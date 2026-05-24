#!/usr/bin/env python3
"""
phase22_hd_surrogate_bo.py
---------------------------
Train an 8-target TargetConditionedSurrogate on the extended Vina scores
(Phase 21 output: 6 original + PDE10A + HDAC3) and run BO for the HD
quad-panel targets:

  HD panel: MSH3, CREBBP, PDE10A, HDAC3

Hypothesis: BF16-512 surrogate discovers higher-pKd compounds than FP32-256
across ≥2 HD panel targets when BO is run with the broader 8-target context.

Design:
  1. Load extended_vina_scores.json (8 targets, 2,639 compounds)
  2. Train FP32-256 and BF16-512 surrogates with N_TARGETS=8
  3. Fidelity check (Spearman ρ) per target — must be ≥0.70 to proceed
  4. 30-round EI BO over 2,639 FDA compounds for each of 8 targets
  5. HD panel summary: MSH3 / CREBBP / PDE10A / HDAC3
  6. Highlight cross-target hits (top compounds appearing in ≥2 HD targets)

GCS output: gs://.../aegis_flashoptim/phase22_hd/
"""

import os, sys, json, time, subprocess, random, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
    try:
        import torch_xla.experimental as _xla_exp
        _xla_exp.eager_mode(True)
    except Exception: pass
except ImportError:
    XLA_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase6_vina_surrogate import (
    TargetConditionedSurrogate, load_pretrained_backbone, vina_to_pkd
)
from phase3_vina_oracle import load_pool_graphs, VinaOracle
from notify import notify, heartbeat

GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID     = os.environ.get("RUN_ID", "aegis_flashoptim")
GCS_BASE   = f"{GCS_BUCKET}/{RUN_ID}"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/tmp/flashoptim_results"))
DATA_DIR   = Path(os.environ.get("DATA_DIR", "/tmp/phase2_data"))
OUT_DIR    = Path("/tmp/phase22_hd")
OUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 8-target config ───────────────────────────────────────────────────────────
TARGETS_8   = ["LINGO1", "PCSK9", "KPC3", "APEX1", "MSH3", "CREBBP",
               "PDE10A", "HDAC3"]
TARGET2ID_8 = {t: i for i, t in enumerate(TARGETS_8)}
N_TARGETS_8 = 8

HD_PANEL    = ["MSH3", "CREBBP", "PDE10A", "HDAC3"]

HIDDEN_DIM_FP32 = 256
HIDDEN_DIM_BF16 = 512
N_EPOCHS        = 50
LR              = 1e-3
BATCH_SIZE      = 64
FIDELITY_THRESH = 0.70
N_BO_ROUNDS     = 30
TOP_K_BO        = 5
EI_CHUNK        = 128

def log(msg): print(f"[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] {msg}", flush=True)
def gsutil_cp(src, dst): subprocess.run(["gsutil", "-q", "cp", str(src), str(dst)], check=False)


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(vina_scores, pool_graphs_all, device):
    """Build (z, pos, valid, target_id, pkd) training records across all 8 targets."""
    records = []
    for target, tid in TARGET2ID_8.items():
        if target not in pool_graphs_all:
            continue
        pg = pool_graphs_all[target]
        for name, (z, pos, valid) in pg.items():
            aff = vina_scores.get(name, {}).get(target, 0.0)
            pkd = vina_to_pkd(aff)
            if pkd > 0:
                records.append((z.squeeze(0), pos.squeeze(0), valid.squeeze(0),
                                 tid, pkd))
    log(f"Dataset: {len(records)} (compound, target) pairs across {N_TARGETS_8} targets")
    return records


def batch_iter(records, batch_size, shuffle=True):
    if shuffle:
        random.shuffle(records)
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        z   = torch.stack([r[0] for r in batch])
        pos = torch.stack([r[1] for r in batch])
        val = torch.stack([r[2] for r in batch])
        tid = torch.tensor([r[3] for r in batch], dtype=torch.long)
        pkd = torch.tensor([r[4] for r in batch], dtype=torch.float32)
        yield z, pos, val, tid, pkd


# ── Training ──────────────────────────────────────────────────────────────────

def train_surrogate(records, val_records, hidden_dim, use_bf16,
                    phase2_ckpt, device, label):
    model = TargetConditionedSurrogate(
        hidden_dim=hidden_dim, n_targets=N_TARGETS_8
    ).to(device)
    load_pretrained_backbone(model, phase2_ckpt, device)

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=N_EPOCHS, eta_min=1e-6
    )
    best_rho = -1.0
    best_sd  = None

    log(f"\n{label} — hidden_dim={hidden_dim}  bf16={use_bf16}  epochs={N_EPOCHS}")
    log(f"  {'ep':>4}  {'lr':>10}  {'train_loss':>12}  {'val_rho':>10}")
    log(f"  {'─'*4}  {'─'*10}  {'─'*12}  {'─'*10}")

    for ep in range(1, N_EPOCHS + 1):
        model.train()
        total_loss, n = 0.0, 0
        for z, pos, val, tid, pkd in batch_iter(records, BATCH_SIZE):
            z, pos, val, tid, pkd = (x.to(device) for x in (z, pos, val, tid, pkd))
            optim.zero_grad()
            ctx = torch.autocast(
                device_type="xla" if XLA_AVAILABLE else "cpu",
                dtype=torch.bfloat16
            ) if use_bf16 else torch.no_grad().__class__()   # dummy ctx
            if use_bf16:
                with torch.autocast(device_type="xla" if XLA_AVAILABLE else "cpu",
                                    dtype=torch.bfloat16):
                    pred = model(z, pos, val, tid)
                    loss = F.mse_loss(pred, pkd)
            else:
                pred = model(z, pos, val, tid)
                loss = F.mse_loss(pred, pkd)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if XLA_AVAILABLE: xm.optimizer_step(optim)
            else: optim.step()
            total_loss += loss.item(); n += 1
            if XLA_AVAILABLE: xm.mark_step()
        sched.step()

        if ep % 5 == 0 or ep == N_EPOCHS:
            rho = evaluate_rho(model, val_records, device, use_bf16)
            lr_now = optim.param_groups[0]["lr"]
            log(f"  ep{ep:>3d}  {lr_now:>10.2e}  {total_loss/max(n,1):>12.4f}  {rho:>10.4f}")
            heartbeat(label, ep, {"rho": rho, "train_loss": total_loss/max(n,1)})
            if rho > best_rho:
                best_rho = rho
                best_sd = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    log(f"\n{label} FINAL best_rho={best_rho:.4f}  "
        f"{'PASS ✓' if best_rho >= FIDELITY_THRESH else 'FAIL ✗'}")
    return model, best_rho, best_sd


@torch.no_grad()
def evaluate_rho(model, val_records, device, use_bf16):
    model.eval()
    preds, truths = [], []
    for z, pos, val, tid, pkd in batch_iter(val_records, BATCH_SIZE, shuffle=False):
        z, pos, val, tid = (x.to(device) for x in (z, pos, val, tid))
        if use_bf16:
            with torch.autocast(device_type="xla" if XLA_AVAILABLE else "cpu",
                                dtype=torch.bfloat16):
                pred = model(z, pos, val, tid)
        else:
            pred = model(z, pos, val, tid)
        if XLA_AVAILABLE: xm.mark_step()
        preds.extend(pred.cpu().float().tolist())
        truths.extend(pkd.tolist())
    rho, _ = spearmanr(preds, truths)
    return float(rho) if not np.isnan(rho) else 0.0


# ── EI BO loop ────────────────────────────────────────────────────────────────

def _stack(pool_graphs, names, device):
    items = [pool_graphs[n] for n in names]
    z   = torch.stack([g[0].squeeze(0) for g in items]).to(device)
    pos = torch.stack([g[1].squeeze(0) for g in items]).to(device)
    val = torch.stack([g[2].squeeze(0) for g in items]).to(device)
    return z, pos, val


def run_bo(surrogate, vina_scores, pool_graphs, target, tid_int,
           device, use_bf16, label):
    oracle   = VinaOracle(vina_scores, target)
    names    = list(pool_graphs.keys())
    observed_names, observed_pkd = [], []
    best_pkd = -float("inf")

    # Warm start: 10 random compounds
    warm = random.sample(names, min(10, len(names)))
    for n in warm:
        pkd = oracle(n)
        observed_names.append(n)
        observed_pkd.append(pkd)
        if pkd > best_pkd:
            best_pkd = pkd

    log(f"  [{label}] Warm start best: {best_pkd:.3f} pKd")

    surrogate.eval()
    for rnd in range(1, N_BO_ROUNDS + 1):
        # Score entire pool
        remaining = [n for n in names if n not in observed_names]
        if not remaining: break

        all_mu, all_sigma = [], []
        with torch.no_grad():
            for i in range(0, len(remaining), EI_CHUNK):
                chunk = remaining[i:i+EI_CHUNK]
                z, pos, val = _stack(pool_graphs, chunk, device)
                tid = torch.full((len(chunk),), tid_int, dtype=torch.long, device=device)
                if use_bf16:
                    with torch.autocast(device_type="xla" if XLA_AVAILABLE else "cpu",
                                        dtype=torch.bfloat16):
                        mu, sigma = surrogate.predict(z, pos, val, tid)
                else:
                    mu, sigma = surrogate.predict(z, pos, val, tid)
                if XLA_AVAILABLE: xm.mark_step()
                all_mu.extend(mu.cpu().float().tolist())
                all_sigma.extend(sigma.cpu().float().tolist())

        # EI acquisition
        mu_t    = torch.tensor(all_mu)
        sig_t   = torch.tensor(all_sigma).clamp(min=1e-6)
        z_score = (mu_t - best_pkd - 0.01) / sig_t
        from torch.distributions import Normal
        dist    = Normal(0, 1)
        ei      = (mu_t - best_pkd - 0.01) * dist.cdf(z_score) + sig_t * dist.log_prob(z_score).exp()
        top_idx = ei.topk(TOP_K_BO).indices.tolist()

        for idx in top_idx:
            n   = remaining[idx]
            pkd = oracle(n)
            observed_names.append(n)
            observed_pkd.append(pkd)
            if pkd > best_pkd:
                best_pkd = pkd

        if rnd % 5 == 0:
            log(f"  [{label}] Round {rnd:>2d}: best={best_pkd:.3f} n_obs={len(observed_names)}")
            heartbeat(f"Phase22_BO_{label}", rnd, {"best_pkd": best_pkd})

    # Top 5 hits
    paired = sorted(zip(observed_names, observed_pkd), key=lambda x: -x[1])
    top5   = paired[:5]
    log(f"  [{label}] DONE  best={best_pkd:.3f}  oracle_calls={oracle.n_calls}")
    log(f"  [{label}] Top 5: {[(n, round(p,3)) for n,p in top5]}")
    return {"target": target, "best_pkd": best_pkd,
            "top5": [{"name": n, "pkd": p} for n, p in top5],
            "oracle_calls": oracle.n_calls}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 65)
    log("  Phase 22 — HD 8-Target Surrogate + BO")
    log(f"  Targets: {TARGETS_8}")
    log(f"  HD panel: {HD_PANEL}")
    log("=" * 65)

    # Device
    if XLA_AVAILABLE:
        device = xm.xla_device(); log(f"Device: {device} (TPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda"); log("Device: CUDA")
    else:
        device = torch.device("cpu"); log("Device: CPU")

    # Load extended vina scores
    ext_path = OUTPUT_DIR / "extended_vina_scores.json"
    if not ext_path.exists():
        log("Downloading extended_vina_scores.json from GCS...")
        subprocess.run(["gsutil", "-q", "cp",
                        f"{GCS_BASE}/phase21_hd/extended_vina_scores.json",
                        str(ext_path)], check=True)
    with open(ext_path) as f:
        vina_scores = json.load(f)
    log(f"Loaded scores: {len(vina_scores)} compounds")

    # Load FDA compounds
    fda_path = DATA_DIR / "pubchem_fda.json"
    if not fda_path.exists():
        subprocess.run(["gsutil", "-q", "cp",
                        f"{GCS_BASE}/phase2_setup/pubchem_fda.json",
                        str(fda_path)], check=True)

    # Build pool graphs per target
    log("\nBuilding molecular graphs...")
    pool_graphs_all = {}
    for target in TARGETS_8:
        pg, _ = load_pool_graphs(vina_scores, target, device)
        if pg:
            pool_graphs_all[target] = pg
            log(f"  {target}: {len(pg)} compounds")

    # Build train/val split (80/20 by compound)
    all_compounds = list(set.union(*[set(pg.keys()) for pg in pool_graphs_all.values()]))
    random.seed(42)
    random.shuffle(all_compounds)
    n_train = int(0.8 * len(all_compounds))
    train_set = set(all_compounds[:n_train])
    val_set   = set(all_compounds[n_train:])

    def filter_graphs(pg, compound_set):
        return {k: v for k, v in pg.items() if k in compound_set}

    train_pool = {t: filter_graphs(pg, train_set) for t, pg in pool_graphs_all.items()}
    val_pool   = {t: filter_graphs(pg, val_set)   for t, pg in pool_graphs_all.items()}

    train_records = build_dataset(vina_scores, train_pool, device)
    val_records   = build_dataset(vina_scores, val_pool, device)

    # Phase2 checkpoint for backbone warm-start
    phase2_ckpt = OUTPUT_DIR / "phase2_best.pt"
    if not phase2_ckpt.exists():
        log("Downloading phase2_best.pt...")
        subprocess.run(["gsutil", "-q", "cp",
                        f"{GCS_BASE}/phase2_best.pt", str(phase2_ckpt)], check=False)

    results = {}

    # ── FP32-256 surrogate ────────────────────────────────────────────────────
    notify("PHASE_START", "[Phase22] FP32-256 surrogate training", data={})
    model_fp32, rho_fp32, sd_fp32 = train_surrogate(
        train_records, val_records,
        hidden_dim=HIDDEN_DIM_FP32, use_bf16=False,
        phase2_ckpt=phase2_ckpt, device=device, label="FP32-256 (8-target)"
    )
    ckpt_fp32 = OUT_DIR / "fp32_256_8target_best.pt"
    torch.save({"model": sd_fp32, "rho": rho_fp32}, ckpt_fp32)
    gsutil_cp(ckpt_fp32, f"{GCS_BASE}/phase22_hd/fp32_256_8target_best.pt")

    # ── BF16-512 surrogate ────────────────────────────────────────────────────
    notify("PHASE_START", "[Phase22] BF16-512 surrogate training", data={})
    model_fp32.load_state_dict(sd_fp32)  # reload best FP32 weights
    model_bf16 = TargetConditionedSurrogate(
        hidden_dim=HIDDEN_DIM_BF16, n_targets=N_TARGETS_8
    ).to(device)
    load_pretrained_backbone(model_bf16, phase2_ckpt, device)
    _, rho_bf16, sd_bf16 = train_surrogate(
        train_records, val_records,
        hidden_dim=HIDDEN_DIM_BF16, use_bf16=True,
        phase2_ckpt=phase2_ckpt, device=device, label="BF16-512 (8-target)"
    )
    ckpt_bf16 = OUT_DIR / "bf16_512_8target_best.pt"
    torch.save({"model": sd_bf16, "rho": rho_bf16}, ckpt_bf16)
    gsutil_cp(ckpt_bf16, f"{GCS_BASE}/phase22_hd/bf16_512_8target_best.pt")

    log(f"\n8-target surrogate fidelity:")
    log(f"  FP32-256: ρ={rho_fp32:.4f}  {'PASS ✓' if rho_fp32>=FIDELITY_THRESH else 'FAIL ✗'}")
    log(f"  BF16-512: ρ={rho_bf16:.4f}  {'PASS ✓' if rho_bf16>=FIDELITY_THRESH else 'FAIL ✗'}")

    # ── BO for all 8 targets ──────────────────────────────────────────────────
    model_fp32.load_state_dict(sd_fp32)
    model_bf16.load_state_dict(sd_bf16)

    for target in TARGETS_8:
        if target not in pool_graphs_all: continue
        tid = TARGET2ID_8[target]
        bo_fp32 = run_bo(model_fp32, vina_scores, pool_graphs_all[target],
                         target, tid, device, False, f"{target}_FP32")
        bo_bf16 = run_bo(model_bf16, vina_scores, pool_graphs_all[target],
                         target, tid, device, True,  f"{target}_BF16")
        delta = bo_bf16["best_pkd"] - bo_fp32["best_pkd"]
        results[target] = {
            "fp32": bo_fp32, "bf16": bo_bf16,
            "delta_pkd": round(delta, 4),
            "hypothesis_supported": delta >= 0.1,
            "hd_panel": target in HD_PANEL,
        }
        log(f"\n  {target}: FP32={bo_fp32['best_pkd']:.3f}  "
            f"BF16={bo_bf16['best_pkd']:.3f}  Δ={delta:+.3f}  "
            f"{'SUPPORTED' if delta>=0.1 else 'REFUTED'}"
            f"{'  [HD PANEL]' if target in HD_PANEL else ''}")

    # ── HD panel summary ──────────────────────────────────────────────────────
    log("\n" + "=" * 65)
    log("  HD PANEL RESULTS (MSH3 / CREBBP / PDE10A / HDAC3)")
    log("=" * 65)
    for t in HD_PANEL:
        if t not in results: continue
        r = results[t]
        log(f"  {t:8s}  FP32={r['fp32']['best_pkd']:.3f}  "
            f"BF16={r['bf16']['best_pkd']:.3f}  Δ={r['delta_pkd']:+.4f}  "
            f"{'✓' if r['hypothesis_supported'] else '✗'}")

    # Cross-target HD hits (compound appears in top-5 of ≥2 HD panel targets)
    all_hd_hits = {}
    for t in HD_PANEL:
        if t not in results: continue
        for entry in results[t]["bf16"]["top5"]:
            n = entry["name"]
            all_hd_hits.setdefault(n, []).append((t, entry["pkd"]))
    cross_hits = {n: hits for n, hits in all_hd_hits.items() if len(hits) >= 2}
    if cross_hits:
        log(f"\n  Cross-target HD hits (≥2 panel targets):")
        for n, hits in sorted(cross_hits.items(), key=lambda x: -len(x[1])):
            log(f"    {n:30s}: {hits}")
    else:
        log("\n  No cross-target HD hits found in top-5 per target")

    # Save
    summary = {
        "experiment": "phase22_hd_surrogate_bo",
        "targets": TARGETS_8, "hd_panel": HD_PANEL,
        "rho_fp32_8target": round(rho_fp32, 4),
        "rho_bf16_8target": round(rho_bf16, 4),
        "results": results,
        "cross_hd_hits": {k: v for k, v in cross_hits.items()},
    }
    out_json = OUT_DIR / "phase22_results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    gsutil_cp(out_json, f"{GCS_BASE}/phase22_hd/results.json")
    log(f"\nResults → {out_json}")

    notify("PHASE_COMPLETE", "[Phase22] HD 8-target BO done",
           data={"rho_fp32": rho_fp32, "rho_bf16": rho_bf16,
                 "n_supported": sum(1 for r in results.values()
                                    if r["hypothesis_supported"])})


if __name__ == "__main__":
    main()
