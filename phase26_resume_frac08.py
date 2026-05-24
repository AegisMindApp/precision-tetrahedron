#!/usr/bin/env python3
"""
phase26_resume_frac08.py
------------------------
Phase 26 ρ-degradation sweep RESUME — runs fracs [0.8, 1.0] only.
Fracs 0.2/0.4/0.6 are already done in GCS:
  phase26_rho_degradation/partial_frac0.4.json
  phase26_rho_degradation/partial_frac0.6.json

For each frac:
  - Subsample first int(frac × n_train) compounds from the full train set
  - For frac=0.8: skip FP32 training (download existing), train BF16-512 only
  - For frac=1.0: train both FP32-256 and BF16-512
  - Run EI BO (20 rounds) for HD_PANEL × [FP32, BF16]
  - Save partial JSON to GCS (phase26_retry/)

At end: compile all fracs into final results.json and upload to phase26_retry/results.json.

GCS output: gs://.../aegis_flashoptim/phase26_retry/
"""

import os, sys, json, time, subprocess, random, math
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
        import torch_xla.experimental as _xe
        _xe.eager_mode(True)
    except Exception:
        pass
except ImportError:
    XLA_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase6_vina_surrogate import TargetConditionedSurrogate, load_pretrained_backbone, vina_to_pkd
from phase3_vina_oracle import load_pool_graphs, VinaOracle
from notify import notify, heartbeat

GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID     = os.environ.get("RUN_ID", "aegis_flashoptim")
GCS_BASE   = f"{GCS_BUCKET}/{RUN_ID}"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/tmp/flashoptim_results"))
DATA_DIR   = Path(os.environ.get("DATA_DIR", "/tmp/phase2_data"))
OUT_DIR    = Path("/tmp/phase26_resume")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS_8   = ["LINGO1", "PCSK9", "KPC3", "APEX1", "MSH3", "CREBBP", "PDE10A", "HDAC3"]
TARGET2ID_8 = {t: i for i, t in enumerate(TARGETS_8)}
N_TARGETS_8 = 8

HD_PANEL        = ["MSH3", "CREBBP", "PDE10A", "HDAC3"]
HIDDEN_DIM_FP32 = 256
HIDDEN_DIM_BF16 = 512
N_EPOCHS        = 30
LR              = 1e-3
BATCH_SIZE      = 64
FIDELITY_THRESH = 0.70
N_BO_ROUNDS     = 20
TOP_K_BO        = 5
EI_CHUNK        = 128


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] {msg}", flush=True)


def gsutil_cp(src, dst):
    subprocess.run(["gsutil", "-q", "cp", str(src), str(dst)], check=False)


def gcs_exists(gcs_path):
    return subprocess.run(["gsutil", "-q", "stat", gcs_path],
                          capture_output=True).returncode == 0


def download_if_missing(local_path, gcs_path):
    if not local_path.exists():
        if gcs_exists(gcs_path):
            subprocess.run(["gsutil", "-q", "cp", gcs_path, str(local_path)], check=False)
            return True
        return False
    return True


# ── Dataset ───────────────────────────────────────────────────────────────────

def build_records(vina_scores, pool_graphs_all, compound_set):
    records = []
    for target, tid in TARGET2ID_8.items():
        if target not in pool_graphs_all:
            continue
        for name, (z, pos, valid) in pool_graphs_all[target].items():
            if name not in compound_set:
                continue
            aff = vina_scores.get(name, {}).get(target, 0.0)
            pkd = vina_to_pkd(aff)
            if pkd > 0:
                records.append((z.squeeze(0), pos.squeeze(0), valid.squeeze(0),
                                tid, pkd))
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
        if XLA_AVAILABLE:
            xm.mark_step()
        preds.extend(pred.cpu().float().tolist())
        truths.extend(pkd.tolist())
    rho, _ = spearmanr(preds, truths)
    return float(rho) if not np.isnan(rho) else 0.0


def train_surrogate(records, val_records, hidden_dim, use_bf16,
                    phase2_ckpt, device, label, frac):
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

    log(f"\n{label} — hidden_dim={hidden_dim}  bf16={use_bf16}  frac={frac}  epochs={N_EPOCHS}")

    for ep in range(1, N_EPOCHS + 1):
        model.train()
        total_loss, n = 0.0, 0
        for z, pos, val, tid, pkd in batch_iter(records, BATCH_SIZE):
            z, pos, val, tid, pkd = (x.to(device) for x in (z, pos, val, tid, pkd))
            optim.zero_grad()
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
            if XLA_AVAILABLE:
                xm.optimizer_step(optim)
            else:
                optim.step()
            total_loss += loss.item()
            n += 1
            if XLA_AVAILABLE:
                xm.mark_step()
        sched.step()

        if ep % 5 == 0 or ep == N_EPOCHS:
            rho = evaluate_rho(model, val_records, device, use_bf16)
            lr_now = optim.param_groups[0]["lr"]
            log(f"  ep{ep:>3d}  lr={lr_now:.2e}  loss={total_loss/max(n,1):.4f}  rho={rho:.4f}")
            heartbeat(label, ep, {"rho": rho, "frac": frac})
            if rho > best_rho:
                best_rho = rho
                best_sd = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    prec = "BF16-512" if use_bf16 else "FP32-256"
    log(f"\n{prec} frac={frac} FINAL best_rho={best_rho:.4f}  "
        f"{'PASS ✓' if best_rho >= FIDELITY_THRESH else 'FAIL ✗'}")
    return model, best_rho, best_sd


# ── BO ────────────────────────────────────────────────────────────────────────

def _stack(pool_graphs, names, device):
    items = [pool_graphs[n] for n in names]
    z   = torch.stack([g[0].squeeze(0) for g in items]).to(device)
    pos = torch.stack([g[1].squeeze(0) for g in items]).to(device)
    val = torch.stack([g[2].squeeze(0) for g in items]).to(device)
    return z, pos, val


def run_bo(surrogate, vina_scores, pool_graphs, target, tid_int,
           device, use_bf16, label, frac):
    from torch.distributions import Normal
    oracle   = VinaOracle(vina_scores, target)
    names    = list(pool_graphs.keys())
    observed_names, observed_pkd = [], []
    best_pkd = -float("inf")

    warm = random.sample(names, min(10, len(names)))
    for n in warm:
        pkd = oracle(n)
        observed_names.append(n)
        observed_pkd.append(pkd)
        if pkd > best_pkd:
            best_pkd = pkd

    log(f"  [{label}_f{frac:.1f}] Warm start best: {best_pkd:.3f} pKd")

    surrogate.eval()
    for rnd in range(1, N_BO_ROUNDS + 1):
        remaining = [n for n in names if n not in observed_names]
        if not remaining:
            break

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
                if XLA_AVAILABLE:
                    xm.mark_step()
                all_mu.extend(mu.cpu().float().tolist())
                all_sigma.extend(sigma.cpu().float().tolist())

        mu_t    = torch.tensor(all_mu)
        sig_t   = torch.tensor(all_sigma).clamp(min=1e-6)
        z_score = (mu_t - best_pkd - 0.01) / sig_t
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

        ts = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        print(f"[{ts}]   [{target}_{'BF16' if use_bf16 else 'FP32'}_f{frac:.1f}] Round {rnd}: best={best_pkd:.3f}  n_obs={len(observed_names)}", flush=True)
        heartbeat(f"Phase26_BO_{target}_{'BF16' if use_bf16 else 'FP32'}", rnd,
                  {"best_pkd": best_pkd})

    ts = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    print(f"[{ts}]   [{target}_{'BF16' if use_bf16 else 'FP32'}_f{frac:.1f}] DONE  best={best_pkd:.3f}  oracle_calls={oracle.n_calls}", flush=True)

    paired = sorted(zip(observed_names, observed_pkd), key=lambda x: -x[1])
    top5   = paired[:5]
    return {"target": target, "best_pkd": best_pkd,
            "top5": [{"name": n, "pkd": p} for n, p in top5],
            "oracle_calls": oracle.n_calls}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 65)
    log("  Phase 26 ρ-degradation RESUME — fracs [0.8, 1.0]")
    log("=" * 65)

    notify("PHASE_START", "[Phase26] ρ-degradation resume fracs 0.8 and 1.0", data={})

    if XLA_AVAILABLE:
        device = xm.xla_device()
        log(f"Device: {device} (TPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        log("Device: CUDA")
    else:
        device = torch.device("cpu")
        log("Device: CPU")

    # Download extended vina scores
    ext_path = OUTPUT_DIR / "extended_vina_scores.json"
    download_if_missing(ext_path, f"{GCS_BASE}/phase21_hd/extended_vina_scores.json")
    with open(ext_path) as f:
        vina_scores = json.load(f)
    log(f"Loaded scores: {len(vina_scores)} compounds")

    # Phase2 checkpoint
    phase2_ckpt = OUTPUT_DIR / "phase2_best.pt"
    download_if_missing(phase2_ckpt, f"{GCS_BASE}/phase2_best.pt")

    # Build pool graphs
    log("\nBuilding molecular graphs...")
    pool_graphs_all = {}
    for target in TARGETS_8:
        pg, _ = load_pool_graphs(vina_scores, target, device)
        if pg:
            pool_graphs_all[target] = pg
            log(f"  {target}: {len(pg)} compounds")

    # Full train/val split (80/20 by compound, seed=42)
    all_compounds = list(set.union(*[set(pg.keys()) for pg in pool_graphs_all.values()]))
    random.seed(42)
    random.shuffle(all_compounds)
    n_total      = len(all_compounds)
    n_train      = int(0.8 * n_total)
    train_list   = all_compounds[:n_train]   # ~2111 compounds
    val_set      = set(all_compounds[n_train:])
    log(f"Total compounds: {n_total}  train: {n_train}  val: {n_total - n_train}")

    val_pool = {t: {k: v for k, v in pg.items() if k in val_set}
                for t, pg in pool_graphs_all.items()}
    val_records = build_records(vina_scores, val_pool, val_set)
    log(f"Val records: {len(val_records)}")

    this_run_results = {}

    for frac in [0.8, 1.0]:
        log(f"\n{'='*60}")
        log(f"  FRAC = {frac}")
        log(f"{'='*60}")

        n_train_frac  = int(frac * n_train)
        frac_compounds = set(train_list[:n_train_frac])
        frac_pool     = {t: {k: v for k, v in pg.items() if k in frac_compounds}
                         for t, pg in pool_graphs_all.items()}
        frac_records  = build_records(vina_scores, frac_pool, frac_compounds)
        log(f"  frac={frac}: {n_train_frac} compounds → {len(frac_records)} records")

        # ── FP32-256 ──────────────────────────────────────────────────────────
        fp32_ckpt_local = OUT_DIR / f"fp32_frac{frac:.1f}.pt"
        fp32_ckpt_gcs   = f"{GCS_BASE}/phase26_rho_degradation/fp32_frac{frac:.1f}.pt"

        if frac == 0.8 and not fp32_ckpt_local.exists():
            if gcs_exists(fp32_ckpt_gcs):
                log(f"  FP32 frac={frac}: downloading from GCS, skipping training")
                subprocess.run(["gsutil", "-q", "cp", fp32_ckpt_gcs, str(fp32_ckpt_local)], check=False)

        if fp32_ckpt_local.exists():
            log(f"  FP32 frac={frac}: loading from {fp32_ckpt_local}")
            ckpt_data = torch.load(fp32_ckpt_local, map_location="cpu")
            rho_fp32  = ckpt_data.get("rho", 0.0)
            sd_fp32   = ckpt_data["model"]
            model_fp32 = TargetConditionedSurrogate(
                hidden_dim=HIDDEN_DIM_FP32, n_targets=N_TARGETS_8
            ).to(device)
            model_fp32.load_state_dict(sd_fp32)
            log(f"  FP32-256 frac={frac} LOADED  rho={rho_fp32:.4f}")
        else:
            log(f"  FP32 frac={frac}: training FP32-256")
            model_fp32, rho_fp32, sd_fp32 = train_surrogate(
                frac_records, val_records,
                hidden_dim=HIDDEN_DIM_FP32, use_bf16=False,
                phase2_ckpt=phase2_ckpt, device=device,
                label=f"FP32-256 frac={frac}", frac=frac
            )
            torch.save({"model": sd_fp32, "rho": rho_fp32}, fp32_ckpt_local)
            gsutil_cp(fp32_ckpt_local, fp32_ckpt_gcs)
            gsutil_cp(fp32_ckpt_local, f"{GCS_BASE}/phase26_retry/fp32_frac{frac:.1f}.pt")

        log(f"FP32-256 frac={frac} FINAL best_rho={rho_fp32:.4f}  "
            f"{'PASS ✓' if rho_fp32 >= FIDELITY_THRESH else 'FAIL ✗'}")

        # ── BF16-512 ──────────────────────────────────────────────────────────
        bf16_ckpt_local = OUT_DIR / f"bf16_frac{frac:.1f}.pt"
        bf16_ckpt_gcs   = f"{GCS_BASE}/phase26_rho_degradation/bf16_frac{frac:.1f}.pt"

        if not bf16_ckpt_local.exists():
            if gcs_exists(bf16_ckpt_gcs):
                log(f"  BF16 frac={frac}: downloading from GCS, skipping training")
                subprocess.run(["gsutil", "-q", "cp", bf16_ckpt_gcs, str(bf16_ckpt_local)], check=False)

        if bf16_ckpt_local.exists():
            log(f"  BF16 frac={frac}: loading from {bf16_ckpt_local}")
            ckpt_data = torch.load(bf16_ckpt_local, map_location="cpu")
            rho_bf16  = ckpt_data.get("rho", 0.0)
            sd_bf16   = ckpt_data["model"]
            model_bf16 = TargetConditionedSurrogate(
                hidden_dim=HIDDEN_DIM_BF16, n_targets=N_TARGETS_8
            ).to(device)
            model_bf16.load_state_dict(sd_bf16)
            log(f"  BF16-512 frac={frac} LOADED  rho={rho_bf16:.4f}")
        else:
            log(f"  BF16 frac={frac}: training BF16-512 (lr=1e-3, 30 epochs)")
            model_bf16, rho_bf16, sd_bf16 = train_surrogate(
                frac_records, val_records,
                hidden_dim=HIDDEN_DIM_BF16, use_bf16=True,
                phase2_ckpt=phase2_ckpt, device=device,
                label=f"BF16-512 frac={frac}", frac=frac
            )
            torch.save({"model": sd_bf16, "rho": rho_bf16}, bf16_ckpt_local)
            gsutil_cp(bf16_ckpt_local, bf16_ckpt_gcs)
            gsutil_cp(bf16_ckpt_local, f"{GCS_BASE}/phase26_retry/bf16_frac{frac:.1f}.pt")

        log(f"BF16-512 frac={frac} FINAL best_rho={rho_bf16:.4f}  "
            f"{'PASS ✓' if rho_bf16 >= FIDELITY_THRESH else 'FAIL ✗'}")

        # ── BO for HD_PANEL ────────────────────────────────────────────────────
        model_fp32.load_state_dict(sd_fp32)
        model_bf16.load_state_dict(sd_bf16)
        bo_results = {}

        for target in HD_PANEL:
            if target not in pool_graphs_all:
                continue
            tid = TARGET2ID_8[target]
            bo_fp32 = run_bo(model_fp32, vina_scores, pool_graphs_all[target],
                             target, tid, device, False, f"{target}_FP32", frac)
            bo_bf16 = run_bo(model_bf16, vina_scores, pool_graphs_all[target],
                             target, tid, device, True,  f"{target}_BF16", frac)
            delta = bo_bf16["best_pkd"] - bo_fp32["best_pkd"]
            bo_results[target] = {
                "fp32": bo_fp32, "bf16": bo_bf16,
                "delta_pkd": round(delta, 4),
            }
            log(f"  {target}: FP32={bo_fp32['best_pkd']:.3f}  "
                f"BF16={bo_bf16['best_pkd']:.3f}  Δ={delta:+.3f}")

        frac_data = {
            "frac": frac,
            "n_train_compounds": n_train_frac,
            "n_records": len(frac_records),
            "rho_fp32": round(rho_fp32, 4),
            "rho_bf16": round(rho_bf16, 4),
            "bo_results": bo_results,
        }
        this_run_results[f"{frac:.1f}"] = frac_data

        partial_path = OUT_DIR / f"partial_frac{frac:.1f}.json"
        partial_path.write_text(json.dumps(frac_data, indent=2))
        gsutil_cp(partial_path, f"{GCS_BASE}/phase26_retry/partial_frac{frac:.1f}.json")
        log(f"  Partial results saved → GCS phase26_retry/partial_frac{frac:.1f}.json")

    # ── Compile all fracs ─────────────────────────────────────────────────────
    log("\n" + "=" * 65)
    log("  COMPILING ALL FRACS")
    log("=" * 65)

    all_results = {}

    # fracs 0.4 and 0.6 from GCS
    for prior_frac in ["0.4", "0.6"]:
        for gcs_prefix in ["phase26_rho_degradation", "phase26_retry"]:
            gcs_p = f"{GCS_BASE}/{gcs_prefix}/partial_frac{prior_frac}.json"
            local_p = OUT_DIR / f"partial_frac{prior_frac}.json"
            if not local_p.exists() and gcs_exists(gcs_p):
                subprocess.run(["gsutil", "-q", "cp", gcs_p, str(local_p)], check=False)
                break
        if local_p.exists():
            with open(local_p) as f:
                all_results[prior_frac] = json.load(f)
            log(f"  Loaded frac={prior_frac} from disk/GCS")
        else:
            log(f"  WARNING: frac={prior_frac} partial not found")

    for k, v in this_run_results.items():
        all_results[k] = v

    # Print summary table
    log("\n  ρ-degradation summary:")
    log(f"  {'frac':>6}  {'n_train':>8}  {'rho_fp32':>10}  {'rho_bf16':>10}")
    log(f"  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*10}")
    for fk in sorted(all_results.keys(), key=float):
        r = all_results[fk]
        log(f"  {float(fk):>6.1f}  {r.get('n_train_compounds', '?'):>8}  "
            f"{r.get('rho_fp32', 0):>10.4f}  {r.get('rho_bf16', 0):>10.4f}")

    final_results = {
        "experiment": "phase26_rho_degradation_sweep",
        "fracs": sorted(all_results.keys(), key=float),
        "results_by_frac": all_results,
    }
    final_path = OUT_DIR / "results.json"
    final_path.write_text(json.dumps(final_results, indent=2))
    gsutil_cp(final_path, f"{GCS_BASE}/phase26_retry/results.json")
    log(f"\nFinal results → GCS phase26_retry/results.json")

    notify("PHASE_COMPLETE", "[Phase26] ρ-degradation sweep done",
           data={"fracs_done": list(all_results.keys())})


if __name__ == "__main__":
    main()
