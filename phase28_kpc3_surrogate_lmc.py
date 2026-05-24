#!/usr/bin/env python3
"""
phase28_kpc3_surrogate_lmc.py
------------------------------
KPC-3 surrogate (MLP on Morgan fingerprints) + LMC.

Reads the enriched vina_scores from /tmp/phase28_kpc3_lmc/vina_scores_chembl.json
  (must be enriched: values are dicts with "smiles" and "vina_score" keys)

Pipeline:
  1. Build Morgan FP dataset (radius=2, nBits=2048)
  2. Train FP32 MLP surrogate (50 epochs)
  3. Train BF16 MLP surrogate (50 epochs)
  4. LMC between FP32 and BF16 (11 α points, MSE on val set)
  5. EI BO for KPC-3 (20 rounds, Vina-warm-started)

GCS output: gs://.../aegis_flashoptim/phase28_kpc3_lmc/results.json
"""

import os, sys, json, time, subprocess, random, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase6_vina_surrogate import vina_to_pkd
from notify import notify, heartbeat

GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID     = os.environ.get("RUN_ID", "aegis_flashoptim")
GCS_BASE   = f"{GCS_BUCKET}/{RUN_ID}"
GCS_OUT    = f"{GCS_BASE}/phase28_kpc3_lmc"

OUT_DIR = Path("/tmp/phase28_kpc3_lmc")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENRICHED_PATH = OUT_DIR / "vina_scores_chembl.json"

N_EPOCHS   = 50
LR         = 1e-3
BATCH_SIZE = 128
N_ALPHA    = 11
N_BO       = 20
TOP_K_BO   = 5
SEED       = 42
VINA_THRESH = -2.0  # only use compounds with vina_score < -2.0 kcal/mol


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] {msg}", flush=True)


def gsutil_cp(src, dst):
    subprocess.run(["gsutil", "-q", "cp", str(src), str(dst)], check=False)


def gcs_exists(gcs_path):
    return subprocess.run(["gsutil", "-q", "stat", gcs_path],
                          capture_output=True).returncode == 0


# ── Model ─────────────────────────────────────────────────────────────────────

class FingerprintSurrogate(nn.Module):
    def __init__(self, in_dim=2048, hidden_dims=(512, 256, 128)):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def predict(self, x):
        """Returns (mu, sigma) for EI BO. Uses MC dropout for uncertainty."""
        self.train()  # enable dropout for MC sampling
        with torch.no_grad():
            preds = torch.stack([self.forward(x) for _ in range(10)], dim=0)
        self.eval()
        return preds.mean(0), preds.std(0).clamp(min=1e-4)


# ── Data ──────────────────────────────────────────────────────────────────────

def build_dataset(enriched):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        log("ERROR: RDKit not available")
        sys.exit(1)

    fps, pkds, names = [], [], []
    n_skipped = 0
    for cid, entry in enriched.items():
        smiles = entry.get("smiles", "")
        score  = entry.get("vina_score", 0.0)
        if not smiles or score >= VINA_THRESH:
            n_skipped += 1
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            n_skipped += 1
            continue
        fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fp_arr = np.array(fp, dtype=np.float32)
        pkd = vina_to_pkd(score)
        if pkd <= 0:
            n_skipped += 1
            continue
        fps.append(fp_arr)
        pkds.append(pkd)
        names.append(cid)

    log(f"Dataset: {len(fps)} compounds  (skipped {n_skipped})")
    X = torch.tensor(np.array(fps), dtype=torch.float32)
    y = torch.tensor(pkds, dtype=torch.float32)
    return X, y, names


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(X_tr, y_tr, X_val, y_val, use_bf16, label):
    torch.manual_seed(SEED)
    model = FingerprintSurrogate()
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=N_EPOCHS, eta_min=1e-5)

    best_mse   = float("inf")
    best_sd    = None

    log(f"\n{label} — bf16={use_bf16}  epochs={N_EPOCHS}")

    for ep in range(1, N_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(X_tr))
        X_s, y_s = X_tr[perm], y_tr[perm]
        total_loss, n = 0.0, 0
        for i in range(0, len(X_s), BATCH_SIZE):
            xb, yb = X_s[i:i+BATCH_SIZE], y_s[i:i+BATCH_SIZE]
            optim.zero_grad()
            if use_bf16:
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    pred = model(xb)
                    loss = F.mse_loss(pred, yb)
            else:
                pred = model(xb)
                loss = F.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()
            n += 1
        sched.step()

        if ep % 10 == 0 or ep == N_EPOCHS:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_mse  = F.mse_loss(val_pred, y_val).item()
                rho, _   = spearmanr(val_pred.numpy(), y_val.numpy())
            rho = float(rho) if not math.isnan(rho) else 0.0
            log(f"  ep{ep:>3d}  train_loss={total_loss/max(n,1):.4f}  val_mse={val_mse:.4f}  rho={rho:.4f}")
            heartbeat(label, ep, {"val_mse": val_mse, "rho": rho})
            if val_mse < best_mse:
                best_mse = val_mse
                best_sd  = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_sd)
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        final_rho, _ = spearmanr(val_pred.numpy(), y_val.numpy())
    final_rho = float(final_rho) if not math.isnan(final_rho) else 0.0
    log(f"\n{label} FINAL  val_mse={best_mse:.4f}  rho={final_rho:.4f}")
    return model, best_sd, final_rho, best_mse


# ── LMC ───────────────────────────────────────────────────────────────────────

def lmc_mse(sd0, sd1, X_val, y_val, label):
    alphas = [round(i / (N_ALPHA - 1), 2) for i in range(N_ALPHA)]
    mse_curve = []
    model = FingerprintSurrogate()

    for alpha in alphas:
        interp = {k: (1 - alpha) * sd0[k].float() + alpha * sd1[k].float()
                  for k in sd0 if k in sd1}
        model.load_state_dict(interp, strict=False)
        model.eval()
        with torch.no_grad():
            pred = model(X_val)
            mse  = F.mse_loss(pred, y_val).item()
        mse_curve.append(round(mse, 6))
        log(f"  {label} α={alpha:.1f}  mse={mse:.6f}")

    base    = min(mse_curve[0], mse_curve[-1])
    barrier = round(max(mse_curve) - base, 6)
    log(f"  {label} LMC barrier (MSE) = {barrier:.6f}")
    return {"alphas": alphas, "mse_curve": mse_curve,
            "barrier_mse": barrier, "baseline_mse": base}


# ── EI BO ─────────────────────────────────────────────────────────────────────

def run_bo(model, X_all, y_all, names, label):
    from torch.distributions import Normal

    indices     = list(range(len(names)))
    # Warm start: top-10 by actual pKd (Vina-guided initialization)
    top10_idx   = sorted(range(len(y_all)), key=lambda i: -y_all[i].item())[:10]
    obs_idx     = set(top10_idx)
    best_pkd    = max(y_all[i].item() for i in obs_idx)

    log(f"  [{label}] Warm start (top-10 Vina): best={best_pkd:.3f}")

    model.eval()
    for rnd in range(1, N_BO + 1):
        remaining = [i for i in indices if i not in obs_idx]
        if not remaining:
            break

        X_rem = X_all[remaining]
        mu, sigma = model.predict(X_rem)

        z_score = (mu - best_pkd - 0.01) / sigma.clamp(min=1e-6)
        dist    = Normal(0, 1)
        ei      = ((mu - best_pkd - 0.01) * dist.cdf(z_score)
                   + sigma * dist.log_prob(z_score).exp())
        top_idx = ei.topk(TOP_K_BO).indices.tolist()

        for idx in top_idx:
            orig_i = remaining[idx]
            pkd    = y_all[orig_i].item()
            obs_idx.add(orig_i)
            if pkd > best_pkd:
                best_pkd = pkd

        log(f"  [{label}] BO round {rnd}: best={best_pkd:.3f}  n_obs={len(obs_idx)}")
        heartbeat(f"Phase28_BO_{label}", rnd, {"best_pkd": best_pkd})

    log(f"  [{label}] BO DONE  best={best_pkd:.3f}  n_obs={len(obs_idx)}")
    return {"best_pkd": best_pkd, "n_obs": len(obs_idx)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 65)
    log("  Phase 28 — KPC-3 FP Surrogate + LMC")
    log("=" * 65)

    notify("PHASE_START", "[Phase28] KPC-3 FP surrogate LMC", data={})

    # Load enriched data
    if not ENRICHED_PATH.exists():
        gcs_path = f"{GCS_BASE}/phase_amr_chembl/vina_scores_chembl_enriched.json"
        if gcs_exists(gcs_path):
            subprocess.run(["gsutil", "-q", "cp", gcs_path, str(ENRICHED_PATH)], check=False)
        else:
            log("ERROR: enriched file not found. Run phase28_enrich_smiles.py first.")
            sys.exit(1)

    with open(ENRICHED_PATH) as f:
        enriched = json.load(f)

    sample = next(iter(enriched.values()))
    if not isinstance(sample, dict):
        log("ERROR: vina_scores_chembl.json is not enriched (values are not dicts). "
            "Run phase28_enrich_smiles.py first.")
        sys.exit(1)

    log(f"Loaded {len(enriched)} enriched compounds")

    # Build dataset
    X, y, names = build_dataset(enriched)
    n = len(y)
    log(f"Dataset after filtering: {n} compounds")

    # 80/20 split
    torch.manual_seed(SEED)
    perm      = torch.randperm(n)
    n_train   = int(0.8 * n)
    tr_idx    = perm[:n_train]
    val_idx   = perm[n_train:]
    X_tr, y_tr   = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    log(f"Split: train={len(tr_idx)}  val={len(val_idx)}")

    # ── FP32 ──────────────────────────────────────────────────────────────────
    fp32_ckpt_local = OUT_DIR / "fp32_best.pt"
    fp32_ckpt_gcs   = f"{GCS_OUT}/fp32_best.pt"

    if fp32_ckpt_local.exists() or gcs_exists(fp32_ckpt_gcs):
        if not fp32_ckpt_local.exists():
            subprocess.run(["gsutil", "-q", "cp", fp32_ckpt_gcs, str(fp32_ckpt_local)], check=False)
        sd_fp32_data = torch.load(fp32_ckpt_local, map_location="cpu")
        sd_fp32  = sd_fp32_data["model"]
        rho_fp32 = sd_fp32_data.get("rho", 0.0)
        log(f"FP32: loaded from checkpoint  rho={rho_fp32:.4f}")
        model_fp32 = FingerprintSurrogate()
        model_fp32.load_state_dict(sd_fp32)
    else:
        model_fp32, sd_fp32, rho_fp32, _ = train_model(
            X_tr, y_tr, X_val, y_val, use_bf16=False, label="FP32-KPC3"
        )
        torch.save({"model": sd_fp32, "rho": rho_fp32}, fp32_ckpt_local)
        gsutil_cp(fp32_ckpt_local, fp32_ckpt_gcs)

    # ── BF16 ──────────────────────────────────────────────────────────────────
    bf16_ckpt_local = OUT_DIR / "bf16_best.pt"
    bf16_ckpt_gcs   = f"{GCS_OUT}/bf16_best.pt"

    if bf16_ckpt_local.exists() or gcs_exists(bf16_ckpt_gcs):
        if not bf16_ckpt_local.exists():
            subprocess.run(["gsutil", "-q", "cp", bf16_ckpt_gcs, str(bf16_ckpt_local)], check=False)
        sd_bf16_data = torch.load(bf16_ckpt_local, map_location="cpu")
        sd_bf16  = sd_bf16_data["model"]
        rho_bf16 = sd_bf16_data.get("rho", 0.0)
        log(f"BF16: loaded from checkpoint  rho={rho_bf16:.4f}")
        model_bf16 = FingerprintSurrogate()
        model_bf16.load_state_dict(sd_bf16)
    else:
        model_bf16, sd_bf16, rho_bf16, _ = train_model(
            X_tr, y_tr, X_val, y_val, use_bf16=True, label="BF16-KPC3"
        )
        torch.save({"model": sd_bf16, "rho": rho_bf16}, bf16_ckpt_local)
        gsutil_cp(bf16_ckpt_local, bf16_ckpt_gcs)

    log(f"\nFP32 rho={rho_fp32:.4f}  BF16 rho={rho_bf16:.4f}")

    # ── LMC ───────────────────────────────────────────────────────────────────
    log("\nComputing LMC (FP32 ↔ BF16)...")
    lmc_results = lmc_mse(sd_fp32, sd_bf16, X_val, y_val, "FP32↔BF16")
    log(f"LMC barrier (MSE): {lmc_results['barrier_mse']:.6f}")

    # ── BO ────────────────────────────────────────────────────────────────────
    log("\nRunning EI BO for KPC-3...")
    model_fp32.eval()
    model_bf16.eval()

    bo_fp32 = run_bo(model_fp32, X, y, names, "FP32")
    bo_bf16 = run_bo(model_bf16, X, y, names, "BF16")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "experiment":      "phase28_kpc3_surrogate_lmc",
        "n_compounds":     n,
        "rho_fp32":        round(rho_fp32, 4),
        "rho_bf16":        round(rho_bf16, 4),
        "lmc_barrier_mse": lmc_results["barrier_mse"],
        "lmc_curve":       lmc_results,
        "bo_fp32_best":    round(bo_fp32["best_pkd"], 4),
        "bo_bf16_best":    round(bo_bf16["best_pkd"], 4),
    }

    out_path = OUT_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    gsutil_cp(out_path, f"{GCS_OUT}/results.json")
    log(f"\nResults → GCS phase28_kpc3_lmc/results.json")
    log(f"  FP32: rho={rho_fp32:.4f}  BO best={bo_fp32['best_pkd']:.3f}")
    log(f"  BF16: rho={rho_bf16:.4f}  BO best={bo_bf16['best_pkd']:.3f}")
    log(f"  LMC barrier (MSE): {lmc_results['barrier_mse']:.6f}")

    notify("PHASE_COMPLETE", "[Phase28] KPC-3 FP surrogate LMC done",
           data={"rho_fp32": rho_fp32, "rho_bf16": rho_bf16,
                 "lmc_barrier": lmc_results["barrier_mse"]})


if __name__ == "__main__":
    main()
