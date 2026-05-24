#!/usr/bin/env python3
"""
phase42_multi_target_bo.py
--------------------------
Phase 42: Multi-target Bayesian Optimisation across KPC-3 and MSH3.

Identifies FDA-approved compounds that score favourably against BOTH targets
simultaneously using surrogates from Phase 41 (loaded from GCS).

Acquisition function: UCB with combined pKd objective
  score(x) = μ_KPC3(x) + μ_MSH3(x) + β·(σ_KPC3(x) + σ_MSH3(x))
plus a diversity penalty that down-weights compounds with Tanimoto > 0.6
to any already-observed hit.

Pipeline:
  1. Load Phase 41 KPC-3 + MSH3 surrogates from GCS (poll until available)
  2. Load FDA compound library + compute Morgan FPs
  3. Run 30-round multi-target BO with EI on combined score
  4. Report top-20 dual-target hits + per-target pKd estimates

GCS output: gs://.../aegis_flashoptim/phase42_multi_target_bo/results.json
"""

import os, sys, json, time, subprocess, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase6_vina_surrogate import vina_to_pkd
from notify import notify, heartbeat

GCS_BUCKET  = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID      = os.environ.get("RUN_ID", "aegis_flashoptim")
GCS_BASE    = f"{GCS_BUCKET}/{RUN_ID}"
GCS_OUT     = f"{GCS_BASE}/phase42_multi_target_bo"
GCS_DONE    = f"{GCS_OUT}/results.json"

# Surrogates from Phase 41
GCS_KPC3_SURR = f"{GCS_BASE}/phase41_lmc_cross_target/kpc3_fp32.pt"
GCS_MSH3_SURR = f"{GCS_BASE}/phase41_lmc_cross_target/msh3_fp32.pt"

# Alternative: phase38/39 MSH3 surrogate if phase41 not done yet
GCS_MSH3_SURR_ALT = f"{GCS_BASE}/phase38_msh3_retry/surrogate.pt"

GCS_FDA     = f"{GCS_BASE}/pubchem_fda.json"
GCS_KPC3    = f"{GCS_BASE}/phase_amr_chembl/vina_scores_chembl_enriched.json"

OUT_DIR  = Path("/tmp/phase42_multi_target_bo")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/phase2_data"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED        = 42
N_BO        = 30
TOP_K       = 5
BETA_UCB    = 0.5        # exploration weight
DIVERSITY_THRESH = 0.6   # Tanimoto threshold for diversity penalty
DIVERSITY_PENALTY = 0.3  # multiply score by (1 - penalty) if too similar

torch.manual_seed(SEED)
np.random.seed(SEED)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] {msg}", flush=True)

def gsutil_cp(src, dst): subprocess.run(["gsutil", "-q", "cp", str(src), str(dst)], check=False)
def gcs_exists(p): return subprocess.run(["gsutil", "-q", "stat", p], capture_output=True).returncode == 0

def gcs_download(gcs, local, retries=3):
    for _ in range(retries):
        if subprocess.run(["gsutil", "-q", "cp", gcs, str(local)],
                          capture_output=True).returncode == 0:
            return True
        time.sleep(5)
    return False


# ── Fingerprint ───────────────────────────────────────────────────────────────

def build_fp(smiles):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), dtype=np.float32)
    except Exception:
        return None


def tanimoto_batch(fp_query: np.ndarray, fp_library: np.ndarray) -> np.ndarray:
    """Tanimoto similarity between one query FP and a library (N × 2048)."""
    inter = fp_library @ fp_query
    union = fp_library.sum(axis=1) + fp_query.sum() - inter
    return inter / np.maximum(union, 1e-9)


# ── Surrogate with MC-dropout uncertainty ─────────────────────────────────────

class FingerprintSurrogate(nn.Module):
    def __init__(self, in_dim=2048, hidden=(512, 256, 128)):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x).squeeze(-1)

    def predict_with_uncertainty(self, x, n_samples=15):
        """MC-dropout: run n_samples forward passes in train mode."""
        self.train()
        with torch.no_grad():
            preds = torch.stack([self.forward(x) for _ in range(n_samples)], 0)
        self.eval()
        return preds.mean(0), preds.std(0).clamp(min=1e-4)


def load_surrogate(gcs_path, local_path):
    """Download and load a FingerprintSurrogate state dict."""
    if not Path(local_path).exists():
        if not gcs_download(gcs_path, local_path):
            return None
    sd = torch.load(str(local_path), map_location="cpu")
    model = FingerprintSurrogate()
    model.load_state_dict(sd)
    model.eval()
    return model


def wait_for_surrogate(gcs_path, local_path, max_wait_min=60):
    """Poll GCS until the surrogate checkpoint appears (phase41 may still be running)."""
    for attempt in range(max_wait_min):
        if gcs_exists(gcs_path):
            model = load_surrogate(gcs_path, local_path)
            if model is not None:
                log(f"  Loaded surrogate from {gcs_path}")
                return model
        if attempt % 5 == 0:
            log(f"  Waiting for {gcs_path} (~{attempt}min elapsed)...")
        time.sleep(60)
    return None


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_fda_compounds():
    fda_path = DATA_DIR / "pubchem_fda.json"
    if not fda_path.exists():
        gcs_download(GCS_FDA, fda_path)
    with open(fda_path) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return {row[0]: row[1] for row in raw if len(row) >= 2}
    return {k: v.get("smiles", "") if isinstance(v, dict) else v for k, v in raw.items()}


def load_kpc3_pkd_lookup():
    """Return {name: pkd} from phase_amr_chembl enriched data."""
    local = OUT_DIR / "kpc3_enriched.json"
    if not local.exists():
        gcs_download(GCS_KPC3, local)
    if not local.exists():
        return {}
    with open(local) as f:
        raw = json.load(f)
    return {k: vina_to_pkd(v.get("vina_score", 0.0))
            for k, v in raw.items() if v.get("vina_score", 0.0) < -1.0}


# ── Multi-target BO ───────────────────────────────────────────────────────────

def run_multi_target_bo(model_kpc3, model_msh3, fps_all, names_all, dev):
    """
    30-round BO maximising combined pKd(KPC-3) + pKd(MSH3) with
    UCB exploration and Tanimoto diversity penalty.
    """
    fps_arr = np.array(fps_all, dtype=np.float32)
    X = torch.tensor(fps_arr, dtype=torch.float32).to(dev)

    # Initial observations: top-10 by average surrogate prediction
    model_kpc3.eval(); model_msh3.eval()
    with torch.no_grad():
        mu_k = model_kpc3(X).cpu().numpy()
        mu_m = model_msh3(X).cpu().numpy()
    combined_init = mu_k + mu_m
    obs_idx = set(np.argsort(combined_init)[-10:].tolist())
    best_combined = max(combined_init[i] for i in obs_idx)
    log(f"  [BO] Warm start best_combined={best_combined:.3f}  n_obs={len(obs_idx)}")

    history = []

    for rnd in range(1, N_BO + 1):
        rem = [i for i in range(len(names_all)) if i not in obs_idx]
        if not rem: break

        X_rem = X[rem]
        mu_k, sig_k = model_kpc3.predict_with_uncertainty(X_rem)
        mu_m, sig_m = model_msh3.predict_with_uncertainty(X_rem)

        # Combined UCB score
        ucb = (mu_k + mu_m + BETA_UCB * (sig_k + sig_m)).cpu().numpy()

        # Diversity penalty: down-weight compounds similar to observed hits
        obs_fps = fps_arr[[list(obs_idx)[i] for i in range(len(obs_idx))]]
        for j, gi in enumerate(rem):
            sims = tanimoto_batch(fps_arr[gi], obs_fps)
            if sims.max() > DIVERSITY_THRESH:
                ucb[j] *= (1.0 - DIVERSITY_PENALTY)

        # Select top-k candidates
        top_local = np.argsort(ucb)[-TOP_K:][::-1]
        round_hits = []
        for local_idx in top_local:
            gi = rem[local_idx]
            obs_idx.add(gi)
            pkd_k = float(mu_k[local_idx].item())
            pkd_m = float(mu_m[local_idx].item())
            combo = pkd_k + pkd_m
            if combo > best_combined: best_combined = combo
            round_hits.append({"name": names_all[gi], "pkd_kpc3": round(pkd_k, 3),
                               "pkd_msh3": round(pkd_m, 3), "combined": round(combo, 3)})

        log(f"  [BO] Round {rnd}: best_combined={best_combined:.3f}  n_obs={len(obs_idx)}")
        heartbeat("Phase42_BO", rnd, {"best_combined": best_combined})
        history.append({"round": rnd, "best_combined": round(best_combined, 4), "hits": round_hits})

        # Checkpoint
        ckpt = OUT_DIR / "bo_checkpoint.json"
        ckpt.write_text(json.dumps({"round": rnd+1, "obs_idx": sorted(obs_idx),
                                    "best_combined": best_combined, "history": history}))
        gsutil_cp(ckpt, f"{GCS_OUT}/bo_checkpoint.json")

    return obs_idx, best_combined, history


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if gcs_exists(GCS_DONE):
        log("Results already in GCS — nothing to do."); return

    notify("PHASE_START", "Phase 42 multi-target BO", data={})
    log("=" * 65)
    log("  Phase 42 — Multi-target BO: KPC-3 + MSH3")
    log("=" * 65)

    dev = torch.device("cpu")  # BO is lightweight; XLA not needed
    try:
        import torch_xla.core.xla_model as xm
        dev = xm.xla_device(); log(f"Device: TPU ({dev})")
    except Exception:
        log("Device: CPU")

    # ── Load surrogates ───────────────────────────────────────────────────────
    log("Loading KPC-3 surrogate (Phase 41)...")
    model_kpc3 = wait_for_surrogate(GCS_KPC3_SURR, OUT_DIR / "kpc3_fp32.pt", max_wait_min=90)
    if model_kpc3 is None:
        log("ERROR: KPC-3 surrogate not available after 90 min"); sys.exit(1)
    model_kpc3 = model_kpc3.to(dev)

    log("Loading MSH3 surrogate (Phase 41, fall back to Phase 38)...")
    model_msh3 = wait_for_surrogate(GCS_MSH3_SURR, OUT_DIR / "msh3_fp32_41.pt", max_wait_min=10)
    if model_msh3 is None:
        log("  Phase 41 MSH3 not ready — trying Phase 38 surrogate...")
        model_msh3 = load_surrogate(GCS_MSH3_SURR_ALT, OUT_DIR / "msh3_fp32_38.pt")
    if model_msh3 is None:
        log("ERROR: no MSH3 surrogate available"); sys.exit(1)
    model_msh3 = model_msh3.to(dev)

    # ── Load FDA compounds ─────────────────────────────────────────────────────
    log("Building FDA compound library...")
    fda_smiles = load_fda_compounds()
    fps_all, names_all = [], []
    for name, smiles in fda_smiles.items():
        if not smiles: continue
        fp = build_fp(smiles)
        if fp is not None:
            fps_all.append(fp)
            names_all.append(name)
    log(f"  {len(fps_all)} FDA compounds with valid FPs")

    # ── Resume BO checkpoint if available ─────────────────────────────────────
    ckpt_local = OUT_DIR / "bo_checkpoint.json"
    if gcs_exists(f"{GCS_OUT}/bo_checkpoint.json"):
        gcs_download(f"{GCS_OUT}/bo_checkpoint.json", ckpt_local)
    # (multi-target BO always starts fresh from warm-start; checkpoint is for crash recovery)

    # ── Run BO ────────────────────────────────────────────────────────────────
    log(f"\nRunning {N_BO}-round multi-target BO...")
    obs_idx, best_combined, history = run_multi_target_bo(
        model_kpc3, model_msh3, fps_all, names_all, dev)

    # ── Build top-20 dual-target hit list ─────────────────────────────────────
    X = torch.tensor(np.array(fps_all, dtype=np.float32)).to(dev)
    model_kpc3.eval(); model_msh3.eval()
    with torch.no_grad():
        mu_k = model_kpc3(X).cpu().numpy()
        mu_m = model_msh3(X).cpu().numpy()
    combined = mu_k + mu_m
    top20_idx = np.argsort(combined)[-20:][::-1]
    top20 = [{"name": names_all[i], "pkd_kpc3": round(float(mu_k[i]), 3),
              "pkd_msh3": round(float(mu_m[i]), 3),
              "combined_pkd": round(float(combined[i]), 3)}
             for i in top20_idx]

    log("\nTop-5 dual-target hits (KPC-3 + MSH3):")
    for i, h in enumerate(top20[:5], 1):
        log(f"  {i}. {h['name']:20s}  pKd_KPC3={h['pkd_kpc3']:.3f}  "
            f"pKd_MSH3={h['pkd_msh3']:.3f}  combined={h['combined_pkd']:.3f}")

    results = {
        "phase": 42,
        "n_fda_compounds": len(fps_all),
        "n_bo_rounds": N_BO,
        "best_combined_pkd": round(best_combined, 4),
        "top20_dual_target": top20,
        "bo_history": [{"round": h["round"], "best_combined": h["best_combined"]}
                       for h in history],
    }

    out = OUT_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    gsutil_cp(out, GCS_DONE)
    log(f"Results → {GCS_DONE}")

    notify("PHASE_COMPLETE", "Phase 42 multi-target BO done", data={
        "n_compounds": len(fps_all), "best_combined": round(best_combined, 3),
        "top_hit": top20[0]["name"] if top20 else "none",
    })


if __name__ == "__main__":
    main()
