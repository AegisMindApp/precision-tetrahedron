#!/usr/bin/env python3
"""
phase41_lmc_cross_target.py
----------------------------
Phase 41: LMC cross-target comparison — KPC-3 vs MSH3 surrogate loss landscapes.

Tests whether drug-target specificity is encoded in loss-landscape geometry.
Trains FingerprintSurrogate (2048-dim Morgan FP → pKd) for each target with
identical architecture and hyperparameters, then computes three LMC curves:

  (a) KPC-3 FP32 ↔ KPC-3 BF16  — intra-target baseline
  (b) MSH3  FP32 ↔ MSH3  BF16  — intra-target baseline
  (c) KPC-3 FP32 ↔ MSH3  FP32  — cross-target (novel)

Cross-target barrier uses a normalised joint loss so the two targets'
MSE scales are comparable. Expected result: barrier_cross >> barrier_intra,
confirming target-specific loss basins.

GCS output: gs://.../aegis_flashoptim/phase41_lmc_cross_target/results.json
"""

import os, sys, json, time, subprocess, math, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase6_vina_surrogate import vina_to_pkd
from notify import notify, heartbeat

warnings.filterwarnings("ignore")

GCS_BUCKET  = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID      = os.environ.get("RUN_ID", "aegis_flashoptim")
GCS_BASE    = f"{GCS_BUCKET}/{RUN_ID}"
GCS_OUT     = f"{GCS_BASE}/phase41_lmc_cross_target"
GCS_DONE    = f"{GCS_OUT}/results.json"

# Input data from prior phases
GCS_KPC3    = f"{GCS_BASE}/phase_amr_chembl/vina_scores_chembl_enriched.json"
GCS_MSH3_39 = f"{GCS_BASE}/phase39_msh3_rdkit/docking_checkpoint.json"
GCS_MSH3_38 = f"{GCS_BASE}/phase38_msh3_retry/docking_checkpoint.json"
GCS_FDA     = f"{GCS_BASE}/pubchem_fda.json"

OUT_DIR  = Path("/tmp/phase41_lmc_cross_target")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/phase2_data"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED        = 42
N_EPOCHS    = 60
LR          = 1e-3
BATCH_SIZE  = 128
N_ALPHA     = 11
VINA_THRESH = -1.0
MAX_COMPOUNDS = 1000

random_state = np.random.seed(SEED)
torch.manual_seed(SEED)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] {msg}", flush=True)

def gsutil_cp(src, dst): subprocess.run(["gsutil", "-q", "cp", str(src), str(dst)], check=False)
def gcs_exists(p): return subprocess.run(["gsutil", "-q", "stat", p], capture_output=True).returncode == 0

def gcs_download(gcs_path, local_path):
    ret = subprocess.run(["gsutil", "-q", "cp", gcs_path, str(local_path)], capture_output=True)
    return ret.returncode == 0


def get_device():
    try:
        import torch_xla.core.xla_model as xm
        dev = xm.xla_device(); log(f"Device: TPU ({dev})"); return dev
    except Exception: pass
    if torch.cuda.is_available(): log("Device: CUDA"); return torch.device("cuda")
    log("Device: CPU"); return torch.device("cpu")


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


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_kpc3_dataset():
    """Load KPC-3 docking data from phase_amr_chembl enriched JSON."""
    local = OUT_DIR / "kpc3_enriched.json"
    if not local.exists():
        if not gcs_download(GCS_KPC3, local):
            log(f"ERROR: cannot download KPC-3 data from {GCS_KPC3}")
            return [], []
    with open(local) as f:
        raw = json.load(f)

    fps, pkds = [], []
    for cid, entry in raw.items():
        smiles = entry.get("smiles", "")
        score  = entry.get("vina_score", 0.0)
        if not smiles or score >= VINA_THRESH: continue
        fp = build_fp(smiles)
        if fp is None: continue
        pkd = vina_to_pkd(score)
        if pkd <= 0: continue
        fps.append(fp); pkds.append(pkd)
        if len(fps) >= MAX_COMPOUNDS: break

    log(f"KPC-3 dataset: {len(fps)} compounds")
    return np.array(fps, dtype=np.float32), np.array(pkds, dtype=np.float32)


def load_msh3_dataset():
    """Load MSH3 docking data: prefer phase39 (rdkit), fall back to phase38."""
    local39 = OUT_DIR / "msh3_ckpt39.json"
    local38 = OUT_DIR / "msh3_ckpt38.json"
    fda_local = DATA_DIR / "pubchem_fda.json"

    # Get docking scores
    raw_scores = None
    if gcs_exists(GCS_MSH3_39):
        if gcs_download(GCS_MSH3_39, local39):
            raw_scores = json.loads(local39.read_text()).get("scores", {})
            log(f"MSH3: loaded Phase 39 docking checkpoint ({len(raw_scores)} compounds)")
    if raw_scores is None:
        if gcs_download(GCS_MSH3_38, local38):
            raw_scores = json.loads(local38.read_text()).get("scores", {})
            log(f"MSH3: fell back to Phase 38 docking checkpoint ({len(raw_scores)} compounds)")
    if raw_scores is None:
        log("ERROR: no MSH3 docking data found"); return [], []

    # Get SMILES from FDA file
    if not fda_local.exists():
        gcs_download(GCS_FDA, fda_local)
    with open(fda_local) as f:
        fda_raw = json.load(f)
    if isinstance(fda_raw, list):
        fda_lookup = {row[0]: row[1] for row in fda_raw if len(row) >= 2}
    else:
        fda_lookup = {k: v.get("smiles", "") if isinstance(v, dict) else v
                      for k, v in fda_raw.items()}

    fps, pkds = [], []
    for name, score in raw_scores.items():
        if score >= VINA_THRESH: continue
        smiles = fda_lookup.get(name, "")
        if not smiles: continue
        fp = build_fp(smiles)
        if fp is None: continue
        pkd = vina_to_pkd(score)
        if pkd <= 0: continue
        fps.append(fp); pkds.append(pkd)
        if len(fps) >= MAX_COMPOUNDS: break

    log(f"MSH3 dataset: {len(fps)} compounds")
    return np.array(fps, dtype=np.float32), np.array(pkds, dtype=np.float32)


# ── Model ──────────────────────────────────────────────────────────────────────

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


def split(X, y, val_frac=0.2, seed=SEED):
    n = len(y)
    idx = np.random.RandomState(seed).permutation(n)
    n_val = max(int(n * val_frac), 20)
    return X[idx[n_val:]], y[idx[n_val:]], X[idx[:n_val]], y[idx[:n_val]]


def train_model(X_tr, y_tr, X_val, y_val, dev, dtype, label):
    Xtr = torch.tensor(X_tr, dtype=dtype).to(dev)
    ytr = torch.tensor(y_tr, dtype=dtype).to(dev)
    Xva = torch.tensor(X_val, dtype=dtype).to(dev)
    yva = torch.tensor(y_val, dtype=dtype).to(dev)

    model = FingerprintSurrogate().to(dev)
    if dtype == torch.bfloat16: model = model.to(dtype)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-5)
    best_mse, best_sd = float("inf"), None

    for ep in range(1, N_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH_SIZE):
            xb, yb = Xtr[perm[i:i+BATCH_SIZE]], ytr[perm[i:i+BATCH_SIZE]]
            opt.zero_grad()
            F.mse_loss(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            try:
                import torch_xla.core.xla_model as xm; xm.mark_step()
            except Exception: pass
        sch.step()
        model.eval()
        with torch.no_grad():
            val_mse = F.mse_loss(model(Xva), yva).item()
        if val_mse < best_mse:
            best_mse = val_mse
            best_sd  = {k: v.float().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0:
            log(f"  [{label}] ep {ep}/{N_EPOCHS}  val_mse={val_mse:.5f}")
        heartbeat(f"Phase41_{label}", ep, {"val_mse": val_mse})

    model.load_state_dict({k: v.to(dtype).to(dev) for k, v in best_sd.items()})
    model.eval()
    log(f"  [{label}] done  best_val={best_mse:.5f}")
    return model, best_sd, best_mse


# ── LMC ───────────────────────────────────────────────────────────────────────

def lmc_intra(sd0, sd1, X_val, y_val, dev, label):
    """Standard intra-target LMC between FP32 and BF16 surrogates."""
    Xva = torch.tensor(X_val, dtype=torch.float32).to(dev)
    yva = torch.tensor(y_val, dtype=torch.float32).to(dev)
    alphas, curve = [round(i/(N_ALPHA-1), 2) for i in range(N_ALPHA)], []
    model = FingerprintSurrogate().to(dev)
    for alpha in alphas:
        sd = {k: (1-alpha)*sd0[k].float() + alpha*sd1[k].float() for k in sd0}
        model.load_state_dict({k: v.to(dev) for k, v in sd.items()})
        model.eval()
        with torch.no_grad():
            mse = F.mse_loss(model(Xva), yva).item()
        curve.append(round(mse, 6))
        log(f"  {label} α={alpha:.1f}  mse={mse:.6f}")
    ep_mean = (curve[0] + curve[-1]) / 2
    barrier = max(curve) - ep_mean
    log(f"  {label} barrier={barrier:.6f}")
    return {"alphas": alphas, "mse_curve": curve,
            "barrier_mse": round(barrier, 6), "endpoint_mean": round(ep_mean, 6)}


def lmc_cross_target(sd_kpc3, sd_msh3,
                     X_val_kpc3, y_val_kpc3,
                     X_val_msh3, y_val_msh3, dev):
    """
    Cross-target LMC: interpolate from KPC-3 model to MSH3 model.
    Uses normalised joint loss so MSE scales are comparable.
    """
    Xk = torch.tensor(X_val_kpc3, dtype=torch.float32).to(dev)
    yk = torch.tensor(y_val_kpc3, dtype=torch.float32).to(dev)
    Xm = torch.tensor(X_val_msh3, dtype=torch.float32).to(dev)
    ym = torch.tensor(y_val_msh3, dtype=torch.float32).to(dev)

    alphas = [round(i/(N_ALPHA-1), 2) for i in range(N_ALPHA)]
    curve_kpc3, curve_msh3, curve_joint = [], [], []
    model = FingerprintSurrogate().to(dev)

    # Endpoint losses for normalisation
    def endpoint_mse(sd, Xv, yv):
        model.load_state_dict({k: v.to(dev) for k, v in sd.items()})
        model.eval()
        with torch.no_grad(): return F.mse_loss(model(Xv), yv).item()

    ep0_kpc3 = endpoint_mse(sd_kpc3, Xk, yk)
    ep1_kpc3 = endpoint_mse(sd_msh3, Xk, yk)
    ep0_msh3 = endpoint_mse(sd_kpc3, Xm, ym)
    ep1_msh3 = endpoint_mse(sd_msh3, Xm, ym)
    norm_kpc3 = (ep0_kpc3 + ep1_kpc3) / 2
    norm_msh3 = (ep0_msh3 + ep1_msh3) / 2
    log(f"  Cross-target endpoints: KPC-3_ep=[{ep0_kpc3:.4f},{ep1_kpc3:.4f}]  "
        f"MSH3_ep=[{ep0_msh3:.4f},{ep1_msh3:.4f}]")

    for alpha in alphas:
        sd = {k: (1-alpha)*sd_kpc3[k].float() + alpha*sd_msh3[k].float() for k in sd_kpc3}
        model.load_state_dict({k: v.to(dev) for k, v in sd.items()})
        model.eval()
        with torch.no_grad():
            mse_k = F.mse_loss(model(Xk), yk).item()
            mse_m = F.mse_loss(model(Xm), ym).item()
        norm_k = mse_k / norm_kpc3 if norm_kpc3 > 0 else mse_k
        norm_m = mse_m / norm_msh3 if norm_msh3 > 0 else mse_m
        joint  = (norm_k + norm_m) / 2
        curve_kpc3.append(round(mse_k, 6))
        curve_msh3.append(round(mse_m, 6))
        curve_joint.append(round(joint, 6))
        log(f"  Cross-target α={alpha:.1f}  kpc3_norm={norm_k:.4f}  msh3_norm={norm_m:.4f}  joint={joint:.4f}")

    ep_mean_joint = (curve_joint[0] + curve_joint[-1]) / 2
    barrier_joint = max(curve_joint) - ep_mean_joint
    log(f"  Cross-target barrier (normalised joint) = {barrier_joint:.4f}")
    return {
        "alphas":         alphas,
        "mse_curve_kpc3": curve_kpc3,
        "mse_curve_msh3": curve_msh3,
        "mse_curve_joint_norm": curve_joint,
        "barrier_joint_norm": round(barrier_joint, 4),
        "endpoint_mean_joint": round(ep_mean_joint, 4),
        "norm_kpc3": round(norm_kpc3, 6),
        "norm_msh3": round(norm_msh3, 6),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if gcs_exists(GCS_DONE):
        log("Results already in GCS — nothing to do."); return

    notify("PHASE_START", "Phase 41 LMC cross-target", data={})
    log("=" * 65)
    log("  Phase 41 — LMC cross-target: KPC-3 vs MSH3 surrogates")
    log("=" * 65)

    # ── Load datasets ─────────────────────────────────────────────────────────
    X_kpc3, y_kpc3 = load_kpc3_dataset()
    X_msh3, y_msh3 = load_msh3_dataset()
    if len(X_kpc3) < 20 or len(X_msh3) < 20:
        log("ERROR: insufficient data"); sys.exit(1)

    X_kpc3_tr, y_kpc3_tr, X_kpc3_val, y_kpc3_val = split(X_kpc3, y_kpc3)
    X_msh3_tr, y_msh3_tr, X_msh3_val, y_msh3_val = split(X_msh3, y_msh3)
    log(f"KPC-3: {len(y_kpc3_tr)} train / {len(y_kpc3_val)} val")
    log(f"MSH3:  {len(y_msh3_tr)} train / {len(y_msh3_val)} val")

    dev = get_device()

    # ── Train KPC-3 FP32 ──────────────────────────────────────────────────────
    log("\n[1/6] KPC-3 FP32 surrogate")
    ckpt = OUT_DIR / "kpc3_fp32.pt"
    if ckpt.exists():
        sd = torch.load(ckpt, map_location="cpu")
        m = FingerprintSurrogate().to(dev); m.load_state_dict({k: v.to(dev) for k,v in sd.items()})
        _, best_mse_k32 = m, 0.0
        sd_kpc3_fp32 = sd
        log("  KPC-3 FP32: loaded from checkpoint")
    else:
        _, sd_kpc3_fp32, best_mse_k32 = train_model(
            X_kpc3_tr, y_kpc3_tr, X_kpc3_val, y_kpc3_val, dev, torch.float32, "KPC3_FP32")
        torch.save(sd_kpc3_fp32, ckpt); gsutil_cp(ckpt, f"{GCS_OUT}/kpc3_fp32.pt")

    # ── Train KPC-3 BF16 ──────────────────────────────────────────────────────
    log("\n[2/6] KPC-3 BF16 surrogate")
    ckpt = OUT_DIR / "kpc3_bf16.pt"
    if ckpt.exists():
        sd_kpc3_bf16 = torch.load(ckpt, map_location="cpu"); log("  KPC-3 BF16: loaded")
    else:
        _, sd_kpc3_bf16, _ = train_model(
            X_kpc3_tr, y_kpc3_tr, X_kpc3_val, y_kpc3_val, dev, torch.bfloat16, "KPC3_BF16")
        torch.save(sd_kpc3_bf16, ckpt); gsutil_cp(ckpt, f"{GCS_OUT}/kpc3_bf16.pt")

    # ── Train MSH3 FP32 ───────────────────────────────────────────────────────
    log("\n[3/6] MSH3 FP32 surrogate")
    ckpt = OUT_DIR / "msh3_fp32.pt"
    # Try loading phase38/39's saved surrogate first
    if not ckpt.exists():
        for gcs_surr in [f"{GCS_BASE}/phase39_msh3_rdkit/surrogate.pt",
                         f"{GCS_BASE}/phase38_msh3_retry/surrogate.pt"]:
            if gcs_exists(gcs_surr):
                gcs_download(gcs_surr, ckpt)
                log(f"  MSH3 FP32: loaded from {gcs_surr}")
                break
    if ckpt.exists():
        sd_msh3_fp32 = torch.load(ckpt, map_location="cpu")
    else:
        _, sd_msh3_fp32, _ = train_model(
            X_msh3_tr, y_msh3_tr, X_msh3_val, y_msh3_val, dev, torch.float32, "MSH3_FP32")
        torch.save(sd_msh3_fp32, ckpt); gsutil_cp(ckpt, f"{GCS_OUT}/msh3_fp32.pt")

    # ── Train MSH3 BF16 ───────────────────────────────────────────────────────
    log("\n[4/6] MSH3 BF16 surrogate")
    ckpt = OUT_DIR / "msh3_bf16.pt"
    if ckpt.exists():
        sd_msh3_bf16 = torch.load(ckpt, map_location="cpu"); log("  MSH3 BF16: loaded")
    else:
        _, sd_msh3_bf16, _ = train_model(
            X_msh3_tr, y_msh3_tr, X_msh3_val, y_msh3_val, dev, torch.bfloat16, "MSH3_BF16")
        torch.save(sd_msh3_bf16, ckpt); gsutil_cp(ckpt, f"{GCS_OUT}/msh3_bf16.pt")

    # ── Spearman ρ for each model ─────────────────────────────────────────────
    def spearman_val(sd, Xv, yv):
        m = FingerprintSurrogate().to(dev)
        m.load_state_dict({k: v.to(dev) for k, v in sd.items()}); m.eval()
        with torch.no_grad():
            p = m(torch.tensor(Xv, dtype=torch.float32).to(dev)).cpu().numpy()
        rho, _ = spearmanr(p, yv)
        return round(float(rho) if not math.isnan(rho) else 0.0, 4)

    rho_k32 = spearman_val(sd_kpc3_fp32, X_kpc3_val, y_kpc3_val)
    rho_k16 = spearman_val(sd_kpc3_bf16, X_kpc3_val, y_kpc3_val)
    rho_m32 = spearman_val(sd_msh3_fp32, X_msh3_val, y_msh3_val)
    rho_m16 = spearman_val(sd_msh3_bf16, X_msh3_val, y_msh3_val)
    log(f"  Spearman ρ: KPC3_FP32={rho_k32}  KPC3_BF16={rho_k16}  "
        f"MSH3_FP32={rho_m32}  MSH3_BF16={rho_m16}")

    # ── LMC: KPC-3 intra ──────────────────────────────────────────────────────
    log("\n[5a/6] LMC KPC-3 FP32 ↔ BF16 (intra-target)")
    lmc_kpc3_intra = lmc_intra(sd_kpc3_fp32, sd_kpc3_bf16, X_kpc3_val, y_kpc3_val, dev, "KPC3 intra")

    # ── LMC: MSH3 intra ───────────────────────────────────────────────────────
    log("\n[5b/6] LMC MSH3 FP32 ↔ BF16 (intra-target)")
    lmc_msh3_intra = lmc_intra(sd_msh3_fp32, sd_msh3_bf16, X_msh3_val, y_msh3_val, dev, "MSH3 intra")

    # ── LMC: cross-target ─────────────────────────────────────────────────────
    log("\n[6/6] LMC KPC-3 ↔ MSH3 (cross-target)")
    lmc_cross = lmc_cross_target(
        sd_kpc3_fp32, sd_msh3_fp32,
        X_kpc3_val, y_kpc3_val,
        X_msh3_val, y_msh3_val, dev)

    barrier_ratio = (lmc_cross["barrier_joint_norm"] /
                     max(lmc_kpc3_intra["barrier_mse"], lmc_msh3_intra["barrier_mse"], 1e-9))
    log(f"\nBarrier summary:")
    log(f"  KPC-3 intra barrier:  {lmc_kpc3_intra['barrier_mse']:.6f} MSE")
    log(f"  MSH3  intra barrier:  {lmc_msh3_intra['barrier_mse']:.6f} MSE")
    log(f"  Cross-target barrier: {lmc_cross['barrier_joint_norm']:.4f} (normalised)")
    log(f"  Cross/intra ratio:    {barrier_ratio:.2f}×")

    results = {
        "phase": 41,
        "n_kpc3": int(len(y_kpc3)),
        "n_msh3": int(len(y_msh3)),
        "surrogates": {
            "kpc3_fp32_rho": rho_k32, "kpc3_bf16_rho": rho_k16,
            "msh3_fp32_rho": rho_m32, "msh3_bf16_rho": rho_m16,
        },
        "lmc_kpc3_intra": lmc_kpc3_intra,
        "lmc_msh3_intra": lmc_msh3_intra,
        "lmc_cross_target": lmc_cross,
        "barrier_cross_over_intra_ratio": round(barrier_ratio, 3),
    }

    out = OUT_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    gsutil_cp(out, GCS_DONE)
    log(f"Results → {GCS_DONE}")

    notify("PHASE_COMPLETE", "Phase 41 LMC cross-target done", data={
        "kpc3_rho": rho_k32, "msh3_rho": rho_m32,
        "barrier_intra_kpc3": lmc_kpc3_intra["barrier_mse"],
        "barrier_intra_msh3": lmc_msh3_intra["barrier_mse"],
        "barrier_cross": lmc_cross["barrier_joint_norm"],
        "ratio": round(barrier_ratio, 2),
    })


if __name__ == "__main__":
    main()
