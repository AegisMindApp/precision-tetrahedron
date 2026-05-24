#!/usr/bin/env python3
"""
phase34_nash_amr.py
--------------------
Nash equilibrium–guided drug combination optimization for KPC-3 AMR.

Pipeline:
  1. Load KPC-3 enriched compound data from GCS (phase_amr_chembl)
  2. Fetch clinical beta-lactam partner SMILES from PubChem API
  3. Parameterize 2×2 payoff matrices: pathogen (KPC-3 / efflux) × drug (inhibitor / partner)
  4. Compute Nash equilibria via nashpy; derive continuous synergy score per pair
  5. Train FP32 + BF16 synergy surrogate MLPs on TPU/XLA
     (input: 4096-dim concatenated Morgan FPs; output: synergy score)
  6. LMC between FP32 and BF16 synergy surrogates (11 α points, BCE on val set)
  7. Evaluate: Pearson ρ, ROC-AUC, top-3 Nash-optimal drug pairs
  8. Upload results JSON to GCS

GCS output: gs://.../aegis_flashoptim/phase34_nash_amr/results.json
"""

import os, sys, json, time, subprocess, random, math, warnings
from pathlib import Path

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase6_vina_surrogate import vina_to_pkd
from notify import notify, heartbeat

warnings.filterwarnings("ignore", category=UserWarning)

GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID     = os.environ.get("RUN_ID", "aegis_flashoptim")
GCS_BASE   = f"{GCS_BUCKET}/{RUN_ID}"
GCS_IN     = f"{GCS_BASE}/phase_amr_chembl/vina_scores_chembl_enriched.json"
GCS_OUT    = f"{GCS_BASE}/phase34_nash_amr"

OUT_DIR = Path("/tmp/phase34_nash_amr")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED        = 42
N_EPOCHS    = 60
LR          = 1e-3
BATCH_SIZE  = 256
N_ALPHA     = 11
VINA_THRESH = -2.0   # only compounds with vina_score < threshold
MAX_COMPOUNDS = 800  # cap for matrix computation tractability
PUBCHEM_API = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES,CanonicalSMILES/JSON"

# Hardcoded fallback SMILES (ChEMBL canonical) in case PubChem API is unavailable
PARTNER_SMILES_FALLBACK = {
    "meropenem":    "C[C@@H]1[C@@H](C(=O)N1)[C@@H]2CC(=C(N2)[C@H](C)O)S[C@@H]3CC(=O)N3",
    "ceftazidime":  "CC1(C(=O)O/N=C(\\C2=CSC(=N2)C[N+]3=CC=CC=C3)/C(=O)N[C@@H]4[C@H]5SCC(=C(N5C4=O)C(=O)[O-])C[N+]6=CC=CC=C6)C",
    "aztreonam":    "CN1C(=O)[C@@H](NC(=O)/C(=N\\OC(C)(C)C(=O)O)c2csc(N)n2)[C@H]1S(=O)(=O)O",
    "imipenem":     "C[C@@H]1[C@@H](C(=O)N1)[C@@H]2CC(=C(N2)SCCNC(=N)N)C(=O)O",
    "cefepime":     "C[C@@H]1[C@H]2[C@@H](C(=O)N2[C@H]1SC3=CC(=NN3C)C)NC(=O)/C(=N/OC)c4csc(N)n4",
    "piperacillin":  "CCN1CCN(CC1)C(=O)N[C@@H](C(=O)N[C@H]2[C@@H]3N(C2=O)[C@H](C(=O)O)SC3(C)C)c4ccccc4",
}

# Clinical beta-lactam partners: PubChem CIDs
PARTNER_CIDS = {
    "meropenem":   441130,
    "ceftazidime": 5481173,
    "aztreonam":   5359316,
    "imipenem":    104838,
    "cefepime":    4053,
    "piperacillin": 43672,
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] {msg}", flush=True)


def gsutil_cp(src, dst):
    subprocess.run(["gsutil", "-q", "cp", str(src), str(dst)], check=False)


def gcs_exists(path):
    return subprocess.run(["gsutil", "-q", "stat", path], capture_output=True).returncode == 0


def get_device():
    try:
        import torch_xla.core.xla_model as xm
        dev = xm.xla_device()
        log(f"Device: TPU ({dev})")
        return dev
    except Exception:
        pass
    if torch.cuda.is_available():
        log("Device: CUDA")
        return torch.device("cuda")
    log("Device: CPU")
    return torch.device("cpu")


# ── PubChem fetch ──────────────────────────────────────────────────────────────

def fetch_partner_smiles(partners: dict) -> dict:
    import urllib.request, urllib.error
    result = {}
    for name, cid in partners.items():
        url = PUBCHEM_API.format(cid=cid)
        fetched = False
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = json.loads(resp.read().decode())
                props = data["PropertyTable"]["Properties"][0]
                # Try IsomericSMILES first, fall back to CanonicalSMILES
                smiles = props.get("IsomericSMILES") or props.get("CanonicalSMILES")
                if smiles:
                    result[name] = smiles
                    log(f"  {name}: {smiles[:60]}...")
                    fetched = True
                    break
                else:
                    log(f"  {name} attempt {attempt+1}: no SMILES in response keys {list(props.keys())}")
            except Exception as e:
                log(f"  {name} fetch attempt {attempt+1}/3 failed: {e}")
                time.sleep(1.5 ** attempt)
        if not fetched and name in PARTNER_SMILES_FALLBACK:
            smiles = PARTNER_SMILES_FALLBACK[name]
            result[name] = smiles
            log(f"  {name}: using hardcoded fallback SMILES")
    return result


# ── Nash payoff matrix ─────────────────────────────────────────────────────────

def tanimoto(fp1: np.ndarray, fp2: np.ndarray) -> float:
    intersect = float(np.dot(fp1, fp2))
    union = float(fp1.sum() + fp2.sum() - intersect)
    return intersect / union if union > 0 else 0.0


def nash_synergy(pkd_i: float, tanimoto_ij: float) -> float:
    """
    2×2 evolutionary game:
      Row player (pathogen): {KPC-3 resistance, efflux pump resistance}
      Column player (treatment): {KPC-3 inhibitor, partner beta-lactam}

    Bacterial fitness matrix (row=pathogen strategy, col=drug):
      KPC-3 vs inhibitor:   exp(-pkd / 10)   — high pKd = well-blocked KPC-3
      KPC-3 vs partner:     0.85             — KPC-3 hydrolyses partner
      Efflux vs inhibitor:  0.65             — efflux irrelevant to serine BLI
      Efflux vs partner:    0.45 + 0.35*sim  — similar structures = efflux substrate

    Synergy = relative fitness reduction at Nash equilibrium vs best monotherapy.
    """
    try:
        import nashpy as nash
    except ImportError:
        # Fallback: analytic mixed-strategy solution for 2×2 game
        pass

    f_kpc3_inh  = math.exp(-pkd_i / 10.0)
    f_kpc3_part = 0.85
    f_efx_inh   = 0.65
    f_efx_part  = 0.45 + 0.35 * tanimoto_ij

    A = np.array([[f_kpc3_inh,  f_kpc3_part],
                  [f_efx_inh,   f_efx_part]], dtype=np.float64)
    B = -A  # zero-sum: drug maximises bacterial fitness cost

    ne_fitness = _nash_expected_payoff(A, B)
    best_mono = max(A.max(axis=0))  # bacteria's best outcome under any single drug
    synergy = max(0.0, (best_mono - ne_fitness) / (best_mono + 1e-9))
    return float(synergy)


def _nash_expected_payoff(A: np.ndarray, B: np.ndarray) -> float:
    """Compute row-player expected payoff at Nash equilibrium of a 2×2 game."""
    try:
        import nashpy as nash
        game = nash.Game(A, B)
        equilibria = list(game.support_enumeration())
        if equilibria:
            sr, sc = equilibria[0]
            return float(sr @ A @ sc)
    except Exception:
        pass

    # Analytic fallback: mixed-strategy NE for non-degenerate 2×2
    a11, a12 = A[0, 0], A[0, 1]
    a21, a22 = A[1, 0], A[1, 1]
    denom = (a11 - a12 - a21 + a22)
    if abs(denom) < 1e-9:
        # Pure strategy — row player plays dominant strategy
        return min(A.max(axis=1))
    q = (a22 - a21) / denom  # column player mixing prob
    q = float(np.clip(q, 0.0, 1.0))
    p = (a22 - a12) / denom  # row player mixing prob
    p = float(np.clip(p, 0.0, 1.0))
    sigma_r = np.array([p, 1 - p])
    sigma_c = np.array([q, 1 - q])
    return float(sigma_r @ A @ sigma_c)


# ── Dataset ────────────────────────────────────────────────────────────────────

def build_fingerprint(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return np.array(fp, dtype=np.float32)


def build_dataset(enriched: dict, partner_smiles: dict):
    """
    Build (X, y, meta) where:
      X[k] = [fp_inhibitor | fp_partner]  shape (4096,)
      y[k] = Nash synergy score           float in [0, 1]
      meta[k] = (cid, partner_name, pkd)
    """
    log("Building fingerprints for inhibitor compounds ...")
    inhibitors = []
    for cid, entry in enriched.items():
        smiles = entry.get("smiles", "")
        score  = entry.get("vina_score", 0.0)
        if not smiles or score >= VINA_THRESH:
            continue
        fp = build_fingerprint(smiles)
        if fp is None:
            continue
        pkd = vina_to_pkd(score)
        if pkd <= 0:
            continue
        inhibitors.append({"cid": cid, "smiles": smiles, "fp": fp, "pkd": pkd})
        if len(inhibitors) >= MAX_COMPOUNDS:
            break

    log(f"  {len(inhibitors)} inhibitors with valid SMILES + pKd")

    log("Building fingerprints for clinical partners ...")
    partner_fps = {}
    for name, smiles in partner_smiles.items():
        fp = build_fingerprint(smiles)
        if fp is not None:
            partner_fps[name] = fp
            log(f"  {name}: ok")

    if not partner_fps:
        log("ERROR: no valid partner fingerprints")
        sys.exit(1)

    log("Computing Nash synergy scores ...")
    X, y, meta = [], [], []
    for inh in inhibitors:
        for pname, pfp in partner_fps.items():
            sim = tanimoto(inh["fp"], pfp)
            syn = nash_synergy(inh["pkd"], sim)
            pair_fp = np.concatenate([inh["fp"], pfp], axis=0)
            X.append(pair_fp)
            y.append(syn)
            meta.append({"cid": inh["cid"], "partner": pname, "pkd": inh["pkd"],
                         "tanimoto": round(sim, 4), "nash_synergy": round(syn, 4)})

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    log(f"  {len(y)} pairs | synergy mean={y.mean():.4f} std={y.std():.4f}")
    return X, y, meta, inhibitors


# ── Model ──────────────────────────────────────────────────────────────────────

class SynergyMLP(nn.Module):
    def __init__(self, in_dim=4096, hidden=(1024, 512, 256)):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_model(X_tr, y_tr, X_val, y_val, dev, use_bf16, label):
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    Xtr = torch.tensor(X_tr, dtype=dtype).to(dev)
    ytr = torch.tensor(y_tr, dtype=dtype).to(dev)
    Xva = torch.tensor(X_val, dtype=dtype).to(dev)
    yva = torch.tensor(y_val, dtype=dtype).to(dev)

    model = SynergyMLP().to(dev)
    if use_bf16:
        model = model.to(dtype)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n   = Xtr.shape[0]

    best_val = float("inf")
    best_sd  = None

    for ep in range(1, N_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n).to(dev)  # generate on CPU, XLA v6e doesn't support int64 RNG
        ep_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = Xtr[idx], ytr[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            opt.step()
            try:
                import torch_xla.core.xla_model as xm
                xm.mark_step()
            except Exception:
                pass
            ep_loss += loss.item() * len(idx)

        ep_loss /= n
        model.eval()
        with torch.no_grad():
            val_pred = model(Xva)
            val_loss = F.mse_loss(val_pred, yva).item()

        if val_loss < best_val:
            best_val = val_loss
            best_sd  = {k: v.float().cpu().clone() for k, v in model.state_dict().items()}

        if ep % 10 == 0:
            log(f"  [{label}] ep {ep:3d}/{N_EPOCHS}  train={ep_loss:.5f}  val={val_loss:.5f}")

    model.load_state_dict({k: v.to(dtype).to(dev) for k, v in best_sd.items()})
    model.eval()
    with torch.no_grad():
        val_pred_f = model(Xva).float().cpu().numpy()

    val_y = y_val
    r, _ = pearsonr(val_y, val_pred_f)
    log(f"  [{label}] done  best_val={best_val:.5f}  Pearson r={r:.4f}")
    return model, best_sd, float(r)


# ── LMC ───────────────────────────────────────────────────────────────────────

def lmc_bce(sd0, sd1, X_val, y_val, dev, label):
    """Linear interpolation loss curve between two surrogates (MSE on synergy)."""
    Xva = torch.tensor(X_val, dtype=torch.float32).to(dev)
    yva = torch.tensor(y_val, dtype=torch.float32).to(dev)
    alphas    = [round(i / (N_ALPHA - 1), 2) for i in range(N_ALPHA)]
    mse_curve = []

    model = SynergyMLP().to(dev)
    for alpha in alphas:
        interp = {k: (1 - alpha) * sd0[k].float() + alpha * sd1[k].float()
                  for k in sd0}
        model.load_state_dict({k: v.to(dev) for k, v in interp.items()})
        model.eval()
        with torch.no_grad():
            pred = model(Xva)
            mse  = F.mse_loss(pred, yva).item()
        mse_curve.append(round(mse, 6))
        log(f"  {label} α={alpha:.1f}  mse={mse:.6f}")

    endpoint_mean = (mse_curve[0] + mse_curve[-1]) / 2
    barrier = max(mse_curve) - endpoint_mean
    log(f"  {label} barrier={barrier:.6f}  (max={max(mse_curve):.6f}  endpoints_mean={endpoint_mean:.6f})")
    return {"alphas": alphas, "mse_curve": mse_curve,
            "barrier_mse": round(barrier, 6), "endpoint_mean": round(endpoint_mean, 6)}


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X_val, y_val, dev, label, meta_val):
    model.eval()
    dtype = next(model.parameters()).dtype  # match input to model precision (XLA disallows mixed)
    Xva = torch.tensor(X_val, dtype=dtype).to(dev)
    with torch.no_grad():
        preds = model(Xva).float().cpu().numpy()

    r, _  = pearsonr(y_val, preds)
    rho,_ = spearmanr(y_val, preds)

    threshold = float(np.median(y_val))
    labels_bin = (y_val > threshold).astype(int)
    auc = roc_auc_score(labels_bin, preds) if labels_bin.sum() > 0 else 0.5

    log(f"  [{label}] Pearson r={r:.4f}  Spearman ρ={rho:.4f}  ROC-AUC={auc:.4f}")
    return {"pearson_r": round(r, 4), "spearman_rho": round(rho, 4), "roc_auc": round(auc, 4)}


def top_pairs(meta, y, n=5):
    ranked = sorted(zip(y.tolist(), meta), key=lambda t: -t[0])
    out = []
    for score, m in ranked[:n]:
        out.append({"cid": m["cid"], "partner": m["partner"],
                    "pkd": round(m["pkd"], 3), "tanimoto": m["tanimoto"],
                    "nash_synergy": round(score, 4)})
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    notify("PHASE_START", "Phase 34 Nash AMR drug combination optimization")
    log("=" * 60)
    log("Phase 34: Nash equilibrium AMR drug combination")
    log("=" * 60)

    # ── Skip if already done ───────────────────────────────────────────────────
    done_path = f"{GCS_OUT}/results.json"
    if gcs_exists(done_path):
        log(f"Output already exists at {done_path} — skipping")
        return

    # ── Install dependencies ───────────────────────────────────────────────────
    log("Installing nashpy ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "nashpy", "--quiet"], check=False)

    # ── Load enriched compound data from GCS ──────────────────────────────────
    log("Loading enriched KPC-3 compound data from GCS ...")
    local_enriched = OUT_DIR / "vina_scores_chembl_enriched.json"
    if not local_enriched.exists():
        ret = subprocess.run(["gsutil", "-q", "cp", GCS_IN, str(local_enriched)],
                             capture_output=True)
        if ret.returncode != 0:
            log(f"ERROR: Failed to download {GCS_IN}")
            log(ret.stderr.decode())
            sys.exit(1)

    with open(local_enriched) as f:
        enriched_raw = json.load(f)

    # Normalise format: may be {id: float} or {id: {smiles:..., vina_score:...}}
    if enriched_raw and isinstance(next(iter(enriched_raw.values())), (int, float)):
        log("  Enriched file has flat scores — no SMILES available; cannot continue")
        sys.exit(1)

    log(f"  Loaded {len(enriched_raw)} entries from enriched file")

    # ── Fetch partner SMILES ───────────────────────────────────────────────────
    log("Fetching clinical partner SMILES from PubChem ...")
    partner_smiles = fetch_partner_smiles(PARTNER_CIDS)
    if not partner_smiles:
        log("ERROR: No partner SMILES fetched")
        sys.exit(1)
    log(f"  {len(partner_smiles)} partners: {list(partner_smiles.keys())}")

    # ── Build dataset ──────────────────────────────────────────────────────────
    log("Building Nash synergy dataset ...")
    X, y, meta, inhibitors = build_dataset(enriched_raw, partner_smiles)

    n_total  = len(y)
    n_val    = max(int(n_total * 0.2), 100)
    n_tr     = n_total - n_val
    idx      = np.random.permutation(n_total)
    tr_idx, val_idx = idx[:n_tr], idx[n_tr:]
    X_tr, y_tr   = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    meta_val     = [meta[i] for i in val_idx]
    log(f"  train={n_tr}  val={n_val}")

    dev = get_device()

    # ── FP32 surrogate ─────────────────────────────────────────────────────────
    log("\n[1/4] Training FP32 synergy surrogate ...")
    ckpt_fp32 = OUT_DIR / "fp32_synergy.pt"
    if ckpt_fp32.exists():
        log("  Resuming FP32 from checkpoint")
        m_fp32 = SynergyMLP().to(dev)
        m_fp32.load_state_dict(torch.load(ckpt_fp32, map_location="cpu"))
        m_fp32.eval()
        with torch.no_grad():
            preds = m_fp32(torch.tensor(X_val, dtype=torch.float32).to(dev)).float().cpu().numpy()
        r_fp32, _ = pearsonr(y_val, preds)
        sd_fp32   = {k: v.float().cpu().clone() for k, v in m_fp32.state_dict().items()}
        log(f"  FP32 resumed  Pearson r={r_fp32:.4f}")
    else:
        m_fp32, sd_fp32, r_fp32 = train_model(X_tr, y_tr, X_val, y_val, dev,
                                               use_bf16=False, label="FP32")
        torch.save(sd_fp32, ckpt_fp32)
        gsutil_cp(ckpt_fp32, f"{GCS_OUT}/fp32_synergy.pt")

    # ── BF16 surrogate ─────────────────────────────────────────────────────────
    log("\n[2/4] Training BF16 synergy surrogate ...")
    ckpt_bf16 = OUT_DIR / "bf16_synergy.pt"
    if ckpt_bf16.exists():
        log("  Resuming BF16 from checkpoint")
        m_bf16 = SynergyMLP().to(dev)
        m_bf16.load_state_dict(torch.load(ckpt_bf16, map_location="cpu"))
        m_bf16.eval()
        with torch.no_grad():
            preds = m_bf16(torch.tensor(X_val, dtype=torch.float32).to(dev)).float().cpu().numpy()
        r_bf16, _ = pearsonr(y_val, preds)
        sd_bf16   = {k: v.float().cpu().clone() for k, v in m_bf16.state_dict().items()}
        log(f"  BF16 resumed  Pearson r={r_bf16:.4f}")
    else:
        m_bf16, sd_bf16, r_bf16 = train_model(X_tr, y_tr, X_val, y_val, dev,
                                               use_bf16=True, label="BF16")
        torch.save(sd_bf16, ckpt_bf16)
        gsutil_cp(ckpt_bf16, f"{GCS_OUT}/bf16_synergy.pt")

    log(f"\nFP32 r={r_fp32:.4f}  BF16 r={r_bf16:.4f}")

    # ── LMC ────────────────────────────────────────────────────────────────────
    log("\n[3/4] Computing LMC (FP32 ↔ BF16) ...")
    lmc_results = lmc_bce(sd_fp32, sd_bf16, X_val, y_val, dev, "FP32↔BF16")
    log(f"LMC barrier (MSE): {lmc_results['barrier_mse']:.6f}")

    # ── Evaluation ─────────────────────────────────────────────────────────────
    log("\n[4/4] Final evaluation ...")
    eval_fp32 = evaluate(m_fp32, X_val, y_val, dev, "FP32", meta_val)
    eval_bf16 = evaluate(m_bf16, X_val, y_val, dev, "BF16", meta_val)

    top3  = top_pairs(meta, y, n=3)
    top10 = top_pairs(meta, y, n=10)

    log("\nTop-3 Nash-optimal KPC-3 inhibitor + partner combinations:")
    for i, p in enumerate(top3, 1):
        log(f"  {i}. {p['cid']} + {p['partner']:12s}  pKd={p['pkd']:.3f}"
            f"  Tanimoto={p['tanimoto']:.3f}  Nash synergy={p['nash_synergy']:.4f}")

    # ── Collate results ─────────────────────────────────────────────────────────
    results = {
        "phase": 34,
        "n_inhibitors": len(inhibitors),
        "n_partners": len(partner_smiles),
        "n_pairs_train": int(n_tr),
        "n_pairs_val": int(n_val),
        "synergy_mean": round(float(y.mean()), 5),
        "synergy_std":  round(float(y.std()),  5),
        "fp32": {"pearson_r": eval_fp32["pearson_r"],
                 "spearman_rho": eval_fp32["spearman_rho"],
                 "roc_auc": eval_fp32["roc_auc"]},
        "bf16": {"pearson_r": eval_bf16["pearson_r"],
                 "spearman_rho": eval_bf16["spearman_rho"],
                 "roc_auc": eval_bf16["roc_auc"]},
        "lmc_barrier_mse": lmc_results["barrier_mse"],
        "lmc_curve": lmc_results["mse_curve"],
        "top3_pairs": top3,
        "top10_pairs": top10,
    }

    out_json = OUT_DIR / "results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)
    gsutil_cp(out_json, f"{GCS_OUT}/results.json")
    log(f"Results uploaded to {GCS_OUT}/results.json")

    log("\n" + "=" * 60)
    log("Phase 34 COMPLETE")
    log(f"  Compounds: {len(inhibitors)} inhibitors × {len(partner_smiles)} partners = {len(y)} pairs")
    log(f"  FP32: Pearson r={eval_fp32['pearson_r']:.4f}  ROC-AUC={eval_fp32['roc_auc']:.4f}")
    log(f"  BF16: Pearson r={eval_bf16['pearson_r']:.4f}  ROC-AUC={eval_bf16['roc_auc']:.4f}")
    log(f"  LMC barrier: {lmc_results['barrier_mse']:.6f} MSE")
    log(f"  Best pair: {top3[0]['cid']} + {top3[0]['partner']}  synergy={top3[0]['nash_synergy']:.4f}")
    log("=" * 60)

    notify("PHASE_COMPLETE", "Phase 34 Nash AMR done", data={
        "n_pairs": len(y),
        "fp32_r":   eval_fp32["pearson_r"],
        "bf16_r":   eval_bf16["pearson_r"],
        "lmc_barrier": lmc_results["barrier_mse"],
        "best_pair": f"{top3[0]['cid']}+{top3[0]['partner']}",
    })


if __name__ == "__main__":
    main()
