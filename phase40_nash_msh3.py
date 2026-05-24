#!/usr/bin/env python3
"""
phase40_nash_msh3.py
--------------------
Phase 40: Nash equilibrium–guided drug combination optimisation for MSH3-driven pathology.

Adapted from Phase 34 (KPC-3 Nash AMR) for the MSH3 ATPase target.
MSH3 drives trinucleotide repeat expansion in Huntington's Disease and
mismatch repair in MSI-H cancers. Combining MSH3 ATPase inhibitors
(from Phase 38 hits) with PARP inhibitors creates orthogonal pressure
on the two adaptation strategies available to cancer/HD cells.

Pipeline:
  1. Load Phase 38 docking scores from GCS + pubchem_fda SMILES
  2. Fetch PARP inhibitor partner SMILES from PubChem
  3. Parameterise 2×2 payoff matrices:
       Row    (cancer): {MSH3-overexpression, RPA/FAN1-bypass}
       Column (drug):   {MSH3 ATPase inhibitor, PARP inhibitor partner}
  4. Compute Nash equilibria → synergy score per inhibitor-partner pair
  5. Train FP32 + BF16 SynergyMLP surrogates on TPU/XLA
  6. LMC between FP32 ↔ BF16 surrogates (11 α points, MSE on val)
  7. Evaluate: Pearson ρ, Spearman ρ, ROC-AUC, top-10 Nash-optimal pairs
  8. Upload results JSON to GCS

GCS output: gs://.../aegis_flashoptim/phase40_nash_msh3/results.json
"""

import os, sys, json, time, subprocess, random, math, warnings
from pathlib import Path

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

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID     = os.environ.get("RUN_ID", "aegis_flashoptim")
GCS_BASE   = f"{GCS_BUCKET}/{RUN_ID}"
# Load Phase 38 docking results as input
GCS_DOCK   = f"{GCS_BASE}/phase38_msh3_retry/docking_checkpoint.json"
GCS_OUT    = f"{GCS_BASE}/phase40_nash_msh3"

OUT_DIR = Path("/tmp/phase40_nash_msh3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/phase2_data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED          = 42
N_EPOCHS      = 60
LR            = 1e-3
BATCH_SIZE    = 256
N_ALPHA       = 11
VINA_THRESH   = -2.0
MAX_COMPOUNDS = 800
PUBCHEM_API   = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES,CanonicalSMILES/JSON"

# PARP inhibitors (clinical) + laquinimod (HD-relevant immunomodulator)
PARTNER_CIDS = {
    "olaparib":    23725625,
    "niraparib":   24958200,
    "veliparib":   11975812,
    "talazoparib": 44151613,
    "rucaparib":    9908089,
    "laquinimod":    213048,
}

PARTNER_SMILES_FALLBACK = {
    "olaparib":    "O=C(c1ccc(N2CC(=O)N3CCCc3c2=O)cc1F)N1CCN(C(=O)c2ccncc2)CC1",
    "niraparib":   "O=C(N1CCC(c2ccc(-c3ccncc3)cc2)CC1)c1c[nH]c2ccccc12",
    "veliparib":   "O=C(N1c2ccc(F)cc2CC1CN1CCCCC1)c1[nH]nc2ccccc12",
    "talazoparib":  "O=c1[nH]nc2c(c1CN1CCN(C(=O)c3ccccc3F)CC1)cccc2",
    "rucaparib":   "O=C(N1CCN(Cc2c[nH]c3ccc(F)cc23)CC1)c1cccc2ccccc12",
    "laquinimod":  "CCN1C(=O)C(NC(=O)c2ccc(Cl)cc2OC)=CC1=O",
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
                smiles = props.get("IsomericSMILES") or props.get("CanonicalSMILES")
                if smiles:
                    result[name] = smiles
                    log(f"  {name}: {smiles[:60]}...")
                    fetched = True
                    break
            except Exception as e:
                log(f"  {name} fetch attempt {attempt+1}/3 failed: {e}")
                time.sleep(1.5 ** attempt)
        if not fetched and name in PARTNER_SMILES_FALLBACK:
            result[name] = PARTNER_SMILES_FALLBACK[name]
            log(f"  {name}: using hardcoded fallback SMILES")
    return result


def tanimoto(fp1: np.ndarray, fp2: np.ndarray) -> float:
    intersect = float(np.dot(fp1, fp2))
    union = float(fp1.sum() + fp2.sum() - intersect)
    return intersect / union if union > 0 else 0.0


def nash_synergy_msh3(pkd_i: float, tanimoto_ij: float) -> float:
    """
    2×2 evolutionary game for MSH3-driven cancer / Huntington's Disease:
      Row player (cancer): {MSH3 overexpression, RPA/FAN1-bypass}
      Column player (drug): {MSH3 ATPase inhibitor, PARP inhibitor}

    Cancer survival matrix (row=cancer strategy, col=drug):
      MSH3-ovexp × MSH3-inh:  exp(-pkd_i / 10)   — high pKd blocks ATPase; reduces MMR expansion
      MSH3-ovexp × PARP-inh:  0.80               — PARP-inh alone cannot suppress MSH3-driven MMR
      RPA/FAN1   × MSH3-inh:  0.65               — ALT path engaged; MSH3 inhibition less potent
      RPA/FAN1   × PARP-inh:  0.40 + 0.35*sim    — PARP hits ALT replication; cross-reactivity bonus

    Synergy = relative cancer survival reduction at Nash equil. vs best monotherapy.
    """
    f_msh3_inh   = math.exp(-pkd_i / 10.0)
    f_msh3_parp  = 0.80
    f_alt_inh    = 0.65
    f_alt_parp   = 0.40 + 0.35 * tanimoto_ij

    A = np.array([[f_msh3_inh,  f_msh3_parp],
                  [f_alt_inh,   f_alt_parp ]], dtype=np.float64)
    B = -A  # zero-sum

    ne_fitness = _nash_expected_payoff(A, B)
    best_mono  = max(A.max(axis=0))
    synergy    = max(0.0, (best_mono - ne_fitness) / (best_mono + 1e-9))
    return float(synergy)


def _nash_expected_payoff(A: np.ndarray, B: np.ndarray) -> float:
    try:
        import nashpy as nash
        game = nash.Game(A, B)
        equilibria = list(game.support_enumeration())
        if equilibria:
            sr, sc = equilibria[0]
            return float(sr @ A @ sc)
    except Exception:
        pass

    # Analytic mixed-strategy NE for non-degenerate 2×2
    a11, a12 = A[0, 0], A[0, 1]
    a21, a22 = A[1, 0], A[1, 1]
    denom = a11 - a12 - a21 + a22
    if abs(denom) < 1e-9:
        return float(min(A.max(axis=1)))
    q = float(np.clip((a22 - a21) / denom, 0.0, 1.0))
    p = float(np.clip((a22 - a12) / denom, 0.0, 1.0))
    return float(np.array([p, 1 - p]) @ A @ np.array([q, 1 - q]))


def build_fingerprint(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048), dtype=np.float32)


def build_dataset(enriched: dict, partner_smiles: dict):
    """
    Build (X, y, meta) where:
      X[k] = [fp_msh3_inhibitor | fp_parp_partner]  shape (4096,)
      y[k] = Nash synergy score                       float in [0, 1]
    """
    log("Building fingerprints for MSH3 inhibitor compounds ...")
    inhibitors = []
    for name, entry in enriched.items():
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
        inhibitors.append({"name": name, "smiles": smiles, "fp": fp, "pkd": pkd})
        if len(inhibitors) >= MAX_COMPOUNDS:
            break

    log(f"  {len(inhibitors)} MSH3 inhibitors with valid SMILES + pKd")

    log("Building fingerprints for PARP inhibitor partners ...")
    partner_fps = {}
    for pname, smiles in partner_smiles.items():
        fp = build_fingerprint(smiles)
        if fp is not None:
            partner_fps[pname] = fp
            log(f"  {pname}: ok")

    if not partner_fps:
        log("ERROR: no valid partner fingerprints")
        sys.exit(1)

    log("Computing Nash synergy scores ...")
    X, y, meta = [], [], []
    for inh in inhibitors:
        for pname, pfp in partner_fps.items():
            sim = tanimoto(inh["fp"], pfp)
            syn = nash_synergy_msh3(inh["pkd"], sim)
            pair_fp = np.concatenate([inh["fp"], pfp], axis=0)
            X.append(pair_fp)
            y.append(syn)
            meta.append({"name": inh["name"], "partner": pname, "pkd": inh["pkd"],
                         "tanimoto": round(sim, 4), "nash_synergy": round(syn, 4)})

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    log(f"  {len(y)} pairs | synergy mean={y.mean():.4f} std={y.std():.4f}")
    return X, y, meta, inhibitors


class SynergyMLP(nn.Module):
    def __init__(self, in_dim=4096, hidden=(1024, 512, 256)):
        super().__init__()
        layers, prev = [], in_dim
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
    best_val, best_sd = float("inf"), None

    for ep in range(1, N_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n).to(dev)
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
            val_loss = F.mse_loss(model(Xva), yva).item()

        if val_loss < best_val:
            best_val = val_loss
            best_sd  = {k: v.float().cpu().clone() for k, v in model.state_dict().items()}

        if ep % 10 == 0:
            log(f"  [{label}] ep {ep:3d}/{N_EPOCHS}  train={ep_loss:.5f}  val={val_loss:.5f}")
        heartbeat(f"Phase40_{label}", ep, {"val_loss": val_loss})

    model.load_state_dict({k: v.to(dtype).to(dev) for k, v in best_sd.items()})
    model.eval()
    with torch.no_grad():
        preds = model(Xva).float().cpu().numpy()
    r, _ = pearsonr(y_val, preds)
    log(f"  [{label}] done  best_val={best_val:.5f}  Pearson r={r:.4f}")
    return model, best_sd, float(r)


def lmc_mse(sd0, sd1, X_val, y_val, dev, label):
    Xva = torch.tensor(X_val, dtype=torch.float32).to(dev)
    yva = torch.tensor(y_val, dtype=torch.float32).to(dev)
    alphas    = [round(i / (N_ALPHA - 1), 2) for i in range(N_ALPHA)]
    mse_curve = []
    model = SynergyMLP().to(dev)
    for alpha in alphas:
        interp = {k: (1 - alpha) * sd0[k].float() + alpha * sd1[k].float() for k in sd0}
        model.load_state_dict({k: v.to(dev) for k, v in interp.items()})
        model.eval()
        with torch.no_grad():
            mse = F.mse_loss(model(Xva), yva).item()
        mse_curve.append(round(mse, 6))
        log(f"  {label} α={alpha:.1f}  mse={mse:.6f}")
    endpoint_mean = (mse_curve[0] + mse_curve[-1]) / 2
    barrier = max(mse_curve) - endpoint_mean
    log(f"  {label} barrier={barrier:.6f}")
    return {"alphas": alphas, "mse_curve": mse_curve,
            "barrier_mse": round(barrier, 6), "endpoint_mean": round(endpoint_mean, 6)}


def evaluate(model, X_val, y_val, dev, label):
    model.eval()
    dtype = next(model.parameters()).dtype
    Xva = torch.tensor(X_val, dtype=dtype).to(dev)
    with torch.no_grad():
        preds = model(Xva).float().cpu().numpy()
    r, _   = pearsonr(y_val, preds)
    rho, _ = spearmanr(y_val, preds)
    threshold  = float(np.median(y_val))
    labels_bin = (y_val > threshold).astype(int)
    auc = roc_auc_score(labels_bin, preds) if labels_bin.sum() > 0 else 0.5
    log(f"  [{label}] Pearson r={r:.4f}  Spearman ρ={rho:.4f}  ROC-AUC={auc:.4f}")
    return {"pearson_r": round(r, 4), "spearman_rho": round(rho, 4), "roc_auc": round(auc, 4)}


def top_pairs(meta, y, n=10):
    ranked = sorted(zip(y.tolist(), meta), key=lambda t: -t[0])
    return [{"name": m["name"], "partner": m["partner"],
             "pkd": round(m["pkd"], 3), "tanimoto": m["tanimoto"],
             "nash_synergy": round(score, 4)}
            for score, m in ranked[:n]]


def main():
    notify("PHASE_START", "Phase 40 Nash MSH3 drug combination optimisation")
    log("=" * 65)
    log("  Phase 40 — Nash MSH3 + PARP inhibitor combination")
    log("=" * 65)

    done_path = f"{GCS_OUT}/results.json"
    if gcs_exists(done_path):
        log(f"Output already exists — skipping"); return

    log("Installing nashpy ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "nashpy", "--quiet"], check=False)

    # ── Load Phase 38 docking scores ──────────────────────────────────────────
    log("Loading Phase 38 MSH3 docking data from GCS ...")
    dock_local = OUT_DIR / "phase38_docking_checkpoint.json"
    if not dock_local.exists():
        ret = subprocess.run(["gsutil", "-q", "cp", GCS_DOCK, str(dock_local)],
                             capture_output=True)
        if ret.returncode != 0:
            log(f"ERROR: failed to download {GCS_DOCK}")
            log(ret.stderr.decode()); sys.exit(1)

    with open(dock_local) as f:
        dock_data = json.load(f)
    raw_scores = dock_data.get("scores", dock_data)  # {name: vina_score}
    log(f"  Loaded {len(raw_scores)} Phase 38 docked compounds")

    # ── Cross-reference with FDA SMILES ───────────────────────────────────────
    log("Cross-referencing with FDA SMILES ...")
    fda_path = DATA_DIR / "pubchem_fda.json"
    if not fda_path.exists():
        subprocess.run(["gsutil", "-q", "cp", f"{GCS_BASE}/pubchem_fda.json", str(fda_path)], check=True)
    with open(fda_path) as f:
        fda_raw = json.load(f)
    if isinstance(fda_raw, list):
        fda_lookup = {row[0]: row[1] for row in fda_raw if len(row) >= 2}
    else:
        fda_lookup = {k: v.get("smiles", "") if isinstance(v, dict) else v
                      for k, v in fda_raw.items()}

    enriched = {}
    for name, vina_score in raw_scores.items():
        smiles = fda_lookup.get(name, "")
        if smiles:
            enriched[name] = {"smiles": smiles, "vina_score": float(vina_score)}
    log(f"  {len(enriched)} compounds with SMILES + vina score")

    # ── Fetch partner SMILES ───────────────────────────────────────────────────
    log("Fetching PARP inhibitor partner SMILES from PubChem ...")
    partner_smiles = fetch_partner_smiles(PARTNER_CIDS)
    if not partner_smiles:
        log("ERROR: No partner SMILES fetched"); sys.exit(1)
    log(f"  {len(partner_smiles)} partners: {list(partner_smiles.keys())}")

    # ── Build dataset ──────────────────────────────────────────────────────────
    log("Building Nash synergy dataset ...")
    X, y, meta, inhibitors = build_dataset(enriched, partner_smiles)

    n_total = len(y)
    n_val   = max(int(n_total * 0.2), 100)
    n_tr    = n_total - n_val
    idx     = np.random.permutation(n_total)
    tr_idx, val_idx = idx[:n_tr], idx[n_tr:]
    X_tr, y_tr   = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]
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
    lmc_results = lmc_mse(sd_fp32, sd_bf16, X_val, y_val, dev, "FP32↔BF16")
    log(f"LMC barrier (MSE): {lmc_results['barrier_mse']:.6f}")

    # ── Evaluation ─────────────────────────────────────────────────────────────
    log("\n[4/4] Final evaluation ...")
    eval_fp32 = evaluate(m_fp32, X_val, y_val, dev, "FP32")
    eval_bf16 = evaluate(m_bf16, X_val, y_val, dev, "BF16")

    top3  = top_pairs(meta, y, n=3)
    top10 = top_pairs(meta, y, n=10)

    log("\nTop-3 Nash-optimal MSH3 inhibitor + PARP partner combinations:")
    for i, p in enumerate(top3, 1):
        log(f"  {i}. {p['name']} + {p['partner']:12s}  pKd={p['pkd']:.3f}"
            f"  Tanimoto={p['tanimoto']:.3f}  Nash synergy={p['nash_synergy']:.4f}")

    results = {
        "phase": 40,
        "input_source": "phase38_msh3_retry docking checkpoint",
        "target": "MSH3 ATPase 3THW chain A (ADP-derived box)",
        "partners": list(partner_smiles.keys()),
        "game": {
            "row_player": "cancer cell",
            "row_strategies": ["MSH3-overexpression", "RPA/FAN1-bypass"],
            "col_player": "treatment",
            "col_strategies": ["MSH3 ATPase inhibitor", "PARP inhibitor"],
        },
        "n_inhibitors": len(inhibitors),
        "n_partners": len(partner_smiles),
        "n_pairs_train": int(n_tr),
        "n_pairs_val": int(n_val),
        "synergy_mean": round(float(y.mean()), 5),
        "synergy_std":  round(float(y.std()),  5),
        "fp32": eval_fp32,
        "bf16": eval_bf16,
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

    log("\n" + "=" * 65)
    log("Phase 40 COMPLETE")
    log(f"  Inhibitors: {len(inhibitors)}  Partners: {len(partner_smiles)}  Pairs: {len(y)}")
    log(f"  FP32: Pearson r={eval_fp32['pearson_r']:.4f}  ROC-AUC={eval_fp32['roc_auc']:.4f}")
    log(f"  BF16: Pearson r={eval_bf16['pearson_r']:.4f}  ROC-AUC={eval_bf16['roc_auc']:.4f}")
    log(f"  LMC barrier: {lmc_results['barrier_mse']:.6f} MSE")
    log(f"  Best pair: {top3[0]['name']} + {top3[0]['partner']}  synergy={top3[0]['nash_synergy']:.4f}")
    log("=" * 65)

    notify("PHASE_COMPLETE", "Phase 40 Nash MSH3 done", data={
        "n_pairs": len(y),
        "fp32_r": eval_fp32["pearson_r"],
        "bf16_r": eval_bf16["pearson_r"],
        "lmc_barrier": lmc_results["barrier_mse"],
        "best_pair": f"{top3[0]['name']}+{top3[0]['partner']}",
    })


if __name__ == "__main__":
    main()
