#!/usr/bin/env python3
"""
phase30_msh3_3thw.py
---------------------
MSH3 3THW molecular docking screen + surrogate BO.

Downloads 3THW.pdb, extracts chain B (MSH3 ATPase), preps receptor with obabel,
screens all 2639 FDA compounds via AutoDock Vina, trains MLP surrogate, runs EI BO.

Docking box (ATP binding site / walker A motif):
  center = (-1.32, 33.12, -36.005)
  size   = (30, 30, 30)

GCS output: gs://.../aegis_flashoptim/phase30_msh3_3thw/results.json
"""

import os, sys, json, time, subprocess, tempfile, random, math
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
GCS_OUT    = f"{GCS_BASE}/phase30_msh3_3thw"
DATA_DIR   = Path(os.environ.get("DATA_DIR", "/tmp/phase2_data"))

OUT_DIR = Path("/tmp/phase30_msh3_3thw")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

RCSB_URL       = "https://files.rcsb.org/download/3THW.pdb"
CHAIN_ID       = "B"

# Docking box — ATP binding site (Walker A motif)
BOX_CENTER     = (-1.32, 33.12, -36.005)
BOX_SIZE       = (30, 30, 30)

CHECKPOINT_EVERY = 100
N_EPOCHS_SURR    = 50
LR_SURR          = 1e-3
BATCH_SIZE       = 128
SEED             = 42
N_BO             = 20
TOP_K_BO         = 5


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] {msg}", flush=True)


def gsutil_cp(src, dst):
    subprocess.run(["gsutil", "-q", "cp", str(src), str(dst)], check=False)


def gcs_exists(gcs_path):
    return subprocess.run(["gsutil", "-q", "stat", gcs_path],
                          capture_output=True).returncode == 0


# ── Receptor preparation ──────────────────────────────────────────────────────

def download_pdb():
    pdb_path = OUT_DIR / "3THW.pdb"
    if not pdb_path.exists():
        log("Downloading 3THW.pdb from RCSB...")
        import urllib.request
        urllib.request.urlretrieve(RCSB_URL, str(pdb_path))
        log(f"  Downloaded → {pdb_path}")
    return pdb_path


def extract_chain_b(pdb_path):
    """Extract chain B using biopython."""
    chain_path = OUT_DIR / "msh3_chain.pdb"
    if chain_path.exists():
        log(f"Chain B already extracted: {chain_path}")
        return chain_path

    try:
        from Bio import PDB
        parser  = PDB.PDBParser(QUIET=True)
        struct  = parser.get_structure("3THW", str(pdb_path))
        io      = PDB.PDBIO()
        io.set_structure(struct)

        class ChainSelect(PDB.Select):
            def accept_chain(self, chain):
                return chain.id == CHAIN_ID

        io.save(str(chain_path), ChainSelect())
        log(f"Extracted chain {CHAIN_ID} → {chain_path}")
    except ImportError:
        log("WARNING: biopython not available — using grep-based extraction")
        lines = []
        with open(pdb_path) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM", "TER", "END")):
                    if len(line) > 21 and line[21] == CHAIN_ID:
                        lines.append(line)
                elif line.startswith("END"):
                    lines.append(line)
        chain_path.write_text("".join(lines))
        log(f"Extracted chain {CHAIN_ID} (grep fallback) → {chain_path}")

    return chain_path


def prep_receptor(chain_path):
    """Convert chain PDB → PDBQT using obabel with rigid receptor flag."""
    receptor_path = OUT_DIR / "msh3_3thw_receptor.pdbqt"
    if receptor_path.exists():
        log(f"Receptor PDBQT already exists: {receptor_path}")
        return receptor_path

    log("Preparing receptor PDBQT with obabel...")
    # -xr: rigid receptor (no ROOT/BRANCH/TORSION — Vina format)
    # -p 7.4: protonate at pH 7.4
    # --partialcharge gasteiger: required for PDBQT
    res = subprocess.run(
        ["obabel", str(chain_path), "-O", str(receptor_path),
         "-xr", "-p", "7.4", "--partialcharge", "gasteiger"],
        capture_output=True, text=True
    )
    if res.returncode != 0 or not receptor_path.exists():
        log(f"  obabel stdout: {res.stdout[:200]}")
        log(f"  obabel stderr: {res.stderr[:200]}")
        log("ERROR: receptor preparation failed")
        return None

    log(f"Receptor PDBQT → {receptor_path}")
    return receptor_path


# ── Ligand docking ────────────────────────────────────────────────────────────

def smiles_to_pdbqt(smiles, out_path):
    """Convert SMILES → 3D PDBQT using RDKit + obabel or obabel alone."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        mol = Chem.AddHs(mol)
        ps  = AllChem.ETKDGv3()
        ps.randomSeed = 42
        if AllChem.EmbedMolecule(mol, ps) < 0:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

        sdf_path = out_path.with_suffix(".sdf")
        writer   = Chem.SDWriter(str(sdf_path))
        writer.write(mol)
        writer.close()

        res = subprocess.run(
            ["obabel", str(sdf_path), "-O", str(out_path), "-p", "7.4"],
            capture_output=True, timeout=30
        )
        sdf_path.unlink(missing_ok=True)
        return res.returncode == 0 and out_path.exists()

    except ImportError:
        # Fallback: obabel SMILES → PDBQT with gen3d
        try:
            res = subprocess.run(
                ["obabel", "-ismi", smiles, "-O", str(out_path),
                 "--gen3d", "--minimize", "-p", "7.4"],
                input=smiles, capture_output=True, text=True, timeout=60
            )
            return res.returncode == 0 and out_path.exists()
        except Exception:
            return False


def dock_compound(receptor_pdbqt, lig_pdbqt):
    """Run AutoDock Vina and return best affinity (kcal/mol). Returns 0.0 on failure."""
    try:
        from vina import Vina
        v = Vina(sf_name="vina", verbosity=0)
        v.set_receptor(str(receptor_pdbqt))
        v.set_ligand_from_file(str(lig_pdbqt))
        v.compute_vina_maps(
            center=list(BOX_CENTER),
            box_size=list(BOX_SIZE)
        )
        v.dock(exhaustiveness=8, n_poses=3)
        energies = v.energies(n_poses=1)
        return float(energies[0][0])
    except Exception as e:
        return 0.0


# ── MLP surrogate ─────────────────────────────────────────────────────────────

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
        self.train()
        with torch.no_grad():
            preds = torch.stack([self.forward(x) for _ in range(10)], dim=0)
        self.eval()
        return preds.mean(0), preds.std(0).clamp(min=1e-4)


def build_fp(smiles):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        return np.array(fp, dtype=np.float32)
    except Exception:
        return None


def train_surrogate(X_tr, y_tr, X_val, y_val):
    torch.manual_seed(SEED)
    model = FingerprintSurrogate()
    optim = torch.optim.AdamW(model.parameters(), lr=LR_SURR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=N_EPOCHS_SURR, eta_min=1e-5)

    best_mse, best_sd = float("inf"), None

    for ep in range(1, N_EPOCHS_SURR + 1):
        model.train()
        perm = torch.randperm(len(X_tr))
        X_s, y_s = X_tr[perm], y_tr[perm]
        for i in range(0, len(X_s), BATCH_SIZE):
            xb, yb = X_s[i:i+BATCH_SIZE], y_s[i:i+BATCH_SIZE]
            optim.zero_grad()
            F.mse_loss(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        sched.step()

        if ep % 10 == 0 or ep == N_EPOCHS_SURR:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_mse  = F.mse_loss(val_pred, y_val).item()
                rho, _   = spearmanr(val_pred.numpy(), y_val.numpy())
            rho = float(rho) if not math.isnan(rho) else 0.0
            log(f"  Surrogate ep{ep:>3d}  val_mse={val_mse:.4f}  rho={rho:.4f}")
            if val_mse < best_mse:
                best_mse = val_mse
                best_sd  = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_sd)
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        final_rho, _ = spearmanr(val_pred.numpy(), y_val.numpy())
    final_rho = float(final_rho) if not math.isnan(final_rho) else 0.0
    return model, final_rho


def run_bo(model, X_all, y_all, names):
    from torch.distributions import Normal

    indices  = list(range(len(names)))
    obs_idx  = set()
    best_pkd = -float("inf")

    # Warm start: top-10 by actual pKd
    top10 = sorted(range(len(y_all)), key=lambda i: -y_all[i].item())[:10]
    for i in top10:
        obs_idx.add(i)
        if y_all[i].item() > best_pkd:
            best_pkd = y_all[i].item()

    log(f"  [BO] Warm start best={best_pkd:.3f}")

    model.eval()
    for rnd in range(1, N_BO + 1):
        remaining = [i for i in indices if i not in obs_idx]
        if not remaining:
            break

        X_rem    = X_all[remaining]
        mu, sigma = model.predict(X_rem)

        z_score  = (mu - best_pkd - 0.01) / sigma.clamp(min=1e-6)
        dist     = Normal(0, 1)
        ei       = ((mu - best_pkd - 0.01) * dist.cdf(z_score)
                    + sigma * dist.log_prob(z_score).exp())
        top_idx  = ei.topk(TOP_K_BO).indices.tolist()

        for idx in top_idx:
            orig_i = remaining[idx]
            pkd    = y_all[orig_i].item()
            obs_idx.add(orig_i)
            if pkd > best_pkd:
                best_pkd = pkd

        log(f"  [BO] Round {rnd}: best={best_pkd:.3f}  n_obs={len(obs_idx)}")
        heartbeat("Phase30_BO_MSH3", rnd, {"best_pkd": best_pkd})

    log(f"  [BO] DONE  best={best_pkd:.3f}  n_obs={len(obs_idx)}")
    return best_pkd


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 65)
    log("  Phase 30 — MSH3 3THW Docking Screen + Surrogate BO")
    log("=" * 65)

    notify("PHASE_START", "[Phase30] MSH3 3THW docking screen", data={})

    # ── Receptor prep ─────────────────────────────────────────────────────────
    pdb_path      = download_pdb()
    chain_path    = extract_chain_b(pdb_path)
    receptor_path = prep_receptor(chain_path)

    if receptor_path is None:
        log("FATAL: receptor preparation failed — cannot proceed")
        sys.exit(1)

    # ── Load FDA compounds ────────────────────────────────────────────────────
    fda_path = DATA_DIR / "pubchem_fda.json"
    if not fda_path.exists():
        log("Downloading pubchem_fda.json from GCS...")
        subprocess.run(["gsutil", "-q", "cp",
                        f"{GCS_BASE}/pubchem_fda.json",
                        str(fda_path)], check=True)

    with open(fda_path) as f:
        fda_data = json.load(f)
    compounds = [(name, entry.get("smiles", ""))
                 for name, entry in fda_data.items()
                 if entry.get("smiles")]
    log(f"Loaded {len(compounds)} FDA compounds with SMILES")

    # ── Load checkpoint if exists ─────────────────────────────────────────────
    ckpt_path   = OUT_DIR / "docking_checkpoint.json"
    scores_path = OUT_DIR / "docking_scores.json"

    if ckpt_path.exists():
        with open(ckpt_path) as f:
            checkpoint = json.load(f)
        docked_scores = checkpoint.get("scores", {})
        log(f"Resumed from checkpoint: {len(docked_scores)} compounds already docked")
    else:
        docked_scores = {}

    # ── Docking loop ──────────────────────────────────────────────────────────
    n_total   = len(compounds)
    n_screened = len(docked_scores)
    n_valid   = sum(1 for v in docked_scores.values() if v < -1.0)

    lig_dir = OUT_DIR / "ligands"
    lig_dir.mkdir(exist_ok=True)

    for idx, (name, smiles) in enumerate(compounds):
        if name in docked_scores:
            continue

        lig_path = lig_dir / f"{name}.pdbqt"
        ok       = smiles_to_pdbqt(smiles, lig_path)
        if not ok:
            docked_scores[name] = 0.0
        else:
            score = dock_compound(receptor_path, lig_path)
            docked_scores[name] = score
            if score < -1.0:
                n_valid += 1
            # cleanup ligand
            lig_path.unlink(missing_ok=True)

        n_screened += 1

        if n_screened % 100 == 0:
            log(f"Docked {n_screened}/{n_total}: {name} score={docked_scores[name]:.3f}")
            # Save checkpoint
            ckpt_path.write_text(json.dumps({"scores": docked_scores}, indent=2))
            # Upload checkpoint to GCS
            gsutil_cp(ckpt_path, f"{GCS_OUT}/docking_checkpoint.json")

    # Final save of all scores
    scores_path.write_text(json.dumps(docked_scores, indent=2))
    gsutil_cp(scores_path, f"{GCS_OUT}/docking_scores.json")
    n_valid = sum(1 for v in docked_scores.values() if v < -1.0)
    log(f"\nDocking complete: {n_screened} screened  {n_valid} valid (score < -1.0)")

    # ── Build surrogate dataset ───────────────────────────────────────────────
    log("\nBuilding surrogate dataset...")
    fps, pkds, names_ok = [], [], []
    fda_dict = {name: entry for name, entry in fda_data.items()}

    for name, score in docked_scores.items():
        if score >= -1.0:
            continue
        entry  = fda_dict.get(name, {})
        smiles = entry.get("smiles", "")
        if not smiles:
            continue
        fp = build_fp(smiles)
        if fp is None:
            continue
        pkd = vina_to_pkd(score)
        if pkd <= 0:
            continue
        fps.append(fp)
        pkds.append(pkd)
        names_ok.append(name)

    log(f"Surrogate dataset: {len(fps)} compounds")

    if len(fps) < 20:
        log("WARNING: too few valid dockings — surrogate will be unreliable")
        results = {
            "experiment":       "phase30_msh3_3thw",
            "n_screened":       n_screened,
            "n_valid_dockings": n_valid,
            "top_10_hits":      [],
            "surrogate_rho":    0.0,
            "bo_best_pkd":      0.0,
            "interpretation":   "Insufficient valid dockings for surrogate training.",
        }
        out_path = OUT_DIR / "results.json"
        out_path.write_text(json.dumps(results, indent=2))
        gsutil_cp(out_path, f"{GCS_OUT}/results.json")
        notify("PHASE_COMPLETE", "[Phase30] Done — insufficient dockings",
               data={"n_valid": n_valid})
        return

    X = torch.tensor(np.array(fps), dtype=torch.float32)
    y = torch.tensor(pkds, dtype=torch.float32)

    torch.manual_seed(SEED)
    perm    = torch.randperm(len(y))
    n_train = int(0.8 * len(y))
    tr_idx  = perm[:n_train]
    val_idx = perm[n_train:]
    X_tr, y_tr   = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    log(f"Train: {len(tr_idx)}  Val: {len(val_idx)}")

    # ── Train surrogate ────────────────────────────────────────────────────────
    log("\nTraining FP32-256 surrogate...")
    model, surrogate_rho = train_surrogate(X_tr, y_tr, X_val, y_val)
    ckpt_surr = OUT_DIR / "surrogate_fp32.pt"
    torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()}, ckpt_surr)
    gsutil_cp(ckpt_surr, f"{GCS_OUT}/surrogate_fp32.pt")
    log(f"Surrogate rho={surrogate_rho:.4f}")

    # ── Top 10 hits ───────────────────────────────────────────────────────────
    top10_raw = sorted(docked_scores.items(), key=lambda x: x[1])[:10]
    top10 = [{"name": n, "vina_score": round(s, 3), "pkd": round(vina_to_pkd(s), 3)}
             for n, s in top10_raw if s < 0]

    # ── BO ────────────────────────────────────────────────────────────────────
    log("\nRunning EI BO for MSH3...")
    bo_best = run_bo(model, X, y, names_ok)

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "experiment":       "phase30_msh3_3thw",
        "receptor":         "3THW chain B (MSH3 ATPase)",
        "box_center":       BOX_CENTER,
        "box_size":         BOX_SIZE,
        "n_screened":       n_screened,
        "n_valid_dockings": n_valid,
        "top_10_hits":      top10,
        "surrogate_rho":    round(surrogate_rho, 4),
        "bo_best_pkd":      round(bo_best, 4),
        "interpretation": (
            f"Screened {n_screened} FDA compounds against MSH3 ATPase (3THW chain B). "
            f"{n_valid} valid dockings (score < -1.0 kcal/mol). "
            f"Best Vina score: {top10[0]['vina_score'] if top10 else 'N/A'} kcal/mol. "
            f"Surrogate ρ={surrogate_rho:.3f}. BO best pKd={bo_best:.3f}."
        ),
    }

    out_path = OUT_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    gsutil_cp(out_path, f"{GCS_OUT}/results.json")

    log(f"\nResults → GCS phase30_msh3_3thw/results.json")
    log(f"  n_screened={n_screened}  n_valid={n_valid}")
    if top10:
        log(f"  Top hit: {top10[0]['name']} score={top10[0]['vina_score']}")
    log(f"  surrogate_rho={surrogate_rho:.4f}  bo_best_pkd={bo_best:.3f}")

    notify("PHASE_COMPLETE", "[Phase30] MSH3 3THW screen done",
           data={"n_valid": n_valid, "surrogate_rho": surrogate_rho, "bo_best": bo_best})


if __name__ == "__main__":
    main()
