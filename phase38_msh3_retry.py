#!/usr/bin/env python3
"""
phase38_msh3_retry.py
---------------------
Phase 38: MSH3 ATPase (3THW) docking retry with corrected box.

Phase 30 produced zero valid dockings because chain B's Walker A backbone
is occluded by a crystal contact. This retry uses:
  - Chain A (open solvent-accessible ATP-binding cavity)
  - Box center derived from the co-crystallised ADP ligand in chain A
    (computed automatically from HETATM ADP coordinates in the PDB)

GCS output: gs://.../aegis_flashoptim/phase38_msh3_retry/results.json
"""

import os, sys, json, time, subprocess, math, random, signal
import multiprocessing as _mp
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase6_vina_surrogate import vina_to_pkd
from notify import notify, heartbeat

GCS_BUCKET    = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID        = os.environ.get("RUN_ID", "aegis_flashoptim")
GCS_BASE      = f"{GCS_BUCKET}/{RUN_ID}"
GCS_OUT       = f"{GCS_BASE}/phase38_msh3_retry"
GCS_DONE      = f"{GCS_OUT}/results.json"
GCS_CKPT_SURR = f"{GCS_OUT}/surrogate.pt"
GCS_CKPT_BO   = f"{GCS_OUT}/bo_checkpoint.json"
DATA_DIR   = Path(os.environ.get("DATA_DIR", "/tmp/phase2_data"))

OUT_DIR = Path("/tmp/phase38_msh3_retry")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

RCSB_URL  = "https://files.rcsb.org/download/3THW.pdb"
CHAIN_ID  = "A"   # fixed from chain B

BOX_SIZE  = (25, 25, 25)   # tighter than phase30's 30×30×30
CHECKPOINT_EVERY = 100
N_EPOCHS_SURR    = 50
LR_SURR          = 1e-3
BATCH_SIZE       = 128
SEED             = 42
N_BO             = 20
TOP_K_BO         = 5


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] {msg}", flush=True)

def gsutil_cp(src, dst): subprocess.run(["gsutil", "-q", "cp", str(src), str(dst)], check=False)
def gcs_exists(p): return subprocess.run(["gsutil", "-q", "stat", p], capture_output=True).returncode == 0


def download_pdb():
    pdb_path = OUT_DIR / "3THW.pdb"
    if not pdb_path.exists():
        log("Downloading 3THW.pdb from RCSB...")
        import urllib.request
        urllib.request.urlretrieve(RCSB_URL, str(pdb_path))
    return pdb_path


def detect_adp_box_center(pdb_path):
    """Extract centroid of ADP ligand atoms in chain A as the docking box centre."""
    ligand_names = {"ADP", "ATP", "AMP", "ANP", "ACP", "ADN"}
    xs, ys, zs = [], [], []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            chain = line[21] if len(line) > 21 else ""
            resname = line[17:20].strip()
            if chain == CHAIN_ID and resname in ligand_names:
                try:
                    xs.append(float(line[30:38]))
                    ys.append(float(line[38:46]))
                    zs.append(float(line[46:54]))
                except (ValueError, IndexError):
                    pass
    if xs:
        cx, cy, cz = sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)
        log(f"ADP/ATP ligand centroid (chain {CHAIN_ID}): ({cx:.2f}, {cy:.2f}, {cz:.2f})  n_atoms={len(xs)}")
        return (round(cx, 3), round(cy, 3), round(cz, 3))
    # Fallback: known ATP-binding pocket coordinates for 3THW chain A from literature
    log(f"WARNING: No ADP/ATP ligand found in chain {CHAIN_ID} — using literature fallback coordinates")
    return (10.5, 28.3, -41.2)


def extract_chain(pdb_path, chain_id):
    chain_path = OUT_DIR / f"msh3_chain{chain_id}.pdb"
    if chain_path.exists():
        return chain_path
    try:
        from Bio import PDB
        parser = PDB.PDBParser(QUIET=True)
        struct = parser.get_structure("3THW", str(pdb_path))
        io     = PDB.PDBIO()
        io.set_structure(struct)

        class ChainSel(PDB.Select):
            def accept_chain(self, c): return c.id == chain_id

        io.save(str(chain_path), ChainSel())
        log(f"Extracted chain {chain_id} → {chain_path}")
    except ImportError:
        lines = []
        with open(pdb_path) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM", "TER")):
                    if len(line) > 21 and line[21] == chain_id:
                        lines.append(line)
        lines.append("END\n")
        chain_path.write_text("".join(lines))
        log(f"Extracted chain {chain_id} (grep) → {chain_path}")
    return chain_path


def prep_receptor(chain_path):
    receptor_path = OUT_DIR / "msh3_chainA_receptor.pdbqt"
    if receptor_path.exists():
        return receptor_path
    log("Preparing receptor PDBQT with obabel...")
    res = subprocess.run(
        ["obabel", str(chain_path), "-O", str(receptor_path),
         "-xr", "-p", "7.4", "--partialcharge", "gasteiger"],
        capture_output=True, text=True
    )
    if res.returncode != 0 or not receptor_path.exists():
        log(f"obabel failed: {res.stderr[:300]}")
        return None
    log(f"Receptor PDBQT → {receptor_path}")
    return receptor_path


def smiles_to_pdbqt(smiles, out_path):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return False
        mol = Chem.AddHs(mol)
        ps = AllChem.ETKDGv3(); ps.randomSeed = 42
        if AllChem.EmbedMolecule(mol, ps) < 0:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        sdf = out_path.with_suffix(".sdf")
        w = Chem.SDWriter(str(sdf)); w.write(mol); w.close()
        res = subprocess.run(["obabel", str(sdf), "-O", str(out_path), "-p", "7.4"],
                             capture_output=True, timeout=30)
        sdf.unlink(missing_ok=True)
        return res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        try:
            res = subprocess.run(
                ["obabel", "-ismi", smiles, "-O", str(out_path),
                 "--gen3d", "--minimize", "-p", "7.4"],
                capture_output=True, text=True, timeout=60
            )
            return res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0
        except Exception:
            return False


def _vina_worker(receptor_pdbqt, lig_pdbqt, box_center, box_size, q):
    try:
        from vina import Vina
        v = Vina(sf_name="vina", verbosity=0)
        v.set_receptor(str(receptor_pdbqt))
        v.set_ligand_from_file(str(lig_pdbqt))
        v.compute_vina_maps(center=list(box_center), box_size=list(box_size))
        v.dock(exhaustiveness=8, n_poses=3)
        q.put(float(v.energies(n_poses=1)[0][0]))
    except Exception:
        q.put(0.0)


def dock_compound(receptor_pdbqt, lig_pdbqt, box_center, timeout=90):
    # Fork a child process so SIGKILL can enforce the timeout even against
    # Vina's pure-C computation loops (SIGALRM cannot interrupt those).
    ctx = _mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_vina_worker,
                    args=(receptor_pdbqt, lig_pdbqt, box_center, BOX_SIZE, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.kill()
        p.join()
        return 0.0
    return q.get(timeout=1) if not q.empty() else 0.0


class FingerprintSurrogate(nn.Module):
    def __init__(self, in_dim=2048, hidden_dims=(512, 256, 128)):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x).squeeze(-1)

    def predict(self, x):
        self.train()
        with torch.no_grad():
            preds = torch.stack([self.forward(x) for _ in range(10)], 0)
        self.eval()
        return preds.mean(0), preds.std(0).clamp(min=1e-4)


def build_fp(smiles):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), dtype=np.float32)
    except Exception:
        return None


def train_surrogate(X_tr, y_tr, X_val, y_val):
    torch.manual_seed(SEED)
    model = FingerprintSurrogate()
    opt   = torch.optim.AdamW(model.parameters(), lr=LR_SURR, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS_SURR, eta_min=1e-5)
    best_mse, best_sd = float("inf"), None
    for ep in range(1, N_EPOCHS_SURR + 1):
        model.train()
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), BATCH_SIZE):
            xb, yb = X_tr[perm[i:i+BATCH_SIZE]], y_tr[perm[i:i+BATCH_SIZE]]
            opt.zero_grad()
            F.mse_loss(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        if ep % 10 == 0 or ep == N_EPOCHS_SURR:
            model.eval()
            with torch.no_grad():
                vp = model(X_val)
                mse = F.mse_loss(vp, y_val).item()
                rho, _ = spearmanr(vp.numpy(), y_val.numpy())
            rho = float(rho) if not math.isnan(rho) else 0.0
            log(f"  Surrogate ep{ep:>3d}  mse={mse:.4f}  rho={rho:.4f}")
            if mse < best_mse:
                best_mse = mse
                best_sd  = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_sd); model.eval()
    with torch.no_grad():
        final_rho, _ = spearmanr(model(X_val).numpy(), y_val.numpy())
    surr_local = OUT_DIR / "surrogate.pt"
    torch.save(best_sd, str(surr_local))
    gsutil_cp(surr_local, GCS_CKPT_SURR)
    log("Surrogate checkpoint → GCS")
    return model, float(final_rho) if not math.isnan(final_rho) else 0.0


def run_bo(model, X_all, y_all, names, start_obs=None, start_round=1, start_best=None):
    from torch.distributions import Normal
    bo_ckpt_path = OUT_DIR / "bo_checkpoint.json"
    if start_obs is not None:
        obs_idx  = set(start_obs)
        best_pkd = start_best
        log(f"  [BO] Resuming from round {start_round}, best={best_pkd:.3f}")
    else:
        obs_idx  = set(sorted(range(len(y_all)), key=lambda i: -y_all[i].item())[:10])
        best_pkd = max(y_all[i].item() for i in obs_idx)
        log(f"  [BO] Warm start best={best_pkd:.3f}")
    model.eval()
    for rnd in range(start_round, N_BO + 1):
        rem = [i for i in range(len(names)) if i not in obs_idx]
        if not rem: break
        mu, sigma = model.predict(X_all[rem])
        z  = (mu - best_pkd - 0.01) / sigma.clamp(min=1e-6)
        d  = Normal(0, 1)
        ei = (mu - best_pkd - 0.01) * d.cdf(z) + sigma * d.log_prob(z).exp()
        for idx in ei.topk(TOP_K_BO).indices.tolist():
            pkd = y_all[rem[idx]].item()
            obs_idx.add(rem[idx])
            if pkd > best_pkd: best_pkd = pkd
        log(f"  [BO] Round {rnd}: best={best_pkd:.3f}  n_obs={len(obs_idx)}")
        heartbeat("Phase38_BO_MSH3", rnd, {"best_pkd": best_pkd})
        bo_ckpt_path.write_text(json.dumps(
            {"round": rnd + 1, "obs_idx": sorted(obs_idx), "best_pkd": best_pkd}
        ))
        gsutil_cp(bo_ckpt_path, GCS_CKPT_BO)
    return best_pkd


def main():
    if gcs_exists(GCS_DONE):
        log("Results already in GCS — nothing to do."); return

    log("=" * 65)
    log("  Phase 38 — MSH3 3THW Docking Retry (chain A, ADP-derived box)")
    log("=" * 65)
    notify("PHASE_START", "[Phase38] MSH3 3THW retry chain A", data={})

    pdb_path   = download_pdb()
    box_center = detect_adp_box_center(pdb_path)
    log(f"Docking box: center={box_center}  size={BOX_SIZE}")

    chain_path    = extract_chain(pdb_path, CHAIN_ID)
    receptor_path = prep_receptor(chain_path)
    if receptor_path is None:
        log("FATAL: receptor prep failed"); sys.exit(1)

    # Load FDA compounds
    fda_path = DATA_DIR / "pubchem_fda.json"
    if not fda_path.exists():
        subprocess.run(["gsutil", "-q", "cp", f"{GCS_BASE}/pubchem_fda.json", str(fda_path)], check=True)
    with open(fda_path) as f:
        raw = json.load(f)
    # Support both list-of-[name,smiles] pairs and {name: {smiles:...}} dict
    if isinstance(raw, list):
        fda_data = {row[0]: {"smiles": row[1]} for row in raw if len(row) >= 2}
    else:
        fda_data = raw
    compounds = [(n, e.get("smiles","")) for n, e in fda_data.items() if e.get("smiles")]
    log(f"Loaded {len(compounds)} FDA compounds")

    # Resume from checkpoint
    ckpt_path = OUT_DIR / "docking_checkpoint.json"
    docked_scores = {}
    if gcs_exists(f"{GCS_OUT}/docking_checkpoint.json"):
        subprocess.run(["gsutil", "-q", "cp", f"{GCS_OUT}/docking_checkpoint.json", str(ckpt_path)], check=False)
    if ckpt_path.exists():
        docked_scores = json.loads(ckpt_path.read_text()).get("scores", {})
        log(f"Resumed: {len(docked_scores)} already docked")

    lig_dir = OUT_DIR / "ligands"; lig_dir.mkdir(exist_ok=True)
    n_total = len(compounds)

    for idx, (name, smiles) in enumerate(compounds):
        if name in docked_scores:
            continue
        lig_path = lig_dir / f"{name}.pdbqt"
        ok = smiles_to_pdbqt(smiles, lig_path)
        docked_scores[name] = dock_compound(receptor_path, lig_path, box_center) if ok else 0.0
        lig_path.unlink(missing_ok=True)
        n_done = len(docked_scores)
        if n_done % CHECKPOINT_EVERY == 0:
            log(f"Docked {n_done}/{n_total}: {name}  score={docked_scores[name]:.3f}")
            ckpt_path.write_text(json.dumps({"scores": docked_scores}, indent=2))
            gsutil_cp(ckpt_path, f"{GCS_OUT}/docking_checkpoint.json")

    n_screened = len(docked_scores)
    n_valid    = sum(1 for v in docked_scores.values() if v < -1.0)
    log(f"\nDocking done: {n_screened} screened  {n_valid} valid")

    # Build surrogate dataset
    fps, pkds, names_ok = [], [], []
    for name, score in docked_scores.items():
        if score >= -1.0: continue
        smiles = fda_data.get(name, {}).get("smiles", "")
        if not smiles: continue
        fp = build_fp(smiles)
        if fp is None: continue
        pkd = vina_to_pkd(score)
        if pkd <= 0: continue
        fps.append(fp); pkds.append(pkd); names_ok.append(name)

    log(f"Surrogate dataset: {len(fps)} compounds")

    results_base = {
        "experiment":    "phase38_msh3_retry",
        "receptor":      f"3THW chain {CHAIN_ID} (MSH3 ATPase, ADP-derived box)",
        "box_center":    box_center,
        "box_size":      BOX_SIZE,
        "n_screened":    n_screened,
        "n_valid_dockings": n_valid,
    }

    if len(fps) < 20:
        log("WARNING: too few valid dockings")
        results_base.update({"top_10_hits": [], "surrogate_rho": 0.0, "bo_best_pkd": 0.0,
                             "note": "Insufficient valid dockings for surrogate."})
        out = OUT_DIR / "results.json"
        out.write_text(json.dumps(results_base, indent=2))
        gsutil_cp(out, GCS_DONE)
        notify("PHASE_COMPLETE", "[Phase38] Done — insufficient dockings", data={"n_valid": n_valid})
        return

    X = torch.tensor(np.array(fps), dtype=torch.float32)
    y = torch.tensor(pkds, dtype=torch.float32)
    perm = torch.randperm(len(y)); n_tr = int(0.8 * len(y))
    X_tr, y_tr   = X[perm[:n_tr]], y[perm[:n_tr]]
    X_val, y_val = X[perm[n_tr:]], y[perm[n_tr:]]

    # Try loading saved surrogate from GCS before retraining
    surr_local = OUT_DIR / "surrogate.pt"
    if gcs_exists(GCS_CKPT_SURR):
        subprocess.run(["gsutil", "-q", "cp", GCS_CKPT_SURR, str(surr_local)], check=False)
    if surr_local.exists():
        log("Restored surrogate model from GCS checkpoint — skipping training")
        sd = torch.load(str(surr_local), map_location="cpu")
        model = FingerprintSurrogate()
        model.load_state_dict(sd); model.eval()
        with torch.no_grad():
            vp = model(X_val)
            rho_val, _ = spearmanr(vp.numpy(), y_val.numpy())
        rho = float(rho_val) if not math.isnan(rho_val) else 0.0
        log(f"Restored surrogate ρ={rho:.4f}")
    else:
        model, rho = train_surrogate(X_tr, y_tr, X_val, y_val)
        log(f"Surrogate ρ={rho:.4f}")

    top10 = [{"name": n, "vina": round(s,3), "pkd": round(vina_to_pkd(s),3)}
             for n, s in sorted(docked_scores.items(), key=lambda x: x[1])[:10] if s < 0]

    # Try loading BO checkpoint from GCS before re-running
    bo_ckpt_local = OUT_DIR / "bo_checkpoint.json"
    start_round, start_obs, start_best = 1, None, None
    if gcs_exists(GCS_CKPT_BO):
        subprocess.run(["gsutil", "-q", "cp", GCS_CKPT_BO, str(bo_ckpt_local)], check=False)
    if bo_ckpt_local.exists():
        bo_state   = json.loads(bo_ckpt_local.read_text())
        start_round = bo_state.get("round", 1)
        start_obs   = bo_state.get("obs_idx")
        start_best  = bo_state.get("best_pkd")
        log(f"Resumed BO from round {start_round}, best={start_best:.3f}")

    bo_best = run_bo(model, X, y, names_ok,
                     start_obs=start_obs, start_round=start_round, start_best=start_best)

    results_base.update({
        "top_10_hits":   top10,
        "surrogate_rho": round(rho, 4),
        "bo_best_pkd":   round(bo_best, 4),
    })
    out = OUT_DIR / "results.json"
    out.write_text(json.dumps(results_base, indent=2))
    gsutil_cp(out, GCS_DONE)
    log("Results → GCS phase38_msh3_retry/results.json")
    notify("PHASE_COMPLETE", "[Phase38] MSH3 retry done",
           data={"n_valid": n_valid, "rho": rho, "bo_best": bo_best})


if __name__ == "__main__":
    main()
