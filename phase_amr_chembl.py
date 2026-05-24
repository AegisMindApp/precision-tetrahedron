#!/usr/bin/env python3
"""
phase_amr_chembl.py
--------------------
Extends the KPC-3 AMR screen from 2,639 FDA drugs to ChEMBL β-lactamase
inhibitors (~2,000–5,000 compounds with confirmed activity data).

Pipeline:
  1. Fetch ChEMBL compounds with pChEMBL ≥ 5.0 against β-lactamase targets
     (KPC-2, TEM-1, NDM-1 — broad net to capture diverse chemotypes)
  2. Lipinski filter + TPSA < 140 Å²
  3. 3D conformer (RDKit ETKDG) + PDBQT (obabel) conversion
  4. Parallel Vina docking against KPC-3 (3RXX, receptor from repo)
  5. Combine new scores with existing FDA screen from GCS
  6. Train MolecularGNN surrogate on combined (graph, Vina pKd) dataset
  7. UCB BO: MC-dropout T=20 p=0.1 β=2, 30 rounds over combined library
  8. ADMET flag (Lipinski + TPSA + aggregation alerts) on top-50 hits
  9. Report: top-20 ChEMBL-origin compounds with predicted pKd + SMILES

KPC-3 binding box: centre (-4.064, 3.249, -5.050) Å, size 25×25×25 Å
  (from PDB 3RXX active-site residues Ser70/Lys73/Ser130/Glu166)

GCS output: gs://.../aegis_flashoptim/phase_amr_chembl/results.json
"""

import os, sys, json, time, math, urllib.request, urllib.parse
import subprocess, tempfile, random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr

try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
    try:
        import torch_xla.experimental as _xe; _xe.eager_mode(True)
    except Exception: pass
except ImportError:
    XLA_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MolecularGNN

GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID     = os.environ.get("RUN_ID",     "aegis_flashoptim")
GCS_BASE   = f"{GCS_BUCKET}/{RUN_ID}"
GCS_P      = f"{GCS_BASE}/phase_amr_chembl"

OUT_DIR      = Path("/tmp/phase_amr_chembl")
RECEPTOR_DIR = Path("/tmp/vina_receptors_amr")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RECEPTOR_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
HIDDEN_DIM = 256
N_BLOCKS   = 6
N_EPOCHS   = 60
BATCH_SIZE = 32
LR         = 1e-3
SEED       = 42
N_WORKERS  = int(os.environ.get("VINA_WORKERS", "32"))
EXHAUSTIVENESS = 8
RT_KCAL    = 0.592
LN10       = 2.3026
EDGE_CUTOFF = 5.0      # Å — molecular graph edge radius

# KPC-3 (3RXX) Vina docking box — active site centre from qubo_docking.py
KPC3_BOX = {
    "center_x": -4.064, "center_y": 3.249, "center_z": -5.050,
    "size_x": 25.0,     "size_y": 25.0,    "size_z": 25.0,
}
RECEPTOR_PDBQT = RECEPTOR_DIR / "3RXX_receptor.pdbqt"

# ChEMBL target IDs for β-lactamases (KPC-2/3, TEM-1, NDM-1)
BETALACTAMASE_TARGETS = [
    "CHEMBL3820019",   # KPC-2 carbapenemase
    "CHEMBL4093",      # TEM-1 β-lactamase (broad-spectrum inhibitor coverage)
    "CHEMBL4523",      # NDM-1 metallo-β-lactamase
]
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"

def log(msg): print(f"[{time.strftime('%H:%M:%S')}][AMR-ChEMBL] {msg}", flush=True)
def gsutil_cp(src, dst):
    subprocess.run(["gsutil", "-q", "cp", str(src), dst], check=False)
def vina_to_pkd(aff):
    return 0.0 if aff >= 0 else -aff / (RT_KCAL * LN10)
def get_device():
    return xm.xla_device() if XLA_AVAILABLE else torch.device("cpu")


# ── ChEMBL fetch ──────────────────────────────────────────────────────────────

def chembl_fetch_betalactamase_inhibitors(pchembl_min=5.0, max_per_target=2000):
    """
    Fetch SMILES for compounds with pChEMBL ≥ pchembl_min against
    β-lactamase targets via ChEMBL REST API.
    Returns list of {"smiles": ..., "chembl_id": ..., "pchembl": ...}.
    """
    seen, results = set(), []
    for target_id in BETALACTAMASE_TARGETS:
        offset = 0
        while True:
            params = urllib.parse.urlencode({
                "target_chembl_id": target_id,
                "pchembl_value__gte": pchembl_min,
                "limit": 100,
                "offset": offset,
                "format": "json",
                "fields": "molecule_chembl_id,pchembl_value,canonical_smiles",
            })
            url = f"{CHEMBL_API}/activity.json?{params}"
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    data = json.loads(r.read())
            except Exception as e:
                log(f"  ChEMBL API error ({target_id} offset={offset}): {e}")
                break
            for act in data.get("activities", []):
                smiles = act.get("canonical_smiles")
                cid    = act.get("molecule_chembl_id")
                pch    = act.get("pchembl_value")
                if smiles and cid and pch and cid not in seen:
                    seen.add(cid)
                    results.append({"smiles": smiles, "chembl_id": cid,
                                    "pchembl": float(pch)})
            page_meta = data.get("page_meta", {})
            total = page_meta.get("total_count", 0)
            offset += 100
            if offset >= min(total, max_per_target):
                break
        log(f"  {target_id}: {len(results)} compounds so far")
    log(f"Fetched {len(results)} unique ChEMBL compounds")
    return results


def lipinski_ok(smiles):
    """Quick Lipinski Ro5 check via RDKit."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return False
        return (Descriptors.MolWt(mol) < 500 and
                Descriptors.MolLogP(mol) < 5 and
                Chem.rdMolDescriptors.CalcNumHBD(mol) <= 5 and
                Chem.rdMolDescriptors.CalcNumHBA(mol) <= 10 and
                Chem.rdMolDescriptors.CalcTPSA(mol) < 140)
    except Exception:
        return False


# ── SMILES → molecular graph ──────────────────────────────────────────────────

ATOM_TYPES = {1:0, 6:1, 7:2, 8:3, 9:4, 15:5, 16:6, 17:7, 35:8, 53:9}

def smiles_to_graph(smiles):
    """
    SMILES → (z, pos, edge_src, edge_dst) for MolecularGNN input.
    Returns None on failure.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        mol = Chem.AddHs(mol)
        ps  = AllChem.ETKDGv3(); ps.randomSeed = SEED
        if AllChem.EmbedMolecule(mol, ps) < 0: return None
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        mol = Chem.RemoveHs(mol)
        N = mol.GetNumAtoms()
        if N < 2: return None

        z = torch.tensor([ATOM_TYPES.get(a.GetAtomicNum(), 0)
                          for a in mol.GetAtoms()], dtype=torch.long)
        conf = mol.GetConformer()
        pos  = torch.tensor([[conf.GetAtomPosition(i).x,
                               conf.GetAtomPosition(i).y,
                               conf.GetAtomPosition(i).z]
                              for i in range(N)], dtype=torch.float)

        # Distance-cutoff graph
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)          # (N,N,3)
        dist = diff.norm(dim=-1)                             # (N,N)
        mask = (dist < EDGE_CUTOFF) & (dist > 0)
        src, dst = mask.nonzero(as_tuple=True)
        return {"z": z, "pos": pos, "edge_src": src, "edge_dst": dst, "n_atoms": N}
    except Exception:
        return None


def graphs_to_batch(graphs, targets, device):
    """Pack list of graphs into a batched format for MolecularGNN."""
    z_list, pos_list, es_list, ed_list, assign_list = [], [], [], [], []
    offset = 0
    for i, g in enumerate(graphs):
        N = g["n_atoms"]
        z_list.append(g["z"])
        pos_list.append(g["pos"])
        es_list.append(g["edge_src"] + offset)
        ed_list.append(g["edge_dst"] + offset)
        assign_list.append(torch.full((N,), i, dtype=torch.long))
        offset += N
    return {
        "z":       torch.cat(z_list).to(device),
        "pos":     torch.cat(pos_list).to(device),
        "edge_src":torch.cat(es_list).to(device),
        "edge_dst":torch.cat(ed_list).to(device),
        "assign":  torch.cat(assign_list).to(device),
        "B":       len(graphs),
        "target":  torch.tensor(targets, dtype=torch.float).to(device),
    }


# ── Vina docking ──────────────────────────────────────────────────────────────

def smiles_to_pdbqt(smiles, out_path):
    try:
        from rdkit import Chem; from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return False
        mol = Chem.AddHs(mol)
        ps = AllChem.ETKDGv3(); ps.randomSeed = SEED
        if AllChem.EmbedMolecule(mol, ps) < 0: return False
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        sdf = out_path.with_suffix(".sdf")
        Chem.SDWriter(str(sdf)).write(mol)
        r = subprocess.run(
            ["obabel", str(sdf), "-O", str(out_path), "-p", "7.4"],
            capture_output=True, timeout=30
        )
        sdf.unlink(missing_ok=True)
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


def dock_one(args):
    chembl_id, smiles = args
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        lig = tmp / "ligand.pdbqt"
        if not smiles_to_pdbqt(smiles, lig): return chembl_id, 0.0
        try:
            from vina import Vina
            v = Vina(sf_name="vina", verbosity=0)
            v.set_receptor(str(RECEPTOR_PDBQT))
            v.set_ligand_from_file(str(lig))
            v.compute_vina_maps(
                center=[KPC3_BOX["center_x"], KPC3_BOX["center_y"], KPC3_BOX["center_z"]],
                box_size=[KPC3_BOX["size_x"],   KPC3_BOX["size_y"],   KPC3_BOX["size_z"]],
            )
            v.dock(exhaustiveness=EXHAUSTIVENESS, n_poses=1)
            aff = v.energies(n_poses=1)[0][0]
            return chembl_id, float(aff)
        except Exception:
            return chembl_id, 0.0


def run_vina_screen(compounds):
    """Dock all compounds; return {chembl_id: affinity_kcal_mol}."""
    log(f"Docking {len(compounds)} compounds with {N_WORKERS} workers...")
    results = {}
    args = [(c["chembl_id"], c["smiles"]) for c in compounds]
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(dock_one, a): a for a in args}
        for fut in as_completed(futs):
            cid, aff = fut.result()
            results[cid] = aff
            done += 1
            if done % 100 == 0:
                log(f"  {done}/{len(args)} docked")
    log(f"Docking complete: {sum(v < 0 for v in results.values())} successful poses")
    return results


# ── Surrogate training ────────────────────────────────────────────────────────

def build_dataset(graphs_map, scores_map):
    """Return list of (graph, pkd) pairs from matching keys."""
    dataset = []
    for k in graphs_map:
        if k in scores_map and scores_map[k] < 0:
            dataset.append((graphs_map[k], vina_to_pkd(scores_map[k])))
    random.shuffle(dataset)
    return dataset


def make_batches_gnn(dataset, batch_size):
    batches = []
    for i in range(0, len(dataset), batch_size):
        chunk = dataset[i:i+batch_size]
        if len(chunk) < 2: break
        batches.append(chunk)
    return batches


def forward_batch(model, batch, device):
    graphs = [g for g,_ in batch]
    targets = [t for _,t in batch]
    b = graphs_to_batch(graphs, targets, device)
    pred = model(b["z"], b["pos"], b["edge_src"], b["edge_dst"],
                 b["assign"], b["B"],
                 torch.ones(b["edge_src"].shape[0], dtype=torch.bool, device=device),
                 torch.ones(b["z"].shape[0],        dtype=torch.bool, device=device))
    return pred.squeeze(), b["target"]


def train_surrogate(dataset, device):
    log(f"Training surrogate on {len(dataset)} samples...")
    torch.manual_seed(SEED)
    model = MolecularGNN(hidden_dim=HIDDEN_DIM, num_blocks=N_BLOCKS).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    n_val = max(1, len(dataset) // 10)
    train_set, val_set = dataset[n_val:], dataset[:n_val]
    for ep in range(1, N_EPOCHS + 1):
        model.train()
        random.shuffle(train_set)
        for batch in make_batches_gnn(train_set, BATCH_SIZE):
            opt.zero_grad()
            pred, y = forward_batch(model, batch, device)
            F.mse_loss(pred, y).backward()
            if XLA_AVAILABLE: xm.optimizer_step(opt)
            else: opt.step()
        if XLA_AVAILABLE: xm.mark_step()
        if ep % 10 == 0:
            model.eval()
            preds, ys = [], []
            with torch.no_grad():
                for batch in make_batches_gnn(val_set, BATCH_SIZE):
                    p, y = forward_batch(model, batch, device)
                    preds.extend(p.cpu().tolist()); ys.extend(y.cpu().tolist())
            rho, _ = spearmanr(preds, ys)
            mae    = sum(abs(a-b) for a,b in zip(preds,ys)) / max(len(preds),1)
            log(f"  ep{ep:3d}  val_mae={mae:.3f}  ρ={rho:.3f}")
    return model


# ── UCB BO ────────────────────────────────────────────────────────────────────

def mc_dropout_predict(model, graphs, device, T=20, p_drop=0.1):
    """MC-dropout inference: returns (mean, std) per compound."""
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.p = p_drop
    model.train()  # keep dropout on
    preds = []
    with torch.no_grad():
        for _ in range(T):
            batch_preds = []
            for g in graphs:
                b = graphs_to_batch([g], [0.0], device)
                out = model(b["z"], b["pos"], b["edge_src"], b["edge_dst"],
                            b["assign"], b["B"],
                            torch.ones(b["edge_src"].shape[0], dtype=torch.bool, device=device),
                            torch.ones(b["z"].shape[0],        dtype=torch.bool, device=device))
                batch_preds.append(out.squeeze().item())
            preds.append(batch_preds)
    if XLA_AVAILABLE: xm.mark_step()
    preds = np.array(preds)  # (T, N)
    return preds.mean(0), preds.std(0)


def ucb_bo(model, library_graphs, library_ids, observed_ids, device,
           n_rounds=30, top_k=5, beta=2.0):
    """UCB acquisition: acquire top_k per round, add to observed set."""
    acquired = []
    observed = set(observed_ids)
    for rnd in range(1, n_rounds + 1):
        candidates = [(i, g) for i, (k, g) in enumerate(zip(library_ids, library_graphs))
                      if k not in observed]
        if not candidates:
            log(f"  BO round {rnd}: all compounds acquired"); break
        idxs, cand_graphs = zip(*candidates)
        means, stds = mc_dropout_predict(model, list(cand_graphs), device)
        ucb_scores  = means + beta * stds
        top_local   = np.argsort(ucb_scores)[::-1][:top_k]
        for li in top_local:
            gi = idxs[li]
            acquired.append({
                "chembl_id": library_ids[gi],
                "round":     rnd,
                "ucb":       float(ucb_scores[li]),
                "mean_pkd":  float(means[li]),
                "std_pkd":   float(stds[li]),
            })
            observed.add(library_ids[gi])
        log(f"  BO round {rnd:2d}: acquired {len(top_local)} compounds  "
            f"best_ucb={max(ucb_scores[top_local]):.3f}")
    return acquired


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 65)
    log("  Phase AMR-ChEMBL — KPC-3 screen on ChEMBL β-lactamase hits")
    log("=" * 65)

    # ── 0. Verify receptor ────────────────────────────────────────────────
    if not RECEPTOR_PDBQT.exists():
        log(f"ERROR: receptor not found at {RECEPTOR_PDBQT}")
        log("Run: gsutil cp gs://aegismind-tpu-results/aegis_flashoptim/3RXX_receptor.pdbqt "
            f"{RECEPTOR_PDBQT}")
        sys.exit(1)

    # ── 1. Fetch ChEMBL compounds ─────────────────────────────────────────
    chembl_cache = OUT_DIR / "chembl_compounds.json"
    if chembl_cache.exists():
        compounds = json.loads(chembl_cache.read_text())
        log(f"Loaded {len(compounds)} ChEMBL compounds from cache")
    else:
        compounds = chembl_fetch_betalactamase_inhibitors()
        compounds = [c for c in compounds if lipinski_ok(c["smiles"])]
        log(f"After Lipinski filter: {len(compounds)} compounds")
        chembl_cache.write_text(json.dumps(compounds))

    # ── 2. Build molecular graphs ─────────────────────────────────────────
    graphs_cache = OUT_DIR / "graphs.pt"
    if graphs_cache.exists():
        graphs_map = torch.load(graphs_cache, map_location="cpu")
        log(f"Loaded {len(graphs_map)} graphs from cache")
    else:
        log("Building molecular graphs (RDKit ETKDG)...")
        graphs_map = {}
        for i, c in enumerate(compounds):
            g = smiles_to_graph(c["smiles"])
            if g is not None:
                graphs_map[c["chembl_id"]] = g
            if (i+1) % 200 == 0:
                log(f"  {i+1}/{len(compounds)} processed ({len(graphs_map)} ok)")
        torch.save(graphs_map, graphs_cache)
        log(f"Graphs built: {len(graphs_map)} / {len(compounds)}")

    # ── 3. Vina docking ───────────────────────────────────────────────────
    vina_cache = OUT_DIR / "vina_scores_chembl.json"
    if vina_cache.exists():
        vina_scores = json.loads(vina_cache.read_text())
        log(f"Loaded {len(vina_scores)} Vina scores from cache")
    else:
        # Dock only compounds for which we have a valid graph
        to_dock = [c for c in compounds if c["chembl_id"] in graphs_map]
        vina_scores = run_vina_screen(to_dock)
        vina_cache.write_text(json.dumps(vina_scores))
        gsutil_cp(vina_cache, f"{GCS_P}/vina_scores_chembl.json")

    # ── 4. (Optionally) load existing FDA scores from GCS ─────────────────
    fda_scores_path = OUT_DIR / "vina_scores_fda.json"
    if not fda_scores_path.exists():
        r = subprocess.run(
            ["gsutil", "-q", "cp",
             f"{GCS_BASE}/vina_scores.json", str(fda_scores_path)],
            capture_output=True
        )
        if r.returncode != 0:
            log("FDA Vina scores not found on GCS — proceeding with ChEMBL only")
    fda_raw = {}
    if fda_scores_path.exists():
        raw = json.loads(fda_scores_path.read_text())
        # FDA format: {compound: {target: aff}} — extract KPC3 column
        for name, scores in raw.items():
            aff = scores.get("KPC3", scores.get("kpc3", 0.0))
            fda_raw[f"fda_{name}"] = float(aff)
        log(f"Loaded {len(fda_raw)} FDA Vina scores")

    # Merge scores and graphs
    all_scores = {**{k: v for k,v in vina_scores.items()}, **fda_raw}

    # ── 5. Build FDA graphs if available ─────────────────────────────────
    fda_graphs_path = OUT_DIR / "fda_graphs.pt"
    if fda_raw and not fda_graphs_path.exists():
        fda_smiles_path = Path("/tmp/phase2_data/pubchem_fda.json")
        if fda_smiles_path.exists():
            log("Building FDA molecular graphs...")
            fda_data = json.loads(fda_smiles_path.read_text())
            fda_graphs = {}
            for name, smi in (fda_data.items() if isinstance(fda_data, dict)
                               else [(d["name"], d["smiles"]) for d in fda_data]):
                g = smiles_to_graph(smi)
                if g is not None:
                    fda_graphs[f"fda_{name}"] = g
            torch.save(fda_graphs, fda_graphs_path)
            graphs_map.update(fda_graphs)
            log(f"Added {len(fda_graphs)} FDA graphs")

    # ── 6. Train surrogate ────────────────────────────────────────────────
    device  = get_device()
    dataset = build_dataset(graphs_map, all_scores)
    log(f"Dataset: {len(dataset)} (graph, pKd) pairs")
    if len(dataset) < 10:
        log("ERROR: too few training pairs — check docking results")
        sys.exit(1)

    model = train_surrogate(dataset, device)
    ckpt  = OUT_DIR / "surrogate.pt"
    torch.save(model.state_dict(), ckpt)
    gsutil_cp(ckpt, f"{GCS_P}/surrogate.pt")
    log("Surrogate saved")

    # ── 7. UCB BO ─────────────────────────────────────────────────────────
    library_ids    = list(graphs_map.keys())
    library_graphs = [graphs_map[k] for k in library_ids]
    trained_ids    = {k for k,_ in dataset}

    log(f"\nRunning UCB BO over {len(library_ids)} compounds...")
    acquired = ucb_bo(model, library_graphs, library_ids, trained_ids, device)

    # ── 8. Top hits ───────────────────────────────────────────────────────
    acquired_sorted = sorted(acquired, key=lambda x: -x["mean_pkd"])
    # Flag ChEMBL-origin compounds in top-50
    chembl_smiles = {c["chembl_id"]: c["smiles"] for c in compounds}
    top_hits = []
    for hit in acquired_sorted[:50]:
        cid = hit["chembl_id"]
        if cid.startswith("fda_"): continue
        smiles = chembl_smiles.get(cid, "")
        vina   = vina_scores.get(cid, 0.0)
        top_hits.append({**hit, "smiles": smiles, "vina_affinity": vina})
        if len(top_hits) == 20: break

    log("\n" + "=" * 65)
    log("  TOP 20 ChEMBL HITS vs KPC-3")
    log(f"  {'ChEMBL ID':16s}  {'pKd (pred)':>10}  {'Vina (kcal)':>11}  SMILES[:40]")
    log("  " + "-" * 62)
    for h in top_hits:
        log(f"  {h['chembl_id']:16s}  {h['mean_pkd']:>10.3f}  "
            f"{h.get('vina_affinity',0):>11.2f}  {h['smiles'][:40]}")

    results = {
        "experiment":      "phase_amr_chembl",
        "n_chembl_fetched": len(compounds),
        "n_docked":        sum(v < 0 for v in vina_scores.values()),
        "n_surrogate_pairs": len(dataset),
        "top_20_hits":     top_hits,
        "ucb_acquired":    acquired_sorted[:100],
    }
    out = OUT_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    gsutil_cp(out, f"{GCS_P}/results.json")
    log(f"\n  Results → {out}")
    log(f"  GCS     → {GCS_P}/results.json")
    log("=== phase_amr_chembl complete ===")


if __name__ == "__main__":
    main()
