#!/usr/bin/env python3
"""
phase21_hd_vina_screen.py
--------------------------
Extend the Phase 6 Vina screen to two Huntington's disease targets:

  PDE10A  — PDB 3HQW  (phosphodiesterase 10A, striatal HD target)
  HDAC3   — PDB 4A69  (histone deacetylase 3, HDAC inhibitor trials in HD)

Strategy:
  1. Prepare receptors for PDE10A and HDAC3 using the same pipeline as
     vina_receptor_prep.py (download PDB, strip water/HETATM, obabel → PDBQT,
     auto-detect binding box from metal/ligand)
  2. Load existing vina_scores.json from GCS (6-target Phase 6 screen)
  3. Dock all 2,639 FDA compounds against the 2 new targets
  4. Merge new scores → extended_vina_scores.json (8 targets)
  5. Upload extended_vina_scores.json to GCS

Both targets have well-defined catalytic zinc sites and many co-crystal
structures, making Vina docking straightforward.

GCS output: gs://.../aegis_flashoptim/phase21_hd/extended_vina_scores.json
"""

import os
import sys
import json
import subprocess
import tempfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

GCS_BUCKET   = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID       = os.environ.get("RUN_ID", "aegis_flashoptim")
GCS_BASE     = f"{GCS_BUCKET}/{RUN_ID}"
OUTPUT_DIR   = Path(os.environ.get("OUTPUT_DIR", "/tmp/flashoptim_results"))
DATA_DIR     = Path(os.environ.get("DATA_DIR", "/tmp/phase2_data"))
RECEPTOR_DIR = Path("/tmp/vina_receptors_hd")
N_WORKERS    = int(os.environ.get("VINA_WORKERS", "32"))
EXHAUSTIVENESS = 8

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RECEPTOR_DIR.mkdir(parents=True, exist_ok=True)

RT_KCAL = 0.592
LN10    = 2.3026

def log(msg): print(f"[Phase21-HD] {msg}", flush=True)

def gsutil_cp(src, dst):
    subprocess.run(["gsutil", "-q", "cp", str(src), str(dst)], check=False)

def vina_to_pkd(aff):
    return 0.0 if aff >= 0 else -aff / (RT_KCAL * LN10)


# ── HD Target configs ─────────────────────────────────────────────────────────

HD_TARGET_CONFIGS = {
    "PDE10A": {
        "pdb_id":     "3HQW",
        "chain":      "A",
        "ref_metal":  "ZN",   # catalytic Zn²⁺ centres box
        "cat_resids": [674, 676, 693, 726, 729, 780],
        "box_size":   [22, 22, 22],
        "note":       "PDE10A2 catalytic domain — Zn/Mg site, Gln726 switch",
    },
    "HDAC3": {
        "pdb_id":     "4A69",
        "chain":      "A",
        "ref_metal":  "ZN",   # catalytic Zn²⁺ at base of binding channel
        "cat_resids": [92, 134, 135, 168, 200, 298],
        "box_size":   [20, 20, 22],
        "note":       "HDAC3 deacetylase active site — Zn-dependent lysine channel",
    },
}


# ── Receptor preparation ──────────────────────────────────────────────────────

def download_pdb(pdb_id, out_path):
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        urllib.request.urlretrieve(url, out_path)
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception as e:
        log(f"  Download failed {pdb_id}: {e}")
        return False


def parse_pdb_atoms(pdb_path, chain):
    atoms = []
    with open(pdb_path) as f:
        for line in f:
            if line[:6] in ("ATOM  ", "HETATM"):
                ch = line[21]
                if ch == chain or chain == "*":
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        name = line[12:16].strip()
                        resid = int(line[22:26].strip())
                        atoms.append((name, resid, x, y, z))
                    except ValueError:
                        pass
    return atoms


def find_box_center(atoms, cfg):
    # Priority 1: metal ion
    metal = cfg.get("ref_metal")
    if metal:
        metal_coords = [(x, y, z) for (name, _, x, y, z) in atoms
                        if name.upper().startswith(metal.upper())]
        if metal_coords:
            xs, ys, zs = zip(*metal_coords)
            return [round(sum(xs)/len(xs), 3),
                    round(sum(ys)/len(ys), 3),
                    round(sum(zs)/len(zs), 3)]
    # Fallback: catalytic residue centroid
    cat_resids = set(cfg.get("cat_resids", []))
    ca_coords = [(x, y, z) for (name, resid, x, y, z) in atoms
                 if resid in cat_resids and name == "CA"]
    if ca_coords:
        xs, ys, zs = zip(*ca_coords)
        return [round(sum(xs)/len(xs), 3),
                round(sum(ys)/len(ys), 3),
                round(sum(zs)/len(zs), 3)]
    # Last resort: centroid of all CA atoms
    ca_all = [(x, y, z) for (name, _, x, y, z) in atoms if name == "CA"]
    if ca_all:
        xs, ys, zs = zip(*ca_all)
        return [round(sum(xs)/len(xs), 3),
                round(sum(ys)/len(ys), 3),
                round(sum(zs)/len(zs), 3)]
    return [0.0, 0.0, 0.0]


def strip_to_protein(pdb_in, pdb_out, chain):
    with open(pdb_in) as f, open(pdb_out, "w") as g:
        for line in f:
            rec = line[:6]
            if rec == "ATOM  " and line[21] == chain:
                g.write(line)
            elif rec in ("TER   ", "END   "):
                g.write(line)


def pdb_to_pdbqt(pdb_path, out_path):
    r = subprocess.run(
        ["obabel", str(pdb_path), "-O", str(out_path), "-xr"],
        capture_output=True, timeout=120)
    return out_path.exists() and out_path.stat().st_size > 0


def prepare_receptor(target, cfg):
    log(f"Preparing {target} ({cfg['pdb_id']})...")
    receptor_pdbqt = RECEPTOR_DIR / f"{target}_receptor.pdbqt"
    box_json_path  = RECEPTOR_DIR / f"{target}_box.json"

    if receptor_pdbqt.exists() and box_json_path.exists():
        log(f"  {target}: cached receptor found")
        return {"target": target, "receptor": str(receptor_pdbqt),
                "box": json.loads(box_json_path.read_text())}

    pdb_raw  = RECEPTOR_DIR / f"{target}_{cfg['pdb_id']}.pdb"
    pdb_prot = RECEPTOR_DIR / f"{target}_{cfg['pdb_id']}_protein.pdb"

    if not pdb_raw.exists():
        if not download_pdb(cfg["pdb_id"], pdb_raw):
            return None

    atoms  = parse_pdb_atoms(pdb_raw, cfg["chain"])
    center = find_box_center(atoms, cfg)
    box    = {"center": center, "size": cfg["box_size"]}
    box_json_path.write_text(json.dumps(box, indent=2))
    log(f"  {target}: box center={center}")

    strip_to_protein(pdb_raw, pdb_prot, cfg["chain"])
    if not pdb_to_pdbqt(pdb_prot, receptor_pdbqt):
        subprocess.run(["obabel", str(pdb_prot), "-O", str(receptor_pdbqt)],
                       capture_output=True)
    if not receptor_pdbqt.exists():
        log(f"  {target}: FAILED PDBQT conversion")
        return None

    log(f"  {target}: ✓ receptor ready  box={box}")
    return {"target": target, "receptor": str(receptor_pdbqt), "box": box}


# ── Ligand prep ───────────────────────────────────────────────────────────────

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
        Chem.SDWriter(str(sdf)).write(mol)
        subprocess.run(["obabel", str(sdf), "-O", str(out_path), "-p", "7.4"],
                       capture_output=True, timeout=30)
        sdf.unlink(missing_ok=True)
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


# ── Docking ───────────────────────────────────────────────────────────────────

def dock_one(args):
    name, smiles, target, receptor_path, box = args
    try:
        from vina import Vina
        with tempfile.TemporaryDirectory() as tmp:
            lig = Path(tmp) / "lig.pdbqt"
            if not smiles_to_pdbqt(smiles, lig):
                return name, target, 0.0
            v = Vina(sf_name="vina", verbosity=0)
            v.set_receptor(str(receptor_path))
            v.set_ligand_from_file(str(lig))
            v.compute_vina_maps(
                center=box["center"],
                box_size=box["size"]
            )
            v.dock(exhaustiveness=EXHAUSTIVENESS, n_poses=3)
            score = v.energies(n_poses=1)[0][0]
            return name, target, float(score) if score < 0 else 0.0
    except Exception:
        return name, target, 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("  Phase 21 — HD Target Vina Screen Extension")
    log("  Targets: PDE10A (3HQW) + HDAC3 (4A69)")
    log("  2,639 FDA compounds × 2 new HD targets")
    log("=" * 60)

    # Prepare receptors
    receptors = {}
    for target, cfg in HD_TARGET_CONFIGS.items():
        r = prepare_receptor(target, cfg)
        if r:
            receptors[target] = r
        else:
            log(f"FATAL: could not prepare {target}")
            sys.exit(1)

    # Load FDA compounds
    fda_path = DATA_DIR / "pubchem_fda.json"
    if not fda_path.exists():
        log("Downloading FDA compounds from GCS...")
        # Try legacy phase2_setup path first, fall back to root
        r = subprocess.run(["gsutil", "-q", "cp",
                            f"{GCS_BASE}/phase2_setup/pubchem_fda.json",
                            str(fda_path)], check=False)
        if r.returncode != 0:
            subprocess.run(["gsutil", "-q", "cp",
                            f"{GCS_BASE}/pubchem_fda.json",
                            str(fda_path)], check=True)
    with open(fda_path) as f:
        fda_raw = json.load(f)
    # Normalise: list of [name, smiles] → dict {name: {smiles: ...}}
    if isinstance(fda_raw, list):
        fda = {item[0]: {"smiles": item[1]} for item in fda_raw if len(item) >= 2}
    else:
        fda = fda_raw
    log(f"FDA compounds: {len(fda)}")

    # Load existing vina_scores
    vina_path = OUTPUT_DIR / "vina_scores.json"
    if not vina_path.exists():
        log("Downloading existing vina_scores.json from GCS...")
        subprocess.run(["gsutil", "-q", "cp",
                        f"{GCS_BASE}/vina_scores.json",
                        str(vina_path)], check=True)
    with open(vina_path) as f:
        scores = json.load(f)
    log(f"Existing scores: {len(scores)} compounds × {len(next(iter(scores.values())))} targets")

    # Build work list (skip already-scored)
    new_targets = list(receptors.keys())
    work = []
    for name, info in fda.items():
        smiles = info.get("smiles", "")
        if not smiles:
            continue
        for target in new_targets:
            existing = scores.get(name, {})
            if target not in existing:
                work.append((name, smiles, target,
                              receptors[target]["receptor"],
                              receptors[target]["box"]))

    log(f"Docking jobs: {len(work)} ({len(fda)} compounds × {len(new_targets)} targets)")

    # Run docking
    n_done = 0
    t0 = time.time()
    checkpoint = OUTPUT_DIR / ".phase21_checkpoint"
    checkpoint_data = {}
    if checkpoint.exists():
        checkpoint_data = json.loads(checkpoint.read_text())
        work = [w for w in work if (w[0], w[2]) not in checkpoint_data]
        log(f"Resuming from checkpoint: {len(checkpoint_data)} already done")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(dock_one, w): w for w in work}
        for fut in as_completed(futures):
            name, target, aff = fut.result()
            if name not in scores:
                scores[name] = {}
            scores[name][target] = aff
            checkpoint_data[(name, target)] = aff
            n_done += 1
            if n_done % 200 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta = (len(work) - n_done) / rate if rate > 0 else 0
                log(f"  {n_done}/{len(work)}  {rate:.1f}/s  ETA {eta/60:.0f} min")
                # Save checkpoint
                out = OUTPUT_DIR / "extended_vina_scores.json"
                out.write_text(json.dumps(scores, indent=2))

    log(f"\nDocking complete. {n_done} new scores added.")

    # Save and upload
    out = OUTPUT_DIR / "extended_vina_scores.json"
    out.write_text(json.dumps(scores, indent=2))
    gsutil_cp(out, f"{GCS_BASE}/phase21_hd/extended_vina_scores.json")
    log(f"Saved → {out}")
    log(f"GCS  → {GCS_BASE}/phase21_hd/extended_vina_scores.json")

    # Summary
    for target in new_targets:
        target_scores = [(n, scores[n][target]) for n in scores
                         if target in scores.get(n, {}) and scores[n][target] < 0]
        target_scores.sort(key=lambda x: x[1])
        log(f"\n  {target} — top 5 hits:")
        for name, aff in target_scores[:5]:
            log(f"    {name:30s}  {aff:.3f} kcal/mol  (pKd={vina_to_pkd(aff):.2f})")


if __name__ == "__main__":
    main()
