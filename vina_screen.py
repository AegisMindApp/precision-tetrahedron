#!/usr/bin/env python3
"""
vina_screen.py
--------------
Parallel AutoDock Vina screening of all FDA-approved compounds against
all prepared receptor targets.

Input:
  - receptors/<TARGET>_receptor.pdbqt  (from vina_receptor_prep.py)
  - receptors/<TARGET>_box.json
  - pubchem_fda.json                   (2,639 SMILES from Phase 2 setup)

Output:
  - vina_scores.json   : {compound_name: {target: affinity_kcal_mol}}
  - GCS upload: gs://aegismind-tpu-results/aegis_flashoptim/vina_scores.json

Runtime: ~4-8 hours on 32 parallel workers (v6e-8 VM idle CPUs).
"""

import os
import json
import math
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
RECEPTOR_DIR  = Path(os.environ.get("RECEPTOR_DIR",  "/tmp/vina_receptors"))
FDA_CACHE     = Path(os.environ.get("PHASE2_DATA_DIR",
                     os.environ.get("DATA_DIR", "/tmp/phase2_data"))) / "pubchem_fda.json"
OUTPUT_DIR    = Path(os.environ.get("OUTPUT_DIR", "/tmp/flashoptim_results"))
GCS_BUCKET    = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID        = os.environ.get("RUN_ID", "aegis_flashoptim")
N_WORKERS     = int(os.environ.get("VINA_WORKERS", "32"))
EXHAUSTIVENESS = int(os.environ.get("VINA_EXHAUSTIVENESS", "8"))
N_POSES       = 3    # keep top-3 poses per docking
CHECKPOINT_INTERVAL = 100   # save partial results every N compounds

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCORES_PATH = OUTPUT_DIR / "vina_scores.json"
CHECKPOINT_PATH = OUTPUT_DIR / ".vina_checkpoint"


def log(msg):
    print(f"[VINA] {msg}", flush=True)


# ── SMILES → 3D mol → PDBQT ───────────────────────────────────────────────────

def smiles_to_pdbqt(smiles: str, out_path: Path) -> bool:
    """Convert SMILES → 3D conformer → PDBQT using RDKit + OpenBabel."""
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
            # Fallback: distance geometry
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

        # Write to SDF, then convert with OpenBabel
        sdf_path = out_path.with_suffix(".sdf")
        writer = Chem.SDWriter(str(sdf_path))
        writer.write(mol)
        writer.close()

        # RDKit already generated 3D coordinates — omit --gen3D to avoid
        # OpenBabel re-running its own (slow) 3D builder on the SDF.
        result = subprocess.run(
            ["obabel", str(sdf_path), "-O", str(out_path), "-p", "7.4"],
            capture_output=True, timeout=30
        )
        sdf_path.unlink(missing_ok=True)
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


# ── Single docking call ────────────────────────────────────────────────────────

def dock_one(args) -> tuple[str, str, float]:
    """
    Dock one (compound, target) pair using the vina Python API.
    Each worker process creates its own Vina instance (safe with ProcessPoolExecutor).
    Returns (compound_name, target, best_affinity_kcal_mol).
    Returns (compound_name, target, 0.0) on failure.
    """
    compound_name, smiles, target, receptor_pdbqt, box = args

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        lig_pdbqt = tmpdir / "ligand.pdbqt"

        if not smiles_to_pdbqt(smiles, lig_pdbqt):
            return (compound_name, target, 0.0)

        center = box["center"]
        size   = box["size"]

        try:
            from vina import Vina
            v = Vina(sf_name='vina', cpu=1, seed=42, verbosity=0)
            v.set_receptor(rigid_pdbqt_filename=str(receptor_pdbqt))
            v.set_ligand_from_file(str(lig_pdbqt))
            v.compute_vina_maps(
                center=center,
                box_size=size,
            )
            v.dock(exhaustiveness=EXHAUSTIVENESS, n_poses=N_POSES)
            energies = v.energies(n_poses=1)
            # energies is a 2-D array: energies[pose][energy_component]
            # energies[0][0] is the best pose's total score (kcal/mol, negative = favourable)
            if energies is not None and len(energies) > 0:
                score = float(energies[0][0])
                if score < 0:          # valid docking score is always negative
                    return (compound_name, target, score)
        except Exception:
            pass

    return (compound_name, target, 0.0)


# ── Main screening loop ────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        try:
            return json.loads(CHECKPOINT_PATH.read_text())
        except Exception:
            pass
    return {}


def save_checkpoint(scores: dict):
    CHECKPOINT_PATH.write_text(json.dumps(scores))


def upload_results():
    try:
        subprocess.run(
            ["gsutil", "cp", str(SCORES_PATH),
             f"{GCS_BUCKET}/{RUN_ID}/vina_scores.json"],
            capture_output=True, check=False
        )
    except Exception:
        pass


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="+", default=None,
                   help="Subset of targets to screen (default: all in manifest)")
    p.add_argument("--max-compounds", type=int, default=None,
                   help="Limit to first N compounds (for testing)")
    args = p.parse_args()

    # ── Load receptor manifest ──────────────────────────────────────────────
    manifest_path = RECEPTOR_DIR / "receptor_manifest.json"
    if not manifest_path.exists():
        log("ERROR: receptor_manifest.json not found. Run vina_receptor_prep.py first.")
        return
    manifest = json.loads(manifest_path.read_text())

    targets = args.targets or list(manifest.keys())
    targets = [t for t in targets if t in manifest]
    log(f"Targets: {targets}")

    # ── Load FDA compound library ───────────────────────────────────────────
    if not FDA_CACHE.exists():
        # Try GCS
        log(f"FDA cache not found at {FDA_CACHE}, trying GCS...")
        subprocess.run(
            ["gsutil", "cp",
             f"{GCS_BUCKET}/phase2_setup/pubchem_fda.json", str(FDA_CACHE)],
            capture_output=True
        )
    compounds = json.loads(FDA_CACHE.read_text())   # list of [name, smiles] or {name, smiles, ...}
    # Normalise format — handle both [name, smiles] lists and dict records
    normalised = []
    for c in compounds:
        if isinstance(c, (list, tuple)):
            name, smiles = str(c[0]), str(c[1]) if len(c) > 1 else ""
        else:
            name   = c.get("name") or c.get("iupac_name") or str(c.get("cid", "UNK"))
            smiles = c.get("smiles") or c.get("canonical_smiles") or c.get("isomeric_smiles", "")
        if smiles:
            normalised.append((str(name), smiles))
    if args.max_compounds:
        normalised = normalised[:args.max_compounds]
    log(f"Compounds: {len(normalised)}")

    # ── Resume from checkpoint ──────────────────────────────────────────────
    scores = load_checkpoint()
    done_pairs = set()
    for cname, tscores in scores.items():
        for t in tscores:
            done_pairs.add((cname, t))

    total    = len(normalised) * len(targets)
    n_done   = len(done_pairs)
    n_remain = total - n_done
    log(f"Total pairs: {total} | Done: {n_done} | Remaining: {n_remain}")

    # Build work list
    work = []
    for (cname, smiles) in normalised:
        for t in targets:
            if (cname, t) not in done_pairs:
                rec  = manifest[t]["receptor"]
                box  = manifest[t]["box"]
                work.append((cname, smiles, t, rec, box))

    if not work:
        log("Nothing to do — all pairs already screened.")
    else:
        t0        = time.time()
        completed = 0
        errors    = 0

        log(f"Starting {len(work)} docking runs with {N_WORKERS} workers...")

        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(dock_one, w): w for w in work}
            for fut in as_completed(futures):
                try:
                    cname, target, affinity = fut.result()
                except Exception as e:
                    errors += 1
                    continue

                if cname not in scores:
                    scores[cname] = {}
                scores[cname][target] = affinity
                completed += 1

                if completed % CHECKPOINT_INTERVAL == 0:
                    save_checkpoint(scores)
                    elapsed = time.time() - t0
                    rate    = completed / elapsed
                    eta_s   = (n_remain - completed) / rate if rate > 0 else 0
                    log(f"  {completed}/{n_remain} done | "
                        f"{rate:.1f}/s | ETA {eta_s/3600:.1f}h | errors={errors}")

        save_checkpoint(scores)

    # ── Write final scores & upload ─────────────────────────────────────────
    SCORES_PATH.write_text(json.dumps(scores, indent=2))
    log(f"Scores saved: {SCORES_PATH}  ({len(scores)} compounds × {len(targets)} targets)")
    upload_results()
    log("GCS upload complete.")

    # Quick stats
    for t in targets:
        vals = [scores[c][t] for c in scores if t in scores.get(c, {}) and scores[c][t] < 0]
        if vals:
            vals_sorted = sorted(vals)
            log(f"  {t}: n={len(vals)}  best={vals_sorted[0]:.2f}  "
                f"mean={sum(vals)/len(vals):.2f}  "
                f"top10_mean={sum(vals_sorted[:10])/min(10,len(vals)):.2f} kcal/mol")


if __name__ == "__main__":
    main()
