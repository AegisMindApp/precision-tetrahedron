#!/usr/bin/env python3
"""
vina_receptor_prep.py
---------------------
Downloads PDB structures for all 6 targets, strips water/heteroatoms,
converts to PDBQT, and auto-detects binding box from co-crystal ligand
or known catalytic residues.

Outputs: receptors/<TARGET>_receptor.pdbqt
         receptors/<TARGET>_box.json
"""

import os
import json
import subprocess
import urllib.request
from pathlib import Path

RECEPTOR_DIR = Path(os.environ.get("RECEPTOR_DIR", "/tmp/vina_receptors"))
RECEPTOR_DIR.mkdir(parents=True, exist_ok=True)

# ── Target configuration ───────────────────────────────────────────────────────
# pdb_id      : best available structure for Vina docking
# chain       : chain to keep (others are crystal contacts / symmetry mates)
# ref_ligand  : 3-letter hetam code of co-crystal ligand to auto-detect box
# ref_metal   : metal ion to use for box center if no ligand
# cat_resids  : fallback — catalytic/binding residues for box centroid
# box_size    : (x, y, z) in Angstroms
# note        : binding site description

TARGET_CONFIGS = {
    "KPC3": {
        "pdb_id":     "3RXX",
        "chain":      "A",
        "ref_ligand": None,
        "ref_metal":  None,
        "cat_resids": [70, 73, 130, 132, 166, 170],   # Ser70 active site
        "box_size":   [26, 28, 24],
        "note":       "KPC-3 beta-lactamase Ser70 active site",
        # use pre-existing receptor if available
        "existing_receptor": "analysis/amr_glass/docking/results/3RXX_receptor.pdbqt",
        "existing_box":      {"center": [-4.064, 3.249, -5.05], "size": [26, 28, 24]},
    },
    "PCSK9": {
        "pdb_id":     "2PMW",
        "chain":      "A",
        "ref_ligand": None,
        "ref_metal":  "CA",
        "cat_resids": [186, 226, 317],   # Asp186-His226-Asn317 catalytic triad
        "box_size":   [30, 30, 30],
        "note":       "PCSK9 catalytic domain (Asp186-His226-Asn317)",
    },
    "APEX1": {
        "pdb_id":     "4IEM",            # APE1 with Mg2+ in active site
        "chain":      "A",
        "ref_ligand": None,
        "ref_metal":  "MG",
        "cat_resids": [70, 96, 210, 212, 308],  # Asp70, Glu96, Asp308
        "box_size":   [25, 25, 25],
        "note":       "APEX1/APE1 nuclease active site (Mg2+-coordinated)",
    },
    "MSH3": {
        "pdb_id":     "2O8B",            # MutSbeta (MSH2-MSH3) ATPase domain
        "chain":      "B",               # MSH3 chain
        "ref_ligand": "ADP",
        "ref_metal":  None,
        "cat_resids": [697, 700, 730, 746],  # Walker A/B motifs approx
        "box_size":   [28, 28, 28],
        "note":       "MSH3 ATPase domain (ADP binding, Walker A/B)",
    },
    "CREBBP": {
        "pdb_id":     "3SVH",            # CREBBP bromodomain + I-BET inhibitor
        "chain":      "A",
        "ref_ligand": "GZ3",             # I-BET compound in 3SVH
        "ref_metal":  None,
        "cat_resids": [1110, 1120, 1125, 1167, 1173],  # bromodomain Asn/Tyr/Pro conserved
        "box_size":   [22, 22, 22],
        "note":       "CREBBP bromodomain acetyl-Lys binding pocket",
    },
    "LINGO1": {
        "pdb_id":     "4DBD",            # LINGO1 ectodomain (LRR domain, X-ray)
        "chain":      "A",
        "ref_ligand": None,
        "ref_metal":  None,
        "cat_resids": list(range(58, 90)),   # concave LRR surface (approx residues 58-90)
        "box_size":   [32, 32, 32],
        "note":       "LINGO1 LRR concave surface (ErbB2 interaction face)",
    },
}

# ── Utilities ──────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[PREP] {msg}", flush=True)


def download_pdb(pdb_id: str, out_path: Path) -> bool:
    """Download PDB file from RCSB."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, str(out_path))
        log(f"Downloaded {pdb_id} → {out_path}")
        return True
    except Exception as e:
        log(f"ERROR downloading {pdb_id}: {e}")
        return False


def parse_pdb_atoms(pdb_path: Path, chain: str):
    """Return list of (resname, resnum, atomname, x, y, z) for ATOM records."""
    atoms = []
    with open(pdb_path) as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            ch = line[21]
            if ch != chain:
                continue
            resname = line[17:20].strip()
            resnum  = int(line[22:26].strip())
            aname   = line[12:16].strip()
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            atoms.append((rec, resname, resnum, aname, x, y, z))
    return atoms


def find_box_center(atoms, cfg: dict):
    """
    Determine binding box center:
    1. Co-crystal ligand centroid (ref_ligand)
    2. Metal ion position (ref_metal)
    3. Catalytic residue centroid (cat_resids)
    """
    ref_lig = cfg.get("ref_ligand")
    ref_met = cfg.get("ref_metal")
    cat_res = set(cfg.get("cat_resids", []))

    coords = []

    # Option 1: co-crystal ligand
    if ref_lig:
        lig_coords = [(x, y, z) for (rec, rn, rnum, an, x, y, z) in atoms
                      if rn.upper() == ref_lig.upper()]
        if lig_coords:
            cx = sum(c[0] for c in lig_coords) / len(lig_coords)
            cy = sum(c[1] for c in lig_coords) / len(lig_coords)
            cz = sum(c[2] for c in lig_coords) / len(lig_coords)
            log(f"  Box from ligand {ref_lig}: center=({cx:.2f},{cy:.2f},{cz:.2f})")
            return [round(cx, 2), round(cy, 2), round(cz, 2)]

    # Option 2: metal ion
    if ref_met:
        met_coords = [(x, y, z) for (rec, rn, rnum, an, x, y, z) in atoms
                      if rn.upper() == ref_met.upper() or an.upper() == ref_met.upper()]
        if met_coords:
            cx = sum(c[0] for c in met_coords) / len(met_coords)
            cy = sum(c[1] for c in met_coords) / len(met_coords)
            cz = sum(c[2] for c in met_coords) / len(met_coords)
            log(f"  Box from metal {ref_met}: center=({cx:.2f},{cy:.2f},{cz:.2f})")
            return [round(cx, 2), round(cy, 2), round(cz, 2)]

    # Option 3: catalytic residue centroid
    if cat_res:
        res_coords = [(x, y, z) for (rec, rn, rnum, an, x, y, z) in atoms
                      if rec == "ATOM" and rnum in cat_res]
        if res_coords:
            cx = sum(c[0] for c in res_coords) / len(res_coords)
            cy = sum(c[1] for c in res_coords) / len(res_coords)
            cz = sum(c[2] for c in res_coords) / len(res_coords)
            log(f"  Box from catalytic residues {sorted(cat_res)}: center=({cx:.2f},{cy:.2f},{cz:.2f})")
            return [round(cx, 2), round(cy, 2), round(cz, 2)]

    # Fallback: protein geometric center
    all_coords = [(x, y, z) for (rec, rn, rnum, an, x, y, z) in atoms if rec == "ATOM"]
    if all_coords:
        cx = sum(c[0] for c in all_coords) / len(all_coords)
        cy = sum(c[1] for c in all_coords) / len(all_coords)
        cz = sum(c[2] for c in all_coords) / len(all_coords)
        log(f"  WARNING: Using protein centroid (no ligand/metal/residues found)")
        return [round(cx, 2), round(cy, 2), round(cz, 2)]

    return [0.0, 0.0, 0.0]


def strip_to_protein(pdb_path: Path, out_path: Path, chain: str):
    """Keep only ATOM records for the specified chain (strip HETATM, water, other chains)."""
    kept = []
    with open(pdb_path) as f:
        for line in f:
            rec = line[:6].strip()
            if rec == "ATOM" and line[21] == chain:
                resname = line[17:20].strip()
                if resname not in ("HOH", "WAT", "DOD"):   # remove crystallographic water
                    kept.append(line)
            elif line.startswith("TER") or line.startswith("END"):
                kept.append(line)
    with open(out_path, "w") as f:
        f.writelines(kept)
    log(f"  Stripped PDB → {out_path} ({len(kept)} lines)")


def pdb_to_pdbqt(pdb_path: Path, pdbqt_path: Path) -> bool:
    """Convert PDB to PDBQT using OpenBabel (adds Gasteiger charges)."""
    result = subprocess.run(
        ["obabel", str(pdb_path), "-O", str(pdbqt_path), "-xr", "--partialcharge", "gasteiger"],
        capture_output=True, text=True
    )
    if pdbqt_path.exists() and pdbqt_path.stat().st_size > 0:
        log(f"  PDBQT → {pdbqt_path}")
        return True
    log(f"  ERROR: obabel failed: {result.stderr[:200]}")
    return False


# ── Main ───────────────────────────────────────────────────────────────────────

def prepare_target(target: str, cfg: dict, repo_root: Path) -> dict | None:
    """Prepare receptor PDBQT and binding box for one target."""
    log(f"\n=== {target} ({cfg['pdb_id']}) ===")

    receptor_pdbqt = RECEPTOR_DIR / f"{target}_receptor.pdbqt"
    box_json       = RECEPTOR_DIR / f"{target}_box.json"

    # ── Use existing KPC3 receptor if available ─────────────────────────────
    existing_rec = cfg.get("existing_receptor")
    if existing_rec:
        src = repo_root / existing_rec
        if src.exists():
            import shutil
            shutil.copy(src, receptor_pdbqt)
            log(f"  Copied existing receptor: {src}")
            box = cfg["existing_box"]
            box_json.write_text(json.dumps(box, indent=2))
            log(f"  Using existing box: {box}")
            return {"target": target, "receptor": str(receptor_pdbqt),
                    "box": box, "note": cfg["note"]}

    # ── Download PDB ────────────────────────────────────────────────────────
    pdb_raw  = RECEPTOR_DIR / f"{target}_{cfg['pdb_id']}.pdb"
    pdb_prot = RECEPTOR_DIR / f"{target}_{cfg['pdb_id']}_protein.pdb"

    if not pdb_raw.exists():
        if not download_pdb(cfg["pdb_id"], pdb_raw):
            log(f"  FAILED: could not download {cfg['pdb_id']}")
            return None

    # ── Parse for box detection ─────────────────────────────────────────────
    atoms = parse_pdb_atoms(pdb_raw, cfg["chain"])
    if not atoms:
        log(f"  WARNING: no atoms found for chain {cfg['chain']} — trying chain A")
        atoms = parse_pdb_atoms(pdb_raw, "A")

    center = find_box_center(atoms, cfg)
    size   = cfg["box_size"]
    box    = {"center": center, "size": size}
    box_json.write_text(json.dumps(box, indent=2))

    # ── Strip to protein, convert to PDBQT ─────────────────────────────────
    strip_to_protein(pdb_raw, pdb_prot, cfg["chain"])

    if not pdb_to_pdbqt(pdb_prot, receptor_pdbqt):
        # Try without charge flag
        r2 = subprocess.run(
            ["obabel", str(pdb_prot), "-O", str(receptor_pdbqt)],
            capture_output=True)
        if not receptor_pdbqt.exists():
            log(f"  FAILED: could not convert {target} to PDBQT")
            return None

    log(f"  ✓ {target}: receptor={receptor_pdbqt.name}  box={box}")
    return {"target": target, "receptor": str(receptor_pdbqt),
            "box": box, "note": cfg["note"]}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="+", default=list(TARGET_CONFIGS.keys()))
    p.add_argument("--repo-root", default=str(Path.cwd()))
    args = p.parse_args()

    repo_root = Path(args.repo_root)
    results = {}
    for target in args.targets:
        if target not in TARGET_CONFIGS:
            log(f"Unknown target: {target}")
            continue
        r = prepare_target(target, TARGET_CONFIGS[target], repo_root)
        if r:
            results[target] = r

    out = RECEPTOR_DIR / "receptor_manifest.json"
    out.write_text(json.dumps(results, indent=2))
    log(f"\nManifest: {out}")
    log(f"Prepared {len(results)}/{len(args.targets)} targets: {list(results.keys())}")
    for t, r in results.items():
        log(f"  {t}: {r['box']}")


if __name__ == "__main__":
    main()
