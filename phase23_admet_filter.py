#!/usr/bin/env python3
"""
phase23_admet_filter.py
------------------------
CNS-focused ADMET filtering on all BO hits from Phase 6 (6 targets) and
Phase 22 (8 targets, including PDE10A + HDAC3).

Filters applied (all RDKit-based, no external API needed):
  1. CNS MPO score ≥ 4.0  (Wager et al. J. Med. Chem. 2010)
       6 weighted physicochemical properties (MW, clogP, TPSA, HBD, pKa, logD)
       Score 0–6; ≥4 is CNS-favorable, ≥5 is excellent
  2. BBB proxy  — TPSA ≤ 90 Å²  AND  MW ≤ 450  AND  logP > -1
  3. Ro5 compliance  — MW < 500, HBD ≤ 5, HBA ≤ 10, logP ≤ 5
  4. P-gp efflux alert  — high MW (>400) + multiple aromatic rings = flag
  5. Aggregation flag  — clogP > 5 = non-specific binding risk

HD panel targets: MSH3, CREBBP, PDE10A, HDAC3
AMR panel:        KPC3

Outputs:
  - admet_results.json  — per-compound ADMET profile + pass/fail flags
  - cns_favorable.json  — compounds passing CNS MPO ≥ 4 per target
  - hd_panel_hits.json  — CNS-favorable hits scoring on ≥2 HD targets
  - amr_hits.json       — KPC3 hits with CNS annotation (selectivity context)

GCS output: gs://.../aegis_flashoptim/phase23_admet/
"""

import os, sys, json, subprocess
from pathlib import Path
from collections import defaultdict

GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID     = os.environ.get("RUN_ID", "aegis_flashoptim")
GCS_BASE   = f"{GCS_BUCKET}/{RUN_ID}"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/tmp/flashoptim_results"))
DATA_DIR   = Path(os.environ.get("DATA_DIR", "/tmp/phase2_data"))
OUT_DIR    = Path("/tmp/phase23_admet")
OUT_DIR.mkdir(exist_ok=True)

HD_PANEL  = {"MSH3", "CREBBP", "PDE10A", "HDAC3"}
AMR_PANEL = {"KPC3"}
CNS_MPO_PASS  = 4.0
BBB_TPSA_MAX  = 90.0
BBB_MW_MAX    = 450.0
BBB_LOGP_MIN  = -1.0

def log(msg): print(f"[Phase23-ADMET] {msg}", flush=True)
def gsutil_cp(src, dst): subprocess.run(["gsutil", "-q", "cp", str(src), str(dst)], check=False)


# ── CNS MPO (Wager et al. 2010) ───────────────────────────────────────────────

def _desirability(val, low_ideal, high_ideal, low_cutoff, high_cutoff):
    """Trapezoidal desirability function: 1.0 in ideal range, 0.0 outside cutoffs."""
    if val <= low_cutoff or val >= high_cutoff:
        return 0.0
    if low_ideal <= val <= high_ideal:
        return 1.0
    if val < low_ideal:
        return (val - low_cutoff) / (low_ideal - low_cutoff + 1e-9)
    return (high_cutoff - val) / (high_cutoff - high_ideal + 1e-9)


def cns_mpo_score(mol):
    """
    CNS MPO score (0–6). Higher = more CNS-favorable.
    Uses: clogP, logD (approx = clogP - 1 at pH7.4 for simple neutral),
          MW, TPSA, HBD count, pKa (approx from amine count).
    """
    try:
        from rdkit.Chem import Descriptors, rdMolDescriptors
        from rdkit.Chem.rdMolDescriptors import CalcTPSA
        mw    = Descriptors.MolWt(mol)
        clogp = Descriptors.MolLogP(mol)
        tpsa  = CalcTPSA(mol)
        hbd   = rdMolDescriptors.CalcNumHBD(mol)
        hba   = rdMolDescriptors.CalcNumHBA(mol)
        # Approximate logD at pH 7.4: clogP - 1 for neutral/weak base
        logd  = clogp - 1.0
        # Approximate pKa: assume ~10 for amines (flag high pKa as problematic)
        n_basic = sum(1 for a in mol.GetAtoms()
                      if a.GetAtomicNum() == 7 and a.GetTotalNumHs() > 0)
        pka_approx = 10.0 if n_basic > 0 else 5.0   # rough heuristic

        # Six desirability components (Wager Table 1)
        d_clogp = _desirability(clogp, 1, 3,    -1,   5)
        d_logd  = _desirability(logd,  1, 2,    -2,   4)
        d_mw    = _desirability(mw,  200, 360,   0, 500)
        d_tpsa  = _desirability(tpsa, 40, 90,    0, 120)
        d_hbd   = _desirability(hbd,   0, 0.5,  -1,   3)
        d_pka   = _desirability(pka_approx, 7, 8, 4, 10.5)

        return round(d_clogp + d_logd + d_mw + d_tpsa + d_hbd + d_pka, 3)
    except Exception:
        return None


def compute_admet(name, smiles):
    """Return dict of ADMET descriptors for a compound."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        from rdkit.Chem.rdMolDescriptors import CalcTPSA

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"name": name, "error": "invalid SMILES"}

        mw    = round(Descriptors.MolWt(mol), 2)
        clogp = round(Descriptors.MolLogP(mol), 3)
        tpsa  = round(CalcTPSA(mol), 2)
        hbd   = rdMolDescriptors.CalcNumHBD(mol)
        hba   = rdMolDescriptors.CalcNumHBA(mol)
        rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        rotb  = rdMolDescriptors.CalcNumRotatableBonds(mol)
        mpo   = cns_mpo_score(mol)

        bbb_pass = (tpsa <= BBB_TPSA_MAX and
                    mw   <= BBB_MW_MAX   and
                    clogp >= BBB_LOGP_MIN)
        ro5_pass = (mw < 500 and hbd <= 5 and hba <= 10 and clogp <= 5)
        pgp_flag = (mw > 400 and rings >= 3)   # rough P-gp efflux alert
        agg_flag = clogp > 5

        return {
            "name":      name,
            "mw":        mw,
            "clogp":     clogp,
            "tpsa":      tpsa,
            "hbd":       hbd,
            "hba":       hba,
            "arom_rings": rings,
            "rot_bonds":  rotb,
            "cns_mpo":   mpo,
            "bbb_pass":  bbb_pass,
            "ro5_pass":  ro5_pass,
            "pgp_flag":  pgp_flag,
            "agg_flag":  agg_flag,
            "cns_favorable": (mpo is not None and mpo >= CNS_MPO_PASS and bbb_pass),
        }
    except Exception as e:
        return {"name": name, "error": str(e)}


# ── Load BO results ───────────────────────────────────────────────────────────

def load_bo_hits(path, label):
    """Load Phase 6 or Phase 22 BO result JSON and extract top hits per target."""
    if not path.exists():
        log(f"  {label}: not found at {path}")
        return {}
    with open(path) as f:
        data = json.load(f)

    hits = {}   # {target: [{name, pkd, fp32/bf16}]}
    results = data.get("results", data.get("combinations", {}))
    for target, res in results.items():
        if not isinstance(res, dict): continue
        target_hits = []
        for precision in ("fp32", "bf16"):
            if precision in res:
                for entry in res[precision].get("top5", []):
                    target_hits.append({
                        "name":      entry["name"],
                        "pkd":       entry["pkd"],
                        "precision": precision,
                        "source":    label,
                    })
        if target_hits:
            hits[target] = sorted(target_hits, key=lambda x: -x["pkd"])
    return hits


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("  Phase 23 — CNS ADMET Filter on HD + AMR BO Hits")
    log(f"  CNS MPO threshold: {CNS_MPO_PASS}  |  BBB: TPSA≤{BBB_TPSA_MAX}, MW≤{BBB_MW_MAX}")
    log("=" * 60)

    # Load FDA SMILES
    fda_path = DATA_DIR / "pubchem_fda.json"
    if not fda_path.exists():
        subprocess.run(["gsutil", "-q", "cp",
                        f"{GCS_BASE}/phase2_setup/pubchem_fda.json",
                        str(fda_path)], check=True)
    with open(fda_path) as f:
        fda = json.load(f)
    smiles_map = {n: info.get("smiles", "") for n, info in fda.items()}
    log(f"FDA compounds: {len(smiles_map)}")

    # Load BO results from Phase 6 and Phase 22
    p6_path  = OUTPUT_DIR / "phase6_bo_results.json"
    p22_path = Path("/tmp/phase22_hd") / "phase22_results.json"

    # Try GCS if local not found
    for path, gcs_key in [(p6_path,  "phase6_bo_v2_results.json"),
                          (p22_path, "phase22_hd/results.json")]:
        if not path.exists():
            log(f"Downloading {path.name} from GCS...")
            subprocess.run(["gsutil", "-q", "cp",
                            f"{GCS_BASE}/{gcs_key}", str(path)], check=False)

    hits_p6  = load_bo_hits(p6_path,  "Phase6")
    hits_p22 = load_bo_hits(p22_path, "Phase22")

    # Merge: combine both phases, deduplicate
    all_hits = {}
    for src in (hits_p6, hits_p22):
        for target, entries in src.items():
            all_hits.setdefault(target, []).extend(entries)

    # Unique compounds across all targets to profile
    all_compounds = set()
    for entries in all_hits.values():
        for e in entries:
            all_compounds.add(e["name"])
    log(f"Unique compounds to profile: {len(all_compounds)}")

    # Run ADMET profiling
    admet = {}
    for name in sorted(all_compounds):
        smi = smiles_map.get(name, "")
        if not smi:
            admet[name] = {"name": name, "error": "no SMILES"}
            continue
        admet[name] = compute_admet(name, smi)

    # Save full profiles
    admet_json = OUT_DIR / "admet_results.json"
    admet_json.write_text(json.dumps(admet, indent=2))
    gsutil_cp(admet_json, f"{GCS_BASE}/phase23_admet/admet_results.json")

    # CNS-favorable hits per target
    cns_hits = {}
    for target, entries in all_hits.items():
        favorable = []
        for e in entries:
            profile = admet.get(e["name"], {})
            if profile.get("cns_favorable"):
                favorable.append({**e, "admet": profile})
        cns_hits[target] = sorted(favorable, key=lambda x: -x["pkd"])

    cns_json = OUT_DIR / "cns_favorable.json"
    cns_json.write_text(json.dumps(cns_hits, indent=2))
    gsutil_cp(cns_json, f"{GCS_BASE}/phase23_admet/cns_favorable.json")

    # HD panel cross-target hits
    hd_compound_targets = defaultdict(list)
    for target in HD_PANEL:
        for entry in cns_hits.get(target, []):
            hd_compound_targets[entry["name"]].append({
                "target": target, "pkd": entry["pkd"],
                "cns_mpo": entry["admet"].get("cns_mpo"),
            })
    hd_cross = {n: v for n, v in hd_compound_targets.items() if len(v) >= 2}

    hd_json = OUT_DIR / "hd_panel_hits.json"
    hd_json.write_text(json.dumps(hd_cross, indent=2))
    gsutil_cp(hd_json, f"{GCS_BASE}/phase23_admet/hd_panel_hits.json")

    # AMR (KPC3) with CNS annotation
    amr_hits = {}
    for target in AMR_PANEL:
        amr_hits[target] = [{**e, "admet": admet.get(e["name"], {})}
                             for e in all_hits.get(target, [])]
    amr_json = OUT_DIR / "amr_hits.json"
    amr_json.write_text(json.dumps(amr_hits, indent=2))
    gsutil_cp(amr_json, f"{GCS_BASE}/phase23_admet/amr_hits.json")

    # ── Report ────────────────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("  CNS ADMET FILTER RESULTS")
    log("=" * 60)

    log(f"\n  {'Target':10s}  {'Total hits':>10}  {'CNS-favorable':>14}  {'Pass rate':>10}")
    log(f"  {'─'*10}  {'─'*10}  {'─'*14}  {'─'*10}")
    for target in sorted(all_hits.keys()):
        total = len(all_hits[target])
        cns   = len(cns_hits.get(target, []))
        rate  = f"{cns/total*100:.0f}%" if total > 0 else "—"
        panel = " [HD]" if target in HD_PANEL else (" [AMR]" if target in AMR_PANEL else "")
        log(f"  {target:10s}  {total:>10d}  {cns:>14d}  {rate:>10}{panel}")

    log(f"\n  HD cross-target hits (CNS-favorable, ≥2 HD panel targets): {len(hd_cross)}")
    if hd_cross:
        for name, targets in sorted(hd_cross.items(),
                                     key=lambda x: -len(x[1])):
            profile = admet.get(name, {})
            tlist   = ", ".join(f"{t['target']} pKd={t['pkd']:.2f}" for t in targets)
            log(f"    {name:30s}  MPO={profile.get('cns_mpo','?')}  "
                f"MW={profile.get('mw','?')}  clogP={profile.get('clogp','?')}")
            log(f"      → {tlist}")

    log("\n  Top CNS-favorable KPC3 (AMR) hits:")
    for entry in amr_hits.get("KPC3", [])[:5]:
        p = entry.get("admet", {})
        log(f"    {entry['name']:30s}  pKd={entry['pkd']:.3f}  "
            f"MPO={p.get('cns_mpo','?')}  CNS={'✓' if p.get('cns_favorable') else '✗'}")

    log(f"\n  Results → {OUT_DIR}")
    log(f"  GCS     → {GCS_BASE}/phase23_admet/")


if __name__ == "__main__":
    main()
