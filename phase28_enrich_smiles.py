#!/usr/bin/env python3
"""
phase28_enrich_smiles.py
------------------------
Enrich raw {chembl_id: float} vina_scores JSON with SMILES from ChEMBL API.

Input:  gs://.../aegis_flashoptim/phase_amr_chembl/vina_scores_chembl.json
        (1899 entries, format {chembl_id: float})

Output: /tmp/phase28_kpc3_lmc/vina_scores_chembl.json  (enriched)
        gs://.../aegis_flashoptim/phase_amr_chembl/vina_scores_chembl_enriched.json
        Format: {chembl_id: {"smiles": str, "vina_score": float}}

ChEMBL API: batch queries of 50 compounds at a time.
"""

import os, sys, json, time, subprocess, urllib.request, urllib.error
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from notify import notify, heartbeat

GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID     = os.environ.get("RUN_ID", "aegis_flashoptim")
GCS_BASE   = f"{GCS_BUCKET}/{RUN_ID}"

OUT_DIR  = Path("/tmp/phase28_kpc3_lmc")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_RAW      = OUT_DIR / "vina_scores_chembl_raw.json"
LOCAL_ENRICHED = OUT_DIR / "vina_scores_chembl.json"
BATCH_SIZE     = 50
SLEEP_BETWEEN  = 0.5
MAX_RETRIES    = 3

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data/molecule"


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] {msg}", flush=True)


def gsutil_cp(src, dst):
    subprocess.run(["gsutil", "-q", "cp", str(src), str(dst)], check=False)


def gcs_exists(gcs_path):
    return subprocess.run(["gsutil", "-q", "stat", gcs_path],
                          capture_output=True).returncode == 0


def chembl_batch_query(chembl_ids: list) -> dict:
    """Query ChEMBL API for a batch of IDs. Returns {chembl_id: smiles or None}."""
    ids_str = ",".join(chembl_ids)
    url = f"{CHEMBL_API}?molecule_chembl_id__in={ids_str}&format=json&limit={len(chembl_ids)}"
    results = {cid: None for cid in chembl_ids}

    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            for mol in data.get("molecules", []):
                cid    = mol.get("molecule_chembl_id")
                structs = mol.get("molecule_structures") or {}
                smiles  = structs.get("canonical_smiles")
                if cid and smiles:
                    results[cid] = smiles
            return results
        except urllib.error.HTTPError as e:
            log(f"  HTTP {e.code} on batch (attempt {attempt+1}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            log(f"  Error on batch (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)

    return results


def main():
    log("=" * 65)
    log("  Phase 28 — Enrich ChEMBL SMILES")
    log("=" * 65)

    notify("PHASE_START", "[Phase28] Enriching vina_scores_chembl with SMILES", data={})

    # Download raw file
    raw_gcs = f"{GCS_BASE}/phase_amr_chembl/vina_scores_chembl.json"
    if not LOCAL_RAW.exists():
        if gcs_exists(raw_gcs):
            log(f"Downloading vina_scores_chembl.json from GCS...")
            subprocess.run(["gsutil", "-q", "cp", raw_gcs, str(LOCAL_RAW)], check=True)
        else:
            log(f"ERROR: {raw_gcs} not found in GCS")
            sys.exit(1)

    with open(LOCAL_RAW) as f:
        raw_data = json.load(f)

    log(f"Loaded {len(raw_data)} entries from vina_scores_chembl.json")

    # Check if already enriched (values are dicts not floats)
    sample_val = next(iter(raw_data.values()))
    if isinstance(sample_val, dict):
        log("File is already enriched — re-uploading")
        gsutil_cp(LOCAL_RAW, f"{GCS_BASE}/phase_amr_chembl/vina_scores_chembl_enriched.json")
        LOCAL_ENRICHED.write_text(json.dumps(raw_data, indent=2))
        notify("PHASE_COMPLETE", "[Phase28] Already enriched, re-uploaded", data={})
        return

    # Load existing enriched file if partial progress exists
    enriched = {}
    if LOCAL_ENRICHED.exists():
        with open(LOCAL_ENRICHED) as f:
            enriched = json.load(f)
        log(f"Resuming from {len(enriched)} already-enriched entries")

    all_ids = list(raw_data.keys())
    remaining = [cid for cid in all_ids if cid not in enriched]
    total     = len(all_ids)

    log(f"Total: {total}  Already done: {total - len(remaining)}  Remaining: {len(remaining)}")

    n_batches  = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
    batch_done = 0

    for i in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[i:i+BATCH_SIZE]
        smiles_map = chembl_batch_query(batch)

        for cid, smiles in smiles_map.items():
            if smiles:
                enriched[cid] = {
                    "smiles": smiles,
                    "vina_score": raw_data[cid],
                }

        batch_done += 1
        time.sleep(SLEEP_BETWEEN)

        if batch_done % 10 == 0:
            n_done = len(enriched)
            log(f"Enriched {n_done}/{total} compounds (batch {batch_done}/{n_batches})")
            # Save partial progress
            LOCAL_ENRICHED.write_text(json.dumps(enriched, indent=2))

    # Final save
    LOCAL_ENRICHED.write_text(json.dumps(enriched, indent=2))

    n_with_smiles = len(enriched)
    log(f"\n{n_with_smiles}/{total} SMILES fetched")

    # Upload to GCS
    gsutil_cp(LOCAL_ENRICHED, f"{GCS_BASE}/phase_amr_chembl/vina_scores_chembl_enriched.json")
    log(f"Uploaded → GCS phase_amr_chembl/vina_scores_chembl_enriched.json")

    notify("PHASE_COMPLETE", "[Phase28] SMILES enrichment done",
           data={"n_enriched": n_with_smiles, "total": total,
                 "pct": round(100 * n_with_smiles / max(total, 1), 1)})


if __name__ == "__main__":
    main()
