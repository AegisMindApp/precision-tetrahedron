#!/usr/bin/env python3
"""
phase13_screen_generated.py
---------------------------
Screen phase12 diffusion-generated molecules (ZINC-250K domain) against
all 6 drug targets using the phase6 target-conditioned surrogate.

Pipeline
--------
1.  Download phase12 samples_final.json from GCS
2.  Download phase6 fp32_best.pt surrogate checkpoint
3.  Load TargetConditionedSurrogate, run batch inference (all mols × 6 targets)
4.  Apply Lipinski filter via RDKit
5.  Rank by predicted KPC3 pKd (primary AMR target) and PCSK9 (secondary)
6.  Report top 50 per target; save full results to GCS

GCS output: gs://aegismind-tpu-results/aegis_flashoptim/phase13_screen_generated/
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── XLA ───────────────────────────────────────────────────────────────────────
try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
    try:
        import torch_xla.experimental as _e
        _e.eager_mode(True)
        print("XLA eager mode: ENABLED", flush=True)
    except Exception as _xe:
        print(f"XLA eager mode unavailable: {_xe}", flush=True)
except ImportError:
    XLA_AVAILABLE = False
    print("torch_xla not found — running on CPU/GPU", flush=True)

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.expanduser("~/flashoptim"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chembl_data import smiles_to_graph
from notify import notify, heartbeat
from phase6_vina_surrogate import TargetConditionedSurrogate

# ── Boilerplate ───────────────────────────────────────────────────────────────
GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID     = os.environ.get("RUN_ID",     "aegis_flashoptim")
GCS_BASE   = f"{GCS_BUCKET}/{RUN_ID}"

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] {msg}", flush=True)

def gsutil_cp(local, gcs):
    subprocess.run(["gsutil", "-q", "cp", str(local), gcs], check=False)

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR   = Path("/tmp/phase13_screen_generated")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PA          = 80          # padding length (must match phase6)
BATCH_SIZE  = 64
TOP_N       = 50          # top hits per target
HIDDEN_DIM  = 256         # fp32 surrogate uses hidden_dim=256

TARGETS   = ["LINGO1", "PCSK9", "KPC3", "APEX1", "MSH3", "CREBBP"]
TARGET2ID = {t: i for i, t in enumerate(TARGETS)}

# ── Lipinski filter ───────────────────────────────────────────────────────────

def lipinski_filter(smiles):
    """Returns (pass: bool, props: dict). props empty if RDKit unavailable."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, {}
        mw   = Descriptors.MolWt(mol)
        hbd  = rdMolDescriptors.CalcNumHBD(mol)
        hba  = rdMolDescriptors.CalcNumHBA(mol)
        logp = Descriptors.MolLogP(mol)
        props = {"mw": round(mw, 1), "hbd": hbd, "hba": hba, "logp": round(logp, 2)}
        return (mw <= 500 and hbd <= 5 and hba <= 10 and logp <= 5), props
    except Exception as e:
        log(f"  lipinski_filter error: {e}")
        return False, {}


# ── Graph building ─────────────────────────────────────────────────────────────

def smiles_to_padded(smiles, pa=PA):
    """Convert SMILES → (z_pad, pos_pad, valid) tensors, or None on failure."""
    try:
        result = smiles_to_graph(smiles)
        if result is None:
            return None
        z, pos = result
        n = min(len(z), pa)
        z_pad   = torch.zeros(pa, dtype=torch.long)
        pos_pad = torch.zeros(pa, 3, dtype=torch.float32)
        valid   = torch.zeros(pa, dtype=torch.bool)
        z_pad[:n]   = torch.as_tensor(z[:n],   dtype=torch.long)
        pos_pad[:n] = torch.as_tensor(pos[:n], dtype=torch.float32)
        valid[:n]   = True
        return z_pad, pos_pad, valid
    except Exception:
        return None


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, graphs, device):
    """
    Run surrogate inference for all molecules × all 6 targets.

    graphs: list of (z_pad, pos_pad, valid, smiles_idx)
    Returns: np.array of shape [N_mols, N_targets] with predicted pKd values.
    """
    model.eval()
    n_mols    = len(graphs)
    n_targets = len(TARGETS)
    all_pkd   = np.zeros((n_mols, n_targets), dtype=np.float32)

    for tid in range(n_targets):
        target_name = TARGETS[tid]
        log(f"  Scoring vs {target_name} (target_id={tid}) ...")
        preds = []
        with torch.no_grad():
            for start in range(0, n_mols, BATCH_SIZE):
                chunk = graphs[start: start + BATCH_SIZE]
                z_b     = torch.stack([g[0] for g in chunk]).to(device)
                pos_b   = torch.stack([g[1] for g in chunk]).to(device)
                valid_b = torch.stack([g[2] for g in chunk]).to(device)
                tid_b   = torch.full((len(chunk),), tid, dtype=torch.long, device=device)
                out = model(z_b, pos_b, valid_b, tid_b)
                if XLA_AVAILABLE:
                    xm.mark_step()
                preds.extend(out.cpu().float().tolist())
        all_pkd[:, tid] = np.array(preds, dtype=np.float32)

    return all_pkd


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    notify("PHASE_START", "Phase 13 — screening phase12 generated molecules vs 6 targets")
    log("=" * 60)
    log("Phase 13 — Virtual screening of diffusion-generated molecules")
    log("=" * 60)

    # ── Device ────────────────────────────────────────────────────────────────
    if XLA_AVAILABLE:
        device = xm.xla_device()
        log(f"Device: XLA ({device})")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        log(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        log("Device: CPU")

    # ── Download inputs ───────────────────────────────────────────────────────
    samples_local  = OUT_DIR / "samples_final.json"
    surrogate_local = OUT_DIR / "fp32_best.pt"

    log("Downloading phase12 samples ...")
    r = subprocess.run(
        ["gsutil", "-q", "cp",
         f"{GCS_BASE}/phase12_diffusion_zinc/samples_final.json",
         str(samples_local)],
        capture_output=True,
    )
    if r.returncode != 0:
        notify("ABORT", "Phase 13: failed to download samples_final.json", urgent=True)
        raise RuntimeError("samples_final.json not found on GCS — did phase12 complete?")

    log("Downloading phase6 fp32_best.pt ...")
    r = subprocess.run(
        ["gsutil", "-q", "cp",
         f"{GCS_BASE}/phase6/fp32_best.pt",
         str(surrogate_local)],
        capture_output=True,
    )
    if r.returncode != 0:
        notify("ABORT", "Phase 13: failed to download phase6 fp32_best.pt", urgent=True)
        raise RuntimeError("phase6/fp32_best.pt not found on GCS — did phase6 complete?")

    # ── Load generated SMILES ─────────────────────────────────────────────────
    raw_samples = json.loads(samples_local.read_text())
    if isinstance(raw_samples, list) and len(raw_samples) > 0:
        if isinstance(raw_samples[0], dict):
            smiles_list = [s["smiles"] for s in raw_samples if "smiles" in s]
        else:
            smiles_list = [str(s) for s in raw_samples]
    else:
        notify("ABORT", "Phase 13: samples_final.json is empty or malformed")
        raise RuntimeError("samples_final.json is empty or unexpected format")

    log(f"Loaded {len(smiles_list)} generated SMILES from phase12")

    # ── Lipinski filter ───────────────────────────────────────────────────────
    log("Applying Lipinski filter ...")
    lipinski_results = []
    for smi in smiles_list:
        passed, props = lipinski_filter(smi)
        lipinski_results.append((passed, props))

    n_lip_pass = sum(1 for p, _ in lipinski_results if p)
    log(f"Lipinski: {n_lip_pass}/{len(smiles_list)} passed (MW≤500, HBD≤5, HBA≤10, logP≤5)")

    # ── Build graph tensors ───────────────────────────────────────────────────
    log("Building graph tensors ...")
    graphs     = []
    graph2smi  = []   # maps graph index → smiles index
    failed     = 0
    for i, smi in enumerate(smiles_list):
        if i % 200 == 0:
            sys.stdout.write(f"\r  {i}/{len(smiles_list)}")
            sys.stdout.flush()
        g = smiles_to_padded(smi)
        if g is None:
            failed += 1
            continue
        graphs.append(g)
        graph2smi.append(i)
    print()
    log(f"Graph build: {len(graphs)} ok, {failed} failed")

    if len(graphs) == 0:
        notify("ABORT", "Phase 13: all SMILES failed graph building — RDKit issue?")
        raise RuntimeError("No valid graphs built")

    # ── Load surrogate model ──────────────────────────────────────────────────
    log("Loading phase6 TargetConditionedSurrogate ...")
    model = TargetConditionedSurrogate(hidden_dim=HIDDEN_DIM, n_targets=len(TARGETS))

    ckpt_raw = torch.load(str(surrogate_local), map_location="cpu")
    if "model_state" in ckpt_raw:
        state = ckpt_raw["model_state"]
    elif "state_dict" in ckpt_raw:
        state = ckpt_raw["state_dict"]
    else:
        state = ckpt_raw
    missing, unexpected = model.load_state_dict(state, strict=False)
    log(f"Checkpoint loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    model = model.to(device)
    model.eval()

    # ── Run inference ─────────────────────────────────────────────────────────
    log("Running surrogate inference ...")
    pkd_matrix = run_inference(model, graphs, device)
    # pkd_matrix shape: [n_valid_graphs, 6]

    log(f"Inference complete. Shape: {pkd_matrix.shape}")
    heartbeat("phase13", 0, {
        "n_generated": len(smiles_list),
        "n_graphs": len(graphs),
        "n_lipinski_pass": n_lip_pass,
    })

    # ── Assemble full results ─────────────────────────────────────────────────
    log("Assembling full results ...")
    full_results = []
    for gi, smi_idx in enumerate(graph2smi):
        smi             = smiles_list[smi_idx]
        lip_pass, props = lipinski_results[smi_idx]
        pkd_per_target  = {TARGETS[t]: round(float(pkd_matrix[gi, t]), 4)
                           for t in range(len(TARGETS))}
        full_results.append({
            "smiles":       smi,
            "pkd_per_target": pkd_per_target,
            "lipinski_pass":  lip_pass,
            "lipinski_props": props,
        })

    # ── Top hits per target ───────────────────────────────────────────────────
    def top_hits(target_name, n=TOP_N):
        tid = TARGET2ID[target_name]
        ranked = sorted(
            [(gi, graph2smi[gi]) for gi in range(len(graphs))],
            key=lambda x: -pkd_matrix[x[0], tid],
        )
        hits = []
        for gi, smi_idx in ranked[:n]:
            smi            = smiles_list[smi_idx]
            lip_pass, props = lipinski_results[smi_idx]
            hits.append({
                "smiles":        smi,
                "predicted_pkd": round(float(pkd_matrix[gi, tid]), 4),
                "lipinski_pass": lip_pass,
                "lipinski_props": props,
            })
        return hits

    top_all = {t: top_hits(t) for t in TARGETS}

    # Summary stats per target
    for t in TARGETS:
        tid   = TARGET2ID[t]
        pkds  = pkd_matrix[:, tid]
        log(f"  {t}: mean pKd={pkds.mean():.3f}  max={pkds.max():.3f}  "
            f"top10_mean={np.sort(pkds)[-10:].mean():.3f}")

    # ── Build output dict ─────────────────────────────────────────────────────
    output = {
        "n_generated":          len(smiles_list),
        "n_graphs_built":       len(graphs),
        "n_lipinski_pass":      n_lip_pass,
        "surrogate_checkpoint": "phase6/fp32_best.pt",
        "targets":              TARGETS,
        "top_hits_KPC3":        top_all["KPC3"],
        "top_hits_PCSK9":       top_all["PCSK9"],
        "top_hits_all_targets": top_all,
        "full_results":         full_results,
    }

    results_path = OUT_DIR / "phase13_results.json"
    results_path.write_text(json.dumps(output, indent=2))
    log(f"Results written: {results_path} ({results_path.stat().st_size // 1024} KB)")

    # ── Upload to GCS ─────────────────────────────────────────────────────────
    log("Uploading to GCS ...")
    gcs_phase13 = f"{GCS_BASE}/phase13_screen_generated"
    gsutil_cp(results_path, f"{gcs_phase13}/phase13_results.json")

    # Also save a compact top-hits-only file
    compact = {
        "n_generated":     len(smiles_list),
        "n_lipinski_pass": n_lip_pass,
        "top_hits_KPC3":   top_all["KPC3"][:10],
        "top_hits_PCSK9":  top_all["PCSK9"][:10],
    }
    compact_path = OUT_DIR / "phase13_top_hits.json"
    compact_path.write_text(json.dumps(compact, indent=2))
    gsutil_cp(compact_path, f"{gcs_phase13}/phase13_top_hits.json")

    log(f"Uploaded to {gcs_phase13}/")

    # ── Final report ─────────────────────────────────────────────────────────
    top_kpc3  = top_all["KPC3"][0]  if top_all["KPC3"]  else {}
    top_pcsk9 = top_all["PCSK9"][0] if top_all["PCSK9"] else {}
    summary = {
        "n_generated":          len(smiles_list),
        "n_lipinski_pass":      n_lip_pass,
        "best_KPC3_pkd":        top_kpc3.get("predicted_pkd"),
        "best_KPC3_smiles":     top_kpc3.get("smiles"),
        "best_PCSK9_pkd":       top_pcsk9.get("predicted_pkd"),
        "best_PCSK9_smiles":    top_pcsk9.get("smiles"),
    }
    log("Phase 13 complete.")
    for k, v in summary.items():
        log(f"  {k}: {v}")

    notify("PHASE_COMPLETE", "Phase 13 screen complete — top KPC3 hits identified",
           data=summary)
    log("PHASE_COMPLETE")


if __name__ == "__main__":
    main()
