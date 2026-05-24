#!/usr/bin/env python3
"""
phase17_rescue_lmc.py
---------------------
Rescue script: INT8sim diverged and is too slow to complete (val_mae=469 eV
at ep5, ~82h projected runtime). This script:
  1. Downloads fp32/bf16/fp16 ep80 checkpoints from GCS
  2. Loads val_loader
  3. Runs LMC for fp32↔bf16 and fp32↔fp16 (2 valid pairs)
  4. Writes results.json with INT8sim flagged as diverged
  5. Uploads results.json to GCS
  6. Touches /tmp/.phase17_done

Run in a new tmux window while (or after) killing the INT8sim process:
  python3 ~/flashoptim/phase17_rescue_lmc.py 2>&1 | tee ~/pipeline_logs/phase17_rescue.log
"""

import os, sys, json, time, subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False

sys.path.insert(0, os.path.expanduser("~/flashoptim"))
from model import MolecularGNN
from data import get_dataloaders, batch_to_graph
from notify import notify

GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID     = os.environ.get("RUN_ID",     "aegis_flashoptim")
GCS_BASE   = f"{GCS_BUCKET}/{RUN_ID}"
GCS_P17    = f"{GCS_BASE}/phase17_precision_dial"

OUT_DIR  = Path("/tmp/phase17_precision_dial")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("/tmp/qm9")

HIDDEN_DIM = 256
N_BLOCKS   = 6
N_EPOCHS   = 80
BATCH_SIZE = 32
SEED       = 42
N_ALPHA    = 11

def log(msg): print(f"[Phase17-Rescue] {msg}", flush=True)
def gsutil_cp(src, dst): subprocess.run(["gsutil", "-q", "cp", str(src), dst], check=False)


def get_device():
    if XLA_AVAILABLE:
        return xm.xla_device()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_model(device):
    torch.manual_seed(SEED)
    return MolecularGNN(hidden_dim=HIDDEN_DIM, num_blocks=N_BLOCKS).to(device)


def eval_mae(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            z, pos, edge_src, edge_dst, assign_mat, B, edge_valid, atom_valid = batch_to_graph(batch, device)
            y = batch['target'].to(device)
            pred = model(z, pos, edge_src, edge_dst, assign_mat, B, edge_valid, atom_valid)
            total += F.l1_loss(pred.squeeze(), y, reduction="sum").item()
            count += y.numel()
    if XLA_AVAILABLE:
        xm.mark_step()
    return total / count if count > 0 else float("inf")


def lmc_barrier(label, sd0_path, sd1_path, val_loader, device):
    log(f"  LMC {label} ...")
    sd0 = torch.load(sd0_path, map_location="cpu")
    sd1 = torch.load(sd1_path, map_location="cpu")
    model  = make_model(device)
    alphas = [round(i / (N_ALPHA - 1), 2) for i in range(N_ALPHA)]
    maes   = []
    for alpha in alphas:
        interp = {k: ((1 - alpha) * sd0[k].float() + alpha * sd1[k].float()).to(device)
                  for k in sd0}
        model.load_state_dict(interp)
        maes.append(round(eval_mae(model, val_loader, device), 5))
    baseline = min(maes[0], maes[-1])
    barrier  = round(max(maes) - baseline, 5)
    peak_a   = alphas[maes.index(max(maes))]
    log(f"    {label:40s}  barrier={barrier:.4f} eV  peak_α={peak_a}")
    return {"label": label, "alphas": alphas, "maes": maes,
            "barrier_ev": barrier, "baseline_ev": baseline, "peak_alpha": peak_a}


def main():
    log("=" * 65)
    log("  Phase 17 Rescue — LMC on FP32 / BF16 / FP16 checkpoints")
    log("  INT8sim excluded: diverged at ep5 (val_mae=469 eV)")
    log("=" * 65)

    device = get_device()
    log(f"Device: {device}")
    _, val_loader, _ = get_dataloaders(DATA_DIR, batch_size=32)

    # Download checkpoints from GCS if not already local
    for label in ["fp32", "bf16", "fp16"]:
        ckpt = OUT_DIR / f"{label}_ep80.pt"
        if not ckpt.exists():
            log(f"  Downloading {label}_ep80.pt from GCS...")
            r = subprocess.run(["gsutil", "-q", "cp",
                                f"{GCS_P17}/{label}_ep80.pt", str(ckpt)], check=False)
            if r.returncode != 0:
                log(f"  ERROR: failed to download {label}_ep80.pt — aborting")
                sys.exit(1)
        log(f"  {label}_ep80.pt: {ckpt.stat().st_size // 1024} KB")

    # Verify final MAEs from checkpoints
    run_mae = {}
    for label in ["fp32", "bf16", "fp16"]:
        ckpt = OUT_DIR / f"{label}_ep80.pt"
        model = make_model(device)
        sd = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(sd)
        model.to(device)
        mae = round(eval_mae(model, val_loader, device), 5)
        run_mae[label] = mae
        log(f"  {label} ep80 val_mae = {mae:.4f} eV")

    # LMC pairs (fp32 is the reference in all pairs)
    fp32_path = OUT_DIR / "fp32_ep80.pt"
    lmc_results = []
    for label, bits in [("bf16", 7), ("fp16", 10)]:
        other_path = OUT_DIR / f"{label}_ep80.pt"
        r = lmc_barrier(f"fp32_ep80 ↔ {label}_ep80", fp32_path, other_path,
                        val_loader, device)
        r["precision"]     = label
        r["mantissa_bits"] = bits
        lmc_results.append(r)

    # Monotonicity check: fp32 baseline, then bf16 (7 bits), fp16 (10 bits)
    # Monotone = barrier decreases as mantissa bits increase
    # i.e. barrier(fp32↔bf16) > barrier(fp32↔fp16) for monotone
    bf16_barrier = next(r["barrier_ev"] for r in lmc_results if r["precision"] == "bf16")
    fp16_barrier = next(r["barrier_ev"] for r in lmc_results if r["precision"] == "fp16")
    # Sorted by mantissa bits ascending: bf16(7) < fp16(10)
    # Monotone (fewer bits → larger barrier) means bf16_barrier > fp16_barrier
    monotone = bf16_barrier > fp16_barrier

    # Results summary
    log("\n" + "=" * 65)
    log("  PRECISION DIAL RESULTS (INT8sim excluded — diverged)")
    log("=" * 65)
    log(f"\n  {'Precision':10s}  {'Mantissa':>10}  {'Final MAE (eV)':>16}  {'LMC barrier (eV)':>18}")
    log(f"  {'─'*10}  {'─'*10}  {'─'*16}  {'─'*18}")
    log(f"  {'FP32':10s}  {'23 bits':>10}  {run_mae['fp32']:>16.4f}  {'—':>18}")
    for r in lmc_results:
        log(f"  {r['precision']:10s}  {r['mantissa_bits']:>10}  "
            f"{run_mae[r['precision']]:>16.4f}  {r['barrier_ev']:>18.4f}")
    log(f"  {'INT8SIM':10s}  {'8 bits':>10}  {'DIVERGED':>16}  {'N/A':>18}")
    log(f"\n  Barrier monotone (fewer bits → larger barrier): {'YES ✓' if monotone else 'NO ✗'}")
    log(f"  bf16 barrier={bf16_barrier:.4f} eV  fp16 barrier={fp16_barrier:.4f} eV")
    log(f"  Phase 7 BF16 reference barrier: 1.447 eV")

    results = {
        "experiment":       "phase17_precision_dial",
        "rescue":           True,
        "rescue_reason":    "INT8sim diverged at ep5 (val_mae=469 eV, ~82h projected runtime)",
        "hidden_dim":       HIDDEN_DIM,
        "n_epochs":         N_EPOCHS,
        "seed":             SEED,
        "runs": {
            "fp32": {"final_mae_ev": run_mae["fp32"], "trajectory": [], "ckpt": str(OUT_DIR / "fp32_ep80.pt")},
            "bf16": {"final_mae_ev": run_mae["bf16"], "trajectory": [], "ckpt": str(OUT_DIR / "bf16_ep80.pt")},
            "fp16": {"final_mae_ev": run_mae["fp16"], "trajectory": [], "ckpt": str(OUT_DIR / "fp16_ep80.pt")},
            "int8sim": {
                "final_mae_ev": None,
                "diverged": True,
                "diverge_ep": 5,
                "diverge_val_mae_ev": 469.0657,
                "diverge_gns": 120.053179,
                "note": "Naive fake-quantize at every step incompatible with cosine LR from scratch; requires QAT-specific schedule",
            },
        },
        "lmc_results":      lmc_results,
        "monotone_barrier": monotone,
        "phase7_ref_barrier_ev": 1.447,
    }

    out = OUT_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    gsutil_cp(out, f"{GCS_P17}/results.json")
    log(f"\n  Results → {out}")
    log(f"  GCS     → {GCS_P17}/results.json")

    # Mark done
    Path("/tmp/.phase17_done").touch()
    log("  /tmp/.phase17_done created")

    notify("phase17_complete", "Phase 17 rescue complete — precision dial (FP32/BF16/FP16) done, INT8sim diverged", data=results)
    log("\n=== Phase 17 rescue complete ===")


if __name__ == "__main__":
    main()
