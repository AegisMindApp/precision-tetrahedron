#!/usr/bin/env python3
"""
phase17b_bf16_fp16_lmc.py
--------------------------
Computes the LMC barrier between bf16_ep80 and fp16_ep80 checkpoints
from Phase 17, using already-saved GCS checkpoints.

Phase 17 measured:
  fp32↔bf16: 0.0142 eV  (shared 8-bit exponent)
  fp32↔fp16: 0.1485 eV  (FP16 has 5-bit exponent)

This adds the third edge of the triangle:
  bf16↔fp16: ?           (8-bit exp vs 5-bit exp — expected large)

If bf16↔fp16 ≈ fp32↔fp16 (~0.15 eV): exponent-range isolation is symmetric;
FP16 is equally isolated from all 8-bit-exponent formats.
If bf16↔fp16 >> fp32↔fp16: BF16's reduced mantissa adds additional separation.

Run in new tmux window (Phase 18 can run concurrently):
  python3 ~/flashoptim/phase17b_bf16_fp16_lmc.py 2>&1 | tee ~/pipeline_logs/phase17b.log
"""

import os, sys, json, subprocess
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

GCS_P17 = "gs://aegismind-tpu-results/aegis_flashoptim/phase17_precision_dial"
OUT_DIR  = Path("/tmp/phase17_precision_dial")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("/tmp/qm9")

HIDDEN_DIM = 256
N_BLOCKS   = 6
BATCH_SIZE = 32
SEED       = 42
N_ALPHA    = 11

def log(msg): print(f"[Phase17b-LMC] {msg}", flush=True)
def gsutil_cp(src, dst): subprocess.run(["gsutil", "-q", "cp", str(src), dst], check=False)


def get_device():
    if XLA_AVAILABLE:
        try:
            return xm.xla_device()
        except RuntimeError:
            log("TPU busy (held by Phase 18) — falling back to CPU")
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
    if XLA_AVAILABLE and str(device) != "cpu":
        xm.mark_step()
    return total / count if count > 0 else float("inf")


def lmc_barrier(label, sd0_path, sd1_path, val_loader, device):
    log(f"Running LMC: {label}")
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
        log(f"  α={alpha:.1f}  mae={maes[-1]:.5f}")
    baseline = min(maes[0], maes[-1])
    barrier  = round(max(maes) - baseline, 5)
    peak_a   = alphas[maes.index(max(maes))]
    log(f"  → barrier={barrier:.5f} eV  peak_α={peak_a}  baseline={baseline:.5f}")
    return {"label": label, "alphas": alphas, "maes": maes,
            "barrier_ev": barrier, "baseline_ev": baseline, "peak_alpha": peak_a}


def main():
    log("=" * 60)
    log("  Phase 17b — BF16↔FP16 LMC (exponent-range triangle)")
    log("  BF16: 7 mantissa bits, 8-bit exponent")
    log("  FP16: 10 mantissa bits, 5-bit exponent")
    log("=" * 60)

    device = get_device()
    log(f"Device: {device}")
    _, val_loader, _ = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)

    for label in ["bf16", "fp16"]:
        ckpt = OUT_DIR / f"{label}_ep80.pt"
        if not ckpt.exists():
            log(f"Downloading {label}_ep80.pt from GCS...")
            subprocess.run(["gsutil", "-q", "cp",
                            f"{GCS_P17}/{label}_ep80.pt", str(ckpt)], check=True)
        log(f"  {label}_ep80.pt: {ckpt.stat().st_size // 1024} KB")

    bf16_path = OUT_DIR / "bf16_ep80.pt"
    fp16_path = OUT_DIR / "fp16_ep80.pt"

    result = lmc_barrier("bf16_ep80 ↔ fp16_ep80", bf16_path, fp16_path, val_loader, device)

    log("\n" + "=" * 60)
    log("  RESULT")
    log("=" * 60)
    log(f"  bf16↔fp16 barrier : {result['barrier_ev']:.5f} eV  (peak α={result['peak_alpha']})")
    log(f"  fp32↔bf16 barrier : 0.01422 eV  (Phase 17 reference)")
    log(f"  fp32↔fp16 barrier : 0.14845 eV  (Phase 17 reference)")
    log(f"  Ratio bf16↔fp16 / fp32↔bf16 : {result['barrier_ev']/0.01422:.1f}×")

    out = OUT_DIR / "phase17b_results.json"
    out.write_text(json.dumps({
        "experiment": "phase17b_bf16_fp16_lmc",
        "lmc": result,
        "phase17_ref": {"fp32_bf16": 0.01422, "fp32_fp16": 0.14845},
    }, indent=2))
    subprocess.run(["gsutil", "-q", "cp", str(out),
                    f"{GCS_P17}/phase17b_results.json"], check=False)
    log(f"  Results → {out}")
    log(f"  GCS     → {GCS_P17}/phase17b_results.json")
    log("\n=== Phase 17b complete ===")


if __name__ == "__main__":
    main()
