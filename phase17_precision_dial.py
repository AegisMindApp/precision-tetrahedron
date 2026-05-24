#!/usr/bin/env python3
"""
phase17_precision_dial.py
-------------------------
Precision dial: train MolecularGNN-256 at four precision levels from
identical initialisations and measure how LMC barrier scales with
mantissa bit-width.

Precisions tested:
  FP32    — 23-bit mantissa (baseline)
  BF16    — 7-bit mantissa (torch.autocast bfloat16)
  FP16    — 10-bit mantissa (torch.autocast float16)
  INT8sim — simulated 8-bit: weights/activations fake-quantised to 8-bit
            dynamic range at each forward pass; accumulation stays FP32

Protocol per precision:
  1. Train 80 epochs from seed 42 (same init, same data order)
  2. Measure GNS every 5 epochs (20 mini-batches, McCandlish 2018)
  3. Save ep80 checkpoint

After all four runs:
  4. Run LMC between fp32_ep80 and each of {bf16, fp16, int8sim}_ep80
     (11 α interpolation points, barrier = max_α val_mae − baseline)
  5. Report barrier vs bit-width table

Hypothesis: if precision-induced sharpening scales monotonically with
reduced mantissa bits, barrier(INT8sim) > barrier(FP16) > barrier(BF16) >> 0.

Phase 7 reference: fp32_ep80 ↔ bf16_ep83 = 1.447 eV;
                   fp32_ep80 ↔ fp32_ep83 = 0.005 eV (273× ratio)

Note: INT8sim uses torch.fake_quantize_per_tensor_affine on weight and
activation tensors at each forward pass (dynamic scale per-tensor).
FP16 may produce NaN on TPU if gradients overflow; gradient clipping to
1.0 is applied for FP16 only.

GCS output: gs://.../aegis_flashoptim/phase17_precision_dial/results.json
"""

import os, sys, json, time, subprocess, copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
    try:
        import torch_xla.experimental as _e
        _e.eager_mode(True)
    except Exception:
        pass
except ImportError:
    XLA_AVAILABLE = False

sys.path.insert(0, os.path.expanduser("~/flashoptim"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MolecularGNN
from data import get_dataloaders, batch_to_graph
from notify import notify, heartbeat

GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID     = os.environ.get("RUN_ID",     "aegis_flashoptim")
GCS_BASE   = f"{GCS_BUCKET}/{RUN_ID}"

OUT_DIR  = Path("/tmp/phase17_precision_dial")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("/tmp/qm9")

HIDDEN_DIM    = 256
N_BLOCKS      = 6
N_EPOCHS      = 80
BATCH_SIZE    = 32
LR_INIT       = 1e-4
WEIGHT_DECAY  = 1e-4
SEED          = 42
N_GNS_BATCHES = 20
GNS_EVERY     = 5
N_ALPHA       = 11
FP16_GRAD_CLIP = 1.0

PRECISION_CONFIGS = [
    {"label": "fp32",    "dtype": None,              "mantissa_bits": 23},
    {"label": "bf16",    "dtype": torch.bfloat16,    "mantissa_bits": 7},
    {"label": "fp16",    "dtype": torch.float16,     "mantissa_bits": 10},
    {"label": "int8sim", "dtype": None,              "mantissa_bits": 8},  # fake quant
]

def log(msg): print(f"[Phase17-Dial] {msg}", flush=True)
def gsutil_cp(src, dst): subprocess.run(["gsutil", "-q", "cp", str(src), dst], check=False)


def get_device():
    if XLA_AVAILABLE:
        return xm.xla_device()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_model(device):
    torch.manual_seed(SEED)
    return MolecularGNN(hidden_dim=HIDDEN_DIM, num_blocks=N_BLOCKS).to(device)


def fake_quantize_model(model):
    """Apply INT8-range fake quantization to all float parameters in-place."""
    with torch.no_grad():
        for p in model.parameters():
            if p.dtype == torch.float32:
                scale = p.abs().max() / 127.0 + 1e-8
                p.copy_(torch.fake_quantize_per_tensor_affine(
                    p, scale.item(), 0, -128, 127))


def eval_mae(model, loader, device, int8sim=False):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            z, pos, edge_src, edge_dst, assign_mat, B, edge_valid, atom_valid = batch_to_graph(batch, device)
            y = batch['target'].to(device)
            if int8sim:
                fake_quantize_model(model)
            pred = model(z, pos, edge_src, edge_dst, assign_mat, B, edge_valid, atom_valid)
            total += F.l1_loss(pred.squeeze(), y, reduction="sum").item()
            count += y.numel()
    if XLA_AVAILABLE:
        xm.mark_step()
    return total / count if count > 0 else float("inf")


def compute_gns(model, loader, device, n_batches, autocast_dtype, int8sim):
    """GNS = ||E[g]||² / Var[g] (McCandlish 2018) from n_batches samples."""
    grad_samples = []
    model.eval()
    loader_iter = iter(loader)
    for _ in range(n_batches):
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        z, pos, edge_src, edge_dst, assign_mat, B, edge_valid, atom_valid = batch_to_graph(batch, device)
        y = batch['target'].to(device)
        model.zero_grad()
        if int8sim:
            fake_quantize_model(model)
        if autocast_dtype is not None:
            dev_type = "xla" if XLA_AVAILABLE else ("cuda" if "cuda" in str(device) else "cpu")
            with torch.autocast(device_type=dev_type, dtype=autocast_dtype):
                loss = F.mse_loss(model(z, pos, edge_src, edge_dst, assign_mat, B, edge_valid, atom_valid).squeeze(), y)
        else:
            loss = F.mse_loss(model(z, pos, edge_src, edge_dst, assign_mat, B, edge_valid, atom_valid).squeeze(), y)
        loss.backward()
        if XLA_AVAILABLE:
            xm.mark_step()
        g = torch.cat([p.grad.detach().float().flatten()
                       for p in model.parameters() if p.grad is not None])
        grad_samples.append(g.cpu())
        model.zero_grad()

    if not grad_samples:
        return None
    G = torch.stack(grad_samples)             # [N, P]
    mean_g  = G.mean(0)                       # [P]
    var_g   = G.var(0, unbiased=True)         # [P]
    numerator   = (mean_g ** 2).sum().item()
    denominator = var_g.sum().item() + 1e-12
    return round(numerator / denominator, 6)


def train_precision(cfg, train_loader, val_loader, device):
    label       = cfg["label"]
    autocast_dt = cfg["dtype"]
    int8sim     = (label == "int8sim")

    log(f"\n  ── {label.upper()} ({cfg['mantissa_bits']} mantissa bits) ──")

    # Resume: check GCS for existing checkpoint and skip training if found
    ckpt_path = OUT_DIR / f"{label}_ep80.pt"
    gcs_ckpt  = f"{GCS_BASE}/phase17_precision_dial/{ckpt_path.name}"
    if not ckpt_path.exists():
        stat_r = subprocess.run(["gsutil", "-q", "stat", gcs_ckpt], capture_output=True)
        if stat_r.returncode == 0:
            log(f"  Checkpoint found on GCS — downloading and skipping training")
            subprocess.run(["gsutil", "-q", "cp", gcs_ckpt, str(ckpt_path)], check=False)
    if ckpt_path.exists():
        model = make_model(device)
        sd = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(sd)
        model.to(device)
        final_mae = eval_mae(model, val_loader, device, int8sim)
        log(f"  {label} resumed val_mae = {final_mae:.4f} eV")
        return {"trajectory": [], "final_mae_ev": final_mae, "ckpt": str(ckpt_path)}

    model = make_model(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-6)

    trajectory = []

    for ep in range(1, N_EPOCHS + 1):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            z, pos, edge_src, edge_dst, assign_mat, B, edge_valid, atom_valid = batch_to_graph(batch, device)
            y = batch['target'].to(device)
            opt.zero_grad()
            if int8sim:
                fake_quantize_model(model)
                pred  = model(z, pos, edge_src, edge_dst, assign_mat, B, edge_valid, atom_valid)
                loss  = F.mse_loss(pred.squeeze(), y)
            elif autocast_dt is not None:
                dev_type = "xla" if XLA_AVAILABLE else ("cuda" if "cuda" in str(device) else "cpu")
                with torch.autocast(device_type=dev_type, dtype=autocast_dt):
                    pred = model(z, pos, edge_src, edge_dst, assign_mat, B, edge_valid, atom_valid)
                    loss = F.mse_loss(pred.squeeze(), y)
            else:
                pred = model(z, pos, edge_src, edge_dst, assign_mat, B, edge_valid, atom_valid)
                loss = F.mse_loss(pred.squeeze(), y)
            loss.backward()
            if label == "fp16":
                torch.nn.utils.clip_grad_norm_(model.parameters(), FP16_GRAD_CLIP)
            if XLA_AVAILABLE:
                xm.optimizer_step(opt)
            else:
                opt.step()
            total += loss.item() * y.numel()
            count += y.numel()
        sched.step()
        if XLA_AVAILABLE:
            xm.mark_step()

        if ep % GNS_EVERY == 0:
            val_mae = eval_mae(model, val_loader, device, int8sim)
            gns     = compute_gns(model, train_loader, device,
                                  N_GNS_BATCHES, autocast_dt, int8sim)
            entry = {"epoch": ep, "precision": label,
                     "train_loss": round(total / count, 6),
                     "val_mae_ev": round(val_mae, 4), "gns": gns}
            trajectory.append(entry)
            log(f"    {label} ep{ep:3d}  val_mae={val_mae:.4f}  gns={gns}")
            heartbeat(f"Phase17_{label}", epoch=ep, metrics={"val_mae": round(val_mae, 4), "gns": gns})
        elif ep % 10 == 0:
            log(f"    {label} ep{ep:3d}  train_loss={total/count:.6f}")

    final_mae = eval_mae(model, val_loader, device, int8sim)
    ckpt_path = OUT_DIR / f"{label}_ep80.pt"
    torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()}, ckpt_path)
    gsutil_cp(ckpt_path, f"{GCS_BASE}/phase17_precision_dial/{ckpt_path.name}")
    log(f"  {label} final val_mae = {final_mae:.4f} eV  → {ckpt_path}")
    return {"trajectory": trajectory, "final_mae_ev": final_mae, "ckpt": str(ckpt_path)}


def lmc_barrier(label, sd0_path, sd1_path, val_loader, device):
    sd0   = torch.load(sd0_path, map_location="cpu")
    sd1   = torch.load(sd1_path, map_location="cpu")
    model = make_model(device)
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
    log(f"  LMC {label:40s}  barrier={barrier:.4f} eV  peak_α={peak_a}")
    return {"label": label, "alphas": alphas, "maes": maes,
            "barrier_ev": barrier, "baseline_ev": baseline, "peak_alpha": peak_a}


def main():
    log("=" * 65)
    log("  Phase 17 — Precision Dial: barrier vs mantissa bit-width")
    log(f"  Precisions: FP32 / BF16 / FP16 / INT8sim  |  {N_EPOCHS} epochs")
    log("=" * 65)

    device = get_device()
    log(f"Device: {device}")
    train_loader, val_loader, _ = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)

    run_results = {}
    for cfg in PRECISION_CONFIGS:
        run_results[cfg["label"]] = train_precision(cfg, train_loader, val_loader, device)

    # LMC: fp32_ep80 ↔ each other precision's ep80
    log("\nRunning LMC (fp32 vs each precision)...")
    lmc_results = []
    fp32_path = OUT_DIR / "fp32_ep80.pt"
    for cfg in PRECISION_CONFIGS[1:]:   # skip fp32 vs fp32
        label = cfg["label"]
        other_path = OUT_DIR / f"{label}_ep80.pt"
        r = lmc_barrier(f"fp32_ep80 ↔ {label}_ep80", fp32_path, other_path,
                        val_loader, device)
        r["precision"] = label
        r["mantissa_bits"] = cfg["mantissa_bits"]
        lmc_results.append(r)

    # ── Report ────────────────────────────────────────────────────────────────
    log("\n" + "=" * 65)
    log("  PRECISION DIAL RESULTS")
    log("=" * 65)
    log(f"\n  {'Precision':10s}  {'Mantissa':>10}  {'Final MAE (eV)':>16}  {'LMC barrier (eV)':>18}")
    log(f"  {'─'*10}  {'─'*10}  {'─'*16}  {'─'*18}")
    log(f"  {'FP32':10s}  {'23 bits':>10}  {run_results['fp32']['final_mae_ev']:>16.4f}  {'—':>18}")
    for r in lmc_results:
        prec = r["precision"]
        log(f"  {prec:10s}  {r['mantissa_bits']:>10}  "
            f"{run_results[prec]['final_mae_ev']:>16.4f}  "
            f"{r['barrier_ev']:>18.4f}")

    # Check monotonicity
    barriers = [(r["mantissa_bits"], r["barrier_ev"]) for r in lmc_results]
    barriers.sort(key=lambda x: -x[0])  # sort descending by bit-width (FP16 > BF16 > INT8sim)
    monotone = all(barriers[i][1] <= barriers[i+1][1]
                   for i in range(len(barriers)-1))
    log(f"\n  Barrier monotone with decreasing bit-width: {'YES ✓' if monotone else 'NO ✗'}")
    log(f"  Phase 7 BF16 reference barrier: 1.447 eV")

    results = {
        "experiment":       "phase17_precision_dial",
        "hidden_dim":       HIDDEN_DIM,
        "n_epochs":         N_EPOCHS,
        "seed":             SEED,
        "runs":             run_results,
        "lmc_results":      lmc_results,
        "monotone_barrier": monotone,
        "phase7_ref_barrier_ev": 1.447,
    }
    out = OUT_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    gsutil_cp(out, f"{GCS_BASE}/phase17_precision_dial/results.json")
    log(f"\n  Results → {out}")
    log(f"  GCS     → {GCS_BASE}/phase17_precision_dial/results.json")
    notify("phase17_complete", "Phase 17 complete — precision dial done", data=results)


if __name__ == "__main__":
    main()
