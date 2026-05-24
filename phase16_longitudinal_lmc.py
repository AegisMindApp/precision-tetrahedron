#!/usr/bin/env python3
"""
phase16_longitudinal_lmc.py
---------------------------
Longitudinal Linear Mode Connectivity sweep: train FP32 and BF16
MolecularGNN-256 from the same seed, save checkpoints every 20 epochs,
and measure LMC barriers within and across precisions.

Key question: at which epoch do FP32 and BF16 weight trajectories first
diverge into topologically distinct loss basins?

Checkpoint pairs tested (12 total):
  Intra-FP32:  ep0↔ep20, ep20↔ep40, ep40↔ep60, ep60↔ep80
  Intra-BF16:  ep0↔ep20, ep20↔ep40, ep40↔ep60, ep60↔ep80
  Cross-prec:  fp32_ep20↔bf16_ep20, fp32_ep40↔bf16_ep40,
               fp32_ep60↔bf16_ep60, fp32_ep80↔bf16_ep80

LMC protocol: θ(α) = (1−α)θ₀ + αθ₁ at 11 α values [0..1];
  barrier = max_α val_mae − min(val_mae[0], val_mae[1])

Reference: Phase 7 established fp32_ep80 ↔ bf16_ep83 barrier = 1.447 eV (273×
the 0.005 eV fp32_ep80 ↔ fp32_ep83 barrier). Phase 16 localises *when* in
training this divergence first appears.

GCS output: gs://.../aegis_flashoptim/phase16_longitudinal_lmc/results.json
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

OUT_DIR  = Path("/tmp/phase16_longitudinal_lmc")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("/tmp/qm9")

HIDDEN_DIM   = 256
N_BLOCKS     = 6
N_EPOCHS     = 80
BATCH_SIZE   = 32
LR_INIT      = 1e-4
WEIGHT_DECAY = 1e-4
SEED         = 42
SAVE_EVERY   = 20   # save checkpoint at ep 20, 40, 60, 80
N_ALPHA      = 11   # LMC interpolation points

def log(msg): print(f"[Phase16-LMC] {msg}", flush=True)
def gsutil_cp(src, dst): subprocess.run(["gsutil", "-q", "cp", str(src), dst], check=False)


def get_device():
    if XLA_AVAILABLE:
        return xm.xla_device()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_model(device):
    torch.manual_seed(SEED)
    model = MolecularGNN(hidden_dim=HIDDEN_DIM, num_blocks=N_BLOCKS).to(device)
    return model


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


def train_one_epoch(model, loader, opt, scheduler, device, use_bf16):
    model.train()
    total, count = 0.0, 0
    ctx = (torch.autocast(device_type="cpu", dtype=torch.bfloat16)
           if use_bf16 and not XLA_AVAILABLE
           else torch.autocast(device_type="xla", dtype=torch.bfloat16)
           if use_bf16 and XLA_AVAILABLE
           else torch.no_grad().__class__())  # fallback – overridden below
    if not use_bf16:
        ctx = None
    for batch in loader:
        z, pos, edge_src, edge_dst, assign_mat, B, edge_valid, atom_valid = batch_to_graph(batch, device)
        y = batch['target'].to(device)
        opt.zero_grad()
        if use_bf16:
            with (torch.autocast(device_type="xla", dtype=torch.bfloat16)
                  if XLA_AVAILABLE
                  else torch.autocast(device_type="cpu", dtype=torch.bfloat16)):
                pred = model(z, pos, edge_src, edge_dst, assign_mat, B, edge_valid, atom_valid)
                loss = F.mse_loss(pred.squeeze(), y)
        else:
            pred = model(z, pos, edge_src, edge_dst, assign_mat, B, edge_valid, atom_valid)
            loss = F.mse_loss(pred.squeeze(), y)
        loss.backward()
        if XLA_AVAILABLE:
            xm.optimizer_step(opt)
        else:
            opt.step()
        total += loss.item() * y.numel()
        count += y.numel()
    scheduler.step()
    if XLA_AVAILABLE:
        xm.mark_step()
    return total / count if count > 0 else float("inf")


def train_run(precision_label, use_bf16, train_loader, val_loader, device):
    """Train for N_EPOCHS, save checkpoints every SAVE_EVERY epochs.
    Returns dict: {ep: state_dict_path, 'ep0': state_dict_path}."""
    log(f"  Training {precision_label} — {N_EPOCHS} epochs")
    model = make_model(device)

    # Save ep0 (random init)
    ep0_path = OUT_DIR / f"{precision_label}_ep0.pt"
    torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()}, ep0_path)
    checkpoints = {0: ep0_path}

    opt = torch.optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-6)

    for ep in range(1, N_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, opt, scheduler, device, use_bf16)
        if ep % SAVE_EVERY == 0:
            val_mae = eval_mae(model, val_loader, device)
            path = OUT_DIR / f"{precision_label}_ep{ep}.pt"
            torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()}, path)
            checkpoints[ep] = path
            log(f"    {precision_label} ep{ep:3d}  train_loss={train_loss:.6f}  val_mae={val_mae:.4f} eV")
            heartbeat(f"Phase16_{precision_label}", epoch=ep, metrics={"val_mae": round(val_mae, 4)})
        elif ep % 10 == 0:
            log(f"    {precision_label} ep{ep:3d}  train_loss={train_loss:.6f}")

    final_mae = eval_mae(model, val_loader, device)
    log(f"  {precision_label} final val_mae = {final_mae:.4f} eV")
    return checkpoints, final_mae


def lmc_pair(label, sd0_path, sd1_path, val_loader, device):
    """Compute LMC barrier between two checkpoint files."""
    sd0 = torch.load(sd0_path, map_location="cpu")
    sd1 = torch.load(sd1_path, map_location="cpu")
    model = make_model(device)

    alphas = [round(i / (N_ALPHA - 1), 2) for i in range(N_ALPHA)]
    maes = []
    for alpha in alphas:
        interp = {k: ((1 - alpha) * sd0[k].float() + alpha * sd1[k].float()).to(device)
                  for k in sd0}
        model.load_state_dict(interp)
        maes.append(round(eval_mae(model, val_loader, device), 5))

    baseline  = min(maes[0], maes[-1])
    barrier   = round(max(maes) - baseline, 5)
    peak_alpha = alphas[maes.index(max(maes))]
    log(f"  LMC {label:45s}  barrier={barrier:.4f} eV  peak_α={peak_alpha}")
    return {"label": label, "alphas": alphas, "maes": maes,
            "barrier_ev": barrier, "baseline_ev": baseline, "peak_alpha": peak_alpha}


def main():
    log("=" * 65)
    log("  Phase 16 — Longitudinal LMC: when do FP32/BF16 diverge?")
    log(f"  {N_EPOCHS} epochs × 2 precisions, checkpoint every {SAVE_EVERY} ep")
    log("=" * 65)

    device = get_device()
    log(f"Device: {device}")

    train_loader, val_loader, _ = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)

    # ── Train both precisions ─────────────────────────────────────────────────
    fp32_ckpts, fp32_final = train_run("fp32", False, train_loader, val_loader, device)
    bf16_ckpts, bf16_final = train_run("bf16", True,  train_loader, val_loader, device)

    # Upload all checkpoints
    for path in OUT_DIR.glob("*.pt"):
        gsutil_cp(path, f"{GCS_BASE}/phase16_longitudinal_lmc/{path.name}")

    # ── LMC sweep ─────────────────────────────────────────────────────────────
    log("\nRunning LMC pairs...")
    lmc_results = []

    epochs = [0, 20, 40, 60, 80]

    # Intra-FP32: consecutive 20-epoch windows
    for i in range(len(epochs) - 1):
        e0, e1 = epochs[i], epochs[i + 1]
        r = lmc_pair(f"fp32_ep{e0}↔fp32_ep{e1}",
                     fp32_ckpts[e0], fp32_ckpts[e1], val_loader, device)
        r["type"] = "intra_fp32"
        lmc_results.append(r)

    # Intra-BF16: consecutive 20-epoch windows
    for i in range(len(epochs) - 1):
        e0, e1 = epochs[i], epochs[i + 1]
        r = lmc_pair(f"bf16_ep{e0}↔bf16_ep{e1}",
                     bf16_ckpts[e0], bf16_ckpts[e1], val_loader, device)
        r["type"] = "intra_bf16"
        lmc_results.append(r)

    # Cross-precision: same epoch, different precision
    for ep in epochs[1:]:
        r = lmc_pair(f"fp32_ep{ep}↔bf16_ep{ep}",
                     fp32_ckpts[ep], bf16_ckpts[ep], val_loader, device)
        r["type"] = "cross_precision"
        r["epoch"] = ep
        lmc_results.append(r)

    # ── Results summary ───────────────────────────────────────────────────────
    log("\n" + "=" * 65)
    log("  LONGITUDINAL LMC RESULTS")
    log("=" * 65)
    log(f"\n  {'Pair':45s}  {'Type':15s}  {'Barrier (eV)':>13}")
    log(f"  {'─'*45}  {'─'*15}  {'─'*13}")
    for r in lmc_results:
        log(f"  {r['label']:45s}  {r['type']:15s}  {r['barrier_ev']:>13.4f}")

    cross = [r for r in lmc_results if r["type"] == "cross_precision"]
    intra_fp32 = [r for r in lmc_results if r["type"] == "intra_fp32"]
    intra_bf16 = [r for r in lmc_results if r["type"] == "intra_bf16"]

    log("\n  Cross-precision barriers by epoch:")
    divergence_ep = None
    for r in cross:
        ep = r["epoch"]
        fp32_intra = next((x["barrier_ev"] for x in intra_fp32
                           if f"ep{ep-20}↔fp32_ep{ep}" in x["label"]), None)
        note = ""
        if fp32_intra is not None and r["barrier_ev"] > 2 * fp32_intra:
            note = " ← DIVERGENCE"
            if divergence_ep is None:
                divergence_ep = ep
        log(f"    ep{ep:2d}: cross={r['barrier_ev']:.4f} eV{note}")

    if divergence_ep:
        log(f"\n  Estimated divergence epoch: {divergence_ep}")
    else:
        log("\n  No clear divergence detected within 80 epochs at 2× threshold")

    results = {
        "experiment":     "phase16_longitudinal_lmc",
        "hidden_dim":     HIDDEN_DIM,
        "n_epochs":       N_EPOCHS,
        "seed":           SEED,
        "save_every":     SAVE_EVERY,
        "fp32_final_mae": fp32_final,
        "bf16_final_mae": bf16_final,
        "lmc_results":    lmc_results,
        "divergence_epoch": divergence_ep,
        "phase7_ref_barrier_ev": 1.447,
    }

    out = OUT_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    gsutil_cp(out, f"{GCS_BASE}/phase16_longitudinal_lmc/results.json")
    log(f"\n  Results → {out}")
    log(f"  GCS     → {GCS_BASE}/phase16_longitudinal_lmc/results.json")
    notify("phase16_complete", "Phase 16 complete — longitudinal LMC done", data=results)


if __name__ == "__main__":
    main()
