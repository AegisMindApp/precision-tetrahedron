#!/usr/bin/env python3
"""
phase11_gradient_noise.py
-------------------------
Track Gradient Noise Scale (GNS) during training for FP32 vs BF16.

GNS = ||E[g]||² / Var[g]   (McCandlish et al. 2018)
A lower GNS means noisier gradients. If BF16 lowers GNS, precision loss
adds gradient noise that could drive sharpening of loss-landscape minima.

Pipeline
--------
1.  Train MolecularGNN (hidden_dim=256) in FP32 for 80 epochs on QM9
    Every 5 epochs: compute GNS from N_GNS_BATCHES=20 individual mini-batches
2.  Train a second MolecularGNN (same init seed) in BF16 for 80 epochs
    Every 5 epochs: compute GNS
3.  Report side-by-side GNS trajectories, gradient norms at ep80
4.  Upload results.json to gs://.../phase11_gradient_noise/

GCS output: gs://aegismind-tpu-results/aegis_flashoptim/phase11_gradient_noise/
"""

import os, sys, json, time, subprocess, contextlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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

# ── Local imports (TPU VM at ~/flashoptim/) ───────────────────────────────────
sys.path.insert(0, os.path.expanduser("~/flashoptim"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MolecularGNN
from data import get_dataloaders, batch_to_graph
from notify import notify, heartbeat

# ── Boilerplate ───────────────────────────────────────────────────────────────
GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID     = os.environ.get("RUN_ID",     "aegis_flashoptim")
GCS_BASE   = f"{GCS_BUCKET}/{RUN_ID}"

OUT_DIR  = Path("/tmp/phase11_gradient_noise")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("/tmp/qm9")


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def gsutil_cp(local, gcs):
    subprocess.run(["gsutil", "-q", "cp", str(local), gcs], check=False)


def get_device():
    if XLA_AVAILABLE:
        return xm.xla_device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Config ────────────────────────────────────────────────────────────────────
HIDDEN_DIM     = 256
N_BLOCKS       = 6
N_GAUSSIANS    = 50
NUM_ATOM_TYPES = 9
CUTOFF         = 5.0
BATCH_SIZE     = 32
N_EPOCHS       = 80
LR_INIT        = 1e-4
WEIGHT_DECAY   = 1e-4
EVAL_EVERY     = 5
GNS_EVERY      = 5       # compute GNS every N epochs
N_GNS_BATCHES  = 20      # individual mini-batches for GNS estimation
INIT_SEED      = 42      # same weight init for both precisions

# Reference barriers
BF16_BARRIER_EV = 1.447
FP32_BARRIER_EV = None   # filled by phase7 — leave as null in output


# ── GNS computation ───────────────────────────────────────────────────────────

def compute_gns(model, loader, device, n_batches=20, use_bf16=False):
    """
    Compute Gradient Noise Scale (McCandlish et al. 2018).
    GNS = ||E[g]||² / Var[g]

    Each of n_batches mini-batches generates an independent gradient vector.
    Returns (gns, grad_norm_of_mean, grad_std).
    """
    model.train()
    grads       = []
    loader_iter = iter(loader)

    for _ in range(n_batches):
        try:
            batch = next(loader_iter)
        except StopIteration:
            break

        z, pos, es, ed, am, ng, ev, av = batch_to_graph(batch, device)
        target = batch["target"].to(device)
        model.zero_grad()

        if use_bf16:
            ctx = (torch.autocast("xla", dtype=torch.bfloat16)
                   if XLA_AVAILABLE
                   else torch.autocast("cpu", dtype=torch.bfloat16))
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            pred = model(z, pos, es, ed, am, ng, ev, av)
            loss = F.mse_loss(pred.float(), target.float())

        loss.backward()

        if XLA_AVAILABLE:
            xm.mark_step()

        # Collect FP32 flat gradient vector
        g = torch.cat([
            p.grad.float().flatten()
            for p in model.parameters()
            if p.grad is not None
        ])
        grads.append(g.detach().cpu())

    if len(grads) == 0:
        return 0.0, 0.0, 0.0

    grads = torch.stack(grads)                      # [n_batches, n_params]
    E_g   = grads.mean(0)                           # [n_params]
    Var_g = ((grads - E_g.unsqueeze(0)) ** 2).mean()
    GNS   = (E_g ** 2).mean() / (Var_g + 1e-10)

    return (
        float(GNS.item()),
        float(E_g.norm().item()),
        float(Var_g.sqrt().item()),
    )


# ── Training helpers ──────────────────────────────────────────────────────────

def train_epoch_fp32(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        z, pos, es, ed, am, ng, ev, av = batch_to_graph(batch, device)
        target = batch["target"].to(device)
        optimizer.zero_grad()
        pred = model(z, pos, es, ed, am, ng, ev, av)
        loss = F.mse_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if XLA_AVAILABLE:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def train_epoch_bf16(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        z, pos, es, ed, am, ng, ev, av = batch_to_graph(batch, device)
        target = batch["target"].to(device)
        optimizer.zero_grad()
        ctx = (torch.autocast("xla", dtype=torch.bfloat16)
               if XLA_AVAILABLE
               else torch.autocast("cpu", dtype=torch.bfloat16))
        with ctx:
            pred = model(z, pos, es, ed, am, ng, ev, av)
            loss = F.mse_loss(pred.float(), target.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if XLA_AVAILABLE:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, std, use_bf16=False):
    model.eval()
    mae_sum, n = 0.0, 0
    for batch in loader:
        z, pos, es, ed, am, ng, ev, av = batch_to_graph(batch, device)
        target = batch["target"].to(device)
        if use_bf16 and XLA_AVAILABLE:
            with torch.autocast("xla", dtype=torch.bfloat16):
                pred = model(z, pos, es, ed, am, ng, ev, av)
        else:
            pred = model(z, pos, es, ed, am, ng, ev, av)
        mae_sum += ((pred.float() - target.float()).abs() * std).sum().item()
        n += ng
        if XLA_AVAILABLE:
            xm.mark_step()
    return mae_sum / max(n, 1)


def build_model(device):
    """Build MolecularGNN with fixed seed."""
    torch.manual_seed(INIT_SEED)
    model = MolecularGNN(
        num_atom_types=NUM_ATOM_TYPES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=N_BLOCKS,
        num_gaussians=N_GAUSSIANS,
        cutoff=CUTOFF,
        num_targets=1,
    ).to(device)
    return model


# ── Per-precision training loop ───────────────────────────────────────────────

def train_with_gns(precision, device, train_loader, val_loader, std):
    """
    Train for N_EPOCHS, recording GNS every GNS_EVERY epochs.
    precision: 'fp32' or 'bf16'
    Returns list of per-epoch records.
    """
    use_bf16   = (precision == "bf16")
    train_fn   = train_epoch_bf16 if use_bf16 else train_epoch_fp32

    model      = build_model(device)
    log(f"  Model ({precision}): params={model.parameter_count():,}")

    optimizer  = torch.optim.AdamW(
        model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY
    )
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-6
    )

    records    = []
    ep0_mae    = evaluate(model, val_loader, device, std, use_bf16=use_bf16)
    log(f"  ep  0  val_mae={ep0_mae:.4f} eV  (initial)")

    log(f"\n  {'ep':>5}  {'lr':>10}  {'train_loss':>12}  "
        f"{'val_mae (eV)':>14}  {'gns':>12}  {'grad_norm':>10}  {'grad_std':>10}")
    log(f"  {'─'*5}  {'─'*10}  {'─'*12}  {'─'*14}  "
        f"{'─'*12}  {'─'*10}  {'─'*10}")

    for ep in range(1, N_EPOCHS + 1):
        lr         = optimizer.param_groups[0]["lr"]
        train_loss = train_fn(model, train_loader, optimizer, device)
        scheduler.step()

        val_mae  = None
        gns_val  = None
        g_norm   = None
        g_std    = None

        if ep % EVAL_EVERY == 0 or ep == N_EPOCHS:
            val_mae = evaluate(model, val_loader, device, std, use_bf16=use_bf16)

        if ep % GNS_EVERY == 0 or ep == N_EPOCHS:
            gns_val, g_norm, g_std = compute_gns(
                model, train_loader, device,
                n_batches=N_GNS_BATCHES, use_bf16=use_bf16
            )

        if val_mae is not None or gns_val is not None:
            rec = {
                "epoch":      ep,
                "precision":  precision,
                "train_loss": round(train_loss, 6),
                "val_mae_ev": round(val_mae, 4) if val_mae is not None else None,
                "gns":        round(gns_val, 6) if gns_val is not None else None,
                "grad_norm":  round(g_norm, 6)  if g_norm  is not None else None,
                "grad_std":   round(g_std, 6)   if g_std   is not None else None,
            }
            records.append(rec)

            val_str  = f"{val_mae:>14.4f}" if val_mae is not None else " " * 14
            gns_str  = f"{gns_val:>12.4e}" if gns_val is not None else " " * 12
            gnorm_str = f"{g_norm:>10.4f}" if g_norm  is not None else " " * 10
            gstd_str  = f"{g_std:>10.4f}"  if g_std   is not None else " " * 10
            log(f"  ep{ep:>3d}  {lr:>10.2e}  {train_loss:>12.6f}  "
                f"{val_str}  {gns_str}  {gnorm_str}  {gstd_str}")

            heartbeat(f"Phase11_{precision}", ep,
                      {"ep": ep, "gns": gns_val, "val_mae": val_mae})
        else:
            log(f"  ep{ep:>3d}  {lr:>10.2e}  {train_loss:>12.6f}")

    # Save final checkpoint
    ckpt_path = OUT_DIR / f"{precision}_ep{N_EPOCHS}.pt"
    torch.save({"epoch": N_EPOCHS, "model": model.state_dict()}, ckpt_path)
    gsutil_cp(ckpt_path,
              f"{GCS_BASE}/phase11_gradient_noise/{precision}_ep{N_EPOCHS}.pt")

    return records, model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 68)
    log("  Phase 11 — Gradient Noise Scale: FP32 vs BF16 on QM9")
    log(f"  hidden_dim={HIDDEN_DIM}, n_blocks={N_BLOCKS}, {N_EPOCHS} epochs each")
    log(f"  GNS measured every {GNS_EVERY} epochs using {N_GNS_BATCHES} mini-batches")
    log(f"  GNS = ||E[g]||^2 / Var[g]  (McCandlish et al. 2018)")
    log("=" * 68)

    device = get_device()
    log(f"Device: {device}" + (" (TPU)" if XLA_AVAILABLE else ""))

    log("Loading QM9 data ...")
    train_loader, val_loader, _ = get_dataloaders(
        str(DATA_DIR), batch_size=BATCH_SIZE, num_workers=4
    )
    std = train_loader.dataset.std
    log(f"QM9 std: {std:.4f} eV")

    notify("PHASE_START", "[Phase11] GNS FP32 vs BF16 training begun",
           data={"hidden_dim": HIDDEN_DIM, "n_epochs": N_EPOCHS,
                 "gns_batches": N_GNS_BATCHES})

    # ── FP32 run ──────────────────────────────────────────────────────────────
    log(f"\n{'─'*68}")
    log("  RUN 1: FP32 training")
    log(f"{'─'*68}")
    fp32_records, model_fp32 = train_with_gns(
        "fp32", device, train_loader, val_loader, std
    )

    # Final GNS at ep80 (re-measure on trained model for clarity)
    gns_fp32_final, gnorm_fp32_final, gstd_fp32_final = compute_gns(
        model_fp32, train_loader, device,
        n_batches=N_GNS_BATCHES, use_bf16=False
    )
    log(f"\n  FP32 ep{N_EPOCHS} GNS={gns_fp32_final:.4e}  "
        f"grad_norm={gnorm_fp32_final:.4f}  grad_std={gstd_fp32_final:.4f}")

    # ── BF16 run ──────────────────────────────────────────────────────────────
    log(f"\n{'─'*68}")
    log("  RUN 2: BF16 training")
    log(f"{'─'*68}")
    bf16_records, model_bf16 = train_with_gns(
        "bf16", device, train_loader, val_loader, std
    )

    gns_bf16_final, gnorm_bf16_final, gstd_bf16_final = compute_gns(
        model_bf16, train_loader, device,
        n_batches=N_GNS_BATCHES, use_bf16=True
    )
    log(f"\n  BF16 ep{N_EPOCHS} GNS={gns_bf16_final:.4e}  "
        f"grad_norm={gnorm_bf16_final:.4f}  grad_std={gstd_bf16_final:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log(f"\n{'='*68}")
    log("  PHASE 11 SUMMARY: Gradient Noise Scale FP32 vs BF16")
    log(f"\n  At ep{N_EPOCHS} (plateau region):")
    log(f"  {'Precision':<10}  {'GNS':>12}  {'grad_norm':>10}  {'grad_std':>10}")
    log(f"  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*10}")
    log(f"  {'FP32':<10}  {gns_fp32_final:>12.4e}  "
        f"{gnorm_fp32_final:>10.4f}  {gstd_fp32_final:>10.4f}")
    log(f"  {'BF16':<10}  {gns_bf16_final:>12.4e}  "
        f"{gnorm_bf16_final:>10.4f}  {gstd_bf16_final:>10.4f}")

    gns_ratio = gns_bf16_final / (gns_fp32_final + 1e-30)
    log(f"\n  GNS ratio (BF16/FP32): {gns_ratio:.4f}")
    if gns_ratio < 1.0:
        log("  -> BF16 has LOWER GNS (noisier gradients) — consistent with "
            "sharpening hypothesis: noise drives exploration of sharper minima")
    elif gns_ratio > 1.0:
        log("  -> BF16 has HIGHER GNS (cleaner gradients) — "
            "precision reduction concentrates gradient signal")
    else:
        log("  -> GNS approximately equal across precisions")

    # GNS trajectory comparison
    fp32_gns_traj = [(r["epoch"], r["gns"]) for r in fp32_records
                     if r["gns"] is not None]
    bf16_gns_traj = [(r["epoch"], r["gns"]) for r in bf16_records
                     if r["gns"] is not None]

    log(f"\n  GNS trajectory (epoch, fp32, bf16):")
    fp32_by_ep = {ep: g for ep, g in fp32_gns_traj}
    bf16_by_ep = {ep: g for ep, g in bf16_gns_traj}
    all_eps    = sorted(set(list(fp32_by_ep) + list(bf16_by_ep)))
    log(f"  {'ep':>5}  {'fp32_gns':>12}  {'bf16_gns':>12}  {'ratio':>10}")
    log(f"  {'─'*5}  {'─'*12}  {'─'*12}  {'─'*10}")
    for ep in all_eps:
        fg = fp32_by_ep.get(ep)
        bg = bf16_by_ep.get(ep)
        ratio_str = f"{bg/fg:>10.4f}" if (fg and bg and fg > 0) else " " * 10
        fp_str    = f"{fg:>12.4e}" if fg is not None else " " * 12
        bf_str    = f"{bg:>12.4e}" if bg is not None else " " * 12
        log(f"  {ep:>5}  {fp_str}  {bf_str}  {ratio_str}")
    log(f"{'='*68}")

    # ── Save and upload ───────────────────────────────────────────────────────
    fp32_final_mae = next(
        (r["val_mae_ev"] for r in reversed(fp32_records)
         if r["val_mae_ev"] is not None), None
    )
    bf16_final_mae = next(
        (r["val_mae_ev"] for r in reversed(bf16_records)
         if r["val_mae_ev"] is not None), None
    )

    summary = {
        "experiment":     "phase11_gradient_noise",
        "dataset":        "QM9",
        "architecture":   "MolecularGNN",
        "hidden_dim":     HIDDEN_DIM,
        "n_blocks":       N_BLOCKS,
        "n_epochs":       N_EPOCHS,
        "batch_size":     BATCH_SIZE,
        "lr_init":        LR_INIT,
        "weight_decay":   WEIGHT_DECAY,
        "gns_every_n_ep": GNS_EVERY,
        "n_gns_batches":  N_GNS_BATCHES,
        "init_seed":      INIT_SEED,
        "qm9_std_ev":     round(float(std), 4),
        "fp32": {
            "trajectory":     fp32_records,
            "final_mae_ev":   fp32_final_mae,
            "gns_at_ep80":    round(gns_fp32_final, 6),
            "grad_norm_ep80": round(gnorm_fp32_final, 6),
            "grad_std_ep80":  round(gstd_fp32_final, 6),
        },
        "bf16": {
            "trajectory":     bf16_records,
            "final_mae_ev":   bf16_final_mae,
            "gns_at_ep80":    round(gns_bf16_final, 6),
            "grad_norm_ep80": round(gnorm_bf16_final, 6),
            "grad_std_ep80":  round(gstd_bf16_final, 6),
        },
        "comparison": {
            "gns_ratio_bf16_over_fp32":   round(gns_ratio, 4),
            "bf16_noisier":               gns_ratio < 1.0,
            "bf16_lmc_barrier_ref_ev":    BF16_BARRIER_EV,
            "fp32_lmc_barrier_ref_ev":    FP32_BARRIER_EV,
            "interpretation": (
                "BF16 gradients are noisier (lower GNS), consistent with "
                "noise-driven sharpening"
                if gns_ratio < 1.0
                else "BF16 gradients have higher or equal GNS — "
                     "sharpening driven by other factors"
            ),
        },
        "gcs_output": f"{GCS_BASE}/phase11_gradient_noise/",
    }

    out_json = OUT_DIR / "results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    log(f"\nResults saved -> {out_json}")
    gsutil_cp(out_json, f"{GCS_BASE}/phase11_gradient_noise/results.json")
    log(f"GCS: {GCS_BASE}/phase11_gradient_noise/")

    notify("PHASE_COMPLETE", "[Phase11] GNS FP32 vs BF16 complete",
           data={"gns_fp32_ep80":  round(gns_fp32_final, 4),
                 "gns_bf16_ep80":  round(gns_bf16_final, 4),
                 "gns_ratio":      round(gns_ratio, 4),
                 "bf16_noisier":   gns_ratio < 1.0})


if __name__ == "__main__":
    main()
