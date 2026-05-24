#!/usr/bin/env python3
"""
phase7_fp32_lmc.py
------------------
FP32 vs BF16 Linear Mode Connectivity comparison.

The BF16 experiment measured a loss barrier of 1.447 eV at α=0.3 between
the ep80 plateau and ep83 post-restart checkpoint. This script repeats the
same measurement for FP32 (Condition A, hidden_dim=256):

  1. Downloads condition_A_epoch80.pt (FP32 plateau checkpoint)
  2. Runs a fresh warm restart for 5 FP32 epochs, saving every epoch
  3. Interpolates linearly between ep80 and ep83 at 11 α points
  4. Reports peak barrier height for comparison with BF16

If BF16 produces a systematically higher barrier than FP32, this supports
the hypothesis that reduced precision sharpens loss-landscape minima.

GCS output: gs://.../aegis_flashoptim/phase7_fp32_lmc/
"""

import os, sys, json, time, subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
    try:
        import torch_xla.experimental as _xla_exp
        _xla_exp.eager_mode(True)
        print("XLA eager mode: ENABLED")
    except Exception as _e:
        print(f"XLA eager mode unavailable: {_e}")
except ImportError:
    XLA_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_model
from data import get_dataloaders, batch_to_graph
from notify import notify, heartbeat

# ── Config ────────────────────────────────────────────────────────────────────
GCS_BUCKET  = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID      = os.environ.get("RUN_ID", "aegis_flashoptim")
GCS_BASE    = f"{GCS_BUCKET}/{RUN_ID}"

OUT_DIR     = Path("/tmp/phase7_fp32_lmc")
OUT_DIR.mkdir(exist_ok=True)

DATA_DIR    = Path("/tmp/qm9")
RESTART_LR  = 5e-5
N_RESTART_EPOCHS = 5   # ep81..ep85; we compare ep80↔ep83

# BF16 reference (from Phase 4 / paper Section 4.6.2)
BF16_BARRIER_EV = 1.447   # peak MAE at α=0.3 for BF16 ep80↔ep83
BF16_ALPHA_PEAK = 0.3

N_INTERP_STEPS = 11   # α ∈ {0.0, 0.1, ..., 1.0}

CONDITION    = "A"   # FP32 256-dim


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def gsutil_cp(local: Path, gcs: str):
    subprocess.run(["gsutil", "-q", "cp", str(local), gcs], check=False)


# ── Training helpers (FP32 only) ──────────────────────────────────────────────

def train_epoch_fp32(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        z, pos, edge_src, edge_dst, assign_mat, num_graphs, edge_valid, atom_valid = \
            batch_to_graph(batch, device)
        target = batch['target'].to(device)
        optimizer.zero_grad()
        pred = model(z, pos, edge_src, edge_dst, assign_mat, num_graphs, edge_valid, atom_valid)
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


@torch.no_grad()
def evaluate_fp32(model, loader, device, std):
    model.eval()
    mae_sum = 0.0
    n_total = 0
    for batch in loader:
        z, pos, edge_src, edge_dst, assign_mat, num_graphs, edge_valid, atom_valid = \
            batch_to_graph(batch, device)
        target = batch['target'].to(device)
        pred = model(z, pos, edge_src, edge_dst, assign_mat, num_graphs, edge_valid, atom_valid)
        mae_sum += ((pred - target).abs() * std).sum().item()
        n_total += num_graphs
        if XLA_AVAILABLE:
            xm.mark_step()
    return mae_sum / max(n_total, 1)


# ── LMC interpolation ─────────────────────────────────────────────────────────

def interpolate_state_dicts(sd0, sd1, alpha):
    result = {}
    for key in sd0:
        v0, v1 = sd0[key].float(), sd1[key].float()
        result[key] = (1 - alpha) * v0 + alpha * v1
    return result


def run_lmc(sd_pre, sd_post, model_template, val_loader, device, std, label):
    """
    Interpolate between sd_pre (α=0) and sd_post (α=1) at N_INTERP_STEPS points.
    Returns list of {alpha, mae_ev} dicts and the peak (barrier) MAE.
    """
    log(f"\nLMC: {label}  ({N_INTERP_STEPS} interpolation points)")
    log(f"  {'α':>6}  {'MAE (eV)':>12}  {'note'}")
    log(f"  {'─'*6}  {'─'*12}  {'─'*20}")

    alphas = [i / (N_INTERP_STEPS - 1) for i in range(N_INTERP_STEPS)]
    records = []

    for alpha in alphas:
        interp = interpolate_state_dicts(sd_pre, sd_post, alpha)
        model_template.load_state_dict(interp)
        mae = evaluate_fp32(model_template, val_loader, device, std)

        note = ""
        if alpha == 0.0:   note = "← pre-restart (ep80)"
        elif alpha == 1.0: note = "← post-restart"
        peak_ref = BF16_ALPHA_PEAK
        if abs(alpha - peak_ref) < 1e-9:
            note += f"  [BF16 peak was here: {BF16_BARRIER_EV:.3f} eV]"

        log(f"  {alpha:>6.2f}  {mae:>12.4f}  {note}")
        records.append({"alpha": round(alpha, 2), "mae_ev": round(mae, 4)})

    maes = [r["mae_ev"] for r in records]
    endpoints_mae = (maes[0] + maes[-1]) / 2
    barrier = max(maes) - endpoints_mae
    peak_alpha = alphas[int(np.argmax(maes))]

    log(f"\n  Endpoint mean MAE:  {endpoints_mae:.4f} eV")
    log(f"  Peak MAE:           {max(maes):.4f} eV  at α={peak_alpha:.1f}")
    log(f"  Barrier height:     {barrier:.4f} eV")
    log(f"  BF16 barrier:       {BF16_BARRIER_EV:.4f} eV")
    if barrier > BF16_BARRIER_EV:
        log(f"  → FP32 barrier HIGHER than BF16 (unexpected)")
    else:
        log(f"  → FP32 barrier LOWER than BF16 (supports hypothesis: "
            f"BF16 sharpens minima)")

    return records, barrier, peak_alpha


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("  Phase 7 — FP32 vs BF16 LMC Comparison")
    log(f"  Condition A (FP32, hidden_dim=256)")
    log(f"  Warm restart from ep80, {N_RESTART_EPOCHS} epochs → compare ep80↔ep83")
    log(f"  BF16 reference barrier: {BF16_BARRIER_EV:.3f} eV at α={BF16_ALPHA_PEAK}")
    log("=" * 60)

    # ── Download pre-restart checkpoint ───────────────────────────────────────
    ep80_path = OUT_DIR / "condition_A_epoch80.pt"
    if not ep80_path.exists():
        log(f"Downloading condition_A_epoch80.pt from GCS...")
        subprocess.run(
            ["gsutil", "-q", "cp",
             f"{GCS_BASE}/condition_A_epoch80.pt", str(ep80_path)],
            check=True
        )
    log(f"Pre-restart checkpoint: {ep80_path}")

    # ── Device ────────────────────────────────────────────────────────────────
    if XLA_AVAILABLE:
        device = xm.xla_device()
        log(f"Device: {device} (TPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        log("Device: CUDA")
    else:
        device = torch.device("cpu")
        log("Device: CPU")

    # ── Data ──────────────────────────────────────────────────────────────────
    log("Loading QM9 data...")
    train_loader, val_loader, _ = get_dataloaders(
        str(DATA_DIR), batch_size=32, num_workers=4
    )
    std = train_loader.dataset.std
    log(f"QM9 std: {std:.4f} eV")

    # ── Build model and load ep80 ─────────────────────────────────────────────
    model = build_model(CONDITION, device)
    log(f"Model: hidden_dim={model.hidden_dim}  params={model.parameter_count():,}")

    ckpt_ep80 = torch.load(ep80_path, map_location="cpu")
    model.load_state_dict(ckpt_ep80["model"])
    sd_ep80 = {k: v.clone().cpu() for k, v in ckpt_ep80["model"].items()}
    log(f"Loaded ep80 weights (epoch={ckpt_ep80.get('epoch', '?')})")

    # Evaluate ep80 as baseline
    mae_ep80 = evaluate_fp32(model, val_loader, device, std)
    log(f"ep80 val MAE (FP32): {mae_ep80:.4f} eV")

    notify("PHASE_START", "[Phase7] FP32 warm restart begun",
           data={"mae_ep80": mae_ep80, "restart_lr": RESTART_LR,
                 "n_epochs": N_RESTART_EPOCHS})

    # ── FP32 warm restart: ep81..ep85 ─────────────────────────────────────────
    log(f"\nFP32 warm restart from ep80, lr={RESTART_LR}")
    model.load_state_dict(ckpt_ep80["model"])   # fresh weights from ep80
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=RESTART_LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_RESTART_EPOCHS, eta_min=1e-6
    )

    saved_checkpoints = {80: sd_ep80}  # {epoch: state_dict}

    log(f"  {'ep':>4}  {'lr':>10}  {'train_loss':>12}  {'val_mae (eV)':>14}")
    log(f"  {'─'*4}  {'─'*10}  {'─'*12}  {'─'*14}")

    for step in range(1, N_RESTART_EPOCHS + 1):
        ep = 80 + step
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_epoch_fp32(model, train_loader, optimizer, device)
        val_mae = evaluate_fp32(model, val_loader, device, std)
        scheduler.step()

        log(f"  ep{ep:>2d}  {lr:>10.2e}  {train_loss:>12.4f}  {val_mae:>14.4f}")

        # Save every epoch state_dict (for LMC)
        saved_checkpoints[ep] = {k: v.clone().cpu()
                                  for k, v in model.state_dict().items()}

        # Also save to disk
        ckpt_path = OUT_DIR / f"condition_A_restart_ep{ep}.pt"
        torch.save({"epoch": ep, "model": model.state_dict(),
                    "val_mae_ev": val_mae}, ckpt_path)

        heartbeat(f"Phase7_FP32_restart",
                  step, {"ep": ep, "val_mae": val_mae})

    log("\nFP32 restart complete.\n")

    # ── LMC: ep80 ↔ ep83 (primary comparison) ─────────────────────────────────
    model_interp = build_model(CONDITION, device)

    results_ep83 = {}
    if 83 in saved_checkpoints:
        records_83, barrier_83, peak_alpha_83 = run_lmc(
            saved_checkpoints[80], saved_checkpoints[83],
            model_interp, val_loader, device, std,
            label="FP32 ep80 ↔ ep83 (primary: matches BF16 comparison)"
        )
        results_ep83 = {
            "label": "fp32_ep80_ep83",
            "comparison": "matches BF16 LMC (ep80 pre-restart, ep83 post-restart)",
            "interpolation": records_83,
            "barrier_ev": round(barrier_83, 4),
            "peak_alpha": peak_alpha_83,
            "bf16_barrier_ev": BF16_BARRIER_EV,
            "bf16_peak_alpha": BF16_ALPHA_PEAK,
            "delta_ev": round(BF16_BARRIER_EV - barrier_83, 4),
            "hypothesis_supported": BF16_BARRIER_EV > barrier_83,
        }
    else:
        log("WARNING: ep83 not in saved checkpoints — unexpected")

    # ── LMC: ep80 ↔ ep85 (full 5-epoch restart context) ──────────────────────
    records_85, barrier_85, peak_alpha_85 = run_lmc(
        saved_checkpoints[80], saved_checkpoints[85],
        model_interp, val_loader, device, std,
        label="FP32 ep80 ↔ ep85 (5 epochs post-restart)"
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("  PHASE 7 SUMMARY: FP32 vs BF16 LMC")
    log("=" * 60)
    log(f"  FP32 ep80↔ep83 barrier:  {results_ep83.get('barrier_ev', '?'):.4f} eV  "
        f"(peak α={results_ep83.get('peak_alpha', '?')})")
    log(f"  BF16 ep80↔ep83 barrier:  {BF16_BARRIER_EV:.4f} eV  "
        f"(peak α={BF16_ALPHA_PEAK})")
    if results_ep83:
        delta = results_ep83["delta_ev"]
        if results_ep83["hypothesis_supported"]:
            log(f"  → BF16 barrier is {delta:.4f} eV HIGHER than FP32  ✓")
            log(f"  → Supports hypothesis: BF16 training sharpens loss-landscape minima")
        else:
            log(f"  → FP32 barrier is {abs(delta):.4f} eV HIGHER than BF16  ✗")
            log(f"  → Does NOT support sharpening hypothesis")
    log("=" * 60)

    summary = {
        "experiment": "phase7_fp32_lmc",
        "condition": CONDITION,
        "precision": "fp32",
        "mae_ep80_ev": round(mae_ep80, 4),
        "restart_lr": RESTART_LR,
        "bf16_barrier_ref_ev": BF16_BARRIER_EV,
        "bf16_peak_alpha_ref": BF16_ALPHA_PEAK,
        "fp32_ep80_ep83": results_ep83,
        "fp32_ep80_ep85": {
            "label": "fp32_ep80_ep85",
            "interpolation": records_85,
            "barrier_ev": round(barrier_85, 4),
            "peak_alpha": peak_alpha_85,
        },
    }

    out_json = OUT_DIR / "phase7_fp32_lmc_results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    gsutil_cp(out_json, f"{GCS_BASE}/phase7_fp32_lmc/results.json")

    # Upload epoch checkpoints
    for ep in range(80, 86):
        p = OUT_DIR / f"condition_A_restart_ep{ep}.pt"
        if p.exists():
            gsutil_cp(p, f"{GCS_BASE}/phase7_fp32_lmc/condition_A_restart_ep{ep}.pt")

    log(f"\nResults saved → {out_json}")
    log(f"GCS: {GCS_BASE}/phase7_fp32_lmc/")

    notify("PHASE_COMPLETE", "[Phase7] FP32 LMC complete",
           data={"fp32_barrier_ev": results_ep83.get("barrier_ev"),
                 "bf16_barrier_ev": BF16_BARRIER_EV,
                 "hypothesis_supported": results_ep83.get("hypothesis_supported")})


if __name__ == "__main__":
    main()
