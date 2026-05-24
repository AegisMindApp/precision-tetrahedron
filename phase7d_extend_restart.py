#!/usr/bin/env python3
"""
phase7d_extend_restart.py
--------------------------
Extend condition A (FP32-256) warm restart from ep100 to ep200.

The a_explicit_restart run stopped at absolute ep100 (ep80 + 20 restart steps).
This script adds 100 more FP32 epochs to reach ep200, using a lower restart LR
(1e-5 vs original 5e-5) since we are already well past the basin crossing.

Steps
-----
1. Download gs://.../a_explicit_restart/condition_A_restart_ep100.pt
2. Build condition A model (FP32, hidden_dim=256)
3. Load weights, run 100 FP32 epochs with lr=1e-5 cosine to eta_min=1e-7
4. Save every 10 epochs → upload to gs://.../phase7d_extend_restart/
5. Report final val MAE trajectory + upload results.json

GCS input:  gs://aegismind-tpu-results/aegis_flashoptim/a_explicit_restart/
                condition_A_restart_ep100.pt
GCS output: gs://aegismind-tpu-results/aegis_flashoptim/phase7d_extend_restart/
"""

import os, sys, json, time, subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── XLA ──────────────────────────────────────────────────────────────────────
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
from model import build_model
from data import get_dataloaders, batch_to_graph
from notify import notify, heartbeat

# ── Config ────────────────────────────────────────────────────────────────────
GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID     = os.environ.get("RUN_ID",     "aegis_flashoptim")
GCS_BASE   = f"{GCS_BUCKET}/{RUN_ID}"

OUT_DIR  = Path("/tmp/phase7d_extend_restart")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("/tmp/qm9")

CONDITION        = "A"    # FP32, hidden_dim=256
START_EP         = 100    # absolute epoch label of the input checkpoint
N_EXTEND_EPOCHS  = 100    # run ep101 … ep200
SAVE_EVERY       = 10     # save checkpoint every N epochs
RESTART_LR       = 1e-5   # lower than original 5e-5 — we are past basin crossing
ETA_MIN          = 1e-7


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def gsutil_cp(local: Path, gcs: str):
    """Upload a single file to GCS (silent, non-fatal on error)."""
    result = subprocess.run(
        ["gsutil", "-q", "cp", str(local), gcs],
        capture_output=True
    )
    if result.returncode != 0:
        log(f"  WARNING: gsutil cp failed for {local.name} → {gcs}")


def get_device():
    if XLA_AVAILABLE:
        return xm.xla_device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Evaluate (FP32 accumulation) ──────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, std):
    model.eval()
    mae_sum, n = 0.0, 0
    for batch in loader:
        z, pos, es, ed, am, ng, ev, av = batch_to_graph(batch, device)
        pred = model(z, pos, es, ed, am, ng, ev, av)
        mae_sum += (
            (pred.float() - batch['target'].to(device).float()).abs() * std
        ).sum().item()
        n += ng
        if XLA_AVAILABLE:
            xm.mark_step()
    return mae_sum / max(n, 1)


# ── FP32 training epoch ───────────────────────────────────────────────────────

def train_epoch_fp32(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        z, pos, es, ed, am, ng, ev, av = batch_to_graph(batch, device)
        target = batch['target'].to(device)
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 68)
    log("  Phase 7d — Condition A (FP32-256) Warm Restart Extension")
    log(f"  Input : ep{START_EP} checkpoint (a_explicit_restart)")
    log(f"  Output: ep{START_EP + 1} … ep{START_EP + N_EXTEND_EPOCHS}")
    log(f"  LR    : {RESTART_LR}  (cosine, eta_min={ETA_MIN})")
    log(f"  Save  : every {SAVE_EVERY} epochs")
    log("=" * 68)

    device = get_device()
    log(f"Device: {device}" + (" (TPU)" if XLA_AVAILABLE else ""))

    # ── 1. Download ep100 checkpoint ──────────────────────────────────────────
    ep100_path = OUT_DIR / f"condition_A_restart_ep{START_EP}.pt"
    if not ep100_path.exists():
        log(f"Downloading condition_A_restart_ep{START_EP}.pt from GCS ...")
        subprocess.run(
            ["gsutil", "-q", "cp",
             f"{GCS_BASE}/a_explicit_restart/condition_A_restart_ep{START_EP}.pt",
             str(ep100_path)],
            check=True
        )
    log(f"Checkpoint: {ep100_path}  ({ep100_path.stat().st_size / 1e6:.1f} MB)")

    # ── 2. Data ───────────────────────────────────────────────────────────────
    log("Loading QM9 data ...")
    train_loader, val_loader, _ = get_dataloaders(
        str(DATA_DIR), batch_size=32, num_workers=4
    )
    std = train_loader.dataset.std
    log(f"QM9 std: {std:.4f} eV")

    # ── 3. Build model and load ep100 weights ─────────────────────────────────
    model = build_model(CONDITION, device)
    log(f"Model: condition={CONDITION}  "
        f"hidden_dim={model.hidden_dim}  "
        f"params={model.parameter_count():,}")

    ckpt100 = torch.load(ep100_path, map_location="cpu")
    model.load_state_dict(ckpt100["model"])
    log(f"Loaded ep{START_EP} weights "
        f"(checkpoint epoch={ckpt100.get('epoch', '?')})")

    # Baseline eval at ep100
    mae_ep100 = evaluate(model, val_loader, device, std)
    log(f"ep{START_EP} val MAE (FP32): {mae_ep100:.4f} eV")

    notify(
        "PHASE_START",
        "[Phase7d] FP32 extend restart ep100→ep200",
        data={
            "mae_ep100": round(mae_ep100, 4),
            "restart_lr": RESTART_LR,
            "n_epochs": N_EXTEND_EPOCHS,
            "start_ep": START_EP,
        }
    )

    # ── 4. Fresh Adam + cosine scheduler for 100 epochs ──────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=RESTART_LR
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EXTEND_EPOCHS, eta_min=ETA_MIN
    )

    val_maes = {START_EP: round(mae_ep100, 4)}
    best_val_mae  = mae_ep100
    best_epoch    = START_EP
    upload_queue  = []  # (local_path, gcs_path) — uploaded in batches

    log(f"\n{'─'*68}")
    log(f"  {'ep':>5}  {'lr':>10}  {'train_loss':>12}  {'val_mae (eV)':>14}  {'best':>6}")
    log(f"  {'─'*5}  {'─'*10}  {'─'*12}  {'─'*14}  {'─'*6}")

    for step in range(1, N_EXTEND_EPOCHS + 1):
        ep  = START_EP + step
        lr  = optimizer.param_groups[0]["lr"]

        train_loss = train_epoch_fp32(model, train_loader, optimizer, device)
        val_mae    = evaluate(model, val_loader, device, std)
        scheduler.step()

        val_maes[ep] = round(val_mae, 4)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch   = ep
            star = "*"
        else:
            star = ""

        log(f"  ep{ep:>3d}  {lr:>10.2e}  {train_loss:>12.6f}  "
            f"{val_mae:>14.4f}  {star:>6}")

        heartbeat("Phase7d_extend", step,
                  {"ep": ep, "val_mae": round(val_mae, 4),
                   "best_mae": round(best_val_mae, 4)})

        # Save every SAVE_EVERY epochs and always at the last epoch
        if step % SAVE_EVERY == 0 or step == N_EXTEND_EPOCHS:
            ckpt_path = OUT_DIR / f"condition_A_extend_ep{ep}.pt"
            torch.save(
                {
                    "epoch": ep,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "val_mae_ev": val_mae,
                    "best_val_mae_ev": best_val_mae,
                    "best_epoch": best_epoch,
                },
                ckpt_path
            )
            gcs_dest = (
                f"{GCS_BASE}/phase7d_extend_restart/"
                f"condition_A_extend_ep{ep}.pt"
            )
            gsutil_cp(ckpt_path, gcs_dest)
            log(f"    -> saved & uploaded condition_A_extend_ep{ep}.pt")

    log(f"\n{'─'*68}")
    log(f"  Extension complete.")
    log(f"  Best val MAE: {best_val_mae:.4f} eV at ep{best_epoch}")
    log(f"  Final val MAE (ep{START_EP + N_EXTEND_EPOCHS}): "
        f"{val_maes[START_EP + N_EXTEND_EPOCHS]:.4f} eV")
    log(f"  MAE improvement vs ep{START_EP}: "
        f"{mae_ep100 - best_val_mae:.4f} eV")
    log(f"{'─'*68}")

    # ── 5. Build results JSON ─────────────────────────────────────────────────
    # Trajectory list sorted by epoch
    trajectory = [
        {"epoch": ep, "val_mae_ev": mae}
        for ep, mae in sorted(val_maes.items())
    ]

    summary = {
        "experiment": "phase7d_extend_restart",
        "condition": CONDITION,
        "precision": "fp32",
        "hidden_dim": 256,
        "input_checkpoint": f"a_explicit_restart/condition_A_restart_ep{START_EP}.pt",
        "start_ep": START_EP,
        "end_ep": START_EP + N_EXTEND_EPOCHS,
        "n_extend_epochs": N_EXTEND_EPOCHS,
        "restart_lr": RESTART_LR,
        "eta_min": ETA_MIN,
        "scheduler": f"CosineAnnealingLR T_max={N_EXTEND_EPOCHS} eta_min={ETA_MIN}",
        "mae_ep100_ev": round(mae_ep100, 4),
        "best_val_mae_ev": round(best_val_mae, 4),
        "best_epoch": best_epoch,
        "final_val_mae_ev": val_maes[START_EP + N_EXTEND_EPOCHS],
        "mae_improvement_from_ep100": round(mae_ep100 - best_val_mae, 4),
        "val_mae_trajectory": trajectory,
        "save_every": SAVE_EVERY,
        "gcs_output": f"{GCS_BASE}/phase7d_extend_restart/",
    }

    out_json = OUT_DIR / "results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    log(f"\nResults saved -> {out_json}")
    gsutil_cp(out_json, f"{GCS_BASE}/phase7d_extend_restart/results.json")
    log(f"GCS: {GCS_BASE}/phase7d_extend_restart/")

    notify(
        "PHASE_COMPLETE",
        "[Phase7d] FP32 extend restart ep100→ep200 complete",
        data={
            "best_val_mae_ev": round(best_val_mae, 4),
            "best_epoch": best_epoch,
            "final_val_mae_ev": val_maes[START_EP + N_EXTEND_EPOCHS],
            "mae_improvement_ev": round(mae_ep100 - best_val_mae, 4),
        }
    )


if __name__ == "__main__":
    main()
