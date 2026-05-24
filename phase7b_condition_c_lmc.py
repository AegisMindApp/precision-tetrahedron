#!/usr/bin/env python3
"""
phase7b_condition_c_lmc.py
--------------------------
LMC comparison for BF16-512 (Condition C).

Condition C (hidden_dim=512, BF16) hit its plateau at ep40.  We have the
ep40 checkpoint from the c_restart run.  This script:

  1. Downloads condition_C_epoch40.pt  (pre-restart plateau)
  2. Runs 5 BF16 restart epochs (ep41-ep45), saving every epoch
  3. LMC primary:   ep40 <-> ep43  (3-epoch, matches BF16-256 comparison)
  4. LMC secondary: ep40 <-> ep45  (5-epoch, full restart)
  5. Prints a comparison table vs BF16-256 barrier (1.447 eV) and FP32-256
  6. Uploads JSON to gs://.../phase7b_condition_c_lmc/results.json

Hypothesis: BF16-512 barrier > BF16-256 barrier > FP32-256 barrier
  (larger hidden dim + reduced precision -> sharper minima)

GCS input:  gs://aegismind-tpu-results/aegis_flashoptim/c_restart/condition_C_epoch40.pt
GCS output: gs://aegismind-tpu-results/aegis_flashoptim/phase7b_condition_c_lmc/
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
        import torch_xla.experimental as _xla_exp
        _xla_exp.eager_mode(True)
        print("XLA eager mode: ENABLED", flush=True)
    except Exception as _e:
        print(f"XLA eager mode unavailable: {_e}", flush=True)
except ImportError:
    XLA_AVAILABLE = False

# ── Local imports (TPU VM) ───────────────────────────────────────────────────
sys.path.insert(0, os.path.expanduser("~/flashoptim"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_model
from data import get_dataloaders, batch_to_graph
from notify import notify, heartbeat

# ── Config ────────────────────────────────────────────────────────────────────
GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID     = os.environ.get("RUN_ID",     "aegis_flashoptim")
GCS_BASE   = f"{GCS_BUCKET}/{RUN_ID}"

OUT_DIR    = Path("/tmp/phase7b_condition_c_lmc")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR   = Path("/tmp/qm9")

CONDITION        = "C"          # BF16, hidden_dim=512
RESTART_LR       = 5e-5
N_RESTART_EPOCHS = 5            # ep41..ep45
PLATEAU_EP       = 40           # pre-restart epoch label
N_INTERP_STEPS   = 11           # α ∈ {0.0, 0.1, ..., 1.0}

# Reference barriers from earlier phases
BF16_256_BARRIER_EV  = 1.447   # Phase 4 / paper Section 4.6.2, BF16-256 ep80↔ep83
BF16_256_PEAK_ALPHA  = 0.3


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def gsutil_cp(local: Path, gcs: str):
    subprocess.run(["gsutil", "-q", "cp", str(local), gcs], check=False)


# ── Evaluate (FP32 accumulation, BF16 forward) ───────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, std):
    model.eval()
    mae_sum = 0.0
    n = 0
    for batch in loader:
        z, pos, edge_src, edge_dst, assign_mat, num_graphs, edge_valid, atom_valid = \
            batch_to_graph(batch, device)
        target = batch['target'].to(device)
        pred = model(z, pos, edge_src, edge_dst, assign_mat,
                     num_graphs, edge_valid, atom_valid)
        mae_sum += ((pred.float() - target.float()).abs() * std).sum().item()
        n += num_graphs
        if XLA_AVAILABLE:
            xm.mark_step()
    return mae_sum / max(n, 1)


# ── BF16 training ─────────────────────────────────────────────────────────────

def train_epoch_bf16(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    use_bf16 = True

    if XLA_AVAILABLE:
        ctx = torch.autocast("xla", dtype=torch.bfloat16, enabled=use_bf16)
    else:
        ctx = torch.autocast("cpu", dtype=torch.bfloat16, enabled=use_bf16)

    for batch in loader:
        z, pos, edge_src, edge_dst, assign_mat, num_graphs, edge_valid, atom_valid = \
            batch_to_graph(batch, device)
        target = batch['target'].to(device)
        optimizer.zero_grad()
        with ctx:
            pred = model(z, pos, edge_src, edge_dst, assign_mat,
                         num_graphs, edge_valid, atom_valid)
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


# ── LMC helpers ───────────────────────────────────────────────────────────────

def interpolate_state_dicts(sd0, sd1, alpha):
    return {k: (1 - alpha) * sd0[k].float() + alpha * sd1[k].float()
            for k in sd0}


def run_lmc(sd_pre, sd_post, model, val_loader, device, std, label, n_steps=11):
    """
    Linearly interpolate between sd_pre (α=0) and sd_post (α=1).
    Returns (records, barrier_ev, peak_alpha).
    """
    log(f"\nLMC: {label}  ({n_steps} interpolation points)")
    log(f"  {'α':>6}  {'MAE (eV)':>12}")
    log(f"  {'─'*6}  {'─'*12}")

    alphas  = [i / (n_steps - 1) for i in range(n_steps)]
    records = []
    model.eval()

    for alpha in alphas:
        model.load_state_dict(interpolate_state_dicts(sd_pre, sd_post, alpha))
        mae = evaluate(model, val_loader, device, std)
        records.append({"alpha": round(alpha, 2), "mae_ev": round(mae, 4)})
        log(f"  {alpha:>6.2f}  {mae:>12.4f}")

    maes        = [r["mae_ev"] for r in records]
    endpoints   = (maes[0] + maes[-1]) / 2.0
    barrier     = max(maes) - endpoints
    peak_alpha  = alphas[int(np.argmax(maes))]

    log(f"\n  Endpoint mean MAE : {endpoints:.4f} eV")
    log(f"  Peak MAE          : {max(maes):.4f} eV  at α={peak_alpha:.2f}")
    log(f"  Barrier height    : {barrier:.4f} eV")

    return records, round(barrier, 4), peak_alpha


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 64)
    log("  Phase 7b — Condition C (BF16-512) LMC Comparison")
    log(f"  Warm restart from ep{PLATEAU_EP}, {N_RESTART_EPOCHS} epochs")
    log(f"  Primary LMC:   ep{PLATEAU_EP} <-> ep{PLATEAU_EP + 3}  "
        f"(matches BF16-256 comparison window)")
    log(f"  Secondary LMC: ep{PLATEAU_EP} <-> ep{PLATEAU_EP + 5}")
    log(f"  BF16-256 reference barrier: {BF16_256_BARRIER_EV:.3f} eV  "
        f"at α={BF16_256_PEAK_ALPHA}")
    log("=" * 64)

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

    # ── Download pre-restart checkpoint ───────────────────────────────────────
    ep40_path = OUT_DIR / f"condition_C_epoch{PLATEAU_EP}.pt"
    if not ep40_path.exists():
        log(f"Downloading condition_C_epoch{PLATEAU_EP}.pt from GCS ...")
        subprocess.run(
            ["gsutil", "-q", "cp",
             f"{GCS_BASE}/c_restart/condition_C_epoch{PLATEAU_EP}.pt",
             str(ep40_path)],
            check=True
        )
    log(f"Pre-restart checkpoint: {ep40_path}")

    # ── Data ──────────────────────────────────────────────────────────────────
    log("Loading QM9 data ...")
    train_loader, val_loader, _ = get_dataloaders(
        str(DATA_DIR), batch_size=32, num_workers=4
    )
    std = train_loader.dataset.std
    log(f"QM9 std: {std:.4f} eV")

    # ── Build model and load ep40 ─────────────────────────────────────────────
    model = build_model(CONDITION, device)
    log(f"Model: condition={CONDITION}  "
        f"params={sum(p.numel() for p in model.parameters()):,}")

    ckpt_ep40 = torch.load(ep40_path, map_location="cpu")
    model.load_state_dict(ckpt_ep40["model"])
    sd_ep40 = {k: v.clone().cpu() for k, v in ckpt_ep40["model"].items()}
    log(f"Loaded ep{PLATEAU_EP} weights  "
        f"(checkpoint epoch={ckpt_ep40.get('epoch', '?')})")

    # Baseline evaluation
    mae_ep40 = evaluate(model, val_loader, device, std)
    log(f"ep{PLATEAU_EP} val MAE (BF16-512): {mae_ep40:.4f} eV")

    notify("PHASE_START", "[Phase7b] BF16-512 LMC warm restart begun",
           data={"mae_ep40": mae_ep40, "restart_lr": RESTART_LR,
                 "n_epochs": N_RESTART_EPOCHS})

    # ── BF16 warm restart: ep41..ep45 ─────────────────────────────────────────
    log(f"\nBF16 warm restart from ep{PLATEAU_EP}, lr={RESTART_LR}")
    model.load_state_dict(ckpt_ep40["model"])   # fresh weights from ep40
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=RESTART_LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_RESTART_EPOCHS, eta_min=1e-6
    )

    saved_checkpoints = {PLATEAU_EP: sd_ep40}   # {epoch: state_dict (cpu)}

    log(f"  {'ep':>4}  {'lr':>10}  {'train_loss':>12}  {'val_mae (eV)':>14}")
    log(f"  {'─'*4}  {'─'*10}  {'─'*12}  {'─'*14}")

    val_maes = {PLATEAU_EP: round(mae_ep40, 4)}

    for step in range(1, N_RESTART_EPOCHS + 1):
        ep  = PLATEAU_EP + step
        lr  = optimizer.param_groups[0]["lr"]
        train_loss = train_epoch_bf16(model, train_loader, optimizer, device)
        val_mae    = evaluate(model, val_loader, device, std)
        scheduler.step()

        log(f"  ep{ep:>2d}  {lr:>10.2e}  {train_loss:>12.4f}  {val_mae:>14.4f}")

        # Save state dict for LMC
        saved_checkpoints[ep] = {k: v.clone().cpu()
                                  for k, v in model.state_dict().items()}
        val_maes[ep] = round(val_mae, 4)

        # Persist checkpoint
        ckpt_path = OUT_DIR / f"condition_C_restart_ep{ep}.pt"
        torch.save({"epoch": ep, "model": model.state_dict(),
                    "val_mae_ev": val_mae}, ckpt_path)

        heartbeat("Phase7b_C_restart", step,
                  {"ep": ep, "val_mae": val_mae})

    log(f"\nBF16-512 warm restart complete.\n")

    # ── Build a fresh model for LMC (avoids in-place state contamination) ─────
    model_lmc = build_model(CONDITION, device)

    # ── Primary LMC: ep40 <-> ep43 ────────────────────────────────────────────
    ep_post_primary = PLATEAU_EP + 3   # ep43
    records_43, barrier_43, peak_alpha_43 = run_lmc(
        saved_checkpoints[PLATEAU_EP],
        saved_checkpoints[ep_post_primary],
        model_lmc, val_loader, device, std,
        label=f"BF16-512 ep{PLATEAU_EP} <-> ep{ep_post_primary}  "
              f"(primary — matches BF16-256 window)",
        n_steps=N_INTERP_STEPS,
    )

    # ── Secondary LMC: ep40 <-> ep45 ──────────────────────────────────────────
    ep_post_secondary = PLATEAU_EP + 5  # ep45
    records_45, barrier_45, peak_alpha_45 = run_lmc(
        saved_checkpoints[PLATEAU_EP],
        saved_checkpoints[ep_post_secondary],
        model_lmc, val_loader, device, std,
        label=f"BF16-512 ep{PLATEAU_EP} <-> ep{ep_post_secondary}  "
              f"(secondary — full 5-epoch restart)",
        n_steps=N_INTERP_STEPS,
    )

    # ── Comparison table ──────────────────────────────────────────────────────
    log("\n" + "=" * 64)
    log("  PHASE 7b SUMMARY: BF16-512 vs BF16-256 vs FP32-256 LMC Barriers")
    log("=" * 64)
    log(f"  {'Condition':<24}  {'Comparison':<16}  {'Barrier (eV)':>14}  {'Peak α':>8}")
    log(f"  {'─'*24}  {'─'*16}  {'─'*14}  {'─'*8}")
    log(f"  {'BF16-512 (C, primary)':<24}  "
        f"{'ep40 <-> ep43':<16}  {barrier_43:>14.4f}  {peak_alpha_43:>8.2f}")
    log(f"  {'BF16-512 (C, secondary)':<24}  "
        f"{'ep40 <-> ep45':<16}  {barrier_45:>14.4f}  {peak_alpha_45:>8.2f}")
    log(f"  {'BF16-256 (B, reference)':<24}  "
        f"{'ep80 <-> ep83':<16}  {BF16_256_BARRIER_EV:>14.4f}  "
        f"{BF16_256_PEAK_ALPHA:>8.2f}")
    log(f"  {'FP32-256 (A)':<24}  "
        f"{'ep80 <-> ep83':<16}  {'null (Phase 7)':>14}  {'TBD':>8}")
    log("=" * 64)

    # Hypothesis check (primary)
    if barrier_43 > BF16_256_BARRIER_EV:
        log(f"  -> BF16-512 barrier ({barrier_43:.4f} eV) HIGHER than BF16-256 "
            f"({BF16_256_BARRIER_EV:.4f} eV)")
        log("  -> Supports hypothesis: larger hidden_dim amplifies BF16 sharpening")
    else:
        log(f"  -> BF16-512 barrier ({barrier_43:.4f} eV) NOT higher than BF16-256 "
            f"({BF16_256_BARRIER_EV:.4f} eV)")
        log("  -> Does not support sharpening-by-width hypothesis")
    log("=" * 64)

    # ── Build summary JSON ────────────────────────────────────────────────────
    summary = {
        "experiment": "phase7b_condition_c_lmc",
        "condition": CONDITION,
        "precision": "bf16",
        "hidden_dim": 512,
        "plateau_epoch": PLATEAU_EP,
        "restart_lr": RESTART_LR,
        "n_restart_epochs": N_RESTART_EPOCHS,
        "mae_ep40_ev": round(mae_ep40, 4),
        "val_maes_per_epoch": val_maes,
        "bf16_512_ep40_ep43_barrier_ev": barrier_43,
        "bf16_512_ep40_ep43_peak_alpha": peak_alpha_43,
        "bf16_512_ep40_ep43_interpolation": records_43,
        "bf16_512_ep40_ep45_barrier_ev": barrier_45,
        "bf16_512_ep40_ep45_peak_alpha": peak_alpha_45,
        "bf16_512_ep40_ep45_interpolation": records_45,
        "bf16_256_ep80_ep83_barrier_ev": BF16_256_BARRIER_EV,
        "fp32_256_ep80_ep83_barrier_ev": None,   # to be filled from Phase 7
        "hypothesis": (
            "BF16-512 barrier > BF16-256 barrier > FP32-256 barrier "
            "if precision sharpens minima"
        ),
        "primary_hypothesis_supported": barrier_43 > BF16_256_BARRIER_EV,
    }

    out_json = OUT_DIR / "results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    log(f"\nResults saved -> {out_json}")
    gsutil_cp(out_json, f"{GCS_BASE}/phase7b_condition_c_lmc/results.json")

    # Upload restart checkpoints
    for step in range(1, N_RESTART_EPOCHS + 1):
        ep = PLATEAU_EP + step
        p  = OUT_DIR / f"condition_C_restart_ep{ep}.pt"
        if p.exists():
            gsutil_cp(p,
                f"{GCS_BASE}/phase7b_condition_c_lmc/"
                f"condition_C_restart_ep{ep}.pt")

    log(f"GCS: {GCS_BASE}/phase7b_condition_c_lmc/")

    notify("PHASE_COMPLETE", "[Phase7b] BF16-512 LMC complete",
           data={"barrier_43": barrier_43, "barrier_45": barrier_45,
                 "bf16_256_ref": BF16_256_BARRIER_EV,
                 "hypothesis_supported": barrier_43 > BF16_256_BARRIER_EV})


if __name__ == "__main__":
    main()
