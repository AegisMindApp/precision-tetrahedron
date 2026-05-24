#!/usr/bin/env python3
"""
phase10_width_scaling.py
------------------------
Train MolecularGNN with hidden_dim in {64, 128, 256, 512} in BF16 on QM9.
Tests whether the BF16 sharpening result (LMC barrier) scales with model
capacity.

Strategy
--------
  hidden_dim=64   → train 80 epochs from scratch, then 5-epoch warm restart
  hidden_dim=128  → train 80 epochs from scratch, then 5-epoch warm restart
  hidden_dim=256  → download condition_B_epoch80.pt, 5-epoch warm restart
  hidden_dim=512  → download condition_C_epoch40.pt (plateau ~ep40), 5-epoch restart

For each width: run LMC between plateau checkpoint ↔ plateau+3 epoch.

Results table
-------------
  hidden_dim | params | ep80_mae | barrier_ev | peak_alpha

GCS output: gs://aegismind-tpu-results/aegis_flashoptim/phase10_width_scaling/
"""

import os, sys, json, time, subprocess
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

OUT_DIR  = Path("/tmp/phase10_width_scaling")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("/tmp/qm9")


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def gsutil_cp(local, gcs):
    subprocess.run(["gsutil", "-q", "cp", str(local), gcs], check=False)


def gsutil_dl(gcs, local):
    """Download from GCS; raises on failure."""
    result = subprocess.run(
        ["gsutil", "-q", "cp", gcs, str(local)],
        capture_output=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"gsutil cp failed: {gcs} -> {local}\n"
            f"{result.stderr.decode()}"
        )


def get_device():
    if XLA_AVAILABLE:
        return xm.xla_device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Config ────────────────────────────────────────────────────────────────────
WIDTHS         = [64, 128, 256, 512]
N_BLOCKS       = 6
N_GAUSSIANS    = 50
NUM_ATOM_TYPES = 9
CUTOFF         = 5.0
BATCH_SIZE     = 32
N_TRAIN_EPOCHS = 80      # for 64 and 128 from scratch
LR_INIT        = 1e-4
WEIGHT_DECAY   = 1e-4
N_RESTART      = 5       # warm restart epochs for every width
RESTART_LR     = 5e-5
RESTART_ETA_MIN = 1e-6
N_INTERP_STEPS = 11
EVAL_EVERY     = 5

# GCS sources for pre-trained checkpoints
GCS_B_EP80  = f"{GCS_BASE}/condition_B_epoch80.pt"      # hidden_dim=256, BF16
GCS_C_EP40  = f"{GCS_BASE}/condition_C_epoch40.pt"      # hidden_dim=512, BF16, plateau ~ep40

# Map width → (plateau_epoch, gcs_ckpt or None)
# None means train from scratch to N_TRAIN_EPOCHS
WIDTH_CONFIG = {
    64:  {"plateau_ep": N_TRAIN_EPOCHS, "gcs_ckpt": None},
    128: {"plateau_ep": N_TRAIN_EPOCHS, "gcs_ckpt": None},
    256: {"plateau_ep": 80,             "gcs_ckpt": GCS_B_EP80},
    512: {"plateau_ep": 40,             "gcs_ckpt": GCS_C_EP40},
}


# ── Training helpers ──────────────────────────────────────────────────────────

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


# ── LMC helpers ───────────────────────────────────────────────────────────────

def interpolate_sds(sd0, sd1, alpha):
    return {k: (1 - alpha) * sd0[k].float() + alpha * sd1[k].float()
            for k in sd0}


def run_lmc(sd0, sd1, model, loader, device, std, label, n_steps=11):
    log(f"LMC: {label}")
    alphas  = [i / (n_steps - 1) for i in range(n_steps)]
    records = []
    model.eval()
    for alpha in alphas:
        model.load_state_dict(interpolate_sds(sd0, sd1, alpha))
        mae = evaluate(model, loader, device, std)
        records.append({"alpha": round(alpha, 2), "mae_ev": round(mae, 4)})
        log(f"  alpha={alpha:.1f}  mae={mae:.4f} eV")
    maes    = [r["mae_ev"] for r in records]
    barrier = max(maes) - (maes[0] + maes[-1]) / 2
    return records, round(barrier, 4), alphas[int(np.argmax(maes))]


# ── Per-width experiment ──────────────────────────────────────────────────────

def run_width(hd, cfg, device, train_loader, val_loader, std):
    """
    Run one hidden_dim experiment end-to-end.
    Returns dict with all metrics for this width.
    """
    plateau_ep = cfg["plateau_ep"]
    gcs_ckpt   = cfg["gcs_ckpt"]

    log(f"\n{'='*68}")
    log(f"  WIDTH={hd}  (plateau_ep={plateau_ep}  "
        f"{'reuse GCS ckpt' if gcs_ckpt else 'train from scratch'})")
    log(f"{'='*68}")

    model = MolecularGNN(
        num_atom_types=NUM_ATOM_TYPES,
        hidden_dim=hd,
        num_blocks=N_BLOCKS,
        num_gaussians=N_GAUSSIANS,
        cutoff=CUTOFF,
        num_targets=1,
    ).to(device)
    n_params = model.parameter_count()
    log(f"  params: {n_params:,}")

    training_trajectory = []

    # ── A: obtain plateau checkpoint ─────────────────────────────────────────
    if gcs_ckpt is not None:
        local_ckpt = OUT_DIR / f"width{hd}_plateau_ep{plateau_ep}.pt"
        if not local_ckpt.exists():
            log(f"  Downloading {gcs_ckpt} ...")
            gsutil_dl(gcs_ckpt, local_ckpt)
        ckpt  = torch.load(local_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        log(f"  Loaded pre-trained ckpt (epoch={ckpt.get('epoch', '?')})")
        mae_plateau = evaluate(model, val_loader, device, std, use_bf16=True)
        log(f"  Plateau val_mae: {mae_plateau:.4f} eV")
    else:
        # Train from scratch
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=N_TRAIN_EPOCHS, eta_min=1e-6
        )

        log(f"  Training {N_TRAIN_EPOCHS} epochs BF16 ...")
        log(f"  {'ep':>5}  {'lr':>10}  {'train_loss':>12}  {'val_mae (eV)':>14}")
        log(f"  {'─'*5}  {'─'*10}  {'─'*12}  {'─'*14}")

        for ep in range(1, N_TRAIN_EPOCHS + 1):
            lr         = optimizer.param_groups[0]["lr"]
            train_loss = train_epoch_bf16(model, train_loader, optimizer, device)
            scheduler.step()

            if ep % EVAL_EVERY == 0 or ep == N_TRAIN_EPOCHS:
                val_mae = evaluate(model, val_loader, device, std, use_bf16=True)
                training_trajectory.append(
                    {"epoch": ep, "val_mae_ev": round(val_mae, 4)}
                )
                log(f"  ep{ep:>3d}  {lr:>10.2e}  {train_loss:>12.6f}  "
                    f"{val_mae:>14.4f}")
                heartbeat("Phase10_train",
                          ep, {"hd": hd, "ep": ep, "val_mae": round(val_mae, 4)})
            else:
                log(f"  ep{ep:>3d}  {lr:>10.2e}  {train_loss:>12.6f}")

        # Save ep80 checkpoint
        local_ckpt = OUT_DIR / f"width{hd}_plateau_ep{plateau_ep}.pt"
        torch.save({"epoch": plateau_ep, "model": model.state_dict(),
                    "val_mae_ev": val_mae}, local_ckpt)
        gsutil_cp(local_ckpt,
                  f"{GCS_BASE}/phase10_width_scaling/width{hd}_ep{plateau_ep}.pt")
        mae_plateau = val_mae

    sd_plateau = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    # ── B: 5-epoch BF16 warm restart ─────────────────────────────────────────
    log(f"\n  Warm restart: lr={RESTART_LR}, {N_RESTART} epochs")
    model.load_state_dict(sd_plateau)
    restart_opt   = torch.optim.AdamW(
        model.parameters(), lr=RESTART_LR, weight_decay=WEIGHT_DECAY
    )
    restart_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        restart_opt, T_max=N_RESTART, eta_min=RESTART_ETA_MIN
    )

    saved_sds     = {plateau_ep: sd_plateau}
    restart_trajectory = [{"epoch": plateau_ep, "val_mae_ev": round(mae_plateau, 4)}]

    log(f"  {'ep':>5}  {'lr':>10}  {'train_loss':>12}  {'val_mae (eV)':>14}")
    log(f"  {'─'*5}  {'─'*10}  {'─'*12}  {'─'*14}")

    for step in range(1, N_RESTART + 1):
        ep         = plateau_ep + step
        lr         = restart_opt.param_groups[0]["lr"]
        train_loss = train_epoch_bf16(model, train_loader, restart_opt, device)
        restart_sched.step()
        val_mae    = evaluate(model, val_loader, device, std, use_bf16=True)

        saved_sds[ep] = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        restart_trajectory.append({"epoch": ep, "val_mae_ev": round(val_mae, 4)})

        ckpt_path = OUT_DIR / f"width{hd}_restart_ep{ep}.pt"
        torch.save({"epoch": ep, "model": model.state_dict(),
                    "val_mae_ev": val_mae}, ckpt_path)
        gsutil_cp(ckpt_path,
                  f"{GCS_BASE}/phase10_width_scaling/width{hd}_restart_ep{ep}.pt")

        log(f"  ep{ep:>3d}  {lr:>10.2e}  {train_loss:>12.6f}  {val_mae:>14.4f}")
        heartbeat("Phase10_restart",
                  step, {"hd": hd, "ep": ep, "val_mae": round(val_mae, 4)})

    # ── C: LMC ───────────────────────────────────────────────────────────────
    log(f"\n  LMC for width={hd}")
    ep_post = plateau_ep + 3
    if ep_post not in saved_sds:
        ep_post = max(k for k in saved_sds if k > plateau_ep)
        log(f"  WARNING: ep+3 not found; using ep{ep_post}")

    model_lmc = MolecularGNN(
        num_atom_types=NUM_ATOM_TYPES,
        hidden_dim=hd,
        num_blocks=N_BLOCKS,
        num_gaussians=N_GAUSSIANS,
        cutoff=CUTOFF,
        num_targets=1,
    ).to(device)

    lmc_label  = f"BF16-{hd}  ep{plateau_ep} <-> ep{ep_post}"
    records, barrier, peak_alpha = run_lmc(
        saved_sds[plateau_ep], saved_sds[ep_post],
        model_lmc, val_loader, device, std, lmc_label, N_INTERP_STEPS
    )
    log(f"  Barrier: {barrier:.4f} eV  peak_alpha={peak_alpha:.2f}")

    return {
        "hidden_dim":          hd,
        "n_params":            n_params,
        "plateau_ep":          plateau_ep,
        "mae_at_plateau_ev":   round(mae_plateau, 4),
        "training_trajectory": training_trajectory,
        "restart_trajectory":  restart_trajectory,
        "lmc": {
            "ep_pre":         plateau_ep,
            "ep_post":        ep_post,
            "n_interp_steps": N_INTERP_STEPS,
            "interpolation":  records,
            "barrier_ev":     barrier,
            "peak_alpha":     peak_alpha,
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 68)
    log("  Phase 10 — Width Scaling: BF16 LMC barrier vs hidden_dim")
    log(f"  Widths: {WIDTHS}")
    log(f"  256 and 512 reuse GCS checkpoints; 64 and 128 trained from scratch")
    log(f"  Each: {N_RESTART}-epoch warm restart, LMC plateau <-> plateau+3")
    log("=" * 68)

    device = get_device()
    log(f"Device: {device}" + (" (TPU)" if XLA_AVAILABLE else ""))

    log("Loading QM9 data ...")
    train_loader, val_loader, _ = get_dataloaders(
        str(DATA_DIR), batch_size=BATCH_SIZE, num_workers=4
    )
    std = train_loader.dataset.std
    log(f"QM9 std: {std:.4f} eV")

    notify("PHASE_START", "[Phase10] Width scaling experiment begun",
           data={"widths": WIDTHS})

    width_results = []
    for hd in WIDTHS:
        result = run_width(hd, WIDTH_CONFIG[hd], device,
                           train_loader, val_loader, std)
        width_results.append(result)

    # ── Results table ─────────────────────────────────────────────────────────
    log(f"\n{'='*68}")
    log("  PHASE 10 SUMMARY: BF16 barrier vs hidden_dim")
    log(f"  {'hidden_dim':>12}  {'params':>10}  {'ep_mae (eV)':>12}  "
        f"{'barrier (eV)':>13}  {'peak_alpha':>11}")
    log(f"  {'─'*12}  {'─'*10}  {'─'*12}  {'─'*13}  {'─'*11}")
    for r in width_results:
        log(f"  {r['hidden_dim']:>12}  {r['n_params']:>10,}  "
            f"{r['mae_at_plateau_ev']:>12.4f}  "
            f"{r['lmc']['barrier_ev']:>13.4f}  "
            f"{r['lmc']['peak_alpha']:>11.2f}")
    log(f"{'='*68}")

    # Monotonicity check: does barrier increase with width?
    barriers    = [r["lmc"]["barrier_ev"] for r in width_results]
    monotone_up = all(barriers[i] <= barriers[i + 1]
                      for i in range(len(barriers) - 1))
    log(f"\n  Barriers: {barriers}")
    log(f"  Monotone increasing with width: {monotone_up}")
    if monotone_up:
        log("  -> BF16 sharpening scales with model capacity")
    else:
        log("  -> No monotone scaling — capacity-independent or non-monotone landscape")

    # ── Save and upload ───────────────────────────────────────────────────────
    summary = {
        "experiment":       "phase10_width_scaling",
        "dataset":          "QM9",
        "widths_tested":    WIDTHS,
        "precision":        "bf16",
        "n_blocks":         N_BLOCKS,
        "n_train_epochs":   N_TRAIN_EPOCHS,
        "n_restart_epochs": N_RESTART,
        "restart_lr":       RESTART_LR,
        "batch_size":       BATCH_SIZE,
        "qm9_std_ev":       round(float(std), 4),
        "results_by_width": width_results,
        "summary_table": [
            {
                "hidden_dim":    r["hidden_dim"],
                "n_params":      r["n_params"],
                "mae_plateau_ev": r["mae_at_plateau_ev"],
                "barrier_ev":    r["lmc"]["barrier_ev"],
                "peak_alpha":    r["lmc"]["peak_alpha"],
            }
            for r in width_results
        ],
        "monotone_barrier_increase": monotone_up,
        "gcs_output": f"{GCS_BASE}/phase10_width_scaling/",
    }

    out_json = OUT_DIR / "results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    log(f"\nResults saved -> {out_json}")
    gsutil_cp(out_json, f"{GCS_BASE}/phase10_width_scaling/results.json")
    log(f"GCS: {GCS_BASE}/phase10_width_scaling/")

    notify("PHASE_COMPLETE", "[Phase10] Width scaling LMC complete",
           data={"barriers": {str(r["hidden_dim"]): r["lmc"]["barrier_ev"]
                              for r in width_results},
                 "monotone_up": monotone_up})


if __name__ == "__main__":
    main()
