#!/usr/bin/env python3
"""
phase18_cyclic_lr_fp32.py
-------------------------
GNS-oscillation control: can FP32 training with a cyclic LR schedule that
induces BF16-like GNS oscillations replicate the BF16 LMC barrier?

Motivation (Phase 11 result)
-----------------------------
Phase 11 showed BF16 has *higher* integrated GNS (ratio 1.596×) and a
~15-epoch oscillation period (amplitude 0.096–2.537) vs FP32's smoother
trajectory. The 273× LMC barrier cannot be attributed to gradient noise.

Phase 18 tests the alternative: does the *oscillatory pattern* (not the
precision per se) drive the basin divergence? Two FP32 runs are trained:

  fp32_baseline  — CosineAnnealingLR(T_max=80), seed 42 (Phase 11 replica)
  fp32_cyclic    — CosineAnnealingWarmRestarts(T_0=15, T_mult=1), seed 42
                   This forces ~15-epoch LR oscillations matching BF16's
                   GNS period. Warm restarts push the model through repeated
                   high-LR phases, expected to induce GNS oscillations.

Measurements:
  1. GNS every 5 epochs for both runs
  2. LMC barriers:
       fp32_baseline_ep80 ↔ fp32_cyclic_ep80  (did cyclic diverge from baseline?)
       fp32_baseline_ep80 ↔ bf16_ep80          (Phase 11 reference, recomputed)
       fp32_cyclic_ep80   ↔ bf16_ep80          (did cyclic converge toward BF16?)

Interpretation:
  If barrier(fp32_cyclic ↔ bf16) << barrier(fp32_baseline ↔ bf16):
      → oscillatory dynamics (not precision) drive convergence to BF16-like basins
  If barrier(fp32_cyclic ↔ bf16) ≈ barrier(fp32_baseline ↔ bf16):
      → precision itself is the driver; dynamics alone cannot replicate the effect

Phase 11 BF16 ep80 checkpoint is downloaded from GCS if available.

GCS output: gs://.../aegis_flashoptim/phase18_cyclic_lr_fp32/results.json
"""

import os, sys, json, time, subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

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

OUT_DIR  = Path("/tmp/phase18_cyclic_lr_fp32")
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
CYCLIC_T0     = 15   # matches BF16 oscillation period from Phase 11

# Phase 11 BF16 ep80 checkpoint on GCS
PHASE11_BF16_GCS = f"{GCS_BUCKET}/{RUN_ID}/phase11_gradient_noise/bf16_ep80.pt"

def log(msg): print(f"[Phase18-Cyclic] {msg}", flush=True)
def gsutil_cp(src, dst): subprocess.run(["gsutil", "-q", "cp", str(src), dst], check=False)
def gsutil_dl(src, dst):
    r = subprocess.run(["gsutil", "-q", "cp", src, str(dst)], capture_output=True)
    return r.returncode == 0 and Path(dst).exists()


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


def compute_gns(model, loader, device, n_batches):
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
    G = torch.stack(grad_samples)
    mean_g = G.mean(0)
    var_g  = G.var(0, unbiased=True)
    return round((mean_g**2).sum().item() / (var_g.sum().item() + 1e-12), 6)


def train_run(label, scheduler_fn, train_loader, val_loader, device):
    """Train FP32 with given scheduler factory. Returns trajectory + final checkpoint."""
    log(f"\n  ── {label} ──")

    # Resume from GCS checkpoint if available — skips full retraining after preemption
    ckpt_path = OUT_DIR / f"{label}_ep80.pt"
    gcs_ckpt  = f"{GCS_BASE}/phase18_cyclic_lr_fp32/{ckpt_path.name}"
    if not ckpt_path.exists():
        stat_r = subprocess.run(["gsutil", "-q", "stat", gcs_ckpt], capture_output=True)
        if stat_r.returncode == 0:
            log(f"  GCS checkpoint found — downloading, skipping training")
            subprocess.run(["gsutil", "-q", "cp", gcs_ckpt, str(ckpt_path)], check=False)
    if ckpt_path.exists():
        model = make_model(device)
        sd = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(sd)
        model.to(device)
        final_mae = round(eval_mae(model, val_loader, device), 5)
        log(f"  {label} resumed from checkpoint  val_mae = {final_mae:.4f} eV")
        return {"trajectory": [], "final_mae_ev": final_mae, "ckpt": str(ckpt_path), "resumed": True}

    model = make_model(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    sched = scheduler_fn(opt)

    trajectory = []
    for ep in range(1, N_EPOCHS + 1):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            z, pos, edge_src, edge_dst, assign_mat, B, edge_valid, atom_valid = batch_to_graph(batch, device)
            y = batch['target'].to(device)
            opt.zero_grad()
            pred = model(z, pos, edge_src, edge_dst, assign_mat, B, edge_valid, atom_valid)
            loss = F.mse_loss(pred.squeeze(), y)
            loss.backward()
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
            val_mae = eval_mae(model, val_loader, device)
            gns     = compute_gns(model, train_loader, device, N_GNS_BATCHES)
            current_lr = opt.param_groups[0]["lr"]
            entry = {"epoch": ep, "label": label,
                     "train_loss": round(total / count, 6),
                     "val_mae_ev": round(val_mae, 4),
                     "gns": gns,
                     "lr": round(current_lr, 8)}
            trajectory.append(entry)
            log(f"    {label} ep{ep:3d}  val_mae={val_mae:.4f}  gns={gns}  lr={current_lr:.2e}")
            heartbeat(f"Phase18_{label}", epoch=ep, metrics={"val_mae": round(val_mae, 4), "gns": gns})
        elif ep % 10 == 0:
            log(f"    {label} ep{ep:3d}  train_loss={total/count:.6f}")

    final_mae  = eval_mae(model, val_loader, device)
    ckpt_path  = OUT_DIR / f"{label}_ep80.pt"
    torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()}, ckpt_path)
    gsutil_cp(ckpt_path, f"{GCS_BASE}/phase18_cyclic_lr_fp32/{ckpt_path.name}")
    log(f"  {label} final val_mae = {final_mae:.4f} eV")
    return {"trajectory": trajectory, "final_mae_ev": final_mae, "ckpt": str(ckpt_path)}


def _load_sd(path):
    """Load state dict, unwrapping training checkpoints saved as {'model': sd, 'epoch': n}."""
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict) and "model" in raw and "epoch" in raw:
        return raw["model"]
    return raw


def lmc_barrier(label, sd0_path, sd1_path, val_loader, device):
    sd0   = _load_sd(sd0_path)
    sd1   = _load_sd(sd1_path)
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
    log(f"  LMC {label:50s}  barrier={barrier:.4f} eV  peak_α={peak_a}")
    return {"label": label, "alphas": alphas, "maes": maes,
            "barrier_ev": barrier, "baseline_ev": baseline, "peak_alpha": peak_a}


def main():
    log("=" * 65)
    log("  Phase 18 — Cyclic LR FP32 control experiment")
    log(f"  Tests: does BF16 GNS oscillation pattern (T≈{CYCLIC_T0} ep) drive")
    log(f"         landscape divergence, or is precision per se the driver?")
    log("=" * 65)

    device = get_device()
    log(f"Device: {device}")
    train_loader, val_loader, _ = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)

    # ── Train fp32_baseline (Phase 11 replica) ────────────────────────────────
    baseline_res = train_run(
        "fp32_baseline",
        lambda opt: CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-6),
        train_loader, val_loader, device)

    # ── Train fp32_cyclic (warm restarts every T0=15 epochs) ─────────────────
    cyclic_res = train_run(
        "fp32_cyclic",
        lambda opt: CosineAnnealingWarmRestarts(opt, T_0=CYCLIC_T0, T_mult=1, eta_min=1e-6),
        train_loader, val_loader, device)

    # ── Download Phase 11 BF16 ep80 checkpoint ────────────────────────────────
    bf16_local = OUT_DIR / "bf16_ep80_phase11.pt"
    if not bf16_local.exists():
        log("Downloading Phase 11 BF16 ep80 from GCS...")
        if not gsutil_dl(PHASE11_BF16_GCS, bf16_local):
            log("WARNING: BF16 checkpoint not found — cross-precision LMC skipped")
            bf16_local = None

    # ── LMC ──────────────────────────────────────────────────────────────────
    log("\nRunning LMC pairs...")
    lmc_results = []
    baseline_path = OUT_DIR / "fp32_baseline_ep80.pt"
    cyclic_path   = OUT_DIR / "fp32_cyclic_ep80.pt"

    # 1. Baseline vs cyclic: did cyclic diverge from standard FP32?
    r1 = lmc_barrier("fp32_baseline_ep80 ↔ fp32_cyclic_ep80",
                     baseline_path, cyclic_path, val_loader, device)
    r1["comparison"] = "cyclic_vs_baseline"
    lmc_results.append(r1)

    if bf16_local:
        # 2. Baseline vs BF16: Phase 11 reference (recomputed)
        r2 = lmc_barrier("fp32_baseline_ep80 ↔ bf16_ep80 (Phase11 ref)",
                         baseline_path, bf16_local, val_loader, device)
        r2["comparison"] = "baseline_vs_bf16"
        lmc_results.append(r2)

        # 3. Cyclic vs BF16: key test
        r3 = lmc_barrier("fp32_cyclic_ep80   ↔ bf16_ep80 (KEY TEST)",
                         cyclic_path, bf16_local, val_loader, device)
        r3["comparison"] = "cyclic_vs_bf16"
        lmc_results.append(r3)

    # ── Report ────────────────────────────────────────────────────────────────
    log("\n" + "=" * 65)
    log("  CYCLIC LR CONTROL RESULTS")
    log("=" * 65)

    # GNS comparison: baseline vs cyclic
    log("\n  GNS trajectories (FP32 baseline vs FP32 cyclic vs BF16 Phase11 ref):")
    log(f"  {'Epoch':>6}  {'FP32-baseline':>15}  {'FP32-cyclic':>13}  {'BF16-Phase11':>14}")
    log(f"  {'─'*6}  {'─'*15}  {'─'*13}  {'─'*14}")
    bf16_ref = {5:0.803,10:0.225,15:1.778,20:1.699,25:0.195,30:2.537,
                35:1.799,40:0.096,45:0.484,50:0.759,55:1.107,60:0.163,
                65:0.663,70:0.777,75:0.646,80:0.367}
    baseline_gns = {e["epoch"]: e["gns"] for e in baseline_res["trajectory"]}
    cyclic_gns   = {e["epoch"]: e["gns"] for e in cyclic_res["trajectory"]}
    for ep in range(5, 81, 5):
        log(f"  {ep:>6}  {str(baseline_gns.get(ep,'—')):>15}  "
            f"{str(cyclic_gns.get(ep,'—')):>13}  {str(bf16_ref.get(ep,'—')):>14}")

    log("\n  LMC barrier summary:")
    log(f"  {'Pair':55s}  {'Barrier (eV)':>14}")
    log(f"  {'─'*55}  {'─'*14}")
    for r in lmc_results:
        log(f"  {r['label']:55s}  {r['barrier_ev']:>14.4f}")

    # Key interpretation
    if bf16_local and len(lmc_results) >= 3:
        b_baseline_bf16 = next(r["barrier_ev"] for r in lmc_results
                               if r["comparison"] == "baseline_vs_bf16")
        b_cyclic_bf16   = next(r["barrier_ev"] for r in lmc_results
                               if r["comparison"] == "cyclic_vs_bf16")
        ratio = b_cyclic_bf16 / (b_baseline_bf16 + 1e-8)
        if ratio < 0.5:
            interpretation = (f"Cyclic barrier vs BF16 = {b_cyclic_bf16:.4f} eV "
                              f"({ratio:.2f}× baseline). SUPPORTS oscillatory-dynamics hypothesis.")
        elif ratio > 0.8:
            interpretation = (f"Cyclic barrier vs BF16 = {b_cyclic_bf16:.4f} eV "
                              f"({ratio:.2f}× baseline). Precision is the driver — dynamics alone insufficient.")
        else:
            interpretation = (f"Cyclic barrier vs BF16 = {b_cyclic_bf16:.4f} eV "
                              f"({ratio:.2f}× baseline). Partial — both dynamics and precision contribute.")
        log(f"\n  Interpretation: {interpretation}")
    else:
        interpretation = "BF16 checkpoint unavailable — cross-precision LMC not run"
        log(f"\n  {interpretation}")

    results = {
        "experiment":     "phase18_cyclic_lr_fp32",
        "hidden_dim":     HIDDEN_DIM,
        "n_epochs":       N_EPOCHS,
        "seed":           SEED,
        "cyclic_T0":      CYCLIC_T0,
        "fp32_baseline":  baseline_res,
        "fp32_cyclic":    cyclic_res,
        "lmc_results":    lmc_results,
        "interpretation": interpretation,
        "phase11_bf16_gns_ref": bf16_ref,
        "phase7_ref_barrier_ev": 1.447,
    }
    out = OUT_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    gsutil_cp(out, f"{GCS_BASE}/phase18_cyclic_lr_fp32/results.json")
    log(f"\n  Results → {out}")
    log(f"  GCS     → {GCS_BASE}/phase18_cyclic_lr_fp32/results.json")
    notify("phase18_complete", "Phase 18 complete — cyclic LR control done", data=results)


if __name__ == "__main__":
    main()
