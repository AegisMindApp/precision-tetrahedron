#!/usr/bin/env python3
"""
phase7c_hessian_sharpness.py
-----------------------------
Loss-landscape sharpness comparison across all three conditions using
random Rademacher perturbations (SAM-style).  Avoids second-order derivatives,
which are unreliable / memory-intensive on XLA.

Method
------
For each condition at its plateau checkpoint:
  1. Compute baseline loss  L(θ)
  2. Draw N_PERTURB=50 Rademacher sign vectors δ, scaled to ||δ||₂ = ε
  3. Compute L(θ + δ) with temporarily perturbed weights (no grad)
  4. Report  mean/max/p95  of  L(θ+δ) − L(θ)  across the 50 samples

Perturbation scales ε ∈ {0.001, 0.005, 0.01, 0.05} — all four are measured
and reported in a single run.

Conditions and plateau checkpoints
-----------------------------------
  A — FP32-256  : gs://.../condition_A_epoch80.pt
  B — BF16-256  : gs://.../condition_B_epoch80.pt
  C — BF16-512  : gs://.../condition_C_epoch40.pt   (c_restart subdir)

GCS output: gs://aegismind-tpu-results/aegis_flashoptim/phase7c_hessian_sharpness/results.json
"""

import os, sys, json, time, subprocess
from pathlib import Path
from typing import Dict, List

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

OUT_DIR  = Path("/tmp/phase7c_hessian_sharpness")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("/tmp/qm9")

N_PERTURB   = 50                          # perturbation samples per (condition, ε)
EPSILONS    = [0.001, 0.005, 0.01, 0.05]  # perturbation scales (L2 norm)

# Plateau checkpoints for each condition
CONDITIONS = [
    {
        "label":    "FP32-256",
        "key":      "A",
        "gcs_path": "{base}/condition_A_epoch80.pt",
        "local":    "condition_A_epoch80.pt",
        "epoch":    80,
    },
    {
        "label":    "BF16-256",
        "key":      "B",
        "gcs_path": "{base}/condition_B_epoch80.pt",
        "local":    "condition_B_epoch80.pt",
        "epoch":    80,
    },
    {
        "label":    "BF16-512",
        "key":      "C",
        "gcs_path": "{base}/c_restart/condition_C_epoch40.pt",
        "local":    "condition_C_epoch40.pt",
        "epoch":    40,
    },
]

# ─────────────────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def gsutil_cp(local: Path, gcs: str):
    subprocess.run(["gsutil", "-q", "cp", str(local), gcs], check=False)


def gsutil_download(gcs: str, local: Path):
    subprocess.run(["gsutil", "-q", "cp", gcs, str(local)], check=True)


# ── Loss evaluation (single pass, FP32 accumulation) ─────────────────────────

@torch.no_grad()
def compute_loss(model, loader, device, n_batches: int = 20) -> float:
    """
    Compute MSE loss on the first `n_batches` validation batches.
    Using a fixed subset keeps sharpness measurement fast and comparable.
    """
    model.eval()
    total_loss = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        z, pos, edge_src, edge_dst, assign_mat, num_graphs, edge_valid, atom_valid = \
            batch_to_graph(batch, device)
        target = batch['target'].to(device)
        pred = model(z, pos, edge_src, edge_dst, assign_mat,
                     num_graphs, edge_valid, atom_valid)
        total_loss += F.mse_loss(pred.float(), target.float()).item()
        n += 1
        if XLA_AVAILABLE:
            xm.mark_step()
    return total_loss / max(n, 1)


# ── Rademacher perturbation ───────────────────────────────────────────────────

def rademacher_perturbation(
    model: torch.nn.Module,
    epsilon: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Build a Rademacher sign-vector perturbation δ scaled so ||δ||₂ = ε.

    Each parameter tensor gets an independent sign vector; then the whole
    concatenated vector is re-scaled to have global L2 norm = epsilon.
    """
    sign_vecs: Dict[str, torch.Tensor] = {}
    total_numel = 0
    for name, p in model.named_parameters():
        signs = (torch.randint(0, 2, p.shape, device=device) * 2 - 1).float()
        sign_vecs[name] = signs
        total_numel += p.numel()

    # Global L2 norm of the concatenated sign vector
    raw_norm = float(np.sqrt(total_numel))   # ||±1||₂ = sqrt(d)

    # Scale so global ||δ||₂ = epsilon
    scale = epsilon / raw_norm
    return {k: v * scale for k, v in sign_vecs.items()}


@torch.no_grad()
def measure_sharpness_single(
    model: torch.nn.Module,
    delta: Dict[str, torch.Tensor],
    baseline_loss: float,
    loader,
    device: torch.device,
    n_batches: int = 20,
) -> float:
    """
    Temporarily add δ to model weights, compute loss, restore weights.
    Returns L(θ+δ) − L(θ).
    """
    # Apply perturbation
    for name, p in model.named_parameters():
        p.data.add_(delta[name].to(p.device))

    perturbed_loss = compute_loss(model, loader, device, n_batches)

    # Restore original weights
    for name, p in model.named_parameters():
        p.data.sub_(delta[name].to(p.device))

    return perturbed_loss - baseline_loss


def measure_sharpness(
    model: torch.nn.Module,
    epsilon: float,
    loader,
    device: torch.device,
    baseline_loss: float,
    n_perturb: int = N_PERTURB,
) -> Dict:
    """
    Run n_perturb Rademacher perturbations at scale epsilon.
    Returns dict with mean, max, p95 sharpness values.
    """
    deltas = [compute_loss.__func__ if False else None]  # placeholder
    sharpness_samples: List[float] = []

    for i in range(n_perturb):
        delta = rademacher_perturbation(model, epsilon, device)
        ds = measure_sharpness_single(model, delta, baseline_loss,
                                      loader, device)
        sharpness_samples.append(ds)
        if (i + 1) % 10 == 0:
            log(f"      perturbation {i+1}/{n_perturb}  "
                f"running mean={np.mean(sharpness_samples):.6f}")
        if XLA_AVAILABLE:
            xm.mark_step()

    arr = np.array(sharpness_samples, dtype=np.float64)
    return {
        "epsilon":       epsilon,
        "n_perturb":     n_perturb,
        "mean":          round(float(arr.mean()), 6),
        "max":           round(float(arr.max()),  6),
        "p95":           round(float(np.percentile(arr, 95)), 6),
        "std":           round(float(arr.std()),  6),
        "min":           round(float(arr.min()),  6),
        "samples":       [round(float(x), 6) for x in arr.tolist()],
    }


# ── Per-condition pipeline ────────────────────────────────────────────────────

def run_condition(cond: dict, val_loader, device) -> dict:
    label    = cond["label"]
    key      = cond["key"]
    gcs_path = cond["gcs_path"].format(base=GCS_BASE)
    local_p  = OUT_DIR / cond["local"]

    log(f"\n{'─'*60}")
    log(f"  Condition {key}: {label}  (plateau ep{cond['epoch']})")
    log(f"{'─'*60}")

    # Download checkpoint
    if not local_p.exists():
        log(f"Downloading {gcs_path} ...")
        gsutil_download(gcs_path, local_p)
    else:
        log(f"Using cached checkpoint: {local_p}")

    # Build model and load weights
    model = build_model(key, device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model params: {n_params:,}")

    ckpt = torch.load(local_p, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    log(f"Checkpoint epoch: {ckpt.get('epoch', '?')}  "
        f"val_mae_ev: {ckpt.get('val_mae_ev', '?')}")

    # Baseline loss (fixed 20-batch subset, consistent across all conditions)
    baseline_loss = compute_loss(model, val_loader, device, n_batches=20)
    log(f"Baseline MSE loss (20 batches): {baseline_loss:.6f}")

    # Sharpness across all epsilon values
    results_by_eps: List[dict] = []
    for eps in EPSILONS:
        log(f"\n  ε={eps}  ({N_PERTURB} perturbations) ...")
        sr = measure_sharpness(model, eps, val_loader, device, baseline_loss)
        results_by_eps.append(sr)
        log(f"  ε={eps:6.3f}  mean={sr['mean']:+.6f}  "
            f"max={sr['max']:+.6f}  p95={sr['p95']:+.6f}  std={sr['std']:.6f}")

        heartbeat(f"Phase7c_{key}_eps{eps}", 0,
                  {"condition": key, "epsilon": eps,
                   "mean_sharpness": sr["mean"], "max_sharpness": sr["max"]})

    return {
        "condition":     key,
        "label":         label,
        "plateau_epoch": cond["epoch"],
        "n_params":      n_params,
        "baseline_loss": round(baseline_loss, 6),
        "val_mae_ev_ckpt": ckpt.get("val_mae_ev"),
        "sharpness_by_epsilon": results_by_eps,
    }


# ── Print comparison table ────────────────────────────────────────────────────

def print_table(all_results: List[dict]):
    log("\n" + "=" * 80)
    log("  PHASE 7c SUMMARY: Random-Perturbation Sharpness")
    log("  Method: N=50 Rademacher vectors, L2 norm = ε")
    log("=" * 80)

    # Header
    col_w = 14
    hdr_cond = f"{'Condition':<16}"
    hdr_eps  = "".join(f"{'ε='+str(e):>{col_w}}" for e in EPSILONS)
    log(f"  {hdr_cond}  Metric     {hdr_eps}")
    log(f"  {'─'*16}  {'─'*9}" + "".join("  " + "─"*(col_w-2) for _ in EPSILONS))

    for res in all_results:
        cond  = res["label"]
        by_ep = {r["epsilon"]: r for r in res["sharpness_by_epsilon"]}
        for metric in ("mean", "max", "p95"):
            tag = f"{cond:<16}" if metric == "mean" else f"{'':16}"
            vals = "".join(
                f"{by_ep[e][metric]:>{col_w}.6f}" for e in EPSILONS
            )
            log(f"  {tag}  {metric:<9}{vals}")
        log("")

    log("=" * 80)
    log("  Interpretation guide:")
    log("  mean_sharpness > 0  -> loss rises on average under perturbation")
    log("  Higher mean/max     -> sharper minimum (less flat basin)")
    log("  Hypothesis: BF16-512 > BF16-256 > FP32-256 (sharpness increases")
    log("              with reduced precision and/or wider network)")
    log("=" * 80)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 64)
    log("  Phase 7c — Loss Landscape Sharpness (Rademacher Perturbation)")
    log(f"  N_PERTURB={N_PERTURB}  ε ∈ {EPSILONS}")
    log(f"  Conditions: A (FP32-256 ep80), B (BF16-256 ep80), "
        f"C (BF16-512 ep40)")
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

    # ── Data (val only needed) ─────────────────────────────────────────────────
    log("Loading QM9 data ...")
    _, val_loader, _ = get_dataloaders(
        str(DATA_DIR), batch_size=32, num_workers=4
    )
    log("Data ready.")

    notify("PHASE_START", "[Phase7c] Rademacher sharpness measurement begun",
           data={"conditions": [c["key"] for c in CONDITIONS],
                 "epsilons": EPSILONS, "n_perturb": N_PERTURB})

    # ── Run all conditions ────────────────────────────────────────────────────
    all_results: List[dict] = []
    for cond in CONDITIONS:
        res = run_condition(cond, val_loader, device)
        all_results.append(res)

    # ── Print consolidated table ──────────────────────────────────────────────
    print_table(all_results)

    # ── Hypothesis verdict ────────────────────────────────────────────────────
    log("\n  Hypothesis check (mean sharpness at ε=0.01):")
    eps_ref = 0.01
    means = {}
    for res in all_results:
        for sr in res["sharpness_by_epsilon"]:
            if abs(sr["epsilon"] - eps_ref) < 1e-9:
                means[res["condition"]] = sr["mean"]
                break
    order_ok = means.get("C", 0) > means.get("B", 0) > means.get("A", 0)
    log(f"  A={means.get('A', '?'):.6f}  B={means.get('B', '?'):.6f}  "
        f"C={means.get('C', '?'):.6f}")
    if order_ok:
        log("  -> BF16-512 > BF16-256 > FP32-256  [HYPOTHESIS SUPPORTED]")
    else:
        log("  -> Order not C > B > A  [HYPOTHESIS NOT SUPPORTED at ε=0.01]")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    summary = {
        "experiment":  "phase7c_hessian_sharpness",
        "method":      "rademacher_perturbation",
        "n_perturb":   N_PERTURB,
        "epsilons":    EPSILONS,
        "n_batches":   20,
        "conditions":  all_results,
        "hypothesis": (
            "BF16-512 sharpness > BF16-256 sharpness > FP32-256 sharpness "
            "if reduced precision and/or wider networks sharpen loss minima"
        ),
        "hypothesis_supported_at_eps_0p01": order_ok,
        "means_at_eps_0p01": means,
    }

    # Strip raw sample lists before saving (large arrays)
    summary_compact = json.loads(json.dumps(summary))
    for cond_res in summary_compact["conditions"]:
        for sr in cond_res["sharpness_by_epsilon"]:
            sr.pop("samples", None)

    out_json = OUT_DIR / "results.json"
    out_json.write_text(json.dumps(summary_compact, indent=2))
    log(f"\nResults saved -> {out_json}")
    gsutil_cp(out_json, f"{GCS_BASE}/phase7c_hessian_sharpness/results.json")

    # Also save a version with full samples for reanalysis
    out_full = OUT_DIR / "results_full_samples.json"
    out_full.write_text(json.dumps(summary, indent=2))
    gsutil_cp(out_full,
              f"{GCS_BASE}/phase7c_hessian_sharpness/results_full_samples.json")

    log(f"GCS: {GCS_BASE}/phase7c_hessian_sharpness/")

    notify("PHASE_COMPLETE", "[Phase7c] Sharpness measurement complete",
           data={"means_at_eps_0p01": means,
                 "hypothesis_supported": order_ok})


if __name__ == "__main__":
    main()
