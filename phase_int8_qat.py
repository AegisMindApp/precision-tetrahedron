#!/usr/bin/env python3
"""
phase_int8_qat.py — INT8 QAT: fourth precision vertex of the LMC triangle.

Phase 17 showed naive INT8sim diverges at ep5 under a vanilla cosine schedule
(val_mae=469 eV). This experiment uses a proper QAT schedule:

  Epochs 0-14  : FP32 warm-up  (stable initialisation, no quantisation)
  Epochs 15-80 : fake-INT8     (weights snapped to INT8 grid after each optimizer step)

Phase 17 GCS checkpoints (fp32/bf16/fp16 ep80) are downloaded and used for
LMC comparison without re-training those conditions.

Key question: does INT8 form a THIRD precision cluster, or merge with {FP16}?

  Exponent-range hypothesis (Phase 17) partitions into:
    {FP32, BF16}  8-bit exponent, barrier ~0.014 eV
    {FP16}        5-bit exponent, barrier ~0.149 eV from either 8-bit format

  INT8 has NO exponent bits (fixed-point, dynamic range ±127/scale). Possible outcomes:
    (A) INT8 equidistant from everything  →  third isolated cluster
    (B) INT8 merges with FP16             →  restricted-range super-cluster
    (C) INT8 merges with {FP32, BF16}    →  scale-invariant; exponent dominates

GCS output: gs://.../aegis_flashoptim/phase_int8_qat/results.json
"""

import os, sys, json, subprocess
from pathlib import Path

import torch
import torch.nn as nn
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

GCS_BUCKET  = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID      = os.environ.get("RUN_ID",     "aegis_flashoptim")
GCS_P17     = f"{GCS_BUCKET}/{RUN_ID}/phase17_precision_dial"
GCS_OUT     = f"{GCS_BUCKET}/{RUN_ID}/phase_int8_qat"

OUT_DIR     = Path("/tmp/phase_int8_qat")
P17_DIR     = Path("/tmp/phase17_ckpts")
DATA_DIR    = Path("/tmp/qm9")
for d in [OUT_DIR, P17_DIR]: d.mkdir(parents=True, exist_ok=True)

# ── Config (identical to Phase 17 for direct comparison) ─────────────────────
HIDDEN_DIM   = 256
N_BLOCKS     = 6
N_EPOCHS     = 80
QAT_WARMUP   = 15   # float FP32 warm-up epochs before fake-quant activates
BATCH_SIZE   = 32
LR_INIT      = 1e-4
WEIGHT_DECAY = 1e-4
SEED         = 42
N_ALPHA      = 11
LMC_BATCH    = 16   # smaller batch for CPU LMC eval

def log(msg): print(f"[INT8-QAT] {msg}", flush=True)
def gsutil_cp(src, dst): subprocess.run(["gsutil", "-q", "cp", str(src), dst], check=False)
def gsutil_dl(gcs, local): subprocess.run(["gsutil", "-q", "cp", gcs, str(local)], check=False)


def get_device():
    if XLA_AVAILABLE:
        return xm.xla_device()
    return torch.device("cpu")


def make_model(device):
    torch.manual_seed(SEED)
    return MolecularGNN(hidden_dim=HIDDEN_DIM, num_blocks=N_BLOCKS).to(device)


def quantize_to_int8_grid(p):
    """Return per-tensor symmetric INT8 fake-quantized copy of p (XLA-safe)."""
    scale = p.abs().max() / 127.0 + 1e-8
    return (p / scale).round().clamp(-128, 127) * scale


def fake_quantize_model(model):
    """Snap all float32 parameters to the INT8 grid in-place.
    Only used for the final checkpoint — NOT during training (use STE loop)."""
    with torch.no_grad():
        for p in model.parameters():
            if p.dtype == torch.float32:
                p.copy_(quantize_to_int8_grid(p))


def eval_mae(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            z, pos, edge_src, edge_dst, assign_mat, B, edge_valid, atom_valid = batch_to_graph(batch, device)
            y    = batch["target"].to(device)
            pred = model(z, pos, edge_src, edge_dst, assign_mat, B, edge_valid, atom_valid)
            total += F.l1_loss(pred.squeeze(), y, reduction="sum").item()
            count += y.numel()
    if XLA_AVAILABLE and str(device) != "cpu":
        xm.mark_step()
    return total / count if count > 0 else float("inf")


# ── Training ──────────────────────────────────────────────────────────────────

def train_int8_qat(train_loader, val_loader, device):
    ckpt = OUT_DIR / "int8qat_ep80.pt"
    gcs  = f"{GCS_OUT}/int8qat_ep80.pt"
    if ckpt.exists():
        log("int8qat_ep80.pt already on disk — skipping training"); return
    if subprocess.run(["gsutil", "-q", "stat", gcs], capture_output=True).returncode == 0:
        log("int8qat_ep80.pt on GCS — downloading")
        gsutil_dl(gcs, ckpt); return

    # Resume from latest GCS epoch checkpoint if available (preemption recovery)
    start_ep = 1
    model = make_model(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-5)
    for resume_ep in range(N_EPOCHS // 10 * 10, 0, -10):
        gcs_ckpt = f"{GCS_OUT}/int8qat_ep{resume_ep:03d}.pt"
        local_ckpt = OUT_DIR / f"int8qat_ep{resume_ep:03d}.pt"
        if subprocess.run(["gsutil", "-q", "stat", gcs_ckpt], capture_output=True).returncode == 0:
            log(f"Resuming from GCS checkpoint ep{resume_ep}")
            gsutil_dl(gcs_ckpt, local_ckpt)
            state = torch.load(local_ckpt, map_location="cpu")
            model.load_state_dict({k: v.to(device) for k, v in state["model"].items()})
            opt.load_state_dict(state["opt"])
            sched.load_state_dict(state["sched"])
            start_ep = resume_ep + 1
            log(f"  Resumed — continuing from ep{start_ep}")
            break

    log(f"Training INT8-QAT  (warm-up={QAT_WARMUP} ep FP32 → ep{QAT_WARMUP+1}-{N_EPOCHS} STE fake-quant, start_ep={start_ep})")

    # Collect trainable params once for STE loop
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for ep in range(start_ep, N_EPOCHS + 1):
        model.train()
        qat_active = ep > QAT_WARMUP

        for batch in train_loader:
            z, pos, edge_src, edge_dst, assign_mat, B, edge_valid, atom_valid = batch_to_graph(batch, device)
            y = batch["target"].to(device)
            opt.zero_grad()

            if qat_active:
                # STE fake-quantization: quantize weights for forward pass only.
                # Save FP32 originals → quantize in-place → forward/backward →
                # restore FP32 → optimizer step on clean FP32 weights.
                fp32_saved = [p.data.clone() for p in trainable_params]
                with torch.no_grad():
                    for p in trainable_params:
                        p.copy_(quantize_to_int8_grid(p))

            pred = model(z, pos, edge_src, edge_dst, assign_mat, B, edge_valid, atom_valid)
            F.mse_loss(pred.squeeze(), y).backward()

            if qat_active:
                # Restore FP32 weights before optimizer so moments track FP32 space
                with torch.no_grad():
                    for p, orig in zip(trainable_params, fp32_saved):
                        p.data.copy_(orig)

            if XLA_AVAILABLE:
                xm.optimizer_step(opt)
            else:
                opt.step()
            if XLA_AVAILABLE:
                xm.mark_step()  # flush per-batch — prevents epoch-level graph accumulation
        sched.step()
        if XLA_AVAILABLE:
            xm.mark_step()

        if ep % 10 == 0 or ep == N_EPOCHS:
            mae = eval_mae(model, val_loader, device)
            quant_status = "QAT-ON" if ep > QAT_WARMUP else "FP32-warmup"
            log(f"  ep{ep:3d}  val_mae={mae:.4f}  [{quant_status}]  lr={sched.get_last_lr()[0]:.2e}")
            # Save epoch checkpoint to GCS for preemption recovery
            ep_ckpt = OUT_DIR / f"int8qat_ep{ep:03d}.pt"
            torch.save({"model": {k: v.cpu().float() for k, v in model.state_dict().items()},
                        "opt": opt.state_dict(), "sched": sched.state_dict(), "ep": ep}, ep_ckpt)
            gsutil_cp(ep_ckpt, f"{GCS_OUT}/int8qat_ep{ep:03d}.pt")
            log(f"  checkpoint saved → GCS ep{ep:03d}")

    mae_final = eval_mae(model, val_loader, device)
    torch.save({k: v.cpu().float() for k, v in model.state_dict().items()}, ckpt)
    gsutil_cp(ckpt, gcs)
    log(f"INT8-QAT final val_mae = {mae_final:.4f} eV  → {ckpt.name}")


# ── Phase 17 checkpoint download ──────────────────────────────────────────────

def download_phase17_ckpts():
    for prec in ["fp32", "bf16", "fp16"]:
        dst = P17_DIR / f"{prec}_ep80.pt"
        if dst.exists():
            log(f"  {prec}_ep80.pt already cached"); continue
        src = f"{GCS_P17}/{prec}_ep80.pt"
        log(f"  Downloading {src}...")
        gsutil_dl(src, dst)
        if dst.exists():
            log(f"  {prec}_ep80.pt OK ({dst.stat().st_size // 1024} KB)")
        else:
            log(f"  WARNING: {prec}_ep80.pt download failed — LMC pair will be skipped")


# ── LMC ───────────────────────────────────────────────────────────────────────

def lmc_barrier(label, path_a, path_b, lmc_loader):
    """LMC on CPU — interpolate two FP32 state_dicts, eval on CPU val loader."""
    sd_a  = torch.load(path_a, map_location="cpu")
    sd_b  = torch.load(path_b, map_location="cpu")
    cpu   = torch.device("cpu")
    model = make_model(cpu)
    alphas = [round(i / (N_ALPHA - 1), 2) for i in range(N_ALPHA)]
    maes   = []
    for alpha in alphas:
        interp = {k: (1 - alpha) * sd_a[k].float() + alpha * sd_b[k].float()
                  for k in sd_a if k in sd_b}
        model.load_state_dict(interp, strict=False)
        mae = eval_mae(model, lmc_loader, cpu)
        maes.append(round(mae, 5))
        log(f"    α={alpha:.1f}  mae={mae:.5f} eV")
    baseline = min(maes[0], maes[-1])
    barrier  = round(max(maes) - baseline, 5)
    peak_a   = alphas[maes.index(max(maes))]
    log(f"  → {label}: barrier={barrier:.5f} eV  peak_α={peak_a:.1f}")
    return {"label": label, "alphas": alphas, "maes": maes,
            "barrier_ev": barrier, "baseline_ev": baseline, "peak_alpha": peak_a}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 65)
    log("  Phase INT8-QAT — Fourth precision vertex of LMC triangle")
    log(f"  Warm-up: {QAT_WARMUP} FP32 epochs; then fake-INT8 post-step quantisation")
    log("=" * 65)

    device = get_device()
    log(f"Device: {device}")

    train_loader, val_loader, _ = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)

    # 1. Train INT8-QAT checkpoint
    train_int8_qat(train_loader, val_loader, device)

    # 2. Download Phase 17 reference checkpoints (FP32/BF16/FP16)
    log("\nDownloading Phase 17 reference checkpoints...")
    download_phase17_ckpts()

    # 3. CPU val loader for LMC (avoids XLA state-dict reload complications)
    _, lmc_loader, _ = get_dataloaders(DATA_DIR, batch_size=LMC_BATCH)

    # 4. Run all LMC pairs involving INT8-QAT
    int8_ckpt = OUT_DIR / "int8qat_ep80.pt"
    pairs = [
        ("INT8-QAT ↔ FP32", int8_ckpt,             P17_DIR / "fp32_ep80.pt"),
        ("INT8-QAT ↔ BF16", int8_ckpt,             P17_DIR / "bf16_ep80.pt"),
        ("INT8-QAT ↔ FP16", int8_ckpt,             P17_DIR / "fp16_ep80.pt"),
        # Also re-measure Phase 17 pairs as cross-validation
        ("FP32 ↔ BF16 (P17 x-val)", P17_DIR / "fp32_ep80.pt", P17_DIR / "bf16_ep80.pt"),
        ("FP32 ↔ FP16 (P17 x-val)", P17_DIR / "fp32_ep80.pt", P17_DIR / "fp16_ep80.pt"),
    ]

    lmc_results = []
    for label, pa, pb in pairs:
        if not pa.exists() or not pb.exists():
            log(f"  Skipping {label}: checkpoint missing"); continue
        log(f"\n  LMC: {label}")
        lmc_results.append(lmc_barrier(label, pa, pb, lmc_loader))

    # 5. Report and interpret
    log("\n" + "=" * 65)
    log("  INT8-QAT LMC RESULTS")
    log("=" * 65)
    log(f"\n  {'Pair':<35}  {'Barrier (eV)':>12}  {'Peak α':>8}")
    log(f"  {'─'*55}")
    for r in lmc_results:
        log(f"  {r['label']:<35}  {r['barrier_ev']:>12.5f}  {r['peak_alpha']:>8.1f}")

    int8_barriers = {r["label"].split(" ↔ ")[1].split(" ")[0]: r["barrier_ev"]
                     for r in lmc_results if r["label"].startswith("INT8")}

    if int8_barriers:
        log("\n  INTERPRETATION:")
        log(f"  Phase 17 reference:  FP32↔BF16 = 0.0142 eV,  FP32↔FP16 = 0.1485 eV")
        b_fp32 = int8_barriers.get("FP32", None)
        b_bf16 = int8_barriers.get("BF16", None)
        b_fp16 = int8_barriers.get("FP16", None)
        if b_fp32 and b_bf16:
            # INT8 equidistant from FP32 and BF16 → exponent-free format is equally isolated from both
            eq = abs(b_fp32 - b_bf16) / max(b_fp32, b_bf16, 1e-8) < 0.20
            log(f"  INT8↔FP32 = {b_fp32:.4f} eV,  INT8↔BF16 = {b_bf16:.4f} eV  ({'equidistant ✓' if eq else 'asymmetric'})")
        if b_fp16 and b_fp32:
            if b_fp16 < b_fp32 * 0.5:
                log(f"  INT8↔FP16 = {b_fp16:.4f} eV < INT8↔FP32/2  →  INT8 merges with FP16 (restricted-range cluster)")
            elif b_fp16 > b_fp32 * 1.5:
                log(f"  INT8↔FP16 = {b_fp16:.4f} eV > INT8↔FP32×1.5  →  INT8 forms distinct 3rd cluster")
            else:
                log(f"  INT8↔FP16 = {b_fp16:.4f} eV ≈ INT8↔FP32  →  ambiguous; INT8 equidistant from all")

    results = {
        "experiment":           "phase_int8_qat",
        "qat_warmup_epochs":    QAT_WARMUP,
        "n_epochs":             N_EPOCHS,
        "seed":                 SEED,
        "hidden_dim":           HIDDEN_DIM,
        "lmc_results":          lmc_results,
        "phase17_ref": {
            "fp32_bf16_ev": 0.0142,
            "fp32_fp16_ev": 0.1485,
            "bf16_fp16_ev": 0.1504,
            "interpretation": "isosceles; exponent range is operative variable",
        },
    }
    out = OUT_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    gsutil_cp(out, f"{GCS_OUT}/results.json")
    log(f"\n  Results → {GCS_OUT}/results.json")
    log("=== phase_int8_qat complete ===")


if __name__ == "__main__":
    main()
