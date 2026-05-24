#!/usr/bin/env python3
"""
phase_resnet_lmc.py — Cross-precision LMC on CIFAR-10 SmallResNet final checkpoints.

§4.17 established that BF16 provides a 0.160-nat val_loss advantage over FP32
on CIFAR-10 ResNet (100 epochs, no restart triggered). This experiment tests
whether that advantage corresponds to a topologically distinct LMC basin or
merely different calibration within the same basin.

Pairs measured:
  FP32 (A_final) ↔ BF16 (B_final)   — does BF16 precision isolate in CV too?
  FP32 (A_final) ↔ FP32 (C_final)   — within-precision control (A=C by design)
  BF16 (B_final) ↔ FP32 (C_final)   — same as A↔B by symmetry

Checkpoints downloaded from GCS:
  gs://aegismind-tpu-results/aegis_flashoptim/phase_resnet_restart/{A,B,C}_final.pt

LMC computed on CPU (interpolate weights, evaluate on CIFAR-10 val, 11 points).

GCS output: gs://aegismind-tpu-results/aegis_flashoptim/phase_resnet_lmc/results.json
"""

import json, os, subprocess, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import tarfile, pickle, urllib.request

GCS_BASE = "gs://aegismind-tpu-results/aegis_flashoptim"
CKPT_DIR = Path("/tmp/resnet_lmc_ckpts")
OUT_DIR  = Path("/tmp/resnet_lmc_out")
CKPT_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

def log(msg): print(f"[ResNetLMC] {msg}", flush=True)
def gsutil_cp(src, dst): subprocess.run(["gsutil", "-q", "cp", str(src), str(dst)], check=False)
def gsutil_dl(src, dst): subprocess.run(["gsutil", "-q", "cp", str(src), str(dst)], check=True)


# ── SmallResNet (must match phase_resnet_restart.py exactly) ─────────────────

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(8, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(8, out_c)
        self.skip  = (nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False)
                      if (in_c != out_c or stride != 1) else nn.Identity())

    def forward(self, x):
        out = F.gelu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.gelu(out + self.skip(x))

class SmallResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.GELU()
        )
        self.layer1 = ResBlock(32,  64,  stride=2)
        self.layer2 = ResBlock(64,  128, stride=2)
        self.layer3 = ResBlock(128, 256, stride=2)
        self.head   = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.head(self.layer3(self.layer2(self.layer1(self.stem(x)))))


def load_cifar10_val():
    cifar_url  = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    data_dir   = Path("/tmp/cifar10_data")
    archive    = data_dir / "cifar-10-python.tar.gz"
    batches    = data_dir / "cifar-10-batches-py"
    data_dir.mkdir(exist_ok=True)
    if not batches.exists():
        if not archive.exists():
            log("Downloading CIFAR-10...")
            urllib.request.urlretrieve(cifar_url, str(archive))
        log("Extracting CIFAR-10...")
        with tarfile.open(archive) as tf:
            tf.extractall(data_dir)
    with open(batches / "test_batch", "rb") as f:
        d = pickle.load(f, encoding="bytes")
    X = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y = np.array(d[b"labels"], dtype=np.int64)
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
    std  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)
    X = (X - mean) / std
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)


def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += F.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def lmc_barrier(label, ckpt_a, ckpt_b, loader, device):
    log(f"  LMC: {label}")
    ma = SmallResNet().to(device)
    mb = SmallResNet().to(device)

    # Load — strip any XLA/BF16 artefacts, cast to FP32 for CPU interpolation
    sa = torch.load(ckpt_a, map_location="cpu")
    sb = torch.load(ckpt_b, map_location="cpu")
    sa = {k: v.float() for k, v in sa.items()}
    sb = {k: v.float() for k, v in sb.items()}
    ma.load_state_dict(sa)
    mb.load_state_dict(sb)

    params_a = {k: v for k, v in ma.named_parameters()}
    params_b = {k: v for k, v in mb.named_parameters()}

    alphas = [round(i * 0.1, 1) for i in range(11)]
    losses, accs = [], []
    mc = SmallResNet().to(device)
    for alpha in alphas:
        sd = {}
        for k in params_a:
            sd[k] = (1 - alpha) * params_a[k].data + alpha * params_b[k].data
        mc.load_state_dict(sd, strict=False)
        loss, acc = evaluate(mc, loader, device)
        losses.append(round(loss, 5))
        accs.append(round(acc, 4))
        log(f"    α={alpha:.1f}  loss={loss:.5f}  acc={acc:.4f}")

    baseline = min(losses[0], losses[-1])
    barrier  = max(losses) - baseline
    peak_a   = alphas[losses.index(max(losses))]
    log(f"  → barrier={barrier:.5f} nats  peak_α={peak_a}")
    return {
        "label": label,
        "alphas": alphas,
        "losses": losses,
        "accs": accs,
        "barrier_nats": round(barrier, 5),
        "baseline_nats": round(baseline, 5),
        "peak_alpha": peak_a,
        "endpoint_a_loss": losses[0],
        "endpoint_b_loss": losses[-1],
    }


def main():
    log("=" * 60)
    log("  Phase ResNet LMC — cross-precision basin isolation test")
    log("  CIFAR-10 SmallResNet: FP32(A) ↔ BF16(B) ↔ FP32(C)")
    log("=" * 60)

    # Download checkpoints
    for cond in ["A", "B", "C"]:
        dst = CKPT_DIR / f"{cond}_final.pt"
        if not dst.exists():
            log(f"Downloading {cond}_final.pt from GCS...")
            gsutil_dl(f"{GCS_BASE}/phase_resnet_restart/{cond}_final.pt", str(dst))
        else:
            log(f"  {cond}_final.pt already cached")

    val_loader = load_cifar10_val()
    device = torch.device("cpu")

    results = []

    # FP32(A) ↔ BF16(B) — key cross-precision pair
    results.append(lmc_barrier(
        "FP32(A) ↔ BF16(B)",
        CKPT_DIR / "A_final.pt",
        CKPT_DIR / "B_final.pt",
        val_loader, device,
    ))

    # FP32(A) ↔ FP32(C) — within-precision control (should be ~0, A=C)
    results.append(lmc_barrier(
        "FP32(A) ↔ FP32(C)",
        CKPT_DIR / "A_final.pt",
        CKPT_DIR / "C_final.pt",
        val_loader, device,
    ))

    # BF16(B) ↔ FP32(C) — same as A↔B by symmetry
    results.append(lmc_barrier(
        "BF16(B) ↔ FP32(C)",
        CKPT_DIR / "B_final.pt",
        CKPT_DIR / "C_final.pt",
        val_loader, device,
    ))

    log("\n" + "=" * 60)
    log("  SUMMARY")
    log("=" * 60)
    for r in results:
        log(f"  {r['label']:35s}  barrier={r['barrier_nats']:.5f} nats  peak_α={r['peak_alpha']}")

    out = {
        "experiment": "phase_resnet_lmc",
        "model": "SmallResNet (GroupNorm, GELU, ~1.2M params)",
        "dataset": "CIFAR-10 val (10K)",
        "checkpoints": "phase_resnet_restart A/B/C _final.pt (ep100)",
        "lmc": results,
        "phase17_gnn_ref": {"fp32_bf16_ev": 0.014, "interpretation": "GNN QM9, with restart"},
        "phase_xarch_ref": {"fp32_bf16_nats": 0.178, "interpretation": "Transformer TinyShakespeare, no restart"},
    }
    out_path = OUT_DIR / "results.json"
    out_path.write_text(json.dumps(out, indent=2))
    gsutil_cp(out_path, f"{GCS_BASE}/phase_resnet_lmc/results.json")
    log(f"\n  Results → {GCS_BASE}/phase_resnet_lmc/results.json")
    log("=== phase_resnet_lmc complete ===")


if __name__ == "__main__":
    main()
