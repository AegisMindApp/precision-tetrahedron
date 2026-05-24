#!/usr/bin/env python3
"""
phase_resnet_restart.py — Plateau-triggered warm restart on CIFAR-10.

Tests whether the plateau-triggered warm restart mechanism (patent AU2026903588),
established for molecular GNNs on QM9, generalises to computer vision.

Architecture: SmallResNet (residual blocks, GroupNorm, GELU) on CIFAR-10.
GroupNorm used throughout — BatchNorm requires cross-device all-reduce on TPU
which triggers recompilation on every batch in eager mode.

Conditions:
  A  — FP32 baseline,   100 epochs, CosineAnnealingLR, no restart
  B  — BF16 + plateau-triggered restart (patience=15)
  C  — FP32 + plateau-triggered restart (patience=15)

Expected outcomes matching the GNN results:
  1. Conditions B and C escape plateau in ≤3 epochs post-restart
  2. LMC barrier B_plateau↔B_post_restart > 0.02 nats (genuine inter-basin)
  3. LMC barrier C_plateau↔C_post_restart < B's barrier (FP32 basins shallower)
  4. Precision-agnostic escape: restart works in both BF16 and FP32

GCS output: gs://.../aegis_flashoptim/phase_resnet_restart/results.json
"""

import argparse, os, sys, json, subprocess, tarfile, pickle, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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

GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID     = os.environ.get("RUN_ID",     "aegis_flashoptim")
GCS_OUT    = f"{GCS_BUCKET}/{RUN_ID}/phase_resnet_restart"

OUT_DIR    = Path("/tmp/phase_resnet_restart")
DATA_DIR   = Path("/tmp/cifar10")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
N_EPOCHS      = 100
RESTART_EPOCHS = 30          # max epochs to train after restart
PATIENCE      = 15           # plateau patience
BATCH_SIZE    = 128
LR_INIT       = 1e-3
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 1.0
SEED          = 42
N_ALPHA       = 11
LOG_EVERY     = 10

# CIFAR-10 normalisation stats
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
CIFAR_STD  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

CIFAR_URL  = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

def log(msg): print(f"[ResNetRestart] {msg}", flush=True)
def gsutil_cp(src, dst): subprocess.run(["gsutil", "-q", "cp", str(src), dst], check=False)


# ── CIFAR-10 download and loading ─────────────────────────────────────────────

def load_cifar10():
    archive = DATA_DIR / "cifar-10-python.tar.gz"
    batches = DATA_DIR / "cifar-10-batches-py"
    if not batches.exists():
        if not archive.exists():
            log("Downloading CIFAR-10...")
            import urllib.request
            urllib.request.urlretrieve(CIFAR_URL, str(archive))
        log("Extracting CIFAR-10...")
        with tarfile.open(archive) as tf:
            tf.extractall(DATA_DIR)

    def unpickle(f):
        with open(f, "rb") as fh:
            return pickle.load(fh, encoding="bytes")

    xs, ys = [], []
    for i in range(1, 6):
        d = unpickle(batches / f"data_batch_{i}")
        xs.append(d[b"data"])
        ys.extend(d[b"labels"])
    X_train = np.concatenate(xs, axis=0).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y_train = np.array(ys, dtype=np.int64)

    d = unpickle(batches / "test_batch")
    X_val = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y_val = np.array(d[b"labels"], dtype=np.int64)

    # Normalise
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
    std  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)
    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std

    T_train = torch.from_numpy(X_train)
    T_val   = torch.from_numpy(X_val)
    y_train = torch.from_numpy(y_train)
    y_val   = torch.from_numpy(y_val)

    log(f"CIFAR-10 loaded: train={len(T_train)} val={len(T_val)}")
    return (TensorDataset(T_train, y_train),
            TensorDataset(T_val,   y_val))


def augment_batch(x):
    """Random horizontal flip + random 32×32 crop from 40×40 padded image."""
    B = x.shape[0]
    # Horizontal flip
    flip_mask = torch.rand(B) > 0.5
    x[flip_mask] = x[flip_mask].flip(-1)
    # Pad 4 each side, random crop
    x_pad = F.pad(x, (4, 4, 4, 4), mode="reflect")
    ox = torch.randint(0, 8, (B,))
    oy = torch.randint(0, 8, (B,))
    out = torch.stack([x_pad[i, :, oy[i]:oy[i]+32, ox[i]:ox[i]+32] for i in range(B)])
    return out


# ── Model ─────────────────────────────────────────────────────────────────────

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
        self.layer1 = ResBlock(32,  64,  stride=2)   # 32→16
        self.layer2 = ResBlock(64,  128, stride=2)   # 16→8
        self.layer3 = ResBlock(128, 256, stride=2)   # 8→4
        self.head   = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.head(self.layer3(self.layer2(self.layer1(self.stem(x)))))


def make_model(device, dtype=None):
    torch.manual_seed(SEED)
    m = SmallResNet().to(device)
    if dtype is not None:
        m = m.to(dtype)
    return m


# ── Plateau detector ──────────────────────────────────────────────────────────

class PlateauDetector:
    def __init__(self, patience, min_delta=5e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = float("inf")
        self.counter   = 0

    def step(self, loss):
        if loss < self.best - self.min_delta:
            self.best    = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ── Eval ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_loss_acc(model, loader, device):
    model.eval()
    total_loss, total_correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss    += F.cross_entropy(logits, y, reduction="sum").item()
        total_correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    if XLA_AVAILABLE and str(device) != "cpu":
        xm.mark_step()
    return total_loss / n, total_correct / n


# ── Natural cosine LR ──────────────────────────────────────────────────────────

def cosine_lr(t, T_max, eta_max, eta_min):
    """η(t) = ηmin + (ηmax-ηmin)/2 × (1 + cos(π t / T_max))"""
    return eta_min + (eta_max - eta_min) * (1 + math.cos(math.pi * t / T_max)) / 2


# ── Training with optional restart ────────────────────────────────────────────

def train_condition(label, device, train_ds, val_ds, use_restart, dtype=None):
    ckpt_plateau = OUT_DIR / f"{label}_plateau.pt"
    ckpt_final   = OUT_DIR / f"{label}_final.pt"

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=256,        shuffle=False, num_workers=0)

    model = make_model(device, dtype)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-5)
    pd    = PlateauDetector(PATIENCE) if use_restart else None

    log(f"\n{'='*55}")
    log(f"  Condition {label}  device={device}  dtype={dtype or 'fp32'}  restart={use_restart}")
    log(f"{'='*55}")

    trajectory = []
    plateau_ep = None
    restart_ep = None
    post_restart_trajectory = []

    for ep in range(1, N_EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x = augment_batch(x)
            x, y = x.to(device), y.to(device)
            if dtype is not None:
                x = x.to(dtype)
            opt.zero_grad()
            F.cross_entropy(model(x).float(), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            if XLA_AVAILABLE:
                xm.optimizer_step(opt)
            else:
                opt.step()
        sched.step()
        if XLA_AVAILABLE:
            xm.mark_step()

        if ep % LOG_EVERY == 0 or ep == N_EPOCHS:
            vl, acc = eval_loss_acc(model, val_loader, device)
            log(f"  {label} ep{ep:3d}  val_loss={vl:.4f}  acc={acc:.3f}  lr={sched.get_last_lr()[0]:.2e}")
            trajectory.append({"epoch": ep, "val_loss": round(vl, 5), "acc": round(acc, 4)})

            if use_restart and plateau_ep is None and pd is not None and pd.step(vl):
                plateau_ep = ep
                log(f"  *** PLATEAU detected at ep{ep} (patience={PATIENCE}) ***")
                torch.save({k: v.cpu().float() for k, v in model.state_dict().items()}, ckpt_plateau)
                gsutil_cp(ckpt_plateau, f"{GCS_OUT}/{label}_plateau.pt")

    # Save final baseline
    torch.save({k: v.cpu().float() for k, v in model.state_dict().items()}, ckpt_final)
    gsutil_cp(ckpt_final, f"{GCS_OUT}/{label}_final.pt")
    vl_final, acc_final = eval_loss_acc(model, val_loader, device)
    log(f"  {label} baseline done: val_loss={vl_final:.4f}  acc={acc_final:.3f}")

    if not use_restart or plateau_ep is None:
        log(f"  No plateau detected — no restart performed")
        return {"label": label, "trajectory": trajectory, "final_val_loss": vl_final,
                "final_acc": acc_final, "plateau_epoch": None, "restart_epoch": None,
                "post_restart_trajectory": []}

    # ── Warm restart ──────────────────────────────────────────────────────────
    log(f"\n  Warm restart from ep{plateau_ep} checkpoint...")
    restart_lr = cosine_lr(plateau_ep, N_EPOCHS, LR_INIT, 1e-5)
    log(f"  Natural cosine LR at ep{plateau_ep}: {restart_lr:.2e}  (self-calibrating restart LR)")

    model2 = make_model(device, dtype)
    sd     = torch.load(ckpt_plateau, map_location="cpu")
    model2.load_state_dict(sd)
    model2.to(device)
    opt2   = torch.optim.AdamW(model2.parameters(), lr=restart_lr, weight_decay=WEIGHT_DECAY)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=RESTART_EPOCHS, eta_min=1e-5)
    restart_ep = plateau_ep

    for ep in range(1, RESTART_EPOCHS + 1):
        model2.train()
        for x, y in train_loader:
            x = augment_batch(x)
            x, y = x.to(device), y.to(device)
            if dtype is not None:
                x = x.to(dtype)
            opt2.zero_grad()
            F.cross_entropy(model2(x).float(), y).backward()
            nn.utils.clip_grad_norm_(model2.parameters(), GRAD_CLIP)
            if XLA_AVAILABLE:
                xm.optimizer_step(opt2)
            else:
                opt2.step()
        sched2.step()
        if XLA_AVAILABLE:
            xm.mark_step()

        vl, acc = eval_loss_acc(model2, val_loader, device)
        log(f"  {label} restart ep{ep:2d}  val_loss={vl:.4f}  acc={acc:.3f}  lr={sched2.get_last_lr()[0]:.2e}")
        post_restart_trajectory.append({"restart_ep": ep, "val_loss": round(vl, 5), "acc": round(acc, 4)})

        if ep == 3:
            # Save ep+3 checkpoint for LMC
            torch.save({k: v.cpu().float() for k, v in model2.state_dict().items()},
                       OUT_DIR / f"{label}_restart_ep3.pt")
            gsutil_cp(OUT_DIR / f"{label}_restart_ep3.pt", f"{GCS_OUT}/{label}_restart_ep3.pt")

    return {"label": label, "trajectory": trajectory, "final_val_loss": vl_final,
            "final_acc": acc_final, "plateau_epoch": plateau_ep, "restart_epoch": restart_ep,
            "restart_lr": restart_lr, "post_restart_trajectory": post_restart_trajectory}


# ── LMC ───────────────────────────────────────────────────────────────────────

@torch.no_grad()
def lmc_barrier(label, path_a, path_b, val_loader, device):
    sd_a  = torch.load(path_a, map_location="cpu")
    sd_b  = torch.load(path_b, map_location="cpu")
    cpu   = torch.device("cpu")
    model = make_model(cpu)
    alphas = [round(i / (N_ALPHA - 1), 2) for i in range(N_ALPHA)]
    losses = []
    for alpha in alphas:
        interp = {k: (1 - alpha) * sd_a[k].float() + alpha * sd_b[k].float()
                  for k in sd_a if k in sd_b}
        model.load_state_dict(interp, strict=False)
        vl, _ = eval_loss_acc(model, val_loader, cpu)
        losses.append(round(vl, 5))
        log(f"    α={alpha:.1f}  loss={vl:.5f}")
    baseline = min(losses[0], losses[-1])
    barrier  = round(max(losses) - baseline, 5)
    peak_a   = alphas[losses.index(max(losses))]
    log(f"  → {label}: barrier={barrier:.5f} nats  peak_α={peak_a:.1f}")
    return {"label": label, "alphas": alphas, "losses": losses,
            "barrier_nats": barrier, "baseline_nats": baseline, "peak_alpha": peak_a}


# ── Main ──────────────────────────────────────────────────────────────────────

def _run_condition_subprocess(cond: str, extra_env: dict | None = None) -> dict:
    """Spawn an isolated subprocess to train one condition.

    Each condition needs its own process so the TPU device is fully released
    between conditions — the parent coordinator never acquires the TPU.
    XLA_USE_BF16=1 must also be set before torch_xla import, so BF16
    conditions (B) additionally require the env var via extra_env.
    """
    out_json = OUT_DIR / f"{cond}_result.json"
    out_json.unlink(missing_ok=True)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    log(f"Launching Condition {cond} subprocess...")
    ret = subprocess.run(
        [sys.executable, __file__, "--condition", cond, "--output-json", str(out_json)],
        env=env,
    )
    if ret.returncode != 0:
        raise RuntimeError(f"Condition {cond} subprocess exited {ret.returncode}")
    return json.loads(out_json.read_text())


def main():
    # ── Subprocess mode (single condition) ───────────────────────────────────
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--condition", default=None)
    parser.add_argument("--output-json", default=None)
    args, _ = parser.parse_known_args()

    if args.condition is not None:
        device = xm.xla_device() if XLA_AVAILABLE else torch.device("cpu")
        train_ds, val_ds = load_cifar10()
        result = train_condition(args.condition, device, train_ds, val_ds,
                                 use_restart=(args.condition != "A"))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(result))
        return

    # ── Orchestrator mode ─────────────────────────────────────────────────────
    # The coordinator never acquires the TPU device — each condition runs in
    # its own subprocess so the device is fully released between conditions.
    log("=" * 65)
    log("  Phase ResNet-Restart — CV generalisation of warm restart")
    log("  Architecture: SmallResNet (GroupNorm) on CIFAR-10")
    log("=" * 65)

    val_ds = load_cifar10()[1]
    val_loader_cpu = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    # ── Three conditions (sequential subprocesses) ────────────────────────────
    results = {}

    # Condition A: FP32 baseline, no restart
    results["A"] = _run_condition_subprocess("A")

    # Condition B: BF16 + plateau-triggered restart (XLA_USE_BF16=1 in subprocess)
    results["B"] = _run_condition_subprocess("B", extra_env={"XLA_USE_BF16": "1"})

    # Condition C: FP32 + plateau-triggered restart
    results["C"] = _run_condition_subprocess("C")

    # ── LMC pairs ────────────────────────────────────────────────────────────
    log("\nRunning LMC pairs (CPU)...")
    lmc_results = []
    for cond in ["B", "C"]:
        plateau_ckpt = OUT_DIR / f"{cond}_plateau.pt"
        restart_ckpt = OUT_DIR / f"{cond}_restart_ep3.pt"
        if plateau_ckpt.exists() and restart_ckpt.exists():
            log(f"\n  LMC: {cond}_plateau ↔ {cond}_restart_ep3")
            lmc_results.append(lmc_barrier(
                f"{cond}: plateau ↔ restart+3",
                plateau_ckpt, restart_ckpt,
                val_loader_cpu, torch.device("cpu")
            ))
        else:
            log(f"  Skipping LMC for {cond}: checkpoint(s) missing "
                f"(plateau={plateau_ckpt.exists()}, restart={restart_ckpt.exists()})")

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n" + "=" * 65)
    log("  SUMMARY")
    log("=" * 65)
    for cond, r in results.items():
        p = r["plateau_epoch"]
        log(f"  Cond {cond}: val_loss={r['final_val_loss']:.4f}  acc={r['final_acc']:.3f}"
            f"  plateau={'ep'+str(p) if p else 'none'}")
    log(f"\n  GNN Phase 7 LMC reference:  BF16 plateau↔restart = 1.447 eV (GNN, molecular)")
    log(f"  GNN Phase 7 LMC reference:  FP32 within-basin   = 0.005 eV")
    log(f"\n  CIFAR-10 SmallResNet LMC:")
    for r in lmc_results:
        log(f"  {r['label']:40s}  {r['barrier_nats']:.5f} nats  peak_α={r['peak_alpha']:.1f}")

    for r in results.values():
        r["post_restart_trajectory"] = r["post_restart_trajectory"][:10]  # cap size

    out_data = {
        "experiment": "phase_resnet_restart",
        "model":      "SmallResNet (GroupNorm, GELU, ~1.2M params)",
        "dataset":    "CIFAR-10 (50K train / 10K test)",
        "conditions": results,
        "lmc":        lmc_results,
        "gnn_ref":    {"bf16_restart_barrier_ev": 1.447, "fp32_intra_basin_ev": 0.005},
    }
    out = OUT_DIR / "results.json"
    out.write_text(json.dumps(out_data, indent=2))
    gsutil_cp(out, f"{GCS_OUT}/results.json")
    log(f"\n  Results → {GCS_OUT}/results.json")
    log("=== phase_resnet_restart complete ===")


if __name__ == "__main__":
    main()
