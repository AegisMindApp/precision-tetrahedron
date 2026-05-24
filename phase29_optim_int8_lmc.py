#!/usr/bin/env python3
"""
phase29_optim_int8_lmc.py
--------------------------
Optimizer-state INT8 LMC — trains GPT char-LM with INT8-snapped Adam moments.

Two training runs:
  1. fp32_clean: standard FP32 Adam, no snapping
  2. int8_snap:  FP32 Adam but snap exp_avg/exp_avg_sq to INT8 every 10 epochs

LMC (11 α points) on CPU between the two checkpoints.

Architecture: same as phase_xarch_lmc.py (6L/256d/8H, TinyShakespeare, 80 epochs)

GCS output: gs://.../aegis_flashoptim/phase29_optim_int8_lmc/results.json
"""

import os, sys, json, math, subprocess, urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from notify import notify, heartbeat

GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID     = os.environ.get("RUN_ID", "aegis_flashoptim")
GCS_P      = f"{GCS_BUCKET}/{RUN_ID}/phase29_optim_int8_lmc"

OUT_DIR  = Path("/tmp/phase29_optim_int8_lmc")
DATA_DIR = Path("/tmp/shakespeare")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Same arch as phase_xarch_lmc.py ──────────────────────────────────────────
N_EMBD       = 256
N_HEAD       = 8
N_LAYER      = 6
BLOCK_SIZE   = 256
DROPOUT      = 0.1
N_EPOCHS     = 80
BATCH_SIZE   = 64
LR_INIT      = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP    = 1.0
SEED         = 42
N_ALPHA      = 11
LOG_EVERY    = 10
INT8_SNAP_EVERY = 10  # snap optimizer moments every N epochs

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)


def log(msg):
    print(f"[Phase29] {msg}", flush=True)


def gsutil_cp(src, dst):
    subprocess.run(["gsutil", "-q", "cp", str(src), dst], check=False)


def gcs_exists(gcs_path):
    return subprocess.run(["gsutil", "-q", "stat", gcs_path],
                          capture_output=True).returncode == 0


# ── Data ──────────────────────────────────────────────────────────────────────

def load_shakespeare():
    path = DATA_DIR / "input.txt"
    if not path.exists():
        log("Downloading TinyShakespeare...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, str(path))
    text  = path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi  = {c: i for i, c in enumerate(chars)}
    data  = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n     = int(0.9 * len(data))
    return data[:n], data[n:], len(chars)


def make_batches(data, batch_size, seq_len):
    stride = batch_size * seq_len
    n  = (len(data) - 1) // stride
    x  = data[:n * stride].view(batch_size, -1)
    y  = data[1:n * stride + 1].view(batch_size, -1)
    return list(zip(x.split(seq_len, dim=1), y.split(seq_len, dim=1)))


# ── Model ─────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv  = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.proj = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.drop = nn.Dropout(DROPOUT)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)).view(1, 1, BLOCK_SIZE, BLOCK_SIZE)
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        h, d = N_HEAD, C // N_HEAD
        q = q.view(B, T, h, d).transpose(1, 2)
        k = k.view(B, T, h, d).transpose(1, 2)
        v = v.view(B, T, h, d).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(d)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, -1e4)
        att = self.drop(F.softmax(att.float(), dim=-1).to(att.dtype))
        return self.proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln2  = nn.LayerNorm(N_EMBD)
        self.ffn  = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTCharLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok    = nn.Embedding(vocab_size, N_EMBD)
        self.pos    = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.drop   = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f   = nn.LayerNorm(N_EMBD)
        self.head   = nn.Linear(N_EMBD, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


def make_model(vocab_size):
    torch.manual_seed(SEED)
    return GPTCharLM(vocab_size)


@torch.no_grad()
def eval_loss(model, batches):
    model.eval()
    total, n = 0.0, 0
    for x, y in batches:
        logits = model(x)
        total += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        ).item()
        n += y.numel()
    return total / n if n > 0 else float("inf")


# ── INT8 snapping ─────────────────────────────────────────────────────────────

def snap_optimizer_moments_to_int8(optimizer):
    """
    Snap Adam exp_avg (first moment) and exp_avg_sq (second moment) to INT8.
    Skips parameters where moments are too small (would cause Adam denom explosion).
    """
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            if 'exp_avg' in state:
                m = state['exp_avg']
                max_val = m.abs().max().item()
                if not math.isfinite(max_val) or max_val < 1e-10:
                    continue  # skip: moments too small, Adam denom would explode
                scale = max_val / 127.0
                m.div_(scale).round_().clamp_(-127, 127).mul_(scale)
            if 'exp_avg_sq' in state:
                m = state['exp_avg_sq']
                max_val = m.max().item()
                if not math.isfinite(max_val) or max_val < 1e-10:
                    continue
                scale = max_val / 255.0
                m.div_(scale).round_().clamp_(0, 255).mul_(scale)


# ── Training ──────────────────────────────────────────────────────────────────

def run_train_only():
    """Called in subprocess via --train-only flag."""
    int8_snap = os.environ.get("INT8_SNAP", "0") == "1"
    precision = "int8_snap" if int8_snap else "fp32_clean"

    ckpt     = OUT_DIR / f"{precision}_ep{N_EPOCHS}.pt"
    gcs_ckpt = f"{GCS_P}/{ckpt.name}"

    if ckpt.exists():
        log(f"[train-only] {precision}: on disk — skip")
        return
    if gcs_exists(gcs_ckpt):
        log(f"[train-only] {precision}: downloading from GCS")
        subprocess.run(["gsutil", "-q", "cp", gcs_ckpt, str(ckpt)], check=False)
        return

    train_data, val_data, vocab_size = load_shakespeare()
    (OUT_DIR / "vocab_size.txt").write_text(str(vocab_size))

    train_batches = make_batches(train_data, BATCH_SIZE, BLOCK_SIZE)
    val_batches   = make_batches(val_data,   BATCH_SIZE, BLOCK_SIZE)

    log(f"[train-only] {precision}: training  int8_snap={int8_snap}")

    m   = make_model(vocab_size)
    opt = torch.optim.AdamW(m.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-5)

    for ep in range(1, N_EPOCHS + 1):
        m.train()
        for x, y in train_batches:
            opt.zero_grad()
            F.cross_entropy(m(x).reshape(-1, vocab_size), y.reshape(-1)).backward()
            nn.utils.clip_grad_norm_(m.parameters(), GRAD_CLIP)
            opt.step()
        sch.step()

        if int8_snap and ep % INT8_SNAP_EVERY == 0:
            snap_optimizer_moments_to_int8(opt)
            log(f"  ep{ep:3d}: snapped optimizer moments to INT8")

        if ep % LOG_EVERY == 0:
            vl = eval_loss(m, val_batches)
            log(f"  {precision} ep{ep:3d}  val_loss={vl:.4f}  bpc={vl/math.log(2):.3f}"
                f"  lr={opt.param_groups[0]['lr']:.2e}")

    vl = eval_loss(m, val_batches)
    torch.save({k: v.cpu().clone() for k, v in m.state_dict().items()}, ckpt)
    gsutil_cp(ckpt, gcs_ckpt)
    log(f"[train-only] {precision}: done  val_loss={vl:.4f}  bpc={vl/math.log(2):.3f}")


# ── LMC ───────────────────────────────────────────────────────────────────────

def lmc_barrier(label, p0, p1, vocab_size, val_batches):
    sd0 = torch.load(p0, map_location="cpu")
    sd1 = torch.load(p1, map_location="cpu")
    m   = make_model(vocab_size)
    alphas = [round(i / (N_ALPHA - 1), 2) for i in range(N_ALPHA)]
    losses = []

    for alpha in alphas:
        interp = {k: (1 - alpha) * sd0[k].float() + alpha * sd1[k].float()
                  for k in sd0 if k in sd1}
        m.load_state_dict(interp, strict=False)
        loss = eval_loss(m, val_batches)
        losses.append(round(loss, 5))
        log(f"  {label} α={alpha:.1f}  loss={loss:.5f}  bpc={loss/math.log(2):.4f}")

    base    = min(losses[0], losses[-1])
    barrier = round(max(losses) - base, 5)
    peak_a  = alphas[losses.index(max(losses))]
    log(f"  → {label} barrier={barrier:.5f} nats  bpc={barrier/math.log(2):.5f}  peak_α={peak_a}")
    return {"label": label, "alphas": alphas, "losses": losses,
            "barrier_nats": barrier, "barrier_bpc": round(barrier / math.log(2), 5),
            "baseline_nats": base, "peak_alpha": peak_a}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 65)
    log("  Phase 29 — Optimizer INT8 Adam Moments LMC")
    log("=" * 65)

    notify("PHASE_START", "[Phase29] Optimizer INT8 Adam LMC", data={})

    # ── Train fp32_clean ──────────────────────────────────────────────────────
    for precision, int8_flag in [("fp32_clean", "0"), ("int8_snap", "1")]:
        ckpt     = OUT_DIR / f"{precision}_ep{N_EPOCHS}.pt"
        gcs_ckpt = f"{GCS_P}/{ckpt.name}"
        if ckpt.exists():
            log(f"{precision}: on disk — skip")
            continue
        if gcs_exists(gcs_ckpt):
            log(f"{precision}: GCS — downloading")
            subprocess.run(["gsutil", "-q", "cp", gcs_ckpt, str(ckpt)], check=False)
            continue

        log(f"\n{'='*50}\nLaunching {precision} subprocess\n{'='*50}")
        env = {**os.environ}
        env["INT8_SNAP"] = int8_flag
        res = subprocess.run([sys.executable, __file__, "--train-only"], env=env)
        if res.returncode != 0:
            log(f"WARNING: {precision} subprocess exited {res.returncode}")

    # ── Load data and compute LMC ─────────────────────────────────────────────
    _, val_data, vocab_size = load_shakespeare()
    vs_path = OUT_DIR / "vocab_size.txt"
    if vs_path.exists():
        vocab_size = int(vs_path.read_text().strip())
    val_batches = make_batches(val_data, BATCH_SIZE, BLOCK_SIZE)

    p_fp32  = OUT_DIR / f"fp32_clean_ep{N_EPOCHS}.pt"
    p_int8  = OUT_DIR / f"int8_snap_ep{N_EPOCHS}.pt"

    if not p_fp32.exists() or not p_int8.exists():
        log("ERROR: one or both checkpoints missing — cannot compute LMC")
        missing = []
        if not p_fp32.exists(): missing.append("fp32_clean")
        if not p_int8.exists(): missing.append("int8_snap")
        log(f"  Missing: {missing}")
        sys.exit(1)

    # Report final val losses
    m = make_model(vocab_size)
    m.load_state_dict(torch.load(p_fp32, map_location="cpu"))
    fp32_vl = eval_loss(m, val_batches)
    fp32_bpc = fp32_vl / math.log(2)

    m.load_state_dict(torch.load(p_int8, map_location="cpu"))
    int8_vl = eval_loss(m, val_batches)
    int8_bpc = int8_vl / math.log(2)

    log(f"\nfp32_clean: val_loss={fp32_vl:.4f}  bpc={fp32_bpc:.4f}")
    log(f"int8_snap:  val_loss={int8_vl:.4f}  bpc={int8_bpc:.4f}")

    log("\nComputing LMC: FP32-clean ↔ INT8-snap...")
    lmc = lmc_barrier("FP32-clean ↔ INT8-snap", p_fp32, p_int8, vocab_size, val_batches)

    # Interpretation
    if lmc["barrier_nats"] < 0.005:
        interpretation = (
            f"Near-zero LMC barrier ({lmc['barrier_nats']:.5f} nats) between FP32-clean "
            f"and INT8-snap optimiser-moment models. INT8-snapped Adam converges to the "
            f"same loss basin — optimizer quantisation noise is negligible at this model size."
        )
    elif lmc["barrier_nats"] < 0.05:
        interpretation = (
            f"Low LMC barrier ({lmc['barrier_nats']:.5f} nats). INT8 moment snapping "
            f"introduces a modest perturbation that lands in an overlapping but slightly "
            f"distinct basin. Barrier bpc={lmc['barrier_bpc']:.5f}."
        )
    else:
        interpretation = (
            f"Large LMC barrier ({lmc['barrier_nats']:.5f} nats). INT8 moment snapping "
            f"drives training into a distinct loss basin. Optimizer-state quantisation "
            f"significantly impacts the optimisation trajectory at this model scale. "
            f"bpc={lmc['barrier_bpc']:.5f}."
        )
    log(f"\nInterpretation: {interpretation}")

    results = {
        "experiment":         "phase29_optim_int8_lmc",
        "model":              {"n_layer": N_LAYER, "n_embd": N_EMBD, "n_head": N_HEAD},
        "fp32_final_val_loss": round(fp32_vl, 5),
        "fp32_final_bpc":      round(fp32_bpc, 5),
        "int8_final_val_loss": round(int8_vl, 5),
        "int8_final_bpc":      round(int8_bpc, 5),
        "lmc_barrier_nats":    lmc["barrier_nats"],
        "lmc_curve":           lmc,
        "interpretation":      interpretation,
    }

    out = OUT_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    gsutil_cp(out, f"{GCS_P}/results.json")
    log(f"\nResults → GCS phase29_optim_int8_lmc/results.json")
    log("=== phase29 complete ===")

    notify("PHASE_COMPLETE", "[Phase29] INT8 Adam LMC done",
           data={"barrier_nats": lmc["barrier_nats"],
                 "fp32_bpc": round(fp32_bpc, 4),
                 "int8_bpc": round(int8_bpc, 4)})


if __name__ == "__main__":
    if "--train-only" in sys.argv:
        run_train_only()
    else:
        main()
