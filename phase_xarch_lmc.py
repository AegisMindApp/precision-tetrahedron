#!/usr/bin/env python3
"""
phase_xarch_lmc.py  —  Cross-architecture LMC precision test
-------------------------------------------------------------
Tests whether the exponent-range isosceles LMC triangle established for GNNs
in Phase 17 (fp32↔bf16: 0.014 eV, fp32↔fp16: 0.149 eV, bf16↔fp16: 0.150 eV,
10.5× ratio, isosceles) generalises to causal language models.

Architecture: GPT-style transformer  6L / 256d / 8H / char-level
Dataset:      TinyShakespeare (~1M chars, vocab≈65)
Metric:       validation cross-entropy (nats); barrier = max(CE path) - min(endpoint)

Triangle prediction (exponent-range hypothesis):
  FP32 ↔ BF16  →  small  (shared 8-bit exponent)
  FP32 ↔ FP16  →  large  (FP16: 5-bit exponent)
  BF16 ↔ FP16  →  large ≈ FP32↔FP16  (isosceles triangle)

Precision via subprocess: each of the three training runs is a child process
with XLA_USE_BF16 / XLA_USE_F16 set before torch_xla import. The orchestrator
never claims the XLA device; LMC runs on CPU.

GCS output: gs://.../aegis_flashoptim/phase_xarch_lmc/results.json
"""

import os, sys, json, math, urllib.request, subprocess
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
    try:
        import torch_xla.experimental as _xe
        _xe.eager_mode(True)
    except Exception:
        pass
except ImportError:
    XLA_AVAILABLE = False

GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID     = os.environ.get("RUN_ID",     "aegis_flashoptim")
GCS_P      = f"{GCS_BUCKET}/{RUN_ID}/phase_xarch_lmc"

OUT_DIR  = Path("/tmp/phase_xarch_lmc")
DATA_DIR = Path("/tmp/shakespeare")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
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

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)

def log(msg): print(f"[XarchLMC] {msg}", flush=True)
def gsutil_cp(src, dst):
    subprocess.run(["gsutil", "-q", "cp", str(src), dst], check=False)


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


def make_model(vocab_size, device):
    torch.manual_seed(SEED)
    return GPTCharLM(vocab_size).to(device)


def cast_fp16_cpu(model):
    """Cast to float16 on CPU; keep LayerNorm in FP32 for numerical stability."""
    model = model.to(torch.float16)
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            m.float()
    return model


@torch.no_grad()
def eval_loss_cpu(model, batches):
    """Eval for CPU FP16 model — logits upcast to FP32 before cross_entropy."""
    model.eval()
    total, n = 0.0, 0
    for x, y in batches:
        logits = model(x)
        total += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction="sum"
        ).item()
        n += y.numel()
    return total / n if n > 0 else float("inf")


def train_fp16_cpu(vocab_size, train_batches, val_batches):
    """Genuine FP16 on CPU — XLA_USE_F16 is a no-op on v6e TPU hardware."""
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    ckpt = OUT_DIR / f"fp16_ep{N_EPOCHS}.pt"
    gcs  = f"{GCS_P}/{ckpt.name}"
    if ckpt.exists():
        log("fp16(CPU): on disk — skip"); return
    if subprocess.run(["gsutil", "-q", "stat", gcs], capture_output=True).returncode == 0:
        log("fp16(CPU): GCS — downloading")
        subprocess.run(["gsutil", "-q", "cp", gcs, str(ckpt)], check=False); return

    log("fp16(CPU): training on CPU with explicit float16 (LayerNorm stays FP32)")
    torch.manual_seed(SEED)
    m   = cast_fp16_cpu(GPTCharLM(vocab_size))
    # Lower LR — FP16 max representable is 65504; 3e-4 causes overflow
    fp16_lr = LR_INIT * (2 / 3)
    opt = torch.optim.AdamW(m.parameters(), lr=fp16_lr, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-5)

    for ep in range(1, N_EPOCHS + 1):
        m.train()
        for x, y in train_batches:
            opt.zero_grad()
            loss = F.cross_entropy(m(x).reshape(-1, vocab_size).float(), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), GRAD_CLIP)
            opt.step()
        sch.step()
        if ep % LOG_EVERY == 0:
            vl = eval_loss_cpu(m, val_batches)
            log(f"    fp16(CPU) ep{ep:3d}  val_loss={vl:.4f}  bpc={vl/math.log(2):.3f}  lr={opt.param_groups[0]['lr']:.2e}")

    vl = eval_loss_cpu(m, val_batches)
    torch.save({k: v.cpu().float() for k, v in m.state_dict().items()}, ckpt)
    gsutil_cp(ckpt, gcs)
    log(f"fp16(CPU): final val_loss={vl:.4f}  bpc={vl/math.log(2):.3f}  → {ckpt.name}")


def get_device():
    if XLA_AVAILABLE:
        return xm.xla_device()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Training / evaluation ─────────────────────────────────────────────────────

def eval_loss(model, batches, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total += F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
            ).item()
            n += y.numel()
    if XLA_AVAILABLE and str(device) != "cpu":
        xm.mark_step()
    return total / n if n > 0 else float("inf")


def train_one_precision(precision, vocab_size, train_batches, val_batches):
    ckpt = OUT_DIR / f"{precision}_ep{N_EPOCHS}.pt"
    gcs  = f"{GCS_P}/{ckpt.name}"

    if not ckpt.exists():
        if subprocess.run(["gsutil", "-q", "stat", gcs], capture_output=True).returncode == 0:
            log(f"  {precision}: GCS ckpt found → downloading")
            subprocess.run(["gsutil", "-q", "cp", gcs, str(ckpt)], check=False)

    if ckpt.exists():
        dev = get_device()
        m   = make_model(vocab_size, dev)
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        m.to(dev)
        vl = eval_loss(m, val_batches, dev)
        log(f"  {precision}: resumed  val_loss={vl:.4f}  bpc={vl/math.log(2):.3f}")
        return {"val_loss": round(vl, 5), "bpc": round(vl/math.log(2), 4), "resumed": True}

    dev = get_device()
    log(f"  {precision}: training on {dev}")
    m   = make_model(vocab_size, dev)
    opt = torch.optim.AdamW(m.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-5)

    for ep in range(1, N_EPOCHS + 1):
        m.train()
        for x, y in train_batches:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            F.cross_entropy(m(x).view(-1, vocab_size), y.view(-1)).backward()
            nn.utils.clip_grad_norm_(m.parameters(), GRAD_CLIP)
            if XLA_AVAILABLE:
                xm.optimizer_step(opt)
            else:
                opt.step()
        sch.step()
        if XLA_AVAILABLE:
            xm.mark_step()
        if ep % LOG_EVERY == 0:
            vl = eval_loss(m, val_batches, dev)
            log(f"    {precision} ep{ep:3d}  val_loss={vl:.4f}  bpc={vl/math.log(2):.3f}  lr={opt.param_groups[0]['lr']:.2e}")

    vl = eval_loss(m, val_batches, dev)
    torch.save({k: v.cpu().clone() for k, v in m.state_dict().items()}, ckpt)
    gsutil_cp(ckpt, gcs)
    log(f"  {precision}: final val_loss={vl:.4f}  bpc={vl/math.log(2):.3f}")
    return {"val_loss": round(vl, 5), "bpc": round(vl/math.log(2), 4)}


def lmc_barrier(label, p0, p1, vocab_size, val_batches):
    sd0 = torch.load(p0, map_location="cpu")
    sd1 = torch.load(p1, map_location="cpu")
    dev = torch.device("cpu")
    m   = make_model(vocab_size, dev)
    alphas = [round(i / (N_ALPHA - 1), 2) for i in range(N_ALPHA)]
    losses = []
    for alpha in alphas:
        interp = {k: (1-alpha)*sd0[k].float() + alpha*sd1[k].float()
                  for k in sd0 if k in sd1}
        m.load_state_dict(interp, strict=False)
        losses.append(round(eval_loss(m, val_batches, dev), 5))
        log(f"    α={alpha:.1f}  loss={losses[-1]:.5f}  bpc={losses[-1]/math.log(2):.4f}")
    base    = min(losses[0], losses[-1])
    barrier = round(max(losses) - base, 5)
    peak_a  = alphas[losses.index(max(losses))]
    log(f"  → barrier={barrier:.5f} nats  bpc={barrier/math.log(2):.5f}  peak_α={peak_a}")
    return {"label": label, "alphas": alphas, "losses": losses,
            "barrier_nats": barrier, "barrier_bpc": round(barrier/math.log(2), 5),
            "baseline_nats": base, "peak_alpha": peak_a}


# ── Entry points ──────────────────────────────────────────────────────────────

def run_train_only():
    """Invoked in subprocess — XLA env vars already set before this import."""
    train_data, val_data, vocab_size = load_shakespeare()
    (OUT_DIR / "vocab_size.txt").write_text(str(vocab_size))
    train_batches = make_batches(train_data, BATCH_SIZE, BLOCK_SIZE)
    val_batches   = make_batches(val_data,   BATCH_SIZE, BLOCK_SIZE)
    if os.environ.get("XLA_USE_F16") == "1":
        # v6e silently promotes FP16→BF16 in XLA; redirect to genuine CPU float16
        log("[train-only] FP16 → CPU path (XLA_USE_F16 no-op on v6e)")
        train_fp16_cpu(vocab_size, train_batches, val_batches)
        return
    precision = "bf16" if os.environ.get("XLA_USE_BF16") == "1" else "fp32"
    log(f"[train-only] precision={precision}")
    result = train_one_precision(precision, vocab_size, train_batches, val_batches)
    log(f"[train-only] done: {result}")


def main():
    log("=" * 65)
    log("  Phase xarch-LMC — GPT char-level on TinyShakespeare")
    log("  Hypothesis: exponent-range isosceles triangle is arch-general")
    log("=" * 65)

    # ── 1. Train each precision in a child subprocess ─────────────────────
    for prec in ["fp32", "bf16", "fp16"]:
        ckpt = OUT_DIR / f"{prec}_ep{N_EPOCHS}.pt"
        gcs  = f"{GCS_P}/{ckpt.name}"
        if ckpt.exists():
            log(f"{prec}: on disk — skip"); continue
        if subprocess.run(["gsutil", "-q", "stat", gcs], capture_output=True).returncode == 0:
            log(f"{prec}: GCS — downloading")
            subprocess.run(["gsutil", "-q", "cp", gcs, str(ckpt)], check=False)
            continue

        log(f"\n{'='*50}\nLaunching {prec} subprocess\n{'='*50}")
        env = {**os.environ}
        env.pop("XLA_USE_BF16", None); env.pop("XLA_USE_F16", None)
        if prec == "bf16":
            env["XLA_USE_BF16"] = "1"
        elif prec == "fp16":
            env["XLA_USE_F16"] = "1"
            env["OMP_NUM_THREADS"] = "4"   # CPU FP16 emulation: cap threads
            env["MKL_NUM_THREADS"] = "4"
        res = subprocess.run([sys.executable, __file__, "--train-only"], env=env)
        if res.returncode != 0:
            log(f"WARNING: {prec} subprocess exited {res.returncode}")

    # ── 2. Load data for LMC (CPU) ────────────────────────────────────────
    _, val_data, vocab_size = load_shakespeare()
    vs_path = OUT_DIR / "vocab_size.txt"
    if vs_path.exists():
        vocab_size = int(vs_path.read_text().strip())
    val_batches = make_batches(val_data, BATCH_SIZE, BLOCK_SIZE)

    # ── 3. LMC pairs ──────────────────────────────────────────────────────
    log("\nRunning LMC pairs (CPU)...")
    p = {prec: OUT_DIR / f"{prec}_ep{N_EPOCHS}.pt" for prec in ["fp32","bf16","fp16"]}
    lmc_results = []
    for label, k0, k1 in [
        ("FP32 ↔ BF16",        "fp32", "bf16"),
        ("FP32 ↔ FP16",        "fp32", "fp16"),
        ("BF16 ↔ FP16  (KEY)", "bf16", "fp16"),
    ]:
        if not p[k0].exists() or not p[k1].exists():
            log(f"  Skipping {label}: ckpt missing"); continue
        log(f"\n  LMC: {label}")
        lmc_results.append(lmc_barrier(label, p[k0], p[k1], vocab_size, val_batches))

    # ── 4. Report ─────────────────────────────────────────────────────────
    log("\n" + "=" * 65)
    log("  TRANSFORMER LMC RESULTS")
    for r in lmc_results:
        log(f"  {r['label']:35s}  {r['barrier_nats']:.5f} nats  ({r['barrier_bpc']:.5f} bpc)")

    interpretation = "Incomplete — some checkpoints missing"
    if len(lmc_results) == 3:
        b_fb, b_fp, b_bp = [r["barrier_nats"] for r in lmc_results]
        ratio       = b_fp / (b_fb + 1e-8)
        iso_ratio   = b_bp / (b_fp + 1e-8)
        isosceles   = abs(b_fp - b_bp) / max(b_fp, b_bp, 1e-8) < 0.20
        log(f"\n  FP32↔FP16 / FP32↔BF16 : {ratio:.1f}×  (GNN Phase17: 10.5×)")
        log(f"  BF16↔FP16 / FP32↔FP16 : {iso_ratio:.3f}  (isosceles if ≈1.0)")
        log(f"  Triangle isosceles     : {'YES' if isosceles else 'NO'}")
        interpretation = (
            f"Isosceles triangle in transformer: fp32↔bf16={b_fb:.4f} fp32↔fp16={b_fp:.4f} "
            f"bf16↔fp16={b_bp:.4f} nats; ratio {ratio:.1f}×; iso={iso_ratio:.3f}. "
            f"Exponent-range partitioning is architecture-general."
            if isosceles else
            f"Non-isosceles in transformer: fp32↔bf16={b_fb:.4f} fp32↔fp16={b_fp:.4f} "
            f"bf16↔fp16={b_bp:.4f} nats; ratio {ratio:.1f}×. "
            f"Architecture may modulate the geometry."
        )
        log(f"\n  {interpretation}")

    results = {
        "experiment": "phase_xarch_lmc",
        "model":      {"n_layer": N_LAYER, "n_embd": N_EMBD, "n_head": N_HEAD,
                       "block_size": BLOCK_SIZE, "vocab_size": vocab_size},
        "dataset":    "tinyshakespeare_charlevel",
        "n_epochs":   N_EPOCHS,
        "lmc":        lmc_results,
        "interpretation": interpretation,
        "phase17_gnn_ref": {"fp32_bf16_ev": 0.0142, "fp32_fp16_ev": 0.1485,
                            "bf16_fp16_ev": 0.1504, "ratio": 10.5},
    }
    out = OUT_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    gsutil_cp(out, f"{GCS_P}/results.json")
    log(f"\n  Results → {out}"); log(f"  GCS     → {GCS_P}/results.json")
    log("=== phase_xarch_lmc complete ===")


if __name__ == "__main__":
    if "--train-only" in sys.argv:
        run_train_only()
    else:
        main()
