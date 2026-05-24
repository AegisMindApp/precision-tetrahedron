#!/usr/bin/env python3
"""
phase31_medium_retry.py
------------------------
Phase 31 model-size scaling — MEDIUM model (38M params) LMC retry.

Reduced memory vs original:
  N_EMBD=512, N_LAYER=12, N_HEAD=16 (38M params)
  BATCH_SIZE=16  (from 64)  — 4× less activation memory
  BLOCK_SIZE=128 (from 256) — 2× less sequence memory
  N_EPOCHS=60

LMC: FP32 ↔ BF16 only (fp16 skipped — CPU FP16 with 38M params too slow).

GCS output: gs://.../aegis_flashoptim/phase31_medium_retry/results.json
"""

import os, sys, json, math, subprocess, urllib.request
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from notify import notify, heartbeat

GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID     = os.environ.get("RUN_ID", "aegis_flashoptim")
GCS_BASE   = f"{GCS_BUCKET}/{RUN_ID}"
GCS_P      = f"{GCS_BASE}/phase31_medium_retry"
GCS_P_OLD  = f"{GCS_BASE}/phase31_scaling/medium"  # fallback for existing BF16

OUT_DIR  = Path("/tmp/phase31_medium_retry")
DATA_DIR = Path("/tmp/shakespeare")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── MEDIUM model hyperparameters ──────────────────────────────────────────────
N_EMBD       = 512
N_HEAD       = 16
N_LAYER      = 12
BLOCK_SIZE   = 128    # reduced from 256
DROPOUT      = 0.1
N_EPOCHS     = 60
BATCH_SIZE   = 16     # reduced from 64
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


def log(msg):
    print(f"[Phase31] {msg}", flush=True)


def gsutil_cp(src, dst):
    subprocess.run(["gsutil", "-q", "cp", str(src), dst], check=False)


def gcs_exists(gcs_path):
    return subprocess.run(["gsutil", "-q", "stat", gcs_path],
                          capture_output=True).returncode == 0


def download_from_gcs(local_path, *gcs_candidates):
    """Try each GCS path in order, download first found."""
    for gcs in gcs_candidates:
        if gcs_exists(gcs):
            log(f"  Downloading {local_path.name} from {gcs}")
            subprocess.run(["gsutil", "-q", "cp", gcs, str(local_path)], check=False)
            if local_path.exists():
                return True
    return False


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


class GPTMedium(nn.Module):
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


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def make_model(vocab_size, device=None):
    torch.manual_seed(SEED)
    m = GPTMedium(vocab_size)
    if device:
        m = m.to(device)
    return m


# ── Training ──────────────────────────────────────────────────────────────────

def get_device():
    if XLA_AVAILABLE:
        return xm.xla_device()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    ckpt_name = f"{precision}_ep{N_EPOCHS}.pt"
    ckpt      = OUT_DIR / ckpt_name

    # Check disk then GCS (new path then old path for BF16)
    if not ckpt.exists():
        gcs_new = f"{GCS_P}/{ckpt_name}"
        gcs_old = f"{GCS_P_OLD}/{ckpt_name}"
        if precision == "bf16":
            download_from_gcs(ckpt, gcs_new, gcs_old)
        else:
            download_from_gcs(ckpt, gcs_new)

    if ckpt.exists():
        dev = get_device()
        m   = make_model(vocab_size, dev)
        try:
            m.load_state_dict(torch.load(ckpt, map_location="cpu"))
            m.to(dev)
            vl = eval_loss(m, val_batches, dev)
            log(f"  {precision}: resumed  val_loss={vl:.4f}  bpc={vl/math.log(2):.3f}")
            return {"val_loss": round(vl, 5), "bpc": round(vl / math.log(2), 4), "resumed": True}
        except RuntimeError as e:
            log(f"  {precision}: checkpoint shape mismatch ({e.__class__.__name__}) — retraining from scratch")
            ckpt.unlink(missing_ok=True)

    dev = get_device()
    log(f"  {precision}: training on {dev}  (N_EMBD={N_EMBD} N_LAYER={N_LAYER} BATCH={BATCH_SIZE})")
    m   = make_model(vocab_size, dev)
    log(f"  Parameters: {count_params(m):,}")

    use_bf16 = (precision == "bf16")
    opt = torch.optim.AdamW(m.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-5)

    for ep in range(1, N_EPOCHS + 1):
        m.train()
        for x, y in train_batches:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            if use_bf16:
                with torch.autocast(device_type="xla" if XLA_AVAILABLE else "cpu",
                                    dtype=torch.bfloat16):
                    loss = F.cross_entropy(m(x).reshape(-1, vocab_size), y.reshape(-1))
            else:
                loss = F.cross_entropy(m(x).reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
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
            log(f"    {precision} ep{ep:3d}  val_loss={vl:.4f}  bpc={vl/math.log(2):.3f}"
                f"  lr={opt.param_groups[0]['lr']:.2e}")

    vl = eval_loss(m, val_batches, dev)
    torch.save({k: v.cpu().clone() for k, v in m.state_dict().items()}, ckpt)
    gsutil_cp(ckpt, f"{GCS_P}/{ckpt_name}")
    log(f"  {precision}: final val_loss={vl:.4f}  bpc={vl/math.log(2):.3f}")
    return {"val_loss": round(vl, 5), "bpc": round(vl / math.log(2), 4)}


# ── LMC ───────────────────────────────────────────────────────────────────────

def lmc_barrier(label, p0, p1, vocab_size, val_batches):
    sd0 = torch.load(p0, map_location="cpu")
    sd1 = torch.load(p1, map_location="cpu")
    dev = torch.device("cpu")
    m   = make_model(vocab_size, dev)
    alphas = [round(i / (N_ALPHA - 1), 2) for i in range(N_ALPHA)]
    losses = []

    for alpha in alphas:
        interp = {k: (1 - alpha) * sd0[k].float() + alpha * sd1[k].float()
                  for k in sd0 if k in sd1}
        m.load_state_dict(interp, strict=False)
        loss = eval_loss(m, val_batches, dev)
        losses.append(round(loss, 5))
        log(f"    α={alpha:.1f}  loss={losses[-1]:.5f}  bpc={losses[-1]/math.log(2):.4f}")

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
    log("  Phase 31 MEDIUM Retry — 38M GPT, reduced BATCH/BLOCK")
    log(f"  N_EMBD={N_EMBD}  N_LAYER={N_LAYER}  N_HEAD={N_HEAD}")
    log(f"  BATCH={BATCH_SIZE}  BLOCK_SIZE={BLOCK_SIZE}  N_EPOCHS={N_EPOCHS}")
    log("=" * 65)

    notify("PHASE_START", "[Phase31] MEDIUM 38M GPT LMC retry", data={})

    # ── Train FP32 and BF16 ───────────────────────────────────────────────────
    train_data, val_data, vocab_size = load_shakespeare()
    (OUT_DIR / "vocab_size.txt").write_text(str(vocab_size))
    train_batches = make_batches(train_data, BATCH_SIZE, BLOCK_SIZE)
    val_batches   = make_batches(val_data,   BATCH_SIZE, BLOCK_SIZE)

    # Estimate params
    dummy = GPTMedium(vocab_size)
    n_params = count_params(dummy)
    log(f"Model parameters: {n_params:,} (~{n_params/1e6:.1f}M)")
    del dummy

    fp32_result = train_one_precision("fp32", vocab_size, train_batches, val_batches)
    bf16_result = train_one_precision("bf16", vocab_size, train_batches, val_batches)

    # ── LMC FP32 ↔ BF16 ──────────────────────────────────────────────────────
    p_fp32 = OUT_DIR / f"fp32_ep{N_EPOCHS}.pt"
    p_bf16 = OUT_DIR / f"bf16_ep{N_EPOCHS}.pt"

    lmc_result = None
    if p_fp32.exists() and p_bf16.exists():
        log("\nRunning LMC: FP32 ↔ BF16 (CPU)...")
        lmc_result = lmc_barrier("FP32 ↔ BF16", p_fp32, p_bf16, vocab_size, val_batches)
    else:
        missing = []
        if not p_fp32.exists(): missing.append("fp32")
        if not p_bf16.exists(): missing.append("bf16")
        log(f"Skipping LMC — missing checkpoints: {missing}")

    # ── Report ────────────────────────────────────────────────────────────────
    log("\n" + "=" * 65)
    log("  PHASE 31 MEDIUM RESULTS")
    log(f"  FP32: val_loss={fp32_result['val_loss']}  bpc={fp32_result['bpc']}")
    log(f"  BF16: val_loss={bf16_result['val_loss']}  bpc={bf16_result['bpc']}")
    if lmc_result:
        log(f"  LMC FP32↔BF16: barrier={lmc_result['barrier_nats']:.5f} nats  bpc={lmc_result['barrier_bpc']:.5f}")

    results = {
        "experiment":  "phase31_medium_retry",
        "n_embd":      N_EMBD,
        "n_layer":     N_LAYER,
        "n_head":      N_HEAD,
        "batch_size":  BATCH_SIZE,
        "block_size":  BLOCK_SIZE,
        "n_epochs":    N_EPOCHS,
        "params_est":  n_params,
        "fp32_bpc":    fp32_result["bpc"],
        "bf16_bpc":    bf16_result["bpc"],
        "lmc_fp32_bf16_barrier_nats": lmc_result["barrier_nats"] if lmc_result else None,
        "lmc_curve":   lmc_result,
    }

    out = OUT_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    gsutil_cp(out, f"{GCS_P}/results.json")
    log(f"\nResults → GCS phase31_medium_retry/results.json")
    log("=== phase31 complete ===")

    notify("PHASE_COMPLETE", "[Phase31] MEDIUM 38M LMC retry done",
           data={"barrier_nats": lmc_result["barrier_nats"] if lmc_result else None,
                 "fp32_bpc": fp32_result["bpc"], "bf16_bpc": bf16_result["bpc"]})


if __name__ == "__main__":
    main()
