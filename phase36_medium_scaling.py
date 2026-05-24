#!/usr/bin/env python3
"""
phase36_medium_scaling.py
--------------------------
Phase 36: MEDIUM GPT (38M params) scaling-law fix.
Previous run (phase31_medium_retry) used BLOCK_SIZE=128 / 60 epochs and gave
bpc=6.26 — clearly underfit, skewing the params^(-0.85) exponent.
This run uses BLOCK_SIZE=256 / 80 epochs (matching all other scaling-law runs).

GCS output: gs://.../aegis_flashoptim/phase36_medium_scaling/results.json
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
from notify import notify

GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID     = os.environ.get("RUN_ID", "aegis_flashoptim")
GCS_BASE   = f"{GCS_BUCKET}/{RUN_ID}"
GCS_P      = f"{GCS_BASE}/phase36_medium_scaling"
GCS_DONE   = f"{GCS_P}/results.json"

OUT_DIR  = Path("/tmp/phase36_medium_scaling")
DATA_DIR = Path("/tmp/shakespeare")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

N_EMBD       = 512
N_HEAD       = 16
N_LAYER      = 12
BLOCK_SIZE   = 256   # fixed from 128
DROPOUT      = 0.1
N_EPOCHS     = 80    # fixed from 60
BATCH_SIZE   = 16
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


def log(msg): print(f"[Phase36] {msg}", flush=True)
def gsutil_cp(src, dst): subprocess.run(["gsutil", "-q", "cp", str(src), dst], check=False)
def gcs_exists(p): return subprocess.run(["gsutil", "-q", "stat", p], capture_output=True).returncode == 0


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
            nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD), nn.Dropout(DROPOUT),
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


def get_device():
    return xm.xla_device() if XLA_AVAILABLE else torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_model(vocab_size, device=None):
    torch.manual_seed(SEED)
    m = GPTMedium(vocab_size)
    return m.to(device) if device else m


def eval_loss(model, batches, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum").item()
            n += y.numel()
    if XLA_AVAILABLE and str(device) != "cpu":
        xm.mark_step()
    return total / n if n > 0 else float("inf")


def train_precision(precision, vocab_size, train_batches, val_batches):
    ckpt = OUT_DIR / f"{precision}_ep{N_EPOCHS}.pt"
    if not ckpt.exists():
        gcs = f"{GCS_P}/{ckpt.name}"
        if gcs_exists(gcs):
            log(f"  {precision}: downloading from GCS")
            subprocess.run(["gsutil", "-q", "cp", gcs, str(ckpt)], check=False)

    if ckpt.exists():
        dev = get_device()
        m   = make_model(vocab_size, dev)
        try:
            m.load_state_dict(torch.load(ckpt, map_location="cpu"))
            m.to(dev)
            vl = eval_loss(m, val_batches, dev)
            log(f"  {precision}: resumed  val={vl:.4f}  bpc={vl/math.log(2):.3f}")
            return {"val_loss": round(vl, 5), "bpc": round(vl / math.log(2), 4), "resumed": True}
        except RuntimeError:
            ckpt.unlink(missing_ok=True)

    dev = get_device()
    log(f"  {precision}: training on {dev}")
    m   = make_model(vocab_size, dev)
    log(f"  Parameters: {sum(p.numel() for p in m.parameters()):,}")

    use_bf16 = (precision == "bf16")
    opt = torch.optim.AdamW(m.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-5)

    for ep in range(1, N_EPOCHS + 1):
        m.train()
        for x, y in train_batches:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            if use_bf16:
                with torch.autocast(device_type="xla" if XLA_AVAILABLE else "cpu", dtype=torch.bfloat16):
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
            log(f"    {precision} ep{ep:3d}/{N_EPOCHS}  val={vl:.4f}  bpc={vl/math.log(2):.3f}  lr={opt.param_groups[0]['lr']:.2e}")

    vl = eval_loss(m, val_batches, dev)
    sd = {k: v.cpu().clone() for k, v in m.state_dict().items()}
    torch.save(sd, ckpt)
    gsutil_cp(ckpt, f"{GCS_P}/{ckpt.name}")
    log(f"  {precision}: done  val={vl:.4f}  bpc={vl/math.log(2):.3f}")
    return {"val_loss": round(vl, 5), "bpc": round(vl / math.log(2), 4)}


def lmc_barrier(label, p0, p1, vocab_size, val_batches):
    sd0 = torch.load(p0, map_location="cpu")
    sd1 = torch.load(p1, map_location="cpu")
    dev = torch.device("cpu")
    m   = make_model(vocab_size, dev)
    alphas = [round(i / (N_ALPHA - 1), 2) for i in range(N_ALPHA)]
    losses = []
    for alpha in alphas:
        interp = {k: (1 - alpha) * sd0[k].float() + alpha * sd1[k].float() for k in sd0 if k in sd1}
        m.load_state_dict(interp, strict=False)
        loss = eval_loss(m, val_batches, dev)
        losses.append(round(loss, 5))
        log(f"    α={alpha:.1f}  loss={losses[-1]:.5f}  bpc={losses[-1]/math.log(2):.4f}")
    base    = (losses[0] + losses[-1]) / 2
    barrier = round(max(losses) - base, 5)
    peak_a  = alphas[losses.index(max(losses))]
    log(f"  → {label}: barrier={barrier:.5f} nats  bpc={barrier/math.log(2):.5f}  peak_α={peak_a}")
    return {"label": label, "alphas": alphas, "losses": losses,
            "barrier_nats": barrier, "barrier_bpc": round(barrier / math.log(2), 5),
            "peak_alpha": peak_a}


def main():
    if gcs_exists(GCS_DONE):
        log("Results already in GCS — nothing to do."); return

    log("=" * 65)
    log(f"  Phase 36 — MEDIUM 38M GPT scaling-law fix")
    log(f"  N_EMBD={N_EMBD}  N_LAYER={N_LAYER}  BLOCK_SIZE={BLOCK_SIZE}  N_EPOCHS={N_EPOCHS}")
    log("=" * 65)
    notify("PHASE_START", "[Phase36] MEDIUM scaling-law fix", data={})

    train_data, val_data, vocab_size = load_shakespeare()
    train_batches = make_batches(train_data, BATCH_SIZE, BLOCK_SIZE)
    val_batches   = make_batches(val_data,   BATCH_SIZE, BLOCK_SIZE)

    n_params = sum(p.numel() for p in GPTMedium(vocab_size).parameters())
    log(f"Parameters: {n_params:,} (~{n_params/1e6:.1f}M)")

    fp32_r = train_precision("fp32", vocab_size, train_batches, val_batches)
    bf16_r = train_precision("bf16", vocab_size, train_batches, val_batches)

    p_fp32 = OUT_DIR / f"fp32_ep{N_EPOCHS}.pt"
    p_bf16 = OUT_DIR / f"bf16_ep{N_EPOCHS}.pt"
    lmc_r  = None
    if p_fp32.exists() and p_bf16.exists():
        log("\nRunning LMC: FP32 ↔ BF16 ...")
        lmc_r = lmc_barrier("FP32 ↔ BF16", p_fp32, p_bf16, vocab_size, val_batches)

    results = {
        "experiment": "phase36_medium_scaling",
        "n_embd": N_EMBD, "n_layer": N_LAYER, "n_head": N_HEAD,
        "block_size": BLOCK_SIZE, "n_epochs": N_EPOCHS, "batch_size": BATCH_SIZE,
        "params": n_params,
        "fp32_bpc": fp32_r["bpc"], "bf16_bpc": bf16_r["bpc"],
        "lmc_fp32_bf16_barrier_nats": lmc_r["barrier_nats"] if lmc_r else None,
        "lmc_curve": lmc_r,
    }
    out = OUT_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    gsutil_cp(out, GCS_DONE)
    log("Results → GCS phase36_medium_scaling/results.json")
    notify("PHASE_COMPLETE", "[Phase36] MEDIUM scaling done",
           data={"barrier_nats": lmc_r["barrier_nats"] if lmc_r else None,
                 "fp32_bpc": fp32_r["bpc"], "bf16_bpc": bf16_r["bpc"]})


if __name__ == "__main__":
    main()
