#!/usr/bin/env python3
"""
phase_cpu_fp16_lmc.py — CPU FP16 LMC triangle (completes Phase 13).

Trains a small GPT char-LM on TinyShakespeare in FP32, BF16, and genuine
FP16 (5-bit exponent) using CPU-native PyTorch — no XLA, no silent promotion.

On v6e TPU hardware XLA_USE_F16 is a no-op; this run provides the missing
FP16 leg of the precision triangle:

    FP32 ↔ BF16  (8-bit exp ↔ 8-bit exp)   expected: small barrier
    FP32 ↔ FP16  (8-bit exp ↔ 5-bit exp)   expected: large barrier
    BF16 ↔ FP16  (8-bit exp ↔ 5-bit exp)   expected: large barrier ≈ FP32↔FP16

If the isosceles triangle from Phase 17 (GNN) generalises to transformers,
the two FP16-involving barriers should be large and roughly equal; the
FP32↔BF16 barrier should be small.
"""

import os, math, json, urllib.request, subprocess
from pathlib import Path

# Use all available cores on cloud VMs; default 4 to avoid starving local SSH
_ncpus = str(os.cpu_count() or 4)
os.environ.setdefault("OMP_NUM_THREADS", _ncpus)
os.environ.setdefault("MKL_NUM_THREADS", _ncpus)

GCS_BASE  = "gs://aegismind-tpu-results/aegis_flashoptim/phase_cpu_fp16_lmc"
GCS_DONE  = f"{GCS_BASE}/results.json"

def gcs_exists(path):
    r = subprocess.run(["gsutil", "-q", "stat", path], capture_output=True)
    return r.returncode == 0

def gcs_upload(local, remote):
    subprocess.run(["gsutil", "cp", str(local), remote], check=True)

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Config ───────────────────────────────────────────────────────────────────
N_LAYERS    = 4
N_HEADS     = 4
N_EMBD      = 128
SEQ_LEN     = 64
BATCH       = 64
LR          = 1e-3
EPOCHS      = 60
LMC_STEPS   = 11
SEED        = 42
EVAL_BATCHES = 30

DATA_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = Path("/tmp/tinyshakespeare.txt")
CKPT_DIR  = Path("/tmp/cpu_fp16_lmc")
CKPT_DIR.mkdir(exist_ok=True)

torch.manual_seed(SEED)

# ── Data ─────────────────────────────────────────────────────────────────────
if not DATA_PATH.exists():
    print("Downloading TinyShakespeare...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

text      = DATA_PATH.read_text()
chars     = sorted(set(text))
vocab_size = len(chars)
stoi      = {c: i for i, c in enumerate(chars)}
data      = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n_train   = int(0.9 * len(data))
train_data, val_data = data[:n_train], data[n_train:]

def get_batch(split):
    d  = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - SEQ_LEN, (BATCH,))
    x  = torch.stack([d[i:i+SEQ_LEN]   for i in ix])
    y  = torch.stack([d[i+1:i+SEQ_LEN+1] for i in ix])
    return x, y

# ── Model ─────────────────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads  = N_HEADS
        self.head_dim = N_EMBD // N_HEADS
        self.qkv  = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.proj = nn.Linear(N_EMBD, N_EMBD,     bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(SEQ_LEN, SEQ_LEN)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        # softmax in fp32 for numerical stability, cast back
        att = F.softmax(att.float(), dim=-1).to(x.dtype)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln2  = nn.LayerNorm(N_EMBD)
        self.ff   = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD, bias=False),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTCharLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, N_EMBD)
        self.pos_emb = nn.Embedding(SEQ_LEN, N_EMBD)
        self.blocks  = nn.Sequential(*[Block() for _ in range(N_LAYERS)])
        self.ln_f    = nn.LayerNorm(N_EMBD)
        self.head    = nn.Linear(N_EMBD, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device)
        h    = self.tok_emb(x) + self.pos_emb(pos)
        h    = self.blocks(h)
        h    = self.ln_f(h)
        return self.head(h)


def cast_model(model, dtype):
    """Cast model to dtype; keep LayerNorm in fp32 for all precisions."""
    model = model.to(dtype)
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            m.float()
    return model


# ── Eval ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_loss(model):
    model.eval()
    losses = []
    for _ in range(EVAL_BATCHES):
        x, y = get_batch("val")
        logits = model(x)
        loss   = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            y.reshape(-1)
        )
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ── Training ─────────────────────────────────────────────────────────────────
def train(name, dtype, lr=None):
    print(f"\n{'='*60}")
    print(f"  {name}  dtype={dtype}  epochs={EPOCHS}")
    print("="*60)

    # FP16 on CPU needs a lower LR — max representable is 65504 and
    # LR=1e-3 causes overflow/NaN in the first few epochs.
    if lr is None:
        lr = 2e-4 if dtype == torch.float16 else LR

    torch.manual_seed(SEED)
    model = cast_model(GPTCharLM(), dtype)

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

    steps = max(1, n_train // (SEQ_LEN * BATCH))

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for _ in range(steps):
            x, y = get_batch("train")
            opt.zero_grad()
            logits = model(x)
            loss   = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y.reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        if epoch % 10 == 0 or epoch == EPOCHS:
            val = eval_loss(model)
            print(f"  ep{epoch:3d}  val_loss={val:.4f}  lr={sched.get_last_lr()[0]:.2e}")

    # Save as fp32 for LMC interpolation
    sd_fp32 = {k: v.float() for k, v in model.state_dict().items()}
    torch.save(sd_fp32, CKPT_DIR / f"{name}.pt")
    final = eval_loss(model)
    print(f"  Final val_loss = {final:.4f}  → saved {name}.pt")
    return final


# ── LMC ─────────────────────────────────────────────────────────────────────
def lmc_barrier(name_a, name_b):
    sd_a = torch.load(CKPT_DIR / f"{name_a}.pt", map_location="cpu")
    sd_b = torch.load(CKPT_DIR / f"{name_b}.pt", map_location="cpu")

    alphas = [i / (LMC_STEPS - 1) for i in range(LMC_STEPS)]
    losses = []
    for alpha in alphas:
        torch.manual_seed(SEED)
        model = GPTCharLM()                         # fp32
        sd    = {k: (1 - alpha) * sd_a[k] + alpha * sd_b[k] for k in sd_a}
        model.load_state_dict(sd, strict=False)
        val   = eval_loss(model)
        losses.append(val)

    barrier = max(losses) - (losses[0] + losses[-1]) / 2
    peak_α  = alphas[losses.index(max(losses))]
    return barrier, peak_α, losses


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if gcs_exists(GCS_DONE):
        print(f"  GCS result already exists at {GCS_DONE} — skipping (delete to rerun).")
        import sys; sys.exit(0)

    print(f"  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}  CPUs={os.cpu_count()}")
    results = {}

    for name, dtype in [("fp32", torch.float32),
                         ("bf16", torch.bfloat16),
                         ("fp16", torch.float16)]:
        if (CKPT_DIR / f"{name}.pt").exists():
            print(f"\n  {name}.pt already exists — skipping training")
            continue
        results[name] = {"final_val_loss": train(name, dtype)}

    print("\n" + "="*60)
    print("  LMC Triangle — CPU genuine FP16")
    print("="*60)

    triangle = {}
    for a, b in [("fp32", "bf16"), ("fp32", "fp16"), ("bf16", "fp16")]:
        label = f"{a.upper()}↔{b.upper()}"
        print(f"\n{label}:")
        barrier, peak_α, curve = lmc_barrier(a, b)
        triangle[label] = {
            "barrier_nats": round(barrier, 4),
            "peak_alpha":   round(peak_α, 2),
            "curve":        [round(v, 4) for v in curve],
        }
        for i, (α, v) in enumerate(zip([i/(LMC_STEPS-1) for i in range(LMC_STEPS)], curve)):
            peak_marker = " ← PEAK" if v == max(curve) else ""
            print(f"  α={α:.1f}  loss={v:.4f}{peak_marker}")
        print(f"  Barrier = {barrier:.4f} nats  peak_α = {peak_α:.1f}")

    print("\n" + "="*60)
    print("  SUMMARY TABLE")
    print("="*60)
    print(f"  {'Pair':<15} {'Barrier (nats)':>15} {'peak α':>8}")
    print(f"  {'-'*40}")
    for label, v in triangle.items():
        print(f"  {label:<15} {v['barrier_nats']:>15.4f} {v['peak_alpha']:>8.1f}")

    out = {"training": results, "lmc_triangle": triangle,
           "hardware": "cpu", "omp_threads": os.environ.get("OMP_NUM_THREADS")}
    local_results = CKPT_DIR / "results.json"
    local_results.write_text(json.dumps(out, indent=2))
    print(f"\n  Results → {local_results}")

    # Upload to GCS
    print(f"  Uploading to {GCS_BASE}/ ...")
    gcs_upload(local_results, GCS_DONE)
    for ck in ["fp32.pt", "bf16.pt", "fp16.pt"]:
        p = CKPT_DIR / ck
        if p.exists():
            gcs_upload(p, f"{GCS_BASE}/{ck}")
    print("  GCS upload complete.")
