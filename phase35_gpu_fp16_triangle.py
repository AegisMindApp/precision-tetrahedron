#!/usr/bin/env python3
"""
phase35_gpu_fp16_triangle.py
-----------------------------
Complete the FP32 ↔ BF16 ↔ FP16 precision triangle on NVIDIA GPU (L4/T4).

On v6e TPUs, torch.float16 silently promotes to float32 (XLA_USE_F16 is a
no-op), making FP16 edges unmeasurable on that hardware. This GPU run
obtains genuine FP16 checkpoints using CUDA native FP16 tensor cores.

Experiments:
  A. GPT char-LM  6L/256d/8H  (~6M params, matches §4.15/§4.24/§4.27)
  B. LSTM char-LM 2L/256d     (~8M params, matches §4.29 Sub-A)

For each: train FP32, BF16, FP16 for 80 epochs on TinyShakespeare.
Compute all 3 LMC edges (11 α-points, val-set MSE).
Report: vertex losses, triangle edge barriers, isosceles check.

GCS output: gs://.../aegis_flashoptim/phase35_gpu_fp16_triangle/results.json
"""

import os, sys, json, time, subprocess, math, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID     = os.environ.get("RUN_ID", "aegis_flashoptim")
GCS_OUT    = f"{GCS_BUCKET}/{RUN_ID}/phase35_gpu_fp16_triangle"

OUT_DIR = Path("/tmp/phase35_gpu_fp16_triangle")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED       = 42
N_EPOCHS   = 80
LR_MAX     = 3e-4
LR_MIN     = 1e-5
BATCH_SIZE = 64
BLOCK_SIZE = 256
N_ALPHA    = 11

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] {msg}", flush=True)


def gsutil_cp(src, dst):
    subprocess.run(["gsutil", "-q", "cp", str(src), str(dst)], check=False)


def gcs_exists(path):
    return subprocess.run(["gsutil", "-q", "stat", path],
                          capture_output=True).returncode == 0


# ── Data ──────────────────────────────────────────────────────────────────────

def load_tinyshakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    local = OUT_DIR / "tinyshakespeare.txt"
    if not local.exists():
        log("Downloading TinyShakespeare ...")
        import urllib.request
        urllib.request.urlretrieve(url, local)
    text = local.read_text()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], vocab_size


def make_batches(data, batch_size, block_size, device):
    batches = []
    for _ in range(max(1, len(data) // (batch_size * block_size))):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x  = torch.stack([data[i:i+block_size] for i in ix]).to(device)
        y  = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
        batches.append((x, y))
    return batches


# ── GPT model ─────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.c_attn  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj  = nn.Linear(n_embd, n_embd, bias=False)
        self.drop    = nn.Dropout(dropout)
        self.register_buffer("mask",
            torch.tril(torch.ones(block_size, block_size))
            .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        nh, hs = self.n_head, C // self.n_head
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, nh, hs).transpose(1, 2)
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att.float(), dim=-1).to(q.dtype))
        return self.c_proj((att @ v).transpose(1, 2).reshape(B, T, C))


class GPTBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.Sequential(*[
            GPTBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.head    = nn.Linear(n_embd, vocab_size, bias=False)
        self.block_size = block_size

    def forward(self, x):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device)
        tok  = self.tok_emb(x)
        h    = self.drop(tok + self.pos_emb(pos))
        h    = self.blocks(h)
        return self.head(self.ln_f(h))


# ── LSTM model ────────────────────────────────────────────────────────────────

class LSTMCharLM(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer, dropout=0.1):
        super().__init__()
        self.emb  = nn.Embedding(vocab_size, n_embd)
        self.lstm = nn.LSTM(n_embd, n_embd, n_layer,
                            batch_first=True, dropout=dropout if n_layer > 1 else 0.0)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x):
        h = self.emb(x)
        out, _ = self.lstm(h)
        return self.head(out)


# ── Training ──────────────────────────────────────────────────────────────────

def cosine_lr(step, total_steps):
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * step / total_steps))


def eval_loss_f32(model_sd, model_fn, val_batches, device):
    """Evaluate a state_dict (float32) by loading into a fresh model."""
    m = model_fn().float().to(device)
    m.load_state_dict(model_sd)
    m.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in val_batches:
            logits = m(xb).float()
            loss   = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            total += loss.item() * xb.size(0)
            n     += xb.size(0)
    return total / n


def train_precision(model_fn, train_data, val_data, vocab_size, device,
                    precision, label, ckpt_path, gcs_ckpt):
    if ckpt_path.exists():
        log(f"  {label}: checkpoint on disk — skip training")
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        return {k: v.float() for k, v in sd.items()}

    if gcs_exists(gcs_ckpt):
        log(f"  {label}: downloading GCS checkpoint")
        subprocess.run(["gsutil", "-q", "cp", gcs_ckpt, str(ckpt_path)], check=True)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        return {k: v.float() for k, v in sd.items()}

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    model = model_fn().to(dtype).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR_MAX)
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16"))

    # Pre-build batches (refresh each epoch for stochasticity)
    n_steps = 0
    best_val  = float("inf")
    best_sd   = None

    for ep in range(1, N_EPOCHS + 1):
        model.train()
        train_batches = make_batches(train_data, BATCH_SIZE, BLOCK_SIZE, device)
        ep_loss = 0.0

        for step, (xb, yb) in enumerate(train_batches):
            lr = cosine_lr(n_steps, N_EPOCHS * len(train_batches))
            for pg in opt.param_groups:
                pg["lr"] = lr

            opt.zero_grad()
            if precision == "fp16":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = model(xb)
                    loss   = F.cross_entropy(
                        logits.float().reshape(-1, logits.size(-1)), yb.reshape(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(xb)
                loss   = F.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)), yb.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            ep_loss += loss.item()
            n_steps += 1

        ep_loss /= len(train_batches)

        model.eval()
        val_batches = make_batches(val_data, BATCH_SIZE, BLOCK_SIZE, device)
        with torch.no_grad():
            val_losses = []
            for xb, yb in val_batches[:20]:
                logits = model(xb)
                val_losses.append(F.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)), yb.reshape(-1)).item())
        val_loss = sum(val_losses) / len(val_losses)

        if val_loss < best_val:
            best_val = val_loss
            best_sd  = {k: v.float().cpu().clone() for k, v in model.state_dict().items()}

        if ep % 10 == 0:
            bpc = val_loss / math.log(2)
            log(f"  {label} ep {ep:3d}/{N_EPOCHS}  train={ep_loss:.4f}  val={val_loss:.4f}  bpc={bpc:.3f}  lr={lr:.2e}")

    torch.save(best_sd, ckpt_path)
    gsutil_cp(ckpt_path, gcs_ckpt)
    log(f"  {label}: done  best_val={best_val:.5f}  bpc={best_val/math.log(2):.4f}")
    return best_sd


# ── LMC ───────────────────────────────────────────────────────────────────────

def lmc_edge(sd_a, sd_b, model_fn, val_data, device, label):
    alphas    = [round(i / (N_ALPHA - 1), 2) for i in range(N_ALPHA)]
    val_batches = make_batches(val_data, BATCH_SIZE, BLOCK_SIZE, device)[:20]
    losses    = []

    for alpha in alphas:
        interp = {k: (1 - alpha) * sd_a[k].float() + alpha * sd_b[k].float()
                  for k in sd_a}
        loss = eval_loss_f32(interp, model_fn, val_batches, device)
        losses.append(round(loss, 6))
        log(f"  {label} α={alpha:.1f}  loss={loss:.5f}  bpc={loss/math.log(2):.4f}")

    ep_mean  = (losses[0] + losses[-1]) / 2
    barrier  = max(losses) - ep_mean
    peak_a   = alphas[losses.index(max(losses))]
    bpc_barrier = barrier / math.log(2)
    log(f"  {label}: barrier={barrier:.5f} nats ({bpc_barrier:.4f} bpc)  peak_α={peak_a}")
    return {
        "label": label,
        "alphas": alphas,
        "losses": losses,
        "barrier_nats": round(barrier, 6),
        "barrier_bpc":  round(bpc_barrier, 6),
        "endpoint_mean": round(ep_mean, 6),
        "peak_alpha": peak_a,
    }


def isosceles_check(edges):
    """Check if triangle is isosceles: one vertex isolated, two edges equal."""
    edge_map = {e["label"]: e["barrier_nats"] for e in edges}
    pairs = [
        ("FP32↔BF16", "FP32↔FP16", "BF16↔FP16"),
        ("FP32↔BF16", "FP32↔FP16", "BF16↔FP16"),
    ]
    b_fp32_bf16 = edge_map.get("FP32↔BF16", 0)
    b_fp32_fp16 = edge_map.get("FP32↔FP16", 0)
    b_bf16_fp16 = edge_map.get("BF16↔FP16", 0)

    barriers = sorted([b_fp32_bf16, b_fp32_fp16, b_bf16_fp16])
    ratio = barriers[2] / (barriers[0] + 1e-9)
    isolated_vertex = None
    if abs(b_fp32_fp16 - b_bf16_fp16) < 0.05 * max(b_fp32_fp16, b_bf16_fp16, 1e-9):
        isolated_vertex = "BF16"
    elif abs(b_fp32_bf16 - b_bf16_fp16) < 0.05 * max(b_fp32_bf16, b_bf16_fp16, 1e-9):
        isolated_vertex = "FP32"
    elif abs(b_fp32_bf16 - b_fp32_fp16) < 0.05 * max(b_fp32_bf16, b_fp32_fp16, 1e-9):
        isolated_vertex = "FP16"

    return {
        "fp32_bf16": round(b_fp32_bf16, 5),
        "fp32_fp16": round(b_fp32_fp16, 5),
        "bf16_fp16": round(b_bf16_fp16, 5),
        "isosceles_isolated_vertex": isolated_vertex,
        "max_to_min_ratio": round(ratio, 2),
    }


# ── Run one architecture ───────────────────────────────────────────────────────

def run_arch(arch_name, model_fn, train_data, val_data, device):
    log(f"\n{'='*60}")
    log(f"Architecture: {arch_name}")
    log(f"{'='*60}")

    done_key = f"{GCS_OUT}/{arch_name}_results.json"
    if gcs_exists(done_key):
        log(f"  Already complete — skipping")
        return json.loads(subprocess.run(
            ["gsutil", "cat", done_key], capture_output=True, text=True).stdout)

    sds = {}
    vertex_losses = {}

    for precision in ["fp32", "bf16", "fp16"]:
        label    = f"{arch_name}_{precision}"
        ckpt     = OUT_DIR / f"{arch_name}_{precision}_ep{N_EPOCHS}.pt"
        gcs_ckpt = f"{GCS_OUT}/{arch_name}_{precision}_ep{N_EPOCHS}.pt"
        log(f"\n[{precision.upper()}] Training {arch_name} ...")
        sd = train_precision(model_fn, train_data, val_data, len(train_data),
                             device, precision, label, ckpt, gcs_ckpt)
        sds[precision] = sd
        val_batches = make_batches(val_data, BATCH_SIZE, BLOCK_SIZE, device)[:20]
        vl = eval_loss_f32(sd, model_fn, val_batches, device)
        vertex_losses[precision] = {"val_loss": round(vl, 5), "bpc": round(vl / math.log(2), 4)}
        log(f"  {precision}: val_loss={vl:.5f}  bpc={vl/math.log(2):.4f}")

    log(f"\nVertex summary for {arch_name}:")
    for p, v in vertex_losses.items():
        log(f"  {p}: val_loss={v['val_loss']}  bpc={v['bpc']}")

    log(f"\nComputing LMC edges for {arch_name} ...")
    edges = [
        lmc_edge(sds["fp32"], sds["bf16"], model_fn, val_data, device, "FP32↔BF16"),
        lmc_edge(sds["fp32"], sds["fp16"], model_fn, val_data, device, "FP32↔FP16"),
        lmc_edge(sds["bf16"], sds["fp16"], model_fn, val_data, device, "BF16↔FP16"),
    ]

    iso = isosceles_check(edges)
    log(f"\nIsosceles check: isolated_vertex={iso['isosceles_isolated_vertex']}  "
        f"max/min ratio={iso['max_to_min_ratio']}×")

    result = {
        "arch": arch_name,
        "n_epochs": N_EPOCHS,
        "vertex_losses": vertex_losses,
        "edges": edges,
        "triangle": iso,
    }

    out_json = OUT_DIR / f"{arch_name}_results.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    gsutil_cp(out_json, done_key)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("Phase 35: GPU FP16 precision triangle")
    log(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"  GPU: {torch.cuda.get_device_name(0)}")
        log(f"  CUDA: {torch.version.cuda}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  Device: {device}")

    done_path = f"{GCS_OUT}/results.json"
    if gcs_exists(done_path):
        log("Already complete — exiting")
        return

    train_data, val_data, vocab_size = load_tinyshakespeare()
    log(f"  Vocab size: {vocab_size}  train: {len(train_data)}  val: {len(val_data)}")

    # ── Architecture A: GPT 6L/256d/8H ────────────────────────────────────────
    def make_gpt6l():
        return GPT(vocab_size=vocab_size, block_size=BLOCK_SIZE,
                   n_layer=6, n_head=8, n_embd=256)

    n_params = sum(p.numel() for p in make_gpt6l().parameters())
    log(f"\nGPT-6L params: {n_params:,} (~{n_params/1e6:.1f}M)")
    gpt6l_result = run_arch("gpt6l", make_gpt6l, train_data, val_data, device)

    # ── Architecture B: LSTM 2L/256d ──────────────────────────────────────────
    def make_lstm2l():
        return LSTMCharLM(vocab_size=vocab_size, n_embd=256, n_layer=2)

    n_params = sum(p.numel() for p in make_lstm2l().parameters())
    log(f"\nLSTM-2L params: {n_params:,} (~{n_params/1e6:.1f}M)")
    lstm2l_result = run_arch("lstm2l", make_lstm2l, train_data, val_data, device)

    # ── Collate and upload ─────────────────────────────────────────────────────
    results = {
        "phase": 35,
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "architectures": {
            "gpt6l": gpt6l_result,
            "lstm2l": lstm2l_result,
        }
    }

    log("\n" + "=" * 60)
    log("Phase 35 COMPLETE")
    for arch, res in results["architectures"].items():
        log(f"\n  {arch}:")
        for p, v in res["vertex_losses"].items():
            log(f"    {p}: bpc={v['bpc']}")
        tri = res["triangle"]
        log(f"    FP32↔BF16: {tri['fp32_bf16']:.4f} nats")
        log(f"    FP32↔FP16: {tri['fp32_fp16']:.4f} nats")
        log(f"    BF16↔FP16: {tri['bf16_fp16']:.4f} nats")
        log(f"    Isolated vertex: {tri['isosceles_isolated_vertex']}")
    log("=" * 60)

    out_json = OUT_DIR / "results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    gsutil_cp(out_json, done_path)
    log(f"Results at {done_path}")


if __name__ == "__main__":
    main()
