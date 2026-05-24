"""
phase35_colab.py  —  GPU FP16 Precision Triangle (Colab version)
================================================================
Run this in a Google Colab notebook with GPU runtime enabled.
(Runtime → Change runtime type → T4 GPU or better)

Paste into a single code cell and run, OR upload this file and run:
    !python phase35_colab.py

Results are saved to /content/phase35_results.json and auto-downloaded
at the end via google.colab.files.download().

Expected runtime: ~30–45 min on T4, ~15–20 min on A100.
"""

# ── Verify GPU ────────────────────────────────────────────────────────────────
import sys

# Python 3.12 rejects C extensions compiled against 3.11; upgrade torch and exit
# cleanly so the user only needs to click "Restart runtime" once.
try:
    import torch
except ValueError:
    import subprocess
    print("PyTorch C-extension incompatible with this Python version — upgrading...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "torch",
         "--index-url", "https://download.pytorch.org/whl/cu121"],
        check=False,
    )
    raise RuntimeError("Torch upgraded. Please do: Runtime → Restart runtime, then re-run.")

if not torch.cuda.is_available():
    raise RuntimeError("No GPU found. Runtime → Change runtime type → GPU")

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.version.cuda}  PyTorch: {torch.__version__}")

# ── Imports ───────────────────────────────────────────────────────────────────
import os, json, math, random, time, urllib.request
from pathlib import Path
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

OUT_DIR = Path("/content/phase35")
OUT_DIR.mkdir(exist_ok=True)

N_EPOCHS   = 80
LR_MAX     = 3e-4
LR_MIN     = 1e-5
BATCH_SIZE = 64
BLOCK_SIZE = 256
N_ALPHA    = 11
DEVICE     = torch.device("cuda")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Data ──────────────────────────────────────────────────────────────────────

def load_tinyshakespeare():
    url   = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    local = OUT_DIR / "tinyshakespeare.txt"
    if not local.exists():
        log("Downloading TinyShakespeare ...")
        urllib.request.urlretrieve(url, local)
    text      = local.read_text()
    chars     = sorted(set(text))
    stoi      = {c: i for i, c in enumerate(chars)}
    data      = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n         = int(0.9 * len(data))
    return data[:n], data[n:], len(chars)


def make_batches(data, device):
    n = max(1, len(data) // (BATCH_SIZE * BLOCK_SIZE))
    ix = torch.randint(len(data) - BLOCK_SIZE, (n * BATCH_SIZE,))
    x  = torch.stack([data[i:i+BLOCK_SIZE]   for i in ix]).to(device)
    y  = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix]).to(device)
    return list(zip(x.split(BATCH_SIZE), y.split(BATCH_SIZE)))


# ── GPT 6L/256d/8H ────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd,     bias=False)
        self.drop   = nn.Dropout(dropout)
        self.register_buffer("mask",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)).view(1,1,BLOCK_SIZE,BLOCK_SIZE))

    def forward(self, x):
        B, T, C = x.shape
        nh, hs  = self.n_head, C // self.n_head
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B,T,nh,hs).transpose(1,2)
        k = k.view(B,T,nh,hs).transpose(1,2)
        v = v.view(B,T,nh,hs).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(hs)
        att = att.masked_fill(self.mask[:,:,:T,:T]==0, float("-inf"))
        att = self.drop(F.softmax(att.float(), dim=-1).to(q.dtype))
        return self.c_proj((att @ v).transpose(1,2).reshape(B,T,C))


class GPTBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2  = nn.LayerNorm(n_embd)
        self.mlp  = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd, bias=False), nn.GELU(),
            nn.Linear(4*n_embd, n_embd, bias=False), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer=6, n_head=8, n_embd=256, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, n_embd)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.Sequential(*[GPTBlock(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.head    = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        h = self.drop(self.tok_emb(x) + self.pos_emb(torch.arange(T, device=x.device)))
        return self.head(self.ln_f(self.blocks(h)))


# ── LSTM 2L/256d ──────────────────────────────────────────────────────────────

class LSTMCharLM(nn.Module):
    def __init__(self, vocab_size, n_embd=256, n_layer=2, dropout=0.1):
        super().__init__()
        self.emb  = nn.Embedding(vocab_size, n_embd)
        self.lstm = nn.LSTM(n_embd, n_embd, n_layer, batch_first=True,
                            dropout=dropout if n_layer > 1 else 0.0)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x):
        return self.head(self.lstm(self.emb(x))[0])


# ── Train ─────────────────────────────────────────────────────────────────────

def cosine_lr(step, total):
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * step / total))


def eval_loss_f32(sd, model_fn, val_data):
    m = model_fn().float().to(DEVICE)
    m.load_state_dict(sd); m.eval()
    batches = make_batches(val_data, DEVICE)[:20]
    with torch.no_grad():
        losses = [F.cross_entropy(m(x).float().reshape(-1, m(x).size(-1) if False else
                  list(sd.values())[-1].shape[0]), y.reshape(-1)).item()
                  for x, y in batches]
    return sum(losses) / len(losses)


def eval_loss_model(model, val_data):
    model.eval()
    batches = make_batches(val_data, DEVICE)[:20]
    with torch.no_grad():
        losses = []
        for x, y in batches:
            logits = model(x)
            losses.append(F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                          y.reshape(-1)).item())
    return sum(losses) / len(losses)


def train_precision(model_fn, train_data, val_data, precision, arch):
    ckpt = OUT_DIR / f"{arch}_{precision}.pt"
    if ckpt.exists():
        log(f"  {arch}/{precision}: loading from disk")
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        return {k: v.float() for k, v in sd.items()}

    dtype  = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    # FP16 uses FP32 master weights; GradScaler cannot unscale FP16 parameter gradients
    master_dtype = torch.float32 if precision == "fp16" else dtype
    model  = model_fn().to(master_dtype).to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=LR_MAX)
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16"))

    total_steps = N_EPOCHS * max(1, len(train_data) // (BATCH_SIZE * BLOCK_SIZE))
    step = 0; best_val = float("inf"); best_sd = None

    for ep in range(1, N_EPOCHS + 1):
        model.train()
        for x, y in make_batches(train_data, DEVICE):
            for pg in opt.param_groups:
                pg["lr"] = cosine_lr(step, total_steps)
            opt.zero_grad()
            if precision == "fp16":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = model(x)
                    loss   = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                logits = model(x)
                loss   = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            step += 1

        if ep % 20 == 0 or ep == N_EPOCHS:
            val = eval_loss_model(model, val_data)
            bpc = val / math.log(2)
            log(f"  {arch}/{precision} ep{ep}/{N_EPOCHS}  val={val:.4f}  bpc={bpc:.3f}")
            if val < best_val:
                best_val = val
                best_sd  = {k: v.float().cpu().clone() for k, v in model.state_dict().items()}

    torch.save(best_sd, ckpt)
    log(f"  {arch}/{precision}: done  best_val={best_val:.5f}  bpc={best_val/math.log(2):.4f}")
    return best_sd


# ── LMC ───────────────────────────────────────────────────────────────────────

def lmc_edge(sd_a, sd_b, model_fn, val_data, label):
    alphas = [round(i / (N_ALPHA - 1), 2) for i in range(N_ALPHA)]
    val_batches = make_batches(val_data, DEVICE)[:20]
    losses = []

    for alpha in alphas:
        interp = {k: (1 - alpha) * sd_a[k].float() + alpha * sd_b[k].float() for k in sd_a}
        m = model_fn().float().to(DEVICE)
        m.load_state_dict(interp); m.eval()
        with torch.no_grad():
            ls = [F.cross_entropy(m(x).reshape(-1, m(x).size(-1) if False else
                  list(interp.values())[-1].shape[0]), y.reshape(-1)).item()
                  for x, y in val_batches]
        loss = sum(ls) / len(ls)
        losses.append(round(loss, 6))

    ep_mean = (losses[0] + losses[-1]) / 2
    barrier = max(losses) - ep_mean
    peak_a  = alphas[losses.index(max(losses))]
    log(f"  {label}: barrier={barrier:.5f} nats  ({barrier/math.log(2):.4f} bpc)  peak_α={peak_a}")
    return {"label": label, "alphas": alphas, "losses": losses,
            "barrier_nats": round(barrier, 6), "barrier_bpc": round(barrier/math.log(2), 6),
            "peak_alpha": peak_a}


def run_arch(arch, model_fn, train_data, val_data):
    log(f"\n{'='*55}\nArchitecture: {arch}\n{'='*55}")
    n_params = sum(p.numel() for p in model_fn().parameters())
    log(f"  Params: {n_params:,} ({n_params/1e6:.1f}M)")

    sds = {}
    vertex_losses = {}
    for prec in ["fp32", "bf16", "fp16"]:
        log(f"\n--- {prec.upper()} ---")
        sd = train_precision(model_fn, train_data, val_data, prec, arch)
        sds[prec] = sd
        m  = model_fn().float().to(DEVICE)
        m.load_state_dict(sd); m.eval()
        vl = eval_loss_model(m, val_data)
        vertex_losses[prec] = {"val_loss": round(vl, 5), "bpc": round(vl/math.log(2), 4)}
        log(f"  vertex: val_loss={vl:.5f}  bpc={vl/math.log(2):.4f}")

    log(f"\n--- LMC edges for {arch} ---")
    edges = [
        lmc_edge(sds["fp32"], sds["bf16"], model_fn, val_data, "FP32↔BF16"),
        lmc_edge(sds["fp32"], sds["fp16"], model_fn, val_data, "FP32↔FP16"),
        lmc_edge(sds["bf16"], sds["fp16"], model_fn, val_data, "BF16↔FP16"),
    ]

    b = {e["label"]: e["barrier_nats"] for e in edges}
    b32_16 = b["FP32↔FP16"]; bb16_16 = b["BF16↔FP16"]; b32_b16 = b["FP32↔BF16"]
    # Condition: both edges FROM vertex X are equal → X is the isosceles apex (isolated).
    # |FP32↔FP16 − BF16↔FP16| small → both edges from FP16 equal → FP16 is isolated apex
    if abs(b32_16 - bb16_16) / max(b32_16, bb16_16, 1e-9) < 0.10:
        isolated = "FP16 isolated (FP32 and BF16 cluster together)"
    # |FP32↔BF16 − FP32↔FP16| small → both edges from FP32 equal → FP32 is isolated apex
    elif abs(b32_b16 - b32_16) / max(b32_b16, b32_16, 1e-9) < 0.10:
        isolated = "FP32 isolated (BF16 and FP16 cluster together)"
    # |FP32↔BF16 − BF16↔FP16| small → both edges from BF16 equal → BF16 is isolated apex
    elif abs(b32_b16 - bb16_16) / max(b32_b16, bb16_16, 1e-9) < 0.10:
        isolated = "BF16 isolated (FP32 and FP16 cluster together)"
    else:
        isolated = "scalene (no clear isosceles pair)"

    triangle = {"fp32_bf16": round(b32_b16,5), "fp32_fp16": round(b32_16,5),
                "bf16_fp16": round(bb16_16,5), "geometry": isolated,
                "max_to_min_ratio": round(max(b.values())/(min(b.values())+1e-9), 2)}
    log(f"\nTriangle ({arch}): FP32↔BF16={b32_b16:.4f}  FP32↔FP16={b32_16:.4f}  BF16↔FP16={bb16_16:.4f}")
    log(f"  → {isolated}")

    return {"arch": arch, "n_params": n_params, "vertex_losses": vertex_losses,
            "edges": edges, "triangle": triangle}


# ── Main ──────────────────────────────────────────────────────────────────────

train_data, val_data, vocab_size = load_tinyshakespeare()
log(f"vocab={vocab_size}  train={len(train_data):,}  val={len(val_data):,}")

results = {
    "phase": 35,
    "hardware": torch.cuda.get_device_name(0),
    "cuda_version": torch.version.cuda,
    "pytorch_version": torch.__version__,
    "architectures": {}
}

def make_gpt():
    return GPT(vocab_size=vocab_size)

def make_lstm():
    return LSTMCharLM(vocab_size=vocab_size)

results["architectures"]["gpt6l"]  = run_arch("gpt6l",  make_gpt,  train_data, val_data)
results["architectures"]["lstm2l"] = run_arch("lstm2l", make_lstm, train_data, val_data)

# ── Save and download ─────────────────────────────────────────────────────────
out_path = Path("/content/phase35_results.json")
out_path.write_text(json.dumps(results, indent=2))

log("\n" + "="*55)
log("PHASE 35 COMPLETE — Summary:")
for arch, res in results["architectures"].items():
    tri = res["triangle"]
    log(f"\n  {arch} ({res['n_params']/1e6:.1f}M):")
    log(f"    FP32↔BF16 = {tri['fp32_bf16']:.4f} nats")
    log(f"    FP32↔FP16 = {tri['fp32_fp16']:.4f} nats")
    log(f"    BF16↔FP16 = {tri['bf16_fp16']:.4f} nats")
    log(f"    → {tri['geometry']}")
log("="*55)
log(f"\nResults saved to {out_path}")

# Auto-download in Colab
try:
    from google.colab import files
    files.download(str(out_path))
    log("Download triggered.")
except ImportError:
    log(f"Not in Colab — results at {out_path}")
    print(out_path.read_text())
