#!/usr/bin/env python3
"""
phase37_int8_gpt.py
--------------------
Phase 37: INT8 QAT on GPT-6L/TinyShakespeare — extends the precision tetrahedron
to the transformer char-LM architecture.

Previous INT8 QAT (phase_int8_qat.py) was on ScatterMolGNN/QM9. This phase
applies identical STE fake-quant to the 4.8M-param GPT used in phase_xarch_lmc,
loading the FP32 ep80 checkpoint from GCS as the starting point.

LMC pairs computed:
  INT8-QAT ↔ FP32   (from phase_xarch_lmc)
  INT8-QAT ↔ BF16
  INT8-QAT ↔ FP16
  FP32 ↔ BF16       (x-val, from phase_xarch_lmc)
  FP32 ↔ FP16
  BF16 ↔ FP16

GCS output: gs://.../aegis_flashoptim/phase37_int8_gpt/results.json
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

GCS_BUCKET  = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID      = os.environ.get("RUN_ID", "aegis_flashoptim")
GCS_BASE    = f"{GCS_BUCKET}/{RUN_ID}"
GCS_P       = f"{GCS_BASE}/phase37_int8_gpt"
GCS_DONE    = f"{GCS_P}/results.json"
GCS_XARCH   = f"{GCS_BASE}/phase_xarch_lmc"   # source of fp32/bf16/fp16 ep80 checkpoints

OUT_DIR  = Path("/tmp/phase37_int8_gpt")
DATA_DIR = Path("/tmp/shakespeare")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# GPT-6L — identical to phase_xarch_lmc
N_EMBD       = 256
N_HEAD       = 8
N_LAYER      = 6
BLOCK_SIZE   = 256
DROPOUT      = 0.1
VOCAB_SIZE   = 65   # TinyShakespeare char-level

N_EPOCHS     = 80
QAT_WARMUP   = 15   # FP32 warm-up epochs before fake-quant
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


def log(msg): print(f"[Phase37] {msg}", flush=True)
def gsutil_cp(src, dst): subprocess.run(["gsutil", "-q", "cp", str(src), dst], check=False)
def gcs_exists(p): return subprocess.run(["gsutil", "-q", "stat", p], capture_output=True).returncode == 0


def fetch_ckpt(local_path, gcs_path):
    if not local_path.exists():
        if gcs_exists(gcs_path):
            log(f"  Fetching {local_path.name} from GCS...")
            subprocess.run(["gsutil", "-q", "cp", gcs_path, str(local_path)], check=False)
    return local_path.exists()


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
    return data[:n], data[n:]


def make_batches(data, batch_size, seq_len):
    stride = batch_size * seq_len
    n  = (len(data) - 1) // stride
    x  = data[:n * stride].view(batch_size, -1)
    y  = data[1:n * stride + 1].view(batch_size, -1)
    return list(zip(x.split(seq_len, dim=1), y.split(seq_len, dim=1)))


# ── GPT-6L (identical architecture to phase_xarch_lmc) ───────────────────────

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


class GPT6L(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok    = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.pos    = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.drop   = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f   = nn.LayerNorm(N_EMBD)
        self.head   = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


def get_device():
    return xm.xla_device() if XLA_AVAILABLE else torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_model(device=None):
    torch.manual_seed(SEED)
    m = GPT6L()
    return m.to(device) if device else m


def eval_loss(model, batches, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in batches[:30]:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total += F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1), reduction="sum").item()
            n += y.numel()
    if XLA_AVAILABLE and str(device) != "cpu":
        xm.mark_step()
    return total / n if n > 0 else float("inf")


# ── INT8 STE fake-quantization ────────────────────────────────────────────────

def fake_quantize_model(model):
    """Snap all FP32 parameters to INT8 grid in-place (XLA-safe)."""
    with torch.no_grad():
        for p in model.parameters():
            if p.dtype == torch.float32:
                scale = p.abs().max() / 127.0 + 1e-8
                q = (p / scale).round().clamp(-128, 127) * scale
                p.copy_(q)


# ── Training ──────────────────────────────────────────────────────────────────

def train_int8_qat(train_batches, val_batches):
    ckpt = OUT_DIR / "int8_qat_ep80.pt"
    if not ckpt.exists() and gcs_exists(f"{GCS_P}/int8_qat_ep80.pt"):
        subprocess.run(["gsutil", "-q", "cp", f"{GCS_P}/int8_qat_ep80.pt", str(ckpt)], check=False)

    if ckpt.exists():
        log("  INT8-QAT: loading from disk/GCS")
        dev = get_device()
        m   = make_model(dev)
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        m.to(dev)
        vl = eval_loss(m, val_batches, dev)
        log(f"  INT8-QAT: val={vl:.4f}  bpc={vl/math.log(2):.3f}  (resumed)")
        return {"val_loss": round(vl, 5), "bpc": round(vl / math.log(2), 4), "resumed": True}

    # Load FP32 ep80 from phase_xarch_lmc as warm-start
    fp32_src = OUT_DIR / "fp32_ep80_xarch.pt"
    fetch_ckpt(fp32_src, f"{GCS_XARCH}/fp32_ep80.pt")

    dev = get_device()
    m   = make_model(dev)
    if fp32_src.exists():
        log("  INT8-QAT: warm-starting from phase_xarch_lmc FP32 ep80")
        m.load_state_dict(torch.load(fp32_src, map_location="cpu"), strict=False)
        m.to(dev)
    else:
        log("  INT8-QAT: phase_xarch_lmc FP32 checkpoint not found — training from scratch")
        m.to(dev)

    log(f"  INT8-QAT: {sum(p.numel() for p in m.parameters()):,} params")
    log(f"  Schedule: ep1-{QAT_WARMUP} FP32 warm-up, ep{QAT_WARMUP+1}-{N_EPOCHS} STE fake-quant")

    opt = torch.optim.AdamW(m.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-5)

    best_val, best_sd = float("inf"), None

    for ep in range(1, N_EPOCHS + 1):
        qat_active = ep > QAT_WARMUP
        m.train()
        for x, y in train_batches:
            x, y = x.to(dev), y.to(dev)
            fp32_backup = None
            if qat_active:
                fp32_backup = {n: p.data.clone() for n, p in m.named_parameters()}
                fake_quantize_model(m)
            opt.zero_grad()
            loss = F.cross_entropy(m(x).reshape(-1, VOCAB_SIZE), y.reshape(-1))
            loss.backward()
            if qat_active:
                for n, p in m.named_parameters():
                    p.data.copy_(fp32_backup[n])
            nn.utils.clip_grad_norm_(m.parameters(), GRAD_CLIP)
            if XLA_AVAILABLE:
                xm.optimizer_step(opt)
            else:
                opt.step()
        sch.step()
        if XLA_AVAILABLE:
            xm.mark_step()
        if ep % LOG_EVERY == 0 or ep == N_EPOCHS:
            vl = eval_loss(m, val_batches, dev)
            status = "QAT-ON" if qat_active else "FP32-warmup"
            log(f"    ep{ep:3d}/{N_EPOCHS}  [{status}]  val={vl:.4f}  bpc={vl/math.log(2):.3f}  lr={opt.param_groups[0]['lr']:.2e}")
            if vl < best_val:
                best_val = vl
                best_sd  = {k: v.float().cpu().clone() for k, v in m.state_dict().items()}

    torch.save(best_sd, ckpt)
    gsutil_cp(ckpt, f"{GCS_P}/int8_qat_ep80.pt")
    log(f"  INT8-QAT: best_val={best_val:.4f}  bpc={best_val/math.log(2):.3f}")
    dev2 = get_device()
    m2   = make_model(dev2)
    m2.load_state_dict(best_sd); m2.to(dev2)
    vl_final = eval_loss(m2, val_batches, dev2)
    return {"val_loss": round(vl_final, 5), "bpc": round(vl_final / math.log(2), 4)}


# ── LMC ───────────────────────────────────────────────────────────────────────

def lmc_edge(label, sd0, sd1, val_batches):
    dev    = torch.device("cpu")
    m      = make_model(dev)
    alphas = [round(i / (N_ALPHA - 1), 2) for i in range(N_ALPHA)]
    losses = []
    for alpha in alphas:
        interp = {k: (1 - alpha) * sd0[k].float() + alpha * sd1[k].float() for k in sd0 if k in sd1}
        m.load_state_dict(interp, strict=False)
        loss = eval_loss(m, val_batches, dev)
        losses.append(round(loss, 5))
    base    = (losses[0] + losses[-1]) / 2
    barrier = round(max(losses) - base, 5)
    peak_a  = alphas[losses.index(max(losses))]
    log(f"  {label}: barrier={barrier:.5f} nats  ({barrier/math.log(2):.4f} bpc)  peak_α={peak_a}")
    return {"label": label, "alphas": alphas, "losses": losses,
            "barrier_nats": barrier, "barrier_bpc": round(barrier / math.log(2), 5),
            "peak_alpha": peak_a}


def load_sd(path): return torch.load(path, map_location="cpu")


def main():
    if gcs_exists(GCS_DONE):
        log("Results already in GCS — nothing to do."); return

    log("=" * 65)
    log("  Phase 37 — INT8 QAT on GPT-6L/TinyShakespeare")
    log(f"  N_EMBD={N_EMBD}  N_LAYER={N_LAYER}  N_EPOCHS={N_EPOCHS}  QAT_WARMUP={QAT_WARMUP}")
    log("=" * 65)
    notify("PHASE_START", "[Phase37] INT8 QAT GPT tetrahedron", data={})

    train_data, val_data = load_shakespeare()
    train_batches = make_batches(train_data, BATCH_SIZE, BLOCK_SIZE)
    val_batches   = make_batches(val_data,   BATCH_SIZE, BLOCK_SIZE)

    # Train INT8-QAT
    int8_result = train_int8_qat(train_batches, val_batches)

    # Fetch FP32 / BF16 / FP16 checkpoints from phase_xarch_lmc
    prec_ckpts = {}
    for prec in ["fp32", "bf16", "fp16"]:
        local = OUT_DIR / f"{prec}_xarch_ep80.pt"
        gcs   = f"{GCS_XARCH}/{prec}_ep80.pt"
        if fetch_ckpt(local, gcs):
            prec_ckpts[prec] = local
        else:
            log(f"  WARNING: {prec} checkpoint not found at {gcs}")

    # Load all state dicts
    int8_ckpt = OUT_DIR / "int8_qat_ep80.pt"
    sd_int8 = load_sd(int8_ckpt) if int8_ckpt.exists() else None
    sds = {prec: load_sd(p) for prec, p in prec_ckpts.items()}

    # Run all LMC pairs
    val_batches_cpu = make_batches(val_data, BATCH_SIZE, BLOCK_SIZE)
    lmc_results = []

    if sd_int8:
        for prec in ["fp32", "bf16", "fp16"]:
            if prec in sds:
                r = lmc_edge(f"INT8-QAT ↔ {prec.upper()}", sd_int8, sds[prec], val_batches_cpu)
                lmc_results.append(r)

    # Cross-val edges from phase_xarch_lmc
    for a, b in [("fp32", "bf16"), ("fp32", "fp16"), ("bf16", "fp16")]:
        if a in sds and b in sds:
            r = lmc_edge(f"{a.upper()} ↔ {b.upper()} (x-val)", sds[a], sds[b], val_batches_cpu)
            lmc_results.append(r)

    # Build tetrahedron summary
    barriers = {r["label"]: r["barrier_nats"] for r in lmc_results}
    log("\n=== INT8 TETRAHEDRON (GPT-6L/TinyShakespeare) ===")
    for label, barrier in barriers.items():
        log(f"  {label}: {barrier:.5f} nats  ({barrier/math.log(2):.4f} bpc)")

    results = {
        "experiment": "phase37_int8_gpt",
        "model": "GPT-6L (n_embd=256, n_layer=6, n_head=8, ~4.8M params)",
        "dataset": "tinyshakespeare_charlevel",
        "n_epochs": N_EPOCHS, "qat_warmup": QAT_WARMUP,
        "int8_qat_val": int8_result,
        "lmc_results": lmc_results,
        "tetrahedron_barriers": barriers,
    }
    out = OUT_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    gsutil_cp(out, GCS_DONE)
    log("Results → GCS phase37_int8_gpt/results.json")
    notify("PHASE_COMPLETE", "[Phase37] INT8 GPT tetrahedron done", data={"barriers": barriers})


if __name__ == "__main__":
    main()
