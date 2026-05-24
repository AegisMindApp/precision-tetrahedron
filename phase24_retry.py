#!/usr/bin/env python3
"""
phase24_retry.py
----------------
INT8 weight tetrahedron — compute LMC between all 4 precision vertices.

Downloads all 4 checkpoints from GCS:
  FP32:     phase_xarch_lmc/fp32_ep80.pt
  BF16:     phase_xarch_lmc/bf16_ep80.pt
  FP16:     phase_xarch_lmc/fp16_ep80.pt
  INT8-QAT: phase_int8_qat/int8qat_ep80.pt

Computes all 6 edges of the tetrahedron (LMC barrier) on CPU:
  FP32 ↔ BF16
  FP32 ↔ FP16
  FP32 ↔ INT8
  BF16 ↔ FP16
  BF16 ↔ INT8
  FP16 ↔ INT8

GCS output: gs://.../aegis_flashoptim/phase24_retry/results.json
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
GCS_BASE   = f"{GCS_BUCKET}/{RUN_ID}"
GCS_P      = f"{GCS_BASE}/phase24_retry"

OUT_DIR  = Path("/tmp/phase24_retry")
DATA_DIR = Path("/tmp/shakespeare")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Same GPT arch as phase_xarch_lmc.py
N_EMBD       = 256
N_HEAD       = 8
N_LAYER      = 6
BLOCK_SIZE   = 256
DROPOUT      = 0.1
BATCH_SIZE   = 64
SEED         = 42
N_ALPHA      = 11
VOCAB_SIZE   = 65   # TinyShakespeare char-level default

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)

# GCS sources for each checkpoint
CHECKPOINT_SOURCES = {
    "fp32":  [f"{GCS_BASE}/phase_xarch_lmc/fp32_ep80.pt"],
    "bf16":  [f"{GCS_BASE}/phase_xarch_lmc/bf16_ep80.pt"],
    "fp16":  [f"{GCS_BASE}/phase_xarch_lmc/fp16_ep80.pt"],
    "int8":  [f"{GCS_BASE}/phase_int8_qat/int8qat_ep80.pt",
              f"{GCS_BASE}/phase_int8_qat/int8_qat_ep80.pt"],
}

TETRAHEDRON_EDGES = [
    ("FP32 ↔ BF16",  "fp32", "bf16"),
    ("FP32 ↔ FP16",  "fp32", "fp16"),
    ("FP32 ↔ INT8",  "fp32", "int8"),
    ("BF16 ↔ FP16",  "bf16", "fp16"),
    ("BF16 ↔ INT8",  "bf16", "int8"),
    ("FP16 ↔ INT8",  "fp16", "int8"),
]


def log(msg):
    print(f"[Phase24] {msg}", flush=True)


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


# ── Model (identical to phase_xarch_lmc.py) ──────────────────────────────────

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
def eval_loss(model, batches, device=None):
    device = device or torch.device("cpu")
    model.eval()
    total, n = 0.0, 0
    for x, y in batches:
        x, y   = x.to(device), y.to(device)
        logits = model(x)
        total += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        ).item()
        n += y.numel()
    return total / n if n > 0 else float("inf")


# ── LMC ───────────────────────────────────────────────────────────────────────

def lmc_barrier(label, p0, p1, vocab_size, val_batches):
    sd0 = torch.load(p0, map_location="cpu")
    sd1 = torch.load(p1, map_location="cpu")
    dev = torch.device("cpu")
    m   = make_model(vocab_size)

    alphas = [round(i / (N_ALPHA - 1), 2) for i in range(N_ALPHA)]
    losses = []

    for alpha in alphas:
        # Float-cast both sides before interpolation (INT8-QAT may have int tensors)
        interp = {k: (1 - alpha) * sd0[k].float() + alpha * sd1[k].float()
                  for k in sd0 if k in sd1}
        m.load_state_dict(interp, strict=False)
        loss = eval_loss(m, val_batches, dev)
        losses.append(round(loss, 5))
        log(f"    {label}  α={alpha:.1f}  loss={losses[-1]:.5f}  bpc={losses[-1]/math.log(2):.4f}")

    base    = min(losses[0], losses[-1])
    barrier = round(max(losses) - base, 5)
    peak_a  = alphas[losses.index(max(losses))]
    log(f"  → {label}  barrier={barrier:.5f} nats  bpc={barrier/math.log(2):.5f}  peak_α={peak_a}")
    return {"label": label, "alphas": alphas, "losses": losses,
            "barrier_nats": barrier, "barrier_bpc": round(barrier / math.log(2), 5),
            "baseline_nats": base, "peak_alpha": peak_a}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 65)
    log("  Phase 24 Retry — INT8 Weight Tetrahedron LMC")
    log("  6 edges: FP32/BF16/FP16/INT8 all pairs")
    log("=" * 65)

    notify("PHASE_START", "[Phase24] INT8 tetrahedron LMC", data={})

    # ── Download checkpoints ──────────────────────────────────────────────────
    local_ckpts = {}
    for key, gcs_sources in CHECKPOINT_SOURCES.items():
        local_path = OUT_DIR / f"{key}_ep80.pt"
        if local_path.exists():
            log(f"  {key}: on disk")
            local_ckpts[key] = local_path
            continue
        found = False
        for gcs in gcs_sources:
            if gcs_exists(gcs):
                log(f"  {key}: downloading from {gcs}")
                subprocess.run(["gsutil", "-q", "cp", gcs, str(local_path)], check=False)
                if local_path.exists():
                    local_ckpts[key] = local_path
                    found = True
                    break
        if not found:
            log(f"  WARNING: {key} checkpoint not found in GCS — will skip edges involving it")

    available = list(local_ckpts.keys())
    log(f"\nAvailable checkpoints: {available}")

    if len(available) < 2:
        log("ERROR: fewer than 2 checkpoints available — cannot compute any LMC edge")
        sys.exit(1)

    # ── Load val data ─────────────────────────────────────────────────────────
    _, val_data, vocab_size = load_shakespeare()
    vs_path = OUT_DIR / "vocab_size.txt"
    if vs_path.exists():
        vocab_size = int(vs_path.read_text().strip())
    # Also check shared location
    shared_vs = Path("/tmp/phase_xarch_lmc/vocab_size.txt")
    if shared_vs.exists():
        vocab_size = int(shared_vs.read_text().strip())
    val_batches = make_batches(val_data, BATCH_SIZE, BLOCK_SIZE)
    log(f"vocab_size={vocab_size}  val_batches={len(val_batches)}")

    # ── Evaluate each vertex loss ─────────────────────────────────────────────
    log("\nEvaluating vertex val losses...")
    vertex_losses = {}
    m = make_model(vocab_size)
    for key, ckpt in local_ckpts.items():
        sd  = torch.load(ckpt, map_location="cpu")
        # Convert any non-float tensors to float32
        sd  = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
        m.load_state_dict(sd, strict=False)
        vl  = eval_loss(m, val_batches)
        bpc = vl / math.log(2)
        vertex_losses[key] = {"val_loss": round(vl, 5), "bpc": round(bpc, 4)}
        log(f"  {key:8s}  val_loss={vl:.4f}  bpc={bpc:.4f}")

    # ── Compute all available edges ───────────────────────────────────────────
    log("\nComputing LMC tetrahedron edges (CPU)...")
    edge_results = []
    skipped_edges = []

    for label, k0, k1 in TETRAHEDRON_EDGES:
        if k0 not in local_ckpts or k1 not in local_ckpts:
            log(f"  Skipping {label}: checkpoint missing")
            skipped_edges.append(label)
            continue
        log(f"\n  Edge: {label}")
        result = lmc_barrier(label, local_ckpts[k0], local_ckpts[k1], vocab_size, val_batches)
        edge_results.append(result)
        heartbeat("Phase24_LMC", len(edge_results), {"edge": label, "barrier": result["barrier_nats"]})

    # ── Tetrahedron analysis ──────────────────────────────────────────────────
    log("\n" + "=" * 65)
    log("  TETRAHEDRON LMC RESULTS")
    for r in edge_results:
        log(f"  {r['label']:30s}  {r['barrier_nats']:.5f} nats  ({r['barrier_bpc']:.5f} bpc)")
    if skipped_edges:
        log(f"\n  Skipped (missing ckpts): {skipped_edges}")

    # Analysis: is INT8 vertex isolated?
    int8_edges   = [r for r in edge_results if "INT8" in r["label"]]
    other_edges  = [r for r in edge_results if "INT8" not in r["label"]]

    int8_isolated = False
    interpretation = ""

    if int8_edges and other_edges:
        int8_mean  = sum(r["barrier_nats"] for r in int8_edges) / len(int8_edges)
        other_mean = sum(r["barrier_nats"] for r in other_edges) / len(other_edges)
        ratio      = int8_mean / (other_mean + 1e-8)
        int8_isolated = ratio > 3.0

        log(f"\n  INT8 edge mean barrier:   {int8_mean:.5f} nats")
        log(f"  Non-INT8 edge mean barrier: {other_mean:.5f} nats")
        log(f"  INT8 isolation ratio:       {ratio:.2f}×")
        log(f"  INT8 vertex isolated:       {'YES' if int8_isolated else 'NO'}")

        interpretation = (
            f"INT8 vertex is ISOLATED (ratio={ratio:.1f}×): "
            f"INT8-QAT weights lie in a distinct basin from FP32/BF16/FP16 weights. "
            f"Mean INT8-edge barrier={int8_mean:.5f} vs non-INT8={other_mean:.5f} nats."
            if int8_isolated else
            f"INT8 vertex is NOT isolated (ratio={ratio:.1f}×): "
            f"INT8-QAT weights share the same loss landscape as higher-precision models. "
            f"Mean INT8-edge barrier={int8_mean:.5f} vs non-INT8={other_mean:.5f} nats."
        )
        log(f"\n  {interpretation}")

    elif len(edge_results) == 0:
        interpretation = "No edges computed — all checkpoints missing."
    else:
        barriers = sorted([r["barrier_nats"] for r in edge_results])
        interpretation = (
            f"{len(edge_results)} edges computed. "
            f"Barrier range: {barriers[0]:.5f} – {barriers[-1]:.5f} nats. "
            f"Incomplete tetrahedron — INT8 ckpt not available for full analysis."
        )

    results = {
        "experiment":    "phase24_retry_int8_tetrahedron",
        "model":         {"n_layer": N_LAYER, "n_embd": N_EMBD, "n_head": N_HEAD},
        "available":     available,
        "skipped_edges": skipped_edges,
        "vertex_losses": vertex_losses,
        "edges":         edge_results,
        "tetrahedron_analysis": {
            "int8_edges_mean_barrier":   round(sum(r["barrier_nats"] for r in int8_edges) / max(len(int8_edges), 1), 5),
            "other_edges_mean_barrier":  round(sum(r["barrier_nats"] for r in other_edges) / max(len(other_edges), 1), 5),
            "int8_vertex_isolated":      int8_isolated,
        },
        "interpretation": interpretation,
    }

    out = OUT_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    gsutil_cp(out, f"{GCS_P}/results.json")
    log(f"\nResults → GCS phase24_retry/results.json")
    log("=== phase24 retry complete ===")

    notify("PHASE_COMPLETE", "[Phase24] INT8 tetrahedron LMC done",
           data={"n_edges": len(edge_results), "int8_isolated": int8_isolated,
                 "available": available})


if __name__ == "__main__":
    main()
