#!/usr/bin/env python3
"""
phase33_finetune_lmc.py
-----------------------
GPT-2 124M fine-tune LMC.

Fine-tunes GPT-2 (124M) from HuggingFace on TinyShakespeare for 10 epochs,
saves FP32 and BF16 checkpoints, then computes LMC barrier (11 α points).

NOTE: Do NOT import torchvision — it conflicts with transformers on this VM.

GCS output: gs://.../aegis_flashoptim/phase33_finetune_lmc/results.json
"""

import os, sys, json, math, subprocess, urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from notify import notify, heartbeat

GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID     = os.environ.get("RUN_ID", "aegis_flashoptim")
GCS_P      = f"{GCS_BUCKET}/{RUN_ID}/phase33_finetune_lmc"

OUT_DIR  = Path("/tmp/phase33_finetune_lmc")
DATA_DIR = Path("/tmp/shakespeare")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

N_EPOCHS     = 10
LR_INIT      = 5e-5   # conservative fine-tune LR
WEIGHT_DECAY = 0.01
GRAD_CLIP    = 1.0
BATCH_SIZE   = 8      # small batch — GPT-2 124M memory
BLOCK_SIZE   = 512    # GPT-2 default context
SEED         = 42
N_ALPHA      = 11
LOG_EVERY    = 1

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)


def log(msg):
    print(f"[Phase33] {msg}", flush=True)


def gsutil_cp(src, dst):
    subprocess.run(["gsutil", "-q", "cp", str(src), dst], check=False)


def gcs_exists(gcs_path):
    return subprocess.run(["gsutil", "-q", "stat", gcs_path],
                          capture_output=True).returncode == 0


# ── Data ──────────────────────────────────────────────────────────────────────

def load_shakespeare_tokens():
    """Tokenise TinyShakespeare using GPT-2 tokenizer."""
    from transformers import GPT2Tokenizer
    path = DATA_DIR / "input.txt"
    if not path.exists():
        log("Downloading TinyShakespeare...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, str(path))

    text = path.read_text(encoding="utf-8")
    tok  = GPT2Tokenizer.from_pretrained("gpt2")
    ids  = tok.encode(text)
    data = torch.tensor(ids, dtype=torch.long)
    n    = int(0.9 * len(data))
    log(f"Tokenised: {len(data)} tokens  (train={n}  val={len(data)-n})")
    return data[:n], data[n:], tok.vocab_size


def make_batches(data, batch_size, seq_len):
    stride = batch_size * seq_len
    n  = (len(data) - 1) // stride
    if n == 0:
        return []
    x  = data[:n * stride].view(batch_size, -1)
    y  = data[1:n * stride + 1].view(batch_size, -1)
    return list(zip(x.split(seq_len, dim=1), y.split(seq_len, dim=1)))


# ── Training ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_loss(model, batches, device):
    model.eval()
    total, n = 0.0, 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        out   = model(x, labels=y)
        total += out.loss.item() * y.numel()
        n     += y.numel()
    return total / n if n > 0 else float("inf")


def fine_tune(precision, train_batches, val_batches, device):
    from transformers import GPT2LMHeadModel

    ckpt_name = f"{precision}_ft_ep{N_EPOCHS}.pt"
    ckpt      = OUT_DIR / ckpt_name
    gcs_ckpt  = f"{GCS_P}/{ckpt_name}"

    if ckpt.exists():
        log(f"{precision}: on disk — skip")
        return ckpt
    if gcs_exists(gcs_ckpt):
        log(f"{precision}: GCS — downloading")
        subprocess.run(["gsutil", "-q", "cp", gcs_ckpt, str(ckpt)], check=False)
        if ckpt.exists():
            return ckpt

    log(f"{precision}: fine-tuning GPT-2 on TinyShakespeare...")
    torch.manual_seed(SEED)
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = model.to(device)

    use_bf16 = (precision == "bf16")
    opt = torch.optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-6)

    for ep in range(1, N_EPOCHS + 1):
        model.train()
        total_loss, n = 0.0, 0
        for x, y in train_batches:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if use_bf16:
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    out  = model(x, labels=y)
                    loss = out.loss
            else:
                out  = model(x, labels=y)
                loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            total_loss += loss.item()
            n          += 1
        sch.step()
        vl = eval_loss(model, val_batches, device)
        log(f"  {precision} ep{ep:2d}  train_loss={total_loss/max(n,1):.4f}  "
            f"val_loss={vl:.4f}  bpc={vl/math.log(2):.3f}")
        heartbeat(f"Phase33_{precision}", ep, {"val_loss": vl})

    # Save FP32 state dict regardless of training precision
    sd = {k: v.cpu().float().clone() for k, v in model.state_dict().items()}
    torch.save(sd, ckpt)
    gsutil_cp(ckpt, gcs_ckpt)
    log(f"{precision}: saved → {ckpt}")
    return ckpt


# ── LMC ───────────────────────────────────────────────────────────────────────

def lmc_barrier(p0, p1, val_batches, device):
    from transformers import GPT2LMHeadModel

    sd0 = torch.load(p0, map_location="cpu")
    sd1 = torch.load(p1, map_location="cpu")

    alphas = [round(i / (N_ALPHA - 1), 2) for i in range(N_ALPHA)]
    losses = []

    # Work on CPU for LMC
    cpu = torch.device("cpu")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = model.to(cpu)

    for alpha in alphas:
        interp = {k: (1 - alpha) * sd0[k].float() + alpha * sd1[k].float()
                  for k in sd0 if k in sd1}
        model.load_state_dict(interp, strict=False)
        loss = eval_loss(model, val_batches, cpu)
        losses.append(round(loss, 5))
        log(f"  α={alpha:.1f}  loss={losses[-1]:.5f}  bpc={losses[-1]/math.log(2):.4f}")

    base    = min(losses[0], losses[-1])
    barrier = round(max(losses) - base, 5)
    peak_a  = alphas[losses.index(max(losses))]
    log(f"  → LMC barrier={barrier:.5f} nats  bpc={barrier/math.log(2):.5f}  peak_α={peak_a}")
    return {"alphas": alphas, "losses": losses, "barrier_nats": barrier,
            "barrier_bpc": round(barrier / math.log(2), 5),
            "baseline_nats": base, "peak_alpha": peak_a}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 65)
    log("  Phase 33 — GPT-2 124M Fine-tune LMC")
    log(f"  N_EPOCHS={N_EPOCHS}  LR={LR_INIT}  BATCH={BATCH_SIZE}")
    log("=" * 65)

    notify("PHASE_START", "[Phase33] GPT-2 124M fine-tune LMC", data={})

    device = torch.device("cpu")  # GPT-2 fine-tune on CPU (no XLA for HuggingFace)
    log(f"Device: {device}")

    # Check if both checkpoints already exist
    fp32_ckpt = OUT_DIR / f"fp32_ft_ep{N_EPOCHS}.pt"
    bf16_ckpt = OUT_DIR / f"bf16_ft_ep{N_EPOCHS}.pt"

    fp32_gcs = f"{GCS_P}/fp32_ft_ep{N_EPOCHS}.pt"
    bf16_gcs = f"{GCS_P}/bf16_ft_ep{N_EPOCHS}.pt"

    for local, gcs in [(fp32_ckpt, fp32_gcs), (bf16_ckpt, bf16_gcs)]:
        if not local.exists() and gcs_exists(gcs):
            subprocess.run(["gsutil", "-q", "cp", gcs, str(local)], check=False)

    both_exist = fp32_ckpt.exists() and bf16_ckpt.exists()

    if both_exist:
        log("Both checkpoints exist — skipping training, computing LMC only")
    else:
        # Load data
        train_data, val_data, _ = load_shakespeare_tokens()
        train_batches = make_batches(train_data, BATCH_SIZE, BLOCK_SIZE)
        val_batches   = make_batches(val_data,   BATCH_SIZE, BLOCK_SIZE)
        log(f"Train batches: {len(train_batches)}  Val batches: {len(val_batches)}")

        if not fp32_ckpt.exists():
            fine_tune("fp32", train_batches, val_batches, device)
        if not bf16_ckpt.exists():
            fine_tune("bf16", train_batches, val_batches, device)

    # Always reload val data for LMC
    train_data, val_data, _ = load_shakespeare_tokens()
    val_batches = make_batches(val_data, BATCH_SIZE, BLOCK_SIZE)

    # Evaluate final losses
    from transformers import GPT2LMHeadModel

    def _load_and_eval(ckpt_path):
        m = GPT2LMHeadModel.from_pretrained("gpt2")
        sd = torch.load(ckpt_path, map_location="cpu")
        m.load_state_dict(sd, strict=False)
        vl = eval_loss(m, val_batches, torch.device("cpu"))
        return round(vl, 5), round(vl / math.log(2), 4)

    fp32_vl, fp32_bpc = _load_and_eval(fp32_ckpt)
    bf16_vl, bf16_bpc = _load_and_eval(bf16_ckpt)
    log(f"\nFP32 fine-tune: val_loss={fp32_vl}  bpc={fp32_bpc}")
    log(f"BF16 fine-tune: val_loss={bf16_vl}  bpc={bf16_bpc}")

    # LMC
    log("\nComputing LMC: FP32 ↔ BF16 (CPU)...")
    lmc = lmc_barrier(fp32_ckpt, bf16_ckpt, val_batches, device)

    results = {
        "experiment":     "phase33_finetune_lmc",
        "model":          "gpt2_124M",
        "n_epochs":       N_EPOCHS,
        "fp32_val_loss":  fp32_vl,
        "fp32_bpc":       fp32_bpc,
        "bf16_val_loss":  bf16_vl,
        "bf16_bpc":       bf16_bpc,
        "lmc_barrier_nats": lmc["barrier_nats"],
        "lmc_barrier_bpc":  lmc["barrier_bpc"],
        "lmc_curve":        lmc,
        "interpretation": (
            f"GPT-2 124M FP32 vs BF16 fine-tune LMC barrier: {lmc['barrier_nats']:.5f} nats "
            f"({lmc['barrier_bpc']:.5f} bpc). "
            + ("Near-zero barrier confirms FP32/BF16 share the same loss basin at 124M scale."
               if lmc["barrier_nats"] < 0.01 else
               "Non-trivial barrier — larger models may exhibit more precision-sensitive basins.")
        ),
    }

    out = OUT_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    gsutil_cp(out, f"{GCS_P}/results.json")
    log(f"\nResults → GCS phase33_finetune_lmc/results.json")
    log(f"  LMC barrier: {lmc['barrier_nats']:.5f} nats")
    log("=== phase33 complete ===")

    notify("PHASE_COMPLETE", "[Phase33] GPT-2 LMC done",
           data={"barrier_nats": lmc["barrier_nats"],
                 "fp32_bpc": fp32_bpc, "bf16_bpc": bf16_bpc})


if __name__ == "__main__":
    main()
