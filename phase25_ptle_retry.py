#!/usr/bin/env python3
"""
phase25_ptle_retry.py
---------------------
PTLE (Precision-Toggling Learning Extension) warm restart — BF16 arm.

Downloads:
  - Source checkpoint: phase25_ptle_restart/fp32_ep80_source.pt
    (fallback: phase_xarch_lmc/fp32_ep80.pt)
  - Arm A:            phase25_ptle_restart/fp32_continued_ep80.pt

Runs arm B: 4 cycles of [BF16 20 epochs → FP32 20 epochs] = 80 total epochs.
Computes LMC between arm A (fp32_continued) and arm B (ptle).
Updates existing results.json in GCS.

Hypothesis: PTLE arm B reaches a flatter basin (lower final val loss) than
arm A (fp32 continued) — BF16 cycles act as implicit regularisation/noise injection.

GCS output: phase25_ptle_restart/results.json (updated)
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
GCS_P      = f"{GCS_BASE}/phase25_ptle_restart"
GCS_XARCH  = f"{GCS_BASE}/phase_xarch_lmc"

OUT_DIR  = Path("/tmp/phase25_ptle_retry")
DATA_DIR = Path("/tmp/shakespeare")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Same arch as phase_xarch_lmc.py ──────────────────────────────────────────
N_EMBD       = 256
N_HEAD       = 8
N_LAYER      = 6
BLOCK_SIZE   = 256
DROPOUT      = 0.1
BATCH_SIZE   = 64
LR_INIT      = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP    = 1.0
SEED         = 42
N_ALPHA      = 11
LOG_EVERY    = 5

# PTLE config
PTLE_CYCLES      = 4        # 4 cycles
EPOCHS_PER_CYCLE = 20       # 20 epochs per half-cycle
PTLE_ORDER       = ["bf16", "fp32", "bf16", "fp32"]  # 4 half-cycles

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)


def log(msg):
    print(f"[Phase25] {msg}", flush=True)


def gsutil_cp(src, dst):
    subprocess.run(["gsutil", "-q", "cp", str(src), dst], check=False)


def gcs_exists(gcs_path):
    return subprocess.run(["gsutil", "-q", "stat", gcs_path],
                          capture_output=True).returncode == 0


def try_download(local_path, *gcs_paths):
    """Download first available GCS path to local_path."""
    if local_path.exists():
        return True
    for gcs in gcs_paths:
        if gcs_exists(gcs):
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


# ── PTLE arm B training ───────────────────────────────────────────────────────

def train_ptle_arm_b(source_sd, vocab_size, train_batches, val_batches, device):
    """
    4 cycles of [BF16 20 epochs → FP32 20 epochs] starting from source checkpoint.
    Returns (final_model, final_val_loss, cycle_logs).
    """
    model = make_model(vocab_size, device)
    model.load_state_dict(source_sd)

    cycle_logs = []
    global_epoch = 0

    for cycle_i, precision in enumerate(PTLE_ORDER):
        use_bf16 = (precision == "bf16")
        log(f"\n  PTLE Cycle {cycle_i+1}/{len(PTLE_ORDER)}: {precision} ({EPOCHS_PER_CYCLE} epochs)")

        # Fresh optimizer each half-cycle (warm restart)
        n_total_remaining = (len(PTLE_ORDER) - cycle_i) * EPOCHS_PER_CYCLE
        opt = torch.optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=EPOCHS_PER_CYCLE, eta_min=1e-5
        )

        for ep in range(1, EPOCHS_PER_CYCLE + 1):
            model.train()
            for x, y in train_batches:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                if use_bf16:
                    with torch.autocast(
                        device_type="xla" if XLA_AVAILABLE else "cpu",
                        dtype=torch.bfloat16
                    ):
                        loss = F.cross_entropy(model(x).reshape(-1, vocab_size), y.reshape(-1))
                else:
                    loss = F.cross_entropy(model(x).reshape(-1, vocab_size), y.reshape(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                if XLA_AVAILABLE:
                    xm.optimizer_step(opt)
                else:
                    opt.step()
            sch.step()
            if XLA_AVAILABLE:
                xm.mark_step()
            global_epoch += 1

            if ep % LOG_EVERY == 0 or ep == EPOCHS_PER_CYCLE:
                vl = eval_loss(model, val_batches, device)
                log(f"    cycle {cycle_i+1} ({precision}) ep{ep:3d}  "
                    f"val_loss={vl:.4f}  bpc={vl/math.log(2):.3f}  "
                    f"lr={opt.param_groups[0]['lr']:.2e}")
                heartbeat(f"Phase25_PTLE_armB", global_epoch, {"val_loss": vl, "cycle": cycle_i+1})

        vl = eval_loss(model, val_batches, device)
        cycle_logs.append({"cycle": cycle_i + 1, "precision": precision,
                            "final_val_loss": round(vl, 5), "bpc": round(vl / math.log(2), 5)})
        log(f"  Cycle {cycle_i+1} ({precision}) DONE  val_loss={vl:.4f}")

    final_vl = eval_loss(model, val_batches, device)
    return model, final_vl, cycle_logs


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
    log("  Phase 25 PTLE Retry — BF16 Arm B")
    log(f"  Cycles: {PTLE_ORDER}  ({EPOCHS_PER_CYCLE} epochs/cycle)")
    log("=" * 65)

    notify("PHASE_START", "[Phase25] PTLE BF16 arm B retry", data={})

    device = get_device()
    log(f"Device: {device}")

    # ── Download source checkpoint ────────────────────────────────────────────
    src_local = OUT_DIR / "fp32_ep80_source.pt"
    ok = try_download(src_local,
                      f"{GCS_P}/fp32_ep80_source.pt",
                      f"{GCS_XARCH}/fp32_ep80.pt")
    if not ok:
        log("ERROR: source checkpoint not found in GCS")
        sys.exit(1)
    log(f"Source checkpoint: {src_local}")

    # ── Download arm A ────────────────────────────────────────────────────────
    arm_a_local = OUT_DIR / "fp32_continued_ep80.pt"
    ok_a = try_download(arm_a_local, f"{GCS_P}/fp32_continued_ep80.pt")
    if not ok_a:
        log("WARNING: arm A checkpoint not found — LMC comparison will be skipped")

    # ── Load data ─────────────────────────────────────────────────────────────
    train_data, val_data, vocab_size = load_shakespeare()
    vs_path = OUT_DIR / "vocab_size.txt"
    shared_vs = Path("/tmp/phase_xarch_lmc/vocab_size.txt")
    if shared_vs.exists():
        vocab_size = int(shared_vs.read_text().strip())
    vs_path.write_text(str(vocab_size))

    train_batches = make_batches(train_data, BATCH_SIZE, BLOCK_SIZE)
    val_batches   = make_batches(val_data,   BATCH_SIZE, BLOCK_SIZE)

    # ── Check if arm B already done ───────────────────────────────────────────
    arm_b_local = OUT_DIR / "ptle_arm_b_ep80.pt"
    arm_b_done  = try_download(arm_b_local, f"{GCS_P}/ptle_arm_b_ep80.pt")

    if arm_b_done:
        log("Arm B checkpoint already exists — loading")
        arm_b_sd   = torch.load(arm_b_local, map_location="cpu")
        m = make_model(vocab_size, torch.device("cpu"))
        m.load_state_dict(arm_b_sd)
        arm_b_vl = eval_loss(m, val_batches, torch.device("cpu"))
        arm_b_cycle_logs = []
        log(f"Arm B val_loss={arm_b_vl:.4f}  bpc={arm_b_vl/math.log(2):.4f}")
    else:
        log("\nTraining PTLE arm B...")
        source_sd = torch.load(src_local, map_location="cpu")
        model_b, arm_b_vl, arm_b_cycle_logs = train_ptle_arm_b(
            source_sd, vocab_size, train_batches, val_batches, device
        )
        arm_b_sd = {k: v.cpu().clone() for k, v in model_b.state_dict().items()}
        torch.save(arm_b_sd, arm_b_local)
        gsutil_cp(arm_b_local, f"{GCS_P}/ptle_arm_b_ep80.pt")
        log(f"\nArm B DONE  val_loss={arm_b_vl:.4f}  bpc={arm_b_vl/math.log(2):.4f}")

    # ── Arm A val loss ────────────────────────────────────────────────────────
    arm_a_vl = None
    if arm_a_local.exists():
        m = make_model(vocab_size, torch.device("cpu"))
        m.load_state_dict(torch.load(arm_a_local, map_location="cpu"))
        arm_a_vl = eval_loss(m, val_batches, torch.device("cpu"))
        log(f"Arm A (fp32_continued) val_loss={arm_a_vl:.4f}  bpc={arm_a_vl/math.log(2):.4f}")

    # ── LMC between arm A and arm B ───────────────────────────────────────────
    lmc_result = None
    if arm_a_local.exists() and arm_b_local.exists():
        log("\nComputing LMC: arm A (fp32_continued) ↔ arm B (ptle_bf16) ...")
        lmc_result = lmc_barrier(
            "FP32-continued ↔ PTLE-BF16",
            arm_a_local, arm_b_local,
            vocab_size, val_batches
        )

    # ── Delta analysis ────────────────────────────────────────────────────────
    delta_val_loss = None
    hypothesis_supported = None

    if arm_a_vl is not None:
        delta_val_loss = arm_b_vl - arm_a_vl
        hypothesis_supported = delta_val_loss < 0

        if hypothesis_supported:
            log(f"\n  PTLE arm B BETTER: Δval_loss={delta_val_loss:+.5f} "
                f"(arm_b={arm_b_vl:.4f} < arm_a={arm_a_vl:.4f})")
            log("  BF16 cycles found a flatter basin — HYPOTHESIS SUPPORTED")
        else:
            log(f"\n  PTLE arm B NO IMPROVEMENT: Δval_loss={delta_val_loss:+.5f} "
                f"(arm_b={arm_b_vl:.4f} vs arm_a={arm_a_vl:.4f})")
            log("  BF16 cycles did not improve over FP32 continuation")

    # ── Load and update existing results.json ────────────────────────────────
    results_local = OUT_DIR / "results.json"
    gcs_results   = f"{GCS_P}/results.json"
    existing = {}
    if gcs_exists(gcs_results):
        subprocess.run(["gsutil", "-q", "cp", gcs_results, str(results_local)], check=False)
        if results_local.exists():
            with open(results_local) as f:
                existing = json.load(f)

    existing["arm_b"] = {
        "description":   "PTLE: 4 cycles [BF16 20ep → FP32 20ep]",
        "cycle_order":   PTLE_ORDER,
        "epochs_per_cycle": EPOCHS_PER_CYCLE,
        "final_val_loss": round(arm_b_vl, 5),
        "final_bpc":      round(arm_b_vl / math.log(2), 5),
        "cycle_logs":     arm_b_cycle_logs,
    }
    if arm_a_vl is not None:
        existing["arm_a_val_loss"] = round(arm_a_vl, 5)
    if delta_val_loss is not None:
        existing["delta_val_loss"] = round(delta_val_loss, 5)
        existing["hypothesis_supported"] = hypothesis_supported
    if lmc_result is not None:
        existing["lmc_arm_a_vs_arm_b"] = lmc_result

    results_local.write_text(json.dumps(existing, indent=2))
    gsutil_cp(results_local, gcs_results)
    log(f"\nResults updated → GCS phase25_ptle_restart/results.json")
    log("=== phase25 PTLE retry complete ===")

    notify("PHASE_COMPLETE", "[Phase25] PTLE arm B done",
           data={"arm_b_vl": round(arm_b_vl, 4),
                 "arm_a_vl": round(arm_a_vl, 4) if arm_a_vl else None,
                 "delta": round(delta_val_loss, 5) if delta_val_loss is not None else None,
                 "hypothesis_supported": hypothesis_supported,
                 "lmc_barrier": lmc_result["barrier_nats"] if lmc_result else None})


if __name__ == "__main__":
    main()
