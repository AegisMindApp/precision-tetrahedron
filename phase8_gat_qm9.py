#!/usr/bin/env python3
"""
phase8_gat_qm9.py
-----------------
Train a Graph Attention Transformer (MolecularTransformer) on QM9 in BF16.

Purpose: test whether the BF16 sharpening / LR self-calibration result
generalises beyond MolecularGNN to a second, independently defined architecture.

Pipeline
--------
1.  Setup : build MolecularTransformer (BF16, hidden_dim=256, 6 blocks)
2.  Train : 100 epochs BF16, AdamW lr=1e-4, CosineAnnealingLR T_max=100
            Eval every 5 epochs; auto-detect plateau (MAE improvement
            < 0.0005 eV for 15 consecutive eval steps)
3.  Warm restart: fresh optimizer at lr=5e-5, CosineAnnealingLR T_max=20
            Save checkpoint at every post-plateau eval step + at plateau
4.  LMC   : measure Linear Mode Connectivity between
            (plateau epoch) and (plateau epoch + 3)
            Compare with MolecularGNN BF16-256 barrier (1.447 eV)
5.  Upload results.json + checkpoints to gs://.../phase8_gat_qm9/

GCS output: gs://aegismind-tpu-results/aegis_flashoptim/phase8_gat_qm9/
"""

import os, sys, json, time, subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── XLA ──────────────────────────────────────────────────────────────────────
try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
    try:
        import torch_xla.experimental as _e
        _e.eager_mode(True)
        print("XLA eager mode: ENABLED", flush=True)
    except Exception as _xe:
        print(f"XLA eager mode unavailable: {_xe}", flush=True)
except ImportError:
    XLA_AVAILABLE = False

# ── Local imports (TPU VM at ~/flashoptim/) ───────────────────────────────────
sys.path.insert(0, os.path.expanduser("~/flashoptim"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# NOTE: model.py is NOT imported — MolecularTransformer is defined below
from data import get_dataloaders, batch_to_graph
from notify import notify, heartbeat

# ── Config ────────────────────────────────────────────────────────────────────
GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID     = os.environ.get("RUN_ID",     "aegis_flashoptim")
GCS_BASE   = f"{GCS_BUCKET}/{RUN_ID}"

OUT_DIR  = Path("/tmp/phase8_gat_qm9")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("/tmp/qm9")

HIDDEN_DIM        = 256
N_BLOCKS          = 6
N_GAUSSIANS       = 50
CUTOFF            = 5.0
BATCH_SIZE        = 32
N_EPOCHS          = 100
LR_INIT           = 1e-4
WEIGHT_DECAY      = 1e-4
ETA_MIN_MAIN      = 1e-6
EVAL_EVERY        = 5       # eval every N epochs
PLATEAU_PATIENCE  = 15      # eval steps without >= 0.0005 eV improvement
PLATEAU_DELTA     = 5e-4    # min improvement to reset patience counter
RESTART_LR        = 5e-5
RESTART_T_MAX     = 20
RESTART_ETA_MIN   = 1e-6
N_INTERP_STEPS    = 11      # α ∈ {0.0, 0.1, ..., 1.0}
MAX_ATOMS         = 29      # QM9 max atoms per molecule

# Reference barrier from MolecularGNN BF16-256 (Phase 4 / paper 4.6.2)
MOLGNN_BF16_256_BARRIER_EV = 1.447
MOLGNN_BF16_256_PEAK_ALPHA = 0.3


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def gsutil_cp(local: Path, gcs: str):
    """Upload file to GCS (silent, non-fatal on error)."""
    result = subprocess.run(
        ["gsutil", "-q", "cp", str(local), gcs],
        capture_output=True
    )
    if result.returncode != 0:
        log(f"  WARNING: gsutil cp failed: {local.name} -> {gcs}")


def get_device():
    if XLA_AVAILABLE:
        return xm.xla_device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ════════════════════════════════════════════════════════════════════════════
#  MolecularTransformer — independent architecture (do NOT import from model.py)
# ════════════════════════════════════════════════════════════════════════════

class DistanceRBF(nn.Module):
    """Expand pairwise distances into Gaussian radial basis functions."""
    def __init__(self, n_gaussians=50, cutoff=5.0):
        super().__init__()
        centers = torch.linspace(0, cutoff, n_gaussians)
        self.register_buffer('centers', centers)
        self.width = (cutoff / n_gaussians) ** 2

    def forward(self, d):  # d: [...] -> [..., n_gaussians]
        return torch.exp(-((d.unsqueeze(-1) - self.centers) ** 2) / self.width)


class AttnBlock(nn.Module):
    """Multi-head attention over atoms with 3D distance bias."""
    def __init__(self, hidden_dim, n_heads=4, n_gaussians=50):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = hidden_dim // n_heads
        self.qkv       = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.dist_proj = nn.Linear(n_gaussians, n_heads, bias=False)
        self.out_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, h, dist_rbf, mask):
        # h: [B,N,D]   dist_rbf: [B,N,N,G]   mask: [B,N] bool (True = real atom)
        B, N, D = h.shape

        # Multi-head attention
        h_ln  = self.norm1(h)
        qkv   = self.qkv(h_ln).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv   = qkv.permute(2, 0, 3, 1, 4)   # [3, B, H, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B,H,N,N]
        dist_bias = self.dist_proj(dist_rbf).permute(0, 3, 1, 2)              # [B,H,N,N]
        attn = attn + dist_bias

        # Mask padding atoms (True = padding → fill with -inf)
        pad_mask = ~mask   # [B, N]
        attn = attn.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), -1e9)
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)                        # [B,H,N,head_dim]
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)
        h = h + self.out_proj(out)

        # Feed-forward
        h = h + self.ffn(self.norm2(h))
        return h


class MolecularTransformer(nn.Module):
    def __init__(self, num_atom_types=9, hidden_dim=256,
                 n_blocks=6, n_gaussians=50, cutoff=5.0):
        super().__init__()
        self._hidden_dim = hidden_dim
        self.embedding   = nn.Embedding(num_atom_types + 1, hidden_dim)
        self.rbf         = DistanceRBF(n_gaussians, cutoff)
        self.blocks      = nn.ModuleList([
            AttnBlock(hidden_dim, n_heads=4, n_gaussians=n_gaussians)
            for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def hidden_dim(self):
        return self._hidden_dim

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, z, pos, edge_src, edge_dst, assign_mat,
                num_graphs, edge_valid, atom_valid):
        # z: [B,N]  pos: [B,N,3]  atom_valid: [B,N] bool
        B, N = z.shape
        h = self.embedding(z)               # [B, N, D]

        # Pairwise distances — static shape [B, N, N]
        diff     = pos.unsqueeze(2) - pos.unsqueeze(1)   # [B,N,N,3]
        dist     = diff.norm(dim=-1)                      # [B,N,N]
        dist_rbf = self.rbf(dist)                         # [B,N,N,G]

        mask = atom_valid                   # [B, N] — True = real atom

        for block in self.blocks:
            h = block(h, dist_rbf, mask)

        # Masked mean-pool over real atoms
        h_masked = h * mask.unsqueeze(-1).float()
        pooled   = h_masked.sum(dim=1)                   # [B, D]
        count    = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        pooled   = pooled / count

        return self.head(pooled).squeeze(-1)              # [B]


# ── Evaluate (FP32 accumulation) ──────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, std):
    model.eval()
    mae_sum, n = 0.0, 0
    for batch in loader:
        z, pos, es, ed, am, ng, ev, av = batch_to_graph(batch, device)
        pred = model(z, pos, es, ed, am, ng, ev, av)
        mae_sum += (
            (pred.float() - batch['target'].to(device).float()).abs() * std
        ).sum().item()
        n += ng
        if XLA_AVAILABLE:
            xm.mark_step()
    return mae_sum / max(n, 1)


# ── BF16 training loop ────────────────────────────────────────────────────────

def train_epoch_bf16(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        z, pos, es, ed, am, ng, ev, av = batch_to_graph(batch, device)
        target = batch['target'].to(device)
        optimizer.zero_grad()
        if XLA_AVAILABLE:
            with torch.autocast('xla', dtype=torch.bfloat16, enabled=True):
                pred = model(z, pos, es, ed, am, ng, ev, av)
                loss = F.mse_loss(pred.float(), target.float())
        else:
            with torch.autocast('cpu', dtype=torch.bfloat16, enabled=True):
                pred = model(z, pos, es, ed, am, ng, ev, av)
                loss = F.mse_loss(pred.float(), target.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if XLA_AVAILABLE:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


# ── LMC helpers ───────────────────────────────────────────────────────────────

def interpolate_sds(sd0, sd1, alpha):
    return {
        k: (1 - alpha) * sd0[k].float() + alpha * sd1[k].float()
        for k in sd0
    }


def run_lmc(sd0, sd1, model, loader, device, std, label, n_steps=11):
    """
    Linearly interpolate between sd0 (alpha=0) and sd1 (alpha=1).
    Returns (records, barrier_ev, peak_alpha).
    """
    log(f"\nLMC: {label}  ({n_steps} interpolation points)")
    log(f"  {'α':>6}  {'MAE (eV)':>12}")
    log(f"  {'─'*6}  {'─'*12}")

    alphas  = [i / (n_steps - 1) for i in range(n_steps)]
    records = []
    model.eval()

    for alpha in alphas:
        model.load_state_dict(interpolate_sds(sd0, sd1, alpha))
        mae = evaluate(model, loader, device, std)
        records.append({"alpha": round(alpha, 2), "mae_ev": round(mae, 4)})
        log(f"  {alpha:>6.2f}  {mae:>12.4f}")

    maes      = [r["mae_ev"] for r in records]
    endpoints = (maes[0] + maes[-1]) / 2.0
    barrier   = max(maes) - endpoints
    peak_alpha = alphas[int(np.argmax(maes))]

    log(f"\n  Endpoint mean MAE : {endpoints:.4f} eV")
    log(f"  Peak MAE          : {max(maes):.4f} eV  at alpha={peak_alpha:.2f}")
    log(f"  Barrier height    : {barrier:.4f} eV")

    return records, round(barrier, 4), peak_alpha


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 68)
    log("  Phase 8 — MolecularTransformer (GAT) on QM9, BF16")
    log(f"  Architecture: hidden_dim={HIDDEN_DIM}, n_blocks={N_BLOCKS}, "
        f"n_gaussians={N_GAUSSIANS}")
    log(f"  Training    : {N_EPOCHS} epochs BF16, lr={LR_INIT}, "
        f"AdamW wd={WEIGHT_DECAY}")
    log(f"  Plateau     : patience={PLATEAU_PATIENCE} eval steps, "
        f"delta={PLATEAU_DELTA} eV")
    log(f"  Warm restart: lr={RESTART_LR}, T_max={RESTART_T_MAX}")
    log(f"  LMC ref     : MolecularGNN BF16-256 barrier = "
        f"{MOLGNN_BF16_256_BARRIER_EV:.3f} eV")
    log("=" * 68)

    device = get_device()
    log(f"Device: {device}" + (" (TPU)" if XLA_AVAILABLE else ""))

    # ── Data ──────────────────────────────────────────────────────────────────
    log("Loading QM9 data ...")
    train_loader, val_loader, _ = get_dataloaders(
        str(DATA_DIR), batch_size=BATCH_SIZE, num_workers=4
    )
    std = train_loader.dataset.std
    log(f"QM9 std: {std:.4f} eV")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MolecularTransformer(
        num_atom_types=9,
        hidden_dim=HIDDEN_DIM,
        n_blocks=N_BLOCKS,
        n_gaussians=N_GAUSSIANS,
        cutoff=CUTOFF,
    ).to(device)
    log(f"MolecularTransformer: hidden_dim={model.hidden_dim}  "
        f"params={model.parameter_count():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=ETA_MIN_MAIN
    )

    notify(
        "PHASE_START",
        "[Phase8] MolecularTransformer BF16 training begun",
        data={
            "hidden_dim": HIDDEN_DIM,
            "n_blocks": N_BLOCKS,
            "n_epochs": N_EPOCHS,
            "lr": LR_INIT,
        }
    )

    # ─────────────────────────────────────────────────────────────────────────
    #  SECTION A: Training with plateau detection
    # ─────────────────────────────────────────────────────────────────────────
    log(f"\n{'─'*68}")
    log("  SECTION A: BF16 Training (100 epochs)")
    log(f"  {'ep':>5}  {'lr':>10}  {'train_loss':>12}  {'val_mae (eV)':>14}  "
        f"{'pat':>5}")
    log(f"  {'─'*5}  {'─'*10}  {'─'*12}  {'─'*14}  {'─'*5}")

    eval_history   = []    # list of (epoch, val_mae)
    saved_sds      = {}    # {epoch: state_dict (cpu clone)}
    plateau_epoch  = None  # set when plateau is triggered
    patience_count = 0
    best_eval_mae  = float("inf")

    # Epoch 0 eval for reference
    mae_ep0 = evaluate(model, val_loader, device, std)
    log(f"  ep  0  (initial val MAE): {mae_ep0:.4f} eV")
    eval_history.append((0, round(mae_ep0, 4)))
    best_eval_mae = mae_ep0

    for ep in range(1, N_EPOCHS + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_epoch_bf16(model, train_loader, optimizer, device)
        scheduler.step()

        if ep % EVAL_EVERY == 0 or ep == N_EPOCHS:
            val_mae = evaluate(model, val_loader, device, std)
            eval_history.append((ep, round(val_mae, 4)))

            # Plateau detection (only before plateau has fired)
            if plateau_epoch is None:
                improvement = best_eval_mae - val_mae
                if improvement >= PLATEAU_DELTA:
                    best_eval_mae  = val_mae
                    patience_count = 0
                else:
                    patience_count += 1

                if patience_count >= PLATEAU_PATIENCE:
                    plateau_epoch = ep
                    log(f"  *** PLATEAU DETECTED at ep{ep} "
                        f"(patience={patience_count}) ***")
                    # Save plateau checkpoint
                    sd_plateau = {k: v.clone().cpu()
                                  for k, v in model.state_dict().items()}
                    saved_sds[ep] = sd_plateau
                    ckpt_path = OUT_DIR / f"transformer_plateau_ep{ep}.pt"
                    torch.save(
                        {"epoch": ep, "model": model.state_dict(),
                         "val_mae_ev": val_mae},
                        ckpt_path
                    )
                    gsutil_cp(
                        ckpt_path,
                        f"{GCS_BASE}/phase8_gat_qm9/transformer_plateau_ep{ep}.pt"
                    )

            log(f"  ep{ep:>3d}  {lr:>10.2e}  {train_loss:>12.6f}  "
                f"{val_mae:>14.4f}  {patience_count:>5d}"
                + ("  [PLATEAU]" if ep == plateau_epoch else ""))

            heartbeat("Phase8_train", ep,
                      {"ep": ep, "val_mae": round(val_mae, 4),
                       "patience": patience_count})

            # Once plateau is detected, stop main training loop
            if plateau_epoch is not None:
                # Save state before breaking if not already saved
                if ep not in saved_sds:
                    saved_sds[ep] = {k: v.clone().cpu()
                                     for k, v in model.state_dict().items()}
                break
        else:
            log(f"  ep{ep:>3d}  {lr:>10.2e}  {train_loss:>12.6f}")

    # If plateau never triggered, set it at the last epoch
    if plateau_epoch is None:
        plateau_epoch = N_EPOCHS
        log(f"  NOTE: No plateau detected in {N_EPOCHS} epochs. "
            f"Using ep{plateau_epoch} as plateau.")
        if plateau_epoch not in saved_sds:
            saved_sds[plateau_epoch] = {k: v.clone().cpu()
                                        for k, v in model.state_dict().items()}

    mae_at_plateau = dict(eval_history).get(plateau_epoch, float("nan"))
    log(f"\n  Plateau ep: {plateau_epoch}   val MAE: {mae_at_plateau:.4f} eV")

    # ─────────────────────────────────────────────────────────────────────────
    #  SECTION B: Warm Restart from plateau
    # ─────────────────────────────────────────────────────────────────────────
    log(f"\n{'─'*68}")
    log(f"  SECTION B: Warm Restart from ep{plateau_epoch}")
    log(f"  lr={RESTART_LR}, CosineAnnealingLR T_max={RESTART_T_MAX}")
    log(f"  {'ep':>5}  {'lr':>10}  {'train_loss':>12}  {'val_mae (eV)':>14}")
    log(f"  {'─'*5}  {'─'*10}  {'─'*12}  {'─'*14}")

    # Fresh optimizer from plateau weights (already loaded in model)
    model.load_state_dict(saved_sds[plateau_epoch])
    restart_opt = torch.optim.AdamW(
        model.parameters(), lr=RESTART_LR, weight_decay=WEIGHT_DECAY
    )
    restart_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        restart_opt, T_max=RESTART_T_MAX, eta_min=RESTART_ETA_MIN
    )

    restart_eval_history = []
    # baseline eval at plateau (re-confirm)
    mae_restart_base = evaluate(model, val_loader, device, std)
    restart_eval_history.append((plateau_epoch, round(mae_restart_base, 4)))

    for step in range(1, RESTART_T_MAX + 1):
        ep  = plateau_epoch + step
        lr  = restart_opt.param_groups[0]["lr"]
        train_loss = train_epoch_bf16(model, train_loader, restart_opt, device)
        restart_sched.step()

        # Eval at every step post-plateau (to allow LMC on ep+3 window)
        val_mae = evaluate(model, val_loader, device, std)
        restart_eval_history.append((ep, round(val_mae, 4)))

        # Save state dict for every step (needed for LMC)
        saved_sds[ep] = {k: v.clone().cpu()
                         for k, v in model.state_dict().items()}

        # Persist checkpoint
        ckpt_path = OUT_DIR / f"transformer_restart_ep{ep}.pt"
        torch.save(
            {"epoch": ep, "model": model.state_dict(),
             "val_mae_ev": val_mae},
            ckpt_path
        )
        gsutil_cp(
            ckpt_path,
            f"{GCS_BASE}/phase8_gat_qm9/transformer_restart_ep{ep}.pt"
        )

        log(f"  ep{ep:>3d}  {lr:>10.2e}  {train_loss:>12.6f}  {val_mae:>14.4f}")

        heartbeat("Phase8_restart", step,
                  {"ep": ep, "val_mae": round(val_mae, 4)})

    log(f"\n  Warm restart complete. ({RESTART_T_MAX} steps)")

    # ─────────────────────────────────────────────────────────────────────────
    #  SECTION C: LMC measurement
    # ─────────────────────────────────────────────────────────────────────────
    log(f"\n{'─'*68}")
    log("  SECTION C: Linear Mode Connectivity")

    ep_post = plateau_epoch + 3
    lmc_label = (
        f"MolecularTransformer BF16-256  "
        f"ep{plateau_epoch} <-> ep{ep_post}  "
        f"(3-epoch window — matches MolecularGNN comparison)"
    )

    # Build a fresh model instance for LMC interpolation
    model_lmc = MolecularTransformer(
        num_atom_types=9,
        hidden_dim=HIDDEN_DIM,
        n_blocks=N_BLOCKS,
        n_gaussians=N_GAUSSIANS,
        cutoff=CUTOFF,
    ).to(device)

    if ep_post not in saved_sds:
        log(f"  WARNING: ep{ep_post} not in saved checkpoints — "
            f"LMC will use last available restart epoch")
        ep_post = max(k for k in saved_sds if k > plateau_epoch)

    records, barrier_ev, peak_alpha = run_lmc(
        saved_sds[plateau_epoch],
        saved_sds[ep_post],
        model_lmc,
        val_loader,
        device,
        std,
        label=lmc_label,
        n_steps=N_INTERP_STEPS,
    )

    # ─────────────────────────────────────────────────────────────────────────
    #  SECTION D: Summary & comparison
    # ─────────────────────────────────────────────────────────────────════════
    log(f"\n{'='*68}")
    log("  PHASE 8 SUMMARY: MolecularTransformer vs MolecularGNN LMC Barriers")
    log(f"  {'Architecture':<36}  {'Barrier (eV)':>14}  {'Peak alpha':>12}")
    log(f"  {'─'*36}  {'─'*14}  {'─'*12}")
    log(f"  {'MolecularTransformer BF16-256 (Phase 8)':<36}  "
        f"{barrier_ev:>14.4f}  {peak_alpha:>12.2f}")
    log(f"  {'MolecularGNN BF16-256 (Phase 4 ref)':<36}  "
        f"{MOLGNN_BF16_256_BARRIER_EV:>14.4f}  "
        f"{MOLGNN_BF16_256_PEAK_ALPHA:>12.2f}")
    log(f"{'='*68}")

    generalises = barrier_ev > 0.05   # non-trivial barrier in second architecture
    delta = barrier_ev - MOLGNN_BF16_256_BARRIER_EV
    if generalises:
        log(f"  -> BF16 sharpening barrier CONFIRMED in MolecularTransformer "
            f"({barrier_ev:.4f} eV)")
    else:
        log(f"  -> Barrier near-zero ({barrier_ev:.4f} eV) — "
            f"transformer may be loss-surface flat in BF16")

    if abs(delta) < 0.2:
        log(f"  -> Barriers agree within 0.2 eV (delta={delta:+.4f} eV) — "
            f"architecture-invariant sharpening")
    else:
        log(f"  -> Barriers differ by {delta:+.4f} eV — "
            f"architecture-specific landscape geometry")
    log(f"{'='*68}")

    # ─────────────────────────────────────────────────────────────────────────
    #  SECTION E: Build and upload results.json
    # ─────────────────────────────────────────────────────────────────────────
    training_trajectory = [
        {"epoch": ep, "val_mae_ev": mae} for ep, mae in eval_history
    ]
    restart_trajectory = [
        {"epoch": ep, "val_mae_ev": mae} for ep, mae in restart_eval_history
    ]

    summary = {
        "experiment": "phase8_gat_qm9",
        "architecture": "MolecularTransformer",
        "precision": "bf16",
        "hidden_dim": HIDDEN_DIM,
        "n_blocks": N_BLOCKS,
        "n_gaussians": N_GAUSSIANS,
        "cutoff": CUTOFF,
        "n_training_epochs": N_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr_init": LR_INIT,
        "weight_decay": WEIGHT_DECAY,
        "plateau_patience_eval_steps": PLATEAU_PATIENCE,
        "plateau_delta_ev": PLATEAU_DELTA,
        "plateau_epoch": plateau_epoch,
        "mae_at_plateau_ev": round(mae_at_plateau, 4),
        "restart_lr": RESTART_LR,
        "restart_t_max": RESTART_T_MAX,
        "training_eval_trajectory": training_trajectory,
        "restart_eval_trajectory": restart_trajectory,
        "lmc": {
            "ep_pre": plateau_epoch,
            "ep_post": ep_post,
            "n_interp_steps": N_INTERP_STEPS,
            "interpolation": records,
            "barrier_ev": barrier_ev,
            "peak_alpha": peak_alpha,
        },
        "comparison": {
            "molgnn_bf16_256_barrier_ev": MOLGNN_BF16_256_BARRIER_EV,
            "molgnn_bf16_256_peak_alpha": MOLGNN_BF16_256_PEAK_ALPHA,
            "transformer_bf16_256_barrier_ev": barrier_ev,
            "transformer_bf16_256_peak_alpha": peak_alpha,
            "delta_ev": round(delta, 4),
            "barrier_generalises_to_transformer": generalises,
            "architectures_agree_within_0p2ev": abs(delta) < 0.2,
        },
        "gcs_output": f"{GCS_BASE}/phase8_gat_qm9/",
    }

    out_json = OUT_DIR / "results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    log(f"\nResults saved -> {out_json}")
    gsutil_cp(out_json, f"{GCS_BASE}/phase8_gat_qm9/results.json")
    log(f"GCS: {GCS_BASE}/phase8_gat_qm9/")

    notify(
        "PHASE_COMPLETE",
        "[Phase8] MolecularTransformer BF16 LMC complete",
        data={
            "plateau_epoch": plateau_epoch,
            "mae_at_plateau_ev": round(mae_at_plateau, 4),
            "lmc_barrier_ev": barrier_ev,
            "molgnn_ref_barrier_ev": MOLGNN_BF16_256_BARRIER_EV,
            "delta_ev": round(delta, 4),
            "generalises": generalises,
        }
    )


if __name__ == "__main__":
    main()
