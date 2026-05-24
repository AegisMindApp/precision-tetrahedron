#!/usr/bin/env python3
"""
phase14_foundation_pretrain.py
-------------------------------
Pre-train a large MolecularGNN (hidden_dim=512, num_blocks=8, ~50M params) on
ZINC-250K in a multi-task supervised setting: predict logP, QED, normalised MW,
and normalised TPSA simultaneously (4 targets).

This gives a powerful molecular representation as a foundation for Phase 15
AMR fine-tuning, far better than starting from scratch.

Pipeline
--------
1.  Load ZINC-250K CSV from /tmp/zinc250k/ (phase12 already fetched it)
    Fall back to QM9 if ZINC not present
2.  Compute 4 molecular descriptors per compound (logP, QED, norm-MW, norm-TPSA)
3.  Train FoundationGNN(hidden_dim=512, num_blocks=8, num_targets=4) in BF16
    AdamW lr=1e-4 wd=1e-5, CosineAnnealingLR T_max=200, 200 epochs, batch=32
4.  Save checkpoint every 10 epochs to GCS

GCS output: gs://aegismind-tpu-results/aegis_flashoptim/phase14_foundation_pretrain/
Approx. runtime: ~7 days on TPU v6e-8 (intentional — large compute spend)
"""

import os
import sys
import csv
import json
import time
import math
import subprocess
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── XLA ───────────────────────────────────────────────────────────────────────
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
    print("torch_xla not found — running on CPU/GPU", flush=True)

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.expanduser("~/flashoptim"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MolecularGNN
from chembl_data import smiles_to_graph
from notify import notify, heartbeat

# ── Boilerplate ───────────────────────────────────────────────────────────────
GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID     = os.environ.get("RUN_ID",     "aegis_flashoptim")
GCS_BASE   = f"{GCS_BUCKET}/{RUN_ID}"

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] {msg}", flush=True)

def gsutil_cp(local, gcs):
    subprocess.run(["gsutil", "-q", "cp", str(local), gcs], check=False)

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR  = Path("/tmp/phase14_foundation_pretrain")
CKPT_DIR = OUT_DIR / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

ZINC_DIR  = Path("/tmp/zinc250k")
ZINC_CSV  = ZINC_DIR / "zinc250k.csv"
ZINC_URL  = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
    "master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
)
QM9_DIR   = Path("/tmp/qm9")

HIDDEN_DIM  = 512
NUM_BLOCKS  = 8
NUM_TARGETS = 4     # logP, QED, norm-MW, norm-TPSA
NUM_GAUSSIANS = 50
CUTOFF      = 5.0

PA          = 80    # padded atoms
BATCH_SIZE  = 32
N_EPOCHS    = 200
LR          = 1e-4
WEIGHT_DECAY = 1e-5
CKPT_EVERY  = 10
TRAIN_FRAC  = 0.90
RANDOM_SEED = 42

GCS_PHASE14 = f"{GCS_BASE}/phase14_foundation_pretrain"


# ── Foundation model ──────────────────────────────────────────────────────────

class FoundationGNN(nn.Module):
    """
    MolecularGNN backbone (hidden_dim=512, num_blocks=8) with a 4-way descriptor
    prediction head.  The backbone is initialised from scratch; no phase6 weights.

    num_targets=4 outputs: [logP, QED, norm_MW, norm_TPSA]
    """

    def __init__(self, hidden_dim=HIDDEN_DIM, num_blocks=NUM_BLOCKS,
                 num_gaussians=NUM_GAUSSIANS, cutoff=CUTOFF,
                 num_targets=NUM_TARGETS):
        super().__init__()
        self.gnn = MolecularGNN(
            num_atom_types=119,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            num_targets=num_targets,
        )

    @property
    def hidden_dim(self):
        return self.gnn.hidden_dim

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, z_pad, pos_pad, atom_valid):
        """Returns [B, num_targets] descriptor predictions."""
        return self.gnn(z_pad, pos_pad, atom_valid)


# ── Descriptor computation ────────────────────────────────────────────────────

def compute_descriptors(smiles: str) -> Optional[List[float]]:
    """
    Compute 4 molecular descriptors via RDKit.
    Returns [logP, QED, norm_MW, norm_TPSA] or None on failure.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, QED as rdkQED
        from rdkit.Chem.rdMolDescriptors import CalcTPSA
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        logp    = Descriptors.MolLogP(mol)
        qed     = rdkQED.qed(mol)
        norm_mw  = Descriptors.MolWt(mol) / 500.0
        norm_tpsa = CalcTPSA(mol) / 200.0
        return [float(logp), float(qed), float(norm_mw), float(norm_tpsa)]
    except Exception:
        return None


# ── Dataset ───────────────────────────────────────────────────────────────────

class DescriptorDataset(Dataset):
    """
    Each item: (z_pad [PA], pos_pad [PA,3], valid [PA], descriptors [4])
    """

    def __init__(self, records):
        # records: list of (z_pad, pos_pad, valid, descriptors_tensor)
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def load_smiles_zinc() -> List[str]:
    """Load SMILES from ZINC-250K CSV; download if not present."""
    ZINC_DIR.mkdir(parents=True, exist_ok=True)

    if not ZINC_CSV.exists():
        log(f"ZINC CSV not found at {ZINC_CSV} — downloading ...")
        try:
            urllib.request.urlretrieve(ZINC_URL, str(ZINC_CSV))
            log(f"Downloaded ZINC CSV ({ZINC_CSV.stat().st_size // 1024} KB)")
        except Exception as e:
            log(f"ZINC download failed: {e}")
            return []

    smiles_col = None
    rows = []
    try:
        with open(ZINC_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            # Column names vary between releases
            for cand in ("smiles", "SMILES", "structure"):
                if cand in headers:
                    smiles_col = cand
                    break
            if smiles_col is None and headers:
                smiles_col = headers[0]  # first column as fallback
            for row in reader:
                smi = (row.get(smiles_col) or "").strip()
                if smi:
                    rows.append(smi)
    except Exception as e:
        log(f"ZINC CSV parse error: {e}")

    log(f"Loaded {len(rows)} SMILES from ZINC-250K")
    return rows


def load_smiles_qm9() -> List[str]:
    """Load SMILES from QM9 gdb9.sdf if available; return empty list otherwise."""
    qm9_sdf = QM9_DIR / "gdb9.sdf"
    if not qm9_sdf.exists():
        log(f"QM9 SDF not found at {qm9_sdf}")
        return []
    try:
        from rdkit import Chem
        suppl = Chem.SDMolSupplier(str(qm9_sdf), removeHs=False)
        smiles_list = []
        for mol in suppl:
            if mol is not None:
                smi = Chem.MolToSmiles(mol)
                if smi:
                    smiles_list.append(smi)
        log(f"Loaded {len(smiles_list)} SMILES from QM9")
        return smiles_list
    except Exception as e:
        log(f"QM9 load error: {e}")
        return []


def build_dataset(smiles_list: List[str]) -> List:
    """Convert SMILES to (z_pad, pos_pad, valid, descriptors) records."""
    records = []
    failed  = 0
    n       = len(smiles_list)
    for i, smi in enumerate(smiles_list):
        if i % 1000 == 0:
            sys.stdout.write(f"\r  Building graphs: {i}/{n}  ok={len(records)}")
            sys.stdout.flush()
        descs = compute_descriptors(smi)
        if descs is None:
            failed += 1
            continue
        g = smiles_to_graph(smi)
        if g is None:
            failed += 1
            continue
        z, pos = g
        na = min(len(z), PA)
        z_pad   = torch.zeros(PA, dtype=torch.long)
        pos_pad = torch.zeros(PA, 3, dtype=torch.float32)
        valid   = torch.zeros(PA, dtype=torch.bool)
        z_pad[:na]   = torch.as_tensor(z[:na],   dtype=torch.long)
        pos_pad[:na] = torch.as_tensor(pos[:na], dtype=torch.float32)
        valid[:na]   = True
        desc_t = torch.tensor(descs, dtype=torch.float32)
        records.append((z_pad, pos_pad, valid, desc_t))
    print()
    log(f"Dataset built: {len(records)} ok, {failed} failed from {n} SMILES")
    return records


def collate_fn(batch):
    z_b     = torch.stack([b[0] for b in batch])
    pos_b   = torch.stack([b[1] for b in batch])
    valid_b = torch.stack([b[2] for b in batch])
    desc_b  = torch.stack([b[3] for b in batch])
    return z_b, pos_b, valid_b, desc_b


# ── BF16 autocast helper ──────────────────────────────────────────────────────

def bf16_autocast(device):
    """Return a context manager for BF16 AMP where supported."""
    if XLA_AVAILABLE:
        import contextlib
        return contextlib.nullcontext()   # XLA handles BF16 via model dtype
    if str(device).startswith("cuda"):
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.no_grad.__class__()      # no-op fallback (never reached in fwd)


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, use_bf16=True):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for z_b, pos_b, valid_b, desc_b in loader:
        z_b     = z_b.to(device)
        pos_b   = pos_b.to(device)
        valid_b = valid_b.to(device)
        desc_b  = desc_b.to(device)

        optimizer.zero_grad()
        if use_bf16 and not XLA_AVAILABLE and str(device).startswith("cuda"):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(z_b, pos_b, valid_b)  # [B, 4]
                loss = F.mse_loss(pred.float(), desc_b)
        else:
            pred = model(z_b, pos_b, valid_b)
            loss = F.mse_loss(pred, desc_b)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if XLA_AVAILABLE:
            xm.mark_step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    with torch.no_grad():
        for z_b, pos_b, valid_b, desc_b in loader:
            z_b     = z_b.to(device)
            pos_b   = pos_b.to(device)
            valid_b = valid_b.to(device)
            desc_b  = desc_b.to(device)
            pred    = model(z_b, pos_b, valid_b)
            loss    = F.mse_loss(pred.float() if not pred.is_floating_point()
                                 else pred, desc_b)
            if XLA_AVAILABLE:
                xm.mark_step()
            total_loss += loss.item()
            n_batches  += 1
    return total_loss / max(n_batches, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    notify("PHASE_START", "Phase 14 — Foundation GNN pre-training (200 epochs, ~7 days)")
    log("=" * 60)
    log("Phase 14 — Foundation GNN pre-training on molecular descriptors")
    log(f"  hidden_dim={HIDDEN_DIM}, num_blocks={NUM_BLOCKS}, num_targets={NUM_TARGETS}")
    log(f"  epochs={N_EPOCHS}, batch={BATCH_SIZE}, lr={LR}")
    log("=" * 60)

    # ── Device ────────────────────────────────────────────────────────────────
    if XLA_AVAILABLE:
        device = xm.xla_device()
        log(f"Device: XLA ({device})")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        log(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        log("Device: CPU")

    # ── Load SMILES ───────────────────────────────────────────────────────────
    smiles_list = load_smiles_zinc()
    if len(smiles_list) < 100:
        log("ZINC-250K unavailable — falling back to QM9")
        smiles_list = load_smiles_qm9()
    if len(smiles_list) < 100:
        notify("ABORT", "Phase 14: no usable SMILES dataset found")
        raise RuntimeError("No SMILES data available (tried ZINC-250K and QM9)")

    # ── Build dataset ─────────────────────────────────────────────────────────
    log("Building descriptor dataset ...")
    all_records = build_dataset(smiles_list)
    if len(all_records) < 100:
        notify("ABORT", "Phase 14: too few valid records after graph build")
        raise RuntimeError(f"Only {len(all_records)} valid records built")

    rng = np.random.RandomState(RANDOM_SEED)
    idx = np.arange(len(all_records))
    rng.shuffle(idx)
    split = int(len(idx) * TRAIN_FRAC)
    train_idx, val_idx = idx[:split].tolist(), idx[split:].tolist()

    train_ds = DescriptorDataset([all_records[i] for i in train_idx])
    val_ds   = DescriptorDataset([all_records[i] for i in val_idx])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               collate_fn=collate_fn, num_workers=2, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                               collate_fn=collate_fn, num_workers=2, pin_memory=False)

    log(f"Train={len(train_ds)}, Val={len(val_ds)}")

    # ── Build model ───────────────────────────────────────────────────────────
    torch.manual_seed(RANDOM_SEED)
    model = FoundationGNN(
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        num_gaussians=NUM_GAUSSIANS,
        cutoff=CUTOFF,
        num_targets=NUM_TARGETS,
    )

    # Cast to BF16 on XLA
    if XLA_AVAILABLE:
        model = model.to(torch.bfloat16)
    model = model.to(device)

    n_params = model.parameter_count()
    log(f"FoundationGNN: {n_params:,} parameters ({n_params/1e6:.1f}M)")

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=LR * 0.01
    )

    # ── Resume from checkpoint if available ───────────────────────────────────
    start_epoch = 0
    best_val    = float("inf")
    metrics_log = []

    # Try to resume latest local checkpoint
    existing_ckpts = sorted(CKPT_DIR.glob("foundation_epoch*.pt"))
    if existing_ckpts:
        latest = existing_ckpts[-1]
        log(f"Resuming from {latest} ...")
        ckpt = torch.load(str(latest), map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("best_val", float("inf"))
        metrics_log = ckpt.get("metrics_log", [])
        # Fast-forward scheduler
        for _ in range(start_epoch):
            scheduler.step()
        log(f"Resumed at epoch {start_epoch}, best_val={best_val:.6f}")

    # ── Training loop ─────────────────────────────────────────────────────────
    log(f"Starting training: epochs {start_epoch}–{N_EPOCHS-1}")
    t0_total = time.time()

    for epoch in range(start_epoch, N_EPOCHS):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, device,
                                  use_bf16=True)
        val_loss   = eval_epoch(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        total_h  = (time.time() - t0_total) / 3600.0
        lr_now   = optimizer.param_groups[0]["lr"]

        log(f"Epoch {epoch:3d}/{N_EPOCHS-1}  "
            f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
            f"lr={lr_now:.2e}  t={elapsed:.1f}s  total={total_h:.2f}h")

        metrics_log.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss,   6),
            "lr":         round(lr_now,     8),
        })

        # Heartbeat every 10 epochs
        if epoch % 10 == 0:
            heartbeat("phase14", epoch, {
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "lr":         lr_now,
                "total_hours": round(total_h, 2),
            })

        # Checkpoint every CKPT_EVERY epochs
        if (epoch + 1) % CKPT_EVERY == 0 or epoch == N_EPOCHS - 1:
            ckpt_path = CKPT_DIR / f"foundation_epoch{epoch:03d}.pt"
            torch.save({
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val":       best_val,
                "val_loss":       val_loss,
                "metrics_log":    metrics_log,
                "config": {
                    "hidden_dim":    HIDDEN_DIM,
                    "num_blocks":    NUM_BLOCKS,
                    "num_targets":   NUM_TARGETS,
                    "num_gaussians": NUM_GAUSSIANS,
                    "cutoff":        CUTOFF,
                },
            }, str(ckpt_path))

            gcs_path = f"{GCS_PHASE14}/foundation_epoch{epoch:03d}.pt"
            gsutil_cp(ckpt_path, gcs_path)
            log(f"  Checkpoint saved → {gcs_path}")

            # Track best
            if val_loss < best_val:
                best_val = val_loss
                best_path = CKPT_DIR / "foundation_best.pt"
                torch.save({
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "val_loss":    val_loss,
                    "config": {
                        "hidden_dim":    HIDDEN_DIM,
                        "num_blocks":    NUM_BLOCKS,
                        "num_targets":   NUM_TARGETS,
                    },
                }, str(best_path))
                gsutil_cp(best_path, f"{GCS_PHASE14}/foundation_best.pt")
                log(f"  New best val_loss={val_loss:.6f} → foundation_best.pt")

    # ── Save training metrics ─────────────────────────────────────────────────
    metrics_path = OUT_DIR / "training_metrics.json"
    metrics_path.write_text(json.dumps({
        "config": {
            "hidden_dim":    HIDDEN_DIM,
            "num_blocks":    NUM_BLOCKS,
            "num_targets":   NUM_TARGETS,
            "n_epochs":      N_EPOCHS,
            "batch_size":    BATCH_SIZE,
            "lr":            LR,
            "weight_decay":  WEIGHT_DECAY,
            "n_params":      n_params,
        },
        "n_train":    len(train_ds),
        "n_val":      len(val_ds),
        "best_val":   best_val,
        "metrics":    metrics_log,
    }, indent=2))
    gsutil_cp(metrics_path, f"{GCS_PHASE14}/training_metrics.json")

    total_h = (time.time() - t0_total) / 3600.0
    summary = {
        "best_val_loss": round(best_val, 6),
        "n_params":      n_params,
        "total_hours":   round(total_h, 2),
        "n_train":       len(train_ds),
        "n_val":         len(val_ds),
    }
    log("Phase 14 complete.")
    for k, v in summary.items():
        log(f"  {k}: {v}")

    notify("PHASE_COMPLETE",
           f"Phase 14 foundation pre-training done — best_val={best_val:.5f} "
           f"in {total_h:.1f}h",
           data=summary)
    log("PHASE_COMPLETE")


if __name__ == "__main__":
    main()
