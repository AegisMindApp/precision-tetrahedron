#!/usr/bin/env python3
"""
phase12_diffusion_zinc.py
-------------------------
E(3)-equivariant diffusion model (EDM-style) trained on ZINC-250K in BF16
on a TPU v6e.  Generates novel drug-like molecules conditioned on molecular
properties (logP, qed, SAS) as a proxy for pKd-conditioned generation.

Based on:  Hoogeboom et al. 2022 — "Equivariant Diffusion for Molecule
           Generation in 3D" (EDM).

Pipeline
--------
1.  Download ZINC-250K CSV  (aspuru-guzik-group/chemical_vae on GitHub)
2.  Build padded (z, pos, valid_mask, cond) dataset via smiles_to_graph
3.  Train EDMDenoiser (EGNN, hidden_dim=128, 6 layers) for 100 epochs BF16
4.  Generate 1000 sample molecules via DDIM-style reverse diffusion
5.  Compute validity / uniqueness / novelty metrics
6.  Upload checkpoints, results, samples to GCS

GCS output: gs://aegismind-tpu-results/aegis_flashoptim/phase12_diffusion_zinc/
"""

import os
import sys
import json
import math
import time
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

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.expanduser("~/flashoptim"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chembl_data import smiles_to_graph
from notify import notify, heartbeat

# ── Global config ──────────────────────────────────────────────────────────────
GCS_BUCKET  = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID      = os.environ.get("RUN_ID",     "aegis_flashoptim")
GCS_BASE    = f"{GCS_BUCKET}/{RUN_ID}/phase12_diffusion_zinc"

OUT_DIR     = Path("/tmp/phase12_diffusion_zinc")
DATA_DIR    = Path("/tmp/zinc250k")
CKPT_DIR    = OUT_DIR / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

ZINC_URL = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
    "master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
)
ZINC_CSV = DATA_DIR / "zinc250k.csv"

# Atom type mapping: atomic number → index 1-9 (0 = pad)
ATOM_MAP    = {1: 1, 6: 2, 7: 3, 8: 4, 9: 5, 15: 6, 16: 7, 17: 8, 35: 9, 53: 9}
MAX_ATOMS   = 80
N_ATOM_TYPES = 10   # indices 0-9; 0 = pad token
COND_DIM    = 3     # (logP, qed, SAS)
T_DIFF      = 500   # diffusion timesteps

HIDDEN_DIM  = 128
N_LAYERS    = 6
BATCH_SIZE  = 32
N_EPOCHS    = 100
LR          = 2e-4
WEIGHT_DECAY = 1e-5
COORD_LOSS_W = 1.0
ATOM_LOSS_W  = 0.1
EVAL_EVERY   = 5
N_SAMPLES    = 1000
N_DDIM_STEPS = 50


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion schedule
# ─────────────────────────────────────────────────────────────────────────────

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule (Nichol & Dhariwal 2021)."""
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T + s) / (1 + s)) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clamp(betas, 1e-4, 0.9999)


class DiffusionSchedule:
    """Pre-computed cosine diffusion schedule tensors."""

    def __init__(self, T: int = T_DIFF):
        self.T = T
        betas = cosine_beta_schedule(T)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod).sqrt()

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self

    def q_sample_coords(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: corrupt coordinates at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        x_t = sqrt_alpha * x0 + sqrt_one_minus * noise
        return x_t, noise

    def q_sample_atoms(
        self,
        z0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Discrete diffusion for atom types.
        With probability (1 - alphas_cumprod[t]) replace each atom type
        with a random type drawn from [1, N_ATOM_TYPES-1].
        """
        B, N = z0.shape
        noise_prob = (1.0 - self.alphas_cumprod[t]).view(-1, 1)  # [B,1]
        # Random atom types in [1, N_ATOM_TYPES-1] (never 0=pad in noise)
        rand_types = torch.randint(1, N_ATOM_TYPES, (B, N), device=z0.device)
        corrupt_mask = torch.bernoulli(noise_prob.expand(B, N).clamp(0, 1)).bool()
        z_t = torch.where(corrupt_mask, rand_types, z0)
        return z_t


# ─────────────────────────────────────────────────────────────────────────────
# EGNN architecture
# ─────────────────────────────────────────────────────────────────────────────

class EGNNLayer(nn.Module):
    """
    E(3)-equivariant graph neural network layer (Satorras et al. 2021).
    Operates on fully-connected graphs (all pairs) with a validity mask.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # edge MLP: [h_i || h_j || d²] → message
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # node MLP: [h_i || agg_msg] → new h_i
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # coordinate update scalar (equivariant because weighted by diff)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,        # [B, N, D]
        x: torch.Tensor,        # [B, N, 3]
        valid_mask: torch.Tensor,  # [B, N] bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = h.shape

        # Pairwise displacement and squared distance
        diff = x.unsqueeze(2) - x.unsqueeze(1)          # [B, N, N, 3]
        dist_sq = (diff ** 2).sum(-1, keepdim=True)      # [B, N, N, 1]

        # Edge features (no per-edge distance stored; use squared dist directly)
        h_i = h.unsqueeze(2).expand(B, N, N, D)
        h_j = h.unsqueeze(1).expand(B, N, N, D)
        edge_in = torch.cat([h_i, h_j, dist_sq], dim=-1)  # [B, N, N, 2D+1]
        m_ij = self.edge_mlp(edge_in)                      # [B, N, N, D]

        # Pair validity mask: both atoms must be real; no self-loops
        mask_2d = (valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)).float()  # [B,N,N]
        eye = torch.eye(N, device=x.device, dtype=x.dtype).unsqueeze(0)
        mask_2d = mask_2d * (1.0 - eye)                   # [B, N, N]

        # Equivariant coordinate update: shift x toward/away from neighbors
        coord_w = self.coord_mlp(m_ij).squeeze(-1)        # [B, N, N]
        coord_w = coord_w * mask_2d
        # [B, N, 3]: weighted sum of displacements, normalised by neighbor count
        n_neighbors = mask_2d.sum(dim=2, keepdim=True).clamp(min=1.0)  # [B, N, 1]
        x_update = (diff * coord_w.unsqueeze(-1)).sum(dim=2) / n_neighbors
        x = x + x_update

        # Invariant node update
        agg = (m_ij * mask_2d.unsqueeze(-1)).sum(dim=2)   # [B, N, D]
        h_new = self.node_mlp(torch.cat([h, agg], dim=-1))
        h = self.norm(h + h_new)

        return h, x


class EDMDenoiser(nn.Module):
    """
    EGNN-based denoiser for equivariant diffusion.

    Input : (z_t [B,N], x_t [B,N,3], t [B], valid [B,N], cond [B,3])
    Output: (atom_logits [B,N,K], coord_noise [B,N,3])
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        n_layers: int = N_LAYERS,
        n_atom_types: int = N_ATOM_TYPES,
        cond_dim: int = COND_DIM,
        T: int = T_DIFF,
    ):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._T = T

        self.atom_embed = nn.Embedding(n_atom_types, hidden_dim, padding_idx=0)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_embed = nn.Linear(cond_dim, hidden_dim)
        self.layers = nn.ModuleList([EGNNLayer(hidden_dim) for _ in range(n_layers)])
        self.atom_head = nn.Linear(hidden_dim, n_atom_types)
        self.coord_head = nn.Linear(hidden_dim, 3)

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        z_t: torch.Tensor,       # [B, N] noisy atom types (long)
        x_t: torch.Tensor,       # [B, N, 3] noisy coords (float)
        t: torch.Tensor,          # [B] timestep indices (long)
        valid_mask: torch.Tensor, # [B, N] bool
        cond: Optional[torch.Tensor] = None,  # [B, COND_DIM]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N = z_t.shape

        h = self.atom_embed(z_t)                                          # [B, N, D]
        t_emb = self.time_embed(t.float().unsqueeze(1) / self._T)         # [B, D]
        h = h + t_emb.unsqueeze(1)                                        # broadcast over N

        if cond is not None:
            h = h + self.cond_embed(cond).unsqueeze(1)

        # Zero out padded atom positions so they don't pollute geometry
        x = x_t * valid_mask.unsqueeze(-1).float()

        for layer in self.layers:
            h, x = layer(h, x, valid_mask)

        atom_logits = self.atom_head(h)    # [B, N, K]
        coord_noise = self.coord_head(h)   # [B, N, 3]

        # Zero-out predictions for padded positions
        pad_float = valid_mask.unsqueeze(-1).float()
        coord_noise = coord_noise * pad_float

        return atom_logits, coord_noise


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ZINCDataset(Dataset):
    """
    Holds padded ZINC-250K records.

    Each record: (z_pad [MAX_ATOMS], pos_pad [MAX_ATOMS,3],
                  valid [MAX_ATOMS], logP, qed, SAS)
    """

    def __init__(self, records: list):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        z, pos, valid, logP, qed, sas = self.records[i]
        return (
            z,                                    # LongTensor [MAX_ATOMS]
            pos,                                  # FloatTensor [MAX_ATOMS, 3]
            valid,                                # BoolTensor [MAX_ATOMS]
            torch.tensor([logP, qed, sas], dtype=torch.float),
        )


def _collate(batch):
    z, pos, valid, cond = zip(*batch)
    return (
        torch.stack(z),
        torch.stack(pos),
        torch.stack(valid),
        torch.stack(cond),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def download_zinc(dest: Path) -> Path:
    """Download ZINC-250K CSV if not already cached."""
    if dest.exists():
        print(f"ZINC CSV already cached at {dest}", flush=True)
        return dest
    print(f"Downloading ZINC-250K from {ZINC_URL} …", flush=True)
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                ZINC_URL, headers={"User-Agent": "AegisMind/1.0"}
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                dest.write_bytes(r.read())
            print(f"  Saved {dest.stat().st_size / 1e6:.1f} MB", flush=True)
            return dest
        except Exception as e:
            print(f"  Download attempt {attempt+1} failed: {e}", flush=True)
            if attempt < 2:
                time.sleep(5)
    raise RuntimeError("Failed to download ZINC-250K after 3 attempts")


def build_zinc_dataset(
    csv_path: Path,
    cache_path: Path,
    max_mols: Optional[int] = None,
) -> ZINCDataset:
    """
    Parse ZINC CSV, convert SMILES → padded (z, pos) tensors.
    Results are cached as a .pt file to avoid re-processing on restart.
    """
    if cache_path.exists():
        print(f"Loading cached ZINC dataset from {cache_path}", flush=True)
        records = torch.load(cache_path, weights_only=False)
        return ZINCDataset(records)

    print("Building ZINC dataset (this takes ~10-20 min on first run) …", flush=True)

    # Parse CSV manually to avoid pandas dependency issues on TPU VM
    lines = csv_path.read_text().splitlines()
    header = [c.strip().strip('"') for c in lines[0].split(",")]
    # Columns: smiles, logP, qed, SAS  (order may vary)
    try:
        idx_smi = header.index("smiles")
        idx_logp = header.index("logP")
        idx_qed  = header.index("qed")
        idx_sas  = header.index("SAS")
    except ValueError:
        # Fallback: try lowercase
        header_lc = [h.lower() for h in header]
        idx_smi  = header_lc.index("smiles")
        idx_logp = header_lc.index("logp")
        idx_qed  = header_lc.index("qed")
        idx_sas  = header_lc.index("sas")

    data_lines = lines[1:]
    if max_mols is not None:
        data_lines = data_lines[:max_mols]

    records = []
    n_total = len(data_lines)
    n_fail = 0
    t0 = time.time()

    for i, line in enumerate(data_lines):
        parts = line.split(",")
        if len(parts) <= max(idx_smi, idx_logp, idx_qed, idx_sas):
            n_fail += 1
            continue

        smiles = parts[idx_smi].strip().strip('"')
        try:
            logP = float(parts[idx_logp])
            qed  = float(parts[idx_qed])
            sas  = float(parts[idx_sas])
        except ValueError:
            n_fail += 1
            continue

        result = smiles_to_graph(smiles)
        if result is None:
            n_fail += 1
            continue

        z_raw, pos_raw = result
        n_atoms = z_raw.shape[0]

        if n_atoms > MAX_ATOMS:
            # Truncate to MAX_ATOMS (rare; cap for TPU shape stability)
            z_raw = z_raw[:MAX_ATOMS]
            pos_raw = pos_raw[:MAX_ATOMS]
            n_atoms = MAX_ATOMS

        # Map atomic numbers to indices
        z_mapped = torch.zeros(MAX_ATOMS, dtype=torch.long)
        for j, an in enumerate(z_raw.tolist()):
            z_mapped[j] = ATOM_MAP.get(int(an), 9)  # unknown → index 9

        pos_pad = torch.zeros(MAX_ATOMS, 3, dtype=torch.float)
        pos_pad[:n_atoms] = pos_raw

        # Centre coordinates at centroid of valid atoms
        centroid = pos_raw.mean(0)
        pos_pad[:n_atoms] = pos_pad[:n_atoms] - centroid

        valid = torch.zeros(MAX_ATOMS, dtype=torch.bool)
        valid[:n_atoms] = True

        records.append((z_mapped, pos_pad, valid, logP, qed, sas))

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_total - i - 1) / max(rate, 1e-6)
            print(
                f"  {i+1}/{n_total} processed | "
                f"{len(records)} valid | {n_fail} failed | "
                f"ETA {eta/60:.1f} min",
                flush=True,
            )

    print(
        f"Dataset built: {len(records)} valid / {n_total} total "
        f"({n_fail} failed)",
        flush=True,
    )

    torch.save(records, cache_path)
    print(f"Cached to {cache_path}", flush=True)
    return ZINCDataset(records)


def compute_cond_stats(dataset: ZINCDataset):
    """Compute mean and std of (logP, qed, SAS) for standardisation."""
    vals = torch.stack([
        torch.tensor([r[3], r[4], r[5]], dtype=torch.float)
        for r in dataset.records
    ])
    mean = vals.mean(0)
    std  = vals.std(0).clamp(min=1e-6)
    return mean, std


# ─────────────────────────────────────────────────────────────────────────────
# GCS upload
# ─────────────────────────────────────────────────────────────────────────────

def gcs_upload(local_path: Path, remote_path: str):
    """Upload a single file to GCS via gsutil."""
    try:
        result = subprocess.run(
            ["gsutil", "-q", "cp", str(local_path), remote_path],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            print(f"  Uploaded {local_path.name} → {remote_path}", flush=True)
        else:
            print(f"  GCS upload failed: {result.stderr.strip()}", flush=True)
    except Exception as e:
        print(f"  GCS upload error: {e}", flush=True)


def gcs_upload_dir(local_dir: Path, remote_dir: str):
    """Upload all files in a directory to GCS."""
    try:
        subprocess.run(
            ["gsutil", "-q", "-m", "cp", "-r", str(local_dir) + "/*", remote_dir + "/"],
            capture_output=True, text=True, timeout=600,
        )
        print(f"  Uploaded {local_dir} → {remote_dir}/", flush=True)
    except Exception as e:
        print(f"  GCS upload_dir error: {e}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def edm_loss(
    model: EDMDenoiser,
    schedule: DiffusionSchedule,
    z0: torch.Tensor,       # [B, N] long
    x0: torch.Tensor,       # [B, N, 3] float
    valid: torch.Tensor,    # [B, N] bool
    cond: torch.Tensor,     # [B, 3] float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single EDM training step.
    Returns (total_loss, coord_loss, atom_loss).
    """
    B = z0.shape[0]
    device = z0.device

    # Sample random timesteps
    t = torch.randint(0, schedule.T, (B,), device=device, dtype=torch.long)

    # Forward diffusion
    x_t, coord_noise = schedule.q_sample_coords(x0, t)
    z_t = schedule.q_sample_atoms(z0, t)

    # Predict
    atom_logits, coord_noise_pred = model(z_t, x_t, t, valid, cond)

    # Coordinate loss (MSE on noise, only over valid atoms)
    valid_f = valid.unsqueeze(-1).float()            # [B, N, 1]
    n_valid = valid.float().sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
    coord_err = (coord_noise_pred - coord_noise) ** 2 * valid_f      # [B, N, 3]
    coord_loss = (coord_err.sum(dim=(-1, -2)) / n_valid).mean()

    # Atom type loss (cross-entropy, only over valid atoms)
    # atom_logits: [B, N, K]; z0: [B, N]
    B2, N2, K = atom_logits.shape
    atom_logits_flat = atom_logits.reshape(B2 * N2, K)
    z0_flat          = z0.reshape(B2 * N2)
    valid_flat       = valid.reshape(B2 * N2)
    # Only compute CE on valid (non-pad) positions
    atom_loss = F.cross_entropy(
        atom_logits_flat[valid_flat],
        z0_flat[valid_flat],
        ignore_index=0,
    ) if valid_flat.any() else torch.tensor(0.0, device=device)

    total = COORD_LOSS_W * coord_loss + ATOM_LOSS_W * atom_loss
    return total, coord_loss, atom_loss


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_molecules(
    model: EDMDenoiser,
    schedule: DiffusionSchedule,
    n_mols: int,
    device,
    cond: Optional[torch.Tensor] = None,
    n_steps: int = N_DDIM_STEPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DDIM-style reverse diffusion sampler.

    Returns:
        x_final: [n_mols, MAX_ATOMS, 3] — predicted coordinates
        z_final: [n_mols, MAX_ATOMS]    — predicted atom types
    """
    model.eval()
    B, N = n_mols, MAX_ATOMS

    x_t = torch.randn(B, N, 3, device=device)
    z_t = torch.randint(1, N_ATOM_TYPES, (B, N), device=device, dtype=torch.long)
    valid = torch.ones(B, N, dtype=torch.bool, device=device)

    if cond is not None:
        cond = cond.to(device)
        if cond.shape[0] == 1:
            cond = cond.expand(B, -1)

    timesteps = torch.linspace(schedule.T - 1, 0, n_steps, dtype=torch.long)

    for step_idx, t_val in enumerate(timesteps.tolist()):
        t_val = int(t_val)
        t = torch.full((B,), t_val, dtype=torch.long, device=device)

        atom_logits, coord_noise_pred = model(z_t, x_t, t, valid, cond)

        # Denoise coordinates (DDIM deterministic update)
        alpha_t = schedule.alphas_cumprod[t_val].clamp(min=1e-8)
        sqrt_one_minus = (1.0 - alpha_t).sqrt()
        sqrt_alpha = alpha_t.sqrt()
        x_t = (x_t - sqrt_one_minus * coord_noise_pred) / sqrt_alpha

        # Update atom types from softmax (temperature=1)
        probs = F.softmax(atom_logits, dim=-1)
        z_t = torch.multinomial(
            probs.reshape(B * N, N_ATOM_TYPES),
            num_samples=1,
        ).reshape(B, N)
        # Clamp: never sample pad token
        z_t = z_t.clamp(min=1)

        if XLA_AVAILABLE and (step_idx + 1) % 10 == 0:
            xm.mark_step()

    z_final = atom_logits.argmax(-1).clamp(min=1)
    return x_t, z_final


# ─────────────────────────────────────────────────────────────────────────────
# SMILES reconstruction
# ─────────────────────────────────────────────────────────────────────────────

# Reverse atom index → atomic number for SMILES reconstruction
_IDX_TO_ATOMNUM = {v: k for k, v in ATOM_MAP.items()}
_IDX_TO_ATOMNUM[0] = 6   # fallback pad → carbon (should never appear)
_ATOMNUM_TO_SYMBOL = {
    1: "H", 6: "C", 7: "N", 8: "O", 9: "F",
    15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I",
}

# Bond-distance thresholds (Angstrom) — very rough but workable for heuristic
_BOND_THRESHOLDS = {
    (6, 6): 1.7,   # C-C
    (6, 7): 1.6,   # C-N
    (6, 8): 1.6,   # C-O
    (6, 9): 1.5,   # C-F
    (6, 16): 1.9,  # C-S
    (6, 17): 1.9,  # C-Cl
    (6, 35): 2.1,  # C-Br
    (6, 53): 2.3,  # C-I
    (7, 7): 1.6,   # N-N
    (7, 8): 1.6,   # N-O
    (8, 15): 1.7,  # O-P
    (16, 16): 2.2, # S-S
    (1, 6): 1.2,   # H-C
    (1, 7): 1.2,   # H-N
    (1, 8): 1.1,   # H-O
}


def _bond_threshold(an1: int, an2: int) -> float:
    key = (min(an1, an2), max(an1, an2))
    return _BOND_THRESHOLDS.get(key, 2.0)


def coords_types_to_smiles(
    x: torch.Tensor,       # [MAX_ATOMS, 3]
    z_types: torch.Tensor, # [MAX_ATOMS] atom-type indices
    valid_mask: Optional[torch.Tensor] = None,
) -> Optional[str]:
    """
    Heuristically convert predicted 3D coords + atom-type indices to SMILES
    via RDKit: add atoms, infer bonds by distance threshold, sanitize.
    Returns SMILES string or None on failure.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return None

    x_np = x.cpu().float().numpy()
    z_np = z_types.cpu().long().numpy()

    # Determine which positions are valid
    if valid_mask is not None:
        valid_np = valid_mask.cpu().bool().numpy()
    else:
        valid_np = (z_np > 0)

    atom_indices = [i for i in range(MAX_ATOMS) if valid_np[i] and z_np[i] > 0]
    if len(atom_indices) < 3:
        return None

    mol = Chem.RWMol()
    conf = Chem.Conformer(len(atom_indices))

    # Map position_in_mol → atom_number
    mol_atom_nums = []
    for new_idx, orig_idx in enumerate(atom_indices):
        type_idx = int(z_np[orig_idx])
        an = _IDX_TO_ATOMNUM.get(type_idx, 6)
        sym = _ATOMNUM_TO_SYMBOL.get(an, "C")
        a = Chem.Atom(sym)
        mol.AddAtom(a)
        pos = x_np[orig_idx]
        conf.SetAtomPosition(new_idx, (float(pos[0]), float(pos[1]), float(pos[2])))
        mol_atom_nums.append(an)

    n_mol_atoms = len(atom_indices)

    # Infer bonds by pairwise distance threshold
    for i in range(n_mol_atoms):
        for j in range(i + 1, n_mol_atoms):
            xi = x_np[atom_indices[i]]
            xj = x_np[atom_indices[j]]
            dist = float(np.linalg.norm(xi - xj))
            thresh = _bond_threshold(mol_atom_nums[i], mol_atom_nums[j])
            if dist < thresh:
                mol.AddBond(i, j, Chem.BondType.SINGLE)

    try:
        mol.AddConformer(conf, assignId=True)
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)
        # Basic validity: at least 3 heavy atoms and parseable
        check = Chem.MolFromSmiles(smiles)
        if check is None or check.GetNumAtoms() < 3:
            return None
        return smiles
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Validity metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_validity_metrics(
    smiles_list: List[Optional[str]],
    training_smiles: Optional[set] = None,
) -> dict:
    """
    Compute valid%, unique%, novel% for a list of generated SMILES.
    novel% is relative to the training set if provided.
    """
    total = len(smiles_list)
    valid = [s for s in smiles_list if s is not None]
    n_valid = len(valid)
    unique = set(valid)
    n_unique = len(unique)
    n_novel = (
        len(unique - training_smiles)
        if training_smiles is not None
        else n_unique
    )

    return {
        "total":       total,
        "n_valid":     n_valid,
        "valid_pct":   100.0 * n_valid / max(total, 1),
        "n_unique":    n_unique,
        "unique_pct":  100.0 * n_unique / max(n_valid, 1),
        "n_novel":     n_novel,
        "novel_pct":   100.0 * n_novel / max(n_unique, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    notify("PHASE_START", "Phase 12: EDM diffusion model training on ZINC-250K", data={
        "model":       "EDMDenoiser (EGNN)",
        "hidden_dim":  HIDDEN_DIM,
        "n_layers":    N_LAYERS,
        "T_diff":      T_DIFF,
        "max_atoms":   MAX_ATOMS,
        "batch_size":  BATCH_SIZE,
        "n_epochs":    N_EPOCHS,
        "bf16":        True,
        "xla":         XLA_AVAILABLE,
        "gcs_output":  GCS_BASE,
    })

    # ── Device ────────────────────────────────────────────────────────────────
    if XLA_AVAILABLE:
        device = xm.xla_device()
        print(f"Device: {device} (XLA TPU)", flush=True)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: {device} (GPU)", flush=True)
    else:
        device = torch.device("cpu")
        print("Device: CPU (no accelerator found)", flush=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n=== STEP 1: Data preparation ===", flush=True)
    download_zinc(ZINC_CSV)

    cache_path = DATA_DIR / "zinc_records.pt"
    dataset = build_zinc_dataset(ZINC_CSV, cache_path)
    print(f"Total molecules: {len(dataset)}", flush=True)

    # Compute conditioning stats for standardisation
    cond_mean, cond_std = compute_cond_stats(dataset)
    print(f"Cond mean: {cond_mean.tolist()}", flush=True)
    print(f"Cond std:  {cond_std.tolist()}", flush=True)

    # Save cond stats for sampling
    cond_stats = {"mean": cond_mean.tolist(), "std": cond_std.tolist()}
    (OUT_DIR / "cond_stats.json").write_text(json.dumps(cond_stats, indent=2))

    # Build normalised dataset (modify records in place)
    for rec in dataset.records:
        z, pos, valid, logP, qed, sas = rec
        # We'll normalise during training/sampling via the stats tensors

    # Collect training SMILES for novelty metric (from raw CSV)
    print("Collecting training SMILES set for novelty scoring …", flush=True)
    lines = ZINC_CSV.read_text().splitlines()
    header_lc = [c.strip().strip('"').lower() for c in lines[0].split(",")]
    try:
        smi_col = header_lc.index("smiles")
    except ValueError:
        smi_col = 0
    training_smiles_set = set()
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) > smi_col:
            training_smiles_set.add(parts[smi_col].strip().strip('"'))
    print(f"Training SMILES set: {len(training_smiles_set)} entries", flush=True)

    # Train / val split (90/10)
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=_collate,
        num_workers=4,
        pin_memory=False,  # XLA requires pin_memory=False
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=_collate,
        num_workers=2,
        pin_memory=False,
        drop_last=False,
    )

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}", flush=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n=== STEP 2: Model setup ===", flush=True)
    model = EDMDenoiser(
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        n_atom_types=N_ATOM_TYPES,
        cond_dim=COND_DIM,
        T=T_DIFF,
    ).to(device)
    print(f"EDMDenoiser parameters: {model.parameter_count():,}", flush=True)

    schedule = DiffusionSchedule(T=T_DIFF).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=LR * 0.01,
    )

    cond_mean_dev = cond_mean.to(device)
    cond_std_dev  = cond_std.to(device)

    # BF16 autocast context
    if XLA_AVAILABLE:
        # XLA BF16 via autocast
        from contextlib import contextmanager

        @contextmanager
        def bf16_ctx():
            with torch.autocast(device_type="xla", dtype=torch.bfloat16):
                yield
    elif torch.cuda.is_available():
        from contextlib import contextmanager

        @contextmanager
        def bf16_ctx():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                yield
    else:
        from contextlib import contextmanager

        @contextmanager
        def bf16_ctx():
            yield  # CPU: no autocast, run FP32

    # ── Training ──────────────────────────────────────────────────────────────
    print("\n=== STEP 3: Training ===", flush=True)
    loss_trajectory = []
    best_val_loss   = float("inf")

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        epoch_loss_sum  = 0.0
        epoch_coord_sum = 0.0
        epoch_atom_sum  = 0.0
        n_batches = 0
        t_ep = time.time()

        for batch in train_loader:
            z0, x0, valid, cond_raw = batch
            z0    = z0.to(device)
            x0    = x0.to(device)
            valid = valid.to(device)
            cond  = ((cond_raw.to(device) - cond_mean_dev) / cond_std_dev)

            optimizer.zero_grad()

            with bf16_ctx():
                loss, coord_loss, atom_loss = edm_loss(
                    model, schedule, z0, x0, valid, cond
                )

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if XLA_AVAILABLE:
                xm.mark_step()

            epoch_loss_sum  += loss.item()
            epoch_coord_sum += coord_loss.item()
            epoch_atom_sum  += atom_loss.item()
            n_batches += 1

        lr_scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        if epoch % EVAL_EVERY == 0 or epoch == N_EPOCHS:
            model.eval()
            val_loss_sum = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    z0, x0, valid, cond_raw = batch
                    z0    = z0.to(device)
                    x0    = x0.to(device)
                    valid = valid.to(device)
                    cond  = (cond_raw.to(device) - cond_mean_dev) / cond_std_dev

                    with bf16_ctx():
                        val_loss, _, _ = edm_loss(model, schedule, z0, x0, valid, cond)

                    if XLA_AVAILABLE:
                        xm.mark_step()

                    val_loss_sum += val_loss.item()
                    n_val_batches += 1

            avg_train = epoch_loss_sum / max(n_batches, 1)
            avg_val   = val_loss_sum   / max(n_val_batches, 1)
            avg_coord = epoch_coord_sum / max(n_batches, 1)
            avg_atom  = epoch_atom_sum  / max(n_batches, 1)
            elapsed   = time.time() - t_ep

            print(
                f"Epoch {epoch:3d}/{N_EPOCHS} | "
                f"train={avg_train:.4f} (coord={avg_coord:.4f} atom={avg_atom:.4f}) | "
                f"val={avg_val:.4f} | "
                f"lr={lr_scheduler.get_last_lr()[0]:.2e} | "
                f"{elapsed:.1f}s",
                flush=True,
            )

            record = {
                "epoch":       epoch,
                "train_loss":  round(avg_train, 6),
                "val_loss":    round(avg_val, 6),
                "coord_loss":  round(avg_coord, 6),
                "atom_loss":   round(avg_atom, 6),
                "lr":          lr_scheduler.get_last_lr()[0],
            }
            loss_trajectory.append(record)

            # Heartbeat notification
            heartbeat("phase12_diffusion_zinc", epoch, {
                "train_loss": avg_train,
                "val_loss":   avg_val,
            })

            # Save checkpoint
            ckpt_path = CKPT_DIR / f"edm_epoch{epoch:04d}.pt"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "sched_state": lr_scheduler.state_dict(),
                "val_loss":    avg_val,
                "cond_mean":   cond_mean.tolist(),
                "cond_std":    cond_std.tolist(),
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}", flush=True)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_ckpt = CKPT_DIR / "edm_best.pt"
                torch.save(torch.load(ckpt_path, weights_only=False), best_ckpt)

            # Upload checkpoint to GCS
            gcs_upload(ckpt_path, f"{GCS_BASE}/checkpoints/{ckpt_path.name}")

        else:
            # Still log train loss every epoch
            avg_train = epoch_loss_sum / max(n_batches, 1)
            avg_coord = epoch_coord_sum / max(n_batches, 1)
            avg_atom  = epoch_atom_sum  / max(n_batches, 1)
            elapsed   = time.time() - t_ep
            print(
                f"Epoch {epoch:3d}/{N_EPOCHS} | "
                f"train={avg_train:.4f} (coord={avg_coord:.4f} atom={avg_atom:.4f}) | "
                f"lr={lr_scheduler.get_last_lr()[0]:.2e} | "
                f"{elapsed:.1f}s",
                flush=True,
            )
            loss_trajectory.append({
                "epoch":      epoch,
                "train_loss": round(avg_train, 6),
                "coord_loss": round(avg_coord, 6),
                "atom_loss":  round(avg_atom, 6),
                "lr":         lr_scheduler.get_last_lr()[0],
            })

    # Save full loss trajectory
    results_path = OUT_DIR / "training_results.json"
    results_path.write_text(json.dumps({
        "config": {
            "hidden_dim":  HIDDEN_DIM,
            "n_layers":    N_LAYERS,
            "T_diff":      T_DIFF,
            "max_atoms":   MAX_ATOMS,
            "n_epochs":    N_EPOCHS,
            "batch_size":  BATCH_SIZE,
            "lr":          LR,
            "weight_decay": WEIGHT_DECAY,
            "coord_loss_w": COORD_LOSS_W,
            "atom_loss_w":  ATOM_LOSS_W,
            "n_params":    model.parameter_count(),
        },
        "best_val_loss":   best_val_loss,
        "loss_trajectory": loss_trajectory,
        "elapsed_s":       round(time.time() - t_start, 1),
    }, indent=2))
    gcs_upload(results_path, f"{GCS_BASE}/training_results.json")
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}", flush=True)

    # ── Generation ────────────────────────────────────────────────────────────
    print("\n=== STEP 4: Generating molecules ===", flush=True)

    # Load best checkpoint for generation
    best_ckpt_path = CKPT_DIR / "edm_best.pt"
    if best_ckpt_path.exists():
        state = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        print(f"Loaded best checkpoint (val_loss={state['val_loss']:.4f})", flush=True)

    # Use zero conditioning (mean molecule) for unconditional-like generation
    # and also sample with two interesting conditions
    cond_configs = [
        ("drug_like",    torch.tensor([[2.5, 0.8, 2.5]])),  # typical drug-like
        ("high_logp",    torch.tensor([[4.5, 0.6, 3.0]])),  # lipophilic
        ("low_sas",      torch.tensor([[1.5, 0.7, 1.5]])),  # easy to synthesize
        ("unconditional", torch.tensor([[0.0, 0.5, 3.0]])), # neutral
    ]

    all_smiles    = []
    samples_meta  = []
    batch_gen     = 64   # generate in batches for memory

    n_per_config = N_SAMPLES // len(cond_configs)
    remainder    = N_SAMPLES - n_per_config * len(cond_configs)

    for cfg_idx, (cfg_name, raw_cond) in enumerate(cond_configs):
        n_this = n_per_config + (remainder if cfg_idx == 0 else 0)
        print(f"  Generating {n_this} molecules (config: {cfg_name}) …", flush=True)
        cfg_smiles = []

        # Standardise the conditioning vector
        std_cond = (raw_cond.to(device) - cond_mean_dev) / cond_std_dev

        generated = 0
        while generated < n_this:
            b = min(batch_gen, n_this - generated)
            cond_batch = std_cond.expand(b, -1)

            with torch.no_grad():
                x_gen, z_gen = sample_molecules(
                    model, schedule, b, device,
                    cond=cond_batch, n_steps=N_DDIM_STEPS,
                )

            for mol_i in range(b):
                smi = coords_types_to_smiles(
                    x_gen[mol_i], z_gen[mol_i], valid_mask=None,
                )
                cfg_smiles.append(smi)
                samples_meta.append({
                    "config": cfg_name,
                    "logP_target": float(raw_cond[0, 0]),
                    "qed_target":  float(raw_cond[0, 1]),
                    "SAS_target":  float(raw_cond[0, 2]),
                    "smiles": smi,
                })
            generated += b
            if XLA_AVAILABLE:
                xm.mark_step()

        n_valid_cfg = sum(1 for s in cfg_smiles if s is not None)
        print(
            f"    Config {cfg_name}: {n_valid_cfg}/{n_this} valid "
            f"({100*n_valid_cfg/max(n_this,1):.1f}%)",
            flush=True,
        )
        all_smiles.extend(cfg_smiles)

    # ── Validity metrics ──────────────────────────────────────────────────────
    print("\n=== STEP 5: Validity metrics ===", flush=True)
    metrics = compute_validity_metrics(all_smiles, training_smiles=training_smiles_set)
    print(
        f"  Valid:  {metrics['valid_pct']:.1f}% ({metrics['n_valid']}/{metrics['total']})",
        flush=True,
    )
    print(
        f"  Unique: {metrics['unique_pct']:.1f}% ({metrics['n_unique']} distinct valid SMILES)",
        flush=True,
    )
    print(
        f"  Novel:  {metrics['novel_pct']:.1f}% "
        f"({metrics['n_novel']} not in training set)",
        flush=True,
    )

    # Save samples
    samples_path = OUT_DIR / "samples_final.json"
    samples_path.write_text(json.dumps({
        "n_generated": N_SAMPLES,
        "n_ddim_steps": N_DDIM_STEPS,
        "samples": samples_meta,
    }, indent=2))

    # Save validity metrics
    metrics_path = OUT_DIR / "validity_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # ── Upload to GCS ─────────────────────────────────────────────────────────
    print("\n=== STEP 6: GCS upload ===", flush=True)
    gcs_upload(samples_path,  f"{GCS_BASE}/samples_final.json")
    gcs_upload(metrics_path,  f"{GCS_BASE}/validity_metrics.json")
    gcs_upload(results_path,  f"{GCS_BASE}/training_results.json")
    gcs_upload(best_ckpt_path, f"{GCS_BASE}/checkpoints/edm_best.pt")
    gcs_upload(OUT_DIR / "cond_stats.json", f"{GCS_BASE}/cond_stats.json")

    total_elapsed = time.time() - t_start
    notify("PHASE_COMPLETE", "Phase 12: EDM diffusion on ZINC-250K complete", data={
        "best_val_loss":   round(best_val_loss, 4),
        "valid_pct":       round(metrics["valid_pct"], 1),
        "unique_pct":      round(metrics["unique_pct"], 1),
        "novel_pct":       round(metrics["novel_pct"], 1),
        "n_samples":       N_SAMPLES,
        "elapsed_min":     round(total_elapsed / 60, 1),
        "gcs_output":      GCS_BASE,
    })
    print(f"\nPhase 12 complete in {total_elapsed/60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
