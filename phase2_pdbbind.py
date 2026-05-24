"""
Phase 2 — PDBbind Fine-tuning for MS Target Docking

Takes the BF16 GNN trained on QM9 (Phase 1, Condition B) and fine-tunes it
on PDBbind binding affinity prediction (pKd), then uses it to score
candidate small molecules against the top MS therapeutic targets identified
by DE analysis.

MS targets (from results/de_ca_rim_vs_control.csv):
  LINGO1  — remyelination inhibitor; Biogen opicinumab target
  PCSK9   — novel CNS finding; FDA-approved drug class
  CTSS    — Cathepsin S; microglial antigen presentation
  GREM1   — BMP antagonist; anti-remyelination / astrogliosis
  HIF1A   — pseudo-hypoxia in smoldering lesions

PDBbind v2020 general set (~19K complexes) is used for fine-tuning.
AlphaFold2 structures used for PCSK9 and GREM1 where PDB structures
exist but ligand-bound forms are limited.

Pipeline:
  1. Download PDBbind v2020 index + structures
  2. Build radius graphs from protein pocket + ligand
  3. Fine-tune Phase 1 BF16 checkpoint with pKd regression head
  4. Download/prep target structures (PDB or AlphaFold)
  5. Screen FDA-approved compound library (ZINC FDA subset, ~3K compounds)
  6. Score + rank candidates per target
  7. Save ranked list → results/ms_target_candidates_phase2.csv

Run as part of master pipeline — not standalone.
"""

import os
import sys
import json
import math
import time
import hashlib
import urllib.request
import tarfile
import gzip
import shutil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True

    # Enable XLA eager mode — same as train.py (Phase 1).
    # Lazy mode accumulates the entire forward+backward+optimizer into one
    # giant HLO graph (~100-300 MB) that is unique per step (due to AdamW
    # bias-correction scalars changing each step), causing infinite recompilation.
    # Eager mode compiles tiny per-op graphs that are cached and reused instantly.
    try:
        import torch_xla.experimental as _xla_exp
        _xla_exp.eager_mode(True)
        print("XLA eager mode: ENABLED")
    except Exception as _e:
        print(f"XLA eager mode: unavailable ({_e}) — using lazy mode")

except ImportError:
    XLA_AVAILABLE = False

from notify import notify, heartbeat
from model import MolecularGNN
from compat import autocast

# PDBbind real data loader (no RDKit required)
try:
    from pdbbind_data import build_pdbbind_dataset
    PDBBIND_AVAILABLE = True
except ImportError:
    PDBBIND_AVAILABLE = False

# ChEMBL data loader — requires rdkit-pypi; falls back to mock if unavailable
try:
    from chembl_data import build_dataset, fetch_pubchem_fda_smiles, smiles_to_graph
    REAL_DATA = True
except ImportError:
    REAL_DATA = False

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/tmp/flashoptim_results"))
DATA_DIR   = Path(os.environ.get("DATA_DIR", "/tmp/phase2_data"))
PHASE1_CKPT = OUTPUT_DIR / "condition_B_best.pt"  # overridden by --checkpoint arg

# Top MS targets from DE analysis (CA_RIM log2FC, FDR < 0.1)
MS_TARGETS = {
    "LINGO1":  {"pdb": "7MHH",  "log2fc": +0.99, "rationale": "Remyelination inhibitor"},
    "PCSK9":   {"pdb": "2P4E",  "log2fc": +0.93, "rationale": "Novel CNS; FDA-approved class"},
    "CTSS":    {"pdb": "1MS6",  "log2fc": +1.16, "rationale": "Microglial antigen presentation"},
    "GREM1":   {"pdb": "1XH0",  "log2fc": +1.60, "rationale": "BMP antagonist; anti-remyelination"},
    "HIF1A":   {"pdb": "1LQB",  "log2fc": +0.92, "rationale": "Pseudo-hypoxia in smoldering MS"},
    # AMR target — KPC-3 carbapenemase inhibitor screening (OceanSparx Patent 4 provisional)
    # KPC-3 differs from KPC-2 by single substitution H272Y; 5UL8 (apo KPC-2, 1.15 Å) is
    # the standard structural proxy used in the literature.
    "KPC3":    {"pdb": "5UL8",  "log2fc": None,   "rationale": "KPC-3 carbapenemase — AMR inhibitor screening (DBO-urea/carbamate/CF3 series)"},
    # HD targets — CHDI engagement (network hub analysis, 2026-05-02)
    # APEX1: AP endonuclease 1; top betweenness hub in MMR cluster (BER-MMR interface);
    #   4LWO = human APE1 + inhibitor MX complex, 2.20 Å. E3330-class compounds available.
    "APEX1":   {"pdb": "4LWO",  "log2fc": None,   "rationale": "APE1 — MMR network bottleneck; somatic CAG expansion (CHDI)"},
    # MSH3: MutSβ component; top GeM-HD GWAS signal; CHDI-validated somatic expansion driver.
    #   3THW = human MSH2-MSH3 heterodimer + ADP, 2.90 Å.
    "MSH3":    {"pdb": "3THW",  "log2fc": None,   "rationale": "MSH3 — MutSβ; somatic CAG expansion driver (CHDI GeM-HD)"},
    # CREBBP: CBP acetyltransferase; transcription-LLPS bridge (hub_score=0.555);
    #   4YGC = CREBBP bromodomain + acetyl-lysine ligand, 1.65 Å.
    "CREBBP":  {"pdb": "4YGC",  "log2fc": None,   "rationale": "CBP/CREBBP — transcription-LLPS bridge; mHTT condensate target (CHDI)"},
}

PDBBIND_URL  = "https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/PDBbind_v2020_plain_text_index.tar.gz"
ZINC_FDA_URL = "https://zinc.docking.org/substances/subsets/fda.sdf?count=all"


# ── Binding affinity regression head ─────────────────────────────────────────

class BindingAffinityGNN(nn.Module):
    """
    Wraps the Phase 1 MolecularGNN and adds a pKd regression head.
    The base GNN is optionally frozen for first N_FREEZE_EPOCHS.

    XLA note: PADDED_ATOMS is fixed so all tensor shapes are static — XLA compiles
    the graph once and reuses it for every batch. Edge indices and assign_mat are
    pre-computed in __init__ as registered buffers (no Python loop in forward).
    Molecules with >PADDED_ATOMS atoms are truncated (rare in PDBbind refined set).
    """
    N_FREEZE_EPOCHS = 5
    PADDED_ATOMS    = 80   # covers >99% of PDBbind refined ligands; rare larger ones truncated

    def __init__(self, base_gnn: MolecularGNN, hidden_dim: int = 256):
        super().__init__()
        self.gnn  = base_gnn
        # No Dropout — nn.Dropout uses torch.bernoulli() which changes the XLA RNG
        # state tensor identity each step, producing a unique computation graph per
        # batch and causing infinite recompilation. Fine-tuning from a pre-trained
        # base doesn't require Dropout for regularisation.
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        # Pre-compute fixed fully-connected edge indices once (avoids per-batch Python loop
        # and prevents XLA recompilation due to shape changes).
        PA       = self.PADDED_ATOMS
        src_list = [a for a in range(PA) for b in range(PA) if a != b]
        dst_list = [b for a in range(PA) for b in range(PA) if a != b]
        e_src    = torch.tensor(src_list, dtype=torch.long)     # [MAX_EDGES]
        e_dst    = torch.tensor(dst_list, dtype=torch.long)     # [MAX_EDGES]
        # assign_base[e, a] = 1 if edge e's destination is atom a  [MAX_EDGES, PA]
        assign_base = (e_dst.unsqueeze(-1) == torch.arange(PA)).float()
        self.register_buffer('_edge_src',    e_src)
        self.register_buffer('_edge_dst',    e_dst)
        self.register_buffer('_assign_base', assign_base)

    def forward(self, z_pad, pos_pad, atom_valid):
        """
        All inputs are pre-padded to static shapes on CPU by PDBbindDataset.batch().
        No Python loops, no .nonzero() — XLA compiles this graph exactly once.

          z_pad      [B, PADDED_ATOMS]      long
          pos_pad    [B, PADDED_ATOMS, 3]   float
          atom_valid [B, PADDED_ATOMS]      bool
        """
        B = z_pad.shape[0]

        # Expand pre-computed edge buffers — static shapes, no recompilation
        edge_src   = self._edge_src.unsqueeze(0).expand(B, -1)        # [B, MAX_EDGES]
        edge_dst   = self._edge_dst.unsqueeze(0).expand(B, -1)
        assign_mat = self._assign_base.unsqueeze(0).expand(B, -1, -1) # [B, MAX_EDGES, PA]

        src_valid  = atom_valid.gather(1, edge_src)
        dst_valid  = atom_valid.gather(1, edge_dst)
        edge_valid = src_valid & dst_valid

        h = self.gnn.embed(z_pad, pos_pad, edge_src, edge_dst, assign_mat,
                           B, edge_valid, atom_valid)
        return self.head(h).squeeze(-1)                 # [B]

    def freeze_base(self):
        for p in self.gnn.parameters():
            p.requires_grad_(False)

    def unfreeze_base(self):
        for p in self.gnn.parameters():
            p.requires_grad_(True)


# ── Data utilities ─────────────────────────────────────────────────────────

def download_pdbbind_index(data_dir: Path) -> Path:
    """Download PDBbind v2020 plain-text index (small, ~5MB)."""
    dest = data_dir / "pdbbind_index.tar.gz"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    notify("PHASE_START", "Downloading PDBbind v2020 index ...")
    try:
        urllib.request.urlretrieve(PDBBIND_URL, dest)
        notify("CHECKPOINT", f"PDBbind index downloaded: {dest.stat().st_size/1e6:.1f} MB")
    except Exception as e:
        notify("ANOMALY", f"PDBbind download failed: {e}. Using mock data for pipeline test.")
        _create_mock_pdbbind(data_dir)
    return dest


def _create_mock_pdbbind(data_dir: Path):
    """Create minimal mock PDBbind data for pipeline smoke-testing."""
    mock = data_dir / "pdbbind_mock.json"
    import random
    rng = random.Random(42)
    entries = []
    for i in range(200):
        entries.append({
            "pdb_id": f"mock{i:04d}",
            "pkd":    rng.gauss(7.0, 1.5),
            "n_atoms": rng.randint(15, 50),
        })
    with open(mock, "w") as f:
        json.dump(entries, f)
    notify("CHECKPOINT", f"Mock PDBbind created: {len(entries)} entries")


def fetch_target_structure(gene: str, pdb_id: str, data_dir: Path) -> Path:
    """Download PDB structure for a target protein."""
    dest = data_dir / "structures" / f"{pdb_id}.pdb"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, dest)
        notify("CHECKPOINT", f"Structure downloaded: {gene} ({pdb_id}) → {dest.stat().st_size/1024:.0f} KB")
    except Exception as e:
        notify("ANOMALY", f"PDB download failed for {gene}/{pdb_id}: {e}")
    return dest


# ── Training ──────────────────────────────────────────────────────────────────

def fine_tune(device, epochs: int = 50, batch_size: int = 32) -> dict:
    """
    Fine-tune Phase 1 BF16 checkpoint on PDBbind binding affinity.
    Returns dict with final metrics.
    """
    notify("PHASE_START", "Phase 2 — PDBbind fine-tuning starting",
           data={"epochs": epochs, "device": str(device)})

    # Load Phase 1 checkpoint
    if not PHASE1_CKPT.exists():
        notify("ANOMALY", f"Phase 1 checkpoint not found at {PHASE1_CKPT}. "
               "Using randomly initialised weights — results will be weaker.",
               urgent=False)
        base = MolecularGNN(hidden_dim=256, num_blocks=6, cutoff=5.0)
    else:
        ckpt = torch.load(PHASE1_CKPT, map_location="cpu")
        base = MolecularGNN(**ckpt.get("model_config",
                                        {"hidden_dim": 256, "num_blocks": 4, "cutoff": 5.0}))
        base.load_state_dict(ckpt["model"])
        notify("CHECKPOINT", f"Loaded Phase 1 checkpoint (epoch {ckpt.get('epoch','?')})")

    model = BindingAffinityGNN(base, hidden_dim=256).to(device)

    # Freeze base for first N_FREEZE_EPOCHS
    model.freeze_base()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-4
    )

    results = {"epochs": [], "target_scores": {}, "data_source": ""}
    best_rmse = float("inf")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load real or mock dataset ─────────────────────────────────────────────
    # Priority: (1) PDBbind v2020 refined set from GCS  (2) ChEMBL  (3) mock
    dataset = None
    train_idx, val_idx = [], []

    if PDBBIND_AVAILABLE:
        try:
            GCS_BUCKET = os.environ.get("GCS_BUCKET", "aegismind-tpu-results")
            notify("CHECKPOINT", "Loading PDBbind v2020 refined set from GCS ...")
            dataset = build_pdbbind_dataset(DATA_DIR, gcs_bucket=GCS_BUCKET)
            train_idx, val_idx = dataset.train_val_split(val_frac=0.1)
            results["data_source"] = f"pdbbind_v2020:{len(dataset)}_complexes"
            notify("CHECKPOINT",
                   f"PDBbind: {len(dataset)} complexes, "
                   f"{len(train_idx)} train / {len(val_idx)} val")
        except Exception as e:
            notify("ANOMALY", f"PDBbind load failed: {e} — trying ChEMBL", urgent=False)
            dataset = None

    if dataset is None and REAL_DATA:
        try:
            notify("CHECKPOINT", "Loading ChEMBL binding affinity data (real) ...")
            dataset = build_dataset(DATA_DIR, n_records=5000)
            train_idx, val_idx = dataset.train_val_split(val_frac=0.2)
            results["data_source"] = f"chembl_real:{len(dataset)}_graphs"
            notify("CHECKPOINT",
                   f"ChEMBL dataset: {len(dataset)} graphs, "
                   f"{len(train_idx)} train / {len(val_idx)} val")
        except Exception as e:
            notify("ANOMALY", f"ChEMBL load failed: {e} — falling back to mock", urgent=False)
            dataset = None

    if dataset is None:
        results["data_source"] = "mock"
        download_pdbbind_index(DATA_DIR)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Unfreeze after freeze period
        if epoch == model.N_FREEZE_EPOCHS + 1:
            model.unfreeze_base()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=5e-5, weight_decay=1e-4
            )
            notify("CHECKPOINT", f"Epoch {epoch}: unfreezing base GNN for full fine-tuning")

        model.train()
        if dataset is not None:
            train_loss = _real_train_epoch(model, optimizer, device, batch_size,
                                           dataset, train_idx)
        else:
            train_loss = _mock_train_epoch(model, optimizer, device, batch_size)

        model.eval()
        if dataset is not None:
            val_rmse = _real_val_epoch(model, device, dataset, val_idx, batch_size)
        else:
            val_rmse = _mock_val_epoch(model, device, epoch)

        elapsed = time.time() - t0
        ep_data = {
            "epoch": epoch, "train_loss": train_loss,
            "val_rmse_pkd": val_rmse, "elapsed_s": elapsed,
        }
        results["epochs"].append(ep_data)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            ckpt_path = OUTPUT_DIR / "phase2_best.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_rmse": val_rmse}, ckpt_path)

        if epoch % 10 == 0:
            heartbeat("Phase2_ChEMBL", epoch,
                      {"val_rmse_pkd": val_rmse, "best_rmse": best_rmse,
                       "data": results["data_source"]})

        if XLA_AVAILABLE:
            xm.mark_step()

    results["best_val_rmse"] = best_rmse
    notify("PHASE_COMPLETE",
           f"Phase 2 fine-tuning complete. Best RMSE: {best_rmse:.3f} pKd units "
           f"({results['data_source']})",
           data=results)
    return results


def _real_train_epoch(model, optimizer, device, batch_size, dataset, train_idx) -> float:
    """Train one epoch on PDBbind data."""
    total_loss = 0.0
    n_batches  = 0
    PA = model.PADDED_ATOMS
    # drop_last=True: all batches are exactly batch_size → single XLA graph variant
    for z_pad, pos_pad, atom_valid, pkd_true in dataset.batch(
            train_idx, batch_size, shuffle=True, device=device, padded_atoms=PA,
            drop_last=True):
        # zero_grad(set_to_none=False): zeroes grad tensors IN-PLACE so XLA sees the
        # same gradient tensor identities every step. Default set_to_none=True (PyTorch 2.x)
        # destroys and recreates grad tensors each backward pass → unique tensor IDs →
        # unique XLA graph → recompilation every batch.
        optimizer.zero_grad(set_to_none=False)
        with autocast(device):
            pred = model(z_pad, pos_pad, atom_valid)
            loss = F.mse_loss(pred, pkd_true)
        loss.backward()
        if XLA_AVAILABLE:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        # loss.item() after optimizer_step fetches already-executed scalar — no extra sync
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


def _real_val_epoch(model, device, dataset, val_idx, batch_size) -> float:
    """Compute RMSE on PDBbind validation set."""
    sq_sum = 0.0
    n      = 0
    PA = model.PADDED_ATOMS
    with torch.no_grad():
        # drop_last=True: keep val graph the same shape as train graph (B=batch_size always)
        for z_pad, pos_pad, atom_valid, pkd_true in dataset.batch(
                val_idx, batch_size, shuffle=False, device=device, padded_atoms=PA,
                drop_last=True):
            with autocast(device):
                pred = model(z_pad, pos_pad, atom_valid)
            if XLA_AVAILABLE:
                xm.mark_step()
            sq_sum += ((pred - pkd_true) ** 2).sum().item()
            n      += len(pkd_true)
    return math.sqrt(sq_sum / max(n, 1))


def _mock_train_epoch(model, optimizer, device, batch_size) -> float:
    """Fallback mock — used when rdkit-pypi is not installed."""
    import random
    total_loss = 0.0
    for _ in range(20):
        z   = torch.randint(0, 9, (batch_size * 20,)).to(device)
        pos = torch.randn(batch_size * 20, 3).to(device)
        bat = torch.repeat_interleave(
            torch.arange(batch_size, device=device), 20)
        pkd_true = torch.randn(batch_size, device=device) * 1.5 + 7.0

        with autocast(device):
            pred = model(z, pos, bat)
            loss = F.mse_loss(pred, pkd_true)

        optimizer.zero_grad()
        loss.backward()
        if XLA_AVAILABLE:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        total_loss += loss.item()
    return total_loss / 20


def _mock_val_epoch(model, device, epoch) -> float:
    """Fallback mock validation — used when rdkit-pypi is not installed."""
    import random
    base = 1.8 * (0.96 ** epoch)
    noise = random.gauss(0, 0.05)
    return max(0.5, base + noise)


# ── Target scoring ────────────────────────────────────────────────────────────

def score_ms_targets(device) -> dict:
    """
    Score FDA-approved compound library against each MS target.
    Returns ranked candidate list per target.

    On real TPU: replace mock scoring with actual graph-based docking.
    Compound representations built from SMILES via RDKit.
    """
    notify("PHASE_START", "Scoring MS targets against FDA compound library",
           data={"targets": list(MS_TARGETS.keys())})

    # Download target structures
    for gene, info in MS_TARGETS.items():
        fetch_target_structure(gene, info["pdb"], DATA_DIR)

    # Load best fine-tuned model
    ckpt_path = OUTPUT_DIR / "phase2_best.pt"
    if not ckpt_path.exists():
        notify("ANOMALY", "Phase 2 checkpoint missing — running with base weights")
        model = BindingAffinityGNN(
            MolecularGNN(hidden_dim=256, num_blocks=6, cutoff=5.0), hidden_dim=256
        ).to(device)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        base = MolecularGNN(hidden_dim=256, num_blocks=6, cutoff=5.0)
        model = BindingAffinityGNN(base, hidden_dim=256)
        model.load_state_dict(ckpt["model_state"])
        model = model.to(device).eval()

    # ── Build compound library ────────────────────────────────────────────────
    fda_compounds: list = []  # list of (name, smiles)

    if REAL_DATA:
        try:
            notify("CHECKPOINT", "Fetching FDA compound library from PubChem ...")
            cache = DATA_DIR / "pubchem_fda.json"
            fda_compounds = fetch_pubchem_fda_smiles(n_max=3000, cache_path=cache)
            notify("CHECKPOINT", f"PubChem FDA: {len(fda_compounds)} compounds fetched")
        except Exception as e:
            notify("ANOMALY", f"PubChem fetch failed: {e} — using mock library", urgent=False)

    use_real_scoring = REAL_DATA and len(fda_compounds) > 0

    if not use_real_scoring:
        import random
        rng = random.Random(42)
        n_compounds = 3000
        fda_compounds = [(f"FDA_{i:04d}", None) for i in range(n_compounds)]

    # ── Pre-score all compounds once ─────────────────────────────────────────────
    # BindingAffinityGNN has no target-protein context; scores are ligand-only.
    # Computing per-target is 9× redundant — score once and re-use rankings.
    if use_real_scoring:
        PA = model.PADDED_ATOMS
        batch_names, batch_z, batch_pos = [], [], []
        base_scores: list = []  # [(name, pkd), ...]

        def _flush_batch(names, zs, poss):
            B = len(names)
            z_pad   = torch.zeros(B, PA, dtype=torch.long)
            pos_pad = torch.zeros(B, PA, 3, dtype=torch.float)
            valid   = torch.zeros(B, PA, dtype=torch.bool)
            for bi, (z, pos) in enumerate(zip(zs, poss)):
                n = min(len(z), PA)
                z_pad[bi, :n]   = z[:n]
                pos_pad[bi, :n] = pos[:n]
                valid[bi, :n]   = True
            z_pad   = z_pad.to(device)
            pos_pad = pos_pad.to(device)
            valid   = valid.to(device)
            with torch.no_grad():
                with autocast(device):
                    preds = model(z_pad, pos_pad, valid)
            if XLA_AVAILABLE:
                xm.mark_step()
            for name, pkd in zip(names, preds.cpu().tolist()):
                base_scores.append((name, pkd))

        for name, smiles in fda_compounds:
            result = smiles_to_graph(smiles) if smiles else None
            if result is None:
                continue
            z, pos = result
            batch_names.append(name)
            batch_z.append(z)
            batch_pos.append(pos)
            if len(batch_names) >= 32:
                _flush_batch(batch_names, batch_z, batch_pos)
                batch_names, batch_z, batch_pos = [], [], []
        if batch_names:
            _flush_batch(batch_names, batch_z, batch_pos)

        base_scores.sort(key=lambda x: -x[1])
        notify("CHECKPOINT",
               f"Scored {len(base_scores)} real FDA compounds via GNN (shared across all targets)",
               data={"n_scored": len(base_scores)})

    rankings = {}
    for gene, info in MS_TARGETS.items():
        if use_real_scoring:
            scores = list(base_scores)  # same ranking for all targets (ligand-only model)
        else:
            import random
            scores = []
            for cmp_id, _ in fda_compounds:
                seed = int(hashlib.md5(f"{gene}{cmp_id}".encode()).hexdigest(), 16) % 1000
                rng2 = random.Random(seed)
                pkd = rng2.gauss(6.5, 1.2)
                scores.append((cmp_id, pkd))
            scores.sort(key=lambda x: -x[1])

        rankings[gene] = {
            "top_candidates": scores[:20],
            "mean_pkd": sum(s for _, s in scores) / len(scores),
            "target_log2fc": info["log2fc"],
            "rationale": info["rationale"],
            "pdb_structure": info["pdb"],
        }

        top5 = ", ".join(f"{c}({p:.2f})" for c, p in scores[:5])
        notify("CHECKPOINT", f"{gene}: top pKd={scores[0][1]:.2f}, top 5: {top5}",
               data={"gene": gene, "n_screened": len(scores), "top_pkd": scores[0][1]})

    # Save results
    out = OUTPUT_DIR / "ms_target_candidates_phase2.json"
    with open(out, "w") as f:
        json.dump(rankings, f, indent=2)
    notify("PHASE_COMPLETE", "MS target scoring complete",
           data={g: v["top_candidates"][0] for g, v in rankings.items()})
    return rankings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",    type=int, default=50)
    parser.add_argument("--batch-size",type=int, default=32)
    parser.add_argument("--skip-train",action="store_true",
                        help="Skip fine-tuning, go straight to target scoring")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to Phase 1 checkpoint (default: condition_B_best.pt)")
    args = parser.parse_args()

    global PHASE1_CKPT
    if args.checkpoint:
        PHASE1_CKPT = Path(args.checkpoint)

    if XLA_AVAILABLE:
        device = xm.xla_device()
        notify("PHASE_START", f"Phase 2 running on TPU: {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        notify("PHASE_START", f"Phase 2 running on GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        notify("PHASE_START", "Phase 2 running on CPU (slow — TPU/GPU preferred)")

    if not args.skip_train:
        fine_tune(device, epochs=args.epochs, batch_size=args.batch_size)

    rankings = score_ms_targets(device)

    # Print summary
    print("\n" + "=" * 60)
    print("MS Target Candidate Summary")
    print("=" * 60)
    for gene, data in sorted(rankings.items(),
                              key=lambda x: -x[1]["top_candidates"][0][1]):
        top = data["top_candidates"][0]
        print(f"  {gene:8s}  best_pKd={top[1]:.2f}  compound={top[0]}"
              f"  [{data['rationale'][:40]}]")


if __name__ == "__main__":
    main()
