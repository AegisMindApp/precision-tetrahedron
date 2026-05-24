"""
QM9 dataset loading and preprocessing for TPU/XLA.

Loads QM9 via PyTorch Geometric, builds radius graphs within cutoff,
then converts to padded fixed-size tensors suitable for XLA's static
computation graph requirement.

Target: property index 4 = HOMO-LUMO gap (eV) — chosen because it is
the most commonly benchmarked QM9 target in molecular ML literature,
giving us published baselines to validate against at Checkpoint A.
"""

import os
import torch
import numpy as np
from typing import Tuple, List, Dict
from torch.utils.data import Dataset, DataLoader


# QM9 target index and published SchNet MAE for sanity check
QM9_TARGET_IDX = 4       # HOMO-LUMO gap, eV
QM9_PUBLISHED_MAE = 0.066  # SchNet published MAE on this target (eV)
CHECKPOINT_A_THRESHOLD = 0.5   # Abort if FP32 MAE exceeds this (day-7 gate)

CUTOFF = 5.0             # Angstroms — edges only within this radius
MAX_ATOMS = 29           # QM9 max atoms per molecule
MAX_EDGES = 400          # Conservative upper bound for padded edge tensor


def _build_radius_graph(pos: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Build edge_index for all pairs within cutoff. Pure PyTorch, XLA-safe."""
    n = pos.shape[0]
    # All pairwise distances
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)   # [N, N, 3]
    dist = diff.norm(dim=-1)                      # [N, N]
    # Self-loops excluded; within cutoff
    mask = (dist < cutoff) & (dist > 0)
    src, dst = mask.nonzero(as_tuple=True)
    return torch.stack([src, dst], dim=0)         # [2, E]


class QM9MoleculeDataset(Dataset):
    """
    Preprocessed QM9 dataset returning fixed-size padded tensors.
    Fixed sizes allow XLA to compile a single static computation graph.

    Each item:
        z:          [MAX_ATOMS]       atom type indices (0 = padding)
        pos:        [MAX_ATOMS, 3]    3D positions (padded with zeros)
        edge_src:   [MAX_EDGES]       source node indices (padded with -1)
        edge_dst:   [MAX_EDGES]       destination node indices (padded with -1)
        num_atoms:  int               actual atom count (for masking)
        num_edges:  int               actual edge count
        target:     float             HOMO-LUMO gap (eV), mean/std normalised
    """

    def __init__(self, root: str, split: str = 'train', seed: int = 42):
        self.split = split
        self.data = self._load(root, seed)

    def _load(self, root: str, seed: int) -> List[Dict]:
        try:
            from torch_geometric.datasets import QM9
        except ImportError:
            raise ImportError("pip install torch_geometric required for data loading")

        dataset = QM9(root=root)
        N = len(dataset)

        rng = np.random.default_rng(seed)
        idx = rng.permutation(N)
        train_end = int(0.8 * N)
        val_end   = int(0.9 * N)

        splits = {
            'train': idx[:train_end],
            'val':   idx[train_end:val_end],
            'test':  idx[val_end:],
        }
        chosen = splits[self.split]

        # Compute normalisation stats on training set
        train_targets = [dataset[int(i)].y[0, QM9_TARGET_IDX].item()
                         for i in splits['train']]
        self.mean = float(np.mean(train_targets))
        self.std  = float(np.std(train_targets))

        records = []
        for i in chosen:
            mol = dataset[int(i)]
            pos = mol.pos                        # [N_atoms, 3]
            # atom type: z is atomic number; map to 0-based index
            z_raw = mol.z                        # [N_atoms] atomic numbers
            z = self._map_atom_types(z_raw)      # [N_atoms] 0-based
            target_raw = mol.y[0, QM9_TARGET_IDX].item()
            target = (target_raw - self.mean) / (self.std + 1e-8)

            edge_index = _build_radius_graph(pos, CUTOFF)
            n_atoms = z.shape[0]
            n_edges = edge_index.shape[1]

            if n_edges > MAX_EDGES:
                # Trim to MAX_EDGES (very rare for small molecules)
                edge_index = edge_index[:, :MAX_EDGES]
                n_edges = MAX_EDGES

            # Pad to fixed sizes
            z_pad = torch.zeros(MAX_ATOMS, dtype=torch.long)
            z_pad[:n_atoms] = z

            pos_pad = torch.zeros(MAX_ATOMS, 3, dtype=torch.float32)
            pos_pad[:n_atoms] = pos

            e_src = torch.full((MAX_EDGES,), -1, dtype=torch.long)
            e_dst = torch.full((MAX_EDGES,), -1, dtype=torch.long)
            e_src[:n_edges] = edge_index[0]
            e_dst[:n_edges] = edge_index[1]

            # Precompute assignment matrix [MAX_EDGES, MAX_ATOMS] on CPU.
            # assign_mat[e, a] = 1 if edge e's destination atom is a, else 0.
            # Passing this as data (not computed in model forward) eliminates the
            # arange+comparison operation from the XLA compiled graph, reducing HLO
            # instruction count from ~17M → ~hundreds of thousands.
            assign_mat = torch.zeros(MAX_EDGES, MAX_ATOMS, dtype=torch.float32)
            if n_edges > 0:
                dst_indices = edge_index[1, :n_edges]          # [n_edges] — dst atom idx
                edge_range  = torch.arange(n_edges)
                assign_mat[edge_range, dst_indices] = 1.0

            records.append({
                'z':          z_pad,
                'pos':        pos_pad,
                'edge_src':   e_src,
                'edge_dst':   e_dst,
                'assign_mat': assign_mat,   # [MAX_EDGES, MAX_ATOMS]
                'num_atoms':  n_atoms,
                'num_edges':  n_edges,
                'target':     torch.tensor(target, dtype=torch.float32),
                'target_raw': target_raw,
            })
        return records

    @staticmethod
    def _map_atom_types(z: torch.Tensor) -> torch.Tensor:
        """Map atomic numbers to 0-based indices. QM9: H=1,C=6,N=7,O=8,F=9."""
        mapping = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 6, 35: 7, 53: 8}
        return torch.tensor([mapping.get(int(a), 8) for a in z], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]

    def denormalise(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std + self.mean


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train/val/test DataLoaders with padded fixed-size tensors."""
    print(f"Loading QM9 dataset from {data_dir} ...")
    train_ds = QM9MoleculeDataset(data_dir, 'train', seed)
    val_ds   = QM9MoleculeDataset(data_dir, 'val',   seed)
    test_ds  = QM9MoleculeDataset(data_dir, 'test',  seed)

    # Store normalisation params on val/test so we can denormalise predictions
    val_ds.mean  = train_ds.mean;  val_ds.std  = train_ds.std
    test_ds.mean = train_ds.mean;  test_ds.std = train_ds.std

    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    print(f"  Target mean={train_ds.mean:.4f} eV, std={train_ds.std:.4f} eV")

    def collate(batch):
        return {
            'z':          torch.stack([b['z']          for b in batch]),
            'pos':        torch.stack([b['pos']         for b in batch]),
            'edge_src':   torch.stack([b['edge_src']    for b in batch]),
            'edge_dst':   torch.stack([b['edge_dst']    for b in batch]),
            'assign_mat': torch.stack([b['assign_mat']  for b in batch]),  # [B, MAX_EDGES, MAX_ATOMS]
            'num_atoms':  torch.tensor([b['num_atoms']  for b in batch]),
            'num_edges':  torch.tensor([b['num_edges']  for b in batch]),
            'target':     torch.stack([b['target']      for b in batch]),
        }

    kwargs = dict(batch_size=batch_size, collate_fn=collate,
                  num_workers=0, pin_memory=False)
    # drop_last=True on train: eliminates partial last batch so XLA only
    # ever sees one fixed shape (B*MAX_ATOMS, B*MAX_EDGES) — prevents a
    # second full recompilation for the tail batch.
    return (
        DataLoader(train_ds, shuffle=True,  drop_last=True,  **kwargs),
        DataLoader(val_ds,   shuffle=False, drop_last=False, **kwargs),
        DataLoader(test_ds,  shuffle=False, drop_last=False, **kwargs),
    )


def batch_to_graph(batch: Dict, device: torch.device):
    """
    Convert a padded batch to per-molecule tensors for MolecularGNN.

    Returns [B, MAX_ATOMS/MAX_EDGES] shaped tensors with LOCAL per-molecule
    edge indices (0..MAX_ATOMS-1). The model uses torch.bmm for aggregation
    instead of scatter_add — bmm is a native TPU primitive that XLA compiles
    in seconds rather than hours.

    The assign_mat is precomputed on CPU in the Dataset, so the edge→atom
    assignment never appears as a dynamic comparison inside the XLA compiled
    graph. This reduces HLO instruction count from ~17M to ~hundreds of
    thousands, making first-batch compilation feasible.

    Returns: z, pos, edge_src, edge_dst, assign_mat, B, edge_valid, atom_valid
      z:          [B, MAX_ATOMS]              atom type indices
      pos:        [B, MAX_ATOMS, 3]           3D positions
      edge_src:   [B, MAX_EDGES]              per-molecule source atom index
      edge_dst:   [B, MAX_EDGES]              per-molecule destination atom index
      assign_mat: [B, MAX_EDGES, MAX_ATOMS]   1 where edge dst == atom, else 0
      edge_valid: [B, MAX_EDGES]              True = real edge (not padding)
      atom_valid: [B, MAX_ATOMS]              True = real atom (not padding)
    """
    B         = batch['z'].shape[0]
    MAX_ATOMS = batch['z'].shape[1]         # 29  — fixed
    MAX_EDGES = batch['edge_src'].shape[1]  # 400 — fixed

    # Atom validity: [B, MAX_ATOMS]
    atom_idx   = torch.arange(MAX_ATOMS).unsqueeze(0).expand(B, MAX_ATOMS)
    atom_valid = (atom_idx < batch['num_atoms'].unsqueeze(1))  # [B, MAX_ATOMS]

    # Edge validity: [B, MAX_EDGES]  (stored as -1 for padding)
    edge_valid = (batch['edge_src'] >= 0)  # [B, MAX_EDGES]

    # Local per-molecule edge indices — clamp -1 padding to 0 (masked out anyway)
    edge_src = batch['edge_src'].clamp(min=0)  # [B, MAX_EDGES]
    edge_dst = batch['edge_dst'].clamp(min=0)  # [B, MAX_EDGES]

    return (
        batch['z'].to(device),              # [B, MAX_ATOMS]
        batch['pos'].to(device),             # [B, MAX_ATOMS, 3]
        edge_src.to(device),                 # [B, MAX_EDGES]
        edge_dst.to(device),                 # [B, MAX_EDGES]
        batch['assign_mat'].to(device),      # [B, MAX_EDGES, MAX_ATOMS]
        B,
        edge_valid.to(device),               # [B, MAX_EDGES]
        atom_valid.to(device),               # [B, MAX_ATOMS]
    )
