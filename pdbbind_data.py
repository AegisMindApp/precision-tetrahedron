"""
PDBbind v2020 refined-set data loader — no RDKit required.

Parses:
  - INDEX_refined_data.2020 → -logKd/Ki (pKd/pKi) values
  - {pdb}_ligand.sdf        → atom types + 3D Cartesian positions

Returns batches compatible with BindingAffinityGNN.forward(z, pos, batch).
"""

import os
import tarfile
import random
import subprocess
import torch
from pathlib import Path

# Atom type map: element symbol → index  (matches QM9/model.py convention)
ATOM_TYPES = {
    'H': 0, 'C': 1, 'N': 2, 'O': 3,
    'F': 4, 'S': 5, 'Cl': 6, 'Br': 7, 'I': 8,
}
_UNKNOWN_IDX = 1  # map rare elements to Carbon


def _parse_index(index_path: Path) -> dict:
    """Parse INDEX_refined_data.2020 → {pdb_code (lower): float pkd}."""
    pkd_map = {}
    with open(index_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            pdb = parts[0].lower()
            try:
                pkd_map[pdb] = float(parts[3])   # column 4 = -logKd/Ki
            except ValueError:
                continue
    return pkd_map


def _parse_sdf(sdf_path: Path):
    """
    Minimal V2000 SDF parser.
    Returns (atom_type_indices: list[int], positions: list[list[float]])
    or (None, None) on any parse failure.
    """
    try:
        with open(sdf_path, encoding='utf-8', errors='replace') as fh:
            lines = fh.readlines()
    except IOError:
        return None, None

    if len(lines) < 5:
        return None, None

    counts = lines[3]
    try:
        n_atoms = int(counts[:3].strip())
    except ValueError:
        return None, None

    if n_atoms < 1 or 4 + n_atoms > len(lines):
        return None, None

    atom_types, positions = [], []
    for line in lines[4: 4 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            return None, None
        try:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            return None, None
        elem = parts[3].strip().capitalize()
        atom_types.append(ATOM_TYPES.get(elem, _UNKNOWN_IDX))
        positions.append([x, y, z])

    return atom_types, positions


class PDBbindDataset:
    """
    In-memory dataset built from PDBbind v2020 refined set.

    Each entry: (pdb_code, pkd, z_tensor [N], pos_tensor [N,3])
    """

    def __init__(self, data_dir: Path, tar_path: Path, index_path: Path,
                 max_atoms: int = 150):
        self.data_dir  = Path(data_dir)
        self.tar_path  = Path(tar_path)
        self.index_path = Path(index_path)
        self.max_atoms = max_atoms
        self.entries: list = []
        self._load()

    def _load(self):
        pkd_map = _parse_index(self.index_path)

        extract_dir = self.data_dir / "refined-set"
        if not extract_dir.exists():
            print(f"[PDBbind] Extracting {self.tar_path} ...", flush=True)
            with tarfile.open(self.tar_path) as tf:
                tf.extractall(self.data_dir)

        n_ok = n_skip = 0
        for pdb, pkd in pkd_map.items():
            sdf = extract_dir / pdb / f"{pdb}_ligand.sdf"
            if not sdf.exists():
                n_skip += 1
                continue
            at, pos = _parse_sdf(sdf)
            if at is None or len(at) < 2 or len(at) > self.max_atoms:
                n_skip += 1
                continue
            z   = torch.tensor(at,  dtype=torch.long)
            xyz = torch.tensor(pos, dtype=torch.float32)
            self.entries.append((pdb, pkd, z, xyz))
            n_ok += 1

        print(f"[PDBbind] Loaded {n_ok} complexes, skipped {n_skip}", flush=True)

    def __len__(self):
        return len(self.entries)

    def train_val_split(self, val_frac: float = 0.1, seed: int = 42):
        idx = list(range(len(self.entries)))
        rng = random.Random(seed)
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_frac))
        return idx[n_val:], idx[:n_val]   # (train_indices, val_indices)

    def batch(self, indices, batch_size: int, shuffle: bool = False, device=None,
              padded_atoms: int = 80, drop_last: bool = False):
        """
        Yield (z_pad, pos_pad, atom_valid, pkd_targets) tuples.

        All tensors are pre-padded to fixed shapes on CPU before being moved to
        device — avoids .nonzero() / dynamic indexing inside the XLA trace, which
        would force 32 graph compilations per forward pass.

          z_pad      [B, padded_atoms]      long
          pos_pad    [B, padded_atoms, 3]   float32
          atom_valid [B, padded_atoms]      bool
          pkd_t      [B]                    float32
        """
        if shuffle:
            indices = list(indices)
            random.shuffle(indices)

        PA = padded_atoms
        for start in range(0, len(indices), batch_size):
            chunk = indices[start: start + batch_size]
            if drop_last and len(chunk) < batch_size:
                break
            B = len(chunk)
            z_pad      = torch.zeros(B, PA, dtype=torch.long)
            pos_pad    = torch.zeros(B, PA, 3, dtype=torch.float32)
            atom_valid = torch.zeros(B, PA, dtype=torch.bool)
            pkd_list   = []

            for g_i, idx in enumerate(chunk):
                _, pkd, z, pos = self.entries[idx]
                n = min(len(z), PA)
                z_pad[g_i, :n]      = z[:n]
                pos_pad[g_i, :n, :] = pos[:n]
                atom_valid[g_i, :n] = True
                pkd_list.append(float(pkd))

            pkd_t = torch.tensor(pkd_list, dtype=torch.float32)

            if device is not None:
                z_pad      = z_pad.to(device)
                pos_pad    = pos_pad.to(device)
                atom_valid = atom_valid.to(device)
                pkd_t      = pkd_t.to(device)

            yield z_pad, pos_pad, atom_valid, pkd_t


def build_pdbbind_dataset(
    data_dir: Path,
    gcs_bucket: str = "aegismind-tpu-results",
) -> PDBbindDataset:
    """
    Build PDBbindDataset, downloading from GCS if local files are absent.

    Expects on GCS:
      gs://{gcs_bucket}/phase2_data/PDBbind_v2020_index.tar.gz
      gs://{gcs_bucket}/phase2_data/PDBbind_v2020_refined.tar.gz
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Normalise bucket: strip any gs:// prefix so we can always prepend it once
    bucket = gcs_bucket.lstrip("gs://").rstrip("/")

    # ── Index ──────────────────────────────────────────────────────────────────
    index_path = data_dir / "index" / "INDEX_refined_data.2020"
    if not index_path.exists():
        idx_tar = data_dir / "PDBbind_v2020_index.tar.gz"
        if not idx_tar.exists():
            gcs = f"gs://{bucket}/phase2_data/PDBbind_v2020_index.tar.gz"
            print(f"[PDBbind] Downloading index from {gcs} ...", flush=True)
            subprocess.run(["gsutil", "cp", gcs, str(idx_tar)], check=True)
        print("[PDBbind] Extracting index ...", flush=True)
        with tarfile.open(idx_tar) as tf:
            tf.extractall(data_dir)

    # ── Refined set structures ──────────────────────────────────────────────────
    tar_path = data_dir / "PDBbind_v2020_refined.tar.gz"
    if not tar_path.exists():
        gcs = f"gs://{bucket}/phase2_data/PDBbind_v2020_refined.tar.gz"
        print(f"[PDBbind] Downloading refined set from {gcs} (~658 MB) ...", flush=True)
        subprocess.run(["gsutil", "cp", gcs, str(tar_path)], check=True)

    return PDBbindDataset(data_dir, tar_path, index_path)
