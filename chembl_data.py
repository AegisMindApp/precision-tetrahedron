"""
ChEMBL + PubChem real data loader for Phase 2 binding affinity fine-tuning.

Replaces mock data in phase2_pdbbind.py with real (SMILES, pKd) pairs from
ChEMBL and real FDA-approved compound library from PubChem.

Requirements: rdkit-pypi (pip install rdkit-pypi)

Data sources:
  - ChEMBL REST API (https://www.ebi.ac.uk/chembl/api/data/) — binding affinities
  - PubChem REST API — FDA-approved compound SMILES
  - Both are public, no authentication required

Graph format matches MolecularGNN input:
  z   — LongTensor of atomic numbers (H=1, C=6, N=7, O=8, ...)
  pos — FloatTensor (N, 3) of 3D coordinates from RDKit ETKDG conformer
  bat — LongTensor (N,) of graph indices (which molecule each atom belongs to)
"""

import json
import math
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import List, Tuple, Optional

import torch

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
PUBCHEM_API = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# Atomic number map — elements common in drug-like molecules
# Anything outside this set is mapped to 0 (unknown) and the atom is skipped
_ATOMIC_NUMBERS = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9,
    "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53,
}


def _rdkit_available() -> bool:
    try:
        from rdkit import Chem  # noqa: F401
        return True
    except ImportError:
        return False


def smiles_to_graph(smiles: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Convert a SMILES string to (z, pos) tensors.

    Returns None if the SMILES is invalid, 3D embedding fails, or RDKit
    is not installed.

    Uses ETKDGv3 for 3D coordinate generation (fast, reasonable geometry).
    Hydrogen atoms are added explicitly so the model sees full atomic detail.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    ret = AllChem.EmbedMolecule(mol, params)
    if ret != 0:
        # Fallback: MMFF optimisation sometimes succeeds where ETKDG fails
        ret = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if ret != 0:
            return None

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass  # geometry is still usable without optimisation

    conf = mol.GetConformer()
    atoms = mol.GetAtoms()

    z_list, pos_list = [], []
    for atom in atoms:
        sym = atom.GetSymbol()
        an = _ATOMIC_NUMBERS.get(sym, 0)
        if an == 0:
            continue  # skip rare elements
        p = conf.GetAtomPosition(atom.GetIdx())
        z_list.append(an)
        pos_list.append([p.x, p.y, p.z])

    if len(z_list) < 3:
        return None

    z   = torch.tensor(z_list, dtype=torch.long)
    pos = torch.tensor(pos_list, dtype=torch.float)
    return z, pos


def _fetch_json(url: str, retries: int = 3) -> Optional[dict]:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None


def fetch_chembl_activities(
    n_records: int = 5000,
    cache_path: Optional[Path] = None,
) -> List[Tuple[str, float]]:
    """
    Fetch (SMILES, pKd) pairs from ChEMBL.

    Uses Ki and Kd measurements (nM), converted to pKd = -log10(Kd_M).
    Filters to:
      - standard_type in (Kd, Ki)
      - standard_units = nM
      - standard_relation = '='  (exact measurements only)
      - standard_value > 0
      - molecule has canonical SMILES

    Returns list of (smiles, pkd) tuples.
    """
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            return [tuple(x) for x in json.load(f)]

    pairs: List[Tuple[str, float]] = []
    offset = 0
    page_size = 1000

    while len(pairs) < n_records:
        params = urllib.parse.urlencode({
            "format": "json",
            "limit": page_size,
            "offset": offset,
            "standard_type__in": "Kd,Ki",
            "standard_units": "nM",
            "standard_relation": "=",
            "assay_type": "B",           # binding assays only
            "pchembl_value__isnull": "false",  # pre-computed pChEMBL value exists
        })
        url = f"{CHEMBL_API}/activity.json?{params}"
        data = _fetch_json(url)
        if not data or not data.get("activities"):
            break

        for act in data["activities"]:
            smiles = act.get("canonical_smiles") or act.get("molecule_structures", {})
            if isinstance(smiles, dict):
                smiles = smiles.get("canonical_smiles", "")
            if not smiles:
                continue

            pchembl = act.get("pchembl_value")
            if pchembl is None:
                continue
            try:
                pkd = float(pchembl)
            except (TypeError, ValueError):
                continue

            if 2.0 <= pkd <= 12.0:  # sanity range
                pairs.append((smiles, pkd))

        page_info = data.get("page_meta", {})
        total = page_info.get("total_count", 0)
        offset += page_size
        if offset >= total or offset >= n_records * 2:
            break
        time.sleep(0.1)  # be polite to the API

    # Deduplicate by SMILES
    seen = set()
    deduped = []
    for s, p in pairs:
        if s not in seen:
            seen.add(s)
            deduped.append((s, p))

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(deduped, f)

    return deduped[:n_records]


def fetch_pubchem_fda_smiles(
    n_max: int = 3000,
    cache_path: Optional[Path] = None,
) -> List[Tuple[str, str]]:
    """
    Fetch (name, SMILES) for FDA-approved drugs from PubChem.

    Returns list of (drug_name, canonical_smiles) tuples.
    Uses the PubChem classification endpoint for FDA-approved drugs.
    """
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            return [tuple(x) for x in json.load(f)]

    # Fetch CIDs for FDA-approved drugs via classification
    url = f"{PUBCHEM_API}/compound/xref/DrugProductIngredient/JSON"
    data = _fetch_json(url)
    if not data:
        # Fallback: use a known list of FDA drug CIDs from a static query
        url2 = (
            f"{PUBCHEM_API}/compound/name/approved+drug/JSON"
            "?name_type=word&MaxRecords=1000"
        )
        data = _fetch_json(url2)

    cids = []
    if data:
        infos = data.get("InformationList", {}).get("Information", [])
        for info in infos:
            if "CID" in info:
                cids.append(info["CID"])

    if not cids:
        return []

    # Batch fetch SMILES (PubChem allows up to 100 CIDs per request)
    results = []
    batch_size = 100
    for i in range(0, min(len(cids), n_max), batch_size):
        batch = cids[i : i + batch_size]
        cid_str = ",".join(str(c) for c in batch)
        url = f"{PUBCHEM_API}/compound/cid/{cid_str}/property/CanonicalSMILES,IUPACName/JSON"
        data = _fetch_json(url)
        if not data:
            continue
        for prop in data.get("PropertyTable", {}).get("Properties", []):
            smiles = prop.get("CanonicalSMILES", "")
            name   = prop.get("IUPACName", f"CID_{prop.get('CID','?')}")
            if smiles:
                results.append((name, smiles))
        time.sleep(0.2)  # PubChem rate limit: 5 req/s

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(results, f)

    return results


class ChEMBLDataset:
    """
    In-memory dataset of (z, pos, pkd) for fine-tuning.

    Converts ChEMBL SMILES to graphs on construction; invalid SMILES are
    dropped silently. Progress is printed every 100 molecules.
    """

    def __init__(self, pairs: List[Tuple[str, float]], verbose: bool = True):
        self.data: List[Tuple[torch.Tensor, torch.Tensor, float]] = []

        if not _rdkit_available():
            raise ImportError(
                "RDKit is required for real data loading. "
                "Run: pip install rdkit-pypi"
            )

        n_total = len(pairs)
        for i, (smiles, pkd) in enumerate(pairs):
            result = smiles_to_graph(smiles)
            if result is not None:
                z, pos = result
                self.data.append((z, pos, pkd))
            if verbose and (i + 1) % 200 == 0:
                print(
                    f"  Graph build: {i+1}/{n_total} SMILES processed, "
                    f"{len(self.data)} valid graphs"
                )

        if verbose:
            print(f"  ChEMBLDataset: {len(self.data)}/{n_total} valid graphs")

    def __len__(self) -> int:
        return len(self.data)

    def train_val_split(self, val_frac: float = 0.2, seed: int = 42):
        """Return (train_dataset, val_dataset) as index lists."""
        import random
        rng = random.Random(seed)
        idx = list(range(len(self.data)))
        rng.shuffle(idx)
        split = int(len(idx) * (1 - val_frac))
        return idx[:split], idx[split:]

    def batch(
        self,
        indices: List[int],
        batch_size: int,
        shuffle: bool = True,
        device=None,
    ):
        """
        Yield (z, pos, batch_idx, pkd) tensors of size batch_size.
        Handles variable-size graphs by concatenating and building batch index.
        """
        import random
        if shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            z_list, pos_list, pkd_list = [], [], []
            batch_idx = []
            for graph_i, idx in enumerate(chunk):
                z, pos, pkd = self.data[idx]
                z_list.append(z)
                pos_list.append(pos)
                pkd_list.append(pkd)
                batch_idx.append(torch.full((len(z),), graph_i, dtype=torch.long))

            z_cat   = torch.cat(z_list)
            pos_cat = torch.cat(pos_list)
            bat_cat = torch.cat(batch_idx)
            pkd_t   = torch.tensor(pkd_list, dtype=torch.float)

            if device is not None:
                z_cat   = z_cat.to(device)
                pos_cat = pos_cat.to(device)
                bat_cat = bat_cat.to(device)
                pkd_t   = pkd_t.to(device)

            yield z_cat, pos_cat, bat_cat, pkd_t


def build_dataset(data_dir: Path, n_records: int = 5000) -> "ChEMBLDataset":
    """
    Download ChEMBL binding data, build graph dataset, cache to disk.

    Subsequent calls load from cache — no re-download needed.
    """
    cache = data_dir / "chembl_pairs.json"
    pairs = fetch_chembl_activities(n_records=n_records, cache_path=cache)
    if not pairs:
        raise RuntimeError(
            "ChEMBL fetch returned no data. Check connectivity to ebi.ac.uk."
        )
    return ChEMBLDataset(pairs, verbose=True)
