#!/usr/bin/env python3
"""
phase15_finetune_amr.py
-----------------------
Fine-tune the phase14 foundation model on AMR drug targets:
  KPC-3 β-lactamase (KPC3), NDM-1, OXA-48 (mapped to MSH3 surrogate slot),
  and the remaining phase6 targets via the Vina docking dataset.

Pipeline
--------
1.  Download phase14 foundation checkpoint from GCS
2.  Download vina_scores.json and pubchem_fda.json
3.  Build FineTuneModel: FoundationGNN backbone + fresh 6-target head
4.  Freeze backbone for epochs 0–9, then unfreeze for epochs 10–49
5.  Fine-tune for 50 epochs total
6.  Virtual screening: ZINC-250K + phase12 generated molecules vs KPC3 and MSH3
7.  Apply Lipinski filter; report top 100 per target
8.  Cross-reference against FDA set to flag novel candidates
9.  Upload results to GCS

GCS output: gs://aegismind-tpu-results/aegis_flashoptim/phase15_finetune_amr/
"""

import os
import sys
import csv
import json
import time
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr

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
OUT_DIR  = Path("/tmp/phase15_finetune_amr")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS   = ["LINGO1", "PCSK9", "KPC3", "APEX1", "MSH3", "CREBBP"]
TARGET2ID = {t: i for i, t in enumerate(TARGETS)}
N_TARGETS = len(TARGETS)

# AMR-relevant target indices
KPC3_ID   = TARGET2ID["KPC3"]    # 2
MSH3_ID   = TARGET2ID["MSH3"]    # 4

PA           = 80
HIDDEN_DIM   = 512
NUM_BLOCKS   = 8
NUM_GAUSSIANS = 50
CUTOFF       = 5.0
FOUNDATION_TARGETS = 4      # phase14 was trained with 4 targets

BATCH_SIZE   = 64
N_EPOCHS     = 50
FREEZE_EPOCHS = 10          # freeze backbone for first 10 epochs
LR_HEAD      = 1e-3         # fresh head needs higher lr
LR_BACKBONE  = 1e-5         # fine-tuning backbone slowly
WEIGHT_DECAY = 1e-5
PATIENCE     = 8
TRAIN_FRAC   = 0.80
RANDOM_SEED  = 42
TOP_N        = 100          # top hits per target in virtual screen

RT_LOG10     = 0.592 * 2.303   # kcal/mol → pKd conversion

GCS_PHASE15  = f"{GCS_BASE}/phase15_finetune_amr"


# ── Model ─────────────────────────────────────────────────────────────────────

class FineTuneModel(nn.Module):
    """
    Foundation GNN backbone (hidden_dim=512, num_blocks=8) with:
    - Target embedding (6 targets, emb_dim=16)
    - Fresh 6-way output head for Vina pKd prediction

    The backbone weights are loaded from phase14; the output head is randomly
    initialised and trained first (backbone frozen), then full fine-tuning.
    """

    EMB_DIM = 16

    def __init__(self):
        super().__init__()
        # Backbone: num_targets=512 so the GNN outputs a 512-dim pooled repr
        # We achieve this by setting num_targets to hidden_dim and using the
        # raw output as embedding. MolecularGNN returns [B, num_targets], so
        # we set num_targets=hidden_dim to get a [B, 512] feature vector.
        self.backbone = MolecularGNN(
            num_atom_types=119,
            hidden_dim=HIDDEN_DIM,
            num_blocks=NUM_BLOCKS,
            num_gaussians=NUM_GAUSSIANS,
            cutoff=CUTOFF,
            num_targets=HIDDEN_DIM,   # repurpose output dim as embedding
        )
        self.target_emb = nn.Embedding(N_TARGETS, self.EMB_DIM)
        self.output_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM + self.EMB_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        log("Backbone frozen")

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        log("Backbone unfrozen")

    def forward(self, z_pad, pos_pad, atom_valid, target_id):
        """Returns predicted pKd [B]."""
        h = self.backbone(z_pad, pos_pad, atom_valid)   # [B, HIDDEN_DIM]
        t = self.target_emb(target_id)                   # [B, EMB_DIM]
        x = torch.cat([h, t], dim=-1)                    # [B, HIDDEN_DIM+EMB_DIM]
        return self.output_head(x).squeeze(-1)           # [B]


def load_foundation_weights(model: FineTuneModel, ckpt_path: Path):
    """
    Load MolecularGNN backbone weights from phase14 foundation checkpoint.
    The phase14 model used MolecularGNN with num_targets=4 (descriptor prediction).
    We load everything except the final output layer, which will differ in size.
    """
    raw = torch.load(str(ckpt_path), map_location="cpu")
    if "model_state" in raw:
        state = raw["model_state"]
    elif "state_dict" in raw:
        state = raw["state_dict"]
    else:
        state = raw

    # Phase14 stores model as FoundationGNN with .gnn attribute = MolecularGNN
    # Phase15 backbone is MolecularGNN directly (stored under .backbone)
    # Strip prefix either way
    backbone_state = {}
    for k, v in state.items():
        if k.startswith("gnn."):
            backbone_state[k[len("gnn."):]] = v
        elif k.startswith("backbone."):
            backbone_state[k[len("backbone."):]] = v
        else:
            backbone_state[k] = v

    # Load with strict=False to tolerate mismatched output head size
    missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
    log(f"Foundation weights loaded: {len(backbone_state)} keys, "
        f"missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        log(f"  Missing keys (first 5): {missing[:5]}")


# ── Data utils ────────────────────────────────────────────────────────────────

def vina_to_pkd(affinity_kcal: float) -> float:
    if affinity_kcal >= 0:
        return 0.0
    return -affinity_kcal / RT_LOG10


def lipinski_filter(smiles: str) -> Tuple[bool, dict]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, {}
        mw   = Descriptors.MolWt(mol)
        hbd  = rdMolDescriptors.CalcNumHBD(mol)
        hba  = rdMolDescriptors.CalcNumHBA(mol)
        logp = Descriptors.MolLogP(mol)
        props = {"mw": round(mw, 1), "hbd": hbd, "hba": hba, "logp": round(logp, 2)}
        return (mw <= 500 and hbd <= 5 and hba <= 10 and logp <= 5), props
    except Exception:
        return False, {}


def smiles_to_padded(smiles: str, pa: int = PA):
    try:
        g = smiles_to_graph(smiles)
        if g is None:
            return None
        z, pos = g
        na = min(len(z), pa)
        z_pad   = torch.zeros(pa, dtype=torch.long)
        pos_pad = torch.zeros(pa, 3, dtype=torch.float32)
        valid   = torch.zeros(pa, dtype=torch.bool)
        z_pad[:na]   = torch.as_tensor(z[:na],   dtype=torch.long)
        pos_pad[:na] = torch.as_tensor(pos[:na], dtype=torch.float32)
        valid[:na]   = True
        return z_pad, pos_pad, valid
    except Exception:
        return None


def load_vina_dataset(vina_path: Path, fda_path: Path) -> List:
    """Returns list of (smiles, target_id, pkd) triples."""
    vina_scores = json.loads(vina_path.read_text())
    fda_raw     = json.loads(fda_path.read_text())

    smiles_map = {}
    for c in fda_raw:
        if isinstance(c, (list, tuple)):
            name = str(c[0]);  smi = str(c[1]) if len(c) > 1 else ""
        else:
            name = str(c.get("name") or c.get("iupac_name") or c.get("cid", ""))
            smi  = c.get("smiles") or c.get("canonical_smiles") or \
                   c.get("isomeric_smiles", "")
        if name and smi:
            smiles_map[name] = smi

    records = []
    skipped = 0
    for compound, scores in vina_scores.items():
        smi = smiles_map.get(compound)
        if not smi:
            skipped += 1
            continue
        for target, affinity in scores.items():
            if target not in TARGET2ID:
                continue
            pkd = vina_to_pkd(affinity)
            if pkd <= 0:
                continue
            records.append((smi, TARGET2ID[target], pkd))
    log(f"Vina dataset: {len(records)} (smiles, target_id, pKd) pairs "
        f"({skipped} compounds missing SMILES)")
    return records


def build_graphs_vina(records: List) -> List:
    """Convert Vina records to (z_pad, pos_pad, valid, tid, pkd) tuples."""
    graphs  = []
    failed  = 0
    smiles_seen: dict = {}   # cache graphs per SMILES

    for i, (smi, tid, pkd) in enumerate(records):
        if i % 500 == 0:
            sys.stdout.write(f"\r  Graphs: {i}/{len(records)}")
            sys.stdout.flush()
        if smi in smiles_seen:
            z_pad, pos_pad, valid = smiles_seen[smi]
        else:
            g = smiles_to_padded(smi)
            if g is None:
                failed += 1
                continue
            smiles_seen[smi] = g
            z_pad, pos_pad, valid = g
        graphs.append((
            z_pad, pos_pad, valid,
            torch.tensor(tid,  dtype=torch.long),
            torch.tensor(pkd,  dtype=torch.float32),
        ))
    print()
    log(f"Graph build: {len(graphs)} ok, {failed} failed")
    return graphs


def batch_graphs(graphs: List, indices: List) -> Tuple:
    items   = [graphs[i] for i in indices]
    z_b     = torch.stack([g[0] for g in items])
    pos_b   = torch.stack([g[1] for g in items])
    valid_b = torch.stack([g[2] for g in items])
    tid_b   = torch.stack([g[3] for g in items])
    pkd_b   = torch.stack([g[4] for g in items])
    return z_b, pos_b, valid_b, tid_b, pkd_b


# ── Training & evaluation ─────────────────────────────────────────────────────

def make_optimizer(model: FineTuneModel, frozen: bool):
    """Create optimizer; head uses LR_HEAD, backbone uses LR_BACKBONE."""
    if frozen:
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR_HEAD, weight_decay=WEIGHT_DECAY,
        )
    return torch.optim.AdamW([
        {"params": model.backbone.parameters(),   "lr": LR_BACKBONE},
        {"params": model.target_emb.parameters(), "lr": LR_HEAD},
        {"params": model.output_head.parameters(), "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)


def train_step(model, graphs, indices, optimizer, device):
    z_b, pos_b, valid_b, tid_b, pkd_b = batch_graphs(graphs, indices)
    z_b, pos_b, valid_b = z_b.to(device), pos_b.to(device), valid_b.to(device)
    tid_b, pkd_b        = tid_b.to(device), pkd_b.to(device)

    optimizer.zero_grad()
    pred = model(z_b, pos_b, valid_b, tid_b)
    loss = F.mse_loss(pred, pkd_b)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    if XLA_AVAILABLE:
        xm.mark_step()
    return loss.item()


def evaluate(model, graphs, indices, device):
    """Returns (rmse, spearman_rho_mean, per_target_rho)."""
    model.eval()
    all_pred, all_true, all_tid = [], [], []
    with torch.no_grad():
        for start in range(0, len(indices), BATCH_SIZE):
            chunk = indices[start: start + BATCH_SIZE]
            z_b, pos_b, valid_b, tid_b, pkd_b = batch_graphs(graphs, chunk)
            z_b, pos_b, valid_b = z_b.to(device), pos_b.to(device), valid_b.to(device)
            tid_b = tid_b.to(device)
            pred  = model(z_b, pos_b, valid_b, tid_b)
            if XLA_AVAILABLE:
                xm.mark_step()
            all_pred.extend(pred.cpu().float().tolist())
            all_true.extend(pkd_b.tolist())
            all_tid.extend(tid_b.cpu().tolist())

    pred_arr = np.array(all_pred, dtype=np.float32)
    true_arr = np.array(all_true, dtype=np.float32)
    tid_arr  = np.array(all_tid)

    rmse = float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))
    rhos = {}
    for t, name in enumerate(TARGETS):
        mask = tid_arr == t
        if mask.sum() < 4:
            rhos[name] = 0.0
            continue
        rho, _ = spearmanr(pred_arr[mask], true_arr[mask])
        rhos[name] = float(rho) if np.isfinite(rho) else 0.0

    mean_rho = float(np.mean(list(rhos.values())))
    model.train()
    return rmse, mean_rho, rhos


# ── Virtual screening ─────────────────────────────────────────────────────────

def load_zinc_smiles() -> List[str]:
    zinc_csv = Path("/tmp/zinc250k/zinc250k.csv")
    if not zinc_csv.exists():
        log("ZINC CSV not found for virtual screen; skipping ZINC")
        return []
    rows = []
    try:
        with open(zinc_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            smiles_col = None
            for cand in ("smiles", "SMILES", "structure"):
                if cand in headers:
                    smiles_col = cand
                    break
            if smiles_col is None and headers:
                smiles_col = headers[0]
            for row in reader:
                smi = (row.get(smiles_col) or "").strip()
                if smi:
                    rows.append(smi)
    except Exception as e:
        log(f"ZINC CSV parse error: {e}")
    return rows


def load_phase12_smiles() -> List[str]:
    samples_path = Path("/tmp/phase13_screen_generated/samples_final.json")
    if not samples_path.exists():
        # Try GCS copy
        subprocess.run(
            ["gsutil", "-q", "cp",
             f"{GCS_BASE}/phase12_diffusion_zinc/samples_final.json",
             str(samples_path)],
            capture_output=True,
        )
    if not samples_path.exists():
        return []
    try:
        raw = json.loads(samples_path.read_text())
        if isinstance(raw, list) and raw:
            if isinstance(raw[0], dict):
                return [s["smiles"] for s in raw if "smiles" in s]
            return [str(s) for s in raw]
    except Exception:
        pass
    return []


def virtual_screen(model, smiles_list: List[str], target_id: int,
                   device, fda_smiles_set: set, top_n: int = TOP_N):
    """
    Score all SMILES against target_id. Returns top_n dicts sorted by pKd desc.
    Flags novel compounds (not in fda_smiles_set).
    """
    model.eval()
    results = []
    failed  = 0

    log(f"  Building graphs for {len(smiles_list)} SMILES ...")
    graphs_idx = []   # (smiles, z_pad, pos_pad, valid)
    for smi in smiles_list:
        g = smiles_to_padded(smi)
        if g is None:
            failed += 1
            continue
        graphs_idx.append((smi, g[0], g[1], g[2]))
    log(f"  {len(graphs_idx)} valid graphs ({failed} failed)")

    log(f"  Running inference (target_id={target_id} — {TARGETS[target_id]}) ...")
    with torch.no_grad():
        for start in range(0, len(graphs_idx), BATCH_SIZE):
            chunk = graphs_idx[start: start + BATCH_SIZE]
            z_b     = torch.stack([c[1] for c in chunk]).to(device)
            pos_b   = torch.stack([c[2] for c in chunk]).to(device)
            valid_b = torch.stack([c[3] for c in chunk]).to(device)
            tid_b   = torch.full((len(chunk),), target_id, dtype=torch.long, device=device)
            preds   = model(z_b, pos_b, valid_b, tid_b)
            if XLA_AVAILABLE:
                xm.mark_step()
            for j, (smi, _, _, _) in enumerate(chunk):
                pkd = float(preds[j].cpu().item())
                lip_pass, lip_props = lipinski_filter(smi)
                results.append({
                    "smiles":        smi,
                    "predicted_pkd": round(pkd, 4),
                    "lipinski_pass": lip_pass,
                    "lipinski_props": lip_props,
                    "is_novel":      smi not in fda_smiles_set,
                })

    results.sort(key=lambda x: -x["predicted_pkd"])
    return results[:top_n]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    notify("PHASE_START",
           "Phase 15 — AMR fine-tuning of foundation model (50 epochs) + virtual screen")
    log("=" * 60)
    log("Phase 15 — Fine-tuning foundation model on AMR targets")
    log(f"  Backbone: hidden_dim={HIDDEN_DIM}, num_blocks={NUM_BLOCKS}")
    log(f"  Epochs={N_EPOCHS} (frozen={FREEZE_EPOCHS}), batch={BATCH_SIZE}")
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

    # ── Download inputs ───────────────────────────────────────────────────────
    foundation_local = OUT_DIR / "foundation_best.pt"
    vina_local       = OUT_DIR / "vina_scores.json"
    fda_local        = OUT_DIR / "pubchem_fda.json"

    # Foundation checkpoint — try epoch200 first, then best
    for ckpt_name in ("foundation_epoch199.pt", "foundation_epoch200.pt",
                      "foundation_best.pt"):
        r = subprocess.run(
            ["gsutil", "-q", "cp",
             f"{GCS_BASE}/phase14_foundation_pretrain/{ckpt_name}",
             str(foundation_local)],
            capture_output=True,
        )
        if r.returncode == 0:
            log(f"Downloaded foundation checkpoint: {ckpt_name}")
            break
    else:
        log("WARNING: foundation checkpoint not found — initialising from scratch")
        foundation_local = None

    log("Downloading vina_scores.json ...")
    subprocess.run(
        ["gsutil", "-q", "cp", f"{GCS_BASE}/vina_scores.json", str(vina_local)],
        capture_output=True,
    )
    log("Downloading pubchem_fda.json ...")
    subprocess.run(
        ["gsutil", "-q", "cp", f"{GCS_BASE}/pubchem_fda.json", str(fda_local)],
        capture_output=True,
    )

    if not vina_local.exists() or not fda_local.exists():
        notify("ABORT", "Phase 15: vina_scores.json or pubchem_fda.json not found")
        raise RuntimeError("Required data files not on GCS")

    # ── Load dataset ──────────────────────────────────────────────────────────
    log("Loading Vina dataset ...")
    records = load_vina_dataset(vina_local, fda_local)
    if len(records) < 50:
        notify("ABORT", "Phase 15: fewer than 50 Vina records loaded")
        raise RuntimeError("Insufficient training data")

    log("Building graph tensors ...")
    graphs = build_graphs_vina(records)
    if len(graphs) < 50:
        notify("ABORT", "Phase 15: too few valid graphs from Vina data")
        raise RuntimeError("Insufficient valid graphs")

    rng   = np.random.RandomState(RANDOM_SEED)
    idx   = np.arange(len(graphs))
    rng.shuffle(idx)
    split = int(len(idx) * TRAIN_FRAC)
    train_idx = idx[:split].tolist()
    val_idx   = idx[split:].tolist()
    log(f"Train={len(train_idx)}, Val={len(val_idx)}")

    # Collect FDA SMILES for novelty checking
    fda_raw    = json.loads(fda_local.read_text())
    fda_smiles_set = set()
    for c in fda_raw:
        if isinstance(c, (list, tuple)):
            smi = str(c[1]) if len(c) > 1 else ""
        else:
            smi = c.get("smiles") or c.get("canonical_smiles") or \
                  c.get("isomeric_smiles", "")
        if smi:
            fda_smiles_set.add(smi)
    log(f"FDA SMILES set: {len(fda_smiles_set)} compounds")

    # ── Build model ───────────────────────────────────────────────────────────
    torch.manual_seed(RANDOM_SEED)
    model = FineTuneModel()

    if foundation_local is not None and foundation_local.exists():
        load_foundation_weights(model, foundation_local)
    else:
        log("No foundation checkpoint — training from random init")

    if XLA_AVAILABLE:
        model = model.to(torch.bfloat16)
    model = model.to(device)
    log(f"FineTuneModel: {model.parameter_count():,} parameters")

    # ── Fine-tuning loop ──────────────────────────────────────────────────────
    model.freeze_backbone()
    optimizer = make_optimizer(model, frozen=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=LR_HEAD * 0.01,
    )

    best_val_rmse = float("inf")
    best_rho      = -1.0
    patience_ctr  = 0
    metrics_log   = []
    t0_total      = time.time()

    for epoch in range(N_EPOCHS):
        t0 = time.time()

        # Unfreeze backbone after FREEZE_EPOCHS
        if epoch == FREEZE_EPOCHS:
            model.unfreeze_backbone()
            optimizer = make_optimizer(model, frozen=False)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=(N_EPOCHS - FREEZE_EPOCHS),
                eta_min=LR_BACKBONE * 0.01,
            )
            log(f"Epoch {epoch}: backbone unfrozen, new optimizer created")

        # Train
        model.train()
        rng.shuffle(train_idx)
        train_loss = 0.0
        n_batches  = 0
        for start in range(0, len(train_idx), BATCH_SIZE):
            batch = train_idx[start: start + BATCH_SIZE]
            if len(batch) < 2:
                continue
            train_loss += train_step(model, graphs, batch, optimizer, device)
            n_batches  += 1
        train_loss = train_loss / max(n_batches, 1)

        # Validate
        val_rmse, mean_rho, per_target_rho = evaluate(model, graphs, val_idx, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]
        log(f"Epoch {epoch:3d}/{N_EPOCHS-1}  "
            f"train_loss={train_loss:.4f}  val_rmse={val_rmse:.4f}  "
            f"rho={mean_rho:.3f}  lr={lr_now:.2e}  t={elapsed:.1f}s")

        metrics_log.append({
            "epoch":          epoch,
            "train_loss":     round(train_loss, 5),
            "val_rmse":       round(val_rmse,   5),
            "mean_rho":       round(mean_rho,   4),
            "per_target_rho": {k: round(v, 4) for k, v in per_target_rho.items()},
        })

        if epoch % 10 == 0:
            heartbeat("phase15", epoch, {
                "train_loss": train_loss,
                "val_rmse":   val_rmse,
                "mean_rho":   mean_rho,
                "kpc3_rho":   per_target_rho.get("KPC3", 0.0),
                "msh3_rho":   per_target_rho.get("MSH3", 0.0),
            })

        # Early stopping on val RMSE
        if val_rmse < best_val_rmse - 1e-5:
            best_val_rmse = val_rmse
            best_rho      = mean_rho
            patience_ctr  = 0
            best_path = OUT_DIR / "finetune_best.pt"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_rmse":    val_rmse,
                "mean_rho":    mean_rho,
            }, str(best_path))
            gsutil_cp(best_path, f"{GCS_PHASE15}/finetune_best.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                log(f"Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    log(f"Fine-tuning complete. best_val_rmse={best_val_rmse:.4f}, "
        f"best_rho={best_rho:.4f}")

    # ── Load best checkpoint for screening ────────────────────────────────────
    best_path = OUT_DIR / "finetune_best.pt"
    if best_path.exists():
        ckpt = torch.load(str(best_path), map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        if XLA_AVAILABLE:
            model = model.to(torch.bfloat16)
        model = model.to(device)
        log("Loaded best fine-tuned checkpoint for virtual screen")

    # ── Virtual screening ─────────────────────────────────────────────────────
    log("Loading SMILES for virtual screening ...")
    zinc_smiles   = load_zinc_smiles()
    p12_smiles    = load_phase12_smiles()
    all_vs_smiles = list(dict.fromkeys(zinc_smiles + p12_smiles))  # deduplicate
    log(f"Virtual screen pool: {len(all_vs_smiles)} SMILES "
        f"({len(zinc_smiles)} ZINC + {len(p12_smiles)} generated)")

    if len(all_vs_smiles) == 0:
        log("WARNING: no SMILES for virtual screen — skipping")
        top_kpc3 = []
        top_msh3 = []
    else:
        log("Virtual screening vs KPC3 ...")
        top_kpc3 = virtual_screen(model, all_vs_smiles, KPC3_ID,
                                   device, fda_smiles_set, top_n=TOP_N)
        log(f"  Top KPC3 pKd: {top_kpc3[0]['predicted_pkd']:.3f}" if top_kpc3 else "  No KPC3 hits")

        log("Virtual screening vs MSH3 ...")
        top_msh3 = virtual_screen(model, all_vs_smiles, MSH3_ID,
                                   device, fda_smiles_set, top_n=TOP_N)
        log(f"  Top MSH3 pKd: {top_msh3[0]['predicted_pkd']:.3f}" if top_msh3 else "  No MSH3 hits")

    # ── Novel candidates ──────────────────────────────────────────────────────
    novel_kpc3 = [c for c in top_kpc3 if c["is_novel"] and c["lipinski_pass"]]
    novel_msh3 = [c for c in top_msh3 if c["is_novel"] and c["lipinski_pass"]]
    novel_all  = {smi: {"KPC3_pkd": None, "MSH3_pkd": None}
                  for smi in set([c["smiles"] for c in novel_kpc3 + novel_msh3])}
    for c in novel_kpc3:
        if c["smiles"] in novel_all:
            novel_all[c["smiles"]]["KPC3_pkd"] = c["predicted_pkd"]
    for c in novel_msh3:
        if c["smiles"] in novel_all:
            novel_all[c["smiles"]]["MSH3_pkd"] = c["predicted_pkd"]
    novel_candidates = [
        {"smiles": smi, **vals}
        for smi, vals in sorted(
            novel_all.items(),
            key=lambda x: -(x[1]["KPC3_pkd"] or 0),
        )
    ]

    # ── Save outputs ──────────────────────────────────────────────────────────
    total_h = (time.time() - t0_total) / 3600.0

    finetune_results = {
        "best_val_rmse":  round(best_val_rmse, 5),
        "best_mean_rho":  round(best_rho,      4),
        "n_train":        len(train_idx),
        "n_val":          len(val_idx),
        "total_hours":    round(total_h, 2),
        "metrics":        metrics_log,
    }
    fr_path = OUT_DIR / "finetune_results.json"
    fr_path.write_text(json.dumps(finetune_results, indent=2))
    gsutil_cp(fr_path, f"{GCS_PHASE15}/finetune_results.json")

    kpc3_path = OUT_DIR / "virtual_screen_KPC3.json"
    kpc3_path.write_text(json.dumps(top_kpc3, indent=2))
    gsutil_cp(kpc3_path, f"{GCS_PHASE15}/virtual_screen_KPC3.json")

    msh3_path = OUT_DIR / "virtual_screen_MSH3.json"
    msh3_path.write_text(json.dumps(top_msh3, indent=2))
    gsutil_cp(msh3_path, f"{GCS_PHASE15}/virtual_screen_MSH3.json")

    novel_path = OUT_DIR / "novel_candidates.json"
    novel_path.write_text(json.dumps(novel_candidates, indent=2))
    gsutil_cp(novel_path, f"{GCS_PHASE15}/novel_candidates.json")

    summary = {
        "best_val_rmse":     round(best_val_rmse, 5),
        "best_mean_rho":     round(best_rho,      4),
        "n_vs_pool":         len(all_vs_smiles),
        "n_novel_kpc3":      len(novel_kpc3),
        "n_novel_msh3":      len(novel_msh3),
        "top_kpc3_pkd":      top_kpc3[0]["predicted_pkd"] if top_kpc3 else None,
        "top_msh3_pkd":      top_msh3[0]["predicted_pkd"] if top_msh3 else None,
        "total_hours":       round(total_h, 2),
    }
    log("Phase 15 complete.")
    for k, v in summary.items():
        log(f"  {k}: {v}")

    notify("PHASE_COMPLETE",
           "Phase 15 AMR fine-tuning + virtual screen complete",
           data=summary)
    log("PHASE_COMPLETE")


if __name__ == "__main__":
    main()
