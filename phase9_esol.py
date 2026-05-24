#!/usr/bin/env python3
"""
phase9_esol.py
--------------
Test whether BF16 precision sharpening generalises to a different molecular
dataset: ESOL (aqueous solubility, ~1 128 compounds).

Pipeline
--------
1.  Download ESOL from DeepChem S3 → /tmp/esol/delaney-processed.csv
2.  Build ESOLDataset: SMILES→graph via chembl_data.smiles_to_graph,
    pad to MAX_ATOMS=40, normalise targets, 80/20 train/val split (seed 42)
3.  Train MolecularGNN (BF16, hidden_dim=256) for up to 100 epochs
    Auto-detect plateau: MAE improvement < 0.003 for 10 consecutive eval steps
4.  Warm restart: lr=5e-5, 20 epochs, save every epoch for LMC
5.  LMC: plateau epoch ↔ plateau+3 epoch
6.  Compare LMC barrier with QM9 BF16-256 reference (1.447 eV)
7.  Upload results.json + checkpoints to gs://.../phase9_esol/

GCS output: gs://aegismind-tpu-results/aegis_flashoptim/phase9_esol/
"""

import os, sys, json, time, subprocess
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

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

# ── Local imports (TPU VM at ~/flashoptim/) ───────────────────────────────────
sys.path.insert(0, os.path.expanduser("~/flashoptim"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MolecularGNN
from chembl_data import smiles_to_graph
from notify import notify, heartbeat

# ── Boilerplate ───────────────────────────────────────────────────────────────
GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID     = os.environ.get("RUN_ID",     "aegis_flashoptim")
GCS_BASE   = f"{GCS_BUCKET}/{RUN_ID}"

OUT_DIR  = Path("/tmp/phase9_esol")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ESOL_DIR = Path("/tmp/esol")
ESOL_DIR.mkdir(parents=True, exist_ok=True)
ESOL_URL  = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
ESOL_CSV  = ESOL_DIR / "delaney-processed.csv"


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def gsutil_cp(local, gcs):
    subprocess.run(["gsutil", "-q", "cp", str(local), gcs], check=False)


def get_device():
    if XLA_AVAILABLE:
        return xm.xla_device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Config ────────────────────────────────────────────────────────────────────
MAX_ATOMS         = 40
CUTOFF            = 5.0
HIDDEN_DIM        = 256
N_BLOCKS          = 6
N_GAUSSIANS       = 50
NUM_ATOM_TYPES    = 9
BATCH_SIZE        = 32
N_EPOCHS          = 100
LR_INIT           = 1e-4
WEIGHT_DECAY      = 1e-4
EVAL_EVERY        = 5
PLATEAU_PATIENCE  = 10     # eval steps with improvement < PLATEAU_DELTA
PLATEAU_DELTA     = 0.003  # log(mol/L) — smaller scale than eV
RESTART_LR        = 5e-5
RESTART_T_MAX     = 20
RESTART_ETA_MIN   = 1e-6
N_INTERP_STEPS    = 11
TRAIN_SPLIT       = 0.8
RANDOM_SEED       = 42

# Reference from Phase 4 / Phase 7 (BF16-256 QM9)
QM9_BF16_256_BARRIER_EV = 1.447
QM9_BF16_256_PEAK_ALPHA = 0.3


# ── ESOL Dataset ──────────────────────────────────────────────────────────────

class ESOLDataset(Dataset):
    PA = MAX_ATOMS  # padded atom count

    def __init__(self, records, mean, std):
        """
        records: list of (z_pad [PA], pos_pad [PA,3], valid [PA], target_norm)
        """
        self.records = records
        self.mean    = float(mean)
        self.std     = float(std)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.records[i]


def esol_collate(batch):
    """Stack list of (z_pad, pos_pad, valid, target) into tensors."""
    z     = torch.stack([b[0] for b in batch])   # [B, PA]
    pos   = torch.stack([b[1] for b in batch])   # [B, PA, 3]
    valid = torch.stack([b[2] for b in batch])   # [B, PA]
    tgt   = torch.tensor([b[3] for b in batch], dtype=torch.float32)  # [B]
    return z, pos, valid, tgt


def download_esol():
    if ESOL_CSV.exists():
        log(f"ESOL CSV already present: {ESOL_CSV}")
        return
    log(f"Downloading ESOL from {ESOL_URL} ...")
    urlretrieve(ESOL_URL, str(ESOL_CSV))
    log(f"Saved to {ESOL_CSV}")


def build_esol_records():
    """Parse CSV, convert SMILES→graph, pad, normalise. Returns (records, mean, std)."""
    import csv
    rows = []
    with open(ESOL_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles  = row["smiles"]
            target  = float(row["measured log solubility in mols per litre"])
            rows.append((smiles, target))

    log(f"ESOL: {len(rows)} rows in CSV")

    records_raw = []
    skipped = 0
    for smiles, target in rows:
        result = smiles_to_graph(smiles)
        if result is None:
            skipped += 1
            continue
        z_tensor, pos_tensor = result   # z: [n], pos: [n, 3]
        n = z_tensor.shape[0]
        if n > MAX_ATOMS:
            skipped += 1
            continue
        # Pad to MAX_ATOMS
        pad = MAX_ATOMS - n
        z_pad   = F.pad(z_tensor.long(),   (0, pad),       value=0)     # [PA]
        pos_pad = F.pad(pos_tensor.float(), (0, 0, 0, pad), value=0.0)  # [PA,3]
        valid   = torch.zeros(MAX_ATOMS, dtype=torch.bool)
        valid[:n] = True
        records_raw.append((z_pad, pos_pad, valid, target))

    log(f"ESOL: {len(records_raw)} valid molecules ({skipped} skipped)")

    targets = np.array([r[3] for r in records_raw], dtype=np.float32)
    mean_t  = float(targets.mean())
    std_t   = float(targets.std())
    if std_t < 1e-6:
        std_t = 1.0

    records = [
        (z, pos, valid, (t - mean_t) / std_t)
        for z, pos, valid, t in records_raw
    ]
    return records, mean_t, std_t


# ── Edge builder ──────────────────────────────────────────────────────────────

def build_edges_from_pos(pos, valid):
    """
    Build edge_src, edge_dst, assign_mat, edge_valid for a padded batch.
    pos:   [B, PA, 3]
    valid: [B, PA] bool
    Returns tensors on the same device as pos.
    """
    B, N, _ = pos.shape
    device   = pos.device
    i_idx    = torch.arange(N, device=device)

    # All ordered pairs (including self) → [N²]
    src = i_idx.unsqueeze(1).expand(N, N).reshape(-1)   # [N²]
    dst = i_idx.unsqueeze(0).expand(N, N).reshape(-1)   # [N²]

    edge_src = src.unsqueeze(0).expand(B, -1)            # [B, N²]
    edge_dst = dst.unsqueeze(0).expand(B, -1)            # [B, N²]

    # edge_valid: both endpoints real AND not a self-loop
    src_v    = valid.gather(1, edge_src)                 # [B, N²]
    dst_v    = valid.gather(1, edge_dst)                 # [B, N²]
    not_self = (edge_src != edge_dst)                    # [B, N²]
    edge_valid = src_v & dst_v & not_self                # [B, N²]

    # assign_mat: [B, N², N] — 1 where edge ends at atom n
    assign_mat = (edge_dst.unsqueeze(2) == i_idx.view(1, 1, N)).float()

    num_graphs = B
    return edge_src, edge_dst, assign_mat, num_graphs, edge_valid


# ── Training helpers ──────────────────────────────────────────────────────────

def train_epoch_bf16(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for z_b, pos_b, valid_b, tgt_b in loader:
        z_b, pos_b, valid_b, tgt_b = (
            z_b.to(device), pos_b.to(device),
            valid_b.to(device), tgt_b.to(device)
        )
        es, ed, am, ng, ev = build_edges_from_pos(pos_b, valid_b)
        optimizer.zero_grad()
        ctx = (torch.autocast("xla", dtype=torch.bfloat16)
               if XLA_AVAILABLE
               else torch.autocast("cpu", dtype=torch.bfloat16))
        with ctx:
            pred = model(z_b, pos_b, es, ed, am, ng, ev, valid_b)
            loss = F.mse_loss(pred.float(), tgt_b.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if XLA_AVAILABLE:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        total_loss += loss.item()
        n += 1
        if XLA_AVAILABLE:
            xm.mark_step()
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, std, use_bf16=False):
    model.eval()
    mae_sum, n = 0.0, 0
    for z_b, pos_b, valid_b, tgt_b in loader:
        z_b, pos_b, valid_b, tgt_b = (
            z_b.to(device), pos_b.to(device),
            valid_b.to(device), tgt_b.to(device)
        )
        es, ed, am, ng, ev = build_edges_from_pos(pos_b, valid_b)
        if use_bf16 and XLA_AVAILABLE:
            with torch.autocast("xla", dtype=torch.bfloat16):
                pred = model(z_b, pos_b, es, ed, am, ng, ev, valid_b)
        else:
            pred = model(z_b, pos_b, es, ed, am, ng, ev, valid_b)
        # Denormalise by std only (absolute MAE in log(mol/L))
        mae_sum += ((pred.float() - tgt_b.float()).abs() * std).sum().item()
        n += ng
        if XLA_AVAILABLE:
            xm.mark_step()
    return mae_sum / max(n, 1)


# ── LMC helpers ───────────────────────────────────────────────────────────────

def interpolate_sds(sd0, sd1, alpha):
    return {k: (1 - alpha) * sd0[k].float() + alpha * sd1[k].float()
            for k in sd0}


def run_lmc(sd0, sd1, model, loader, device, std, label, n_steps=11):
    log(f"LMC: {label}")
    alphas  = [i / (n_steps - 1) for i in range(n_steps)]
    records = []
    model.eval()
    for alpha in alphas:
        model.load_state_dict(interpolate_sds(sd0, sd1, alpha))
        mae = evaluate(model, loader, device, std)
        records.append({"alpha": round(alpha, 2), "mae_ev": round(mae, 4)})
        log(f"  alpha={alpha:.1f}  mae={mae:.4f}")
    maes    = [r["mae_ev"] for r in records]
    barrier = max(maes) - (maes[0] + maes[-1]) / 2
    return records, round(barrier, 4), alphas[int(np.argmax(maes))]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 68)
    log("  Phase 9 — BF16 MolecularGNN on ESOL (aqueous solubility)")
    log(f"  hidden_dim={HIDDEN_DIM}, n_blocks={N_BLOCKS}, MAX_ATOMS={MAX_ATOMS}")
    log(f"  {N_EPOCHS} epochs BF16, AdamW lr={LR_INIT}, wd={WEIGHT_DECAY}")
    log(f"  Plateau: patience={PLATEAU_PATIENCE} eval steps, delta={PLATEAU_DELTA}")
    log(f"  Warm restart: lr={RESTART_LR}, T_max={RESTART_T_MAX}")
    log(f"  QM9 BF16-256 ref barrier: {QM9_BF16_256_BARRIER_EV:.3f} eV "
        f"(peak alpha={QM9_BF16_256_PEAK_ALPHA})")
    log("=" * 68)

    device = get_device()
    log(f"Device: {device}" + (" (TPU)" if XLA_AVAILABLE else ""))

    # ── ESOL data ─────────────────────────────────────────────────────────────
    download_esol()
    log("Building ESOL graph records ...")
    records, mean_t, std_t = build_esol_records()
    log(f"Target stats: mean={mean_t:.4f}  std={std_t:.4f} log(mol/L)")

    dataset = ESOLDataset(records, mean_t, std_t)
    n_train = int(len(dataset) * TRAIN_SPLIT)
    n_val   = len(dataset) - n_train
    gen     = torch.Generator().manual_seed(RANDOM_SEED)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)
    log(f"Split: {n_train} train / {n_val} val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=esol_collate, num_workers=4,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=esol_collate, num_workers=4)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MolecularGNN(
        num_atom_types=NUM_ATOM_TYPES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=N_BLOCKS,
        num_gaussians=N_GAUSSIANS,
        cutoff=CUTOFF,
        num_targets=1,
    ).to(device)
    log(f"MolecularGNN: hidden_dim={model.hidden_dim}  "
        f"params={model.parameter_count():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-6
    )

    notify("PHASE_START", "[Phase9] ESOL BF16 training begun",
           data={"hidden_dim": HIDDEN_DIM, "n_train": n_train, "n_val": n_val,
                 "std_t": round(std_t, 4)})

    # ─────────────────────────────────────────────────────────────────────────
    #  SECTION A: BF16 training with plateau detection
    # ─────────────────────────────────────────────────────────────────────────
    log(f"\n{'─'*68}")
    log("  SECTION A: BF16 training (up to 100 epochs)")
    log(f"  {'ep':>5}  {'lr':>10}  {'train_loss':>12}  "
        f"{'val_mae':>12}  {'pat':>5}")
    log(f"  {'─'*5}  {'─'*10}  {'─'*12}  {'─'*12}  {'─'*5}")

    eval_history   = []
    saved_sds      = {}
    plateau_epoch  = None
    patience_count = 0
    best_mae       = float("inf")

    mae_ep0 = evaluate(model, val_loader, device, std_t)
    eval_history.append((0, round(mae_ep0, 4)))
    best_mae = mae_ep0
    log(f"  ep  0  (initial): {mae_ep0:.4f}")

    for ep in range(1, N_EPOCHS + 1):
        lr         = optimizer.param_groups[0]["lr"]
        train_loss = train_epoch_bf16(model, train_loader, optimizer, device)
        scheduler.step()

        if ep % EVAL_EVERY == 0 or ep == N_EPOCHS:
            val_mae = evaluate(model, val_loader, device, std_t)
            eval_history.append((ep, round(val_mae, 4)))

            if plateau_epoch is None:
                improvement = best_mae - val_mae
                if improvement >= PLATEAU_DELTA:
                    best_mae       = val_mae
                    patience_count = 0
                else:
                    patience_count += 1

                if patience_count >= PLATEAU_PATIENCE:
                    plateau_epoch = ep
                    log(f"  *** PLATEAU at ep{ep} (patience={patience_count}) ***")
                    saved_sds[ep] = {k: v.clone().cpu()
                                     for k, v in model.state_dict().items()}
                    ckpt_path = OUT_DIR / f"esol_plateau_ep{ep}.pt"
                    torch.save({"epoch": ep, "model": model.state_dict(),
                                "val_mae": val_mae}, ckpt_path)
                    gsutil_cp(ckpt_path,
                              f"{GCS_BASE}/phase9_esol/esol_plateau_ep{ep}.pt")

            log(f"  ep{ep:>3d}  {lr:>10.2e}  {train_loss:>12.6f}  "
                f"{val_mae:>12.4f}  {patience_count:>5d}"
                + ("  [PLATEAU]" if ep == plateau_epoch else ""))

            heartbeat("Phase9_train", ep,
                      {"ep": ep, "val_mae": round(val_mae, 4),
                       "patience": patience_count})

            if plateau_epoch is not None:
                if ep not in saved_sds:
                    saved_sds[ep] = {k: v.clone().cpu()
                                     for k, v in model.state_dict().items()}
                break
        else:
            log(f"  ep{ep:>3d}  {lr:>10.2e}  {train_loss:>12.6f}")

    if plateau_epoch is None:
        plateau_epoch = N_EPOCHS
        log(f"  NOTE: No plateau in {N_EPOCHS} epochs; using ep{plateau_epoch}.")
        if plateau_epoch not in saved_sds:
            saved_sds[plateau_epoch] = {k: v.clone().cpu()
                                        for k, v in model.state_dict().items()}

    mae_at_plateau = dict(eval_history).get(plateau_epoch, float("nan"))
    log(f"\n  Plateau ep={plateau_epoch}  val_mae={mae_at_plateau:.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    #  SECTION B: Warm restart
    # ─────────────────────────────────────────────────────────────────────────
    log(f"\n{'─'*68}")
    log(f"  SECTION B: Warm restart from ep{plateau_epoch}")
    log(f"  lr={RESTART_LR}, CosineAnnealingLR T_max={RESTART_T_MAX}")

    model.load_state_dict(saved_sds[plateau_epoch])
    restart_opt   = torch.optim.AdamW(
        model.parameters(), lr=RESTART_LR, weight_decay=WEIGHT_DECAY
    )
    restart_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        restart_opt, T_max=RESTART_T_MAX, eta_min=RESTART_ETA_MIN
    )

    restart_history = []
    mae_restart_base = evaluate(model, val_loader, device, std_t)
    restart_history.append((plateau_epoch, round(mae_restart_base, 4)))

    log(f"  {'ep':>5}  {'lr':>10}  {'train_loss':>12}  {'val_mae':>12}")
    log(f"  {'─'*5}  {'─'*10}  {'─'*12}  {'─'*12}")

    for step in range(1, RESTART_T_MAX + 1):
        ep         = plateau_epoch + step
        lr         = restart_opt.param_groups[0]["lr"]
        train_loss = train_epoch_bf16(model, train_loader, restart_opt, device)
        restart_sched.step()
        val_mae    = evaluate(model, val_loader, device, std_t)
        restart_history.append((ep, round(val_mae, 4)))

        saved_sds[ep] = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        ckpt_path = OUT_DIR / f"esol_restart_ep{ep}.pt"
        torch.save({"epoch": ep, "model": model.state_dict(),
                    "val_mae": val_mae}, ckpt_path)
        gsutil_cp(ckpt_path, f"{GCS_BASE}/phase9_esol/esol_restart_ep{ep}.pt")

        log(f"  ep{ep:>3d}  {lr:>10.2e}  {train_loss:>12.6f}  {val_mae:>12.4f}")
        heartbeat("Phase9_restart", step,
                  {"ep": ep, "val_mae": round(val_mae, 4)})

    # ─────────────────────────────────────────────────────────────────────────
    #  SECTION C: LMC
    # ─────────────────────────────────────────────────────────────────────────
    log(f"\n{'─'*68}")
    log("  SECTION C: LMC measurement")

    ep_post = plateau_epoch + 3
    if ep_post not in saved_sds:
        ep_post = max(k for k in saved_sds if k > plateau_epoch)
        log(f"  WARNING: ep+3 missing; using ep{ep_post}")

    model_lmc = MolecularGNN(
        num_atom_types=NUM_ATOM_TYPES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=N_BLOCKS,
        num_gaussians=N_GAUSSIANS,
        cutoff=CUTOFF,
        num_targets=1,
    ).to(device)

    lmc_label = (f"ESOL BF16-256  ep{plateau_epoch} <-> ep{ep_post}  "
                 f"(3-epoch window)")
    records_lmc, barrier, peak_alpha = run_lmc(
        saved_sds[plateau_epoch], saved_sds[ep_post],
        model_lmc, val_loader, device, std_t, lmc_label,
        n_steps=N_INTERP_STEPS,
    )

    # ─────────────────────────────────────────────────────────────────────────
    #  SECTION D: Summary
    # ─────────────────────────────────────────────────────────────────────────
    log(f"\n{'='*68}")
    log("  PHASE 9 SUMMARY: ESOL BF16-256 vs QM9 BF16-256 LMC Barriers")
    log(f"  {'Experiment':<38}  {'Barrier':>8}  {'Peak alpha':>12}")
    log(f"  {'─'*38}  {'─'*8}  {'─'*12}")
    log(f"  {'ESOL BF16-256 (Phase 9)':<38}  {barrier:>8.4f}  {peak_alpha:>12.2f}")
    log(f"  {'QM9  BF16-256 (Phase 4 ref)':<38}  "
        f"{QM9_BF16_256_BARRIER_EV:>8.4f}  {QM9_BF16_256_PEAK_ALPHA:>12.2f}")
    generalises = barrier > 0.05
    delta       = barrier - QM9_BF16_256_BARRIER_EV
    if generalises:
        log(f"  -> BF16 sharpening barrier confirmed on ESOL ({barrier:.4f})")
    else:
        log(f"  -> Barrier near-zero ({barrier:.4f}) — flat landscape on ESOL")
    log(f"  -> Delta vs QM9: {delta:+.4f}  "
        f"({'within 0.2' if abs(delta) < 0.2 else 'differs by >0.2'})")
    log(f"{'='*68}")

    # ─────────────────────────────────────────────────────────────────────────
    #  SECTION E: Save and upload
    # ─────────────────────────────────────────────────────────────────────────
    summary = {
        "experiment":          "phase9_esol",
        "dataset":             "ESOL (delaney-processed.csv)",
        "n_train":             n_train,
        "n_val":               n_val,
        "target_mean":         round(mean_t, 4),
        "target_std":          round(std_t, 4),
        "architecture":        "MolecularGNN",
        "precision":           "bf16",
        "hidden_dim":          HIDDEN_DIM,
        "num_blocks":          N_BLOCKS,
        "max_atoms":           MAX_ATOMS,
        "n_training_epochs":   N_EPOCHS,
        "batch_size":          BATCH_SIZE,
        "lr_init":             LR_INIT,
        "weight_decay":        WEIGHT_DECAY,
        "plateau_patience":    PLATEAU_PATIENCE,
        "plateau_delta":       PLATEAU_DELTA,
        "plateau_epoch":       plateau_epoch,
        "mae_at_plateau":      round(mae_at_plateau, 4),
        "restart_lr":          RESTART_LR,
        "restart_t_max":       RESTART_T_MAX,
        "training_trajectory": [{"epoch": e, "val_mae": m}
                                 for e, m in eval_history],
        "restart_trajectory":  [{"epoch": e, "val_mae": m}
                                 for e, m in restart_history],
        "lmc": {
            "ep_pre":          plateau_epoch,
            "ep_post":         ep_post,
            "n_interp_steps":  N_INTERP_STEPS,
            "interpolation":   records_lmc,
            "barrier":         barrier,
            "peak_alpha":      peak_alpha,
        },
        "comparison": {
            "qm9_bf16_256_barrier_ev":     QM9_BF16_256_BARRIER_EV,
            "qm9_bf16_256_peak_alpha":     QM9_BF16_256_PEAK_ALPHA,
            "esol_bf16_256_barrier":       barrier,
            "esol_bf16_256_peak_alpha":    peak_alpha,
            "delta_vs_qm9":               round(delta, 4),
            "generalises_to_esol":         generalises,
            "barriers_agree_within_0p2":   abs(delta) < 0.2,
        },
        "gcs_output": f"{GCS_BASE}/phase9_esol/",
    }

    out_json = OUT_DIR / "results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    log(f"\nResults saved -> {out_json}")
    gsutil_cp(out_json, f"{GCS_BASE}/phase9_esol/results.json")
    log(f"GCS: {GCS_BASE}/phase9_esol/")

    notify("PHASE_COMPLETE", "[Phase9] ESOL BF16 LMC complete",
           data={"plateau_epoch": plateau_epoch,
                 "mae_at_plateau": round(mae_at_plateau, 4),
                 "lmc_barrier":    barrier,
                 "qm9_ref_ev":     QM9_BF16_256_BARRIER_EV,
                 "generalises":    generalises})


if __name__ == "__main__":
    main()
