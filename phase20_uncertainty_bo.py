#!/usr/bin/env python3
"""
phase20_uncertainty_bo.py
-------------------------
Uncertainty-gated BO: adds MC-dropout to the Phase 6 target-conditioned
surrogate and runs BO with Upper Confidence Bound (UCB) acquisition to
test whether proper uncertainty quantification fixes the Spearman ρ
degradation seen in Phase 6.

Phase 6 BO pool ρ (vs training ρ):
  MSH3:   0.583 (vs 0.832)  ← worst degradation
  APEX1:  0.507 (vs 0.641)
  CREBBP: 0.755 (vs 0.801)
  KPC3:   0.778 (vs 0.826)
  PCSK9:  0.640 (vs 0.763)
  LINGO1: 0.875 (vs 0.876)  ← only BO-supported target

Hypothesis: EI acquisition ignores surrogate uncertainty, confidently
acquiring compounds in low-fidelity regions of the pool. UCB (μ + κσ)
will avoid high-uncertainty regions, maintaining ρ > 0.70 across all
rounds and potentially finding better lead compounds.

Protocol
--------
1. Retrain TargetConditionedSurrogateDropout (Phase 6 arch + Dropout(0.10)
   in output head) on vina_scores.json, FP32, hidden_dim=256, 50 epochs.
2. Run EI BO × 6 targets   (Phase 6 replica — in-script baseline)
3. Run UCB BO × 6 targets  (κ=1.0, MC-dropout N_MC=20 for σ estimates)
4. Report per target per variant: best pKd, ρ_pool per round, mean σ.
5. Calibration check: Pearson r between predicted σ and |μ − vina| on
   the held-out validation set.

GCS output: gs://.../aegis_flashoptim/phase20_uncertainty_bo/results.json
"""

import os, sys, json, random, subprocess, time
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr

try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
    try:
        import torch_xla.experimental as _e
        _e.eager_mode(True)
    except Exception:
        pass
except ImportError:
    XLA_AVAILABLE = False

sys.path.insert(0, os.path.expanduser("~/flashoptim"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase3_surrogate_bayes import SurrogateGNN
from phase2_pdbbind import BindingAffinityGNN
from phase3_vina_oracle import load_pool_graphs, vina_to_pkd
from phase6_vina_surrogate import TARGETS, TARGET2ID, N_TARGETS
from notify import notify, heartbeat
from compat import autocast

GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID     = os.environ.get("RUN_ID", "aegis_flashoptim")
GCS_BASE   = f"{GCS_BUCKET}/{RUN_ID}"

DATA_DIR   = Path("/tmp/flashoptim_results")
OUT_DIR    = Path("/tmp/phase20_uncertainty_bo")
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

PA           = BindingAffinityGNN.PADDED_ATOMS
HIDDEN_DIM   = 256
TARGET_EMB   = 16
DROPOUT_RATE = 0.10
N_MC         = 20          # MC-dropout forward passes
UCB_KAPPA    = 1.0         # exploration weight
N_INIT       = 10
N_ROUNDS     = 30
K_BATCH      = 5
RETRAIN_EPOCHS = 20
RETRAIN_BATCH  = 32
TRAIN_FRAC   = 0.80
RANDOM_SEED  = 42
LR           = 1e-4
WEIGHT_DECAY = 1e-5
N_EPOCHS     = 50
BATCH_SIZE   = 64
FIDELITY_RHO = 0.70

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}] [Phase20] {msg}",
          flush=True)

def gsutil_cp(src, dst):
    subprocess.run(["gsutil", "-q", "cp", str(src), dst], check=False)

def gsutil_dl(src, dst):
    r = subprocess.run(["gsutil", "-q", "cp", src, str(dst)], capture_output=True)
    return r.returncode == 0 and Path(dst).exists()


# ── Architecture ──────────────────────────────────────────────────────────────

class TargetConditionedSurrogateDropout(nn.Module):
    """Phase 6 TargetConditionedSurrogate + Dropout in output head."""
    def __init__(self, hidden_dim=HIDDEN_DIM, n_targets=N_TARGETS,
                 emb_dim=TARGET_EMB, dropout=DROPOUT_RATE):
        super().__init__()
        self.base       = SurrogateGNN(hidden_dim=hidden_dim)
        self.target_emb = nn.Embedding(n_targets, emb_dim)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim + emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),      # MC-dropout point
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),      # second MC-dropout point
            nn.Linear(64, 1),
        )

    def forward(self, z_pad, pos_pad, atom_valid, target_id):
        h = self.base._embed(z_pad, pos_pad, atom_valid)   # [B, hidden_dim]
        t = self.target_emb(target_id)                      # [B, emb_dim]
        x = torch.cat([h, t], dim=-1)
        return self.output_head(x).squeeze(-1)              # [B]


@contextmanager
def mc_dropout_mode(model):
    """Enable dropout during inference without updating BN running stats."""
    was_training = model.training
    model.train()
    # Freeze any BN layers
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
    try:
        yield
    finally:
        model.train(was_training)


def mc_predict(model, z, pos, val, tid, n_mc=N_MC, device=None):
    """Return (mean, std) over N_MC stochastic forward passes."""
    preds = []
    with mc_dropout_mode(model):
        with torch.no_grad():
            for _ in range(n_mc):
                p = model(z, pos, val, tid)
                if XLA_AVAILABLE:
                    xm.mark_step()
                preds.append(p.cpu().float())
    stacked = torch.stack(preds, dim=0)   # [N_MC, B]
    return stacked.mean(0), stacked.std(0)


def _stack_to_device(pool_graphs, names, device):
    items = [pool_graphs[n] for n in names]
    z   = torch.stack([g[0].squeeze(0) for g in items]).to(device)
    pos = torch.stack([g[1].squeeze(0) for g in items]).to(device)
    val = torch.stack([g[2].squeeze(0) for g in items]).to(device)
    return z, pos, val


# ── Surrogate training ────────────────────────────────────────────────────────

def train_surrogate_dropout(vina_scores, pool_graphs, device):
    """Train TargetConditionedSurrogateDropout on vina_scores."""
    log("Training surrogate with MC-dropout...")

    # Build (name, target_id, pkd) triples
    triples = []
    for name, target_scores in vina_scores.items():
        if name not in pool_graphs:
            continue
        for target, aff in target_scores.items():
            if target in TARGET2ID and aff < 0:
                triples.append((name, TARGET2ID[target], vina_to_pkd(aff)))

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(triples)
    split = int(len(triples) * TRAIN_FRAC)
    train_data, val_data = triples[:split], triples[split:]
    log(f"  {len(train_data)} train / {len(val_data)} val pairs")

    model = TargetConditionedSurrogateDropout().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_rho, best_sd, patience_cnt = -1.0, None, 0

    for ep in range(1, N_EPOCHS + 1):
        model.train()
        rng.shuffle(train_data)
        total, count = 0.0, 0
        for i in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[i:i + BATCH_SIZE]
            names   = [b[0] for b in batch]
            tids    = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
            targets = torch.tensor([b[2] for b in batch], dtype=torch.float, device=device)
            z, pos, val = _stack_to_device(pool_graphs, names, device)
            opt.zero_grad()
            pred = model(z, pos, val, tids)
            loss = F.mse_loss(pred, targets)
            loss.backward()
            if XLA_AVAILABLE:
                xm.optimizer_step(opt)
            else:
                opt.step()
            total += loss.item() * len(batch)
            count += len(batch)
        if XLA_AVAILABLE:
            xm.mark_step()

        if ep % 5 == 0 or ep == N_EPOCHS:
            model.eval()
            val_preds, val_true = [], []
            with torch.no_grad():
                for i in range(0, len(val_data), BATCH_SIZE):
                    batch = val_data[i:i + BATCH_SIZE]
                    names   = [b[0] for b in batch]
                    tids    = torch.tensor([b[1] for b in batch],
                                           dtype=torch.long, device=device)
                    z, pos, v = _stack_to_device(pool_graphs, names, device)
                    val_preds.extend(model(z, pos, v, tids).cpu().tolist())
                    val_true.extend([b[2] for b in batch])
            per_target_rho = []
            for tid, tname in enumerate(TARGETS):
                idx = [i for i, b in enumerate(val_data) if b[1] == tid]
                if len(idx) >= 5:
                    rho, _ = spearmanr([val_preds[i] for i in idx],
                                       [val_true[i]  for i in idx])
                    per_target_rho.append(rho)
            mean_rho = float(np.mean(per_target_rho)) if per_target_rho else 0.0
            log(f"  ep{ep:3d}  loss={total/count:.5f}  val_ρ={mean_rho:.3f}")
            heartbeat("Phase20_surrogate", epoch=ep, metrics={"val_rho": round(mean_rho, 3)})

            if mean_rho > best_val_rho:
                best_val_rho = mean_rho
                best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= 3:
                    log(f"  Early stop at ep{ep} (patience=3 × 5ep)")
                    break

    model.load_state_dict(best_sd)
    log(f"  Best surrogate val ρ = {best_val_rho:.3f}")
    return model, best_val_rho, val_data


def calibration_check(model, val_data, pool_graphs, device):
    """Pearson r between predicted σ and |μ − true_pkd| on validation set."""
    sigmas, errors = [], []
    for i in range(0, len(val_data), 32):
        batch = val_data[i:i + 32]
        names = [b[0] for b in batch]
        tids  = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
        true  = [b[2] for b in batch]
        z, pos, val = _stack_to_device(pool_graphs, names, device)
        mu, sigma = mc_predict(model, z, pos, val, tids, n_mc=N_MC)
        sigmas.extend(sigma.tolist())
        errors.extend([abs(mu[j].item() - true[j]) for j in range(len(batch))])
    if len(sigmas) < 5:
        return None
    r, p = pearsonr(sigmas, errors)
    return {"pearson_r": round(r, 4), "p_value": round(p, 4), "n": len(sigmas)}


# ── BO variants ───────────────────────────────────────────────────────────────

def run_bo_variant(label, use_ucb, model, pool_graphs, vina_scores,
                   target, target_id, device):
    """Run 30-round BO. use_ucb=True → UCB(κ=1.0); False → EI (Phase 6 replica)."""
    pool_names = list(pool_graphs.keys())
    true_pkd   = {n: vina_to_pkd(vina_scores[n].get(target, 0))
                  for n in pool_names if vina_scores.get(n, {}).get(target, 0) < 0}
    pool_names = [n for n in pool_names if n in true_pkd]

    rng = random.Random(RANDOM_SEED)
    observed = rng.sample(pool_names, N_INIT)
    obs_pkd  = [true_pkd[n] for n in observed]
    remaining = [n for n in pool_names if n not in set(observed)]

    best_pkd   = max(obs_pkd)
    round_log  = []
    total_sigma = []

    for rnd in range(1, N_ROUNDS + 1):
        # Surrogate prediction on remaining pool
        mu_all, sigma_all = [], []
        for i in range(0, len(remaining), 64):
            chunk = remaining[i:i + 64]
            z, pos, val = _stack_to_device(pool_graphs, chunk, device)
            tid = torch.full((len(chunk),), target_id, dtype=torch.long, device=device)
            if use_ucb:
                mu, sigma = mc_predict(model, z, pos, val, tid)
            else:
                model.eval()
                with torch.no_grad():
                    mu = model(z, pos, val, tid).cpu().float()
                sigma = torch.zeros_like(mu)
            mu_all.append(mu); sigma_all.append(sigma)
        mu_all    = torch.cat(mu_all)
        sigma_all = torch.cat(sigma_all)

        # Acquisition
        if use_ucb:
            scores = mu_all + UCB_KAPPA * sigma_all
        else:
            best_obs = max(obs_pkd)
            improvement = mu_all - best_obs
            # Simple EI approximation (deterministic surrogate → use μ only)
            scores = improvement.clamp(min=0)

        top_idx = scores.topk(K_BATCH).indices.tolist()
        acquired = [remaining[i] for i in top_idx]

        # Oracle
        for name in acquired:
            pkd = true_pkd[name]
            obs_pkd.append(pkd)
            observed.append(name)
            best_pkd = max(best_pkd, pkd)
        remaining = [n for n in remaining if n not in set(acquired)]

        # ρ on remaining pool (surrogate fidelity during acquisition)
        if len(remaining) >= 10:
            sample = rng.sample(remaining, min(200, len(remaining)))
            mu_s, _ = [], []
            for i in range(0, len(sample), 64):
                chunk = sample[i:i + 64]
                z, pos, val = _stack_to_device(pool_graphs, chunk, device)
                tid = torch.full((len(chunk),), target_id, dtype=torch.long, device=device)
                model.eval()
                with torch.no_grad():
                    mu_s.append(model(z, pos, val, tid).cpu().float())
            mu_s   = torch.cat(mu_s).tolist()
            true_s = [true_pkd[n] for n in sample]
            rho, _ = spearmanr(mu_s, true_s)
        else:
            rho = float("nan")

        mean_sig = sigma_all.mean().item() if use_ucb else 0.0
        total_sigma.append(mean_sig)
        round_log.append({"round": rnd, "best_pkd": round(best_pkd, 3),
                          "rho_pool": round(rho, 3), "mean_sigma": round(mean_sig, 4)})

        if rnd % 5 == 0:
            log(f"  [{label}/{target}] rnd{rnd:2d}  best={best_pkd:.3f}  "
                f"ρ_pool={rho:.3f}  σ̄={mean_sig:.3f}")

        # Surrogate retrain every 5 rounds on observed data
        if rnd % 5 == 0 and len(observed) >= 20:
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
            obs_names = observed[-min(len(observed), 200):]
            obs_y     = obs_pkd[-len(obs_names):]
            for _ in range(RETRAIN_EPOCHS):
                idx = list(range(len(obs_names)))
                rng.shuffle(idx)
                for i in range(0, len(idx), RETRAIN_BATCH):
                    b_idx = idx[i:i + RETRAIN_BATCH]
                    names = [obs_names[j] for j in b_idx]
                    tids  = torch.full((len(names),), target_id,
                                       dtype=torch.long, device=device)
                    y = torch.tensor([obs_y[j] for j in b_idx],
                                     dtype=torch.float, device=device)
                    z, pos, val = _stack_to_device(pool_graphs, names, device)
                    opt.zero_grad()
                    loss = F.mse_loss(model(z, pos, val, tids), y)
                    loss.backward()
                    if XLA_AVAILABLE:
                        xm.optimizer_step(opt)
                    else:
                        opt.step()
            if XLA_AVAILABLE:
                xm.mark_step()

    final_rho, _ = spearmanr(
        [true_pkd.get(n, 0) for n in pool_names],
        _predict_all(model, pool_names, target_id, pool_graphs, device))

    return {"label": label, "target": target, "use_ucb": use_ucb,
            "best_pkd": round(best_pkd, 3),
            "final_rho": round(final_rho, 3),
            "round_log": round_log,
            "mean_sigma": round(float(np.mean(total_sigma)), 4)}


def _predict_all(model, names, target_id, pool_graphs, device):
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(names), 64):
            chunk = names[i:i + 64]
            z, pos, val = _stack_to_device(pool_graphs, chunk, device)
            tid = torch.full((len(chunk),), target_id, dtype=torch.long, device=device)
            preds.extend(model(z, pos, val, tid).cpu().tolist())
    return preds


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 65)
    log("  Phase 20 — Uncertainty-gated BO (MC-dropout + UCB)")
    log(f"  Targets: {TARGETS}")
    log(f"  Dropout={DROPOUT_RATE}, N_MC={N_MC}, UCB κ={UCB_KAPPA}")
    log("=" * 65)

    device = xm.xla_device() if XLA_AVAILABLE else torch.device("cpu")
    log(f"Device: {device}")

    # Load vina scores
    vina_path = DATA_DIR / "vina_scores.json"
    if not vina_path.exists():
        log("Downloading vina_scores.json from GCS...")
        subprocess.run(["gsutil", "-q", "cp",
                        f"{GCS_BASE}/vina_scores.json", str(vina_path)], check=True)
    with open(vina_path) as f:
        vina_scores = json.load(f)
    log(f"Vina scores: {len(vina_scores)} compounds")

    # Load pool graphs — load_pool_graphs is per-target; merge across all targets
    log("Loading pool graphs (per target, then merged)...")
    fda_path = DATA_DIR / "pubchem_fda.json"
    if not fda_path.exists():
        subprocess.run(["gsutil", "-q", "cp",
                        f"{GCS_BASE}/pubchem_fda.json", str(fda_path)], check=True)
    pool_graphs = {}
    for t in TARGETS:
        t_graphs, _ = load_pool_graphs(vina_scores, t, device)
        pool_graphs.update(t_graphs)
    log(f"Pool graphs: {len(pool_graphs)} unique compounds across {len(TARGETS)} targets")

    # Train surrogate with dropout
    model, best_rho, val_data = train_surrogate_dropout(vina_scores, pool_graphs, device)
    ckpt = OUT_DIR / "phase20_dropout_surrogate.pt"
    torch.save(model.state_dict(), ckpt)
    gsutil_cp(ckpt, f"{GCS_BASE}/phase20_uncertainty_bo/surrogate.pt")

    # Calibration check
    log("Running calibration check...")
    calib = calibration_check(model, val_data, pool_graphs, device)
    log(f"  Calibration: σ vs |μ−true| Pearson r = {calib['pearson_r']:.3f} "
        f"(p={calib['p_value']:.3f})" if calib else "  Calibration: skipped")

    # BO runs: EI then UCB for each target
    all_results = []
    phase6_rho_ref = {"LINGO1": 0.875, "PCSK9": 0.640, "KPC3": 0.778,
                      "APEX1": 0.507,  "MSH3": 0.583,  "CREBBP": 0.755}

    for target in TARGETS:
        target_id = TARGET2ID[target]
        log(f"\n  === {target} ===")

        # Reload fresh model for each target/variant to avoid state bleed
        for use_ucb in (False, True):
            variant_model = TargetConditionedSurrogateDropout().to(device)
            variant_model.load_state_dict(
                torch.load(ckpt, map_location="cpu"))
            variant_model.to(device)
            label = "UCB" if use_ucb else "EI"
            res = run_bo_variant(label, use_ucb, variant_model, pool_graphs,
                                 vina_scores, target, target_id, device)
            res["phase6_rho_ref"] = phase6_rho_ref.get(target)
            all_results.append(res)

    # ── Summary table ──────────────────────────────────────────────────────────
    log("\n" + "=" * 65)
    log("  PHASE 20 RESULTS — EI vs UCB BO")
    log("=" * 65)
    log(f"\n  {'Target':8s}  {'Variant':6s}  {'Best pKd':>9}  "
        f"{'Final ρ':>8}  {'Phase6 ρ ref':>13}  {'Δρ':>6}")
    log(f"  {'─'*8}  {'─'*6}  {'─'*9}  {'─'*8}  {'─'*13}  {'─'*6}")
    for r in all_results:
        ref  = r.get("phase6_rho_ref", 0)
        delta = round(r["final_rho"] - ref, 3) if ref else "—"
        log(f"  {r['target']:8s}  {r['label']:6s}  {r['best_pkd']:>9.3f}  "
            f"{r['final_rho']:>8.3f}  {str(ref):>13}  {str(delta):>6}")

    # Did UCB fix the ρ degradation?
    ucb_rhos = {r["target"]: r["final_rho"] for r in all_results if r["use_ucb"]}
    ei_rhos  = {r["target"]: r["final_rho"] for r in all_results if not r["use_ucb"]}
    n_ucb_above_threshold = sum(1 for rho in ucb_rhos.values() if rho >= FIDELITY_RHO)
    n_ei_above_threshold  = sum(1 for rho in ei_rhos.values()  if rho >= FIDELITY_RHO)
    log(f"\n  Targets with ρ_pool ≥ {FIDELITY_RHO}: EI={n_ei_above_threshold}/6, "
        f"UCB={n_ucb_above_threshold}/6")
    hypothesis_supported = n_ucb_above_threshold > n_ei_above_threshold
    log(f"  UCB ρ-degradation fix: {'SUPPORTED ✓' if hypothesis_supported else 'NOT SUPPORTED ✗'}")

    results = {
        "experiment":       "phase20_uncertainty_bo",
        "surrogate_val_rho": round(best_rho, 3),
        "calibration":       calib,
        "bo_results":        all_results,
        "n_ucb_rho_pass":    n_ucb_above_threshold,
        "n_ei_rho_pass":     n_ei_above_threshold,
        "hypothesis_supported": hypothesis_supported,
        "phase6_rho_ref":    phase6_rho_ref,
        "fidelity_threshold": FIDELITY_RHO,
    }
    out = OUT_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    gsutil_cp(out, f"{GCS_BASE}/phase20_uncertainty_bo/results.json")
    log(f"\n  Results → {out}")
    log(f"  GCS     → {GCS_BASE}/phase20_uncertainty_bo/results.json")
    notify("phase20_complete", "Phase 20 complete — uncertainty-gated BO", data=results)


if __name__ == "__main__":
    main()
