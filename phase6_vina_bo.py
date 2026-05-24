#!/usr/bin/env python3
"""
phase6_vina_bo.py
-----------------
Phase 6 Bayesian optimisation using the target-conditioned surrogate
trained in phase6_vina_surrogate.py.

Loads phase6_fp32_best.pt and phase6_bf16_best.pt from GCS, then runs
the same 30-round BO comparison as Phase 5 but with surrogates that
actually know which target they're optimising (Spearman ρ ≥ 0.70).

Output: gs://.../aegis_flashoptim/phase6/phase6_vina_bo_results.json
"""

import os, sys, json, random, subprocess, time, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
    try:
        import torch_xla.experimental as _xla_exp
        _xla_exp.eager_mode(True)
        print("XLA eager mode: ENABLED")
    except Exception:
        pass
except ImportError:
    XLA_AVAILABLE = False

from notify import notify, heartbeat
from phase2_pdbbind import BindingAffinityGNN
from phase3_surrogate_bayes import expected_improvement
from phase3_vina_oracle import VinaOracle, load_pool_graphs, vina_to_pkd
from phase6_vina_surrogate import TargetConditionedSurrogate, TARGETS, TARGET2ID
from compat import autocast

# ── Config ────────────────────────────────────────────────────────────────────
GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID     = os.environ.get("RUN_ID", "aegis_flashoptim")
GCS_BASE   = f"{GCS_BUCKET}/{RUN_ID}"

DATA_DIR   = Path("/tmp/flashoptim_results")
DATA_DIR.mkdir(exist_ok=True)

N_INIT    = 10
N_ROUNDS  = 30
K_BATCH   = 5
EI_CHUNK  = 64
RETRAIN_EPOCHS = 20
RETRAIN_BATCH  = 32

PA = BindingAffinityGNN.PADDED_ATOMS
FIDELITY_RHO = 0.70


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def gsutil_cp(local, gcs):
    subprocess.run(["gsutil", "-q", "cp", str(local), gcs], check=False)


def _stack_to_device(pool_graphs, names, device):
    items = [pool_graphs[n] for n in names]
    z   = torch.stack([g[0] for g in items]).to(device)
    pos = torch.stack([g[1] for g in items]).to(device)
    val = torch.stack([g[2] for g in items]).to(device)
    return z, pos, val


# ── Target-conditioned BO loop ────────────────────────────────────────────────

class Phase6BayesOptLoop:
    """
    BO loop using TargetConditionedSurrogate.
    target_id is fixed for the duration of the loop (one target per run).
    """

    def __init__(self, surrogate: TargetConditionedSurrogate,
                 oracle: VinaOracle, pool_graphs: dict,
                 target_id: int, device, label: str, use_bf16: bool = False):
        self.surrogate   = surrogate.to(device)
        self.oracle      = oracle
        self.pool_graphs = pool_graphs
        self.target_id   = target_id          # constant: int
        self.device      = device
        self.label       = label
        self.use_bf16    = use_bf16

        self.observed_names: list = []
        self.observed_pkd:   list = []
        self.best_pkd = -float("inf")
        self.history  = []

    def _tid_tensor(self, batch_size: int) -> torch.Tensor:
        """Broadcast the fixed target_id to a [B] LongTensor on device."""
        return torch.full((batch_size,), self.target_id,
                          dtype=torch.long, device=self.device)

    def _score_oracle(self, name: str) -> float:
        return self.oracle(name)

    def _retrain_surrogate(self):
        n_obs = len(self.observed_names)
        if n_obs < 5:
            return

        opt = torch.optim.AdamW(self.surrogate.parameters(), lr=3e-4)
        self.surrogate.train()
        pkd_arr = torch.tensor(self.observed_pkd, dtype=torch.float)

        for _ in range(RETRAIN_EPOCHS):
            if n_obs >= RETRAIN_BATCH:
                idx = random.sample(range(n_obs), RETRAIN_BATCH)
            else:
                idx = random.choices(range(n_obs), k=RETRAIN_BATCH)

            z_b, pos_b, valid_b = _stack_to_device(
                self.pool_graphs, [self.observed_names[i] for i in idx], self.device)
            pkd_b = pkd_arr[idx].to(self.device)
            tid_b = self._tid_tensor(len(idx))

            opt.zero_grad()
            if self.use_bf16:
                with torch.autocast(
                    device_type="xla" if XLA_AVAILABLE else "cpu",
                    dtype=torch.bfloat16, enabled=True):
                    pred = self.surrogate(z_b, pos_b, valid_b, tid_b)
                    loss = F.mse_loss(pred.float(), pkd_b.float())
            else:
                pred = self.surrogate(z_b, pos_b, valid_b, tid_b)
                loss = F.mse_loss(pred, pkd_b)

            loss.backward()
            opt.step()
            if XLA_AVAILABLE:
                xm.mark_step()

    def _ei_select(self, pool_names: list) -> list:
        unobserved = [n for n in pool_names if n not in set(self.observed_names)]
        if not unobserved:
            return []

        self.surrogate.eval()
        ei_scores = []

        with torch.no_grad():
            for start in range(0, len(unobserved), EI_CHUNK):
                chunk = unobserved[start: start + EI_CHUNK]
                z_b, pos_b, valid_b = _stack_to_device(
                    self.pool_graphs, chunk, self.device)
                tid_b = self._tid_tensor(len(chunk))

                if self.use_bf16:
                    with torch.autocast(
                        device_type="xla" if XLA_AVAILABLE else "cpu",
                        dtype=torch.bfloat16, enabled=True):
                        mu, sigma = self.surrogate.predict(
                            z_b, pos_b, valid_b, tid_b)
                else:
                    mu, sigma = self.surrogate.predict(
                        z_b, pos_b, valid_b, tid_b)

                ei = expected_improvement(mu.float(), sigma.float(), self.best_pkd)
                if XLA_AVAILABLE:
                    xm.mark_step()

                for name, score in zip(chunk, ei.cpu().tolist()):
                    ei_scores.append((name, score))

        ei_scores.sort(key=lambda x: -x[1])
        return [n for n, _ in ei_scores[:K_BATCH]]

    def run(self, pool_names: list) -> dict:
        notify("PHASE_START",
               f"[{self.label}] BO: {N_INIT} init + {N_ROUNDS} rounds × {K_BATCH}",
               data={"pool_size": len(pool_names), "label": self.label})

        for name in random.sample(pool_names, min(N_INIT, len(pool_names))):
            pkd = self._score_oracle(name)
            self.observed_names.append(name)
            self.observed_pkd.append(pkd)
            if pkd > self.best_pkd:
                self.best_pkd = pkd

        for rnd in range(1, N_ROUNDS + 1):
            t0 = time.time()
            self._retrain_surrogate()
            candidates = self._ei_select(pool_names)

            for name in candidates:
                pkd = self._score_oracle(name)
                self.observed_names.append(name)
                self.observed_pkd.append(pkd)
                if pkd > self.best_pkd:
                    self.best_pkd = pkd
                    notify("CHECKPOINT",
                           f"[{self.label}] New best rd{rnd}: {name} pKd={pkd:.3f}",
                           data={"round": rnd, "name": name, "pkd": pkd})

            elapsed = time.time() - t0
            self.history.append({"round": rnd, "best_pkd": self.best_pkd,
                                  "n_observed": len(self.observed_names),
                                  "elapsed_s": elapsed})

            if rnd % 5 == 0 or rnd == 1:
                heartbeat(f"Phase6_{self.label}", rnd,
                          {"best_pkd": self.best_pkd,
                           "oracle_calls": len(self.observed_names),
                           "elapsed_s": elapsed})
                print(f"  [{self.label}] Round {rnd:2d}: best={self.best_pkd:.3f} "
                      f"n_obs={len(self.observed_names)} t={elapsed:.1f}s")

        top_idx = sorted(range(len(self.observed_pkd)),
                         key=lambda i: -self.observed_pkd[i])[:20]
        top_candidates = [{"name": self.observed_names[i],
                            "pkd":  self.observed_pkd[i]}
                           for i in top_idx]

        notify("PHASE_COMPLETE",
               f"[{self.label}] BO done. Best: {self.best_pkd:.3f} "
               f"after {len(self.observed_names)} oracle calls",
               data={"top": top_candidates[:5], "label": self.label})

        return {"label": self.label, "best_pkd": self.best_pkd,
                "n_oracle_calls": len(self.observed_names),
                "top_candidates": top_candidates, "history": self.history}


# ── Spearman ρ on val compounds ───────────────────────────────────────────────

def measure_fidelity(surrogate, vina_scores, pool_graphs, target: str,
                     target_id: int, device, use_bf16: bool) -> float:
    """
    Measure Spearman ρ between surrogate pKd predictions and Vina ground truth
    on the pool_graphs compounds. Returns ρ.
    """
    names  = list(pool_graphs.keys())
    preds, truths = [], []

    surrogate.eval()
    with torch.no_grad():
        for start in range(0, len(names), EI_CHUNK):
            chunk = names[start: start + EI_CHUNK]
            z_b, pos_b, valid_b = _stack_to_device(pool_graphs, chunk, device)
            tid_b = torch.full((len(chunk),), target_id,
                               dtype=torch.long, device=device)
            if use_bf16:
                with torch.autocast(
                    device_type="xla" if XLA_AVAILABLE else "cpu",
                    dtype=torch.bfloat16):
                    mu, _ = surrogate.predict(z_b, pos_b, valid_b, tid_b)
            else:
                mu, _ = surrogate.predict(z_b, pos_b, valid_b, tid_b)
            if XLA_AVAILABLE:
                xm.mark_step()
            preds.extend(mu.cpu().float().tolist())
            for n in chunk:
                aff = vina_scores.get(n, {}).get(target, 0.0)
                truths.append(vina_to_pkd(aff))

    rho, _ = spearmanr(preds, truths)
    return float(rho)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=TARGETS)
    args = parser.parse_args()

    random.seed(42)

    out_dir = DATA_DIR / "phase6"
    out_dir.mkdir(exist_ok=True)

    # Download inputs
    vina_path   = DATA_DIR / "vina_scores.json"
    fda_path    = DATA_DIR / "pubchem_fda.json"
    fp32_ckpt   = out_dir  / "fp32_best.pt"
    bf16_ckpt   = out_dir  / "bf16_best.pt"

    for local, gcs in [
        (vina_path, f"{GCS_BASE}/vina_scores.json"),
        (fda_path,  f"{GCS_BUCKET}/phase2_setup/pubchem_fda.json"),
        (fp32_ckpt, f"{GCS_BASE}/phase6/fp32_best.pt"),
        (bf16_ckpt, f"{GCS_BASE}/phase6/bf16_best.pt"),
    ]:
        if not local.exists():
            log(f"Downloading {gcs}")
            subprocess.run(["gsutil", "-q", "cp", gcs, str(local)], check=True)

    vina_scores = json.loads(vina_path.read_text())

    # Device
    if XLA_AVAILABLE:
        device = xm.xla_device()
        log(f"Device: {device} (TPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load surrogates
    sur_fp32 = TargetConditionedSurrogate(hidden_dim=256).to(device)
    sur_fp32.load_state_dict(torch.load(fp32_ckpt, map_location="cpu"))
    log("Loaded FP32 surrogate")

    sur_bf16 = TargetConditionedSurrogate(hidden_dim=512).to(device)
    sur_bf16.load_state_dict(torch.load(bf16_ckpt, map_location="cpu"))
    sur_bf16 = sur_bf16.to(torch.bfloat16)
    log("Loaded BF16 surrogate")

    all_results = {}

    for target in args.targets:
        target_id = TARGET2ID[target]
        log(f"\n{'='*60}")
        log(f"  Target: {target}  (id={target_id})")
        log(f"{'='*60}")

        # Build pool (same as Phase 5 — compounds with Vina score < -2 kcal/mol)
        pool_graphs, pool_names = load_pool_graphs(vina_scores, target, device)
        if not pool_graphs:
            log(f"  Skipping {target} — empty pool")
            continue

        oracle = VinaOracle(vina_scores, target)

        # Measure fidelity before BO
        rho_fp32 = measure_fidelity(sur_fp32, vina_scores, pool_graphs,
                                     target, target_id, device, use_bf16=False)
        rho_bf16 = measure_fidelity(sur_bf16, vina_scores, pool_graphs,
                                     target, target_id, device, use_bf16=True)
        log(f"  Pre-BO Spearman ρ: FP32={rho_fp32:.3f}  BF16={rho_bf16:.3f}  "
            f"threshold={FIDELITY_RHO}")

        # FP32 BO
        loop_fp32 = Phase6BayesOptLoop(
            sur_fp32, oracle, pool_graphs, target_id, device,
            label=f"{target}_FP32", use_bf16=False)
        result_fp32 = loop_fp32.run(list(pool_names))

        # BF16 BO (fresh oracle call counter)
        oracle2 = VinaOracle(vina_scores, target)
        loop_bf16 = Phase6BayesOptLoop(
            sur_bf16, oracle2, pool_graphs, target_id, device,
            label=f"{target}_BF16", use_bf16=True)
        result_bf16 = loop_bf16.run(list(pool_names))

        delta    = result_bf16["best_pkd"] - result_fp32["best_pkd"]
        fp32_top = result_fp32["top_candidates"][0]
        bf16_top = result_bf16["top_candidates"][0]
        supported = delta > 0.1 and rho_bf16 >= FIDELITY_RHO

        log(f"\n  {target} RESULT:")
        log(f"    FP32: {fp32_top['name']} pKd={result_fp32['best_pkd']:.3f}  ρ={rho_fp32:.3f}")
        log(f"    BF16: {bf16_top['name']} pKd={result_bf16['best_pkd']:.3f}  ρ={rho_bf16:.3f}")
        log(f"    Δ={delta:+.3f}  hypothesis={'SUPPORTED' if supported else 'REFUTED'}")

        all_results[target] = {
            "fp32": result_fp32,
            "bf16": result_bf16,
            "comparison": {
                "target": target,
                "fp32_best_pkd": result_fp32["best_pkd"],
                "bf16_best_pkd": result_bf16["best_pkd"],
                "fp32_best_compound": fp32_top["name"],
                "bf16_best_compound": bf16_top["name"],
                "fp32_spearman_rho": round(rho_fp32, 4),
                "bf16_spearman_rho": round(rho_bf16, 4),
                "delta_pkd": round(delta, 4),
                "fidelity_threshold": FIDELITY_RHO,
                "fp32_fidelity_pass": rho_fp32 >= FIDELITY_RHO,
                "bf16_fidelity_pass": rho_bf16 >= FIDELITY_RHO,
                "hypothesis_supported": supported,
            },
        }

        # Checkpoint results to GCS after each target
        partial_path = out_dir / "phase6_vina_bo_results.json"
        partial_path.write_text(json.dumps(all_results, indent=2))
        gsutil_cp(partial_path, f"{GCS_BASE}/phase6/phase6_vina_bo_results.json")

    # Final summary
    n_supported = sum(1 for v in all_results.values()
                      if v["comparison"]["hypothesis_supported"])
    log(f"\n{'='*60}")
    log(f"  PHASE 6 BO COMPLETE")
    log(f"  Hypothesis supported: {n_supported}/{len(all_results)}")
    for t, v in all_results.items():
        c = v["comparison"]
        log(f"  {t}: FP32={c['fp32_best_pkd']:.3f}  BF16={c['bf16_best_pkd']:.3f}  "
            f"Δ={c['delta_pkd']:+.3f}  ρ_fp32={c['fp32_spearman_rho']:.3f}  "
            f"ρ_bf16={c['bf16_spearman_rho']:.3f}  "
            f"{'SUPPORTED' if c['hypothesis_supported'] else 'REFUTED'}")
    log("="*60)

    notify("DONE", f"[Phase6 BO] Complete. Hypothesis supported {n_supported}/{len(all_results)}",
           data={"n_supported": n_supported, "n_targets": len(all_results)})


if __name__ == "__main__":
    main()
