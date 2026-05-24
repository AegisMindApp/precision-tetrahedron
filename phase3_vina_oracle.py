#!/usr/bin/env python3
"""
phase3_vina_oracle.py
---------------------
Phase 3 re-run with a protein-aware oracle: AutoDock Vina scores (kcal/mol)
from a pre-computed lookup table replace the ligand-only GNN oracle.

Key differences from phase3_surrogate_bayes.py:
  - Oracle: Vina affinity (kcal/mol, lower = better binding) from vina_scores.json
    converted to pKd via ΔG = -RT ln(Kd) → pKd = ΔG / (RT·log10(e))
    Using RT = 0.592 kcal/mol at 298K, log10(e) = 0.4343:
    pKd = -affinity / (0.592 × 2.303) ≈ -affinity / 1.364
  - Oracle calls are instant table lookups — no GPU/TPU needed for oracle
  - Surrogate GNN (BF16 2×, hidden_dim=512) still trains on TPU
  - Target-specific: each target has its own Vina score column → different rankings
  - Physically calibrated: Vina scores map to pKd 3–9 range for typical drugs

Design:
  - Pool: all compounds with valid Vina scores for a given target (≤ 2,639)
  - Oracle: vina_affinity_to_pkd(vina_scores[compound][target])
  - Surrogate: same SurrogateGNN as Phase 3 (BF16 512-dim on TPU)
  - 30 rounds × top-5 per round = 150 oracle calls per target
  - Comparison: FP32 1× surrogate vs BF16 2× surrogate (same oracle)
"""

import os
import sys
import json
import math
import time
import random
import argparse
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
    try:
        import torch_xla.experimental as _xla_exp
        _xla_exp.eager_mode(True)
        print("XLA eager mode: ENABLED")
    except Exception as _e:
        print(f"XLA eager mode unavailable: {_e}")
except ImportError:
    XLA_AVAILABLE = False

from notify import notify, heartbeat
from model import MolecularGNN
from phase2_pdbbind import BindingAffinityGNN
from phase3_surrogate_bayes import SurrogateGNN, BayesOptLoop, expected_improvement
from compat import autocast

try:
    from chembl_data import smiles_to_graph
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

OUTPUT_DIR   = Path(os.environ.get("OUTPUT_DIR", "/tmp/flashoptim_results"))
DATA_DIR     = Path(os.environ.get("PHASE2_DATA_DIR",
                    os.environ.get("DATA_DIR", "/tmp/phase2_data")))
PHASE2_CKPT  = OUTPUT_DIR / "phase2_best.pt"
GCS_BUCKET   = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
RUN_ID       = os.environ.get("RUN_ID", "aegis_flashoptim")
VINA_SCORES  = OUTPUT_DIR / "vina_scores.json"

RT_KCAL      = 0.592   # kcal/mol at 298K (RT)
LN10         = 2.3026
# pKd = -ΔG / (RT × ln10)  where ΔG = Vina affinity (negative kcal/mol)
# pKd = -vina_affinity / (0.592 × 2.303) ≈ -vina_affinity / 1.364
# e.g., vina = -7.0 → pKd = 5.13  (Kd ≈ 7.4 nM — excellent)
#        vina = -5.0 → pKd = 3.67  (Kd ≈ 210 nM — moderate)

def vina_to_pkd(affinity_kcal: float) -> float:
    """Convert Vina binding energy (kcal/mol) to pKd. Returns 0 if invalid."""
    if affinity_kcal >= 0:
        return 0.0
    return -affinity_kcal / (RT_KCAL * LN10)


class VinaOracle:
    """
    Wraps the pre-computed Vina score table as a callable oracle.
    Returns pKd (higher = better) for a given compound and target.
    """
    def __init__(self, scores: dict, target: str):
        self.scores = scores      # {compound_name: {target: affinity_kcal}}
        self.target = target
        self._calls = 0

    def __call__(self, compound_name: str) -> float:
        self._calls += 1
        ts = self.scores.get(compound_name, {})
        affinity = ts.get(self.target, 0.0)
        return vina_to_pkd(affinity)

    @property
    def n_calls(self):
        return self._calls


class VinaBayesOptLoop(BayesOptLoop):
    """
    Overrides BayesOptLoop to use VinaOracle instead of the GNN oracle.
    Pool compounds are identified by name; oracle calls are O(1) lookups.
    """
    def __init__(self, surrogate, oracle: VinaOracle, pool_graphs: dict,
                 device, label: str, use_bf16: bool = False):
        # Don't call super().__init__ — set attributes manually
        self.surrogate    = surrogate.to(device)
        self.oracle       = oracle
        self.pool_graphs  = pool_graphs
        self.device       = device
        self.label        = label
        self.use_bf16     = use_bf16
        self.observed_names: list = []
        self.observed_pkd:   list = []
        self.best_pkd     = -float("inf")
        self.history      = []
        self.RETRAIN_BATCH  = 32
        self.RETRAIN_EPOCHS = 20
        self.K_BATCH        = 5
        self.N_INIT         = 10
        self.N_ROUNDS       = 30

    def _score_oracle(self, name: str) -> float:
        return self.oracle(name)


def load_pool_graphs(vina_scores: dict, target: str, device):
    """
    Build padded-tensor pool from compounds that have valid Vina scores for target.
    Same format as phase3_surrogate_bayes.py but keyed by compound name.
    """
    PA = BindingAffinityGNN.PADDED_ATOMS

    # Filter to compounds with meaningful Vina scores for this target
    valid = {
        name: scores[target]
        for name, scores in vina_scores.items()
        if target in scores and scores[target] < -2.0   # only real binders (< -2 kcal/mol)
    }
    log(f"  [{target}] Pool: {len(valid)} compounds with Vina score < -2.0 kcal/mol "
        f"(of {len(vina_scores)} total)")

    if not valid:
        log(f"  WARNING: No valid compounds for {target}")
        return {}, []

    # Build padded graphs using SMILES from FDA cache
    fda_cache = DATA_DIR / "pubchem_fda.json"
    compounds_list = json.loads(fda_cache.read_text()) if fda_cache.exists() else []
    smiles_map = {}
    for c in compounds_list:
        if isinstance(c, (list, tuple)):
            # pubchem_fda.json stored as [name, smiles] pairs
            name, smiles = (str(c[0]) if len(c) > 0 else ""), (str(c[1]) if len(c) > 1 else "")
        else:
            name   = str(c.get("name") or c.get("iupac_name") or c.get("cid", ""))
            smiles = c.get("smiles") or c.get("canonical_smiles") or c.get("isomeric_smiles", "")
        if name and smiles:
            smiles_map[name] = smiles

    pool_graphs = {}
    rng = torch.Generator()
    rng.manual_seed(42)

    n_real = 0
    for name in valid:
        smiles = smiles_map.get(name, "")
        z_pad   = torch.zeros(1, PA, dtype=torch.long)
        pos_pad = torch.zeros(1, PA, 3, dtype=torch.float32)
        atom_valid = torch.zeros(1, PA, dtype=torch.bool)

        if smiles and RDKIT_OK:
            try:
                result = smiles_to_graph(smiles)
                if result:
                    z, pos = result
                    n = min(len(z), PA)
                    z_pad[0, :n]   = torch.tensor(z[:n], dtype=torch.long).clamp(0, 8)
                    pos_pad[0, :n] = torch.tensor(pos[:n], dtype=torch.float32)
                    atom_valid[0, :n] = True
                    n_real += 1
            except Exception:
                pass

        if not atom_valid[0].any():
            # Random fallback (maintains pool size, surrogate treats as noise)
            n = random.randint(3, min(20, PA))
            z_pad[0, :n]   = torch.randint(1, 9, (n,), generator=rng)
            pos_pad[0, :n] = torch.randn(n, 3, generator=rng) * 2.0
            atom_valid[0, :n] = True

        pool_graphs[name] = (
            z_pad.to(device), pos_pad.to(device), atom_valid.to(device)
        )

    pool_names = list(pool_graphs.keys())
    log(f"  [{target}] Graphs: {len(pool_names)} ({n_real} real RDKit, "
        f"{len(pool_names)-n_real} random fallback)")
    return pool_graphs, pool_names


def log(msg):
    print(f"[PHASE3-VINA] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets",   nargs="+",
                   default=["LINGO1", "PCSK9", "KPC3", "APEX1", "MSH3", "CREBBP"])
    p.add_argument("--rounds",    type=int, default=30)
    p.add_argument("--pool-size", type=int, default=None,
                   help="Max pool size per target (default: all valid Vina hits)")
    args = p.parse_args()

    # ── Load Vina scores ────────────────────────────────────────────────────
    if not VINA_SCORES.exists():
        log(f"Fetching vina_scores.json from GCS...")
        subprocess.run(
            ["gsutil", "cp",
             f"{GCS_BUCKET}/{RUN_ID}/vina_scores.json", str(VINA_SCORES)],
            check=False, capture_output=True
        )
    if not VINA_SCORES.exists():
        log("ERROR: vina_scores.json not found. Run vina_screen.py first.")
        sys.exit(1)

    vina_scores = json.loads(VINA_SCORES.read_text())
    log(f"Loaded Vina scores: {len(vina_scores)} compounds")

    # ── Device ──────────────────────────────────────────────────────────────
    if XLA_AVAILABLE:
        device = xm.xla_device()
        log(f"Device: {device} (TPU)")
    else:
        device = torch.device("cpu")
        log("Device: CPU (no XLA)")

    all_results = {}

    for target in args.targets:
        notify("PHASE_START", f"[Vina BO] {target}")
        log(f"\n{'='*60}\nTarget: {target}\n{'='*60}")

        oracle = VinaOracle(vina_scores, target)

        pool_graphs, pool_names = load_pool_graphs(vina_scores, target, device)
        if not pool_names:
            log(f"  Skipping {target} — no valid compounds")
            continue

        if args.pool_size and len(pool_names) > args.pool_size:
            pool_names = pool_names[:args.pool_size]
            pool_graphs = {n: pool_graphs[n] for n in pool_names}

        # Print Vina score distribution for this target
        target_affinities = sorted(
            [vina_scores[c][target] for c in pool_names if target in vina_scores.get(c, {})],
        )
        if target_affinities:
            log(f"  Vina range: {target_affinities[0]:.2f} to {target_affinities[-1]:.2f} kcal/mol")
            log(f"  pKd range:  {vina_to_pkd(target_affinities[0]):.2f} to "
                f"{vina_to_pkd(target_affinities[-1]):.2f}")

        # ── FP32 surrogate (256-dim) ──────────────────────────────────────
        surr_fp32 = SurrogateGNN(hidden_dim=256, num_blocks=6)
        loop_fp32 = VinaBayesOptLoop(
            surr_fp32, oracle, pool_graphs, device,
            label=f"{target}_FP32_1x", use_bf16=False)
        loop_fp32.N_ROUNDS = args.rounds
        result_fp32 = loop_fp32.run(list(pool_names))

        # ── BF16 surrogate (512-dim) ──────────────────────────────────────
        oracle2 = VinaOracle(vina_scores, target)   # fresh call counter
        surr_bf16 = SurrogateGNN(hidden_dim=512, num_blocks=6)
        loop_bf16 = VinaBayesOptLoop(
            surr_bf16, oracle2, pool_graphs, device,
            label=f"{target}_BF16_2x", use_bf16=True)
        loop_bf16.N_ROUNDS = args.rounds
        result_bf16 = loop_bf16.run(list(pool_names))

        # ── Surrogate fidelity: Spearman ρ vs Vina ground truth ──────────────
        # Score entire pool with each final surrogate and correlate against
        # Vina lookup scores — measures how well the surrogate learned the
        # fitness landscape (success criterion: ρ ≥ 0.70).
        def surrogate_fidelity(loop, pool_names, vina_scores, target, device):
            """Compute Spearman ρ between surrogate pKd and Vina pKd over pool."""
            surr = loop.surrogate.eval()
            vina_pkds, surr_pkds = [], []
            with torch.no_grad():
                for name in pool_names:
                    v_pkd = vina_to_pkd(vina_scores.get(name, {}).get(target, 0.0))
                    if v_pkd <= 0:
                        continue
                    z_pad, pos_pad, valid = pool_graphs[name]
                    try:
                        mu, _ = surr(z_pad, pos_pad, valid)
                        surr_pkds.append(float(mu.mean().cpu()))
                        vina_pkds.append(v_pkd)
                    except Exception:
                        pass
            if len(vina_pkds) < 10:
                return float("nan")
            # Spearman ρ via rank correlation
            n = len(vina_pkds)
            def rank(lst):
                s = sorted(range(n), key=lambda i: lst[i])
                r = [0] * n
                for rank_i, orig_i in enumerate(s):
                    r[orig_i] = rank_i
                return r
            rv = rank(vina_pkds)
            rs = rank(surr_pkds)
            mean_rv = sum(rv) / n
            mean_rs = sum(rs) / n
            num = sum((rv[i]-mean_rv)*(rs[i]-mean_rs) for i in range(n))
            dv  = (sum((rv[i]-mean_rv)**2 for i in range(n)) ** 0.5)
            ds  = (sum((rs[i]-mean_rs)**2 for i in range(n)) ** 0.5)
            return num / (dv * ds) if dv * ds > 0 else float("nan")

        rho_fp32 = surrogate_fidelity(loop_fp32, pool_names, vina_scores, target, device)
        rho_bf16 = surrogate_fidelity(loop_bf16, pool_names, vina_scores, target, device)
        log(f"  Surrogate fidelity  FP32 ρ={rho_fp32:.3f}  BF16 ρ={rho_bf16:.3f}  "
            f"(threshold ρ≥0.70)")

        improvement = result_bf16["best_pkd"] - result_fp32["best_pkd"]
        comparison = {
            "target":               target,
            "fp32_best_pkd":        result_fp32["best_pkd"],
            "bf16_best_pkd":        result_bf16["best_pkd"],
            "fp32_best_vina":       -result_fp32["best_pkd"] * RT_KCAL * LN10,
            "bf16_best_vina":       -result_bf16["best_pkd"] * RT_KCAL * LN10,
            "improvement":          improvement,
            "hypothesis_supported": improvement > 0,
            "fp32_spearman_rho":    round(rho_fp32, 4) if rho_fp32 == rho_fp32 else None,
            "bf16_spearman_rho":    round(rho_bf16, 4) if rho_bf16 == rho_bf16 else None,
            "fidelity_threshold":   0.70,
            "fp32_fidelity_pass":   rho_fp32 >= 0.70 if rho_fp32 == rho_fp32 else None,
            "bf16_fidelity_pass":   rho_bf16 >= 0.70 if rho_bf16 == rho_bf16 else None,
            "oracle":               "vina_1.2",
            "pool_size":            len(pool_names),
        }
        all_results[target] = {
            "fp32":       result_fp32,
            "bf16":       result_bf16,
            "comparison": comparison,
        }

        verdict = "BETTER" if improvement > 0 else "WORSE"
        notify("CHECKPOINT",
               f"[VinaBO] {target}: BF16 {verdict} by {improvement:+.3f} pKd "
               f"(FP32={result_fp32['best_pkd']:.2f} BF16={result_bf16['best_pkd']:.2f})",
               data=comparison)
        log(f"  {target}: FP32={result_fp32['best_pkd']:.3f}  "
            f"BF16={result_bf16['best_pkd']:.3f}  Δ={improvement:+.3f} pKd")

    # ── Save ────────────────────────────────────────────────────────────────
    out = OUTPUT_DIR / "phase3_vina_bo_results.json"
    out.write_text(json.dumps(all_results, indent=2))
    try:
        subprocess.run(
            ["gsutil", "cp", str(out),
             f"{GCS_BUCKET}/{RUN_ID}/phase3_vina_bo_results.json"],
            check=False, capture_output=True
        )
    except Exception:
        pass

    # ── Summary ─────────────────────────────────────────────────────────────
    supported = [t for t, r in all_results.items()
                 if r["comparison"]["hypothesis_supported"]]
    notify("DONE",
           f"[VinaBO] Phase 3 (Vina oracle) complete. "
           f"Hypothesis supported {len(supported)}/{len(all_results)}: {supported}")
    log("\n" + "="*60)
    log("FINAL SUMMARY (Vina oracle, physically calibrated pKd):")
    log("="*60)
    for t, r in all_results.items():
        c = r["comparison"]
        v = "SUPPORTED" if c["hypothesis_supported"] else "REFUTED"
        log(f"  {t}: FP32={c['fp32_best_pkd']:.2f}  BF16={c['bf16_best_pkd']:.2f}  "
            f"Δ={c['improvement']:+.2f} pKd  [{v}]")
        log(f"        (Vina: FP32={c['fp32_best_vina']:.2f}  "
            f"BF16={c['bf16_best_vina']:.2f} kcal/mol)")


if __name__ == "__main__":
    main()
