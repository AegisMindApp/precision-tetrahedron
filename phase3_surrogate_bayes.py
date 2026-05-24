"""
Phase 3 — Surrogate Bayesian Optimisation (FP32 1× vs BF16 2× surrogate comparison)

Tests the FlashOptim unlocked hypothesis:
  A BF16 surrogate GNN (2× wider, same memory budget as FP32 baseline)
  achieves better sample efficiency in BO over molecular space.

Design:
  - Pool: 2,639 FDA-approved drugs (ChEMBL max_phase=4, from GCS cache)
  - Oracle: Phase 2 BindingAffinityGNN (live GNN calls, padded tensor format)
  - Surrogate: SurrogateGNN (same padded format) with Gaussian head (mean + log_var)
  - Acquisition: Expected Improvement
  - 30 rounds × top-5 oracle calls per round per target
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

    # Eager mode is essential — lazy mode bakes step-count scalars into the graph,
    # causing unique HLO per step and infinite recompilation (same fix as Phase 2).
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
from compat import autocast

try:
    from chembl_data import smiles_to_graph
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

OUTPUT_DIR  = Path(os.environ.get("OUTPUT_DIR", "/tmp/flashoptim_results"))
# PHASE2_DATA_DIR is set by tpu_master.sh; DATA_DIR is the QM9 dir — Phase 3 needs Phase 2's dir
DATA_DIR    = Path(os.environ.get("PHASE2_DATA_DIR",
                   os.environ.get("DATA_DIR", "/tmp/phase2_data")))
PHASE2_CKPT = OUTPUT_DIR / "phase2_best.pt"

PA = BindingAffinityGNN.PADDED_ATOMS   # 80 — static atom count, must match oracle


# ── Surrogate model ───────────────────────────────────────────────────────────

class SurrogateGNN(nn.Module):
    """
    Gaussian surrogate: outputs (mean, log_var) for pKd.

    Uses the same padded-batch format as BindingAffinityGNN so XLA compiles
    both oracle and surrogate forward passes with identical graph shapes.
    Pre-computes the fully-connected edge structure once at __init__.
    """
    PADDED_ATOMS = PA

    def __init__(self, hidden_dim: int = 512, num_blocks: int = 6):
        super().__init__()
        self.gnn = MolecularGNN(hidden_dim=hidden_dim, num_blocks=num_blocks, cutoff=5.0)
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.logv_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 1))

        # Pre-compute fully-connected edge structure (same as BindingAffinityGNN).
        # Registered as buffers so they move to device with .to(device).
        src_list = [a for a in range(PA) for b in range(PA) if a != b]
        dst_list = [b for a in range(PA) for b in range(PA) if a != b]
        e_src       = torch.tensor(src_list, dtype=torch.long)
        e_dst       = torch.tensor(dst_list, dtype=torch.long)
        assign_base = (e_dst.unsqueeze(-1) == torch.arange(PA)).float()
        self.register_buffer('_edge_src',    e_src)       # [MAX_EDGES]
        self.register_buffer('_edge_dst',    e_dst)
        self.register_buffer('_assign_base', assign_base) # [MAX_EDGES, PA]

    def _embed(self, z_pad, pos_pad, atom_valid):
        """Return pooled graph embedding [B, hidden_dim] using gnn.embed()."""
        B = z_pad.shape[0]
        edge_src   = self._edge_src.unsqueeze(0).expand(B, -1)
        edge_dst   = self._edge_dst.unsqueeze(0).expand(B, -1)
        assign_mat = self._assign_base.unsqueeze(0).expand(B, -1, -1)
        src_valid  = atom_valid.gather(1, edge_src)
        dst_valid  = atom_valid.gather(1, edge_dst)
        edge_valid = src_valid & dst_valid
        return self.gnn.embed(z_pad, pos_pad, edge_src, edge_dst,
                              assign_mat, B, edge_valid, atom_valid)

    def forward(self, z_pad, pos_pad, atom_valid):
        """
        z_pad      [B, PA]      long
        pos_pad    [B, PA, 3]   float
        atom_valid [B, PA]      bool
        returns (mean [B], log_var [B])
        """
        h = self._embed(z_pad, pos_pad, atom_valid)
        return self.mean_head(h).squeeze(-1), self.logv_head(h).squeeze(-1)

    def predict(self, z_pad, pos_pad, atom_valid):
        """Returns (mean [B], std [B])."""
        mu, lv = self(z_pad, pos_pad, atom_valid)
        return mu, torch.exp(0.5 * lv)


# ── Acquisition function ──────────────────────────────────────────────────────

def expected_improvement(mu: torch.Tensor, sigma: torch.Tensor,
                          best: float, xi: float = 0.01) -> torch.Tensor:
    z   = (mu - best - xi) / (sigma + 1e-8)
    phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2)))
    pdf = torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    return torch.clamp((mu - best - xi) * phi + sigma * pdf, min=0.0)


# ── Graph building ─────────────────────────────────────────────────────────────

def smiles_to_padded_cpu(smiles: str):
    """
    Convert SMILES → padded CPU tensors (z_pad[1,PA], pos_pad[1,PA,3], valid[1,PA]).
    Returns None if RDKit unavailable or SMILES invalid.
    """
    if not RDKIT_OK:
        return None
    result = smiles_to_graph(smiles)
    if result is None:
        return None
    z_raw, pos_raw = result
    n = min(len(z_raw), PA)
    z_pad   = torch.zeros(1, PA, dtype=torch.long)
    pos_pad = torch.zeros(1, PA, 3, dtype=torch.float)
    valid   = torch.zeros(1, PA, dtype=torch.bool)
    z_pad[0,   :n] = z_raw[:n]
    pos_pad[0, :n] = pos_raw[:n]
    valid[0,   :n] = True
    return z_pad, pos_pad, valid  # all on CPU


def build_pool_graphs(compounds: list) -> dict:
    """
    Pre-build padded graph tensors for all pool compounds (CPU).
    Returns {name: (z_pad[1,PA], pos_pad[1,PA,3], valid[1,PA])}.
    """
    pool = {}
    n_fail = 0
    for i, (name, smiles) in enumerate(compounds):
        result = smiles_to_padded_cpu(smiles) if smiles else None
        if result is not None:
            pool[name] = result
        else:
            n_fail += 1
        if (i + 1) % 300 == 0:
            print(f"  Pool build: {i+1}/{len(compounds)} — {len(pool)} valid, {n_fail} failed")
    print(f"  Pool built: {len(pool)}/{len(compounds)} valid graphs")
    return pool


def _stack_to_device(pool_graphs: dict, names: list, device) -> tuple:
    """Stack named graph tensors into a single batch on device."""
    zs, poss, valids = [], [], []
    for name in names:
        z_pad, pos_pad, valid = pool_graphs[name]
        zs.append(z_pad)        # [1, PA]
        poss.append(pos_pad)    # [1, PA, 3]
        valids.append(valid)    # [1, PA]
    return (torch.cat(zs,     dim=0).to(device),
            torch.cat(poss,   dim=0).to(device),
            torch.cat(valids, dim=0).to(device))


# ── Bayesian optimisation loop ─────────────────────────────────────────────────

class BayesOptLoop:
    """
    BO loop with a GNN surrogate and Phase 2 GNN oracle.

    Comparison: FP32 surrogate (hidden_dim=256) vs BF16 surrogate (hidden_dim=512)
    to test whether 2× capacity at equal memory unlocks better sample efficiency.
    """
    N_INIT       = 50   # initial random oracle queries
    N_ROUNDS     = 30
    K_BATCH      = 5    # oracle calls per round
    RETRAIN_EPOCHS  = 30
    RETRAIN_BATCH   = 32   # fixed batch size — keeps XLA graph shapes static
    EI_CHUNK     = 64   # EI scoring batch size

    def __init__(self, surrogate: SurrogateGNN, oracle: BindingAffinityGNN,
                 pool_graphs: dict, device, label: str, use_bf16: bool = False):
        self.surrogate   = surrogate.to(device)
        self.oracle      = oracle
        self.pool_graphs = pool_graphs
        self.device      = device
        self.label       = label
        self.use_bf16    = use_bf16

        self.observed_names: list = []
        self.observed_pkd:   list = []
        self.best_pkd = -float("inf")
        self.history  = []

    def _score_oracle(self, name: str) -> float:
        """Score a single compound with the Phase 2 GNN oracle."""
        z_pad, pos_pad, valid = self.pool_graphs[name]
        z_pad   = z_pad.to(self.device)
        pos_pad = pos_pad.to(self.device)
        valid   = valid.to(self.device)
        with torch.no_grad():
            with autocast(self.device):
                pred = self.oracle(z_pad, pos_pad, valid)
            if XLA_AVAILABLE:
                xm.mark_step()
        return float(pred.item())

    def _retrain_surrogate(self):
        """Retrain surrogate on all observed (mol, pKd) pairs."""
        n_obs = len(self.observed_names)
        if n_obs < 5:
            return
        opt = torch.optim.AdamW(self.surrogate.parameters(), lr=3e-4)
        self.surrogate.train()

        pkd_arr = torch.tensor(self.observed_pkd, dtype=torch.float)

        for epoch in range(self.RETRAIN_EPOCHS):
            # Fixed batch size — use sampling with replacement if needed
            if n_obs >= self.RETRAIN_BATCH:
                idx = random.sample(range(n_obs), self.RETRAIN_BATCH)
            else:
                idx = random.choices(range(n_obs), k=self.RETRAIN_BATCH)

            names_b   = [self.observed_names[i] for i in idx]
            targets_b = pkd_arr[idx].to(self.device)

            z_b, pos_b, valid_b = _stack_to_device(self.pool_graphs, names_b, self.device)

            opt.zero_grad(set_to_none=False)   # in-place zero keeps tensor IDs stable
            if self.use_bf16:
                with autocast(self.device):
                    mu, lv = self.surrogate(z_b, pos_b, valid_b)
            else:
                mu, lv = self.surrogate(z_b, pos_b, valid_b)

            nll  = 0.5 * (lv + (mu - targets_b) ** 2 / (torch.exp(lv) + 1e-8))
            loss = nll.mean()
            loss.backward()

            if XLA_AVAILABLE:
                xm.optimizer_step(opt)
            else:
                opt.step()

        self.surrogate.eval()
        if XLA_AVAILABLE:
            xm.mark_step()

    def _ei_select(self, pool_names: list) -> list:
        """Score all unobserved pool molecules by EI; return top K_BATCH."""
        observed_set = set(self.observed_names)
        unobserved   = [n for n in pool_names if n not in observed_set]
        if not unobserved:
            return []

        ei_scores = []
        self.surrogate.eval()

        with torch.no_grad():
            for start in range(0, len(unobserved), self.EI_CHUNK):
                chunk = unobserved[start: start + self.EI_CHUNK]
                z_b, pos_b, valid_b = _stack_to_device(self.pool_graphs, chunk, self.device)

                if self.use_bf16:
                    with autocast(self.device):
                        mu, sigma = self.surrogate.predict(z_b, pos_b, valid_b)
                else:
                    mu, sigma = self.surrogate.predict(z_b, pos_b, valid_b)

                ei = expected_improvement(mu, sigma, self.best_pkd)
                if XLA_AVAILABLE:
                    xm.mark_step()

                for name, score in zip(chunk, ei.cpu().tolist()):
                    ei_scores.append((name, score))

        ei_scores.sort(key=lambda x: -x[1])
        return [m for m, _ in ei_scores[:self.K_BATCH]]

    def run(self, pool_names: list) -> dict:
        notify("PHASE_START",
               f"[{self.label}] BO: {self.N_INIT} init + {self.N_ROUNDS} rounds × {self.K_BATCH}",
               data={"pool_size": len(pool_names), "label": self.label})

        # Initial random oracle evaluations
        init_pool = random.sample(pool_names, min(self.N_INIT, len(pool_names)))
        for name in init_pool:
            pkd = self._score_oracle(name)
            self.observed_names.append(name)
            self.observed_pkd.append(pkd)
            if pkd > self.best_pkd:
                self.best_pkd = pkd

        notify("CHECKPOINT",
               f"[{self.label}] Init done. Best pKd: {self.best_pkd:.3f}",
               data={"n_obs": len(self.observed_names), "best_pkd": self.best_pkd})

        for rnd in range(1, self.N_ROUNDS + 1):
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
            self.history.append({
                "round": rnd, "best_pkd": self.best_pkd,
                "n_observed": len(self.observed_names), "elapsed_s": elapsed,
            })

            if rnd % 5 == 0 or rnd == 1:
                heartbeat(f"Phase3_{self.label}", rnd,
                          {"best_pkd": self.best_pkd,
                           "oracle_calls": len(self.observed_names),
                           "elapsed_s": elapsed})
                print(f"  [{self.label}] Round {rnd:2d}: best={self.best_pkd:.3f} "
                      f"n_obs={len(self.observed_names)} t={elapsed:.1f}s")

        top_idx = sorted(range(len(self.observed_pkd)),
                         key=lambda i: -self.observed_pkd[i])[:20]
        top_candidates = [
            {"name": self.observed_names[i], "pkd": self.observed_pkd[i]}
            for i in top_idx
        ]

        notify("PHASE_COMPLETE",
               f"[{self.label}] BO done. Best: {self.best_pkd:.3f} "
               f"after {len(self.observed_names)} oracle calls",
               data={"top": top_candidates[:5], "label": self.label})

        return {
            "label":          self.label,
            "best_pkd":       self.best_pkd,
            "n_oracle_calls": len(self.observed_names),
            "top_candidates": top_candidates,
            "history":        self.history,
        }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+",
                        default=["LINGO1", "PCSK9", "KPC3", "APEX1", "MSH3", "CREBBP"])
    parser.add_argument("--pool-size", type=int, default=5000)
    parser.add_argument("--rounds",    type=int, default=30)
    args = parser.parse_args()

    # ── Device setup ───────────────────────────────────────────────────────────
    if XLA_AVAILABLE:
        device = xm.xla_device()
        notify("PHASE_START", f"Phase 3 on TPU: {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        notify("PHASE_START", "Phase 3 on CUDA")
    else:
        device = torch.device("cpu")
        notify("PHASE_START", "Phase 3 on CPU")

    # ── Load Phase 2 oracle ────────────────────────────────────────────────────
    # Oracle: BindingAffinityGNN(num_blocks=6) with Phase 2 fine-tuned weights.
    # num_blocks=6 matches the condition_B training config — NOT the old num_blocks=4 typo.
    ckpt_path = PHASE2_CKPT
    if ckpt_path.exists():
        ckpt  = torch.load(ckpt_path, map_location="cpu")
        base  = MolecularGNN(hidden_dim=256, num_blocks=6, cutoff=5.0)   # ← was 4, now 6
        oracle = BindingAffinityGNN(base, hidden_dim=256)
        oracle.load_state_dict(ckpt["model_state"])
        oracle = oracle.to(device).eval()
        val_rmse = ckpt.get("val_rmse", "?")
        notify("CHECKPOINT", f"Phase 2 oracle loaded (val_rmse={val_rmse})")
    else:
        notify("ANOMALY", f"Phase 2 checkpoint missing: {ckpt_path}. "
               "Oracle will return random scores — pipeline smoke-test only.")
        base  = MolecularGNN(hidden_dim=256, num_blocks=6, cutoff=5.0)
        oracle = BindingAffinityGNN(base, hidden_dim=256).to(device).eval()

    # ── Load compound pool ─────────────────────────────────────────────────────
    fda_cache = DATA_DIR / "pubchem_fda.json"
    # GCS fallback: if cache is missing locally, try to restore from GCS
    if not fda_cache.exists():
        gcs_bucket = os.environ.get("GCS_BUCKET", "gs://aegismind-tpu-results")
        gcs_src = f"{gcs_bucket}/phase2_setup/pubchem_fda.json"
        fda_cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(["gsutil", "cp", gcs_src, str(fda_cache)],
                           check=True, capture_output=True, timeout=60)
            notify("CHECKPOINT", f"FDA cache restored from GCS: {gcs_src}")
        except Exception as _e:
            notify("ANOMALY", f"GCS restore failed ({_e}): {gcs_src}")

    if fda_cache.exists():
        with open(fda_cache) as f:
            compounds = [tuple(x) for x in json.load(f)]  # [(name, smiles), ...]
        notify("CHECKPOINT", f"FDA compound library: {len(compounds)} compounds")
    else:
        notify("ANOMALY", f"FDA cache missing: {fda_cache}. Using mock pool.")
        # Fallback: small mock pool so the pipeline doesn't silently die
        compounds = [(f"MOCK{i:04d}", "C1CCCCC1") for i in range(200)]

    # Limit pool to --pool-size
    if len(compounds) > args.pool_size:
        random.seed(42)
        compounds = random.sample(compounds, args.pool_size)

    # ── Pre-build all pool graphs (once, on CPU) ───────────────────────────────
    if RDKIT_OK:
        notify("PHASE_START", f"Building graphs for {len(compounds)} compounds ...")
        pool_graphs = build_pool_graphs(compounds)
    else:
        notify("ANOMALY", "RDKit not available — using deterministic random graphs")
        pool_graphs = {}
        rng = torch.Generator()
        for name, smiles in compounds:
            rng.manual_seed(hash(smiles) % (2 ** 31))
            n = torch.randint(10, 35, (1,), generator=rng).item()
            z_pad   = torch.zeros(1, PA, dtype=torch.long)
            pos_pad = torch.zeros(1, PA, 3, dtype=torch.float)
            valid   = torch.zeros(1, PA, dtype=torch.bool)
            z_pad[0, :n]   = torch.randint(0, 9, (n,), generator=rng)
            pos_pad[0, :n] = torch.randn(n, 3, generator=rng)
            valid[0, :n]   = True
            pool_graphs[name] = (z_pad, pos_pad, valid)

    pool_names = [name for name, _ in compounds if name in pool_graphs]
    notify("CHECKPOINT", f"Pool ready: {len(pool_names)} valid graphs from "
           f"{len(compounds)} compounds")

    # ── Run per-target BO comparison ──────────────────────────────────────────
    all_results = {}

    for target in args.targets:
        notify("PHASE_START", f"BO comparison for target: {target}")

        # FP32 surrogate — hidden_dim=256, same capacity as Condition A
        surrogate_fp32 = SurrogateGNN(hidden_dim=256, num_blocks=6)
        loop_fp32 = BayesOptLoop(
            surrogate_fp32, oracle, pool_graphs, device,
            label=f"{target}_FP32_1x", use_bf16=False)
        loop_fp32.N_ROUNDS = args.rounds
        result_fp32 = loop_fp32.run(list(pool_names))

        # BF16 surrogate — hidden_dim=512, 2× wider, same memory as FP32 256
        surrogate_bf16 = SurrogateGNN(hidden_dim=512, num_blocks=6)
        loop_bf16 = BayesOptLoop(
            surrogate_bf16, oracle, pool_graphs, device,
            label=f"{target}_BF16_2x", use_bf16=True)
        loop_bf16.N_ROUNDS = args.rounds
        result_bf16 = loop_bf16.run(list(pool_names))

        improvement = result_bf16["best_pkd"] - result_fp32["best_pkd"]
        comparison = {
            "target":            target,
            "fp32_best_pkd":     result_fp32["best_pkd"],
            "bf16_best_pkd":     result_bf16["best_pkd"],
            "improvement":       improvement,
            "hypothesis_supported": improvement > 0,
        }
        all_results[target] = {
            "fp32":       result_fp32,
            "bf16":       result_bf16,
            "comparison": comparison,
        }

        verdict = "BETTER" if improvement > 0 else "WORSE"
        notify("CHECKPOINT",
               f"{target}: BF16 2× surrogate {verdict} than FP32 1× "
               f"by {improvement:+.3f} pKd",
               data=comparison, urgent=(improvement < -0.5))

    # ── Save & upload results ─────────────────────────────────────────────────
    out = OUTPUT_DIR / "phase3_surrogate_bo_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)

    try:
        subprocess.run(
            ["gsutil", "cp", str(out),
             "gs://aegismind-tpu-results/aegis_flashoptim/phase3_surrogate_bo_results.json"],
            check=False, capture_output=True)
    except Exception:
        pass

    # ── Summary ───────────────────────────────────────────────────────────────
    targets_supported = [t for t, r in all_results.items()
                         if r["comparison"]["hypothesis_supported"]]
    notify("DONE",
           f"Phase 3 complete. Hypothesis supported for "
           f"{len(targets_supported)}/{len(args.targets)} targets: {targets_supported}",
           data={"results_file": str(out),
                 "summary": {t: r["comparison"] for t, r in all_results.items()}})

    print("\n" + "=" * 60)
    print("Phase 3 — Surrogate BO Summary")
    print("=" * 60)
    for target, res in all_results.items():
        c       = res["comparison"]
        verdict = "SUPPORTED" if c["hypothesis_supported"] else "REFUTED"
        print(f"  {target}: FP32={c['fp32_best_pkd']:.3f}  BF16_2x={c['bf16_best_pkd']:.3f}  "
              f"delta={c['improvement']:+.3f}  [{verdict}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
