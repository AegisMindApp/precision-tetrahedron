# The Precision Tetrahedron

**Loss Landscape Topology Across Number Formats and Multi-Target Drug Discovery**

John Goodman — AegisMind Research  
*Supported by the Google TPU Research Cloud (TRC), project `aegismind-tpu`*

---

## Paper

📄 [The Precision Tetrahedron.pdf](./The%20Precision%20Tetrahedron.pdf)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20363636.svg)](https://doi.org/10.5281/zenodo.20363636)

### Abstract

Numerical precision is a systematic variable in neural network loss landscape geometry. We characterise inter-precision Linear Mode Connectivity (LMC) barriers across FP32, BF16, FP16, and INT8 for models from 1M to 124M parameters, identifying two structural findings:

1. **A model-size scaling law**: FP32↔BF16 barrier ∝ params^(−0.85) (R²=0.98), placing ~10M parameters as the practical basin-separator / regulariser boundary for mixed-precision training.
2. **An isosceles precision triangle** extended to an **irregular four-vertex INT8 tetrahedron** — exponent range, not mantissa width, is the operative variable governing basin isolation.

The framework is applied to a multi-target drug discovery pipeline. A target-conditioned surrogate (Spearman ρ=0.827) drives UCB Bayesian Optimisation across six therapeutic targets. The AMR pipeline screens KPC-3 carbapenemase inhibitors; the MSH3 ATPase pipeline (Huntington's disease) identifies PONATINIB, EPTIFIBATIDE, ENTRECTINIB, and IMATINIB as novel binders. Nash equilibrium 2×2 payoff matrices identify EPTIFIBATIDE+laquinimod as a synergistic dual-pathway inhibitor pair (synergy=0.250). Multi-target BO recovers EPTIFIBATIDE as a convergent tri-method candidate independently selected by Vina docking, Nash synergy, and surrogate BO.

All 42 experimental phases ran on Google v6e-8 TPUs.

---

## Repository structure

```
phase*.py          — The 42 experimental phases (TPU training, LMC, docking, BO)
model.py           — MolecularGNN / SurrogateGNN / TargetConditionedSurrogate
train.py           — Core training loop (BF16/FP32 precision, plateau-triggered restart)
data.py            — QM9 and PDBbind data loaders
pdbbind_data.py    — PDBbind Vina docking data pipeline
vina_screen.py     — AutoDock Vina virtual screening pipeline (FDA compound library)
vina_receptor_prep.py  — Receptor preparation for Vina docking
phase3_vina_oracle.py  — Vina oracle for Bayesian Optimisation loop
chembl_data.py     — ChEMBL KPC-3 inhibitor data pipeline
checkpoint_interpolation.py  — LMC barrier measurement (linear weight interpolation)
compat.py          — XLA/PyTorch-XLA compatibility helpers
*.sh               — TPU VM launch and orchestration scripts
```

---

## Reproducing experiments

### Requirements

```bash
pip install torch torch_xla rdkit-pypi scipy numpy pandas requests
# AutoDock Vina 1.2+ required for docking phases
```

### TPU access

Experiments were run on Google v6e-8 TPUs via the [TPU Research Cloud](https://sites.research.google/trc/). Each phase script is self-contained and reads/writes checkpoints to GCS (`gs://aegismind-tpu-results/aegis_flashoptim/`).

### Running a phase

```bash
# Example: Phase 39 (MSH3 rdkit docking + surrogate)
python phase39_msh3_rdkit.py

# Example: Phase 42 (multi-target BO)
python phase42_multi_target_bo.py
```

Phases 1–9 establish the core BF16/FP32 LMC results. Phases 10–33 extend to architectures, scales, and precisions. Phases 34–42 cover the drug discovery pipeline (KPC-3, MSH3, Nash, multi-target BO).

---

## Key results

| Finding | Phase | Section |
|---|---|---|
| BF16 + restart: 0.0215 eV (2.7× over FP32) | 1–4 | §4.1–4.7 |
| 273× LMC barrier ratio (BF16 vs FP32) | 7 | §4.8 |
| Isosceles precision triangle (exponent range drives isolation) | 17, 17b | §4.12 |
| Irregular INT8 tetrahedron (4.344 nat isolation) | 16b | §4.16 |
| Scaling law: barrier ∝ params^(−0.85), R²=0.98 | 31 | §4.28 |
| KPC-3 ChEMBL screen: CHEMBL3931277 pKd=6.19 | AMR | §4.14 |
| MSH3 screen: PONATINIB top hit pKd=6.359, ρ=0.854 | 39 | §4.37 |
| Nash MSH3+PARP: EPTIFIBATIDE+laquinimod synergy=0.250 | 40 | §4.38 |
| Cross-target LMC: KPC-3↔MSH3 barrier = 15.2% of intra-target | 41 | §4.39 |
| Multi-target BO: EPTIFIBATIDE convergent tri-method hit | 42 | §4.40 |

---

## Citation

```bibtex
@article{goodman2026precision,
  title   = {The Precision Tetrahedron: Loss Landscape Topology Across Number Formats and Multi-Target Drug Discovery},
  author  = {Goodman, John},
  year    = {2026},
  doi     = {10.5281/zenodo.20363636},
  url     = {https://doi.org/10.5281/zenodo.20363636},
  note    = {Supported by Google TPU Research Cloud}
}
```

---

## License

MIT License — see [LICENSE](./LICENSE)

## Acknowledgements

Google TPU Research Cloud (TRC) — GCP project `aegismind-tpu`, v6e-8 TPUs across us-east5-b, us-east1-d, us-central1-b, europe-west4-a.
