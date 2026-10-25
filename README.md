# Structured Off-Task Learning Intervals (SOTLI): Experimental Reproducibility Kit

This repository provides the official source code, configuration files, dataset split manifests, experimental logs, and statistical verification tools for the empirical study on Structured Off-Task Learning Intervals (SOTLI) in deep image classification.

---

## 1. Overview

The study evaluates whether temporary off-task auxiliary learning intervals (e.g., STL-10 transfer intermediate between initial and relearning phases on CIFAR-100) confer downstream generalization benefits under matched computational budgets and target exposures.

Key evaluated conditions:
- **C0 (Continuous Target):** Continuous CIFAR-100 training for 35,000 updates.
- **C1 (Momentum Reset Control):** On-target CIFAR-100 training with optimizer momentum zeroed at boundary updates (15,000 and 20,000).
- **C2 (Head Substitution Control):** On-target CIFAR-100 training with temporary head substitution during Phase 2; momentum carried through continuously.
- **C3 (Bundled Procedural Control):** On-target CIFAR-100 training with both temporary head substitution and momentum clearing.
- **S1 (Auxiliary STL-10 Transfer):** Auxiliary detour on STL-10 during Phase 2 (15,000 to 20,000 updates) with temporary head and momentum clearing.

---

## 2. Repository Structure

```
├── primary.csv                 # Terminal accuracies across 20 paired seeds (C0, C3, S1)
├── factorial.csv               # Seed-level accuracies across 6 seeds for 2x2 factorial arms (F00, F01, F10, F11)
├── paa_verify_stats.py         # Standalone statistical auditor for paired tests and factorial ANOVA
├── verification_results.json   # Machine-readable output of statistical verification
├── requirements.txt            # Python environment dependencies
├── src/                        # Core implementation
│   ├── data/                   # Data loaders, deterministic splitters, paired batch samplers
│   ├── models/                 # GroupNorm ResNet-18 and Compact Vision Transformer (CViT)
│   ├── training/               # Multi-phase training loops, checkpoint transitions, evaluators
│   └── utils/                  # Rank, CKA, and representation metrics
├── scripts/                    # Experiment execution and analysis runners
│   ├── run_confirmatory.py     # Confirmatory runner
│   ├── run_factorial.py        # 2x2 factorial ablation runner
│   ├── run_screening.py        # Multi-domain screening spectrum runner
│   └── run_vit_pilot.py        # Vision Transformer cross-architecture runner
├── PAA/scripts/                # Figure and table generation scripts
│   ├── generate_paired_contrasts_forest_figure.py
│   ├── generate_longitudinal_dynamics_figure.py
│   ├── generate_monotonic_spectrum_figure.py
│   └── generate_paa_tables_and_stats.py
├── configs/                    # Protocol and locked hyperparameter configurations
├── data/
│   └── split_manifest.json     # Deterministic CIFAR-100 45k/5k split indices and SHA-256 hashes
└── results/                    # Curated seed-level results and contrast statistics
```

---

## 3. Quick Start & Statistical Verification

To verify the manuscript's primary paired comparisons and 2x2 factorial decomposition without needing GPUs:

### Environment Setup
```bash
pip install -r requirements.txt
```

### Run the Statistical Auditor
```bash
python3 paa_verify_stats.py \
  --primary primary.csv \
  --factorial factorial.csv \
  --output verification_results.json
```

**Expected Verification Outputs:**
- **Primary S1 - C0 Contrast (N = 20):** Mean Difference = -0.4325 pp, Paired t(19) = -1.7873, p = 0.0898, Exact Sign-Flip p = 0.0906 (95,034 / 1,048,576 permutations).
- **Factorial 2x2 Decomposition (N = 6):**
  - Head Substitution Main Effect: -2.1008 pp, t = -4.9947, p = 0.0041
  - Momentum Clearing Main Effect: +0.5692 pp, t = 1.2142, p = 0.2789
  - Head x Momentum Interaction: +0.0217 pp, t = 0.0440, p = 0.9666

---

## 4. Reproducing Figures

Figure scripts generate publication-grade PDF and 300 DPI PNG plots:

```bash
# Forest Plot & Exact Permutation Null (Figure 2)
python3 PAA/scripts/generate_paired_contrasts_forest_figure.py

# Longitudinal Representation Dynamics (Figure 3)
python3 PAA/scripts/generate_longitudinal_dynamics_figure.py

# Multi-Domain Screening Spectrum (Figure 4)
python3 PAA/scripts/generate_monotonic_spectrum_figure.py
```

Outputs are saved to `PAA/figures/`.

---

## 5. Dataset Splitting & Integrity

All experiments use deterministic, non-overlapping splits:
- **CIFAR-100:** 45,000 training images and 5,000 validation images partitioned using `numpy.random.RandomState(271828)`.
- **Representation Diagnostics:** Unlabeled 2,000-image geometry subsets sampled with `RandomState(271829)` from validation inputs (`test_access: "not constructed; geometry uses validation inputs"`).
- Exact index arrays and file hashes are cataloged in `data/split_manifest.json`.

---

## 6. Training from Scratch

To launch experimental arms on GPU (CUDA enabled):

```bash
# Execute 2x2 Factorial Study
python3 scripts/run_factorial.py --config configs/focused_p2br_13h_released_20261017.json

# Execute Confirmatory Seeds
python3 scripts/run_confirmatory.py --device cuda
```
