# Preregistered Protocol Specification: Applied Vision Fine-Tuning Regime (12k Updates)

- **Protocol ID**: `SOTLI-APPLIED-R50-12K-20261107`
- **Target Venue**: Transactions on Machine Learning Research (TMLR) / Applied Cohort Expansion
- **Registration Date**: 7 November 2026
- **Principal Investigator**: Vishesh Panghal
- **Status**: FROZEN & PREREGISTERED PRIOR TO PRIMARY / FACTORIAL RUNS

---

## 1. Study Objective and Framing

This study evaluates whether the "procedural reset vs. data transfer" confound identified in from-scratch training on CIFAR-100 persists in an **applied, pretrained-encoder fine-tuning regime** representative of real-world computer vision deployments.

The primary research question is:
> When an ImageNet-pretrained vision encoder undergoes a temporary off-task detour (Task B) before returning to its primary clinical classification task (Task A), does any observed terminal accuracy difference stem from data transfer or from the procedural mechanics of the boundary (head substitution and optimizer momentum reset)?

---

## 2. Model Architecture and Hyperparameters

- **Backbone**: ResNet-50 initialized from official torchvision `IMAGENET1K_V2` pretrained weights.
- **Input Dimensions**: 224 x 224 x 3 RGB.
- **Augmentation**: `RandomResizedCrop(224, scale=(0.7, 1.0))`, `RandomHorizontalFlip()`.
- **Normalization**: Standard ImageNet channel statistics (`mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`).
- **Classification Heads**:
  - Head A: `Linear(2048, 8)` initialized with Kaiming normal weights and 0 bias.
  - Head B (Auxiliary): `Linear(2048, 10)` initialized fresh at Boundary 1.
  - Head Temp (Reset Control): `Linear(2048, 8)` initialized fresh at Boundary 1.
- **Optimizer**: SGD, momentum 0.9, weight decay 1e-4, batch size 64.
- **Learning Rate Schedule**: Base learning rate 0.01 with a single 500-step linear warmup starting at step 1 for all arms. Constant 0.01 thereafter; no per-arm tuning or boundary schedule adjustments.
- **Mixed Precision**: PyTorch Automatic Mixed Precision (`torch.amp.autocast('cuda')` with `GradScaler`).
- **BatchNorm Policy**: BatchNorm running mean and variance statistics are preserved across all phase boundaries without reset.

---

## 3. Update Budget & Phase Structure (12,000 Updates Total)

| Phase | Updates | Interval | Content |
|---|---|---|---|
| **Phase 1** (Primary Task A Warmup) | u1 = 5,000 | 1 to 5,000 | Target Task A (ISIC 2019, 8 classes), active Head A |
| **Phase 2** (Interval Task) | uB = 2,000 | 5,001 to 7,000 | Arm-dependent (see Experimental Arms) |
| **Phase 3** (Primary Task A Relearning) | u2 = 5,000 | 7,001 to 12,000 | Target Task A (ISIC 2019), restored Head A |

- **Milestone Checkpoints**:
  - $\theta_1$: step 5,000 (end of Phase 1)
  - $\theta_2$: step 7,000 (end of Phase 2)
  - $\theta_{\text{equal-target}}$: step 10,000 on Condition C0 (equal target exposure: u1 + u2 = 10,000 updates on Task A)
  - $\theta_3$: step 12,000 (terminal evaluation)

---

## 4. Experimental Arms

| Condition ID | Description | Phase 2 Data | Phase 2 Head | Momentum at Boundaries |
|---|---|---|---|---|
| **C0** | Continuous Baseline | Task A (ISIC 2019) | Head A (continuous) | Preserved |
| **C3** | On-Target Reset Control | Task A (ISIC 2019) | Head Temp (8 classes) | Zeroed at 5k and 7k |
| **S1** | Off-Task Detour | Task B (Imagenette) | Head B (10 classes) | Zeroed at 5k and 7k |
| **S1-shuf** | Shuffled-Label Control | Task B (Label-shuffled) | Head B (10 classes) | Zeroed at 5k and 7k |
| **F00** | Factorial (No Swap, No Reset) | Task A | Head A | Preserved |
| **F01** | Factorial (Head Swap Only) | Task A | Head Temp | Preserved |
| **F10** | Factorial (Momentum Reset Only) | Task A | Head A | Zeroed at 5k and 7k |
| **F11** | Factorial (Both Swap and Reset) | Task A | Head Temp | Zeroed at 5k and 7k |

---

## 5. Preregistered Seed Lists

### 5.1 Pilot Study (N = 2 seeds x 4 conditions = 8 runs)
- `[900001, 900002]`
- Conditions: `C0`, `C3`, `S1`, `S1-shuf`

### 5.2 Primary Confirmatory Study (N = 10 seeds x 3 conditions = 30 runs)
- Seeds: `[910001, 910002, 910003, 910004, 910005, 910006, 910007, 910008, 910009, 910010]`
- Conditions: `C0`, `C3`, `S1`

### 5.3 Factorial Study (N = 6 seeds x 4 conditions = 24 runs)
- Seeds: `[920001, 920002, 920003, 920004, 920005, 920006]`
- Conditions: `F00`, `F01`, `F10`, `F11`

---

## 6. Endpoints, Metrics, and Statistical Decision Rules

### 6.1 Primary Endpoint
- **Metric**: Terminal Top-1 Validation Accuracy (%) on Target Task A at $\theta_3$ (12,000 updates).
- **Primary Hypothesis Test (H-A1)**: Paired contrast $S1 - C0$ at 12,000 updates across 10 paired seeds.
- **Statistical Tests**: Two-sided paired Student's $t$-test ($\alpha = 0.05$), confirmed by exact sign-flip permutation test ($2^{10} = 1024$ permutations) and 10,000-sample bootstrap 95% confidence intervals.

### 6.2 Secondary Endpoints
- **Secondary Contrasts** (controlled for family-wise error rate via Holm step-down correction):
  1. $S1 - C3$: Effect of off-task data semantics isolated from boundary mechanics.
  2. $C3 - C0$: Net penalty of head substitution and momentum zeroing.
  3. $S1(12\text{k}) - C0(10\text{k})$: Equal Target-A exposure contrast.
- **Applied Vision & Clinical Metrics**:
  - Macro one-vs-rest AUC across all 8 classes.
  - Balanced Accuracy (macro mean recall across 8 imbalanced classes).
  - Expected Calibration Error (ECE, 10 equal-width confidence bins).
  - Multiclass Brier Score.
- **Representation Diagnostics**:
  - Effective Representation Rank on centered 2,000-image diagnostic probe set.
  - Linear CKA representation drift versus $\theta_1$.
  - Linear probe AUC on frozen penultimate representations.

---

## 7. Data Provenance & Verification Hashes

| Artifact | File Name | SHA-256 Checksum |
|---|---|---|
| ISIC 2019 Ground Truth | `ISIC_2019_Training_GroundTruth.csv` | `aa88e9638fe4df9ef330dc4ba22fa4bd475c44af692fffb870c73091ab25cdcd` |
| ISIC 2019 Stratified Splits | `isic2019_splits.json` | `5cc940e3d2ef1003188e42edf3b2ca9f14d252e45b92a222ae9724c702f9726a` |
| ISIC 2019 Diagnostic Probe Set | `isic2019_probe_2000.json` | `1d7b1d69ecea80adeb4823f89d7ce248298b00c14f9d29b3f12a38d203b312d3` |
| Imagenette Stratified Splits | `imagenette_splits.json` | `d539a406ea81da45c068015403edc556d62c9dc34444c0a5d6fdf8a49036d9c3` |

---

## 8. Compute and Preflight Benchmarks

- **Benchmarked Execution Platform**: NVIDIA Quadro RTX 8000 (48 GB VRAM), PyTorch 2.12 + CUDA 13.2.
- **Measured Empirical Latency**: 209.0 ms per update (batch size 64 with AMP).
- **Measured Memory Peak**: 4,256 MiB VRAM.
- **Per-Run 12k Runtime**: 41.8 minutes (0.70 GPU hours).
- **Smoke Test Verification**: All primary conditions (`C0`, `C3`, `S1`) verified clean forward/backward execution and metric logging on 7 November 2026.
