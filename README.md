# FR-PhyTCA: Fidelity-Residual Physics-Structured Tensor Component Adaptation

This repository implements **FR-PhyTCA**, a parameter-efficient continual-learning framework for evolving materials databases. FR-PhyTCA first learns a **parent** model on a low-cost, abundant fidelity and then freezes its effective route permanently. Each subsequent higher fidelity is learned as a small, structured Tucker residual on top of the frozen parent, so the parent fidelity route is exactly invariant by construction and forgetting is identically zero. The symmetric **PhyTCA** baseline (shared Tucker factors updated for every task) is retained for comparison.

## Project structure

```
Continual Learning/
├── data.py                        # JARVIS loaders, protocols, periodic graph builder
├── data_audit.py                  # Data auditing and GO/NO-GO gate
├── phytca.py                      # Tucker4DAdapter, AdapterCrystalGraphLayer, PhyTCAModel (symmetric baseline)
├── diagnostics.py                 # FR-PhyTCA model variants and diagnostic experiments
├── train_phytca.py                # Continual training/evaluation entry point
├── baselines.py                   # Phase 0 baseline methods
├── scripts/
│   ├── run_phase0.py              # Multi-seed full Phase 0 runner
│   ├── run_phase0_b_screening.py  # Two-task Protocol B screening (symmetric PhyTCA baselines)
│   └── run_phase0_diagnostics.py  # D1–D6 diagnostic experiments (FR-PhyTCA redesign)
├── tests/
│   ├── test_periodic_graph.py     # PBC correctness and protocol splitting tests
│   ├── test_protocol_semantics.py # Protocol A/B semantic invariants
│   └── test_diagnostics.py        # FR-PhyTCA parent-route invariance and replay accounting
├── configs/
│   ├── jarvis_protocol_a.yaml     # Database-evolution protocol
│   └── jarvis_protocol_b.yaml     # Multi-fidelity band-gap protocol
└── reports/                       # Audit manifests, gate JSONs, and Phase 0 results
```

## Installation

Use the `EGNN` conda environment (or create an equivalent one):

```bash
conda activate EGNN
pip install jarvis-tools pymatgen egnn-pytorch torch
```

JARVIS datasets are loaded through `jarvis-tools` with a fallback to cached zip files in `data_cache/jarvis/`:

- `jdft_3d-8-18-2021.json.zip` — JARVIS 2021 (`dft_3d_2021`), ~55k structures
- `jdft_3d-12-12-2022.json.zip` — JARVIS 2022 (`dft_3d`), ~76k structures

If the cache is missing, `data.py` will attempt to download via `jarvis-tools`.

## Data audit and GO/NO-GO gate

Audit a protocol before training:

```bash
python data_audit.py --protocol a --report-dir reports
python data_audit.py --protocol b --report-dir reports
```

Artifacts produced:

- `reports/audit_protocol_{a,b}.md`
- `reports/gate_protocol_{a,b}.json`
- `reports/manifest_protocol_{a,b}.json`

Both protocols currently pass the gate (sufficient samples, finite targets, disjoint splits, working periodic graph builder).

## Continual learning protocols

### Protocol A — Database evolution (data-incremental)

Task sequence:

1. JARVIS-2021 formation energy (OptB88vdW)
2. JARVIS-2022 formation energy (OptB88vdW)
3. JARVIS-2021 band gap (OptB88vdW)
4. JARVIS-2022 band gap (OptB88vdW)

A1 → A2 and A3 → A4 are **data-incremental** snapshots of the same property/fidelity. They share the same property embedding, fidelity embedding, prediction head, and adapter route. The route is frozen only after the last occurrence of that (property, fidelity) in the task sequence.

### Protocol B — Multi-fidelity band gap

Task sequence:

1. JARVIS-2021 band gap / OptB88vdW
2. JARVIS-2021 band gap / TB-mBJ
3. JARVIS-2022 band gap / OptB88vdW
4. JARVIS-2022 band gap / TB-mBJ

Only structures with both band-gap fidelities are retained.

## Training

Full-scale training from a config:

```bash
python train_phytca.py --protocol a --hidden-dim 64 --adapter-rank 8 --epochs 20 --device cuda
python train_phytca.py --protocol b --hidden-dim 64 --adapter-rank 8 --epochs 20 --device cuda
```

Small GPU smoke test (≤500 samples/task, 2 epochs):

```bash
python train_phytca.py --protocol a --cap 500 --epochs 2 --batch-size 16 --hidden-dim 32 --adapter-rank 4 --device cuda
python train_phytca.py --protocol b --cap 500 --epochs 2 --batch-size 16 --hidden-dim 32 --adapter-rank 4 --device cuda
```

To log per-parameter-group gradient norms during the A2 training step, add `--log-gradients`:

```bash
python train_phytca.py --protocol a --cap 500 --epochs 2 --log-gradients --device cuda
```

## Phase 0 baseline comparison

`baselines.py` implements the comparison methods: joint training, independent models, sequential fine-tuning, frozen encoder + independent heads, EWC, experience replay, independent LoRA, shared LoRA bank, and PhyTCA.

Run the full multi-seed Phase 0 comparison:

```bash
python scripts/run_phase0.py --protocol b --cap 2000 --seeds 42 43 44 --epochs 15
```

Run the small-scale Protocol B two-task screening (used for GO/NO-GO gating):

```bash
python scripts/run_phase0_b_screening.py \
  --train-cap 2000 --val-cap 500 --test-cap 1000 \
  --seed 42 --epochs 10 --patience 3 --batch-size 32 \
  --hidden-dim 64 --adapter-rank 8 --device cuda --with-joint
```

Run the paired recheck (PhyTCA `μ=0` vs `μ=0.01`) to verify identical Task-1 trajectories:

```bash
python scripts/run_phase0_b_screening.py --paired-recheck --device cuda
```

Tune the stability coefficient `μ` on the held-out `continual_dev` split:

```bash
python scripts/run_phase0_b_screening.py --mu-grid --device cuda
```

The grid is `[0, 1e-5, 1e-4, 1e-3, 1e-2]`; the coefficient that gives the lowest absolute forgetting on Task 1 after Task 2 (with competitive Task-2 nMAE) is selected. `final_test` is not used for tuning or GO decisions.

All methods in the screening start from the same canonical frozen-encoder checkpoint per seed (`artifacts/init/seed_{seed}_base.pt`), and the held-out split is divided into `continual_dev` (used for reporting) and `final_test` (frozen and unused for tuning or GO decisions). Results are saved to `reports/phase0_b_screening/screening_results.json`.

### Round 1 screening outcome (corrected)

Protocol B (2021 OPT → 2021 MBJ) at 2k train / 500 val / 500 continual_dev / 500 final_test per task, seed 42. Metrics are reported on `continual_dev`; `final_test` is held out and unused for tuning or GO decisions. All methods start from the same canonical frozen-encoder checkpoint (`artifacts/init/seed_42_base.pt`) and use the same batch order. BWT is reported for nMAE as `BWT = -absolute_forgetting`, so negative values indicate forgetting.

| Method | T1@T1 | T1@T2 | T2 final | abs forgetting | BWT | avg final nMAE | trainable params | stored params |
|---|---|---|---|---|---|---|---|---|
| Sequential FT | 0.791 | 1.421 | 0.885 | 0.630 | -0.630 | 1.153 | 285,283 | 285,283 |
| + replay (1%) | 0.782 | 2.873 | 0.848 | 2.091 | -2.091 | 1.860 | 3,988 | 285,283 |
| Shared LoRA bank | 0.812 | 2.197 | 0.818 | 1.385 | -1.385 | 1.507 | 5,132 | 286,427 |
| PhyTCA ($\mu=0$) | 0.782 | 2.940 | 0.794 | 2.159 | -2.159 | 1.867 | 3,858 | 285,283 |
| PhyTCA ($\mu=0.01$) | 0.782 | 2.584 | 0.798 | 1.803 | -1.803 | 1.691 | 3,858 | 285,283 |
| Joint (upper bound) | 0.356 | 0.356 | 0.261 | 0.000 | 0.000 | 0.309 | 285,283 | 285,283 |

**Decision:** `NO_GO_PHYTCA_NO_ADVANTAGE_AT_2K`. Under this corrected single-seed screen, PhyTCA's absolute forgetting (1.803) is larger than sequential fine-tuning's forgetting (0.630), so the symmetric PhyTCA configuration does not provide a basic advantage signal. The next redesign is FR-PhyTCA, not a larger $\mu$ grid or 5k training.

### Diagnostic redesign (D1–D6)

After the NO-GO, a set of diagnostic experiments on the same 2k/seed-42 Protocol B split isolates capacity, backbone quality, sequential failure, and the OPT→MBJ parent-child relationship. A single shared OPT parent checkpoint is trained once and loaded by D4–D6 so that any Task-2 difference comes from the correction module, not a different Task-1 starting point. All metrics are on `continual_dev`.

| Experiment | T1@T1 | T1@T2 | T2 final | abs forgetting | BWT | avg final nMAE | raw MAE (eV) | incr. params |
|---|---|---|---|---|---|---|---|---|
| D1 Full joint (upper bound) | 0.339 | 0.339 | 0.248 | 0.000 | 0.000 | 0.294 | 0.441 | 0 |
| D2 Joint PhyTCA (symmetric) | 0.471 | 0.471 | 0.357 | 0.000 | 0.000 | 0.414 | 0.621 | 0 |
| D3 Sequential PhyTCA (symmetric) | 1.406 | 2.511 | 2.106 | 1.105 | -1.105 | 2.309 | 1.818 | 165 |
| D4 Frozen OPT + affine MBJ | 0.881 | 0.881 | 0.835 | 0.000 | 0.000 | 0.858 | 1.288 | 8,450 |
| D5 Frozen OPT + residual MBJ | 0.881 | 0.881 | 0.838 | 0.000 | 0.000 | 0.859 | 1.290 | 4,225 |
| D6 FR-PhyTCA, no distillation | 0.881 | 0.881 | 0.446 | 0.000 | 0.000 | 0.664 | 0.996 | 3,923 |

**Diagnosis gates:**
- `DIAGNOSIS_ADAPTER_ON_RANDOM_BACKBONE` — the canonical encoder is randomly initialized, so the adapter is operating on a weak backbone.
- `DIAGNOSIS_SEQUENTIAL_OPTIMIZATION_FAILURE` — D3 already fails on Task 1 (T1@T1 = 1.406 vs. D2 = 0.471), so the failure is better described as sequential optimization failure than pure Task-2 interference.
- `GO_TO_FIDELITY_RESIDUAL_PHYTCA` — preserving the OPT route eliminates forgetting (D4–D6 share T1@T1 = 0.881 and the same prediction hash), and the structured Tucker residual (D6) improves over both frozen-OPT correction baselines while using only ~4k incremental parameters.

Run the diagnostic experiments:

```bash
python scripts/run_phase0_diagnostics.py --device cuda
```

The script writes `reports/phase0_b_screening/diagnostic_experiments.json` and prints the gates. It also emits the shared OPT parent checkpoint (`artifacts/init/seed_42_base.pt`), parameter names for the Task-2 module, and `opt_route_drift` for every D6 variant.

#### D6 ablations

| Experiment | T1@T1 | T1@T2 | T2 final | abs forgetting | BWT | avg final nMAE | raw MAE (eV) | incr. params |
|---|---|---|---|---|---|---|---|---|
| D6-a FR-PhyTCA, no distillation | 0.881 | 0.881 | 0.446 | 0.000 | 0.000 | 0.664 | 0.996 | 3,923 |
| D6-b FR-PhyTCA + distillation ($\lambda=1.0$) | 0.881 | 0.881 | 0.446 | 0.000 | 0.000 | 0.664 | 0.996 | 3,923 |
| D6-c Independent low-rank residual | 0.881 | 0.881 | 0.905 | 0.000 | 0.000 | 0.893 | 1.341 | 520 |
| D6-d Parameter-matched residual MLP | 0.881 | 0.881 | 0.908 | 0.000 | 0.000 | 0.895 | 1.343 | 1,057 |
| D6-e Orthogonal Tucker residual | 0.881 | 0.881 | 0.446 | 0.000 | 0.000 | 0.664 | 0.996 | 3,923 |
| D6-f Shared factors + top-layer update | 0.881 | 1.180 | 0.342 | 0.298 | -0.298 | 0.761 | 1.142 | 92,632 |

Distillation across $\lambda \in \{0, 0.1, 1.0, 10.0\}$ leaves performance unchanged, exactly as expected when the OPT route is truly frozen. The low-rank and parameter-matched MLP residuals lag FR-PhyTCA, showing that the improvement is not merely from freezing the parent. Sharing factors and updating the top encoder layer (D6-f) improves Task-2 nMAE but perturbs the OPT route and incurs forgetting.

### 2k×3-seed reproducibility

After the diagnostic GO gate, we re-ran D1–D6 on seeds 42, 43, and 44 (2k train / task). Results on `continual_dev`:

| Experiment | T1@T1 | T1@T2 | T2 final | abs forgetting | avg final nMAE | raw MAE (eV) | incr. params |
|---|---|---|---|---|---|---|---|
| D1 Full joint | 0.337 ± 0.036 | 0.337 ± 0.036 | 0.225 ± 0.027 | 0.000 | 0.281 ± 0.031 | 0.424 ± 0.043 | 0 |
| D2 Joint PhyTCA (symmetric) | 0.481 ± 0.048 | 0.481 ± 0.048 | 0.348 ± 0.032 | 0.000 | 0.415 ± 0.040 | 0.626 ± 0.052 | 0 |
| D3 Sequential PhyTCA (symmetric) | 1.482 ± 0.069 | 1.882 ± 0.517 | 2.039 ± 0.240 | 0.400 ± 0.511 | 1.961 ± 0.269 | 1.527 ± 0.210 | 165 |
| D4 Frozen OPT + affine MBJ | 0.824 ± 0.044 | 0.824 ± 0.044 | 0.840 ± 0.032 | 0.000 | 0.832 ± 0.036 | 1.258 ± 0.061 | 8,450 |
| D5 Frozen OPT + residual MBJ | 0.824 ± 0.044 | 0.824 ± 0.044 | 0.841 ± 0.034 | 0.000 | 0.832 ± 0.036 | 1.258 ± 0.062 | 4,225 |
| D6 FR-PhyTCA | 0.824 ± 0.044 | 0.824 ± 0.044 | 0.382 ± 0.038 | 0.000 | 0.603 ± 0.004 | 0.912 ± 0.009 | 3,923 |
| D6-c Independent low-rank residual | 0.824 ± 0.044 | 0.824 ± 0.044 | 0.921 ± 0.013 | 0.000 | 0.873 ± 0.021 | 1.319 ± 0.040 | 520 |
| D6-d Parameter-matched residual MLP | 0.824 ± 0.044 | 0.824 ± 0.044 | 0.925 ± 0.013 | 0.000 | 0.875 ± 0.019 | 1.322 ± 0.029 | 1,057 |
| D6-e Orthogonal Tucker residual | 0.824 ± 0.044 | 0.824 ± 0.044 | 0.382 ± 0.038 | 0.000 | 0.603 ± 0.004 | 0.912 ± 0.010 | 3,923 |
| D6-f Shared factors + top-layer update | 0.824 ± 0.044 | 0.911 ± 0.035 | 0.324 ± 0.031 | 0.086 ± 0.047 | 0.617 ± 0.004 | 0.933 ± 0.018 | 92,632 |

Parent-route invariance holds exactly across all three seeds (T1@T1 spread within each seed is 0.00, OPT-route drift is 0.00 for every D6 variant, and forgetting is identically zero). FR-PhyTCA's average final nMAE is stable at 0.603 ± 0.004 and clearly outperforms both frozen-OPT correction (0.832 ± 0.036) and symmetric sequential PhyTCA (1.961 ± 0.269). The structured Tucker residual is therefore robust and not an artifact of seed 42.

### Stage 2: 5k×3-seed scaling validation

After the 2k reproducibility gate passed, we scaled Protocol B to 5,000 training samples per task (seeds 42, 43, 44) while freezing the FR-PhyTCA design. All frozen-parent methods share one trained OPT parent per seed, so differences reflect the correction module, not initialization. `final_test` remains unopened.

```bash
python scripts/run_phase2_b_scaling.py \
  --train-cap 5000 --val-cap 500 --test-cap 1000 \
  --seeds 42 43 44 --epochs 10 --patience 3 --batch-size 32 \
  --hidden-dim 64 --adapter-rank 8 --device cuda
```

| Method | T1@T1 | T1@T2 | T2 final | avg final nMAE | raw MAE (eV) | incr. params |
|---|---|---|---|---|---|---|
| Full joint (upper bound) | 0.293 ± 0.034 | 0.293 ± 0.034 | 0.188 ± 0.009 | 0.241 ± 0.017 | 0.367 ± 0.025 | 0 |
| Joint PhyTCA (upper bound) | 0.471 ± 0.022 | 0.471 ± 0.022 | 0.310 ± 0.024 | 0.390 ± 0.017 | 0.595 ± 0.027 | 0 |
| MBJ-only training | 1.056 ± 0.228 | 1.056 ± 0.228 | 0.758 ± 0.037 | 0.907 ± 0.133 | 1.630 ± 0.240 | 3,988 |
| OPT pretrain → MBJ full fine-tune | 0.839 ± 0.030 | 1.793 ± 0.812 | 0.895 ± 0.045 | 1.344 ± 0.404 | 2.048 ± 0.607 | 285,283 |
| Frozen OPT + affine correction | 0.839 ± 0.030 | 0.839 ± 0.030 | 0.842 ± 0.017 | 0.840 ± 0.022 | 1.281 ± 0.038 | 8,450 |
| Frozen OPT + matched MLP residual | 0.839 ± 0.030 | 0.839 ± 0.030 | 0.829 ± 0.033 | 0.834 ± 0.028 | 1.271 ± 0.048 | 3,895 |
| Frozen OPT + matched low-rank residual | 0.839 ± 0.030 | 0.839 ± 0.030 | 0.831 ± 0.017 | 0.835 ± 0.015 | 1.273 ± 0.029 | 3,900 |
| **FR-PhyTCA** | **0.839 ± 0.030** | **0.839 ± 0.030** | **0.342 ± 0.022** | **0.590 ± 0.026** | **0.900 ± 0.040** | **3,923** |
| Orthogonal FR-PhyTCA | 0.839 ± 0.030 | 0.839 ± 0.030 | 0.342 ± 0.021 | 0.590 ± 0.025 | 0.900 ± 0.039 | 3,923 |
| Feature-transfer baseline | 0.839 ± 0.030 | 0.839 ± 0.030 | 0.823 ± 0.017 | 0.831 ± 0.013 | 1.267 ± 0.018 | 8,321 |

FR-PhyTCA retains exact parent-route invariance (OPT-route drift 0.00 on every seed). Under that constraint, its Task-2 nMAE is more than 10% lower than the parameter-matched MLP and low-rank residuals, it beats both on all three seeds, raw MAE drops versus the 2k stage, and the gap to Joint Tucker does not widen. The Stage-2 gate is `GO_TO_REALISTIC_FIDELITY_SCALING`.

## Periodic graph construction

`PeriodicGraphBuilder` expands each unit cell into a supercell (default `2x2x2`) so that nearest-neighbor messages can cross periodic boundaries. Pooling is restricted to atoms with zero image offset via `original_mask`.

Run the correctness tests:

```bash
python -m pytest tests/ -v
```

Tests cover:

- JARVIS record → pymatgen `Structure` conversion
- Supercell size and original-mask consistency
- PBC invariance at encoder depths 1, 2, and 4
- Periodic halo convergence (2×2×2 vs 3×3×3 vs 4×4×4)
- Primitive vs. supercell equivalence
- Integer-lattice separation of periodic replicas
- Crystal graph encoder forward pass on real JARVIS periodic graphs
- Protocol A data-incremental semantics (shared head, shared adapter route, no snapshot conditioning, JID-disjoint snapshots)
- Protocol B OPT/MBJ pairing and shared-head semantics
- Formula-disjoint train/val/continual_dev/final_test splits within each task
- FR-PhyTCA parent-route invariance and replay-buffer parameter accounting

## Key design notes

- **FR-PhyTCA** freezes a parent fidelity route and learns higher-fidelity residuals. The parent route is invariant by construction: the child slice for the parent fidelity is zero-initialized and gradient-zeroed after every step.
- **Symmetric PhyTCA** is the baseline where shared Tucker factors are updated for every task.
- **No tensor properties / MP access in current protocols.** The implemented Protocols A and B use scalar formation energy and band-gap targets from JARVIS only.
- **Data-incremental Protocol A** shares property/fidelity embeddings, adapter route, and prediction head across A1↔A2 and A3↔A4. Freezing happens only after the final occurrence of a (property, fidelity).
- **Formula-disjoint splits** prevent data leakage across train/val/continual_dev/final_test within each task.
- **Fair initialization.** All Phase 0 baselines start from the same canonical frozen-encoder checkpoint per seed; the same batch order is used across methods.
- **Target parser** rejects `None`, `NaN`, `inf`, `"na"`, empty strings, etc.
- **Data loader** uses one-hot element features (dim 92) by default.

## Citation

See `E:\PAPER\Parameter-Efficient Continual Learning via Structured Tensor Decomposition\example_paper.tex` for the accompanying manuscript.
