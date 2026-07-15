# CLAUDE.md — FR-PhyTCA

## Project context

This repository implements **FR-PhyTCA** (Fidelity-Residual Physics-Structured Tensor Component Adaptation), a continual-learning framework for evolving materials databases. The method first learns a **parent** model on a low-cost, abundant fidelity and then freezes its effective route permanently. Each subsequent higher fidelity is learned as a small, structured Tucker residual on top of the frozen parent, so the parent fidelity route is exactly invariant by construction and forgetting is identically zero.

The symmetric **PhyTCA** baseline (shared Tucker factors updated for every task) is retained for comparison in `phytca.py` and `baselines.py`. The FR-PhyTCA redesign lives in `diagnostics.py` and is executed by `scripts/run_phase0_diagnostics.py`.

## Environment

- Activate the `EGNN` conda environment before running Python commands: `source activate EGNN` (bash) or `conda activate EGNN` (cmd).
- Required packages: `torch`, `egnn-pytorch`, `pymatgen`, `jarvis-tools`, `numpy`, `pytest`.

## Code conventions

- Python 3.11+ with `from __future__ import annotations`.
- Prefer explicit type hints.
- Keep modules flat and focused: `data.py` for data, `phytca.py` for the symmetric baseline, `diagnostics.py` for FR-PhyTCA variants, `train_phytca.py` for training, `data_audit.py` for audits, `baselines.py` for comparison methods.
- Avoid creating many small files; reuse existing modules.
- Default to no comments; only explain non-obvious invariants.

## Data workflow

1. JARVIS data lives in `data_cache/jarvis/` as zip files.
2. Use `data.load_jarvis_dataset(name)` which loads directly from zip and falls back to `jarvis-tools`.
3. Run `python data_audit.py --protocol {a,b}` to produce manifests, reports, and the GO/NO-GO gate.
4. Training uses `JARVISCrystalDataset` and `collate_crystals`, which return a 5-tuple: `(node_feats, coords, mask, original_mask, targets)`. The model must pool using `original_mask` only.

## Protocols

- **Protocol A** — database evolution across JARVIS 2021/2022 for formation energy and band gap (OptB88vdW). A1↔A2 and A3↔A4 are data-incremental: they share property/fidelity embeddings, adapter route, and head. The route is frozen only after its final occurrence.
- **Protocol B** — multi-fidelity band gap: OptB88vdW → TB-mBJ across both JARVIS versions, with paired records sharing splits.

Protocol A snapshots are JID-disjoint; each task's train/val/test are formula-disjoint. Do not introduce Materials Project data or tensor properties unless explicitly requested.

## Testing

Run PBC, protocol, and FR-PhyTCA tests with:

```bash
python -m pytest tests/ -v
```

`tests/test_periodic_graph.py` covers multi-layer PBC invariance, halo convergence, and splitting correctness. `tests/test_protocol_semantics.py` covers data-incremental routing and freezing invariants. `tests/test_diagnostics.py` covers FR-PhyTCA parent-route invariance, canonical-to-progressive state remapping, residual identity initialization, and replay storage accounting.

## Training

```bash
python train_phytca.py --protocol a --hidden-dim 64 --adapter-rank 8 --epochs 20 --device cuda
```

For quick smoke tests use `--cap 500 --epochs 2 --hidden-dim 32 --adapter-rank 4`. Add `--log-gradients` to print per-parameter-group gradient norms during the A2 training step.

## Baselines and Phase 0 comparison

`baselines.py` implements joint, independent, sequential fine-tuning, frozen encoder + independent heads, EWC, replay, independent LoRA, shared LoRA bank, and symmetric PhyTCA.

Run the Protocol B two-task screening:

```bash
python scripts/run_phase0_b_screening.py \
  --train-cap 2000 --val-cap 500 --test-cap 1000 \
  --seed 42 --epochs 10 --patience 3 --batch-size 32 \
  --hidden-dim 64 --adapter-rank 8 --device cuda --with-joint
```

Run the paired recheck of PhyTCA `μ=0` vs `μ=0.01`:

```bash
python scripts/run_phase0_b_screening.py --paired-recheck --device cuda
```

Tune `μ` on `continual_dev` over `[0, 1e-5, 1e-4, 1e-3, 1e-2]`:

```bash
python scripts/run_phase0_b_screening.py --mu-grid --device cuda
```

Round 1 produced `NO_GO_PHYTCA_NO_ADVANTAGE_AT_2K`: under the corrected full-method screen, symmetric PhyTCA ($\mu=0.01$) still forgets more than sequential FT (absolute forgetting 1.803 vs. 0.630), so the current configuration does not provide a basic advantage signal. A stability-coefficient grid search on `continual_dev` over $[0, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}]$ selected $\mu=0.01$ as the best PhyTCA setting, but it still does not beat sequential FT. The paired recheck confirms that PhyTCA with and without stability loss share an identical Task-1 trajectory by construction. All methods use the same canonical base checkpoint per seed and the same batch order; held-out data is split into `continual_dev` (for reporting and $\mu$ tuning) and `final_test` (frozen). Do not scale to the 5k×3-seed stage until a clear advantage is demonstrated.

### Diagnostic redesign (D1–D6)

After the NO-GO, run the diagnostic experiments to locate the failure mode before scaling:

```bash
python scripts/run_phase0_diagnostics.py --device cuda
```

The diagnostic script writes `reports/phase0_b_screening/diagnostic_experiments.json` and prints gates. It trains a single shared OPT parent checkpoint once and loads it into D4–D6 so that Task-2 differences reflect the correction module, not the parent initialization. The most recent run produced:

- `DIAGNOSIS_ADAPTER_ON_RANDOM_BACKBONE`
- `DIAGNOSIS_SEQUENTIAL_OPTIMIZATION_FAILURE`
- `GO_TO_FIDELITY_RESIDUAL_PHYTCA`

Key results (2k/seed 42, `continual_dev`):

| Experiment | T1@T1 | T1@T2 | T2 final | abs forgetting | avg final nMAE | incr. params |
|---|---|---|---|---|---|---|
| D1 Full joint | 0.339 | 0.339 | 0.248 | 0.000 | 0.294 | 0 |
| D2 Joint PhyTCA (symmetric) | 0.471 | 0.471 | 0.357 | 0.000 | 0.414 | 0 |
| D3 Sequential PhyTCA (symmetric) | 1.406 | 2.511 | 2.106 | 1.105 | 2.309 | 165 |
| D4 Frozen OPT + affine MBJ | 0.881 | 0.881 | 0.835 | 0.000 | 0.858 | 8,450 |
| D5 Frozen OPT + residual MBJ | 0.881 | 0.881 | 0.838 | 0.000 | 0.859 | 4,225 |
| D6 FR-PhyTCA | 0.881 | 0.881 | 0.446 | 0.000 | 0.664 | 3,923 |

The architecture gap (D2–D1) is 0.120 nMAE (41% relative), so the low-rank adapter has non-negligible capacity limitations on a random backbone. D3 already fails on Task 1 (T1@T1 = 1.406), so the failure is better described as sequential optimization failure than pure Task-2 interference. Preserving the OPT route eliminates forgetting: D4, D5, and D6 share the exact same T1@T1 (0.881), prediction hash, and state-dict hash from the common OPT parent, and D6 reports $\max|\hat y_{\mathrm{OPT}}^{\mathrm{after}} - \hat y_{\mathrm{OPT}}^{\mathrm{before}}| = 0.00$. FR-PhyTCA (D6) closes most of the sequential gap while using only ~4k incremental parameters. Distillation across $\lambda \in \{0, 0.1, 1.0, 10.0\}$ leaves performance unchanged, as expected when the parent route is truly frozen.

Run the diagnostic and replay unit tests with the rest of the suite:

```bash
python -m pytest tests/ -v
```

The next step after the diagnostic GO gate is the **2k×3-seed reproducibility stage** (seeds 42, 43, 44). Across seeds, FR-PhyTCA achieves an average final nMAE of 0.603 ± 0.004, the frozen-OPT correction baselines average 0.832 ± 0.036, and sequential symmetric PhyTCA averages 1.961 ± 0.269. Parent-route invariance holds exactly (T1@T1 spread within each seed is 0.00, OPT-route drift is 0.00 for every D6 variant, forgetting is identically zero).

After the 2k stage passed, we ran **Stage 2: 5k×3-seed scaling validation** on Protocol B (train_cap=5000/task). All frozen-parent methods share one trained OPT parent per seed. FR-PhyTCA achieves an average final nMAE of 0.590 ± 0.026 and a Task-2 nMAE of 0.342 ± 0.022, more than 10% lower than the parameter-matched MLP (0.829 ± 0.033) and low-rank (0.831 ± 0.017) residuals, while using 3,923 incremental parameters. Raw MAE improves from 0.912 ± 0.009 eV (2k) to 0.900 ± 0.040 eV (5k), the gap to Joint Tucker (0.390 ± 0.017) stays within 0.25 nMAE, and OPT-route drift is 0.00 on every seed. The Stage-2 gate is `GO_TO_REALISTIC_FIDELITY_SCALING`.

Run the Stage-2 scaling study with:

```bash
python scripts/run_phase2_b_scaling.py \
  --train-cap 5000 --val-cap 500 --test-cap 1000 \
  --seeds 42 43 44 --epochs 10 --patience 3 --batch-size 32 \
  --hidden-dim 64 --adapter-rank 8 --device cuda
```

## Avoid

- Adding tensor-property protocols or MP data without explicit approval.
- Modifying the periodic graph builder without updating `tests/test_periodic_graph.py`.
- Changing FR-PhyTCA v1 design (adapter rank, residual placement, distillation coefficient, hidden dim, freezing policy) without explicit approval; the design is frozen after Stage 2.
- Using the `llm` conda environment; always prefer `EGNN`.
