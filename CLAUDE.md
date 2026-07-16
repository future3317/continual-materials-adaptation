# CLAUDE.md — Backward-Compatible Model Serving for Evolving Materials Databases

## Project context

This repository studies **backward-compatible model serving** for evolving materials databases. The core observation is that exact retention of an old prediction endpoint and continual learning of new labels for the same `(material, property, fidelity)` are fundamentally incompatible when no version identifier is provided at inference: a single deterministic function cannot output two different values for the same input. The project therefore formalizes the problem as serving a set of **versioned endpoints**. Each endpoint is identified by `(version, property, fidelity)`. Old endpoints are frozen after publication; the latest endpoint is allowed to learn revised data and new fidelities.

The repository contains both (1) an initial task-incremental FR-PhyTCA implementation (`diagnostics.py`, `train_phytca.py`) and (2) a new `VersionedFidelityGraph` (`versioned_graph.py`) that implements structural exact retention for published endpoints via shared low-rank bases and route-private coefficients.

The symmetric **PhyTCA** baseline and the old Phase-0 baselines have been moved to `legacy/`; they are retained only for reference and should not be used in new work.

## Problem statement (formal)

**Theorem (Impossibility of exact retention without version IDs).** Let $f$ be a deterministic predictor queried with a fixed input $x$ and a fixed task descriptor $(p,f)$ but no version identifier. Suppose the label for $(x,p,f)$ changes from $y_{\text{old}}$ to $y_{\text{new}} \neq y_{\text{old}}$ between two database snapshots. There is no single model $f$ that simultaneously (i) exactly retains the old prediction, $f(x,p,f)=y_{\text{old}}$, and (ii) learns the new label, $f(x,p,f)=y_{\text{new}}$.

**Proof.** A function maps each input to exactly one output. The two requirements demand two distinct outputs for the same input, a contradiction. Therefore exact retention plus learning requires either a version identifier at inference or multiple endpoint functions.

This motivates the versioned-endpoint design: inference must specify which published endpoint to query.

## Environment

- Activate the `EGNN` conda environment before running Python commands: `source activate EGNN` (bash) or `conda activate EGNN` (cmd).
- Required packages: `torch`, `egnn-pytorch`, `pymatgen`, `jarvis-tools`, `numpy`, `pytest`.

## Code conventions

- Python 3.11+ with `from __future__ import annotations`.
- Prefer explicit type hints.
- Keep modules flat and focused: `data.py` for data, `models.py`/`versioned_graph.py` for the new architecture, `train_utils.py` for shared training utilities, `diagnostics.py` for experiments, `data_audit.py` for audits.
- Legacy code lives in `legacy/` and should not be imported by new modules.
- Default to no comments; only explain non-obvious invariants.

## Data workflow

1. JARVIS data lives in `data_cache/jarvis/` as zip files.
2. Use `data.load_jarvis_dataset(name)` which loads directly from zip and falls back to `jarvis-tools`.
3. Target fields are defined in `data.TARGET_FIELDS` and include band gap (OptB88vdW, TB-mBJ, HSE), formation energy, bulk modulus, shear modulus, and dielectric tensor components.
4. Run `python data_audit.py --protocol {a,b}` to produce manifests, reports, and the GO/NO-GO gate for the legacy Protocol A/B benchmarks.
5. Use `data.build_versioned_protocol(...)` to construct three-axis `(version, property, fidelity)` tasks with canonical material-group splits.
6. Training uses `JARVISCrystalDataset` and `collate_crystals`, which return a 5-tuple: `(node_feats, coords, mask, original_mask, targets)`. The model must pool using `original_mask` only.

## Protocols

- **Protocol A** — database evolution across JARVIS 2021/2022 for formation energy and band gap (OptB88vdW). A1↔A2 and A3↔A4 are data-incremental: they share property/fidelity embeddings, adapter route, and head. The route is frozen only after its final occurrence. *Limitation:* because the same `(property, fidelity)` route is reused across snapshots, exact retention cannot hold when a target value is revised.
- **Protocol B** — multi-fidelity band gap: OptB88vdW → TB-mBJ across both JARVIS versions, with paired records sharing splits.
- **Versioned Protocol** — the new three-axis benchmark built by `data.build_versioned_protocol`. Each `(version, property, fidelity)` combination is a distinct endpoint, so 2021 OPT and 2022 OPT are separate routes and exact retention of the 2021 endpoint is well defined.

Do not introduce Materials Project data or tensor properties unless explicitly requested.

## Architecture

### `models.ContinualCrystalModel`

Frozen crystal-graph encoder with per-task adapter banks and heads. Exact retention by structural isolation: old tasks are frozen via `requires_grad=False` and excluded from the child optimizer.

### `versioned_graph.VersionedFidelityGraph`

Frozen encoder plus shared low-rank bases `U_in`, `U_out`. Each versioned endpoint owns a private middle matrix `M_r` and a head. `publish_route(...)` freezes the route's coefficients and the shared bases, guaranteeing exact retention. New routes add new `M_r` matrices on top of the frozen bases. Incremental parameter count per endpoint: $L r^2 + (d+1)$.

## Testing

Run core tests with:

```bash
python -m pytest tests/test_versioned_graph.py tests/test_versioned_runner.py tests/test_adapters_models.py tests/test_global_splits.py tests/test_fidelity_graph.py -v
```

`tests/test_versioned_graph.py` covers:
- exact retention of published endpoints under training of new routes,
- the impossibility sanity check without version IDs,
- incremental parameter accounting.

`tests/test_versioned_runner.py` covers the end-to-end `scripts/run_versioned_protocol.py` runner on capped JARVIS data.

`tests/test_global_splits.py` covers canonical material-group splits and cross-year leakage prevention.

## Training

Versioned-endpoint training:

```bash
python scripts/run_versioned_protocol.py --snapshots dft_3d_2021 dft_3d --properties band_gap --fidelities OptB88vdW TB-mBJ --hidden-dim 64 --rank 8 --epochs 15 --device cuda
```

For smoke tests, add `--cap 50` to limit per-split records.

Legacy task-incremental training:

```bash
python train_phytca.py --protocol a --hidden-dim 64 --adapter-rank 8 --epochs 20 --device cuda
```

`scripts/run_versioned_protocol.py` consumes `data.build_versioned_protocol(...)`, trains each `(version, property, fidelity)` route, publishes it, and evaluates all published endpoints. Metrics are written to `reports/versioned_protocol/metrics.json`.

## Baselines and Phase 0 comparison

Old baselines (joint, independent, sequential, EWC, replay, LoRA variants, symmetric PhyTCA) have been moved to `legacy/baselines.py` and `legacy/phytca.py`. They are no longer part of the recommended workflow.

The new evaluation will compare:
- Independent models per endpoint (upper bound on accuracy, no sharing),
- Joint model on all endpoints (upper bound, no exact retention),
- `VersionedFidelityGraph` with published frozen routes,
- Copy-on-write full child models,
- Standard LoRA-AB and LoRA-ABA adapters,
- Replay / distillation / functional regularization for the same-fidelity revision case.

Metrics must include: latest-task MAE, per-endpoint drift, forward/backward transfer, calibration error, parameter count, checkpoint size, inference latency, and top-k recall.

## Status and next steps

1. **Done in this refactor**
   - Moved `phytca.py` and `baselines.py` to `legacy/`.
   - Extracted shared training utilities into `train_utils.py`.
   - Added `data.TARGET_FIELDS` and `data.build_versioned_protocol` for three-axis tasks.
   - Implemented `versioned_graph.VersionedFidelityGraph` with exact retention by structural isolation.
   - Added formal impossibility statement and unit tests.
   - Implemented `scripts/run_versioned_protocol.py`, the first end-to-end three-axis benchmark runner.
   - Verified `data_audit.py --protocol a/b` still pass and selected unit tests pass.

2. **Remaining work**
   - Implement the Pareto evaluation harness (calibration, latency, checkpoint size, top-k recall, forward transfer).
   - Compare against proper baselines on the versioned protocol (independent, joint, copy-on-write, LoRA-AB/ABA, replay).
   - Rewrite the paper around backward-compatible model serving, the impossibility theorem, and the three-axis benchmark.

## Avoid

- Adding tensor-property protocols or MP data without explicit approval.
- Modifying the periodic graph builder without updating `tests/test_periodic_graph.py`.
- Using the `llm` conda environment; always prefer `EGNN`.
