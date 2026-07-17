# CLAUDE.md — Backward-Compatible Model Serving for Evolving Materials Databases

## Project context

This repository studies **backward-compatible model serving** for evolving materials databases. The core observation is that exact retention of an old prediction endpoint and continual learning of new labels for the same `(material, property, fidelity)` are fundamentally incompatible when no version identifier is provided at inference: a single deterministic function cannot output two different values for the same input. The project therefore formalizes the problem as serving a set of **versioned endpoints**. Each endpoint is identified by `(version, property, fidelity)`. Old endpoints are frozen after publication; the latest endpoint is allowed to learn revised data and new fidelities.

The current architecture is the **Persistent Consolidation Graph (PCG)**, implemented in `persistent_consolidation_graph.py`. PCG uses a frozen backbone (MatGL by default), an append-only shared basis bank, route-private coefficients, and typed parent edges (temporal revision, fidelity transfer). New endpoints first learn through a temporary fast adapter, then project that knowledge onto the basis bank; when the residual is too novel, new basis blocks are appended. Old blocks are never modified, so every published endpoint is structurally immutable.

The symmetric **PhyTCA** baseline and the old Phase-0 baselines, plus the older `VersionedFidelityGraph`, have been moved to or replaced by code in `legacy/`; they are retained only for reference and should not be used in new work.

## Problem statement (formal)

**Theorem (Impossibility of exact retention without version IDs).** Let $f$ be a deterministic predictor queried with a fixed input $x$ and a fixed task descriptor $(p,f)$ but no version identifier. Suppose the label for $(x,p,f)$ changes from $y_{\text{old}}$ to $y_{\text{new}} \neq y_{\text{old}}$ between two database snapshots. There is no single model $f$ that simultaneously (i) exactly retains the old prediction, $f(x,p,f)=y_{\text{old}}$, and (ii) learns the new label, $f(x,p,f)=y_{\text{new}}$.

**Proof.** A function maps each input to exactly one output. The two requirements demand two distinct outputs for the same input, a contradiction. Therefore exact retention plus learning requires either a version identifier at inference or multiple endpoint functions.

This motivates the versioned-endpoint design: inference must specify which published endpoint to query.

## Environment

- Activate the `EGNN` conda environment before running Python commands: `source activate EGNN` (bash) or `conda activate EGNN` (cmd).
- Required packages: `torch`, `egnn-pytorch`, `pymatgen`, `jarvis-tools`, `numpy`, `pytest`, `matgl` (for the MatGL backbone).

## Code conventions

- Python 3.11+ with `from __future__ import annotations`.
- Prefer explicit type hints.
- Keep modules flat and focused: `data.py` for data, `persistent_consolidation_graph.py` for the PCG architecture, `protocols.py` for protocol builders, `pcg_runner.py` for shared runner utilities, `train_utils.py` for shared training utilities, `diagnostics.py` for experiments, `data_audit.py` for audits.
- Legacy code lives in `legacy/` and should not be imported by new modules.
- Default to no comments; only explain non-obvious invariants.

## Data workflow

1. JARVIS data lives in `data_cache/jarvis/` as zip files.
2. Use `data.load_jarvis_dataset(name)` which loads directly from zip and falls back to `jarvis-tools`.
3. Target fields are defined in `data.TARGET_FIELDS` and include band gap (OptB88vdW, TB-mBJ, HSE), formation energy, bulk modulus, shear modulus, and dielectric tensor components.
4. Run `python data_audit.py --protocol {a,b}` to produce manifests, reports, and the GO/NO-GO gate for the legacy Protocol A/B benchmarks.
5. Use `protocols.build_combined_protocol(...)`, `build_revision_protocol(...)`, `build_addition_protocol(...)`, or `build_fidelity_expansion_protocol(...)` to construct three-axis `(version, property, fidelity)` tasks with canonical material-group splits.
6. Training uses `JARVISCrystalDataset` and `collate_crystals`, which return a 5-tuple: `(node_feats, coords, mask, original_mask, targets)`. The model must pool using `original_mask` only.

## Protocols

- **Protocol A** — database evolution across JARVIS 2021/2022 for formation energy and band gap (OptB88vdW). A1↔A2 and A3↔A4 are data-incremental: they share property/fidelity embeddings, adapter route, and head. The route is frozen only after its final occurrence. *Limitation:* because the same `(property, fidelity)` route is reused across snapshots, exact retention cannot hold when a target value is revised.
- **Protocol B** — multi-fidelity band gap: OptB88vdW → TB-mBJ across both JARVIS versions, with paired records sharing splits.
- **PCG protocols** — three-axis `(version, property, fidelity)` benchmarks built by `protocols.py`. Each combination is a distinct endpoint, so 2021 OPT and 2022 OPT are separate routes and exact retention of the 2021 endpoint is well defined. The combined protocol interleaves fidelities within each version.

Do not introduce Materials Project data or tensor properties unless explicitly requested.

## Architecture

### `persistent_consolidation_graph.PersistentConsolidationGraph`

Frozen backbone (MatGL or EGNN) plus an append-only `BasisBank` of low-rank `(U_in, U_out)` blocks. Each endpoint owns a `RouteSpec` with parent IDs, selected basis block IDs, private middle matrices `M_e`, a head, and a normalizer. `publish_route(...)` freezes the route's coefficients and its basis blocks, guaranteeing exact retention. New routes may reuse existing blocks or trigger novelty-gated expansion. Incremental parameter count per endpoint: $\sum_k r_k^2 + (d+1)$ for the private coefficients/head plus $2 d \sum_k r_k$ when new basis blocks are appended.

### `models.CrystalEncoder`

Small EGNN-based encoder used for fast smoke tests and diagnostics. The main experiments use `backbones.MatGLBackbone`.

### `models.CopyOnWriteFullChildModel`

Each versioned endpoint owns a deep copy of the full crystal encoder and a private head. This is the strongest exact-retention baseline (no cross-route interference) but pays a full encoder per endpoint.

## Testing

Run core PCG tests with:

```bash
python -m pytest tests/test_persistent_consolidation_graph.py tests/test_pcg_matgl.py tests/test_protocols.py tests/test_snapshot_diff.py tests/test_pcg_revision_runner.py tests/test_pcg_addition_runner.py tests/test_pcg_fidelity_runner.py tests/test_pcg_baselines.py -v
```

`tests/test_persistent_consolidation_graph.py` covers:
- append-only isolation of old routes under training of new routes,
- endpoint registry hash detection,
- typed parent gates with physical-unit aggregation,
- zero forgetting after publication.

`tests/test_pcg_matgl.py` covers MatGL-backbone + PCG smoke tests.

`tests/test_protocols.py` and `tests/test_snapshot_diff.py` cover protocol builders and snapshot diff classification.

`tests/test_pcg_baselines.py` covers all comparison methods on a tiny capped protocol.

## Training

Combined PCG benchmark:

```bash
python -m scripts.run_pcg_combined --properties band_gap --fidelities OptB88vdW TB-mBJ --hidden-dim 64 --rank 8 --epochs-fast 5 --epochs-cons 15 --device cuda
```

Single-axis protocols:

```bash
python -m scripts.run_pcg_revision --properties band_gap --fidelities OptB88vdW --hidden-dim 64 --rank 8 --device cuda
python -m scripts.run_pcg_addition --properties band_gap --fidelities OptB88vdW --hidden-dim 64 --rank 8 --device cuda
python -m scripts.run_pcg_fidelity_expansion --properties band_gap --fidelities OptB88vdW TB-mBJ --hidden-dim 64 --rank 8 --device cuda
```

Baseline comparison:

```bash
python -m scripts.run_pcg_baselines --properties band_gap --fidelities OptB88vdW TB-mBJ --hidden-dim 64 --rank 8 --device cuda
```

The harness automatically skips methods whose `reports/pcg_baselines/<method>/metrics.json` already exists, so re-running the same command resumes from the last incomplete method.

For smoke tests, add `--cap 50` and use `--encoder-type egnn` for faster CPU execution.

Legacy task-incremental training:

```bash
python train_phytca.py --protocol a --hidden-dim 64 --adapter-rank 8 --epochs 20 --device cuda
```

`scripts/run_pcg_combined.py` consumes `protocols.build_combined_protocol(...)`, trains each `(version, property, fidelity)` route with typed parents, publishes it, and evaluates all published endpoints. Metrics are written to `reports/pcg_combined/metrics.json`.

## Baselines and Phase 0 comparison

Old baselines (joint, independent, sequential, EWC, replay, LoRA variants, symmetric PhyTCA) have been moved to `legacy/baselines.py` and `legacy/phytca.py`. They are no longer part of the recommended workflow.

The new evaluation compares:
- Independent models per endpoint (upper bound on accuracy, no sharing),
- Joint model on all endpoints (upper bound, no exact retention),
- `PersistentConsolidationGraph` with novelty-gated expandable basis,
- Copy-on-write full child models,
- Fixed shared-basis and always-expand ablations,
- Replay / distillation / functional regularization for the same-fidelity revision case.

Metrics must include: latest-task MAE, per-endpoint drift, forward/backward transfer, calibration error, parameter count, basis growth, checkpoint size, inference latency, and top-k recall.

## Status and next steps

1. **Done**
   - Snapshot diff foundation (`snapshot_diff.py`) and protocol builders (`protocols.py`).
   - Persistent Consolidation Graph core (`persistent_consolidation_graph.py`) with append-only basis bank, fast adapter, novelty gate, and typed parent edges.
   - MatGL backbone integration (`tests/test_pcg_matgl.py`, `pcg_runner.py`).
   - Protocol-specific runners (`scripts/run_pcg_combined.py`, `run_pcg_revision.py`, `run_pcg_addition.py`, `run_pcg_fidelity_expansion.py`).
   - Pareto harness extended with physical-target forward kwargs.

2. **Done**
   - Cap-500 baseline comparison (`scripts/run_pcg_baselines.py --cap 500`) completed and filled `tab:baselines` in `main.tex`.

3. **Remaining work**
   - Full-scale (no-cap) experiments for the camera-ready tables.
   - Finalize Pareto metrics and paper figures.

## Avoid

- Adding tensor-property protocols or MP data without explicit approval.
- Modifying the periodic graph builder without updating `tests/test_periodic_graph.py`.
- Using the `llm` conda environment; always prefer `EGNN`.
