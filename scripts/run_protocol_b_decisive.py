"""Decisive single-snapshot Protocol B experiment.

This script runs the Protocol B comparison requested in 反馈_2.md and the
user's latest instructions.  It uses only JARVIS-2022 band-gap data, builds an
OPT parent on all available OPT training data, and then compares child
fidelity-learning methods across variable amounts of MBJ training data and
multiple ranks/seeds.

Usage:
    python scripts/run_protocol_b_decisive.py --smoke
    python scripts/run_protocol_b_decisive.py --output-dir runs/protocol_b
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Allow importing from the project root regardless of where the script is run.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters import ADAPTER_REGISTRY
from baselines import _make_loaders
from data import (
    JARVISCrystalDataset,
    assign_global_splits,
    build_protocol_b,
    collate_crystals,
    jarvis_record_to_structure,
    load_jarvis_dataset,
    parse_target,
)
from diagnostics import (
    _evaluate_on_dev,
    _load_canonical_base,
    _name_to_id,
    _run_label_and_metrics,
    _snapshot_opt_predictions,
    _state_dict_hash,
    _train_single_task,
)
from models import (
    ContinualCrystalModel,
    CopyOnWriteTopBlock,
    compute_mad,
    normalized_mae,
)
from train_phytca import evaluate_loader


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_LOW_TRAIN_SIZES = [10000, 30000, None]
DEFAULT_HIGH_TRAIN_SIZES = [100, 500, 1000, 2000, 5000, None]
DEFAULT_METHODS = [
    "frozen_parent_only",
    "frozen_parent_affine_correction",
    "frozen_parent_residual_correction",
    "fr_lora_ab",
    "fr_lora_aba",
    "fr_single_child_tucker",
    "param_matched_bottleneck",
    "param_matched_mlp",
    "private_top_block",
    "full_child_finetuning",
    "joint_opt_mbj",
]
DEFAULT_RANKS = [2, 4, 8, 16, 32]
DEFAULT_SEEDS = [42, 43, 44]


def int_or_none(value: str) -> int | None:
    """Argument helper: 'all' or empty -> None, otherwise int."""
    if value is None or value.lower() in ("", "all", "none"):
        return None
    return int(value)


def parse_size_list(raw: list[str] | None) -> list[int | None]:
    """Parse a list of size strings into int/None values."""
    if raw is None:
        return []
    return [int_or_none(v) for v in raw]


# ---------------------------------------------------------------------------
# Data preparation: 2022-only paired band gaps with global canonical split
# ---------------------------------------------------------------------------


def _pair_bandgaps_2022(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return paired OPT/MBJ records for JARVIS-2022 only."""
    opt_recs, mbj_recs = [], []
    for r in records:
        opt = parse_target(r.get("optb88vdw_bandgap"))
        mbj = parse_target(r.get("mbj_bandgap"))
        if opt is None or mbj is None:
            continue
        struct = jarvis_record_to_structure(r)
        base = {
            "jid": r.get("jid"),
            "structure": struct,
            "formula": struct.composition.reduced_formula,
            "dataset": "dft_3d",
            "property": "band_gap",
        }
        opt_recs.append({**base, "fidelity": "OptB88vdW", "target": opt})
        mbj_recs.append({**base, "fidelity": "TB-mBJ", "target": mbj})
    return opt_recs, mbj_recs


def _assign_global_paired_splits(
    opt_recs: list[dict],
    mbj_recs: list[dict],
    seed: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> None:
    """Assign a single global formula-disjoint split shared by both fidelities.

    This is the "global canonical material group" fallback described in the
    instructions: the same reduced formula always lands in the same split, so
    OPT and MBJ records for the same material share train/val/test membership.
    """
    rng = np.random.default_rng(seed)
    formulas = list({r["formula"] for r in opt_recs})
    rng.shuffle(formulas)

    n = len(formulas)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    train_formulas = set(formulas[:n_train])
    val_formulas = set(formulas[n_train : n_train + n_val])
    test_formulas = set(formulas[n_train + n_val :])

    for r in opt_recs + mbj_recs:
        f = r["formula"]
        if f in train_formulas:
            r["split"] = "train"
        elif f in val_formulas:
            r["split"] = "val"
        else:
            r["split"] = "test"


def _cap_split(records: list[dict], cap: int | None, seed: int) -> list[dict]:
    """Cap the train split to ``cap`` records while keeping other splits intact."""
    if cap is None:
        return records
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for split in ("train", "val", "continual_dev", "test"):
        recs = [r for r in records if r.get("split") == split]
        if split == "train" and len(recs) > cap:
            idx = np.arange(len(recs))
            rng.shuffle(idx)
            recs = [recs[i] for i in idx[:cap]]
        out.extend(recs)
    return out


def _ensure_continual_dev(records: list[dict], frac: float = 0.5, seed: int = 0) -> list[dict]:
    """Move a fraction of val records to a new ``continual_dev`` split.

    Several diagnostics helpers evaluate on ``continual_dev``; this function
    creates that split deterministically from the existing val pool so that
    early-stopping val data and audit data both exist.
    """
    rng = np.random.default_rng(seed)
    val_indices = [i for i, r in enumerate(records) if r.get("split") == "val"]
    if len(val_indices) <= 1:
        return records
    rng.shuffle(val_indices)
    n_dev = max(1, int(len(val_indices) * frac))
    dev_indices = set(val_indices[:n_dev])
    for i in dev_indices:
        records[i]["split"] = "continual_dev"
    return records


def build_protocol_b_2022(
    cache_dir: str | None = None,
    seed: int = 42,
    n_low_train: int | None = None,
    n_high_train: int | None = None,
    split_mode: str = "global_structure_group",
    raw_cap: int | None = None,
) -> tuple[list[tuple[str, str, str]], list[list[dict]], dict[str, Any]]:
    """Build Protocol B restricted to the 2022 snapshot.

    Args:
        cache_dir: JARVIS cache directory.
        seed: Random seed for splitting.
        n_low_train: Cap on OPT training records (None = all).
        n_high_train: Cap on MBJ training records (None = all).
        split_mode: ``global_structure_group`` uses a single global split;
            anything else falls back to paired formula-disjoint splits.
        raw_cap: Optional cap on raw records loaded before pairing (for smoke
            tests).  Applied deterministically with ``seed``.

    Returns:
        tasks, task_records, audit.
    """
    d22 = load_jarvis_dataset("dft_3d", cache_dir)
    if raw_cap is not None and len(d22) > raw_cap:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(d22))
        rng.shuffle(idx)
        d22 = [d22[i] for i in idx[:raw_cap]]
    opt_recs, mbj_recs = _pair_bandgaps_2022(d22)

    if split_mode == "global_structure_group":
        # Use the data.py canonical material group split feature.
        combined = opt_recs + mbj_recs
        assign_global_splits(combined, seed=seed, train_frac=0.70, val_frac=0.15)
    else:
        # Fallback to the paired formula-disjoint split already in data.py.
        from data import assign_paired_splits
        assign_paired_splits(opt_recs, mbj_recs, seed=seed)

    # Create continual_dev split needed by diagnostics helpers.
    opt_recs = _ensure_continual_dev(opt_recs, frac=0.5, seed=seed)
    mbj_recs = _ensure_continual_dev(mbj_recs, frac=0.5, seed=seed + 1)

    opt_recs = _cap_split(opt_recs, n_low_train, seed=seed)
    mbj_recs = _cap_split(mbj_recs, n_high_train, seed=seed + 1)

    tasks = [
        ("dft_3d", "band_gap", "OptB88vdW"),
        ("dft_3d", "band_gap", "TB-mBJ"),
    ]
    task_records = [opt_recs, mbj_recs]

    def split_counts(recs: list[dict]) -> dict[str, int]:
        return {
            "train": sum(1 for r in recs if r.get("split") == "train"),
            "val": sum(1 for r in recs if r.get("split") == "val"),
            "continual_dev": sum(1 for r in recs if r.get("split") == "continual_dev"),
            "test": sum(1 for r in recs if r.get("split") == "test"),
        }

    audit = {
        "snapshot": "2022",
        "split_mode": split_mode,
        "seed": seed,
        "n_low_train_cap": n_low_train,
        "n_high_train_cap": n_high_train,
        "raw_cap": raw_cap,
        "opt_split": split_counts(opt_recs),
        "mbj_split": split_counts(mbj_recs),
    }
    return tasks, task_records, audit


# ---------------------------------------------------------------------------
# Reproducibility / environment artifacts
# ---------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def _git_status() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def _environment_info() -> dict[str, Any]:
    info = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
    for pkg in ("numpy", "pandas", "pyarrow"):
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            info[pkg] = "not installed"
    return info


def _checkpoint_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _save_json(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    torch.save(state, path)


def _save_predictions(path: Path, df: Any) -> None:
    """Save predictions as parquet if possible, otherwise CSV."""
    suffix = path.suffix
    try:
        import pyarrow  # noqa: F401
        if suffix != ".parquet":
            path = path.with_suffix(".parquet")
        df.to_parquet(path, index=False)
    except Exception:
        path = path.with_suffix(".csv")
        df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Metrics and accounting
# ---------------------------------------------------------------------------


def _evaluate_split(
    model: nn.Module,
    records: list[dict],
    prop_id: int,
    fid_id: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    mad: float,
    batch_size: int,
    device: torch.device,
    split: str = "test",
) -> tuple[float, float, torch.Tensor]:
    """Return (MAE in eV, nMAE, predictions) for a split."""
    ds = JARVISCrystalDataset(records, split=split)
    ds.target_mean = float(mean)
    ds.target_std = float(std)
    ds.normalize_target = True
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            node_feats, coords, mask, original_mask, y = batch
            node_feats = node_feats.to(device)
            coords = coords.to(device)
            mask = mask.to(device)
            original_mask = original_mask.to(device)
            pred_norm = model(node_feats, coords, mask, original_mask, prop_id, fid_id)
            pred = pred_norm * std.to(device) + mean.to(device)
            preds.append(pred.cpu())
            targets.append(y)
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    mae = float(torch.abs(preds - targets).mean())
    nmae = float(normalized_mae(preds, targets, mad))
    return mae, nmae, preds


def _parent_route_drift(
    model: ContinualCrystalModel,
    opt_records: list[dict],
    opt_prop_id: int,
    opt_fid_id: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[float, str, str]:
    """Compute max absolute prediction drift plus state-dict hashes before/after.

    This helper snapshots the parent route, evaluates it, and reports the drift.
    For methods that do not modify the parent route the drift should be zero.
    """
    state_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    hash_before = _state_dict_hash(state_before)
    preds_before = _snapshot_opt_predictions(
        model, [opt_records], opt_prop_id, opt_fid_id, mean, std, batch_size, device
    )

    # Re-evaluate to get "after" hashes/predictions.
    state_after = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    hash_after = _state_dict_hash(state_after)
    preds_after = _snapshot_opt_predictions(
        model, [opt_records], opt_prop_id, opt_fid_id, mean, std, batch_size, device
    )
    drift = float((preds_after - preds_before).abs().max())
    return drift, hash_before, hash_after


def _measure_inference_latency(
    model: nn.Module,
    loader: DataLoader,
    prop_id: int,
    fid_id: int,
    device: torch.device,
    n_batches: int = 10,
) -> float:
    """Average per-batch inference latency on the CPU/GPU timer."""
    model.eval()
    times = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            node_feats, coords, mask, original_mask, _ = batch
            node_feats = node_feats.to(device)
            coords = coords.to(device)
            mask = mask.to(device)
            original_mask = original_mask.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = model(node_feats, coords, mask, original_mask, prop_id, fid_id)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)
    return float(np.mean(times[1:])) if len(times) > 1 else float(np.mean(times))


def _optimizer_state_bytes(optimizer: torch.optim.Optimizer | None) -> int:
    if optimizer is None:
        return 0
    total = 0
    for group in optimizer.param_groups:
        for p in group.get("params", []):
            state = optimizer.state.get(p, {})
            for v in state.values():
                if isinstance(v, torch.Tensor):
                    total += v.numel() * v.element_size()
    return total


def _model_size_bytes(model: nn.Module) -> int:
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total


def _count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _count_total(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _fr_phytca_incremental_params(hidden_dim: int, rank: int, n_layers: int = 3) -> int:
    per_layer = (
        hidden_dim * rank  # u_in
        + rank * rank  # core
        + hidden_dim * rank  # u_out
    )
    return n_layers * per_layer + (hidden_dim + 1)  # new head


def _find_bottleneck_for_target(hidden_dim: int, target_params: int) -> int:
    """Bottleneck width for a 2-layer SiLU MLP with ~target_params trainable weights."""
    best = None
    best_diff = float("inf")
    for mid in range(1, hidden_dim * 4 + 1):
        params = (hidden_dim * mid + mid) + (mid * 1 + 1)
        diff = abs(params - target_params)
        if diff < best_diff:
            best_diff = diff
            best = mid
    assert best is not None
    return best


# ---------------------------------------------------------------------------
# OPT parent training (shared across all child methods)
# ---------------------------------------------------------------------------


def _train_opt_parent(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Train the shared OPT parent and return a checkpoint bundle.

    Uses the val split for the parent validation metric (the original
    ``diagnostics.train_opt_parent`` expects a ``continual_dev`` split that is
    not produced by the 2022-only builder).
    """
    prop2id, fid2id = _name_to_id(tasks)
    opt_prop_id = prop2id["band_gap"]
    opt_fid_id = fid2id["OptB88vdW"]

    epochs = args.parent_epochs if not args.smoke else 2
    batch_size = args.batch_size
    patience = args.patience if not args.smoke else 1

    model = ContinualCrystalModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=len(prop2id),
        n_fidelities=len(fid2id),
        adapter_name="single_child_tucker",
        adapter_rank=args.parent_adapter_rank,
        n_layers=args.n_layers,
        num_nearest_neighbors=args.num_nearest_neighbors,
    ).to(device)
    model.add_task(opt_prop_id, opt_fid_id)

    train_loader, val_loader, _, mean, std, mad = _make_eval_loaders(
        task_records[0], batch_size
    )

    optimizer = torch.optim.AdamW(
        model.current_trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_nmae = float("inf")
    best_state = None
    patience_counter = 0

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            node_feats, coords, mask, original_mask, y = batch
            node_feats = node_feats.to(device)
            coords = coords.to(device)
            mask = mask.to(device)
            original_mask = original_mask.to(device)
            y_norm = ((y.to(device) - mean.to(device)) / std.to(device)).float()
            optimizer.zero_grad()
            pred = model(node_feats, coords, mask, original_mask, opt_prop_id, opt_fid_id)
            loss = F.mse_loss(pred, y_norm)
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_nmae = evaluate_loader(
            model, val_loader, opt_prop_id, opt_fid_id, mean, std, mad, device
        )
        if val_nmae < best_nmae:
            best_nmae = val_nmae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Evaluate on val as the T1@T1 proxy.
    t1_after_t1 = evaluate_loader(
        model, val_loader, opt_prop_id, opt_fid_id, mean, std, mad, device
    )

    model.eval()
    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in val_loader:
            node_feats, coords, mask, original_mask, _ = batch
            node_feats = node_feats.to(device)
            coords = coords.to(device)
            mask = mask.to(device)
            original_mask = original_mask.to(device)
            pred = model(node_feats, coords, mask, original_mask, opt_prop_id, opt_fid_id)
            preds.append(pred.detach().cpu())
    opt_predictions = torch.cat(preds)

    state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return {
        "state_dict": state_dict,
        "mean": mean,
        "std": std,
        "mad": mad,
        "task1_after_task1": t1_after_t1,
        "opt_predictions": opt_predictions,
        "state_dict_hash": _state_dict_hash(state_dict),
        "prediction_hash": _state_dict_hash({"pred": opt_predictions}),
    }


# ---------------------------------------------------------------------------
# Method implementations
# ---------------------------------------------------------------------------


def _make_eval_loaders(
    recs: list[dict], batch_size: int
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, torch.Tensor, float]:
    """Thin wrapper around diagnostics._make_loaders for a single task."""
    return _make_loaders(recs, batch_size)


def run_frozen_parent_only(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    opt_parent_state: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate the frozen OPT parent on OPT and MBJ test sets (no child training)."""
    prop2id, fid2id = _name_to_id(tasks)
    opt_prop_id = prop2id["band_gap"]
    opt_fid_id = fid2id["OptB88vdW"]
    mbj_fid_id = fid2id["TB-mBJ"]

    model = ContinualCrystalModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=len(prop2id),
        n_fidelities=len(fid2id),
        adapter_name="single_child_tucker",
        adapter_rank=args.adapter_rank,
        n_layers=args.n_layers,
        num_nearest_neighbors=args.num_nearest_neighbors,
    ).to(device)
    model.add_task(opt_prop_id, opt_fid_id)
    model.load_state_dict(
        {k: v.to(device) for k, v in opt_parent_state["state_dict"].items()},
        strict=False,
    )
    for p in model.parameters():
        p.requires_grad = False

    mean, std, mad = opt_parent_state["mean"], opt_parent_state["std"], opt_parent_state["mad"]
    t1_after_t1 = opt_parent_state["task1_after_task1"]

    opt_mae, opt_nmae, opt_preds = _evaluate_split(
        model, task_records[0], opt_prop_id, opt_fid_id, mean, std, mad, args.batch_size, device
    )
    # Evaluate OPT predictions on MBJ targets as a zero-transfer baseline.
    mbj_mae, mbj_nmae, mbj_preds = _evaluate_split(
        model, task_records[1], opt_prop_id, opt_fid_id, mean, std, mad, args.batch_size, device
    )

    return {
        "method": "frozen_parent_only",
        "opt_mae": opt_mae,
        "opt_nmae": opt_nmae,
        "mbj_mae": mbj_mae,
        "mbj_nmae": mbj_nmae,
        "task1_after_task1": t1_after_t1,
        "task1_after_task2": opt_nmae,
        "absolute_forgetting": 0.0,
        "opt_route_drift": 0.0,
        "incremental_params": 0,
        "trainable_params": 0,
        "stored_params": _count_total(model),
        "wall_train_seconds": 0.0,
        "opt_predictions": opt_preds,
        "mbj_predictions": mbj_preds,
    }


def run_private_top_block(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    opt_parent_state: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """FR-PhyTCA + copy-on-write top encoder block for the MBJ child."""
    from diagnostics import d6_progressive_tucker

    prop2id, fid2id = _name_to_id(tasks)
    opt_prop_id = prop2id["band_gap"]
    opt_fid_id = fid2id["OptB88vdW"]
    mbj_fid_id = fid2id["TB-mBJ"]

    model = ContinualCrystalModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=len(prop2id),
        n_fidelities=len(fid2id),
        adapter_name=args.adapter_name,
        adapter_rank=args.adapter_rank,
        n_layers=args.n_layers,
        num_nearest_neighbors=args.num_nearest_neighbors,
    ).to(device)

    start = time.perf_counter()
    model.add_task(opt_prop_id, opt_fid_id)
    model.load_state_dict(
        {k: v.to(device) for k, v in opt_parent_state["state_dict"].items()},
        strict=False,
    )
    mean, std, mad = opt_parent_state["mean"], opt_parent_state["std"], opt_parent_state["mad"]
    t1_after_t1 = opt_parent_state["task1_after_task1"]

    # Snapshot before adding the private top block / MBJ task.
    opt_preds_before = _snapshot_opt_predictions(
        model, task_records, opt_prop_id, opt_fid_id, mean, std, args.batch_size, device
    )
    hash_before = _state_dict_hash({k: v.cpu() for k, v in model.state_dict().items()})

    model.freeze_task(opt_prop_id, opt_fid_id)
    model.add_task(opt_prop_id, mbj_fid_id)
    model.add_private_top_block(opt_prop_id, mbj_fid_id)

    train_loader2, val_loader2, _, mean2, std2, mad2 = _make_eval_loaders(
        task_records[1], args.batch_size
    )
    optimizer = torch.optim.AdamW(
        model.current_trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_nmae = float("inf")
    best_state = None
    patience_counter = 0
    wall_train_start = time.perf_counter()
    for _ in range(args.epochs):
        model.train()
        for batch in train_loader2:
            node_feats, coords, mask, original_mask, y = batch
            node_feats = node_feats.to(device)
            coords = coords.to(device)
            mask = mask.to(device)
            original_mask = original_mask.to(device)
            y_norm = ((y.to(device) - mean2.to(device)) / std2.to(device)).float()
            optimizer.zero_grad()
            pred = model(node_feats, coords, mask, original_mask, opt_prop_id, mbj_fid_id)
            loss = F.mse_loss(pred, y_norm)
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_nmae = evaluate_loader(
            model, val_loader2, opt_prop_id, mbj_fid_id, mean2, std2, mad2, device
        )
        if val_nmae < best_nmae:
            best_nmae = val_nmae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            break
    wall_train = time.perf_counter() - wall_train_start

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    opt_preds_after = _snapshot_opt_predictions(
        model, task_records, opt_prop_id, opt_fid_id, mean, std, args.batch_size, device
    )
    hash_after = _state_dict_hash({k: v.cpu() for k, v in model.state_dict().items()})
    opt_route_drift = float((opt_preds_after - opt_preds_before).abs().max())

    opt_mae, opt_nmae, opt_preds = _evaluate_split(
        model, task_records[0], opt_prop_id, opt_fid_id, mean, std, mad, args.batch_size, device
    )
    mbj_mae, mbj_nmae, mbj_preds = _evaluate_split(
        model, task_records[1], opt_prop_id, mbj_fid_id, mean2, std2, mad2, args.batch_size, device
    )

    # Inference latency on MBJ test loader.
    test_loader_mbj = DataLoader(
        JARVISCrystalDataset(task_records[1], split="test"),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_crystals,
    )
    latency = _measure_inference_latency(
        model, test_loader_mbj, opt_prop_id, mbj_fid_id, device, n_batches=args.latency_batches
    )

    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        torch.cuda.reset_peak_memory_stats(device)

    return {
        "method": "private_top_block",
        "opt_mae": opt_mae,
        "opt_nmae": opt_nmae,
        "mbj_mae": mbj_mae,
        "mbj_nmae": mbj_nmae,
        "task1_after_task1": t1_after_t1,
        "task1_after_task2": opt_nmae,
        "absolute_forgetting": opt_nmae - t1_after_t1,
        "opt_route_drift": opt_route_drift,
        "state_dict_hash_before": hash_before,
        "state_dict_hash_after": hash_after,
        "incremental_params": model.count_incremental_parameters(opt_prop_id, mbj_fid_id),
        "trainable_params": _count_trainable(model),
        "stored_params": _count_total(model),
        "checkpoint_bytes": _model_size_bytes(model),
        "optimizer_state_bytes": _optimizer_state_bytes(optimizer),
        "peak_memory_mb": peak_memory_mb,
        "wall_train_seconds": wall_train,
        "inference_latency_seconds": latency,
        "opt_predictions": opt_preds,
        "mbj_predictions": mbj_preds,
    }


def run_fr_phytca_variant(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    opt_parent_state: dict[str, Any],
    args: argparse.Namespace,
    adapter_name: str,
    method_label: str,
) -> dict[str, Any]:
    """Run d6_progressive_tucker with a chosen adapter and extra bookkeeping."""
    from diagnostics import d6_progressive_tucker

    result = d6_progressive_tucker(
        tasks=tasks,
        task_records=task_records,
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        adapter_rank=args.adapter_rank,
        num_nearest_neighbors=args.num_nearest_neighbors,
        opt_parent_state=opt_parent_state,
        experiment_label=method_label,
        adapter_name=adapter_name,
    )

    prop2id, fid2id = _name_to_id(tasks)
    opt_prop_id = prop2id["band_gap"]
    opt_fid_id = fid2id["OptB88vdW"]
    mbj_fid_id = fid2id["TB-mBJ"]

    mean, std, mad = opt_parent_state["mean"], opt_parent_state["std"], opt_parent_state["mad"]
    _, _, _, mean2, std2, mad2 = _make_eval_loaders(task_records[1], args.batch_size)

    model = result.get("model")
    opt_mae, opt_nmae, opt_preds = _evaluate_split(
        model, task_records[0], opt_prop_id, opt_fid_id, mean, std, mad, args.batch_size, device
    )
    mbj_mae, mbj_nmae, mbj_preds = _evaluate_split(
        model, task_records[1], opt_prop_id, mbj_fid_id, mean2, std2, mad2, args.batch_size, device
    )

    test_loader_mbj = DataLoader(
        JARVISCrystalDataset(task_records[1], split="test"),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_crystals,
    )
    latency = _measure_inference_latency(
        model, test_loader_mbj, opt_prop_id, mbj_fid_id, device, n_batches=args.latency_batches
    )

    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        torch.cuda.reset_peak_memory_stats(device)

    result.update(
        {
            "method": method_label,
            "opt_mae": opt_mae,
            "opt_nmae": opt_nmae,
            "mbj_mae": mbj_mae,
            "mbj_nmae": mbj_nmae,
            "inference_latency_seconds": latency,
            "peak_memory_mb": peak_memory_mb,
            "checkpoint_bytes": _model_size_bytes(model),
            "stored_params": _count_total(model),
            "trainable_params": _count_trainable(model),
            "opt_predictions": opt_preds,
            "mbj_predictions": mbj_preds,
        }
    )
    return result


def run_frozen_correction(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    opt_parent_state: dict[str, Any],
    args: argparse.Namespace,
    affine: bool,
) -> dict[str, Any]:
    """Frozen parent + affine/residual correction using diagnostics helpers."""
    from diagnostics import d4_frozen_opt_affine, d5_frozen_opt_residual

    fn = d4_frozen_opt_affine if affine else d5_frozen_opt_residual
    label = "frozen_parent_affine_correction" if affine else "frozen_parent_residual_correction"
    result = fn(
        tasks=tasks,
        task_records=task_records,
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        adapter_rank=args.adapter_rank,
        num_nearest_neighbors=args.num_nearest_neighbors,
        opt_parent_state=opt_parent_state,
    )

    prop2id, fid2id = _name_to_id(tasks)
    opt_prop_id = prop2id["band_gap"]
    opt_fid_id = fid2id["OptB88vdW"]
    mbj_fid_id = fid2id["TB-mBJ"]

    mean, std, mad = opt_parent_state["mean"], opt_parent_state["std"], opt_parent_state["mad"]
    _, _, _, mean2, std2, mad2 = _make_eval_loaders(task_records[1], args.batch_size)

    model = result.get("model")
    opt_mae, opt_nmae, opt_preds = _evaluate_split(
        model, task_records[0], opt_prop_id, opt_fid_id, mean, std, mad, args.batch_size, device
    )
    mbj_mae, mbj_nmae, mbj_preds = _evaluate_split(
        model, task_records[1], opt_prop_id, mbj_fid_id, mean2, std2, mad2, args.batch_size, device
    )

    test_loader_mbj = DataLoader(
        JARVISCrystalDataset(task_records[1], split="test"),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_crystals,
    )
    latency = _measure_inference_latency(
        model, test_loader_mbj, opt_prop_id, mbj_fid_id, device, n_batches=args.latency_batches
    )

    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        torch.cuda.reset_peak_memory_stats(device)

    result.update(
        {
            "method": label,
            "opt_mae": opt_mae,
            "opt_nmae": opt_nmae,
            "mbj_mae": mbj_mae,
            "mbj_nmae": mbj_nmae,
            "inference_latency_seconds": latency,
            "peak_memory_mb": peak_memory_mb,
            "checkpoint_bytes": _model_size_bytes(model),
            "stored_params": _count_total(model),
            "trainable_params": _count_trainable(model),
            "opt_predictions": opt_preds,
            "mbj_predictions": mbj_preds,
        }
    )
    return result


def run_param_matched_mlp(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    opt_parent_state: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Frozen parent + parameter-matched MLP residual."""
    from diagnostics import d6h_matched_mlp_residual

    result = d6h_matched_mlp_residual(
        tasks=tasks,
        task_records=task_records,
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        adapter_rank=args.adapter_rank,
        num_nearest_neighbors=args.num_nearest_neighbors,
        opt_parent_state=opt_parent_state,
    )
    return _postprocess_baseline_result(result, tasks, task_records, opt_parent_state, args, "param_matched_mlp", device)


def run_param_matched_bottleneck(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    opt_parent_state: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Frozen parent + parameter-matched bottleneck adapter inserted at the pooled representation."""
    from diagnostics import _frozen_opt_residual_experiment

    target = _fr_phytca_incremental_params(hidden_dim, args.adapter_rank, n_layers=args.n_layers)
    bottleneck = _find_bottleneck_for_target(hidden_dim, target)
    correction = nn.Sequential(
        nn.Linear(hidden_dim, bottleneck),
        nn.SiLU(),
        nn.Linear(bottleneck, 1),
    )
    result = _frozen_opt_residual_experiment(
        tasks=tasks,
        task_records=task_records,
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        device=device,
        base_state_dict=None,
        correction_module=correction,
        label="param_matched_bottleneck",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        adapter_rank=args.adapter_rank,
        num_nearest_neighbors=args.num_nearest_neighbors,
        opt_parent_state=opt_parent_state,
    )
    return _postprocess_baseline_result(result, tasks, task_records, opt_parent_state, args, "param_matched_bottleneck", device)


def run_full_child_finetuning(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    opt_parent_state: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Pre-train on OPT, then unfreeze everything and fine-tune on MBJ."""
    from diagnostics import opt_pretrain_mbj_full_finetune

    result = opt_pretrain_mbj_full_finetune(
        tasks=tasks,
        task_records=task_records,
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        adapter_rank=args.adapter_rank,
        num_nearest_neighbors=args.num_nearest_neighbors,
        opt_parent_state=opt_parent_state,
    )
    return _postprocess_baseline_result(result, tasks, task_records, opt_parent_state, args, "full_child_finetuning", device)


def run_joint_opt_mbj(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    opt_parent_state: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Joint training upper bound: OPT + MBJ together with full fine-tuning."""
    from diagnostics import d1_full_joint

    result = d1_full_joint(
        tasks=tasks,
        task_records=task_records,
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        adapter_rank=args.adapter_rank,
        num_nearest_neighbors=args.num_nearest_neighbors,
    )
    return _postprocess_baseline_result(result, tasks, task_records, opt_parent_state, args, "joint_opt_mbj", device)


def _postprocess_baseline_result(
    result: dict[str, Any],
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    opt_parent_state: dict[str, Any],
    args: argparse.Namespace,
    method_label: str,
    device: torch.device,
) -> dict[str, Any]:
    """Add uniform OPT/MBJ MAE metrics and latency to a diagnostics result dict."""
    prop2id, fid2id = _name_to_id(tasks)
    opt_prop_id = prop2id["band_gap"]
    opt_fid_id = fid2id["OptB88vdW"]
    mbj_fid_id = fid2id["TB-mBJ"]

    mean, std, mad = opt_parent_state["mean"], opt_parent_state["std"], opt_parent_state["mad"]
    _, _, _, mean2, std2, mad2 = _make_eval_loaders(task_records[1], args.batch_size)

    model = result.get("model")
    opt_mae, opt_nmae, opt_preds = _evaluate_split(
        model, task_records[0], opt_prop_id, opt_fid_id, mean, std, mad, args.batch_size, device
    )
    mbj_mae, mbj_nmae, mbj_preds = _evaluate_split(
        model, task_records[1], opt_prop_id, mbj_fid_id, mean2, std2, mad2, args.batch_size, device
    )

    test_loader_mbj = DataLoader(
        JARVISCrystalDataset(task_records[1], split="test"),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_crystals,
    )
    latency = _measure_inference_latency(
        model, test_loader_mbj, opt_prop_id, mbj_fid_id, device, n_batches=args.latency_batches
    )

    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        torch.cuda.reset_peak_memory_stats(device)

    result.update(
        {
            "method": method_label,
            "opt_mae": opt_mae,
            "opt_nmae": opt_nmae,
            "mbj_mae": mbj_mae,
            "mbj_nmae": mbj_nmae,
            "inference_latency_seconds": latency,
            "peak_memory_mb": peak_memory_mb,
            "checkpoint_bytes": _model_size_bytes(model),
            "stored_params": _count_total(model),
            "trainable_params": _count_trainable(model),
            "opt_predictions": opt_preds,
            "mbj_predictions": mbj_preds,
            "wall_train_seconds": _wall_train_seconds(result),
        }
    )
    return result


METHOD_RUNNERS: dict[str, Any] = {
    "frozen_parent_only": run_frozen_parent_only,
    "frozen_parent_affine_correction": lambda *a, **k: run_frozen_correction(*a, **k, affine=True),
    "frozen_parent_residual_correction": lambda *a, **k: run_frozen_correction(*a, **k, affine=False),
    "fr_lora_ab": lambda *a, **k: run_fr_phytca_variant(*a, **k, adapter_name="lora_ab", method_label="fr_lora_ab"),
    "fr_lora_aba": lambda *a, **k: run_fr_phytca_variant(*a, **k, adapter_name="lora_aba", method_label="fr_lora_aba"),
    "fr_single_child_tucker": lambda *a, **k: run_fr_phytca_variant(
        *a, **k, adapter_name="single_child_tucker", method_label="fr_single_child_tucker"
    ),
    "param_matched_bottleneck": run_param_matched_bottleneck,
    "param_matched_mlp": run_param_matched_mlp,
    "private_top_block": run_private_top_block,
    "full_child_finetuning": run_full_child_finetuning,
    "joint_opt_mbj": run_joint_opt_mbj,
}


# ---------------------------------------------------------------------------
# Per-run artifact saving
# ---------------------------------------------------------------------------


def _wall_train_seconds(result: dict[str, Any]) -> float:
    """Return wall-clock training time, falling back from diagnostics naming."""
    if result.get("wall_train_seconds") is not None:
        return float(result["wall_train_seconds"])
    if result.get("wall_time_seconds") is not None:
        return float(result["wall_time_seconds"])
    return 0.0


def _save_run_artifacts(
    run_dir: Path,
    config: dict[str, Any],
    result: dict[str, Any],
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    opt_parent_state: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Write config, metrics, predictions, timing, memory, hashes, etc."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # config.yaml
    _save_yaml(run_dir / "config.yaml", config)

    # git_commit.txt
    (run_dir / "git_commit.txt").write_text(
        f"{_git_commit()}\n\nStatus:\n{_git_status()}", encoding="utf-8"
    )

    # environment.json
    _save_json(run_dir / "environment.json", _environment_info())

    # split_manifest.json
    prop2id, fid2id = _name_to_id(tasks)
    manifest = {
        "tasks": [f"{t[0]}_{t[1]}_{t[2]}" for t in tasks],
        "splits": {
            f"{t[0]}_{t[1]}_{t[2]}": {
                "train": sum(1 for r in recs if r.get("split") == "train"),
                "val": sum(1 for r in recs if r.get("split") == "val"),
                "test": sum(1 for r in recs if r.get("split") == "test"),
            }
            for t, recs in zip(tasks, task_records)
        },
    }
    _save_json(run_dir / "split_manifest.json", manifest)

    # parameter_breakdown.json
    model = result.get("model")
    param_breakdown: dict[str, Any] = {"incremental_params": result.get("incremental_params")}
    if hasattr(model, "get_parameter_group_counts"):
        param_breakdown.update(model.get_parameter_group_counts())
    else:
        param_breakdown["trainable_params"] = result.get("trainable_params")
        param_breakdown["stored_params"] = result.get("stored_params")
    _save_json(run_dir / "parameter_breakdown.json", param_breakdown)

    # metrics.json
    metrics = {
        "method": result.get("method"),
        "opt_mae": result.get("opt_mae"),
        "opt_nmae": result.get("opt_nmae"),
        "mbj_mae": result.get("mbj_mae"),
        "mbj_nmae": result.get("mbj_nmae"),
        "task1_after_task1": result.get("task1_after_task1"),
        "task1_after_task2": result.get("task1_after_task2"),
        "absolute_forgetting": result.get("absolute_forgetting"),
        "opt_route_drift": result.get("opt_route_drift"),
        "incremental_params": result.get("incremental_params"),
        "trainable_params": result.get("trainable_params"),
        "stored_params": result.get("stored_params"),
        "checkpoint_bytes": result.get("checkpoint_bytes"),
        "optimizer_state_bytes": result.get("optimizer_state_bytes"),
        "peak_memory_mb": result.get("peak_memory_mb"),
        "wall_train_seconds": _wall_train_seconds(result),
        "inference_latency_seconds": result.get("inference_latency_seconds"),
    }
    _save_json(run_dir / "metrics.json", metrics)

    # predictions.parquet / csv
    try:
        import pandas as pd
    except Exception:
        pd = None
    if pd is not None:
        opt_preds = result.get("opt_predictions")
        mbj_preds = result.get("mbj_predictions")
        rows = []
        opt_test_recs = [r for r in task_records[0] if r.get("split") == "test"]
        mbj_test_recs = [r for r in task_records[1] if r.get("split") == "test"]
        if opt_preds is not None and len(opt_preds) == len(opt_test_recs):
            for r, p in zip(opt_test_recs, opt_preds.tolist()):
                rows.append({"jid": r.get("jid"), "split": "test", "fidelity": "OptB88vdW", "prediction": p})
        if mbj_preds is not None and len(mbj_preds) == len(mbj_test_recs):
            for r, p in zip(mbj_test_recs, mbj_preds.tolist()):
                rows.append({"jid": r.get("jid"), "split": "test", "fidelity": "TB-mBJ", "prediction": p})
        if rows:
            _save_predictions(run_dir / "predictions.parquet", pd.DataFrame(rows))

    # timing.json
    timing = {
        "wall_train_seconds": _wall_train_seconds(result),
        "inference_latency_seconds": result.get("inference_latency_seconds"),
    }
    _save_json(run_dir / "timing.json", timing)

    # memory.json
    memory = {"peak_memory_mb": result.get("peak_memory_mb", 0.0)}
    _save_json(run_dir / "memory.json", memory)

    # checkpoint_hashes.json
    checkpoint_path = run_dir / "checkpoint.pt"
    if model is not None:
        _save_checkpoint(checkpoint_path, {"model_state": model.state_dict(), "config": config})
    hashes = {
        "state_dict_hash_before": result.get("state_dict_hash_before"),
        "state_dict_hash_after": result.get("state_dict_hash_after"),
    }
    if checkpoint_path.exists():
        hashes["checkpoint_sha256"] = _checkpoint_hash(checkpoint_path)
    _save_json(run_dir / "checkpoint_hashes.json", hashes)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decisive Protocol B experiment")
    parser.add_argument("--snapshot", default="2022", help="JARVIS snapshot year (only 2022 supported)")
    parser.add_argument(
        "--low-train-sizes",
        nargs="+",
        type=int_or_none,
        default=DEFAULT_LOW_TRAIN_SIZES,
        help="OPT training-set sizes to sweep (use 'all' for no cap)",
    )
    parser.add_argument(
        "--high-train-sizes",
        nargs="+",
        type=int_or_none,
        default=DEFAULT_HIGH_TRAIN_SIZES,
        help="MBJ training-set sizes to sweep (use 'all' for no cap)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="Methods to compare",
    )
    parser.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        default=DEFAULT_RANKS,
        help="Adapter ranks to sweep",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds",
    )
    parser.add_argument(
        "--split-mode",
        default="global_structure_group",
        choices=["global_structure_group", "paired_formula"],
        help="Split strategy",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "runs" / "protocol_b_decisive"),
        help="Directory to save run artifacts",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension of the crystal encoder",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=3,
        help="Number of crystal-graph encoder layers",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Max child-training epochs",
    )
    parser.add_argument(
        "--parent-epochs",
        type=int,
        default=20,
        help="Max OPT parent-training epochs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early-stopping patience",
    )
    parser.add_argument(
        "--parent-adapter-rank",
        type=int,
        default=8,
        help="Rank used for the OPT parent adapter bank",
    )
    parser.add_argument(
        "--num-nearest-neighbors",
        type=int,
        default=8,
        help="EGNN kNN parameter",
    )
    parser.add_argument(
        "--latency-batches",
        type=int,
        default=10,
        help="Number of test batches used to estimate inference latency",
    )
    parser.add_argument(
        "--raw-cap",
        type=int,
        default=None,
        help="Cap raw records before pairing (useful for fast mini-runs without --smoke)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast smoke test: cap samples and epochs",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.snapshot != "2022":
        raise ValueError(f"Only snapshot 2022 is supported; got {args.snapshot}")

    # Smoke-test overrides.
    if args.smoke:
        args.low_train_sizes = [50]
        args.high_train_sizes = [50]
        args.methods = ["frozen_parent_only", "fr_single_child_tucker", "private_top_block"]
        args.ranks = [4]
        args.seeds = [42]
        args.epochs = 2
        args.parent_epochs = 2
        args.patience = 1
        args.latency_batches = 2
        print("SMOKE TEST MODE: using minimal grid")

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    node_dim = 92
    hidden_dim = args.hidden_dim

    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print(f"Sweep grid: N_L={args.low_train_sizes}, N_H={args.high_train_sizes}")
    print(f"Methods: {args.methods}")
    print(f"Ranks: {args.ranks}, Seeds: {args.seeds}")

    summary_rows = []

    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Build the data once per seed (full OPT, varying MBJ caps are applied later).
        raw_cap = args.raw_cap if args.raw_cap is not None else (500 if args.smoke else None)
        tasks, task_records_full, audit = build_protocol_b_2022(
            cache_dir=None,
            seed=seed,
            n_low_train=None,
            n_high_train=None,
            split_mode=args.split_mode,
            raw_cap=raw_cap,
        )

        for rank in args.ranks:
            # Train one OPT parent per rank so that the parent state dict is
            # compatible with the child models trained at this rank.
            args.adapter_rank = rank
            args.parent_adapter_rank = rank
            opt_parent_state = _train_opt_parent(
                tasks, task_records_full, node_dim, hidden_dim, device, args
            )

            for n_low in args.low_train_sizes:
                for n_high in args.high_train_sizes:
                    # Apply caps while preserving the global split.
                    opt_recs = _cap_split(copy.deepcopy(task_records_full[0]), n_low, seed=seed + 10)
                    mbj_recs = _cap_split(copy.deepcopy(task_records_full[1]), n_high, seed=seed + 11)
                    task_records = [opt_recs, mbj_recs]

                    args.adapter_name = "single_child_tucker"

                    for method in args.methods:
                        if method not in METHOD_RUNNERS:
                            warnings.warn(f"Unknown method {method}; skipping")
                            continue

                        config = {
                            "snapshot": args.snapshot,
                            "split_mode": args.split_mode,
                            "seed": seed,
                            "n_low_train": n_low,
                            "n_high_train": n_high,
                            "rank": rank,
                            "method": method,
                            "hidden_dim": hidden_dim,
                            "n_layers": args.n_layers,
                            "batch_size": args.batch_size,
                            "lr": args.lr,
                            "weight_decay": args.weight_decay,
                            "epochs": args.epochs,
                            "parent_epochs": args.parent_epochs,
                            "patience": args.patience,
                            "parent_adapter_rank": args.parent_adapter_rank,
                            "num_nearest_neighbors": args.num_nearest_neighbors,
                        }

                        run_name = (
                            f"seed{seed}_Nl{n_low or 'all'}_Nh{n_high or 'all'}_r{rank}_{method}"
                        )
                        run_dir = output_dir / run_name

                        print(f"\n=== Running {run_name} ===")
                        try:
                            runner = METHOD_RUNNERS[method]
                            result = runner(
                                tasks=tasks,
                                task_records=task_records,
                                node_dim=node_dim,
                                hidden_dim=hidden_dim,
                                device=device,
                                opt_parent_state=opt_parent_state,
                                args=args,
                            )
                            _save_run_artifacts(
                                run_dir, config, result, tasks, task_records, opt_parent_state, args
                            )
                            summary_rows.append(
                                {
                                    "seed": seed,
                                    "n_low": n_low,
                                    "n_high": n_high,
                                    "rank": rank,
                                    "method": method,
                                    "opt_nmae": result.get("opt_nmae"),
                                    "mbj_nmae": result.get("mbj_nmae"),
                                    "opt_route_drift": result.get("opt_route_drift"),
                                    "incremental_params": result.get("incremental_params"),
                                    "stored_params": result.get("stored_params"),
                                    "wall_train_seconds": _wall_train_seconds(result),
                                    "status": "ok",
                                }
                            )
                            print(
                                f"  OPT nMAE={result.get('opt_nmae'):.3f} "
                                f"MBJ nMAE={result.get('mbj_nmae'):.3f} "
                                f"drift={result.get('opt_route_drift', 0):.2e} "
                                f"params={result.get('incremental_params')}"
                            )
                        except Exception as exc:
                            warnings.warn(f"Run {run_name} failed: {exc}")
                            summary_rows.append(
                                {
                                    "seed": seed,
                                    "n_low": n_low,
                                    "n_high": n_high,
                                    "rank": rank,
                                    "method": method,
                                    "status": f"failed: {exc}",
                                }
                            )

    # Save a top-level summary CSV/JSON for the whole sweep.
    try:
        import pandas as pd
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / "summary.csv", index=False)
        _save_json(output_dir / "summary.json", {"rows": summary_rows, "audit": audit})
    except Exception:
        _save_json(output_dir / "summary.json", {"rows": summary_rows, "audit": audit})

    print("\n=== Sweep complete ===")
    print(f"Summary saved to {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
