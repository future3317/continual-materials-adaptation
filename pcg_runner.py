"""Shared helpers for PCG protocol runners."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data import JARVISCrystalDataset, PeriodicGraphBuilder, collate_crystals
from pareto_harness import evaluate_pareto_metrics
from persistent_consolidation_graph import PersistentConsolidationGraph
from train_utils import compute_mad, forgetting, backward_transfer


def build_pcg_encoder_and_graph_builder(
    encoder_type: str,
    hidden_dim: int,
    node_dim: int = 92,
) -> tuple[torch.nn.Module, Any | None]:
    """Create a frozen PCG encoder and optional graph builder for real data."""
    if encoder_type == "matgl":
        from backbones import _MATGL_AVAILABLE, build_matgl_backbone

        if not _MATGL_AVAILABLE:
            raise ImportError("MatGL is not installed; use --encoder-type egnn")
        encoder = build_matgl_backbone(None, hidden_dim=hidden_dim, freeze=True)
        node_feature_dim = encoder._max_element_z
        graph_builder = PeriodicGraphBuilder(node_feature_dim=node_feature_dim)
        return encoder, graph_builder

    if encoder_type == "egnn":
        from models import CrystalEncoder

        encoder = CrystalEncoder(node_dim=node_dim, hidden_dim=hidden_dim, n_layers=3)
        return encoder, None

    raise ValueError(f"Unknown encoder_type: {encoder_type}")


def filter_records_for_encoder(
    recs: list[dict],
    encoder_type: str,
    encoder: torch.nn.Module,
) -> list[dict]:
    """Drop records that the chosen encoder cannot represent."""
    if encoder_type == "matgl":
        element_symbols = set(encoder.element_types)
        return filter_records_by_elements(recs, element_symbols)
    return recs


def cap_records(recs: list[dict], cap: int | None) -> list[dict]:
    """Cap records while preserving train/val/test split membership."""
    if cap is None or len(recs) <= cap:
        return recs
    by_split: dict[str, list[dict]] = {}
    for r in recs:
        by_split.setdefault(r.get("split", "train"), []).append(r)
    capped: list[dict] = []
    per_split_cap = max(1, cap // len(by_split))
    for split_recs in by_split.values():
        capped.extend(split_recs[:per_split_cap])
    return capped


def filter_records_by_elements(recs: list[dict], element_symbols: set[str]) -> list[dict]:
    """Drop records whose structures contain elements outside ``element_symbols``."""
    out: list[dict] = []
    for r in recs:
        struct = r["structure"]
        if all(str(site.specie) in element_symbols for site in struct):
            out.append(r)
    return out


def make_pcg_loaders(
    recs: list[dict],
    batch_size: int,
    graph_builder: Any | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, tuple[float, float], float]:
    """Return train/val/test loaders and normalizer stats for PCG.

    PCG expects physical targets; the endpoint normalizer is stored separately.
    """
    train_recs = [r for r in recs if r.get("split") == "train"]
    targets = torch.tensor([r["target"] for r in train_recs], dtype=torch.float32)
    mean = float(targets.mean())
    std = float(targets.std().clamp_min(1e-6))
    mad = compute_mad(targets)

    train_ds = JARVISCrystalDataset(recs, split="train", normalize_target=False, graph_builder=graph_builder)
    val_ds = JARVISCrystalDataset(recs, split="val", normalize_target=False, graph_builder=graph_builder)
    test_ds = JARVISCrystalDataset(recs, split="test", normalize_target=False, graph_builder=graph_builder)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_crystals)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals)

    return train_loader, val_loader, test_loader, (mean, std), mad


def determine_parents_combined(
    tasks: list[tuple[str, str, str, str]],
    t: int,
    route_keys: list[str],
) -> list[str]:
    """Return parent route keys for task ``t`` in the canonical combined protocol."""
    if t == 0:
        return []
    version, prop, fid, _ = tasks[t]
    prev_version = "dft_3d_2021" if version == "dft_3d" else None

    parents: list[str] = []
    # Temporal revision parent: same property/fidelity, previous version.
    if prev_version is not None:
        for i, (v, p, f, _) in enumerate(tasks[:t]):
            if v == prev_version and p == prop and f == fid:
                parents.append(route_keys[i])
                break

    # Fidelity-transfer parent: most recent same-property endpoint with different fidelity.
    for i in range(t - 1, -1, -1):
        v, p, f, _ = tasks[i]
        if p == prop and f != fid:
            parents.append(route_keys[i])
            break

    return parents


def determine_parents_revision(
    tasks: list[tuple[str, str, str, str]],
    t: int,
    route_keys: list[str],
) -> list[str]:
    """Temporal revision parent only."""
    if t == 0:
        return []
    version, prop, fid, _ = tasks[t]
    prev_version = "dft_3d_2021" if version == "dft_3d" else None
    if prev_version is None:
        return []
    for i, (v, p, f, _) in enumerate(tasks[:t]):
        if v == prev_version and p == prop and f == fid:
            return [route_keys[i]]
    return []


def determine_parents_addition(
    tasks: list[tuple[str, str, str, str]],
    t: int,
    route_keys: list[str],
) -> list[str]:
    """Use the previous same-fidelity endpoint as a parent."""
    return determine_parents_revision(tasks, t, route_keys)


def determine_parents_fidelity_expansion(
    tasks: list[tuple[str, str, str, str]],
    t: int,
    route_keys: list[str],
) -> list[str]:
    """Previous fidelity in the same version as parent."""
    if t == 0:
        return []
    version, prop, fid, _ = tasks[t]
    for i in range(t - 1, -1, -1):
        v, p, f, _ = tasks[i]
        if v == version and p == prop:
            return [route_keys[i]]
    return []


def evaluate_endpoint(
    model: PersistentConsolidationGraph,
    recs: list[dict],
    version: str,
    prop_id: int,
    fid_id: int,
    normalizer: tuple[float, float],
    device: torch.device,
    batch_size: int,
    graph_builder: Any | None = None,
    change_type: str | None = None,
    test_loader: DataLoader | None = None,
) -> dict[str, float]:
    """Evaluate a single endpoint on (a subset of) its test records."""
    if change_type is not None:
        recs = [r for r in recs if r.get("change_type") == change_type]
        if not recs:
            return {"n": 0, "mae": float("nan"), "nmae": float("nan")}
        ds = JARVISCrystalDataset(recs, split="test", normalize_target=False, graph_builder=graph_builder)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals)
    elif test_loader is not None:
        loader = test_loader
    else:
        ds = JARVISCrystalDataset(recs, split="test", normalize_target=False, graph_builder=graph_builder)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals)

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for node_feats, coords, mask, original_mask, y in loader:
            node_feats = node_feats.to(device)
            coords = coords.to(device)
            mask = mask.to(device)
            original_mask = original_mask.to(device)
            pred_phys = model(node_feats, coords, mask, original_mask, version, prop_id, fid_id, physical=True)
            preds.append(pred_phys.cpu())
            targets.append(y)

    if not preds:
        return {"n": 0, "mae": float("nan"), "nmae": float("nan")}

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    mae = float(torch.abs(preds - targets).mean())
    mad = compute_mad(targets)
    nmae = mae / max(mad, 1e-8)
    return {"n": len(recs), "mae": mae, "nmae": nmae}


def run_pcg_protocol(
    protocol_name: str,
    tasks: list[tuple[str, str, str, str]],
    task_records: list[list[dict]],
    model: PersistentConsolidationGraph,
    prop2id: dict[str, int],
    fid2id: dict[str, int],
    device: torch.device,
    batch_size: int,
    epochs_fast: int,
    epochs_cons: int,
    lr: float,
    output_dir: Path,
    parent_fn: Any,
    graph_builder: Any | None = None,
    eval_subsets: dict[int, list[str]] | None = None,
) -> dict[str, Any]:
    """Train a sequence of PCG endpoints and evaluate them."""
    output_dir.mkdir(parents=True, exist_ok=True)

    route_keys: list[str] = []
    normalizers: list[tuple[float, float]] = []
    mads: list[float] = []
    test_loaders: list[DataLoader] = []
    wall_times: list[float] = []
    nmaes: list[list[float]] = []
    per_task_subset_metrics: list[dict[str, dict[str, float]]] = []
    per_route_info: list[dict[str, Any]] = []

    start_all = time.perf_counter()
    for t, (version, prop, fid, _) in enumerate(tasks):
        pid = prop2id[prop]
        fid_id = fid2id[fid]
        parents = parent_fn(tasks, t, route_keys)
        key = model.add_route(version, pid, fid_id, parent_ids=parents)
        route_keys.append(key)

        print(f"\n=== Route {t + 1}/{len(tasks)}: {version} / {prop} / {fid} ===")
        print(f"  parents: {parents}")

        train_loader, val_loader, test_loader, normalizer, mad = make_pcg_loaders(
            task_records[t], batch_size, graph_builder=graph_builder
        )
        normalizers.append(normalizer)
        mads.append(mad)
        test_loaders.append(test_loader)

        # Normalize targets inside the model via the route normalizer.
        model.registry.routes[key].normalizer = normalizer

        train_start = time.perf_counter()
        result = model.learn_endpoint(
            version,
            pid,
            fid_id,
            train_loader,
            val_loader,
            device,
            epochs_fast=epochs_fast,
            epochs_cons=epochs_cons,
            lr=lr,
        )
        train_time = time.perf_counter() - train_start
        wall_times.append(train_time)
        per_route_info.append({
            "new_basis_blocks": result["new_basis_blocks"],
            "selected_basis_blocks": result["selected_basis_blocks"],
            "best_val_loss": result["best_val_loss"],
        })
        print(f"  New basis blocks: {result['new_basis_blocks']}")
        print(f"  Best val loss: {result['best_val_loss']:.4f}  ({train_time:.1f}s)")

        # Evaluate on all seen endpoints.
        task_nmaes: list[float] = []
        for prev_t in range(t + 1):
            prev_version, prev_prop, prev_fid, _ = tasks[prev_t]
            ev = evaluate_endpoint(
                model,
                task_records[prev_t],
                prev_version,
                prop2id[prev_prop],
                fid2id[prev_fid],
                normalizers[prev_t],
                device,
                batch_size,
                graph_builder=graph_builder,
                test_loader=test_loaders[prev_t],
            )
            task_nmaes.append(ev["nmae"])
        nmaes.append(task_nmaes)
        print(f"  Test nMAEs: {[f'{x:.3f}' for x in task_nmaes]}")

        # Subset evaluations (e.g. revision protocol error decomposition).
        subset_metrics: dict[str, dict[str, float]] = {}
        if eval_subsets is not None and t in eval_subsets:
            for subset_name in eval_subsets[t]:
                subset_metrics[subset_name] = evaluate_endpoint(
                    model,
                    task_records[t],
                    version,
                    pid,
                    fid_id,
                    normalizer,
                    device,
                    batch_size,
                    graph_builder=graph_builder,
                    change_type=subset_name,
                )
            print(f"  Subsets: {subset_metrics}")
        per_task_subset_metrics.append(subset_metrics)

        model.publish_route(version, pid, fid_id)
        print(f"  Published route {key}")

    total_time = time.perf_counter() - start_all

    metrics: dict[str, Any] = {
        "protocol": protocol_name,
        "tasks": [{"version": v, "property": p, "fidelity": f, "target_field": tf} for v, p, f, tf in tasks],
        "route_keys": route_keys,
        "nmaes": nmaes,
        "average_final_nmae": sum(nmaes[-1]) / len(nmaes[-1]) if nmaes else float("nan"),
        "forgetting": forgetting(nmaes),
        "backward_transfer": backward_transfer(nmaes),
        "total_parameters": model.total_parameters(),
        "incremental_parameters": [model.incremental_parameters(v, prop2id[p], fid2id[f]) for v, p, f, _ in tasks],
        "train_wall_times_seconds": wall_times,
        "total_wall_time_seconds": total_time,
        "subset_metrics": per_task_subset_metrics,
        "per_route_info": per_route_info,
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    return metrics


def evaluate_pareto_for_endpoint(
    model: PersistentConsolidationGraph,
    recs: list[dict],
    version: str,
    prop_id: int,
    fid_id: int,
    device: torch.device,
    batch_size: int,
    graph_builder: Any | None = None,
) -> dict[str, float]:
    """Compute Pareto metrics for a published endpoint using physical targets."""
    ds = JARVISCrystalDataset(recs, split="test", normalize_target=False, graph_builder=graph_builder)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals)
    return evaluate_pareto_metrics(
        model,
        loader,
        optimizer=None,
        device=device,
        forward_args=(version, prop_id, fid_id),
        forward_kwargs={"physical": True},
    )
