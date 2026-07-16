"""Baseline comparison runner for the versioned (version, property, fidelity) protocol.

Methods:
- independent: one fresh model trained per endpoint (upper bound, no sharing).
- joint: a single ContinualCrystalModel trained jointly on all endpoints (upper bound,
  no exact retention).
- continual_tucker: ContinualCrystalModel with single-child Tucker adapters and
  freeze_task after each endpoint.
- continual_lora_ab / continual_lora_aba: same with LoRA-AB / LoRA-ABA adapters.
- versioned_graph: VersionedFidelityGraph with published frozen routes.
- copy_on_write: CopyOnWriteFullChildModel with independent full child encoders.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch

from data import JARVISCrystalDataset, build_versioned_protocol, collate_crystals
from models import ContinualCrystalModel, CopyOnWriteFullChildModel
from train_utils import (
    _evaluate_all_seen_versioned,
    _evaluate_loader,
    _make_loaders,
    _train_one_task_trainable,
    backward_transfer,
    forgetting,
)
from versioned_graph import VersionedFidelityGraph


def _name_to_id(names: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(sorted(set(names)))}


def _cap_records(recs: list[dict], cap: int | None) -> list[dict]:
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


def _flat_task_ids(tasks: list[tuple[str, str, str, str]]) -> list[tuple[int, int]]:
    """Map each versioned task to a unique (prop_id, fid_id=0) pair.

    ContinualCrystalModel only understands (property, fidelity) pairs, so we
    flatten the three-axis task into a single property ID.
    """
    return [(i, 0) for i in range(len(tasks))]


MethodFn = Callable[
    [
        list[tuple[str, str, str, str]],
        list[list[dict]],
        dict[str, int],
        dict[str, int],
        dict[str, int],
        torch.device,
        argparse.Namespace,
    ],
    dict[str, Any],
]


def _build_graph_model(args: argparse.Namespace, device: torch.device) -> VersionedFidelityGraph:
    return VersionedFidelityGraph(
        node_dim=92,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        rank=args.rank,
        num_nearest_neighbors=args.num_nearest_neighbors,
        bases_trainable=True,
    ).to(device)


def _run_versioned_graph(
    tasks: list[tuple[str, str, str, str]],
    task_records: list[list[dict]],
    version2id: dict[str, int],
    prop2id: dict[str, int],
    fid2id: dict[str, int],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """VersionedFidelityGraph baseline (already the main runner, included for comparison)."""
    model = _build_graph_model(args, device)
    task_stats: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    nmaes: list[list[float]] = []

    start = time.perf_counter()
    for t, (version, prop, fid, _) in enumerate(tasks):
        pid = prop2id[prop]
        fid_id = fid2id[fid]
        model.add_route(version, pid, fid_id)
        train_loader, val_loader, _, mean, std, mad = _make_loaders(
            task_records[t], args.batch_size
        )
        _, mean, std, mad, _ = _train_one_task_trainable(
            model,
            train_loader,
            val_loader,
            forward_extra_args=(version, pid, fid_id),
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
        )
        task_stats.append((mean, std, mad))
        nmaes.append(
            _evaluate_all_seen_versioned(
                model, tasks, task_records, task_stats, prop2id, fid2id,
                args.batch_size, device, t,
            )
        )
        model.publish_route(version, pid, fid_id)

    return {
        "nmaes": nmaes,
        "average_final_nmae": sum(nmaes[-1]) / len(nmaes[-1]),
        "forgetting": forgetting(nmaes),
        "backward_transfer": backward_transfer(nmaes),
        "total_parameters": model.total_parameters(),
        "wall_time_seconds": time.perf_counter() - start,
    }


def _run_copy_on_write(
    tasks: list[tuple[str, str, str, str]],
    task_records: list[list[dict]],
    version2id: dict[str, int],
    prop2id: dict[str, int],
    fid2id: dict[str, int],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Copy-on-write full child baseline."""
    model = CopyOnWriteFullChildModel(
        node_dim=92,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        num_nearest_neighbors=args.num_nearest_neighbors,
    ).to(device)
    task_stats: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    nmaes: list[list[float]] = []

    start = time.perf_counter()
    for t, (version, prop, fid, _) in enumerate(tasks):
        pid = prop2id[prop]
        fid_id = fid2id[fid]
        model.add_route(version, pid, fid_id)
        train_loader, val_loader, _, mean, std, mad = _make_loaders(
            task_records[t], args.batch_size
        )
        _, mean, std, mad, _ = _train_one_task_trainable(
            model,
            train_loader,
            val_loader,
            forward_extra_args=(version, pid, fid_id),
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
        )
        task_stats.append((mean, std, mad))
        nmaes.append(
            _evaluate_all_seen_versioned(
                model, tasks, task_records, task_stats, prop2id, fid2id,
                args.batch_size, device, t,
            )
        )
        model.freeze_route(version, pid, fid_id)

    return {
        "nmaes": nmaes,
        "average_final_nmae": sum(nmaes[-1]) / len(nmaes[-1]),
        "forgetting": forgetting(nmaes),
        "backward_transfer": backward_transfer(nmaes),
        "total_parameters": model.total_parameters(),
        "wall_time_seconds": time.perf_counter() - start,
    }


def _run_continual_model(
    tasks: list[tuple[str, str, str, str]],
    task_records: list[list[dict]],
    version2id: dict[str, int],
    prop2id: dict[str, int],
    fid2id: dict[str, int],
    device: torch.device,
    args: argparse.Namespace,
    adapter_name: str,
    freeze_after_each: bool = True,
) -> dict[str, Any]:
    """ContinualCrystalModel with a given adapter, optionally freezing after each task."""
    flat_ids = _flat_task_ids(tasks)
    n_tasks = len(tasks)
    model = ContinualCrystalModel(
        node_dim=92,
        hidden_dim=args.hidden_dim,
        n_properties=n_tasks,
        n_fidelities=1,
        adapter_name=adapter_name,
        adapter_rank=args.rank,
        n_layers=args.n_layers,
        num_nearest_neighbors=args.num_nearest_neighbors,
    ).to(device)

    task_stats: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    nmaes: list[list[float]] = []

    start = time.perf_counter()
    for t, _ in enumerate(tasks):
        prop_id, fid_id = flat_ids[t]
        model.add_task(prop_id, fid_id)
        train_loader, val_loader, _, mean, std, mad = _make_loaders(
            task_records[t], args.batch_size
        )
        _, mean, std, mad, _ = _train_one_task_trainable(
            model,
            train_loader,
            val_loader,
            forward_extra_args=(prop_id, fid_id),
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
        )
        task_stats.append((mean, std, mad))

        # Evaluate all seen tasks using their flat IDs.
        seen_nmaes: list[float] = []
        for prev_t in range(t + 1):
            prev_pid, prev_fid = flat_ids[prev_t]
            mean_p, std_p, mad_p = task_stats[prev_t]
            test_ds = JARVISCrystalDataset(task_records[prev_t], split="test")
            test_ds.target_mean = float(mean_p)
            test_ds.target_std = float(std_p)
            test_ds.normalize_target = True
            loader = torch.utils.data.DataLoader(
                test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_crystals
            )
            seen_nmaes.append(
                _evaluate_loader(model, loader, (prev_pid, prev_fid), mean_p, std_p, mad_p, device)
            )
        nmaes.append(seen_nmaes)

        if freeze_after_each:
            model.freeze_task(prop_id, fid_id)

    return {
        "nmaes": nmaes,
        "average_final_nmae": sum(nmaes[-1]) / len(nmaes[-1]),
        "forgetting": forgetting(nmaes),
        "backward_transfer": backward_transfer(nmaes),
        "total_parameters": model.count_total_parameters(),
        "wall_time_seconds": time.perf_counter() - start,
    }


def _run_independent(
    tasks: list[tuple[str, str, str, str]],
    task_records: list[list[dict]],
    version2id: dict[str, int],
    prop2id: dict[str, int],
    fid2id: dict[str, int],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Independent model per endpoint: fresh ContinualCrystalModel for each task."""
    flat_ids = _flat_task_ids(tasks)
    n_tasks = len(tasks)
    nmaes: list[list[float]] = []
    total_params = 0
    start = time.perf_counter()

    for t, (version, prop, fid, _) in enumerate(tasks):
        prop_id, fid_id = flat_ids[t]
        model = ContinualCrystalModel(
            node_dim=92,
            hidden_dim=args.hidden_dim,
            n_properties=n_tasks,
            n_fidelities=1,
            adapter_name="single_child_tucker",
            adapter_rank=args.rank,
            n_layers=args.n_layers,
            num_nearest_neighbors=args.num_nearest_neighbors,
        ).to(device)
        model.add_task(prop_id, fid_id)
        train_loader, val_loader, _, mean, std, mad = _make_loaders(
            task_records[t], args.batch_size
        )
        _train_one_task_trainable(
            model,
            train_loader,
            val_loader,
            forward_extra_args=(prop_id, fid_id),
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
        )
        total_params += model.count_total_parameters()

        # Independent models have no sharing: only the current task is evaluated.
        test_ds = JARVISCrystalDataset(task_records[t], split="test")
        test_ds.target_mean = float(mean)
        test_ds.target_std = float(std)
        test_ds.normalize_target = True
        loader = torch.utils.data.DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_crystals
        )
        nmae = _evaluate_loader(model, loader, (prop_id, fid_id), mean, std, mad, device)

        row = [float("nan")] * len(tasks)
        row[t] = nmae
        nmaes.append(row)

    return {
        "nmaes": nmaes,
        "average_final_nmae": nmaes[-1][-1],
        "forgetting": 0.0,
        "backward_transfer": 0.0,
        "total_parameters": total_params,
        "wall_time_seconds": time.perf_counter() - start,
    }


METHODS: dict[str, MethodFn] = {
    "versioned_graph": _run_versioned_graph,
    "copy_on_write": _run_copy_on_write,
    "continual_tucker": lambda t, r, v, p, f, d, a: _run_continual_model(t, r, v, p, f, d, a, "single_child_tucker"),
    "continual_lora_ab": lambda t, r, v, p, f, d, a: _run_continual_model(t, r, v, p, f, d, a, "lora_ab"),
    "continual_lora_aba": lambda t, r, v, p, f, d, a: _run_continual_model(t, r, v, p, f, d, a, "lora_aba"),
    "joint": lambda t, r, v, p, f, d, a: _run_continual_model(t, r, v, p, f, d, a, "single_child_tucker", freeze_after_each=False),
    "independent": _run_independent,
}


def run_versioned_baselines(
    snapshots: list[str],
    properties: list[str],
    fidelities: list[str],
    methods: list[str],
    hidden_dim: int,
    rank: int,
    n_layers: int,
    num_nearest_neighbors: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
    seed: int,
    cap: int | None,
    output_dir: Path,
) -> dict[str, Any]:
    """Run selected baselines and return aggregated results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks, task_records, audit = build_versioned_protocol(
        snapshots=snapshots,
        properties=properties,
        fidelities=fidelities,
        seed=seed,
        global_split=True,
    )
    if cap is not None:
        task_records = [_cap_records(recs, cap) for recs in task_records]

    version2id = _name_to_id([v for v, _, _, _ in tasks])
    prop2id = _name_to_id([p for _, p, _, _ in tasks])
    fid2id = _name_to_id([f for _, _, f, _ in tasks])

    results: dict[str, Any] = {
        "tasks": [
            {"version": v, "property": p, "fidelity": f, "target_field": tf}
            for v, p, f, tf in tasks
        ],
        "audit": audit,
        "methods": {},
    }

    for method_name in methods:
        print(f"\n=== Running baseline: {method_name} ===")
        fn = METHODS[method_name]
        method_results = fn(
            tasks, task_records, version2id, prop2id, fid2id, device,
            argparse.Namespace(
                hidden_dim=hidden_dim,
                rank=rank,
                n_layers=n_layers,
                num_nearest_neighbors=num_nearest_neighbors,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                patience=patience,
            ),
        )
        results["methods"][method_name] = method_results
        print(f"  avg final nMAE: {method_results['average_final_nmae']:.3f}")
        print(f"  forgetting: {method_results['forgetting']:.3f}")
        print(f"  total params: {method_results['total_parameters']:,}")

    out_path = output_dir / "baseline_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Versioned protocol baseline comparison")
    parser.add_argument(
        "--snapshots",
        nargs="+",
        default=["dft_3d_2021", "dft_3d"],
    )
    parser.add_argument(
        "--properties",
        nargs="+",
        default=["band_gap"],
    )
    parser.add_argument(
        "--fidelities",
        nargs="+",
        default=["OptB88vdW", "TB-mBJ"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS.keys()),
        choices=list(METHODS.keys()),
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--num-nearest-neighbors", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cap", type=int, default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/versioned_baselines"),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    run_versioned_baselines(
        snapshots=args.snapshots,
        properties=args.properties,
        fidelities=args.fidelities,
        methods=args.methods,
        hidden_dim=args.hidden_dim,
        rank=args.rank,
        n_layers=args.n_layers,
        num_nearest_neighbors=args.num_nearest_neighbors,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=device,
        seed=args.seed,
        cap=args.cap,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
