"""Train and evaluate a VersionedFidelityGraph on the three-axis benchmark.

Each endpoint is identified by ``(version, property, fidelity)``.  After an
endpoint is trained it is published (frozen), guaranteeing exact retention of
all previously published endpoints while the latest endpoint continues to
learn.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from data import build_versioned_protocol
from train_utils import (
    _evaluate_all_seen_versioned,
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
    # Cap proportionally, ensuring every split gets at least one record if present.
    capped: list[dict] = []
    per_split_cap = max(1, cap // len(by_split))
    for split_recs in by_split.values():
        capped.extend(split_recs[:per_split_cap])
    return capped


def run_versioned_protocol(
    snapshots: list[str],
    properties: list[str],
    fidelities: list[str],
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
    """Run the versioned benchmark and return metrics."""
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

    node_dim = 92  # One-hot atom types for JARVIS.
    model = VersionedFidelityGraph(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        rank=rank,
        num_nearest_neighbors=num_nearest_neighbors,
        bases_trainable=True,
    ).to(device)

    task_stats: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    nmaes: list[list[float]] = []
    route_keys: list[str] = []
    wall_times: list[float] = []

    start_all = time.perf_counter()
    for t, (version, prop, fid, _) in enumerate(tasks):
        vid = version2id[version]
        pid = prop2id[prop]
        fid_id = fid2id[fid]
        key = model.add_route(version, pid, fid_id)
        route_keys.append(key)

        print(f"\n=== Route {t + 1}/{len(tasks)}: {version} / {prop} / {fid} ===")
        train_loader, val_loader, test_loader, mean, std, mad = _make_loaders(
            task_records[t], batch_size
        )

        train_start = time.perf_counter()
        best_val_nmae, mean, std, mad = _train_one_task_trainable(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            forward_extra_args=(version, pid, fid_id),
            device=device,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
        )
        train_time = time.perf_counter() - train_start
        wall_times.append(train_time)

        task_stats.append((mean, std, mad))
        print(f"  Best val nMAE: {best_val_nmae:.3f}  ({train_time:.1f}s)")

        # Evaluate on all routes seen so far (including the new one).
        task_nmaes = _evaluate_all_seen_versioned(
            model=model,
            tasks=tasks,
            task_records=task_records,
            task_stats=task_stats,
            prop2id=prop2id,
            fid2id=fid2id,
            batch_size=batch_size,
            device=device,
            t=t,
        )
        nmaes.append(task_nmaes)
        print(f"  Test nMAEs: {[f'{x:.3f}' for x in task_nmaes]}")

        # Publish the endpoint before moving to the next route.
        model.publish_route(version, pid, fid_id)
        print(f"  Published route {key}")

    total_time = time.perf_counter() - start_all

    metrics: dict[str, Any] = {
        "tasks": [
            {"version": v, "property": p, "fidelity": f, "target_field": tf}
            for v, p, f, tf in tasks
        ],
        "route_keys": route_keys,
        "nmaes": nmaes,
        "average_final_nmae": sum(nmaes[-1]) / len(nmaes[-1]),
        "forgetting": forgetting(nmaes),
        "backward_transfer": backward_transfer(nmaes),
        "total_parameters": model.total_parameters(),
        "incremental_parameters": [
            model.incremental_parameters(v, prop2id[p], fid2id[f])
            for v, p, f, _ in tasks
        ],
        "train_wall_times_seconds": wall_times,
        "total_wall_time_seconds": total_time,
        "audit": audit,
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Versioned backward-compatible benchmark runner"
    )
    parser.add_argument(
        "--snapshots",
        nargs="+",
        default=["dft_3d_2021", "dft_3d"],
        help="JARVIS snapshots in chronological order",
    )
    parser.add_argument(
        "--properties",
        nargs="+",
        default=["band_gap"],
        help="Properties to include",
    )
    parser.add_argument(
        "--fidelities",
        nargs="+",
        default=["OptB88vdW", "TB-mBJ"],
        help="Fidelities to include",
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
    parser.add_argument(
        "--cap",
        type=int,
        default=None,
        help="Per-task record cap for smoke tests",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/versioned_protocol"),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    run_versioned_protocol(
        snapshots=args.snapshots,
        properties=args.properties,
        fidelities=args.fidelities,
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
