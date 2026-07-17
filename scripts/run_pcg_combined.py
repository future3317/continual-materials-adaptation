"""Run the combined PCG benchmark across all three evolution axes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from persistent_consolidation_graph import PersistentConsolidationGraph
from pcg_runner import (
    build_pcg_encoder_and_graph_builder,
    cap_records,
    determine_parents_combined,
    evaluate_pareto_for_endpoint,
    filter_records_for_encoder,
    run_pcg_protocol,
)
from protocols import build_combined_protocol


def _name_to_id(names: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(sorted(set(names)))}


def main() -> None:
    parser = argparse.ArgumentParser(description="PCG combined protocol runner")
    parser.add_argument("--properties", nargs="+", default=["band_gap"])
    parser.add_argument("--fidelities", nargs="+", default=["OptB88vdW", "TB-mBJ"])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--encoder-type", choices=["matgl", "egnn"], default="matgl")
    parser.add_argument("--epochs-fast", type=int, default=5)
    parser.add_argument("--epochs-cons", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cap", type=int, default=None, help="Per-task record cap for smoke tests")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/pcg_combined"))
    parser.add_argument("--pareto", action="store_true", help="Compute Pareto metrics for the final endpoint")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tasks, task_records, audit = build_combined_protocol(
        properties=args.properties,
        fidelities=args.fidelities,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )

    encoder, graph_builder = build_pcg_encoder_and_graph_builder(
        args.encoder_type, args.hidden_dim
    )
    task_records = [filter_records_for_encoder(recs, args.encoder_type, encoder) for recs in task_records]

    if args.cap is not None:
        task_records = [cap_records(recs, args.cap) for recs in task_records]

    tasks = [t for t, recs in zip(tasks, task_records) if recs]
    task_records = [recs for recs in task_records if recs]

    prop2id = _name_to_id([p for _, p, _, _ in tasks])
    fid2id = _name_to_id([f for _, _, f, _ in tasks])

    model = PersistentConsolidationGraph(encoder, args.hidden_dim, rank=args.rank).to(device)

    metrics = run_pcg_protocol(
        protocol_name="combined",
        tasks=tasks,
        task_records=task_records,
        model=model,
        prop2id=prop2id,
        fid2id=fid2id,
        device=device,
        batch_size=args.batch_size,
        epochs_fast=args.epochs_fast,
        epochs_cons=args.epochs_cons,
        lr=args.lr,
        output_dir=args.output_dir,
        parent_fn=determine_parents_combined,
        graph_builder=graph_builder,
    )
    metrics["audit"] = audit

    if args.pareto:
        latest_version, latest_prop, latest_fid, _ = tasks[-1]
        latest_pid = prop2id[latest_prop]
        latest_fid_id = fid2id[latest_fid]
        metrics["pareto"] = evaluate_pareto_for_endpoint(
            model,
            task_records[-1],
            latest_version,
            latest_pid,
            latest_fid_id,
            device,
            args.batch_size,
            graph_builder=graph_builder,
        )
        metrics_path = args.output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Pareto metrics appended to {metrics_path}")


if __name__ == "__main__":
    main()
