"""Run the addition protocol with PCG."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from persistent_consolidation_graph import PersistentConsolidationGraph
from pcg_runner import (
    build_pcg_encoder_and_graph_builder,
    cap_records,
    determine_parents_addition,
    filter_records_for_encoder,
    run_pcg_protocol,
)
from protocols import build_addition_protocol


def _name_to_id(names: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(sorted(set(names)))}


def main() -> None:
    parser = argparse.ArgumentParser(description="PCG addition protocol runner")
    parser.add_argument("--properties", nargs="+", default=["band_gap"])
    parser.add_argument("--fidelities", nargs="+", default=["OptB88vdW", "TB-mBJ"])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--encoder-type", choices=["matgl", "egnn"], default="matgl")
    parser.add_argument("--matgl-model", default=None, help="Pre-trained MatGL model name/path; None uses a small random-init M3GNet")
    parser.add_argument("--epochs-fast", type=int, default=5)
    parser.add_argument("--epochs-cons", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cap", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/pcg_addition"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tasks, task_records, audit = build_addition_protocol(
        properties=args.properties,
        fidelities=args.fidelities,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )

    encoder, graph_builder = build_pcg_encoder_and_graph_builder(
        args.encoder_type, args.hidden_dim, matgl_model=args.matgl_model
    )
    task_records = [filter_records_for_encoder(recs, args.encoder_type, encoder) for recs in task_records]

    if args.cap is not None:
        task_records = [cap_records(recs, args.cap) for recs in task_records]

    tasks = [t for t, recs in zip(tasks, task_records) if recs]
    task_records = [recs for recs in task_records if recs]

    prop2id = _name_to_id([p for _, p, _, _ in tasks])
    fid2id = _name_to_id([f for _, _, f, _ in tasks])

    model = PersistentConsolidationGraph(encoder, args.hidden_dim, rank=args.rank, max_rank=args.rank).to(device)

    metrics = run_pcg_protocol(
        protocol_name="addition",
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
        parent_fn=determine_parents_addition,
        graph_builder=graph_builder,
    )
    metrics["audit"] = audit


if __name__ == "__main__":
    main()
