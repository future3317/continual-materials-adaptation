"""Phase 0 small-scale baseline comparison runner for PhyTCA.

Example:
    python scripts/run_phase0.py --protocol a --cap 5000 --seeds 42 43 44 \
        --methods phytca joint sequential frozen_heads --epochs 10 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Allow imports from project root when running inside scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from legacy.baselines import BASELINE_REGISTRY
from data import build_protocol_a, build_protocol_b
from train_utils import backward_transfer, forgetting
from train_phytca import continual_experiment


METHODS_DEFAULT = [
    "phytca",
    "joint",
    "independent",
    "sequential",
    "frozen_heads",
    "ewc",
    "replay",
    "independent_lora",
    "shared_lora",
]


def run_method(
    method: str,
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    adapter_rank: int,
    num_nearest_neighbors: int,
    mu: float = 0.01,
) -> dict[str, Any]:
    """Run one method and return metrics."""
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    start = time.time()
    try:
        if method == "phytca":
            nmaes, info = continual_experiment(
                tasks=tasks,
                task_records=task_records,
                node_dim=node_dim,
                hidden_dim=hidden_dim,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                mu=mu,
                adapter_rank=adapter_rank,
                num_nearest_neighbors=num_nearest_neighbors,
            )
        else:
            fn = BASELINE_REGISTRY[method]
            nmaes, info = fn(
                tasks=tasks,
                task_records=task_records,
                node_dim=node_dim,
                hidden_dim=hidden_dim,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                adapter_rank=adapter_rank,
                num_nearest_neighbors=num_nearest_neighbors,
            )
        elapsed = time.time() - start
        peak_mem = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0
        return {
            "method": method,
            "status": "ok",
            "nmaes": nmaes,
            "final_nmaes": nmaes[-1],
            "avg_final_nmae": sum(nmaes[-1]) / len(nmaes[-1]),
            "forgetting": forgetting(nmaes),
            "backward_transfer": backward_transfer(nmaes),
            "parameters": info.get("adapter_params", 0),
            "replay_storage": info.get("replay_storage", 0),
            "time_seconds": elapsed,
            "peak_gpu_mb": peak_mem,
        }
    except Exception as e:
        return {
            "method": method,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["a", "b"], default="a")
    parser.add_argument("--cap", type=int, default=5000, help="Train samples per task cap")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--methods", nargs="+", default=METHODS_DEFAULT, help="Baseline methods to run")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--num-nearest-neighbors", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="reports/phase0")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        if args.protocol == "a":
            tasks, task_records, audit = build_protocol_a(seed=seed, n_train_val_per_task=args.cap)
        else:
            tasks, task_records, audit = build_protocol_b(seed=seed, n_train_val_per_task=args.cap)

        print(f"\n=== Seed {seed} ===")
        for t, desc in enumerate(tasks):
            print(f"  Task {t + 1} {desc}: {len(task_records[t])} structures")

        for method in args.methods:
            print(f"\n  Running {method}...")
            result = run_method(
                method=method,
                tasks=tasks,
                task_records=task_records,
                node_dim=92,
                hidden_dim=args.hidden_dim,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                adapter_rank=args.adapter_rank,
                num_nearest_neighbors=args.num_nearest_neighbors,
                mu=args.mu,
            )
            result["seed"] = seed
            result["protocol"] = args.protocol
            result["cap"] = args.cap
            all_results.append(result)

            if result["status"] == "ok":
                print(
                    f"    avg_final={result['avg_final_nmae']:.3f} "
                    f"forget={result['forgetting']:.3f} "
                    f"bwt={result['backward_transfer']:.3f} "
                    f"params={result['parameters']:,} "
                    f"time={result['time_seconds']:.1f}s"
                )
            else:
                print(f"    ERROR: {result['error']}")

    summary_path = output_dir / f"phase0_{args.protocol}_cap{args.cap}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote results to {summary_path}")

    # Print aggregate table.
    print("\n=== Aggregate per method ===")
    from collections import defaultdict
    agg: dict[str, list[dict]] = defaultdict(list)
    for r in all_results:
        if r["status"] == "ok":
            agg[r["method"]].append(r)

    print(f"{'method':<20} {'avg_final':<10} {'forget':<10} {'bwt':<10} {'params':<12} {'time':<10}")
    for method in args.methods:
        if method not in agg:
            continue
        vals = agg[method]
        avg_final = sum(r["avg_final_nmae"] for r in vals) / len(vals)
        avg_forget = sum(r["forgetting"] for r in vals) / len(vals)
        avg_bwt = sum(r["backward_transfer"] for r in vals) / len(vals)
        params = vals[0]["parameters"]
        avg_time = sum(r["time_seconds"] for r in vals) / len(vals)
        print(f"{method:<20} {avg_final:<10.3f} {avg_forget:<10.3f} {avg_bwt:<10.3f} {params:<12,} {avg_time:<10.1f}")


if __name__ == "__main__":
    main()
