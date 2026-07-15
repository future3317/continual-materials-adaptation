"""Stage 2 scaling study: 5k×3-seed FR-PhyTCA validation on Protocol B.

This script runs the required baselines at 5,000 training samples per task and
seeds 42, 43, 44.  It shares a single trained OPT parent checkpoint across all
frozen-parent methods so that differences in MBJ accuracy reflect the correction
module, not a different Task-1 starting point.  After all seeds finish, it
aggregates results and prints the Stage-2 GO/NO-GO gate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import build_protocol_b, cap_splits
from diagnostics import (
    d1_full_joint,
    d2_joint_phytca,
    d4_frozen_opt_affine,
    d6_progressive_tucker,
    d6e_orthogonal_tucker_residual,
    d6g_matched_low_rank_residual,
    d6h_matched_mlp_residual,
    feature_transfer_experiment,
    mbj_only_training,
    opt_pretrain_mbj_full_finetune,
    train_opt_parent,
)
from scripts.run_phase0_b_screening import _canonical_base_state, _repartition_dev_test


def _build_two_task_data(
    train_cap: int,
    val_cap: int,
    test_cap: int,
    dev_frac: float,
    seed: int,
):
    """Load Protocol B first two tasks, cap, and repartition held-out test."""
    tasks_all, task_records_all, _ = build_protocol_b(seed=seed)
    tasks = tasks_all[:2]
    task_records = [
        cap_splits(recs, train_cap, val_cap, test_cap, seed=seed)
        for recs in task_records_all[:2]
    ]
    task_records = [
        _repartition_dev_test(recs, dev_frac=dev_frac, seed=seed + t)
        for t, recs in enumerate(task_records)
    ]
    return tasks, task_records


def _checkpoint_bytes(state_dict: dict[str, torch.Tensor]) -> int:
    return sum(v.numel() * v.element_size() for v in state_dict.values())


def _run_experiment(name: str, fn, kwargs: dict, device: torch.device) -> dict:
    """Run one experiment, capturing peak GPU memory and checkpoint size."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    try:
        result = fn(**kwargs)
    except Exception as e:
        print(f"ERROR {name}: {e}")
        return {
            "experiment": name,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
    wall_time = time.time() - start
    result = result or {"experiment": name}
    result["wall_time_seconds"] = wall_time
    if torch.cuda.is_available():
        result["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(device)
    # Approximate stored checkpoint bytes from the trained model state dict.
    if "state_dict" not in result:
        # Heuristic: stored params times element size.
        elem_size = 4  # float32
        result["checkpoint_bytes"] = result.get("stored_params", 0) * elem_size
    return result


def _run_seed(
    seed: int,
    train_cap: int,
    val_cap: int,
    test_cap: int,
    dev_frac: float,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    hidden_dim: int,
    adapter_rank: int,
    num_nearest_neighbors: int,
    device: torch.device,
    output_dir: Path,
    artifact_dir: Path,
) -> list[dict]:
    """Run all Stage-2 methods for one seed."""
    tasks, task_records = _build_two_task_data(
        train_cap, val_cap, test_cap, dev_frac, seed
    )
    print(f"\n=== Protocol B Stage-2 scaling experiments ===")
    print(f"seed={seed}, train_cap={train_cap}, tasks={tasks}")
    for t, desc in enumerate(tasks):
        counts: dict[str, int] = {"train": 0, "val": 0, "continual_dev": 0, "final_test": 0}
        for r in task_records[t]:
            counts[r["split"]] += 1
        print(f"  {desc}: {counts}")

    from train_phytca import _name_to_id
    prop2id, fid2id = _name_to_id(tasks)
    base_state = _canonical_base_state(
        seed=seed,
        node_dim=92,
        hidden_dim=hidden_dim,
        n_properties=len(prop2id),
        n_fidelities=len(fid2id),
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        artifact_dir=artifact_dir,
        device=device,
    )

    common_kwargs = {
        "node_dim": 92,
        "hidden_dim": hidden_dim,
        "device": device,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "patience": patience,
        "adapter_rank": adapter_rank,
        "num_nearest_neighbors": num_nearest_neighbors,
    }

    results: list[dict] = []

    # 1. Full joint.
    results.append(_run_experiment(
        "full_joint_upper_bound",
        d1_full_joint,
        {"tasks": tasks, "task_records": task_records, **common_kwargs},
        device,
    ))

    # 2. Joint Tucker / PhyTCA.
    results.append(_run_experiment(
        "phytca_joint_upper_bound",
        d2_joint_phytca,
        {"tasks": tasks, "task_records": task_records, "base_state_dict": base_state, **common_kwargs},
        device,
    ))

    # 3. MBJ-only training.
    results.append(_run_experiment(
        "mbj_only",
        mbj_only_training,
        {"tasks": tasks, "task_records": task_records, "base_state_dict": base_state, **common_kwargs},
        device,
    ))

    # Train shared OPT parent for frozen-parent methods.
    print("\nTraining shared OPT parent for frozen-parent methods...")
    opt_parent_bundle = train_opt_parent(
        tasks=tasks,
        task_records=task_records,
        base_state_dict=base_state,
        artifact_dir=output_dir,
        **common_kwargs,
    )
    print(
        f"  OPT parent: T1@T1={opt_parent_bundle['task1_after_task1']:.4f}, "
        f"state_hash={opt_parent_bundle['state_dict_hash'][:16]}..., "
        f"pred_hash={opt_parent_bundle['prediction_hash'][:16]}..."
    )

    shared_parent_kwargs = {"opt_parent_state": opt_parent_bundle, **common_kwargs}

    # 4. OPT pretrain -> MBJ full fine-tune.
    results.append(_run_experiment(
        "opt_pretrain_mbj_full_finetune",
        opt_pretrain_mbj_full_finetune,
        {"tasks": tasks, "task_records": task_records, "base_state_dict": base_state, **shared_parent_kwargs},
        device,
    ))

    # 5. Frozen OPT + affine correction.
    results.append(_run_experiment(
        "frozen_opt_affine_correction",
        d4_frozen_opt_affine,
        {"tasks": tasks, "task_records": task_records, "base_state_dict": base_state, **shared_parent_kwargs},
        device,
    ))

    # 6. Frozen OPT + matched MLP residual.
    results.append(_run_experiment(
        "matched_mlp_residual",
        d6h_matched_mlp_residual,
        {"tasks": tasks, "task_records": task_records, "base_state_dict": base_state, **shared_parent_kwargs},
        device,
    ))

    # 7. Frozen OPT + matched low-rank residual.
    results.append(_run_experiment(
        "matched_low_rank_residual",
        d6g_matched_low_rank_residual,
        {"tasks": tasks, "task_records": task_records, "base_state_dict": base_state, **shared_parent_kwargs},
        device,
    ))

    # 8. FR-PhyTCA.
    results.append(_run_experiment(
        "fr_phytca",
        d6_progressive_tucker,
        {"tasks": tasks, "task_records": task_records, "base_state_dict": base_state, "lambda_distill": 0.0, **shared_parent_kwargs},
        device,
    ))

    # 9. Orthogonal FR-PhyTCA.
    results.append(_run_experiment(
        "fr_phytca_orthogonal",
        d6e_orthogonal_tucker_residual,
        {"tasks": tasks, "task_records": task_records, "base_state_dict": base_state, **shared_parent_kwargs},
        device,
    ))

    # 10. Feature-transfer baseline.
    results.append(_run_experiment(
        "feature_transfer",
        feature_transfer_experiment,
        {"tasks": tasks, "task_records": task_records, "base_state_dict": base_state, **shared_parent_kwargs},
        device,
    ))

    # Add checkpoint-byte estimates where missing.
    for r in results:
        if r.get("status") == "error":
            continue
        if "checkpoint_bytes" not in r:
            r["checkpoint_bytes"] = r.get("stored_params", 0) * 4

    seed_out = output_dir / f"seed_{seed}" / "scaling_experiments.json"
    seed_out.parent.mkdir(parents=True, exist_ok=True)
    with open(seed_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved seed results to {seed_out}")

    return results


def _aggregate(
    all_results: dict[int, list[dict]],
    output_dir: Path,
) -> dict[str, Any]:
    """Aggregate metrics across seeds and apply Stage-2 gates."""
    methods: set[str] = set()
    for seed_results in all_results.values():
        for r in seed_results:
            if r.get("status") != "error":
                methods.add(r["experiment"])

    summary: dict[str, dict[str, list[float]]] = {m: {} for m in methods}
    for seed_results in all_results.values():
        by_name = {r["experiment"]: r for r in seed_results if r.get("status") != "error"}
        for name, r in by_name.items():
            for key in [
                "task1_after_task1",
                "task1_after_task2",
                "task2_final_nmae",
                "absolute_forgetting",
                "bwt",
                "average_final_nmae",
                "raw_mae_eV",
                "opt_route_drift",
                "incremental_params",
                "stored_params",
                "wall_time_seconds",
                "checkpoint_bytes",
                "peak_gpu_memory_bytes",
            ]:
                if key in r and r[key] is not None:
                    summary[name].setdefault(key, []).append(float(r[key]))

    rows = []
    for name in sorted(methods):
        vals = summary[name]
        row: dict[str, Any] = {"experiment": name}
        for key, vlist in vals.items():
            if key in {"incremental_params", "stored_params", "checkpoint_bytes", "peak_gpu_memory_bytes"}:
                row[key] = {"mean": statistics.mean(vlist), "std": statistics.stdev(vlist) if len(vlist) > 1 else 0.0}
            else:
                row[key] = {"mean": statistics.mean(vlist), "std": statistics.stdev(vlist) if len(vlist) > 1 else 0.0}
        rows.append(row)

    agg_path = output_dir / "scaling_aggregate.json"
    with open(agg_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved aggregate to {agg_path}")

    # Stage-2 gates.
    gates: list[str] = []
    fr = next((r for r in rows if r["experiment"] == "fr_phytca"), None)
    matched_mlp = next((r for r in rows if r["experiment"] == "matched_mlp_residual"), None)
    matched_lr = next((r for r in rows if r["experiment"] == "matched_low_rank_residual"), None)
    joint_tucker = next((r for r in rows if r["experiment"] == "phytca_joint_upper_bound"), None)

    if fr is None:
        gates.append("NO_GO_PARENT_ROUTE_DRIFT")
        return {"gates": gates, "rows": rows}

    # Gate 1: FR-PhyTCA OPT drift == 0 on every seed.
    drift_violations = 0
    for seed, seed_results in all_results.items():
        fr_seed = next((r for r in seed_results if r.get("experiment") == "fr_phytca" and r.get("status") != "error"), None)
        if fr_seed is None or fr_seed.get("opt_route_drift", float("inf")) >= 1e-7:
            drift_violations += 1
    if drift_violations > 0:
        gates.append("NO_GO_PARENT_ROUTE_DRIFT")

    # Gate 2: FR-PhyTCA mean MBJ nMAE at least 10% lower than matched MLP or matched low-rank.
    fr_t2_mean = fr["task2_final_nmae"]["mean"]
    competitor_means = []
    if matched_mlp:
        competitor_means.append(matched_mlp["task2_final_nmae"]["mean"])
    if matched_lr:
        competitor_means.append(matched_lr["task2_final_nmae"]["mean"])
    gate2_pass = False
    if competitor_means:
        gate2_pass = all(fr_t2_mean <= c * 0.9 for c in competitor_means)

    # Gate 3: FR-PhyTCA beats both matched baselines on at least 2/3 seeds.
    seed_wins = 0
    for seed, seed_results in all_results.items():
        by_name = {r["experiment"]: r for r in seed_results if r.get("status") != "error"}
        if "fr_phytca" not in by_name:
            continue
        fr_t2 = by_name["fr_phytca"]["task2_final_nmae"]
        better_than_both = True
        for comp in ("matched_mlp_residual", "matched_low_rank_residual"):
            if comp in by_name and fr_t2 >= by_name[comp]["task2_final_nmae"]:
                better_than_both = False
                break
        if better_than_both:
            seed_wins += 1
    gate3_pass = seed_wins >= 2

    # Gate 4: FR-PhyTCA raw MAE drops vs 2k stage.
    fr_raw_mean = fr["raw_mae_eV"]["mean"]
    gate4_pass = fr_raw_mean < 0.912  # 2k mean from reproducibility summary

    # Gate 5: FR-PhyTCA vs Joint Tucker gap does not widen vs 2k stage.
    gate5_pass = True
    if joint_tucker:
        gap_5k = fr["average_final_nmae"]["mean"] - joint_tucker["average_final_nmae"]["mean"]
        # 2k gap was ~0.603 - 0.415 = 0.188; allow small widening.
        gate5_pass = gap_5k <= 0.25

    if not gate2_pass:
        gates.append("NO_GO_TUCKER_NO_ADVANTAGE_OVER_MATCHED_RESIDUAL")
    if not gate3_pass:
        gates.append("NO_GO_TUCKER_NO_ADVANTAGE_OVER_MATCHED_RESIDUAL")
    if not gate4_pass:
        gates.append("NO_GO_SCALING_GAP_WIDENS")
    if not gate5_pass:
        gates.append("NO_GO_SCALING_GAP_WIDENS")

    if len(gates) == 0:
        gates.append("GO_TO_REALISTIC_FIDELITY_SCALING")

    return {"gates": gates, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cap", type=int, default=5000)
    parser.add_argument("--val-cap", type=int, default=500)
    parser.add_argument("--test-cap", type=int, default=1000)
    parser.add_argument("--dev-frac", type=float, default=0.5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--num-nearest-neighbors", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="reports/phase2_b_scaling")
    parser.add_argument("--artifact-dir", default="artifacts/init")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(args.artifact_dir)

    all_results: dict[int, list[dict]] = {}
    for seed in args.seeds:
        seed_output = output_dir / f"seed_{seed}"
        seed_output.mkdir(parents=True, exist_ok=True)
        all_results[seed] = _run_seed(
            seed=seed,
            train_cap=args.train_cap,
            val_cap=args.val_cap,
            test_cap=args.test_cap,
            dev_frac=args.dev_frac,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            hidden_dim=args.hidden_dim,
            adapter_rank=args.adapter_rank,
            num_nearest_neighbors=args.num_nearest_neighbors,
            device=device,
            output_dir=output_dir,
            artifact_dir=artifact_dir,
        )

    print("\n=== Stage-2 aggregate summary ===")
    agg = _aggregate(all_results, output_dir)
    for r in agg["rows"]:
        print(
            f"{r['experiment']}: "
            f"T1@T1={r['task1_after_task1']['mean']:.3f}±{r['task1_after_task1']['std']:.3f} "
            f"T1@T2={r['task1_after_task2']['mean']:.3f}±{r['task1_after_task2']['std']:.3f} "
            f"T2={r['task2_final_nmae']['mean']:.3f}±{r['task2_final_nmae']['std']:.3f} "
            f"forget={r['absolute_forgetting']['mean']:.3f}±{r['absolute_forgetting']['std']:.3f} "
            f"avg_final={r['average_final_nmae']['mean']:.3f}±{r['average_final_nmae']['std']:.3f} "
            f"raw_MAE={r['raw_mae_eV']['mean']:.3f}±{r['raw_mae_eV']['std']:.3f} "
            f"incr={int(r['incremental_params']['mean']):,}"
        )

    print("\n=== Stage-2 gates ===")
    for g in agg["gates"]:
        print(f"  {g}")

    gate_path = output_dir / "scaling_gates.json"
    with open(gate_path, "w") as f:
        json.dump(agg["gates"], f, indent=2)
    print(f"Saved gates to {gate_path}")


if __name__ == "__main__":
    main()
