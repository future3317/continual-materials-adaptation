"""Run the six Phase-0 diagnostic experiments on Protocol B.

All experiments use the same 2k/500/500 train/val/continual_dev split and the
same canonical frozen-encoder checkpoint for PhyTCA-derived methods.  Results
are reported on ``continual_dev``; ``final_test`` is held out.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import build_protocol_b, cap_splits
from diagnostics import (
    DIAGNOSTIC_REGISTRY,
    d6_progressive_tucker,
    d6c_independent_low_rank_residual,
    d6d_parameter_matched_mlp_residual,
    d6e_orthogonal_tucker_residual,
    d6f_shared_factor_top_layer,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cap", type=int, default=2000)
    parser.add_argument("--val-cap", type=int, default=500)
    parser.add_argument("--test-cap", type=int, default=1000)
    parser.add_argument("--dev-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--num-nearest-neighbors", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="reports/phase0_b_screening")
    parser.add_argument("--artifact-dir", default="artifacts/init")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks, task_records = _build_two_task_data(
        args.train_cap, args.val_cap, args.test_cap, args.dev_frac, args.seed
    )
    print("=== Protocol B diagnostic experiments ===")
    print(f"seed={args.seed}, tasks={tasks}")
    for t, desc in enumerate(tasks):
        counts = {"train": 0, "val": 0, "continual_dev": 0, "final_test": 0}
        for r in task_records[t]:
            counts[r["split"]] += 1
        print(f"  {desc}: {counts}")

    from train_phytca import _name_to_id
    prop2id, fid2id = _name_to_id(tasks)
    base_state = _canonical_base_state(
        seed=args.seed,
        node_dim=92,
        hidden_dim=args.hidden_dim,
        n_properties=len(prop2id),
        n_fidelities=len(fid2id),
        adapter_rank=args.adapter_rank,
        num_nearest_neighbors=args.num_nearest_neighbors,
        artifact_dir=Path(args.artifact_dir),
        device=device,
    )

    common_kwargs = {
        "node_dim": 92,
        "hidden_dim": args.hidden_dim,
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "patience": args.patience,
        "adapter_rank": args.adapter_rank,
        "num_nearest_neighbors": args.num_nearest_neighbors,
    }

    results: list[dict] = []

    # D1: full joint upper bound.
    try:
        results.append(DIAGNOSTIC_REGISTRY["full_joint_upper_bound"](
            tasks=tasks, task_records=task_records, **common_kwargs
        ))
    except Exception as e:
        print(f"ERROR full_joint_upper_bound: {e}")
        results.append({"experiment": "full_joint_upper_bound", "status": "error", "error": str(e), "traceback": traceback.format_exc()})

    # D2: joint PhyTCA.
    try:
        results.append(DIAGNOSTIC_REGISTRY["phytca_joint_upper_bound"](
            tasks=tasks, task_records=task_records, base_state_dict=base_state, **common_kwargs
        ))
    except Exception as e:
        print(f"ERROR phytca_joint_upper_bound: {e}")
        results.append({"experiment": "phytca_joint_upper_bound", "status": "error", "error": str(e), "traceback": traceback.format_exc()})

    # D3: sequential PhyTCA.
    try:
        results.append(DIAGNOSTIC_REGISTRY["phytca_sequential"](
            tasks=tasks, task_records=task_records, base_state_dict=base_state, mu=0.01, **common_kwargs
        ))
    except Exception as e:
        print(f"ERROR phytca_sequential: {e}")
        results.append({"experiment": "phytca_sequential", "status": "error", "error": str(e), "traceback": traceback.format_exc()})

    # D4/D5/D6 share a single trained OPT parent checkpoint so that any
    # difference in Task-2 performance comes from the correction module, not a
    # different Task-1 starting point.
    print("\nTraining shared OPT parent for D4/D5/D6...")
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

    # D4: frozen OPT + affine correction.
    try:
        results.append(DIAGNOSTIC_REGISTRY["frozen_opt_affine_correction"](
            tasks=tasks, task_records=task_records, base_state_dict=base_state,
            opt_parent_state=opt_parent_bundle, **common_kwargs
        ))
    except Exception as e:
        print(f"ERROR frozen_opt_affine_correction: {e}")
        results.append({"experiment": "frozen_opt_affine_correction", "status": "error", "error": str(e), "traceback": traceback.format_exc()})

    # D5: frozen OPT + residual correction.
    try:
        results.append(DIAGNOSTIC_REGISTRY["frozen_opt_residual_correction"](
            tasks=tasks, task_records=task_records, base_state_dict=base_state,
            opt_parent_state=opt_parent_bundle, **common_kwargs
        ))
    except Exception as e:
        print(f"ERROR frozen_opt_residual_correction: {e}")
        results.append({"experiment": "frozen_opt_residual_correction", "status": "error", "error": str(e), "traceback": traceback.format_exc()})

    # D6: progressive Tucker residual + OPT distillation grid.
    d6_candidates: list[dict] = []
    for lam in [0.0, 0.1, 1.0, 10.0]:
        try:
            r = d6_progressive_tucker(
                tasks=tasks,
                task_records=task_records,
                base_state_dict=base_state,
                lambda_distill=lam,
                opt_parent_state=opt_parent_bundle,
                **common_kwargs,
            )
            d6_candidates.append(r)
            print(
                f"  D6 lambda={lam}: avg_final={r['average_final_nmae']:.4f}, "
                f"opt_route_drift={r.get('opt_route_drift', float('nan')):.2e}"
            )
        except Exception as e:
            print(f"ERROR fr_phytca lambda={lam}: {e}")
            d6_candidates.append({"experiment": f"fr_phytca_distill_{lam}", "status": "error", "error": str(e), "traceback": traceback.format_exc()})

    best_d6 = min(
        (r for r in d6_candidates if r.get("status") != "error"),
        key=lambda r: r["average_final_nmae"],
        default=None,
    )
    if best_d6 is not None:
        results.append(best_d6)
    results.extend([r for r in d6_candidates if r is not best_d6])

    # D6 ablations on the same frozen MBJ parent.
    ablation_candidates: list[dict] = []
    for ablation_fn, name in [
        (d6c_independent_low_rank_residual, "low_rank_residual"),
        (d6d_parameter_matched_mlp_residual, "param_matched_mlp"),
        (d6e_orthogonal_tucker_residual, "orthogonal"),
        (d6f_shared_factor_top_layer, "shared_factor_top_layer"),
    ]:
        try:
            r = ablation_fn(
                tasks=tasks,
                task_records=task_records,
                base_state_dict=base_state,
                opt_parent_state=opt_parent_bundle,
                **common_kwargs,
            )
            ablation_candidates.append(r)
            print(f"  D6-{name}: avg_final={r['average_final_nmae']:.4f}")
        except Exception as e:
            print(f"ERROR D6-{name}: {e}")
            ablation_candidates.append({"experiment": f"fr_phytca_{name}", "status": "error", "error": str(e), "traceback": traceback.format_exc()})
    results.extend(ablation_candidates)

    # Save raw results.
    out_path = output_dir / "diagnostic_experiments.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")

    # Summary and gates.
    print("\n=== Diagnostic summary ===")
    ok_results = [r for r in results if r.get("status") != "error"]
    for r in ok_results:
        print(
            f"{r['experiment']}: "
            f"T1@T1={r['task1_after_task1']:.3f} "
            f"T1@T2={r['task1_after_task2']:.3f} "
            f"T2={r['task2_final_nmae']:.3f} "
            f"forget={r['absolute_forgetting']:.3f} "
            f"avg_final={r['average_final_nmae']:.3f} "
            f"trainable={r['trainable_params']:,} "
            f"incremental={r.get('incremental_params', 0):,}"
        )

    full_joint = next((r for r in ok_results if r["experiment"] == "full_joint_upper_bound"), None)
    phytca_joint = next((r for r in ok_results if r["experiment"] == "phytca_joint_upper_bound"), None)
    phytca_seq = next((r for r in ok_results if r["experiment"] == "phytca_sequential"), None)
    frozen_res = next((r for r in ok_results if r["experiment"] == "frozen_opt_residual_correction"), None)
    best_d6_obj = next((r for r in ok_results if r["experiment"].startswith("fr_phytca")), None)

    gates: list[str] = ["DIAGNOSIS_ADAPTER_ON_RANDOM_BACKBONE"]

    if phytca_joint and full_joint:
        arch_gap = phytca_joint["average_final_nmae"] - full_joint["average_final_nmae"]
        rel_arch_gap = arch_gap / full_joint["average_final_nmae"]
        print(f"\narchitecture_gap = {arch_gap:.4f} ({rel_arch_gap*100:.1f}%)")
        if arch_gap > 0.2:
            gates.append("DIAGNOSIS_CAPACITY_OR_REPRESENTATION_LIMIT")

    if phytca_seq and phytca_joint:
        cont_gap = phytca_seq["average_final_nmae"] - phytca_joint["average_final_nmae"]
        print(f"continual_gap = {cont_gap:.4f}")
        # T1@T1 must be reasonable before blaming Task-2 interference.
        if cont_gap > 0.2 and phytca_seq["task1_after_task1"] <= phytca_joint["task1_after_task1"] + 0.2:
            gates.append("DIAGNOSIS_SEQUENTIAL_INTERFERENCE")
        elif cont_gap > 0.2:
            gates.append("DIAGNOSIS_SEQUENTIAL_OPTIMIZATION_FAILURE")

    # Fairness audit: D4/D5/D6 must share the same T1@T1 from the common parent.
    frozen_methods = [r for r in ok_results if r["experiment"] in (
        "frozen_opt_affine_correction", "frozen_opt_residual_correction"
    ) or r["experiment"].startswith("fr_phytca")]
    if len(frozen_methods) >= 2:
        t1_values = [r["task1_after_task1"] for r in frozen_methods]
        max_t1_spread = max(t1_values) - min(t1_values)
        print(f"T1@T1 spread across D4/D5/D6 = {max_t1_spread:.2e}")
        if max_t1_spread > 1e-6:
            gates.append("NO_GO_PARENT_CHECKPOINT_NOT_IDENTICAL")

    # Frozen-parent hard gate.
    if best_d6_obj:
        drift = best_d6_obj.get("opt_route_drift", float("inf"))
        print(f"best D6 ({best_d6_obj['experiment']}): avg_final={best_d6_obj['average_final_nmae']:.4f}, opt_route_drift={drift:.2e}")
        if drift >= 1e-7:
            gates.append("NO_GO_OPT_ROUTE_DRIFT")

    if frozen_res:
        print(f"frozen OPT residual: forgetting={frozen_res['absolute_forgetting']:.4f}, T2={frozen_res['task2_final_nmae']:.4f}")
        if abs(frozen_res["absolute_forgetting"]) < 0.05 and frozen_res["task2_final_nmae"] <= 0.885:
            gates.append("GO_TO_FIDELITY_RESIDUAL_PHYTCA")

    print("\nGates:")
    for g in gates:
        print(f"  {g}")


if __name__ == "__main__":
    main()
