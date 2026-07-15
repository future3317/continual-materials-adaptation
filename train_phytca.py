"""Continual learning training script for exact-retention crystal models.

This script uses ``models.ContinualCrystalModel`` and the adapter zoo from
``adapters.py``.  It enforces exact retention by structural isolation: each
(property, fidelity) task owns a private adapter bank + head, and old tasks are
frozen by ``requires_grad=False`` and excluded from the optimizer.

Normalization policy (addresses 反馈_2.md 5.1):
* Each fidelity is normalized with its own training-set mean/std.
* The model predicts child-normalized targets directly; no parent and child
  normalized quantities are added.
* Correction baselines that explicitly add parent + residual must use
  ``models.PredictionResidualHead`` to work in physical units.
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import (
    JARVISCrystalDataset,
    build_protocol_a,
    build_protocol_b,
    collate_crystals,
)
from models import (
    ContinualCrystalModel,
    backward_transfer,
    compute_mad,
    forgetting,
    normalized_mae,
)


def _name_to_id(tasks: list[tuple[str, str, str]]) -> tuple[dict[str, int], dict[str, int]]:
    """Map property and fidelity names to contiguous integer IDs."""
    props = []
    fids = []
    for _, p, f in tasks:
        if p not in props:
            props.append(p)
        if f not in fids:
            fids.append(f)
    return {p: i for i, p in enumerate(props)}, {f: i for i, f in enumerate(fids)}


def _last_occurrences(tasks: list[tuple[str, str, str]]) -> set[tuple[int, int]]:
    """Return task indices that are the last occurrence of their (prop, fid)."""
    prop2id, fid2id = _name_to_id(tasks)
    last: dict[tuple[int, int], int] = {}
    for t, (_, p, f) in enumerate(tasks):
        last[(prop2id[p], fid2id[f])] = t
    return set(last.values())


def evaluate_loader(
    model: ContinualCrystalModel,
    loader: DataLoader,
    prop_id: int,
    fid_id: int,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    mad: float,
    device: torch.device,
) -> float:
    """Evaluate nMAE on a loader for a given task."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for node_feats, coords, mask, original_mask, y in loader:
            node_feats = node_feats.to(device)
            coords = coords.to(device)
            mask = mask.to(device)
            original_mask = original_mask.to(device)
            pred_norm = model(node_feats, coords, mask, original_mask, prop_id, fid_id)
            pred = pred_norm * target_std.to(device) + target_mean.to(device)
            preds.append(pred.cpu())
            targets.append(y)
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    return float(normalized_mae(preds, targets, mad))


def train_task(
    model: ContinualCrystalModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    prop_id: int,
    fid_id: int,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
) -> tuple[float, torch.Tensor, torch.Tensor, float]:
    """Train one continual task and return best validation nMAE + stats."""
    model.add_task(prop_id, fid_id)

    # Use training-set statistics for normalization.
    all_targets = []
    for _, _, _, _, y in train_loader:
        all_targets.append(y)
    all_targets = torch.cat(all_targets)
    target_mean = all_targets.mean()
    target_std = all_targets.std().clamp_min(1e-6)
    mad = compute_mad(all_targets)

    trainable = model.current_trainable_parameters()
    if not trainable:
        raise RuntimeError(
            f"No trainable parameters for task (prop={prop_id}, fid={fid_id}); "
            "the task may have been frozen already."
        )

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_nmae = float("inf")
    best_state = None
    patience_counter = 0

    for _ in range(epochs):
        model.train()
        for node_feats, coords, mask, original_mask, y in train_loader:
            node_feats = node_feats.to(device)
            coords = coords.to(device)
            mask = mask.to(device)
            original_mask = original_mask.to(device)
            y_norm = ((y.to(device) - target_mean) / target_std).float()

            optimizer.zero_grad()
            pred = model(node_feats, coords, mask, original_mask, prop_id, fid_id)
            loss = F.mse_loss(pred, y_norm)
            loss.backward()
            optimizer.step()

        val_nmae = evaluate_loader(
            model, val_loader, prop_id, fid_id, target_mean, target_std, mad, device
        )
        if val_nmae < best_nmae:
            best_nmae = val_nmae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        scheduler.step()
        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return best_nmae, target_mean, target_std, mad


def continual_experiment(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    adapter_name: str = "single_child_tucker",
    adapter_rank: int = 8,
    n_layers: int = 3,
    num_nearest_neighbors: int = 8,
    update_coors: bool = False,
) -> tuple[list[list[float]], dict]:
    """Run sequential continual learning over JARVIS (property, fidelity) tasks.

    Tasks that share the same (property, fidelity) are treated as data-
    incremental snapshots: they reuse the same adapter bank and head and are
    frozen only after the last occurrence.
    """
    prop2id, fid2id = _name_to_id(tasks)
    n_props = len(prop2id)
    n_fids = len(fid2id)
    model = ContinualCrystalModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=n_props,
        n_fidelities=n_fids,
        adapter_name=adapter_name,
        adapter_rank=adapter_rank,
        n_layers=n_layers,
        num_nearest_neighbors=num_nearest_neighbors,
        update_coors=update_coors,
    ).to(device)

    freeze_steps = _last_occurrences(tasks)

    task_stats: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    nmaes: list[list[float]] = []

    for t, (dataset_tag, prop_name, fid_name) in enumerate(tasks):
        prop_id = prop2id[prop_name]
        fid_id = fid2id[fid_name]
        print(f"\n=== Task {t + 1}/{len(tasks)}: {dataset_tag} / {prop_name} / {fid_name} ===")

        recs = task_records[t]
        train_dataset = JARVISCrystalDataset(recs, split="train")
        val_dataset = JARVISCrystalDataset(recs, split="val")
        train_mean = torch.tensor(train_dataset.target_mean)
        train_std = torch.tensor(train_dataset.target_std)
        for ds in (train_dataset, val_dataset):
            ds.target_mean = float(train_mean)
            ds.target_std = float(train_std)
            ds.normalize_target = True

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_crystals,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_crystals,
        )

        best_nmae, mean, std, mad = train_task(
            model, train_loader, val_loader, prop_id, fid_id, device,
            epochs=epochs, lr=lr, weight_decay=weight_decay, patience=5,
        )
        task_stats.append((mean, std, mad))
        print(f"  Best val nMAE on current task: {best_nmae:.3f}")

        if t in freeze_steps:
            model.freeze_task(prop_id, fid_id)
            print(
                f"  Frozen task (prop={prop_name}, fid={fid_name}); "
                f"incremental params for next task will be fresh."
            )

        # Evaluate on test sets of all tasks seen so far.
        task_nmaes = []
        for prev_t in range(t + 1):
            prev_dataset_tag, prev_prop, prev_fid = tasks[prev_t]
            pid = prop2id[prev_prop]
            pfid = fid2id[prev_fid]
            mean_p, std_p, mad_p = task_stats[prev_t]
            prev_test_recs = task_records[prev_t]
            prev_dataset = JARVISCrystalDataset(prev_test_recs, split="test")
            prev_dataset.target_mean = float(mean_p)
            prev_dataset.target_std = float(std_p)
            prev_dataset.normalize_target = True
            prev_loader = DataLoader(
                prev_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_crystals,
            )
            nmae = evaluate_loader(
                model, prev_loader, pid, pfid, mean_p, std_p, mad_p, device
            )
            task_nmaes.append(nmae)
        nmaes.append(task_nmaes)
        print(f"  test nMAEs after task {t + 1}: {[f'{x:.3f}' for x in task_nmaes]}")

    info = {
        "model": model,
        "adapter_params": sum(
            model.count_task_parameters(p, f) for p, f in model._task_order
        ),
        "parameter_groups": model.get_parameter_group_counts(),
    }
    return nmaes, info


def main():
    from adapters import ADAPTER_REGISTRY

    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["a", "b"], default="a")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--adapter-name",
        choices=list(ADAPTER_REGISTRY.keys()),
        default="single_child_tucker",
    )
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cap", type=int, default=None, help="Per-task sample cap for smoke tests")
    parser.add_argument("--num-nearest-neighbors", type=int, default=8)
    parser.add_argument("--update-coors", action="store_true", help="Allow EGNN to update coordinates")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.protocol == "a":
        tasks, task_records, _ = build_protocol_a(seed=args.seed, n_train_val_per_task=args.cap)
    else:
        tasks, task_records, _ = build_protocol_b(seed=args.seed, n_train_val_per_task=args.cap)

    for t, task_desc in enumerate(tasks):
        print(f"  Task {t + 1} {task_desc}: {len(task_records[t])} structures")

    device = torch.device(args.device)
    nmaes, info = continual_experiment(
        tasks=tasks,
        task_records=task_records,
        node_dim=92,
        hidden_dim=args.hidden_dim,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        adapter_name=args.adapter_name,
        adapter_rank=args.adapter_rank,
        n_layers=args.n_layers,
        num_nearest_neighbors=args.num_nearest_neighbors,
        update_coors=args.update_coors,
    )

    print("\n=== Final Results ===")
    print(f"Parameter groups: {info['parameter_groups']}")
    print(f"Adapter + head parameters: {info['adapter_params']:,}")
    print(f"Average final nMAE: {sum(nmaes[-1]) / len(nmaes[-1]):.3f}")
    print(f"Average forgetting: {forgetting(nmaes):.3f}")
    print(f"Average backward transfer: {backward_transfer(nmaes):.3f}")


if __name__ == "__main__":
    main()
