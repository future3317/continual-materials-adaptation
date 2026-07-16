"""Shared training and evaluation utilities.

This module contains the reusable pieces that were previously scattered across
``phytca.py`` and ``baselines.py``. Those legacy modules are kept in
``legacy/`` for reference but should not be used in new code.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import JARVISCrystalDataset, collate_crystals
from train_phytca import evaluate_loader


def compute_mad(targets: torch.Tensor) -> float:
    """Mean absolute deviation of a target tensor."""
    return float(torch.abs(targets - targets.mean()).mean())


def normalized_mae(pred: torch.Tensor, target: torch.Tensor, mad: float) -> torch.Tensor:
    """Normalized MAE by mean absolute deviation."""
    return torch.abs(pred - target).mean() / max(mad, 1e-8)


def forgetting(nmaes: list[list[float]]) -> float:
    """Average per-task forgetting across a continual run."""
    T = len(nmaes)
    if T <= 1:
        return 0.0
    vals = []
    for i in range(T):
        best = min(nmaes[t][i] for t in range(i, T))
        final = nmaes[T - 1][i]
        vals.append(max(0.0, final - best))
    return sum(vals) / len(vals)


def backward_transfer(nmaes: list[list[float]]) -> float:
    """Average backward transfer for an error metric (lower is better)."""
    T = len(nmaes)
    if T <= 1:
        return 0.0
    vals = []
    for i in range(T - 1):
        best = nmaes[i][i]
        final = nmaes[T - 1][i]
        vals.append(best - final)
    return sum(vals) / len(vals)


def forward_transfer(nmaes: list[list[float]], scratch_nmaes: list[float]) -> float:
    """Average forward transfer vs training each task from scratch."""
    if not nmaes:
        return 0.0
    return sum(scratch_nmaes[t] - nmaes[t][t] for t in range(len(nmaes))) / len(nmaes)


def _make_loaders(
    recs: list[dict],
    batch_size: int,
    mean: float | None = None,
    std: float | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, torch.Tensor, float]:
    """Return train/val/test loaders plus normalization stats for one task."""
    train_ds = JARVISCrystalDataset(recs, split="train")
    val_ds = JARVISCrystalDataset(recs, split="val")
    test_ds = JARVISCrystalDataset(recs, split="test")

    if mean is None or std is None:
        all_targets = torch.tensor(
            [r["target"] for r in recs if r.get("split") == "train"], dtype=torch.float32
        )
        mean = float(all_targets.mean())
        std = float(all_targets.std().clamp_min(1e-6))

    for ds in (train_ds, val_ds, test_ds):
        ds.target_mean = mean
        ds.target_std = std
        ds.normalize_target = True

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_crystals
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals
    )

    targets_for_mad = torch.tensor(
        [r["target"] for r in recs if r.get("split") == "train"], dtype=torch.float32
    )
    mad = compute_mad(targets_for_mad)
    return train_loader, val_loader, test_loader, torch.tensor(mean), torch.tensor(std), mad


def _train_one_task(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    prop_id: int,
    fid_id: int,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    patience: int = 5,
    extra_loss_fn: Callable | None = None,
) -> tuple[float, torch.Tensor, torch.Tensor, float]:
    """Generic single-task training with early stopping.

    ``model`` must have ``forward(node_feats, coords, mask, original_mask, prop_id, fid_id)``.
    ``extra_loss_fn(model)`` adds regularization terms such as EWC.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    all_targets = []
    for _, _, _, _, y in train_loader:
        all_targets.append(y)
    all_targets = torch.cat(all_targets)
    target_mean = all_targets.mean()
    target_std = all_targets.std().clamp_min(1e-6)
    mad = compute_mad(all_targets)

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
            if extra_loss_fn is not None:
                loss += extra_loss_fn(model)
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


def _evaluate_all_seen(
    model: nn.Module,
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    task_stats: list[tuple[torch.Tensor, torch.Tensor, float]],
    prop2id: dict[str, int],
    fid2id: dict[str, int],
    batch_size: int,
    device: torch.device,
    t: int,
) -> list[float]:
    """Evaluate model on test sets of tasks 0..t."""
    nmaes: list[float] = []
    for prev_t in range(t + 1):
        _, prev_prop, prev_fid = tasks[prev_t]
        pid = prop2id[prev_prop]
        fid = fid2id[prev_fid]
        mean_p, std_p, mad_p = task_stats[prev_t]
        test_ds = JARVISCrystalDataset(task_records[prev_t], split="test")
        test_ds.target_mean = float(mean_p)
        test_ds.target_std = float(std_p)
        test_ds.normalize_target = True
        loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals
        )
        nmaes.append(evaluate_loader(model, loader, pid, fid, mean_p, std_p, mad_p, device))
    return nmaes


class LoRALinear(nn.Module):
    """Simple LoRA-augmented linear layer used by legacy screening scripts."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.lora_a = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_b = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** (1.0 / 3))
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.alpha * (x @ self.lora_a @ self.lora_b)
