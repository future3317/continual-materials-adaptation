"""Baseline continual-learning methods for PhyTCA Phase 0 comparison.

Each baseline exposes a ``run`` function with the same signature as
``continual_experiment`` in ``train_phytca.py`` and returns a matrix
``nmaes[t][i]`` of test nMAE on task ``i`` after training task ``t``.
"""

from __future__ import annotations

import copy
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from data import JARVISCrystalDataset, collate_crystals
from legacy.phytca import PhyTCAModel, compute_mad, normalized_mae
from train_phytca import _name_to_id, continual_experiment, evaluate_loader


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


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
        all_targets = torch.tensor([r["target"] for r in recs if r.get("split") == "train"], dtype=torch.float32)
        mean = float(all_targets.mean())
        std = float(all_targets.std().clamp_min(1e-6))

    for ds in (train_ds, val_ds, test_ds):
        ds.target_mean = mean
        ds.target_std = std
        ds.normalize_target = True

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_crystals)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals)

    targets_for_mad = torch.tensor([r["target"] for r in recs if r.get("split") == "train"], dtype=torch.float32)
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

        val_nmae = evaluate_loader(model, val_loader, prop_id, fid_id, target_mean, target_std, mad, device)
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
        loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals)
        nmaes.append(evaluate_loader(model, loader, pid, fid, mean_p, std_p, mad_p, device))
    return nmaes


# ---------------------------------------------------------------------------
# 1. Joint training (upper bound)
# ---------------------------------------------------------------------------


def joint_training(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
) -> tuple[list[list[float]], dict]:
    """Train one model jointly on all tasks; evaluate sequentially for comparison."""
    prop2id, fid2id = _name_to_id(tasks)
    n_props, n_fids = len(prop2id), len(fid2id)
    model = PhyTCAModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=n_props,
        n_fidelities=n_fids,
        n_layers=3,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        freeze_encoder_weights=False,
    ).to(device)

    task_stats: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    loaders: list[tuple[DataLoader, DataLoader]] = []

    for recs in task_records:
        train_loader, val_loader, _, mean, std, mad = _make_loaders(recs, batch_size)
        task_stats.append((mean, std, mad))
        loaders.append((train_loader, val_loader))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_state = None
    best_nmae = float("inf")

    for _ in range(epochs):
        model.train()
        for t, (dataset_tag, prop_name, fid_name) in enumerate(tasks):
            prop_id = prop2id[prop_name]
            fid_id = fid2id[fid_name]
            train_loader, _ = loaders[t]
            mean, std, _ = task_stats[t]
            for node_feats, coords, mask, original_mask, y in train_loader:
                node_feats = node_feats.to(device)
                coords = coords.to(device)
                mask = mask.to(device)
                original_mask = original_mask.to(device)
                y_norm = ((y.to(device) - mean.to(device)) / std.to(device)).float()
                optimizer.zero_grad()
                pred = model(node_feats, coords, mask, original_mask, prop_id, fid_id)
                loss = F.mse_loss(pred, y_norm)
                loss.backward()
                optimizer.step()

        # Validation averaged across tasks.
        val_nmaes = []
        for t, (dataset_tag, prop_name, fid_name) in enumerate(tasks):
            prop_id = prop2id[prop_name]
            fid_id = fid2id[fid_name]
            _, val_loader = loaders[t]
            mean, std, mad = task_stats[t]
            val_nmaes.append(evaluate_loader(model, val_loader, prop_id, fid_id, mean, std, mad, device))
        avg_val = sum(val_nmaes) / len(val_nmaes)
        if avg_val < best_nmae:
            best_nmae = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Evaluate sequentially on each prefix for fair comparison.
    nmaes: list[list[float]] = []
    for t in range(len(tasks)):
        nmaes.append(_evaluate_all_seen(model, tasks, task_records, task_stats, prop2id, fid2id, batch_size, device, t))
    return nmaes, {"model": model, "adapter_params": sum(p.numel() for p in model.parameters())}


# ---------------------------------------------------------------------------
# 2. Independent model per task
# ---------------------------------------------------------------------------


def independent_models(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
) -> tuple[list[list[float]], dict]:
    """Train a separate model for each task."""
    prop2id, fid2id = _name_to_id(tasks)
    n_props, n_fids = len(prop2id), len(fid2id)
    models: list[PhyTCAModel] = []
    task_stats: list[tuple[torch.Tensor, torch.Tensor, float]] = []

    for t, (dataset_tag, prop_name, fid_name) in enumerate(tasks):
        prop_id = prop2id[prop_name]
        fid_id = fid2id[fid_name]
        recs = task_records[t]
        train_loader, val_loader, _, mean, std, mad = _make_loaders(recs, batch_size)
        task_stats.append((mean, std, mad))

        model = PhyTCAModel(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            n_properties=n_props,
            n_fidelities=n_fids,
            n_layers=3,
            adapter_rank=adapter_rank,
            num_nearest_neighbors=num_nearest_neighbors,
            freeze_encoder_weights=False,
        ).to(device)
        _train_one_task(model, train_loader, val_loader, prop_id, fid_id, device, epochs=epochs, lr=lr)
        models.append(model)

    nmaes: list[list[float]] = []
    for t in range(len(tasks)):
        row: list[float] = []
        for i in range(t + 1):
            _, prop_name, fid_name = tasks[i]
            pid = prop2id[prop_name]
            fid = fid2id[fid_name]
            mean, std, mad = task_stats[i]
            test_ds = JARVISCrystalDataset(task_records[i], split="test")
            test_ds.target_mean = float(mean)
            test_ds.target_std = float(std)
            test_ds.normalize_target = True
            loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals)
            row.append(evaluate_loader(models[i], loader, pid, fid, mean, std, mad, device))
        nmaes.append(row)

    total_params = sum(sum(p.numel() for p in m.parameters()) for m in models)
    return nmaes, {"models": models, "adapter_params": total_params}


# ---------------------------------------------------------------------------
# 3. Sequential fine-tuning
# ---------------------------------------------------------------------------


def sequential_finetuning(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
) -> tuple[list[list[float]], dict]:
    """Train one model sequentially; all weights are updated each task."""
    prop2id, fid2id = _name_to_id(tasks)
    n_props, n_fids = len(prop2id), len(fid2id)
    model = PhyTCAModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=n_props,
        n_fidelities=n_fids,
        n_layers=3,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        freeze_encoder_weights=False,
    ).to(device)

    task_stats: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    nmaes: list[list[float]] = []

    for t, (dataset_tag, prop_name, fid_name) in enumerate(tasks):
        prop_id = prop2id[prop_name]
        fid_id = fid2id[fid_name]
        train_loader, val_loader, _, mean, std, mad = _make_loaders(task_records[t], batch_size)
        task_stats.append((mean, std, mad))
        _train_one_task(model, train_loader, val_loader, prop_id, fid_id, device, epochs=epochs, lr=lr)
        nmaes.append(_evaluate_all_seen(model, tasks, task_records, task_stats, prop2id, fid2id, batch_size, device, t))

    return nmaes, {"model": model, "adapter_params": sum(p.numel() for p in model.parameters())}


# ---------------------------------------------------------------------------
# 4. Frozen encoder + independent heads
# ---------------------------------------------------------------------------


def frozen_encoder_independent_heads(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
) -> tuple[list[list[float]], dict]:
    """Freeze crystal graph encoder; train a separate head per (property, fidelity)."""
    prop2id, fid2id = _name_to_id(tasks)
    n_props, n_fids = len(prop2id), len(fid2id)
    model = PhyTCAModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=n_props,
        n_fidelities=n_fids,
        n_layers=3,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        freeze_encoder_weights=True,
    ).to(device)

    # Only heads are trainable.
    for p in model.parameters():
        p.requires_grad = False
    for head in model.heads.values():
        for p in head.parameters():
            p.requires_grad = True

    task_stats: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    nmaes: list[list[float]] = []

    for t, (dataset_tag, prop_name, fid_name) in enumerate(tasks):
        prop_id = prop2id[prop_name]
        fid_id = fid2id[fid_name]
        train_loader, val_loader, _, mean, std, mad = _make_loaders(task_records[t], batch_size)
        task_stats.append((mean, std, mad))
        _train_one_task(model, train_loader, val_loader, prop_id, fid_id, device, epochs=epochs, lr=lr)
        nmaes.append(_evaluate_all_seen(model, tasks, task_records, task_stats, prop2id, fid2id, batch_size, device, t))

    return nmaes, {"model": model, "adapter_params": sum(p.numel() for p in model.parameters() if p.requires_grad)}


# ---------------------------------------------------------------------------
# 5. EWC
# ---------------------------------------------------------------------------


class EWCLearner:
    """EWC regularizer storing diagonal Fisher per parameter."""

    def __init__(self, model: nn.Module, lam: float = 1e4) -> None:
        self.lam = lam
        self.means: list[torch.Tensor] = []
        self.fishers: list[torch.Tensor] = []
        self.param_names: list[str] = []
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.param_names.append(name)
                self.means.append(p.data.clone())
                self.fishers.append(torch.zeros_like(p.data))

    def update_fisher(self, model: nn.Module, train_loader: DataLoader, prop_id: int, fid_id: int, device: torch.device) -> None:
        """Accumulate diagonal Fisher on the current task."""
        model.eval()
        for i in range(len(self.fishers)):
            self.fishers[i].zero_()
        count = 0
        for node_feats, coords, mask, original_mask, y in train_loader:
            node_feats = node_feats.to(device)
            coords = coords.to(device)
            mask = mask.to(device)
            original_mask = original_mask.to(device)
            y = y.to(device)
            model.zero_grad()
            pred = model(node_feats, coords, mask, original_mask, prop_id, fid_id)
            loss = F.mse_loss(pred, y)
            loss.backward()
            for i, (name, p) in enumerate((n, p) for n, p in model.named_parameters() if p.requires_grad):
                if p.grad is not None:
                    self.fishers[i] += p.grad.data.pow(2)
            count += 1
        for i in range(len(self.fishers)):
            if count > 0:
                self.fishers[i] /= count
            self.means[i] = next(p for n, p in model.named_parameters() if n == self.param_names[i]).data.clone()

    def penalty(self, model: nn.Module) -> torch.Tensor:
        loss = 0.0
        for i, (name, p) in enumerate((n, p) for n, p in model.named_parameters() if p.requires_grad):
            loss += (self.fishers[i] * (p - self.means[i].to(p.device)).pow(2)).sum()
        return self.lam * loss


def ewc(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    lam: float = 1e4,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
) -> tuple[list[list[float]], dict]:
    """Sequential fine-tuning with EWC regularization."""
    prop2id, fid2id = _name_to_id(tasks)
    n_props, n_fids = len(prop2id), len(fid2id)
    model = PhyTCAModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=n_props,
        n_fidelities=n_fids,
        n_layers=3,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        freeze_encoder_weights=False,
    ).to(device)

    ewc_learner = EWCLearner(model, lam=lam)
    task_stats: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    nmaes: list[list[float]] = []

    for t, (dataset_tag, prop_name, fid_name) in enumerate(tasks):
        prop_id = prop2id[prop_name]
        fid_id = fid2id[fid_name]
        train_loader, val_loader, _, mean, std, mad = _make_loaders(task_records[t], batch_size)
        task_stats.append((mean, std, mad))
        _train_one_task(
            model, train_loader, val_loader, prop_id, fid_id, device,
            epochs=epochs, lr=lr,
            extra_loss_fn=lambda m: ewc_learner.penalty(m),
        )
        ewc_learner.update_fisher(model, train_loader, prop_id, fid_id, device)
        nmaes.append(_evaluate_all_seen(model, tasks, task_records, task_stats, prop2id, fid2id, batch_size, device, t))

    return nmaes, {"model": model, "adapter_params": sum(p.numel() for p in model.parameters())}


# ---------------------------------------------------------------------------
# 6. Experience replay
# ---------------------------------------------------------------------------


def experience_replay(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 5,
    buffer_size_per_task: int = 100,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
) -> tuple[list[list[float]], dict]:
    """Sequential fine-tuning with a small replay buffer of past training samples."""
    prop2id, fid2id = _name_to_id(tasks)
    n_props, n_fids = len(prop2id), len(fid2id)
    model = PhyTCAModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=n_props,
        n_fidelities=n_fids,
        n_layers=3,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        freeze_encoder_weights=False,
    ).to(device)

    buffer: list[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    task_stats: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    nmaes: list[list[float]] = []

    for t, (dataset_tag, prop_name, fid_name) in enumerate(tasks):
        prop_id = prop2id[prop_name]
        fid_id = fid2id[fid_name]
        train_loader, val_loader, _, mean, std, mad = _make_loaders(task_records[t], batch_size)
        task_stats.append((mean, std, mad))

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
                y_norm = ((y.to(device) - mean.to(device)) / std.to(device)).float()

                optimizer.zero_grad()
                pred = model(node_feats, coords, mask, original_mask, prop_id, fid_id)
                loss = F.mse_loss(pred, y_norm)

                # Replay old tasks.
                if buffer:
                    n_replay = min(len(buffer), batch_size)
                    replay_samples = buffer[:n_replay]
                    for rp, rf, r_nf, r_c, r_m, r_om, r_y in replay_samples:
                        r_pred = model(r_nf, r_c, r_m, r_om, rp, rf)
                        loss += F.mse_loss(r_pred, r_y)

                loss.backward()
                optimizer.step()

            val_nmae = evaluate_loader(model, val_loader, prop_id, fid_id, mean, std, mad, device)
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

        # Store a few batches in the replay buffer.
        stored = 0
        for node_feats, coords, mask, original_mask, y in train_loader:
            if stored >= buffer_size_per_task:
                break
            node_feats = node_feats.to(device)
            coords = coords.to(device)
            mask = mask.to(device)
            original_mask = original_mask.to(device)
            y_norm = ((y.to(device) - mean.to(device)) / std.to(device)).float()
            buffer.append((prop_id, fid_id, node_feats, coords, mask, original_mask, y_norm))
            stored += node_feats.size(0)

        nmaes.append(_evaluate_all_seen(model, tasks, task_records, task_stats, prop2id, fid2id, batch_size, device, t))

    return nmaes, {"model": model, "adapter_params": sum(p.numel() for p in model.parameters()), "replay_storage": len(buffer)}


# ---------------------------------------------------------------------------
# 7. Independent LoRA
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """LoRA adaptation of a linear layer."""

    def __init__(self, base: nn.Linear, rank: int = 4) -> None:
        super().__init__()
        self.base = base
        self.lora_a = nn.Parameter(torch.randn(base.in_features, rank) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(rank, base.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + x @ self.lora_a @ self.lora_b


def _attach_lora(model: PhyTCAModel, rank: int, active: bool = True) -> None:
    """Replace selected Linear layers with LoRA wrappers for the current task."""
    # Only adapt node_embed and heads; encoder internals are left frozen.
    if active:
        model.node_embed = LoRALinear(model.node_embed, rank=rank)
        for key, head in model.heads.items():
            model.heads[key] = LoRALinear(head, rank=rank)


def independent_lora(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    lora_rank: int = 4,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
) -> tuple[list[list[float]], dict]:
    """Separate LoRA adapter per task on a frozen crystal graph encoder."""
    prop2id, fid2id = _name_to_id(tasks)
    n_props, n_fids = len(prop2id), len(fid2id)
    base_model = PhyTCAModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=n_props,
        n_fidelities=n_fids,
        n_layers=3,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        freeze_encoder_weights=True,
    ).to(device)

    task_models: list[PhyTCAModel] = []
    task_stats: list[tuple[torch.Tensor, torch.Tensor, float]] = []

    for t, (dataset_tag, prop_name, fid_name) in enumerate(tasks):
        prop_id = prop2id[prop_name]
        fid_id = fid2id[fid_name]
        train_loader, val_loader, _, mean, std, mad = _make_loaders(task_records[t], batch_size)
        task_stats.append((mean, std, mad))

        model = copy.deepcopy(base_model)
        _attach_lora(model, rank=lora_rank)
        model.to(device)
        _train_one_task(model, train_loader, val_loader, prop_id, fid_id, device, epochs=epochs, lr=lr)
        task_models.append(model)

    nmaes: list[list[float]] = []
    for t in range(len(tasks)):
        row: list[float] = []
        for i in range(t + 1):
            _, prop_name, fid_name = tasks[i]
            pid = prop2id[prop_name]
            fid = fid2id[fid_name]
            mean, std, mad = task_stats[i]
            test_ds = JARVISCrystalDataset(task_records[i], split="test")
            test_ds.target_mean = float(mean)
            test_ds.target_std = float(std)
            test_ds.normalize_target = True
            loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals)
            row.append(evaluate_loader(task_models[i], loader, pid, fid, mean, std, mad, device))
        nmaes.append(row)

    total_params = sum(sum(p.numel() for p in m.parameters()) for m in task_models)
    return nmaes, {"models": task_models, "adapter_params": total_params}


# ---------------------------------------------------------------------------
# 8. Shared LoRA bank
# ---------------------------------------------------------------------------


def shared_lora_bank(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 5,
    lora_rank: int = 4,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
) -> tuple[list[list[float]], dict]:
    """One LoRA bank shared across tasks; task-specific scaling is learned."""
    prop2id, fid2id = _name_to_id(tasks)
    n_props, n_fids = len(prop2id), len(fid2id)
    model = PhyTCAModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=n_props,
        n_fidelities=n_fids,
        n_layers=3,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        freeze_encoder_weights=True,
    ).to(device)

    _attach_lora(model, rank=lora_rank)
    model.to(device)

    # Add a task-specific scalar gate per LoRA parameter.
    task_gates: list[dict[str, nn.Parameter]] = []
    for t, _ in enumerate(tasks):
        gates: dict[str, nn.Parameter] = {}
        for name, p in model.named_parameters():
            if "lora" in name:
                gates[name] = nn.Parameter(torch.ones_like(p))
        task_gates.append(gates)

    task_stats: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    nmaes: list[list[float]] = []

    for t, (dataset_tag, prop_name, fid_name) in enumerate(tasks):
        prop_id = prop2id[prop_name]
        fid_id = fid2id[fid_name]
        train_loader, val_loader, _, mean, std, mad = _make_loaders(task_records[t], batch_size)
        task_stats.append((mean, std, mad))

        # Train only current task's gates plus the shared LoRA weights.
        params = list(model.parameters()) + list(task_gates[t].values())
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
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
                y_norm = ((y.to(device) - mean.to(device)) / std.to(device)).float()

                # Apply current task gates.
                gate_state = {}
                for name, p in model.named_parameters():
                    if name in task_gates[t]:
                        gate_state[name] = p.data.clone()
                        p.data = p.data * task_gates[t][name]

                optimizer.zero_grad()
                pred = model(node_feats, coords, mask, original_mask, prop_id, fid_id)
                loss = F.mse_loss(pred, y_norm)
                loss.backward()
                optimizer.step()

                # Restore base weights.
                for name, p in model.named_parameters():
                    if name in gate_state:
                        p.data = gate_state[name]

            val_nmae = evaluate_loader(model, val_loader, prop_id, fid_id, mean, std, mad, device)
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
        nmaes.append(_evaluate_all_seen(model, tasks, task_records, task_stats, prop2id, fid2id, batch_size, device, t))

    return nmaes, {"model": model, "adapter_params": sum(p.numel() for p in model.parameters())}


# ---------------------------------------------------------------------------
# 9. Architecture-matched FR-PhyTCA baselines (new ContinualCrystalModel)
# ---------------------------------------------------------------------------


def _fr_phytca_baseline(
    adapter_name: str,
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
) -> tuple[list[list[float]], dict]:
    """Run ``train_phytca.continual_experiment`` with the requested adapter."""
    return continual_experiment(
        tasks=tasks,
        task_records=task_records,
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        adapter_name=adapter_name,
        adapter_rank=adapter_rank,
        n_layers=3,
        num_nearest_neighbors=num_nearest_neighbors,
        update_coors=False,
    )


def fr_lora_ab(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
) -> tuple[list[list[float]], dict]:
    """FR-PhyTCA-style training with LoRA-AB adapters."""
    return _fr_phytca_baseline(
        "lora_ab",
        tasks,
        task_records,
        node_dim,
        hidden_dim,
        device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
    )


def fr_lora_aba(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
) -> tuple[list[list[float]], dict]:
    """FR-PhyTCA-style training with LoRA-ABA adapters."""
    return _fr_phytca_baseline(
        "lora_aba",
        tasks,
        task_records,
        node_dim,
        hidden_dim,
        device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
    )


def fr_single_child_tucker(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
) -> tuple[list[list[float]], dict]:
    """FR-PhyTCA: exact-retention continual learning with single-child Tucker."""
    return _fr_phytca_baseline(
        "single_child_tucker",
        tasks,
        task_records,
        node_dim,
        hidden_dim,
        device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
    )


def fr_multi_axis_tucker(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
) -> tuple[list[list[float]], dict]:
    """FR-PhyTCA-style training with full multi-axis Tucker adapters.

    This is most meaningful when ``n_properties >= 2`` or ``n_fidelities >= 3``;
    otherwise the property/fidelity modes have nothing to share across.
    """
    return _fr_phytca_baseline(
        "multi_axis_tucker",
        tasks,
        task_records,
        node_dim,
        hidden_dim,
        device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
    )


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------


BASELINE_REGISTRY: dict[str, Callable] = {
    "phytca": None,  # handled separately to allow mu/grid choices
    "joint": joint_training,
    "independent": independent_models,
    "sequential": sequential_finetuning,
    "frozen_heads": frozen_encoder_independent_heads,
    "ewc": ewc,
    "replay": experience_replay,
    "independent_lora": independent_lora,
    "shared_lora": shared_lora_bank,
    "fr_lora_ab": fr_lora_ab,
    "fr_lora_aba": fr_lora_aba,
    "fr_single_child_tucker": fr_single_child_tucker,
    "fr_multi_axis_tucker": fr_multi_axis_tucker,
}
