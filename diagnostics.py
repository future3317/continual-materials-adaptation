"""Diagnostic experiments for the exact-retention continual crystal model.

This module implements the D1–D6 diagnostic experiments from Protocol B using
``models.ContinualCrystalModel`` and the adapter zoo from ``adapters.py``.
All PhyTCA-derived methods enforce exact retention by structural isolation:
each task owns a private adapter bank + head, and completed tasks are frozen
via ``requires_grad=False`` so they are never touched by later optimizers.
"""

from __future__ import annotations

import copy
import hashlib
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from adapters import ADAPTER_REGISTRY
from train_utils import _evaluate_all_seen, _make_loaders, _train_one_task
from data import JARVISCrystalDataset, collate_crystals
from models import (
    ContinualCrystalModel,
    PredictionResidualHead,
    backward_transfer,
    compute_mad,
    forgetting,
    normalized_mae,
)
from train_phytca import _name_to_id, evaluate_loader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_canonical_base(
    model: nn.Module,
    base_state_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    """Load a canonical frozen-encoder state into a ContinualCrystalModel.

    The canonical state comes from ``PhyTCAModel`` (``phytca.py``), so we map
    its key names to the encoder keys of ``ContinualCrystalModel``.  Task-
    specific adapters and heads are not loaded here; they are created later
    with ``add_task``.
    """
    mapped: dict[str, torch.Tensor] = {}
    for k, v in base_state_dict.items():
        if k.startswith("node_embed."):
            mapped["encoder." + k] = v.to(device)
        elif k.startswith("layers.") and ".encoder." in k:
            # layers.0.encoder.conv... -> encoder.layers.0.conv...
            parts = k.split(".")
            new_key = f"encoder.layers.{parts[1]}." + ".".join(parts[3:])
            mapped[new_key] = v.to(device)
        # Adapter / head keys are task-specific and intentionally skipped.

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    # Missing encoder parameters are expected if the canonical state only
    # contains a subset, but missing adapter/head keys are normal.
    if unexpected:
        print(f"  Warning: canonical base load unexpected={unexpected}")


def _train_jointly(
    model: nn.Module,
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    prop2id: dict[str, int],
    fid2id: dict[str, int],
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 3,
    extra_loss_fn: Callable[[nn.Module], torch.Tensor] | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
    """Train ``model`` jointly on all ``tasks`` and return per-task stats."""
    # For ContinualCrystalModel, allocate every task before joint training.
    if hasattr(model, "add_task"):
        for _, prop_name, fid_name in tasks:
            model.add_task(prop2id[prop_name], fid2id[fid_name])

    task_stats: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    loaders: list[tuple[DataLoader, DataLoader]] = []
    for recs in task_records:
        train_loader, val_loader, _, mean, std, mad = _make_loaders(recs, batch_size)
        task_stats.append((mean, std, mad))
        loaders.append((train_loader, val_loader))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_state = None
    best_nmae = float("inf")
    patience_counter = 0

    for _ in range(epochs):
        model.train()
        for t, (_, prop_name, fid_name) in enumerate(tasks):
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
                if extra_loss_fn is not None:
                    loss += extra_loss_fn(model)
                loss.backward()
                optimizer.step()

        val_nmaes = []
        for t, (_, prop_name, fid_name) in enumerate(tasks):
            prop_id = prop2id[prop_name]
            fid_id = fid2id[fid_name]
            _, val_loader = loaders[t]
            mean, std, mad = task_stats[t]
            val_nmaes.append(
                evaluate_loader(model, val_loader, prop_id, fid_id, mean, std, mad, device)
            )
        avg_val = sum(val_nmaes) / len(val_nmaes)

        if avg_val < best_nmae:
            best_nmae = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        scheduler.step()
        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return task_stats


def _train_single_task(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    prop_id: int,
    fid_id: int,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    patience: int = 3,
    extra_loss_fn: Callable[[nn.Module], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Train one task and return normalization stats (mean, std, mad)."""
    all_targets = []
    for _, _, _, _, y in train_loader:
        all_targets.append(y)
    all_targets = torch.cat(all_targets)
    target_mean = all_targets.mean()
    target_std = all_targets.std().clamp_min(1e-6)
    mad = compute_mad(all_targets)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
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

    return torch.tensor(target_mean), torch.tensor(target_std), mad


def _evaluate_on_dev(
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
    """Evaluate on ``continual_dev`` splits of tasks 0..t."""
    nmaes: list[float] = []
    for prev_t in range(t + 1):
        _, prev_prop, prev_fid = tasks[prev_t]
        pid = prop2id[prev_prop]
        fid = fid2id[prev_fid]
        mean_p, std_p, mad_p = task_stats[prev_t]
        dev_ds = JARVISCrystalDataset(task_records[prev_t], split="continual_dev")
        dev_ds.target_mean = float(mean_p)
        dev_ds.target_std = float(std_p)
        dev_ds.normalize_target = True
        loader = DataLoader(
            dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals
        )
        nmaes.append(evaluate_loader(model, loader, pid, fid, mean_p, std_p, mad_p, device))
    return nmaes


def _count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _count_total(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _trainable_param_names(model: nn.Module) -> list[str]:
    """Return names of parameters that currently require gradients."""
    return [name for name, p in model.named_parameters() if p.requires_grad]


def _state_dict_hash(state_dict: dict[str, torch.Tensor]) -> str:
    """SHA256 over a sorted concatenation of CPU tensor bytes."""
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        h.update(key.encode("utf-8"))
        h.update(state_dict[key].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _predictions_hash(predictions: torch.Tensor) -> str:
    return hashlib.sha256(predictions.detach().cpu().numpy().tobytes()).hexdigest()


def train_opt_parent(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    base_state_dict: dict[str, torch.Tensor] | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 3,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """Train the OPT parent once and return a reproducible checkpoint bundle.

    The bundle contains the trained ``ContinualCrystalModel`` state dict,
    normalizer statistics, T1@T1 nMAE on ``continual_dev``, and hashes for
    audit comparisons across D4–D6.
    """
    prop2id, fid2id = _name_to_id(tasks)
    opt_prop_id = prop2id["band_gap"]
    opt_fid_id = fid2id["OptB88vdW"]

    model = ContinualCrystalModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=len(prop2id),
        n_fidelities=len(fid2id),
        adapter_name="single_child_tucker",
        adapter_rank=adapter_rank,
        n_layers=3,
        num_nearest_neighbors=num_nearest_neighbors,
    ).to(device)
    if base_state_dict is not None:
        _load_canonical_base(model, base_state_dict, device)

    model.add_task(opt_prop_id, opt_fid_id)

    train_loader, val_loader, _, mean, std, mad = _make_loaders(task_records[0], batch_size)
    _train_single_task(
        model,
        train_loader,
        val_loader,
        opt_prop_id,
        opt_fid_id,
        device,
        epochs=epochs,
        lr=lr,
        patience=patience,
    )

    # T1@T1 on continual_dev.
    dev_ds = JARVISCrystalDataset(task_records[0], split="continual_dev")
    dev_ds.target_mean = float(mean)
    dev_ds.target_std = float(std)
    dev_ds.normalize_target = True
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals
    )
    t1_after_t1 = evaluate_loader(
        model, dev_loader, opt_prop_id, opt_fid_id, mean, std, mad, device
    )

    # OPT predictions on dev for hash comparison.
    model.eval()
    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for node_feats, coords, mask, original_mask, _ in dev_loader:
            node_feats = node_feats.to(device)
            coords = coords.to(device)
            mask = mask.to(device)
            original_mask = original_mask.to(device)
            pred = model(node_feats, coords, mask, original_mask, opt_prop_id, opt_fid_id)
            preds.append(pred.detach().cpu())
    opt_predictions = torch.cat(preds)

    state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    bundle = {
        "state_dict": state_dict,
        "mean": mean,
        "std": std,
        "mad": mad,
        "task1_after_task1": t1_after_t1,
        "opt_predictions": opt_predictions,
        "state_dict_hash": _state_dict_hash(state_dict),
        "prediction_hash": _predictions_hash(opt_predictions),
    }

    if artifact_dir is not None:
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        torch.save(bundle, artifact_dir / "opt_parent_bundle.pt")

    return bundle


def _run_label_and_metrics(
    label: str,
    model: nn.Module,
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    task_stats: list[tuple[torch.Tensor, torch.Tensor, float]],
    prop2id: dict[str, int],
    fid2id: dict[str, int],
    batch_size: int,
    device: torch.device,
    start_time: float,
    nmaes: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Common metric extraction after a two-task continual run."""
    if nmaes is None:
        nmaes = []
        for t in range(len(tasks)):
            nmaes.append(
                _evaluate_on_dev(
                    model, tasks, task_records, task_stats, prop2id, fid2id, batch_size, device, t
                )
            )

    task1_after_t1 = nmaes[0][0]
    task1_after_t2 = nmaes[1][0]
    task2_final = nmaes[1][1]
    abs_forgetting = task1_after_t2 - task1_after_t1
    bwt = backward_transfer(nmaes)
    avg_final = sum(nmaes[-1]) / len(nmaes[-1])

    # Raw MAE in eV: avg_final * average per-task MAD.
    avg_mad = sum(stats[2] for stats in task_stats) / len(task_stats)
    raw_mae_eV = avg_final * avg_mad

    return {
        "experiment": label,
        "model": model,
        "nmaes": nmaes,
        "task1_after_task1": task1_after_t1,
        "task1_after_task2": task1_after_t2,
        "task2_final_nmae": task2_final,
        "absolute_forgetting": abs_forgetting,
        "bwt": bwt,
        "average_final_nmae": avg_final,
        "raw_mae_eV": raw_mae_eV,
        "trainable_params": _count_trainable(model),
        "stored_params": _count_total(model),
        "wall_time_seconds": time.time() - start_time,
    }


# ---------------------------------------------------------------------------
# D1: Full joint (full fine-tuning upper bound)
# ---------------------------------------------------------------------------


def d1_full_joint(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 3,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
) -> dict[str, Any]:
    """Joint training with the encoder unfrozen (upper bound)."""
    prop2id, fid2id = _name_to_id(tasks)
    model = ContinualCrystalModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=len(prop2id),
        n_fidelities=len(fid2id),
        adapter_name="single_child_tucker",
        adapter_rank=adapter_rank,
        n_layers=3,
        num_nearest_neighbors=num_nearest_neighbors,
    ).to(device)

    # Add every task so the model has all heads and adapter banks.
    for _, prop_name, fid_name in tasks:
        model.add_task(prop2id[prop_name], fid2id[fid_name])

    # Unfreeze the crystal-graph encoder for full fine-tuning.
    for p in model.encoder.parameters():
        p.requires_grad = True

    start = time.time()
    task_stats = _train_jointly(
        model,
        tasks,
        task_records,
        prop2id,
        fid2id,
        device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
    )
    result = _run_label_and_metrics(
        "full_joint_upper_bound",
        model,
        tasks,
        task_records,
        task_stats,
        prop2id,
        fid2id,
        batch_size,
        device,
        start,
    )
    result["incremental_params"] = 0
    return result


# ---------------------------------------------------------------------------
# D2: Joint PhyTCA (frozen encoder, all tasks share adapter banks)
# ---------------------------------------------------------------------------


def d2_joint_phytca(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    base_state_dict: dict[str, torch.Tensor] | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 3,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
) -> dict[str, Any]:
    """Joint training on all tasks with only adapter/head parameters trainable."""
    prop2id, fid2id = _name_to_id(tasks)
    model = ContinualCrystalModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=len(prop2id),
        n_fidelities=len(fid2id),
        adapter_name="single_child_tucker",
        adapter_rank=adapter_rank,
        n_layers=3,
        num_nearest_neighbors=num_nearest_neighbors,
    ).to(device)
    if base_state_dict is not None:
        _load_canonical_base(model, base_state_dict, device)

    for _, prop_name, fid_name in tasks:
        model.add_task(prop2id[prop_name], fid2id[fid_name])

    start = time.time()
    task_stats = _train_jointly(
        model,
        tasks,
        task_records,
        prop2id,
        fid2id,
        device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
    )
    result = _run_label_and_metrics(
        "phytca_joint_upper_bound",
        model,
        tasks,
        task_records,
        task_stats,
        prop2id,
        fid2id,
        batch_size,
        device,
        start,
    )
    result["incremental_params"] = 0
    return result


# ---------------------------------------------------------------------------
# D3: Sequential PhyTCA (structural isolation, no stability loss)
# ---------------------------------------------------------------------------


def d3_sequential_phytca(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    base_state_dict: dict[str, torch.Tensor] | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 3,
    mu: float = 0.0,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
) -> dict[str, Any]:
    """Sequential PhyTCA with exact retention via structural isolation.

    The ``mu`` argument is kept for API compatibility but is ignored: the new
    architecture guarantees zero forgetting by freezing old tasks, so a
    stability loss is no longer required.
    """
    _ = mu
    prop2id, fid2id = _name_to_id(tasks)
    model = ContinualCrystalModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=len(prop2id),
        n_fidelities=len(fid2id),
        adapter_name="single_child_tucker",
        adapter_rank=adapter_rank,
        n_layers=3,
        num_nearest_neighbors=num_nearest_neighbors,
    ).to(device)
    if base_state_dict is not None:
        _load_canonical_base(model, base_state_dict, device)

    start = time.time()
    task_stats: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    nmaes: list[list[float]] = []

    for t, (_, prop_name, fid_name) in enumerate(tasks):
        prop_id = prop2id[prop_name]
        fid_id = fid2id[fid_name]
        model.add_task(prop_id, fid_id)

        train_loader, val_loader, _, mean, std, mad = _make_loaders(task_records[t], batch_size)
        mean, std, mad = _train_single_task(
            model,
            train_loader,
            val_loader,
            prop_id,
            fid_id,
            device,
            epochs=epochs,
            lr=lr,
            patience=patience,
        )
        task_stats.append((mean, std, mad))

        nmaes.append(
            _evaluate_on_dev(
                model, tasks, task_records, task_stats, prop2id, fid2id, batch_size, device, t
            )
        )

        model.freeze_task(prop_id, fid_id)

    result = _run_label_and_metrics(
        "phytca_sequential",
        model,
        tasks,
        task_records,
        task_stats,
        prop2id,
        fid2id,
        batch_size,
        device,
        start,
        nmaes=nmaes,
    )
    if len(tasks) >= 1:
        _, last_prop, last_fid = tasks[-1]
        result["incremental_params"] = model.count_incremental_parameters(
            prop2id[last_prop], fid2id[last_fid]
        )
    else:
        result["incremental_params"] = 0
    return result


# ---------------------------------------------------------------------------
# D4/D5: Frozen OPT + correction branch
# ---------------------------------------------------------------------------


class FrozenOptCorrectionModel(nn.Module):
    """Freeze the OPT route and learn a small MBJ correction on top.

    For the affine variant two MLPs (alpha, beta) operate in **physical units**;
    for the residual variant ``models.PredictionResidualHead`` is used to
    de-normalize the parent prediction, add a physical residual, and
    re-normalize to the child (MBJ) space.  This avoids the cross-fidelity
    normalization bug described in ``反馈_2.md`` 5.1.
    """

    def __init__(
        self,
        base_model: ContinualCrystalModel,
        opt_prop_id: int,
        opt_fid_id: int,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.base = base_model
        self.opt_prop_id = opt_prop_id
        self.opt_fid_id = opt_fid_id
        self.affine = affine
        hidden_dim = base_model.hidden_dim

        # Freeze the entire base model.
        for p in self.base.parameters():
            p.requires_grad = False

        if affine:
            mlp = lambda: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.alpha = mlp()
            self.beta = mlp()
            # Initialize alpha near 1, beta near 0.
            for m in self.alpha.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
            self.alpha[-1].bias.data.fill_(1.0)
            for m in self.beta.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
        else:
            self.residual_head = PredictionResidualHead(hidden_dim)

    def set_normalizers(
        self,
        parent_mean: torch.Tensor,
        parent_std: torch.Tensor,
        child_mean: torch.Tensor,
        child_std: torch.Tensor,
    ) -> None:
        device = next(self.parameters()).device
        self.register_buffer("parent_mean", parent_mean.to(device))
        self.register_buffer("parent_std", parent_std.to(device))
        self.register_buffer("child_mean", child_mean.to(device))
        self.register_buffer("child_std", child_std.to(device))

    def forward(
        self,
        node_feats: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        original_mask: torch.Tensor,
        prop_id: int,
        fid_id: int,
    ) -> torch.Tensor:
        h = self.base.encode(
            node_feats, coords, mask, original_mask, self.opt_prop_id, self.opt_fid_id
        )
        key = self.base._task_key(self.opt_prop_id, self.opt_fid_id)
        y_opt = self.base.heads[key](h).squeeze(-1)
        if fid_id == self.opt_fid_id:
            return y_opt

        if self.affine:
            alpha = self.alpha(h).squeeze(-1)
            beta = self.beta(h).squeeze(-1)
            parent_pred_phys = y_opt * self.parent_std + self.parent_mean
            child_pred_phys = alpha * parent_pred_phys + beta
            return (child_pred_phys - self.child_mean) / self.child_std

        return self.residual_head(
            h, y_opt, self.parent_mean, self.parent_std, self.child_mean, self.child_std
        )


def _frozen_opt_correction(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    base_state_dict: dict[str, torch.Tensor] | None,
    affine: bool,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    adapter_rank: int,
    num_nearest_neighbors: int,
    opt_parent_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run frozen-OPT correction experiment from a shared OPT parent checkpoint."""
    prop2id, fid2id = _name_to_id(tasks)
    opt_prop_id = prop2id["band_gap"]
    opt_fid_id = fid2id["OptB88vdW"]
    mbj_fid_id = fid2id["TB-mBJ"]

    base_model = ContinualCrystalModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=len(prop2id),
        n_fidelities=len(fid2id),
        adapter_name="single_child_tucker",
        adapter_rank=adapter_rank,
        n_layers=3,
        num_nearest_neighbors=num_nearest_neighbors,
    ).to(device)

    start = time.time()

    if opt_parent_state is not None:
        base_model.add_task(opt_prop_id, opt_fid_id)
        base_model.load_state_dict(
            {k: v.to(device) for k, v in opt_parent_state["state_dict"].items()},
            strict=False,
        )
        mean, std, mad = opt_parent_state["mean"], opt_parent_state["std"], opt_parent_state["mad"]
        t1_after_t1 = opt_parent_state["task1_after_task1"]
    else:
        if base_state_dict is not None:
            _load_canonical_base(base_model, base_state_dict, device)
        base_model.add_task(opt_prop_id, opt_fid_id)
        train_loader, val_loader, _, mean, std, mad = _make_loaders(task_records[0], batch_size)
        _train_single_task(
            base_model,
            train_loader,
            val_loader,
            opt_prop_id,
            opt_fid_id,
            device,
            epochs=epochs,
            lr=lr,
            patience=patience,
        )
        dev_ds = JARVISCrystalDataset(task_records[0], split="continual_dev")
        dev_ds.target_mean = float(mean)
        dev_ds.target_std = float(std)
        dev_ds.normalize_target = True
        dev_loader = DataLoader(
            dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals
        )
        t1_after_t1 = evaluate_loader(
            base_model, dev_loader, opt_prop_id, opt_fid_id, mean, std, mad, device
        )

    task1_stats = (mean, std, mad)

    # Freeze encoder and OPT route.
    for p in base_model.parameters():
        p.requires_grad = False

    correction = FrozenOptCorrectionModel(
        base_model, opt_prop_id, opt_fid_id, affine=affine
    ).to(device)

    # Task 2: train correction branch only.
    train_loader2, val_loader2, _, mean2, std2, mad2 = _make_loaders(task_records[1], batch_size)
    task2_stats = (mean2, std2, mad2)
    correction.set_normalizers(mean, std, mean2, std2)
    task2_trainable_names = _trainable_param_names(correction)

    _train_single_task(
        correction,
        train_loader2,
        val_loader2,
        opt_prop_id,
        mbj_fid_id,
        device,
        epochs=epochs,
        lr=lr,
        patience=patience,
    )

    label = "frozen_opt_affine_correction" if affine else "frozen_opt_residual_correction"
    result = _run_label_and_metrics(
        label,
        correction,
        tasks,
        task_records,
        [task1_stats, task2_stats],
        prop2id,
        fid2id,
        batch_size,
        device,
        start,
    )
    # Override T1@T1 with the value from the shared OPT parent checkpoint.
    result["task1_after_task1"] = t1_after_t1
    result["absolute_forgetting"] = result["task1_after_task2"] - t1_after_t1
    result["bwt"] = backward_transfer([[t1_after_t1], result["nmaes"][1]])
    result["incremental_params"] = _count_trainable(correction)
    result["task2_trainable_param_names"] = task2_trainable_names
    return result


def d4_frozen_opt_affine(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    base_state_dict: dict[str, torch.Tensor] | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 3,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
    opt_parent_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _frozen_opt_correction(
        tasks,
        task_records,
        node_dim,
        hidden_dim,
        device,
        base_state_dict,
        affine=True,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        opt_parent_state=opt_parent_state,
    )


def d5_frozen_opt_residual(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    base_state_dict: dict[str, torch.Tensor] | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 3,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
    opt_parent_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _frozen_opt_correction(
        tasks,
        task_records,
        node_dim,
        hidden_dim,
        device,
        base_state_dict,
        affine=False,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        opt_parent_state=opt_parent_state,
    )


# ---------------------------------------------------------------------------
# Correction modules for residual baselines
# ---------------------------------------------------------------------------


class PhysicalResidualCorrection(nn.Module):
    """Wrap an arbitrary residual module so it operates in physical units."""

    def __init__(self, correction_module: nn.Module) -> None:
        super().__init__()
        self.correction_module = correction_module

    def forward(
        self,
        h: torch.Tensor,
        parent_pred_norm: torch.Tensor,
        parent_mean: torch.Tensor,
        parent_std: torch.Tensor,
        child_mean: torch.Tensor,
        child_std: torch.Tensor,
    ) -> torch.Tensor:
        parent_pred_phys = parent_pred_norm * parent_std + parent_mean
        residual_phys = self.correction_module(h).squeeze(-1)
        child_pred_phys = parent_pred_phys + residual_phys
        return (child_pred_phys - child_mean) / child_std


class FrozenOptResidualModel(nn.Module):
    """Freeze the OPT route and add a physical-residual correction for MBJ."""

    def __init__(
        self,
        base_model: ContinualCrystalModel,
        opt_prop_id: int,
        opt_fid_id: int,
        correction_module: nn.Module,
    ) -> None:
        super().__init__()
        self.base = base_model
        self.opt_prop_id = opt_prop_id
        self.opt_fid_id = opt_fid_id
        for p in self.base.parameters():
            p.requires_grad = False
        self.correction = PhysicalResidualCorrection(correction_module)

    def set_normalizers(
        self,
        parent_mean: torch.Tensor,
        parent_std: torch.Tensor,
        child_mean: torch.Tensor,
        child_std: torch.Tensor,
    ) -> None:
        device = next(self.parameters()).device
        self.register_buffer("parent_mean", parent_mean.to(device))
        self.register_buffer("parent_std", parent_std.to(device))
        self.register_buffer("child_mean", child_mean.to(device))
        self.register_buffer("child_std", child_std.to(device))

    def forward(
        self,
        node_feats: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        original_mask: torch.Tensor,
        prop_id: int,
        fid_id: int,
    ) -> torch.Tensor:
        h = self.base.encode(
            node_feats, coords, mask, original_mask, self.opt_prop_id, self.opt_fid_id
        )
        key = self.base._task_key(self.opt_prop_id, self.opt_fid_id)
        y_opt = self.base.heads[key](h).squeeze(-1)
        if fid_id == self.opt_fid_id:
            return y_opt
        return self.correction(
            h, y_opt, self.parent_mean, self.parent_std, self.child_mean, self.child_std
        )


class LowRankResidual(nn.Module):
    """Low-rank residual: delta(h) = h @ A @ B."""

    def __init__(self, hidden_dim: int, rank: int) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.randn(hidden_dim, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.A @ self.B


def _make_parameter_matched_mlp(hidden_dim: int, target_params: int) -> nn.Sequential:
    """Return a two-layer MLP whose trainable count is close to ``target_params``."""
    best = None
    best_diff = float("inf")
    for mid in range(1, hidden_dim + 1):
        params = (hidden_dim * mid + mid) + (mid * 1 + 1)
        diff = abs(params - target_params)
        if diff < best_diff:
            best_diff = diff
            best = mid
    assert best is not None
    return nn.Sequential(
        nn.Linear(hidden_dim, best),
        nn.SiLU(),
        nn.Linear(best, 1),
    )


def _frozen_opt_residual_experiment(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    base_state_dict: dict[str, torch.Tensor] | None,
    correction_module: nn.Module,
    label: str,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    adapter_rank: int,
    num_nearest_neighbors: int,
    opt_parent_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a physical-residual correction module on top of the frozen OPT route."""
    prop2id, fid2id = _name_to_id(tasks)
    opt_prop_id = prop2id["band_gap"]
    opt_fid_id = fid2id["OptB88vdW"]
    mbj_fid_id = fid2id["TB-mBJ"]

    base_model = ContinualCrystalModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=len(prop2id),
        n_fidelities=len(fid2id),
        adapter_name="single_child_tucker",
        adapter_rank=adapter_rank,
        n_layers=3,
        num_nearest_neighbors=num_nearest_neighbors,
    ).to(device)

    start = time.time()

    if opt_parent_state is not None:
        base_model.add_task(opt_prop_id, opt_fid_id)
        base_model.load_state_dict(
            {k: v.to(device) for k, v in opt_parent_state["state_dict"].items()},
            strict=False,
        )
        mean, std, mad = opt_parent_state["mean"], opt_parent_state["std"], opt_parent_state["mad"]
        t1_after_t1 = opt_parent_state["task1_after_task1"]
    else:
        if base_state_dict is not None:
            _load_canonical_base(base_model, base_state_dict, device)
        base_model.add_task(opt_prop_id, opt_fid_id)
        train_loader, val_loader, _, mean, std, mad = _make_loaders(task_records[0], batch_size)
        _train_single_task(
            base_model,
            train_loader,
            val_loader,
            opt_prop_id,
            opt_fid_id,
            device,
            epochs=epochs,
            lr=lr,
            patience=patience,
        )
        dev_ds = JARVISCrystalDataset(task_records[0], split="continual_dev")
        dev_ds.target_mean = float(mean)
        dev_ds.target_std = float(std)
        dev_ds.normalize_target = True
        dev_loader = DataLoader(
            dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals
        )
        t1_after_t1 = evaluate_loader(
            base_model, dev_loader, opt_prop_id, opt_fid_id, mean, std, mad, device
        )

    task1_stats = (mean, std, mad)

    for p in base_model.parameters():
        p.requires_grad = False

    correction_model = FrozenOptResidualModel(
        base_model, opt_prop_id, opt_fid_id, correction_module.to(device)
    ).to(device)
    task2_trainable_names = _trainable_param_names(correction_model)

    train_loader2, val_loader2, _, mean2, std2, mad2 = _make_loaders(task_records[1], batch_size)
    task2_stats = (mean2, std2, mad2)
    correction_model.set_normalizers(mean, std, mean2, std2)
    _train_single_task(
        correction_model,
        train_loader2,
        val_loader2,
        opt_prop_id,
        mbj_fid_id,
        device,
        epochs=epochs,
        lr=lr,
        patience=patience,
    )

    result = _run_label_and_metrics(
        label,
        correction_model,
        tasks,
        task_records,
        [task1_stats, task2_stats],
        prop2id,
        fid2id,
        batch_size,
        device,
        start,
    )
    result["task1_after_task1"] = t1_after_t1
    result["absolute_forgetting"] = result["task1_after_task2"] - t1_after_t1
    result["bwt"] = backward_transfer([[t1_after_t1], result["nmaes"][1]])
    result["incremental_params"] = _count_trainable(correction_model)
    result["task2_trainable_param_names"] = task2_trainable_names
    return result


def d6c_independent_low_rank_residual(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    base_state_dict: dict[str, torch.Tensor] | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 3,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
    opt_parent_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Frozen OPT parent + low-rank residual on the pooled representation."""
    correction = LowRankResidual(hidden_dim=hidden_dim, rank=adapter_rank)
    return _frozen_opt_residual_experiment(
        tasks,
        task_records,
        node_dim,
        hidden_dim,
        device,
        base_state_dict,
        correction=correction,
        label="fr_phytca_low_rank_residual",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        opt_parent_state=opt_parent_state,
    )


def d6d_parameter_matched_mlp_residual(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    base_state_dict: dict[str, torch.Tensor] | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 3,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
    opt_parent_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Frozen OPT parent + small residual MLP matched to FR-PhyTCA size."""
    prop2id, fid2id = _name_to_id(tasks)
    target_params = _fr_phytca_incremental_params(
        hidden_dim, adapter_rank, len(prop2id), len(fid2id), n_layers=3
    )
    correction = _make_parameter_matched_mlp(hidden_dim, target_params)
    return _frozen_opt_residual_experiment(
        tasks,
        task_records,
        node_dim,
        hidden_dim,
        device,
        base_state_dict,
        correction_module=correction,
        label="fr_phytca_param_matched_mlp",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        opt_parent_state=opt_parent_state,
    )


def _fr_phytca_incremental_params(
    hidden_dim: int,
    adapter_rank: int,
    n_properties: int,
    n_fidelities: int,
    n_layers: int = 3,
) -> int:
    """Incremental parameter count for FR-PhyTCA (single-child Tucker + head)."""
    _ = n_properties, n_fidelities
    per_layer = (
        hidden_dim * adapter_rank  # u_in
        + adapter_rank * adapter_rank  # core
        + hidden_dim * adapter_rank  # u_out
    )
    return n_layers * per_layer + (hidden_dim + 1)  # new head


def _find_low_rank_for_target(hidden_dim: int, target_params: int) -> int:
    """Return rank so that LowRankResidual params are within 5% of target."""
    denom = hidden_dim + 1
    ideal = target_params / denom
    candidates = [max(1, int(ideal)), max(1, int(ideal) + 1)]
    best = min(candidates, key=lambda r: abs(r * denom - target_params))
    assert abs(best * denom - target_params) / target_params <= 0.05
    return best


def _find_mlp_bottleneck_for_target(hidden_dim: int, target_params: int) -> int:
    """Return bottleneck so that two-layer MLP params are within 5% of target."""
    denom = hidden_dim + 2
    ideal = (target_params - 1) / denom
    candidates = [max(1, int(ideal)), max(1, int(ideal) + 1)]
    best = min(candidates, key=lambda b: abs(b * denom + 1 - target_params))
    assert abs(best * denom + 1 - target_params) / target_params <= 0.05
    return best


def d6g_matched_low_rank_residual(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    base_state_dict: dict[str, torch.Tensor] | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 3,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
    opt_parent_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Frozen OPT parent + low-rank residual with parameter count matched to FR-PhyTCA."""
    prop2id, fid2id = _name_to_id(tasks)
    target = _fr_phytca_incremental_params(
        hidden_dim, adapter_rank, len(prop2id), len(fid2id), n_layers=3
    )
    rank = _find_low_rank_for_target(hidden_dim, target)
    correction = LowRankResidual(hidden_dim=hidden_dim, rank=rank)
    result = _frozen_opt_residual_experiment(
        tasks,
        task_records,
        node_dim,
        hidden_dim,
        device,
        base_state_dict,
        correction=correction,
        label="matched_low_rank_residual",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        opt_parent_state=opt_parent_state,
    )
    result["matched_target_params"] = target
    result["matched_actual_params"] = _count_params(correction)
    return result


def d6h_matched_mlp_residual(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    base_state_dict: dict[str, torch.Tensor] | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 3,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
    opt_parent_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Frozen OPT parent + MLP residual with parameter count matched to FR-PhyTCA."""
    prop2id, fid2id = _name_to_id(tasks)
    target = _fr_phytca_incremental_params(
        hidden_dim, adapter_rank, len(prop2id), len(fid2id), n_layers=3
    )
    bottleneck = _find_mlp_bottleneck_for_target(hidden_dim, target)
    correction = nn.Sequential(
        nn.Linear(hidden_dim, bottleneck),
        nn.SiLU(),
        nn.Linear(bottleneck, 1),
    )
    result = _frozen_opt_residual_experiment(
        tasks,
        task_records,
        node_dim,
        hidden_dim,
        device,
        base_state_dict,
        correction_module=correction,
        label="matched_mlp_residual",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        opt_parent_state=opt_parent_state,
    )
    result["matched_target_params"] = target
    result["matched_actual_params"] = _count_params(correction)
    return result


# ---------------------------------------------------------------------------
# D6: Progressive Tucker residual (now structural isolation / single-child Tucker)
# ---------------------------------------------------------------------------


def _snapshot_opt_predictions(
    model: ContinualCrystalModel,
    task_records: list[list[dict]],
    opt_prop_id: int,
    opt_fid_id: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return normalized OPT predictions on the continual_dev split."""
    dev_ds = JARVISCrystalDataset(task_records[0], split="continual_dev")
    dev_ds.target_mean = float(mean)
    dev_ds.target_std = float(std)
    dev_ds.normalize_target = True
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals
    )
    preds: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for node_feats, coords, mask, original_mask, _ in dev_loader:
            node_feats = node_feats.to(device)
            coords = coords.to(device)
            mask = mask.to(device)
            original_mask = original_mask.to(device)
            pred = model(node_feats, coords, mask, original_mask, opt_prop_id, opt_fid_id)
            preds.append(pred.detach().cpu())
    return torch.cat(preds)


def d6_progressive_tucker(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    base_state_dict: dict[str, torch.Tensor] | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 3,
    lambda_distill: float = 0.0,
    orthogonal_child: bool = False,
    shared_factors: bool = False,
    unfreeze_top_layer: bool = False,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
    opt_parent_state: dict[str, Any] | None = None,
    experiment_label: str | None = None,
    adapter_name: str = "single_child_tucker",
) -> dict[str, Any]:
    """FR-PhyTCA via exact-retention structural isolation.

    The old progressive Tucker parent/child construction is replaced by
    ``ContinualCrystalModel``: each task has a private adapter bank, and
    freezing the old bank guarantees zero drift of the OPT route.

    The ``orthogonal_child``, ``shared_factors``, and ``unfreeze_top_layer``
    flags are kept for API compatibility but are no-ops in the new design:
    structural isolation already enforces the invariant they were meant to
    approximate.  ``lambda_distill`` is also ignored because the parent route
    is physically frozen, so distillation is redundant.
    """
    _ = lambda_distill, orthogonal_child, shared_factors, unfreeze_top_layer
    prop2id, fid2id = _name_to_id(tasks)
    opt_prop_id = prop2id["band_gap"]
    opt_fid_id = fid2id["OptB88vdW"]
    mbj_fid_id = fid2id["TB-mBJ"]

    model = ContinualCrystalModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=len(prop2id),
        n_fidelities=len(fid2id),
        adapter_name=adapter_name,
        adapter_rank=adapter_rank,
        n_layers=3,
        num_nearest_neighbors=num_nearest_neighbors,
    ).to(device)

    if base_state_dict is not None:
        _load_canonical_base(model, base_state_dict, device)

    start = time.time()

    if opt_parent_state is not None:
        model.add_task(opt_prop_id, opt_fid_id)
        model.load_state_dict(
            {k: v.to(device) for k, v in opt_parent_state["state_dict"].items()},
            strict=False,
        )
        mean, std, mad = opt_parent_state["mean"], opt_parent_state["std"], opt_parent_state["mad"]
        t1_after_t1 = opt_parent_state["task1_after_task1"]
    else:
        model.add_task(opt_prop_id, opt_fid_id)
        train_loader, val_loader, _, mean, std, mad = _make_loaders(task_records[0], batch_size)
        _train_single_task(
            model,
            train_loader,
            val_loader,
            opt_prop_id,
            opt_fid_id,
            device,
            epochs=epochs,
            lr=lr,
            patience=patience,
        )
        dev_ds = JARVISCrystalDataset(task_records[0], split="continual_dev")
        dev_ds.target_mean = float(mean)
        dev_ds.target_std = float(std)
        dev_ds.normalize_target = True
        dev_loader = DataLoader(
            dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals
        )
        t1_after_t1 = evaluate_loader(
            model, dev_loader, opt_prop_id, opt_fid_id, mean, std, mad, device
        )

    task1_stats = (mean, std, mad)

    # Exact-retention gate: snapshot OPT predictions before freezing/adding MBJ.
    opt_preds_before = _snapshot_opt_predictions(
        model, task_records, opt_prop_id, opt_fid_id, mean, std, batch_size, device
    )

    # Freeze the OPT task; its adapter bank and head are excluded from all
    # future optimizers by structural isolation.
    model.freeze_task(opt_prop_id, opt_fid_id)

    # Task 2: add a fresh MBJ adapter bank + head and train it.
    model.add_task(opt_prop_id, mbj_fid_id)
    train_loader2, val_loader2, _, mean2, std2, mad2 = _make_loaders(task_records[1], batch_size)
    task2_stats = (mean2, std2, mad2)
    _train_single_task(
        model,
        train_loader2,
        val_loader2,
        opt_prop_id,
        mbj_fid_id,
        device,
        epochs=epochs,
        lr=lr,
        patience=patience,
    )

    # Exact-retention gate: OPT predictions must be identical after Task 2.
    opt_preds_after = _snapshot_opt_predictions(
        model, task_records, opt_prop_id, opt_fid_id, mean, std, batch_size, device
    )
    opt_route_drift = float((opt_preds_after - opt_preds_before).abs().max())

    nmaes = [
        _evaluate_on_dev(
            model, tasks, task_records, [task1_stats], prop2id, fid2id, batch_size, device, 0
        ),
        _evaluate_on_dev(
            model,
            tasks,
            task_records,
            [task1_stats, task2_stats],
            prop2id,
            fid2id,
            batch_size,
            device,
            1,
        ),
    ]

    label = experiment_label or "fr_phytca"
    result = _run_label_and_metrics(
        label, model, tasks, task_records, [task1_stats, task2_stats], prop2id, fid2id, batch_size, device, start, nmaes=nmaes
    )
    result["task1_after_task1"] = t1_after_t1
    result["absolute_forgetting"] = result["task1_after_task2"] - t1_after_t1
    result["bwt"] = backward_transfer([[t1_after_t1], result["nmaes"][1]])
    result["incremental_params"] = model.count_incremental_parameters(opt_prop_id, mbj_fid_id)
    result["opt_route_drift"] = opt_route_drift
    return result


def d6e_orthogonal_tucker_residual(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    base_state_dict: dict[str, torch.Tensor] | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 3,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
    opt_parent_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """FR-PhyTCA variant with the orthogonal-child label (now a structural alias)."""
    return d6_progressive_tucker(
        tasks=tasks,
        task_records=task_records,
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        device=device,
        base_state_dict=base_state_dict,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        opt_parent_state=opt_parent_state,
        experiment_label="fr_phytca_orthogonal",
    )


def d6f_shared_factor_top_layer(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    base_state_dict: dict[str, torch.Tensor] | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 3,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
    opt_parent_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """FR-PhyTCA variant with the shared-factor-top-layer label (structural alias)."""
    return d6_progressive_tucker(
        tasks=tasks,
        task_records=task_records,
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        device=device,
        base_state_dict=base_state_dict,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        opt_parent_state=opt_parent_state,
        experiment_label="fr_phytca_shared_factor_top_layer",
    )


# ---------------------------------------------------------------------------
# D6i-k: architecture-matched adapter baselines
# ---------------------------------------------------------------------------


def d6i_lora_ab(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    base_state_dict: dict[str, torch.Tensor] | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 3,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
    opt_parent_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """FR-PhyTCA-style training with ``adapter_name='lora_ab'``."""
    return d6_progressive_tucker(
        tasks=tasks,
        task_records=task_records,
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        device=device,
        base_state_dict=base_state_dict,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        opt_parent_state=opt_parent_state,
        experiment_label="fr_phytca_lora_ab",
        adapter_name="lora_ab",
    )


def d6j_lora_aba(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    base_state_dict: dict[str, torch.Tensor] | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 3,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
    opt_parent_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """FR-PhyTCA-style training with ``adapter_name='lora_aba'``."""
    return d6_progressive_tucker(
        tasks=tasks,
        task_records=task_records,
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        device=device,
        base_state_dict=base_state_dict,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        opt_parent_state=opt_parent_state,
        experiment_label="fr_phytca_lora_aba",
        adapter_name="lora_aba",
    )


def d6k_multi_axis_tucker(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    base_state_dict: dict[str, torch.Tensor] | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 3,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
    opt_parent_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """FR-PhyTCA-style training with ``adapter_name='multi_axis_tucker'``."""
    return d6_progressive_tucker(
        tasks=tasks,
        task_records=task_records,
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        device=device,
        base_state_dict=base_state_dict,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        opt_parent_state=opt_parent_state,
        experiment_label="fr_phytca_multi_axis_tucker",
        adapter_name="multi_axis_tucker",
    )


# ---------------------------------------------------------------------------
# Additional baselines for scaling study
# ---------------------------------------------------------------------------


class FeatureTransferModel(nn.Module):
    """Freeze OPT route and train an MBJ head on concatenated OPT+MBJ representations."""

    def __init__(
        self,
        base_model: ContinualCrystalModel,
        opt_prop_id: int,
        opt_fid_id: int,
        mbj_fid_id: int,
    ) -> None:
        super().__init__()
        self.base = base_model
        self.opt_prop_id = opt_prop_id
        self.opt_fid_id = opt_fid_id
        self.mbj_fid_id = mbj_fid_id
        hidden_dim = base_model.hidden_dim
        self.mbj_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(
        self,
        node_feats: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        original_mask: torch.Tensor,
        prop_id: int,
        fid_id: int,
    ) -> torch.Tensor:
        h_opt = self.base.encode(
            node_feats, coords, mask, original_mask, prop_id, self.opt_fid_id
        )
        if fid_id == self.opt_fid_id:
            key = self.base._task_key(prop_id, self.opt_fid_id)
            return self.base.heads[key](h_opt).squeeze(-1)
        h_mbj = self.base.encode(
            node_feats, coords, mask, original_mask, prop_id, self.mbj_fid_id
        )
        return self.mbj_head(torch.cat([h_mbj, h_opt.detach()], dim=-1)).squeeze(-1)


def feature_transfer_experiment(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    base_state_dict: dict[str, torch.Tensor] | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 3,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
    opt_parent_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Frozen OPT parent + trainable MBJ head on concatenated representations."""
    prop2id, fid2id = _name_to_id(tasks)
    opt_prop_id = prop2id["band_gap"]
    opt_fid_id = fid2id["OptB88vdW"]
    mbj_fid_id = fid2id["TB-mBJ"]

    base_model = ContinualCrystalModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=len(prop2id),
        n_fidelities=len(fid2id),
        adapter_name="single_child_tucker",
        adapter_rank=adapter_rank,
        n_layers=3,
        num_nearest_neighbors=num_nearest_neighbors,
    ).to(device)

    start = time.time()

    if opt_parent_state is not None:
        base_model.add_task(opt_prop_id, opt_fid_id)
        base_model.load_state_dict(
            {k: v.to(device) for k, v in opt_parent_state["state_dict"].items()},
            strict=False,
        )
        mean, std, mad = opt_parent_state["mean"], opt_parent_state["std"], opt_parent_state["mad"]
        t1_after_t1 = opt_parent_state["task1_after_task1"]
    else:
        if base_state_dict is not None:
            _load_canonical_base(base_model, base_state_dict, device)
        base_model.add_task(opt_prop_id, opt_fid_id)
        train_loader, val_loader, _, mean, std, mad = _make_loaders(task_records[0], batch_size)
        _train_single_task(
            base_model,
            train_loader,
            val_loader,
            opt_prop_id,
            opt_fid_id,
            device,
            epochs=epochs,
            lr=lr,
            patience=patience,
        )
        dev_ds = JARVISCrystalDataset(task_records[0], split="continual_dev")
        dev_ds.target_mean = float(mean)
        dev_ds.target_std = float(std)
        dev_ds.normalize_target = True
        dev_loader = DataLoader(
            dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals
        )
        t1_after_t1 = evaluate_loader(
            base_model, dev_loader, opt_prop_id, opt_fid_id, mean, std, mad, device
        )

    task1_stats = (mean, std, mad)

    for p in base_model.parameters():
        p.requires_grad = False

    # Add an untrained MBJ route so its representation can be used by the head.
    base_model.add_task(opt_prop_id, mbj_fid_id)

    ft_model = FeatureTransferModel(
        base_model, opt_prop_id, opt_fid_id, mbj_fid_id
    ).to(device)
    task2_trainable_names = _trainable_param_names(ft_model)

    train_loader2, val_loader2, _, mean2, std2, mad2 = _make_loaders(task_records[1], batch_size)
    task2_stats = (mean2, std2, mad2)
    _train_single_task(
        ft_model,
        train_loader2,
        val_loader2,
        opt_prop_id,
        mbj_fid_id,
        device,
        epochs=epochs,
        lr=lr,
        patience=patience,
    )

    result = _run_label_and_metrics(
        "feature_transfer",
        ft_model,
        tasks,
        task_records,
        [task1_stats, task2_stats],
        prop2id,
        fid2id,
        batch_size,
        device,
        start,
    )
    result["task1_after_task1"] = t1_after_t1
    result["absolute_forgetting"] = result["task1_after_task2"] - t1_after_t1
    result["bwt"] = backward_transfer([[t1_after_t1], result["nmaes"][1]])
    result["incremental_params"] = _count_trainable(ft_model)
    result["task2_trainable_param_names"] = task2_trainable_names
    return result


def mbj_only_training(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    base_state_dict: dict[str, torch.Tensor] | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 3,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
) -> dict[str, Any]:
    """Train only on MBJ; OPT predictions come from an untrained route."""
    prop2id, fid2id = _name_to_id(tasks)
    opt_prop_id = prop2id["band_gap"]
    opt_fid_id = fid2id["OptB88vdW"]
    mbj_fid_id = fid2id["TB-mBJ"]

    model = ContinualCrystalModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=len(prop2id),
        n_fidelities=len(fid2id),
        adapter_name="single_child_tucker",
        adapter_rank=adapter_rank,
        n_layers=3,
        num_nearest_neighbors=num_nearest_neighbors,
    ).to(device)
    if base_state_dict is not None:
        _load_canonical_base(model, base_state_dict, device)

    start = time.time()
    # Add an untrained OPT route and the MBJ task to be trained.
    model.add_task(opt_prop_id, opt_fid_id)
    model.add_task(opt_prop_id, mbj_fid_id)

    train_loader, val_loader, _, mean, std, mad = _make_loaders(task_records[1], batch_size)
    _train_single_task(
        model,
        train_loader,
        val_loader,
        opt_prop_id,
        mbj_fid_id,
        device,
        epochs=epochs,
        lr=lr,
        patience=patience,
    )

    task_stats = [(mean, std, mad), (mean, std, mad)]
    result = _run_label_and_metrics(
        "mbj_only",
        model,
        tasks,
        task_records,
        task_stats,
        prop2id,
        fid2id,
        batch_size,
        device,
        start,
    )
    result["incremental_params"] = result["trainable_params"]
    return result


def opt_pretrain_mbj_full_finetune(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    base_state_dict: dict[str, torch.Tensor] | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 3,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
    opt_parent_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pre-train on OPT, then unfreeze everything and fine-tune on MBJ."""
    prop2id, fid2id = _name_to_id(tasks)
    opt_prop_id = prop2id["band_gap"]
    opt_fid_id = fid2id["OptB88vdW"]
    mbj_fid_id = fid2id["TB-mBJ"]

    model = ContinualCrystalModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=len(prop2id),
        n_fidelities=len(fid2id),
        adapter_name="single_child_tucker",
        adapter_rank=adapter_rank,
        n_layers=3,
        num_nearest_neighbors=num_nearest_neighbors,
    ).to(device)

    start = time.time()

    if opt_parent_state is not None:
        model.add_task(opt_prop_id, opt_fid_id)
        model.load_state_dict(
            {k: v.to(device) for k, v in opt_parent_state["state_dict"].items()},
            strict=False,
        )
        mean, std, mad = opt_parent_state["mean"], opt_parent_state["std"], opt_parent_state["mad"]
        t1_after_t1 = opt_parent_state["task1_after_task1"]
    else:
        if base_state_dict is not None:
            _load_canonical_base(model, base_state_dict, device)
        model.add_task(opt_prop_id, opt_fid_id)
        train_loader, val_loader, _, mean, std, mad = _make_loaders(task_records[0], batch_size)
        _train_single_task(
            model,
            train_loader,
            val_loader,
            opt_prop_id,
            opt_fid_id,
            device,
            epochs=epochs,
            lr=lr,
            patience=patience,
        )
        dev_ds = JARVISCrystalDataset(task_records[0], split="continual_dev")
        dev_ds.target_mean = float(mean)
        dev_ds.target_std = float(std)
        dev_ds.normalize_target = True
        dev_loader = DataLoader(
            dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals
        )
        t1_after_t1 = evaluate_loader(
            model, dev_loader, opt_prop_id, opt_fid_id, mean, std, mad, device
        )

    task1_stats = (mean, std, mad)

    # Unfreeze everything for full MBJ fine-tuning (demonstrates forgetting).
    for p in model.parameters():
        p.requires_grad = True

    model.add_task(opt_prop_id, mbj_fid_id)
    train_loader2, val_loader2, _, mean2, std2, mad2 = _make_loaders(task_records[1], batch_size)
    task2_stats = (mean2, std2, mad2)
    _train_single_task(
        model,
        train_loader2,
        val_loader2,
        opt_prop_id,
        mbj_fid_id,
        device,
        epochs=epochs,
        lr=lr,
        patience=patience,
    )

    result = _run_label_and_metrics(
        "opt_pretrain_mbj_full_finetune",
        model,
        tasks,
        task_records,
        [task1_stats, task2_stats],
        prop2id,
        fid2id,
        batch_size,
        device,
        start,
    )
    result["task1_after_task1"] = t1_after_t1
    result["absolute_forgetting"] = result["task1_after_task2"] - t1_after_t1
    result["bwt"] = backward_transfer([[t1_after_t1], result["nmaes"][1]])
    result["incremental_params"] = result["trainable_params"]
    return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


DIAGNOSTIC_REGISTRY: dict[str, Callable] = {
    "full_joint_upper_bound": d1_full_joint,
    "phytca_joint_upper_bound": d2_joint_phytca,
    "phytca_sequential": d3_sequential_phytca,
    "frozen_opt_affine_correction": d4_frozen_opt_affine,
    "frozen_opt_residual_correction": d5_frozen_opt_residual,
    "fr_phytca": d6_progressive_tucker,
    "fr_phytca_orthogonal": d6e_orthogonal_tucker_residual,
    "fr_phytca_shared_factor_top_layer": d6f_shared_factor_top_layer,
    "fr_phytca_low_rank_residual": d6c_independent_low_rank_residual,
    "fr_phytca_param_matched_mlp": d6d_parameter_matched_mlp_residual,
    "matched_low_rank_residual": d6g_matched_low_rank_residual,
    "matched_mlp_residual": d6h_matched_mlp_residual,
    "fr_phytca_lora_ab": d6i_lora_ab,
    "fr_phytca_lora_aba": d6j_lora_aba,
    "fr_phytca_multi_axis_tucker": d6k_multi_axis_tucker,
    "feature_transfer": feature_transfer_experiment,
    "mbj_only": mbj_only_training,
    "opt_pretrain_mbj_full_finetune": opt_pretrain_mbj_full_finetune,
}
