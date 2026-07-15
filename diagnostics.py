"""Diagnostic model variants and experiments for PhyTCA redesign.

This module implements the six 2k/seed-42 diagnostic experiments requested for
Protocol B: full joint, joint PhyTCA, sequential PhyTCA, frozen-OPT correction
(affine and residual), and progressive Tucker residual with optional OPT
distillation.  All variants start from the same canonical frozen-encoder
checkpoint where applicable, and all metrics are reported on the held-out
``continual_dev`` split.
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

from baselines import _evaluate_all_seen, _make_loaders, _train_one_task
from data import JARVISCrystalDataset, collate_crystals
from phytca import (
    AdapterCrystalGraphLayer,
    PhyTCAModel,
    Tucker4DAdapter,
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
    """Load a canonical frozen-encoder state into a model on ``device``."""
    missing, unexpected = model.load_state_dict(copy.deepcopy(base_state_dict), strict=False)
    if missing or unexpected:
        print(f"  Warning: canonical base load missing={missing} unexpected={unexpected}")


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
            val_nmaes.append(evaluate_loader(model, val_loader, prop_id, fid_id, mean, std, mad, device))
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
        loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals)
        nmaes.append(evaluate_loader(model, loader, pid, fid, mean_p, std_p, mad_p, device))
    return nmaes


def _count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _count_total(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _trainable_param_names(model: nn.Module) -> list[str]:
    """Return names of parameters that currently require gradients."""
    return [name for name, p in model.named_parameters() if p.requires_grad]


def _project_orthogonal(child: torch.Tensor, parent: torch.Tensor) -> torch.Tensor:
    """Return ``child`` columns projected onto the complement of ``parent`` columns."""
    with torch.no_grad():
        q_parent, _ = torch.linalg.qr(parent)
        residual = child - q_parent @ (q_parent.T @ child)
        q_child, _ = torch.linalg.qr(residual)
        # QR may drop columns if residual is rank-deficient; pad to keep rank.
        if q_child.shape[1] < child.shape[1]:
            need = child.shape[1] - q_child.shape[1]
            pad = torch.randn(child.shape[0], need, device=child.device, dtype=child.dtype)
            pad = pad - q_parent @ (q_parent.T @ pad) - q_child @ (q_child.T @ pad)
            q_pad, _ = torch.linalg.qr(pad)
            q_child = torch.cat([q_child, q_pad[:, :need]], dim=1)
        return q_child[:, : child.shape[1]]


def _orthogonalize_child_factors(model: nn.Module) -> None:
    """Project each child Tucker factor to be orthogonal to its parent factor."""
    for layer in model.layers:
        child = layer.adapter.child
        parent = layer.adapter.parent
        child.U_in.data = _project_orthogonal(child.U_in.data, parent.U_in.data)
        child.U_out.data = _project_orthogonal(child.U_out.data, parent.U_out.data)


def _tie_child_factors_to_parent(model: nn.Module) -> None:
    """Share ``U_in``/``U_out`` between child and parent (parent remains frozen)."""
    for layer in model.layers:
        layer.adapter.child.U_in = layer.adapter.parent.U_in
        layer.adapter.child.U_out = layer.adapter.parent.U_out


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

    The bundle contains the trained model state dict, normalizer statistics,
    T1@T1 nMAE on continual_dev, and hashes for audit comparisons across D4-D6.
    """
    prop2id, fid2id = _name_to_id(tasks)
    opt_prop_id = prop2id["band_gap"]
    opt_fid_id = fid2id["OptB88vdW"]

    model = PhyTCAModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=len(prop2id),
        n_fidelities=len(fid2id),
        n_layers=3,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        freeze_encoder_weights=True,
    ).to(device)
    if base_state_dict is not None:
        _load_canonical_base(model, base_state_dict, device)

    train_loader, val_loader, _, mean, std, mad = _make_loaders(task_records[0], batch_size)
    _train_single_task(
        model, train_loader, val_loader, opt_prop_id, opt_fid_id, device,
        epochs=epochs, lr=lr, patience=patience,
    )

    # T1@T1 on continual_dev.
    dev_ds = JARVISCrystalDataset(task_records[0], split="continual_dev")
    dev_ds.target_mean = float(mean)
    dev_ds.target_std = float(std)
    dev_ds.normalize_target = True
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals)
    t1_after_t1 = evaluate_loader(model, dev_loader, opt_prop_id, opt_fid_id, mean, std, mad, device)
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
    """Common metric extraction after a two-task continual run.

    For sequential methods ``nmaes`` can be precomputed after each task so that
    T1@T1 reflects the model right after Task 1 rather than after Task 2.
    """
    if nmaes is None:
        nmaes = []
        for t in range(len(tasks)):
            nmaes.append(_evaluate_on_dev(model, tasks, task_records, task_stats, prop2id, fid2id, batch_size, device, t))

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
    """Joint training with the full model fine-tuned (285k trainable)."""
    prop2id, fid2id = _name_to_id(tasks)
    model = PhyTCAModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=len(prop2id),
        n_fidelities=len(fid2id),
        n_layers=3,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        freeze_encoder_weights=False,
    ).to(device)

    start = time.time()
    task_stats = _train_jointly(
        model, tasks, task_records, prop2id, fid2id, device,
        epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
    )
    result = _run_label_and_metrics(
        "full_joint_upper_bound", model, tasks, task_records, task_stats,
        prop2id, fid2id, batch_size, device, start,
    )
    result["incremental_params"] = 0
    return result


# ---------------------------------------------------------------------------
# D2: Joint PhyTCA (frozen backbone + Tucker adapter)
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
    """Joint training on OPT and MBJ with only adapter/head trainable."""
    prop2id, fid2id = _name_to_id(tasks)
    model = PhyTCAModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=len(prop2id),
        n_fidelities=len(fid2id),
        n_layers=3,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        freeze_encoder_weights=True,
    ).to(device)
    if base_state_dict is not None:
        _load_canonical_base(model, base_state_dict, device)

    start = time.time()
    task_stats = _train_jointly(
        model, tasks, task_records, prop2id, fid2id, device,
        epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
    )
    result = _run_label_and_metrics(
        "phytca_joint_upper_bound", model, tasks, task_records, task_stats,
        prop2id, fid2id, batch_size, device, start,
    )
    result["incremental_params"] = 0
    return result


# ---------------------------------------------------------------------------
# D3: Sequential PhyTCA (existing behavior)
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
    mu: float = 0.01,
    adapter_rank: int = 8,
    num_nearest_neighbors: int = 8,
) -> dict[str, Any]:
    """Sequential PhyTCA with stability loss."""
    prop2id, fid2id = _name_to_id(tasks)
    model = PhyTCAModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=len(prop2id),
        n_fidelities=len(fid2id),
        n_layers=3,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        freeze_encoder_weights=True,
    ).to(device)
    if base_state_dict is not None:
        _load_canonical_base(model, base_state_dict, device)

    start = time.time()
    task_stats: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    nmaes: list[list[float]] = []
    anchor: dict = {}
    for t, (_, prop_name, fid_name) in enumerate(tasks):
        prop_id = prop2id[prop_name]
        fid_id = fid2id[fid_name]
        train_loader, val_loader, _, mean, std, mad = _make_loaders(task_records[t], batch_size)
        task_stats.append((mean, std, mad))

        def extra_loss_fn(m: nn.Module) -> torch.Tensor:
            return m.stability_loss(mu, anchor)

        mean, std, mad = _train_single_task(
            model, train_loader, val_loader, prop_id, fid_id, device,
            epochs=epochs, lr=lr, patience=patience, extra_loss_fn=extra_loss_fn,
        )
        task_stats[-1] = (mean, std, mad)

        # Evaluate on all seen tasks using the current model state.
        nmaes.append(_evaluate_on_dev(model, tasks, task_records, task_stats, prop2id, fid2id, batch_size, device, t))

        model.freeze_task(prop_id, fid_id)
        anchor = model.anchor_state()

    result = _run_label_and_metrics(
        "phytca_sequential", model, tasks, task_records, task_stats,
        prop2id, fid2id, batch_size, device, start, nmaes=nmaes,
    )
    # After Task 1 the trainable count is the Task-1 adapter+head.
    # Incremental for Task 2 is the MBJ head + its slice of the adapter.
    # Approximate using the model's per-task adapter parameter count.
    result["incremental_params"] = int(model.get_parameter_group_counts()["heads"] / len(tasks)) + 100  # rough
    return result


# ---------------------------------------------------------------------------
# D4/D5: Frozen OPT + correction branch
# ---------------------------------------------------------------------------


class FrozenOptCorrectionModel(nn.Module):
    """Freeze OPT route and learn a small MBJ correction on top of OPT predictions.

    The forward pass returns the OPT prediction for the OPT fidelity.  For the
    MBJ fidelity it returns either
        y_mbj = alpha(h) * y_opt.detach() + beta(h)      (affine)
    or
        y_mbj = y_opt.detach() + delta(h)                (residual)
    where ``h`` is the pooled representation from the frozen OPT route.
    """

    def __init__(
        self,
        base_model: PhyTCAModel,
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

        # Small correction MLP.
        mlp = lambda: nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        if affine:
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
            self.delta = mlp()
            for m in self.delta.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)

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
            node_feats, coords, mask, original_mask, prop_id, self.opt_fid_id
        )
        y_opt = self.base.heads[f"p{prop_id}_f{self.opt_fid_id}"](h).squeeze(-1)
        if fid_id == self.opt_fid_id:
            return y_opt
        if self.affine:
            alpha = self.alpha(h).squeeze(-1)
            beta = self.beta(h).squeeze(-1)
            return alpha * y_opt.detach() + beta
        return y_opt.detach() + self.delta(h).squeeze(-1)


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

    base_model = PhyTCAModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=len(prop2id),
        n_fidelities=len(fid2id),
        n_layers=3,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        freeze_encoder_weights=True,
    ).to(device)

    start = time.time()

    if opt_parent_state is not None:
        base_model.load_state_dict({k: v.to(device) for k, v in opt_parent_state["state_dict"].items()})
        mean, std, mad = opt_parent_state["mean"], opt_parent_state["std"], opt_parent_state["mad"]
        t1_after_t1 = opt_parent_state["task1_after_task1"]
    else:
        if base_state_dict is not None:
            _load_canonical_base(base_model, base_state_dict, device)
        train_loader, val_loader, _, mean, std, mad = _make_loaders(task_records[0], batch_size)
        _train_single_task(
            base_model, train_loader, val_loader, opt_prop_id, opt_fid_id, device,
            epochs=epochs, lr=lr, patience=patience,
        )
        dev_ds = JARVISCrystalDataset(task_records[0], split="continual_dev")
        dev_ds.target_mean = float(mean)
        dev_ds.target_std = float(std)
        dev_ds.normalize_target = True
        dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals)
        t1_after_t1 = evaluate_loader(base_model, dev_loader, opt_prop_id, opt_fid_id, mean, std, mad, device)

    task1_stats = (mean, std, mad)

    # Freeze encoder and OPT head.
    for p in base_model.parameters():
        p.requires_grad = False

    correction = FrozenOptCorrectionModel(
        base_model, opt_prop_id, opt_fid_id, affine=affine
    ).to(device)
    task2_trainable_names = _trainable_param_names(correction)

    # Task 2: train correction branch only.
    train_loader2, val_loader2, _, mean2, std2, mad2 = _make_loaders(task_records[1], batch_size)
    task2_stats = (mean2, std2, mad2)
    _train_single_task(
        correction, train_loader2, val_loader2, opt_prop_id, mbj_fid_id, device,
        epochs=epochs, lr=lr, patience=patience,
    )

    task_stats = [task1_stats, task2_stats]
    label = "frozen_opt_affine_correction" if affine else "frozen_opt_residual_correction"
    result = _run_label_and_metrics(
        label, correction, tasks, task_records, task_stats,
        prop2id, fid2id, batch_size, device, start,
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
        tasks, task_records, node_dim, hidden_dim, device, base_state_dict,
        affine=True, epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
        adapter_rank=adapter_rank, num_nearest_neighbors=num_nearest_neighbors,
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
        tasks, task_records, node_dim, hidden_dim, device, base_state_dict,
        affine=False, epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
        adapter_rank=adapter_rank, num_nearest_neighbors=num_nearest_neighbors,
        opt_parent_state=opt_parent_state,
    )


# ---------------------------------------------------------------------------
# D6 ablations: corrections on the frozen MBJ parent representation
# ---------------------------------------------------------------------------


class FrozenMbjCorrectionModel(nn.Module):
    """Use the frozen MBJ parent representation and apply a trainable correction.

    The forward returns the parent OPT prediction for OPT fidelity.  For MBJ it
    returns ``y_mbj_parent + correction(h_mbj_parent)``.
    """

    def __init__(
        self,
        base_model: PhyTCAModel,
        opt_prop_id: int,
        opt_fid_id: int,
        mbj_fid_id: int,
        correction: nn.Module,
    ) -> None:
        super().__init__()
        self.base = base_model
        self.opt_prop_id = opt_prop_id
        self.opt_fid_id = opt_fid_id
        self.mbj_fid_id = mbj_fid_id
        self.correction = correction

    def forward(
        self,
        node_feats: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        original_mask: torch.Tensor,
        prop_id: int,
        fid_id: int,
    ) -> torch.Tensor:
        if fid_id == self.opt_fid_id:
            h = self.base.encode(node_feats, coords, mask, original_mask, prop_id, self.opt_fid_id)
            return self.base.heads[f"p{prop_id}_f{self.opt_fid_id}"](h).squeeze(-1)
        h = self.base.encode(node_feats, coords, mask, original_mask, prop_id, self.mbj_fid_id)
        y_mbj_parent = self.base.heads[f"p{prop_id}_f{self.mbj_fid_id}"](h).squeeze(-1)
        return y_mbj_parent + self.correction(h).squeeze(-1)


class LowRankResidual(nn.Module):
    """Low-rank residual: delta(h) = h @ A @ B."""

    def __init__(self, hidden_dim: int, rank: int) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.randn(hidden_dim, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.A @ self.B


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _make_parameter_matched_mlp(hidden_dim: int, target_params: int) -> nn.Sequential:
    """Return a two-layer MLP whose trainable count is close to ``target_params``."""
    # Binary search over bottleneck size.
    lo, hi = 1, hidden_dim
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


def _frozen_mbj_correction_experiment(
    tasks: list[tuple[str, str, str]],
    task_records: list[list[dict]],
    node_dim: int,
    hidden_dim: int,
    device: torch.device,
    base_state_dict: dict[str, torch.Tensor] | None,
    correction: nn.Module,
    label: str,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    adapter_rank: int,
    num_nearest_neighbors: int,
    opt_parent_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a correction module on top of the frozen MBJ parent representation."""
    prop2id, fid2id = _name_to_id(tasks)
    opt_prop_id = prop2id["band_gap"]
    opt_fid_id = fid2id["OptB88vdW"]
    mbj_fid_id = fid2id["TB-mBJ"]

    base_model = PhyTCAModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=len(prop2id),
        n_fidelities=len(fid2id),
        n_layers=3,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        freeze_encoder_weights=True,
    ).to(device)

    start = time.time()

    if opt_parent_state is not None:
        base_model.load_state_dict({k: v.to(device) for k, v in opt_parent_state["state_dict"].items()})
        mean, std, mad = opt_parent_state["mean"], opt_parent_state["std"], opt_parent_state["mad"]
        t1_after_t1 = opt_parent_state["task1_after_task1"]
    else:
        if base_state_dict is not None:
            _load_canonical_base(base_model, base_state_dict, device)
        train_loader, val_loader, _, mean, std, mad = _make_loaders(task_records[0], batch_size)
        _train_single_task(
            base_model, train_loader, val_loader, opt_prop_id, opt_fid_id, device,
            epochs=epochs, lr=lr, patience=patience,
        )
        dev_ds = JARVISCrystalDataset(task_records[0], split="continual_dev")
        dev_ds.target_mean = float(mean)
        dev_ds.target_std = float(std)
        dev_ds.normalize_target = True
        dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals)
        t1_after_t1 = evaluate_loader(base_model, dev_loader, opt_prop_id, opt_fid_id, mean, std, mad, device)

    task1_stats = (mean, std, mad)

    for p in base_model.parameters():
        p.requires_grad = False

    correction_model = FrozenMbjCorrectionModel(
        base_model, opt_prop_id, opt_fid_id, mbj_fid_id, correction.to(device)
    ).to(device)
    task2_trainable_names = _trainable_param_names(correction_model)

    train_loader2, val_loader2, _, mean2, std2, mad2 = _make_loaders(task_records[1], batch_size)
    task2_stats = (mean2, std2, mad2)
    _train_single_task(
        correction_model, train_loader2, val_loader2, opt_prop_id, mbj_fid_id, device,
        epochs=epochs, lr=lr, patience=patience,
    )

    result = _run_label_and_metrics(
        label, correction_model, tasks, task_records, [task1_stats, task2_stats],
        prop2id, fid2id, batch_size, device, start,
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
    """Frozen MBJ parent + low-rank residual on the pooled representation."""
    correction = LowRankResidual(hidden_dim=hidden_dim, rank=adapter_rank)
    return _frozen_mbj_correction_experiment(
        tasks, task_records, node_dim, hidden_dim, device, base_state_dict,
        correction=correction, label="fr_phytca_low_rank_residual",
        epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
        adapter_rank=adapter_rank, num_nearest_neighbors=num_nearest_neighbors,
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
    """Frozen MBJ parent + small residual MLP (not matched to FR-PhyTCA)."""
    # Target parameter count matches the FR-PhyTCA Tucker core increment only.
    target_params = adapter_rank * (2 * hidden_dim + adapter_rank) + 1
    correction = _make_parameter_matched_mlp(hidden_dim, target_params)
    return _frozen_mbj_correction_experiment(
        tasks, task_records, node_dim, hidden_dim, device, base_state_dict,
        correction=correction, label="fr_phytca_param_matched_mlp",
        epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
        adapter_rank=adapter_rank, num_nearest_neighbors=num_nearest_neighbors,
        opt_parent_state=opt_parent_state,
    )


def _fr_phytca_incremental_params(
    hidden_dim: int,
    adapter_rank: int,
    n_properties: int,
    n_fidelities: int,
    n_layers: int = 3,
) -> int:
    """Incremental parameter count for FR-PhyTCA (child adapters + new head)."""
    rank_prop = max(2, n_properties)
    rank_fid = max(2, n_fidelities)
    per_layer = (
        hidden_dim * adapter_rank  # U_in
        + hidden_dim * adapter_rank  # U_out
        + adapter_rank * adapter_rank * rank_prop * rank_fid  # G
        + n_properties * rank_prop  # E_prop
        + n_fidelities * rank_fid  # E_fid
    )
    return n_layers * per_layer + (hidden_dim + 1)  # new head


def _find_low_rank_for_target(hidden_dim: int, target_params: int) -> int:
    """Return rank so that LowRankResidual params are within 5% of target."""
    # Params = rank * (hidden_dim + 1)
    denom = hidden_dim + 1
    ideal = target_params / denom
    candidates = [max(1, int(ideal)), max(1, int(ideal) + 1)]
    best = min(candidates, key=lambda r: abs(r * denom - target_params))
    assert abs(best * denom - target_params) / target_params <= 0.05
    return best


def _find_mlp_bottleneck_for_target(hidden_dim: int, target_params: int) -> int:
    """Return bottleneck so that two-layer MLP params are within 5% of target."""
    # Params = (hidden_dim * b + b) + (b * 1 + 1) = b * (hidden_dim + 2) + 1
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
    result = _frozen_mbj_correction_experiment(
        tasks, task_records, node_dim, hidden_dim, device, base_state_dict,
        correction=correction, label="matched_low_rank_residual",
        epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
        adapter_rank=adapter_rank, num_nearest_neighbors=num_nearest_neighbors,
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
    result = _frozen_mbj_correction_experiment(
        tasks, task_records, node_dim, hidden_dim, device, base_state_dict,
        correction=correction, label="matched_mlp_residual",
        epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
        adapter_rank=adapter_rank, num_nearest_neighbors=num_nearest_neighbors,
        opt_parent_state=opt_parent_state,
    )
    result["matched_target_params"] = target
    result["matched_actual_params"] = _count_params(correction)
    return result


# ---------------------------------------------------------------------------
# D6: Progressive Tucker residual + optional OPT distillation
# ---------------------------------------------------------------------------


class ProgressiveTuckerAdapter(nn.Module):
    """Parent adapter frozen after Task 1; child adapter is a random-init residual.

    The child uses the default Tucker initialization so that the MBJ fidelity
    slice can actually learn.  Only the parent OPT slice is forced to zero so
    the frozen OPT route is not perturbed during Task-2 training.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_properties: int,
        n_fidelities: int,
        rank: int = 8,
    ) -> None:
        super().__init__()
        self.parent = Tucker4DAdapter(
            d_in, d_out, n_properties, n_fidelities,
            rank_out=rank, rank_in=rank, rank_prop=max(2, n_properties), rank_fid=max(2, n_fidelities),
        )
        self.child = Tucker4DAdapter(
            d_in, d_out, n_properties, n_fidelities,
            rank_out=rank, rank_in=rank, rank_prop=max(2, n_properties), rank_fid=max(2, n_fidelities),
        )
        # Slices of the child that must remain zero (e.g. the parent OPT slice).
        self.child_frozen_slices: set[tuple[int, int]] = set()

    def forward(self, x: torch.Tensor, prop_id: int, fid_id: int) -> torch.Tensor:
        return self.parent(x, prop_id, fid_id) + self.child(x, prop_id, fid_id)

    def freeze_parent_slice(self, prop_id: int, fid_id: int) -> None:
        self.parent.freeze_slice(prop_id, fid_id)

    def zero_and_freeze_child_slice(self, prop_id: int, fid_id: int) -> None:
        """Make the child's slice for a parent fidelity permanently zero.

        This guarantees that the OPT forward path does not depend on Task-2
        updates to the shared child ``U_in``/``U_out`` factors.  We do not set
        ``requires_grad`` on slices (PyTorch forbids it for non-leaf views);
        gradients for these slices are zeroed in the post-backward hook.
        """
        self.child_frozen_slices.add((int(prop_id), int(fid_id)))
        with torch.no_grad():
            self.child.G[:, :, prop_id, fid_id].zero_()
            self.child.E_prop.weight[prop_id].zero_()
            self.child.E_fid.weight[fid_id].zero_()

    def zero_child_gradients_for_parent(self) -> None:
        """Zero gradients that would update the child for already-frozen slices."""
        if self.child.G.grad is not None:
            for p, f in self.child_frozen_slices:
                self.child.G.grad[:, :, p, f].zero_()
        if self.child.E_prop.weight.grad is not None:
            for p, _ in self.child_frozen_slices:
                self.child.E_prop.weight.grad[p].zero_()
        if self.child.E_fid.weight.grad is not None:
            for _, f in self.child_frozen_slices:
                self.child.E_fid.weight.grad[f].zero_()


class ProgressiveAdapterCrystalGraphLayer(nn.Module):
    """Crystal graph layer with a progressive parent+residual Tucker adapter."""

    def __init__(
        self,
        dim: int,
        n_properties: int,
        n_fidelities: int,
        rank: int = 8,
        num_nearest_neighbors: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = AdapterCrystalGraphLayer(dim, n_properties, n_fidelities, rank, num_nearest_neighbors).encoder
        self.adapter = ProgressiveTuckerAdapter(
            dim, dim, n_properties, n_fidelities, rank=rank,
        )

    def forward(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        prop_id: int,
        fid_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        new_feats, new_coords = self.encoder(feats, coords, mask=mask)
        delta = self.adapter(new_feats, prop_id, fid_id)
        return new_feats + delta, new_coords


class ProgressivePhyTCAModel(nn.Module):
    """PhyTCA variant where each fidelity slice is parent + zero-init residual."""

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        n_properties: int,
        n_fidelities: int,
        n_layers: int = 3,
        adapter_rank: int = 8,
        num_nearest_neighbors: int = 8,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.n_properties = n_properties
        self.n_fidelities = n_fidelities
        self.n_layers = n_layers

        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList([
            ProgressiveAdapterCrystalGraphLayer(
                hidden_dim, n_properties, n_fidelities, rank=adapter_rank,
                num_nearest_neighbors=num_nearest_neighbors,
            )
            for _ in range(n_layers)
        ])
        self.heads = nn.ModuleDict()
        for p in range(n_properties):
            for f in range(n_fidelities):
                self.heads[f"p{p}_f{f}"] = nn.Linear(hidden_dim, 1)

        # Keep crystal graph encoder frozen; only adapters/heads are trained.
        for p in self.encoder_parameters():
            p.requires_grad = False

    def encoder_parameters(self) -> list[nn.Parameter]:
        params = list(self.node_embed.parameters())
        for layer in self.layers:
            params.extend(layer.encoder.parameters())
        return params

    def encode(
        self,
        node_feats: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        original_mask: torch.Tensor,
        prop_id: int,
        fid_id: int,
    ) -> torch.Tensor:
        h = self.node_embed(node_feats)
        for layer in self.layers:
            h, coords = layer(h, coords, mask, prop_id, fid_id)
        mask_exp = original_mask.unsqueeze(-1).float()
        return (h * mask_exp).sum(dim=1) / (mask_exp.sum(dim=1).clamp_min(1.0))

    def forward(
        self,
        node_feats: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        original_mask: torch.Tensor,
        prop_id: int,
        fid_id: int,
    ) -> torch.Tensor:
        pooled = self.encode(node_feats, coords, mask, original_mask, prop_id, fid_id)
        return self.heads[f"p{prop_id}_f{fid_id}"](pooled).squeeze(-1)

    def freeze_parent_task(self, prop_id: int, fid_id: int) -> None:
        """Freeze the parent adapter slice and the corresponding head."""
        for layer in self.layers:
            layer.adapter.freeze_parent_slice(prop_id, fid_id)
            # Zero-grad guard for the child is handled by the post-backward hook.
        for p in self.heads[f"p{prop_id}_f{fid_id}"].parameters():
            p.requires_grad = False

        # Freeze all parent adapter parameters.
        for layer in self.layers:
            for p in layer.adapter.parent.parameters():
                p.requires_grad = False

    def count_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _remap_phytca_to_progressive_state(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map a PhyTCAModel state dict into a ProgressivePhyTCAModel parent state."""
    remapped: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("layers.") and ".adapter." in k:
            remapped[k.replace(".adapter.", ".adapter.parent.", 1)] = v
        else:
            remapped[k] = v
    return remapped


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
) -> dict[str, Any]:
    """Progressive Tucker residual with optional OPT distillation.

    When ``opt_parent_state`` is provided, the OPT parent is loaded from that
    shared checkpoint instead of being trained again, so D4/D5/D6 share the
    exact same frozen OPT route.

    Ablations:
        - ``orthogonal_child``: project child ``U_in``/``U_out`` to be orthogonal
          to the parent factors (no distillation).
        - ``shared_factors``: tie child ``U_in``/``U_out`` to the frozen parent
          factors and optionally unfreeze the top encoder layer.
    """
    if orthogonal_child and shared_factors:
        raise ValueError("orthogonal_child and shared_factors are mutually exclusive")
    prop2id, fid2id = _name_to_id(tasks)
    opt_prop_id = prop2id["band_gap"]
    opt_fid_id = fid2id["OptB88vdW"]
    mbj_fid_id = fid2id["TB-mBJ"]

    model = ProgressivePhyTCAModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=len(prop2id),
        n_fidelities=len(fid2id),
        n_layers=3,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
    ).to(device)

    start = time.time()

    if opt_parent_state is not None:
        parent_state = _remap_phytca_to_progressive_state(opt_parent_state["state_dict"])
        model.load_state_dict({k: v.to(device) for k, v in parent_state.items()}, strict=False)
        for layer in model.layers:
            layer.adapter.zero_and_freeze_child_slice(opt_prop_id, opt_fid_id)
        mean, std, mad = opt_parent_state["mean"], opt_parent_state["std"], opt_parent_state["mad"]
        t1_after_t1 = opt_parent_state["task1_after_task1"]
    else:
        if base_state_dict is not None:
            encoder_state = {k: v for k, v in base_state_dict.items() if "adapter" not in k}
            parent_state = {}
            for k, v in base_state_dict.items():
                if "adapter" in k:
                    parent_key = k.replace("adapter.", "adapter.parent.", 1)
                    parent_state[parent_key] = v
            mapped_state = {**encoder_state, **parent_state}
            _load_canonical_base(model, mapped_state, device)

        # Make the child's OPT slice permanently zero so the OPT path stays pure parent.
        for layer in model.layers:
            layer.adapter.zero_and_freeze_child_slice(opt_prop_id, opt_fid_id)

        # Task 1: train OPT parent only (child frozen).
        for layer in model.layers:
            for p in layer.adapter.child.parameters():
                p.requires_grad = False

        train_loader, val_loader, _, mean, std, mad = _make_loaders(task_records[0], batch_size)
        _train_single_task(
            model, train_loader, val_loader, opt_prop_id, opt_fid_id, device,
            epochs=epochs, lr=lr, patience=patience,
        )

        dev_ds = JARVISCrystalDataset(task_records[0], split="continual_dev")
        dev_ds.target_mean = float(mean)
        dev_ds.target_std = float(std)
        dev_ds.normalize_target = True
        dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals)
        t1_after_t1 = evaluate_loader(model, dev_loader, opt_prop_id, opt_fid_id, mean, std, mad, device)

    task1_stats = (mean, std, mad)

    # Freeze parent and enable child.
    model.freeze_parent_task(opt_prop_id, opt_fid_id)

    if orthogonal_child:
        _orthogonalize_child_factors(model)
    if shared_factors:
        _tie_child_factors_to_parent(model)
    if unfreeze_top_layer:
        for p in model.layers[-1].encoder.parameters():
            p.requires_grad = True

    for layer in model.layers:
        # The OPT slice of the child core/embedding is kept zero by the guard;
        # MBJ slice uses these parameters freely.
        layer.adapter.child.G.requires_grad = True
        layer.adapter.child.E_prop.weight.requires_grad = True
        layer.adapter.child.E_fid.weight.requires_grad = True
        if not shared_factors:
            layer.adapter.child.U_in.requires_grad = True
            layer.adapter.child.U_out.requires_grad = True
    # Unfreeze MBJ head.
    for p in model.heads[f"p{opt_prop_id}_f{mbj_fid_id}"].parameters():
        p.requires_grad = True

    task2_trainable_names = _trainable_param_names(model)

    # Snapshot OPT predictions before Task 2 for the invariance gate.
    dev_ds = JARVISCrystalDataset(task_records[0], split="continual_dev")
    dev_ds.target_mean = float(mean)
    dev_ds.target_std = float(std)
    dev_ds.normalize_target = True
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals)
    opt_preds_before: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for node_feats, coords, mask, original_mask, _ in dev_loader:
            node_feats = node_feats.to(device)
            coords = coords.to(device)
            mask = mask.to(device)
            original_mask = original_mask.to(device)
            pred = model(node_feats, coords, mask, original_mask, opt_prop_id, opt_fid_id)
            opt_preds_before.append(pred.detach().cpu())
    opt_preds_before = torch.cat(opt_preds_before)

    # Task 2: train child residual (+ optional distillation).
    train_loader2, val_loader2, _, mean2, std2, mad2 = _make_loaders(task_records[1], batch_size)
    task2_stats = (mean2, std2, mad2)

    teacher_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    teacher = ProgressivePhyTCAModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=len(prop2id),
        n_fidelities=len(fid2id),
        n_layers=3,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
    ).to(device)
    teacher.load_state_dict({k: v.to(device) for k, v in teacher_state.items()})
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    def post_backward(m: nn.Module) -> None:
        for layer in m.layers:
            layer.adapter.zero_child_gradients_for_parent()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_nmae = float("inf")
    best_state = None
    patience_counter = 0

    for _ in range(epochs):
        model.train()
        for node_feats, coords, mask, original_mask, y in train_loader2:
            node_feats = node_feats.to(device)
            coords = coords.to(device)
            mask = mask.to(device)
            original_mask = original_mask.to(device)
            y_norm = ((y.to(device) - mean2.to(device)) / std2.to(device)).float()

            optimizer.zero_grad()
            pred_mbj = model(node_feats, coords, mask, original_mask, opt_prop_id, mbj_fid_id)
            loss = F.mse_loss(pred_mbj, y_norm)

            if lambda_distill > 0.0:
                with torch.no_grad():
                    teacher_opt = teacher(node_feats, coords, mask, original_mask, opt_prop_id, opt_fid_id)
                pred_opt = model(node_feats, coords, mask, original_mask, opt_prop_id, opt_fid_id)
                loss += lambda_distill * F.smooth_l1_loss(pred_opt, teacher_opt)

            loss.backward()
            post_backward(model)
            optimizer.step()

        val_nmae = evaluate_loader(model, val_loader2, opt_prop_id, mbj_fid_id, mean2, std2, mad2, device)
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

    # OPT route invariance gate: predictions must be identical before/after Task 2.
    opt_preds_after: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for node_feats, coords, mask, original_mask, _ in dev_loader:
            node_feats = node_feats.to(device)
            coords = coords.to(device)
            mask = mask.to(device)
            original_mask = original_mask.to(device)
            pred = model(node_feats, coords, mask, original_mask, opt_prop_id, opt_fid_id)
            opt_preds_after.append(pred.detach().cpu())
    opt_preds_after = torch.cat(opt_preds_after)
    opt_route_drift = float((opt_preds_after - opt_preds_before).abs().max())

    task_stats = [task1_stats, task2_stats]
    label = "fr_phytca"
    if orthogonal_child:
        label += "_orthogonal"
    if shared_factors:
        label += "_shared_factor_top_layer"
    if lambda_distill > 0.0:
        label += f"_distill_{lambda_distill}"
    result = _run_label_and_metrics(
        label, model, tasks, task_records, task_stats,
        prop2id, fid2id, batch_size, device, start,
    )
    result["task1_after_task1"] = t1_after_t1
    result["absolute_forgetting"] = result["task1_after_task2"] - t1_after_t1
    result["bwt"] = backward_transfer([[t1_after_t1], result["nmaes"][1]])
    result["incremental_params"] = result["trainable_params"]
    result["task2_trainable_param_names"] = task2_trainable_names
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
    """FR-PhyTCA with child Tucker factors orthogonal to the parent factors."""
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
        lambda_distill=0.0,
        orthogonal_child=True,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        opt_parent_state=opt_parent_state,
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
    """FR-PhyTCA with shared Tucker factors and a trainable top encoder layer."""
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
        lambda_distill=0.0,
        shared_factors=True,
        unfreeze_top_layer=True,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        opt_parent_state=opt_parent_state,
    )


# ---------------------------------------------------------------------------
# Additional baselines for 5k scaling study
# ---------------------------------------------------------------------------


class FeatureTransferModel(nn.Module):
    """Freeze OPT route and train an MBJ head on concatenated OPT+MBJ representations.

    This baseline distinguishes output-delta transfer (D5), latent-feature
    transfer (this model), and structured tensor residuals (FR-PhyTCA).
    """

    def __init__(
        self,
        base_model: PhyTCAModel,
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
            return self.base.heads[f"p{prop_id}_f{self.opt_fid_id}"](h_opt).squeeze(-1)
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

    base_model = PhyTCAModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=len(prop2id),
        n_fidelities=len(fid2id),
        n_layers=3,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        freeze_encoder_weights=True,
    ).to(device)

    start = time.time()

    if opt_parent_state is not None:
        base_model.load_state_dict({k: v.to(device) for k, v in opt_parent_state["state_dict"].items()})
        mean, std, mad = opt_parent_state["mean"], opt_parent_state["std"], opt_parent_state["mad"]
        t1_after_t1 = opt_parent_state["task1_after_task1"]
    else:
        if base_state_dict is not None:
            _load_canonical_base(base_model, base_state_dict, device)
        train_loader, val_loader, _, mean, std, mad = _make_loaders(task_records[0], batch_size)
        _train_single_task(
            base_model, train_loader, val_loader, opt_prop_id, opt_fid_id, device,
            epochs=epochs, lr=lr, patience=patience,
        )
        dev_ds = JARVISCrystalDataset(task_records[0], split="continual_dev")
        dev_ds.target_mean = float(mean)
        dev_ds.target_std = float(std)
        dev_ds.normalize_target = True
        dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals)
        t1_after_t1 = evaluate_loader(base_model, dev_loader, opt_prop_id, opt_fid_id, mean, std, mad, device)

    task1_stats = (mean, std, mad)

    for p in base_model.parameters():
        p.requires_grad = False

    ft_model = FeatureTransferModel(base_model, opt_prop_id, opt_fid_id, mbj_fid_id).to(device)
    task2_trainable_names = _trainable_param_names(ft_model)

    train_loader2, val_loader2, _, mean2, std2, mad2 = _make_loaders(task_records[1], batch_size)
    task2_stats = (mean2, std2, mad2)
    _train_single_task(
        ft_model, train_loader2, val_loader2, opt_prop_id, mbj_fid_id, device,
        epochs=epochs, lr=lr, patience=patience,
    )

    result = _run_label_and_metrics(
        "feature_transfer", ft_model, tasks, task_records, [task1_stats, task2_stats],
        prop2id, fid2id, batch_size, device, start,
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

    model = PhyTCAModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=len(prop2id),
        n_fidelities=len(fid2id),
        n_layers=3,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        freeze_encoder_weights=True,
    ).to(device)
    if base_state_dict is not None:
        _load_canonical_base(model, base_state_dict, device)

    start = time.time()
    train_loader, val_loader, _, mean, std, mad = _make_loaders(task_records[1], batch_size)
    _train_single_task(
        model, train_loader, val_loader, opt_prop_id, mbj_fid_id, device,
        epochs=epochs, lr=lr, patience=patience,
    )

    # Evaluate on both tasks using the single trained model.
    task_stats = [(mean, std, mad), (mean, std, mad)]
    result = _run_label_and_metrics(
        "mbj_only", model, tasks, task_records, task_stats,
        prop2id, fid2id, batch_size, device, start,
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
    """Pre-train on OPT, then full fine-tune on MBJ (shows catastrophic forgetting)."""
    prop2id, fid2id = _name_to_id(tasks)
    opt_prop_id = prop2id["band_gap"]
    opt_fid_id = fid2id["OptB88vdW"]
    mbj_fid_id = fid2id["TB-mBJ"]

    model = PhyTCAModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=len(prop2id),
        n_fidelities=len(fid2id),
        n_layers=3,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        freeze_encoder_weights=False,
    ).to(device)

    start = time.time()

    if opt_parent_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in opt_parent_state["state_dict"].items()})
        mean, std, mad = opt_parent_state["mean"], opt_parent_state["std"], opt_parent_state["mad"]
        t1_after_t1 = opt_parent_state["task1_after_task1"]
    else:
        if base_state_dict is not None:
            _load_canonical_base(model, base_state_dict, device)
        train_loader, val_loader, _, mean, std, mad = _make_loaders(task_records[0], batch_size)
        _train_single_task(
            model, train_loader, val_loader, opt_prop_id, opt_fid_id, device,
            epochs=epochs, lr=lr, patience=patience,
        )
        dev_ds = JARVISCrystalDataset(task_records[0], split="continual_dev")
        dev_ds.target_mean = float(mean)
        dev_ds.target_std = float(std)
        dev_ds.normalize_target = True
        dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals)
        t1_after_t1 = evaluate_loader(model, dev_loader, opt_prop_id, opt_fid_id, mean, std, mad, device)

    task1_stats = (mean, std, mad)

    # Unfreeze everything for full MBJ fine-tuning.
    for p in model.parameters():
        p.requires_grad = True

    train_loader2, val_loader2, _, mean2, std2, mad2 = _make_loaders(task_records[1], batch_size)
    task2_stats = (mean2, std2, mad2)
    _train_single_task(
        model, train_loader2, val_loader2, opt_prop_id, mbj_fid_id, device,
        epochs=epochs, lr=lr, patience=patience,
    )

    result = _run_label_and_metrics(
        "opt_pretrain_mbj_full_finetune", model, tasks, task_records, [task1_stats, task2_stats],
        prop2id, fid2id, batch_size, device, start,
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
}
