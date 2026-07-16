"""Phase 0 Protocol B two-task screening with fair initialization and paired recheck.

Runs a small-scale (2k train / 500 val / 1k held-out per task), single-seed
comparison of continual-learning methods on Protocol B task 1 (2021 OPT)
followed by task 2 (2021 MBJ).  All comparable methods start from the same
canonical base checkpoint per seed and see the same batch order.  The held-out
split is further divided into ``continual_dev`` (used for reporting and GO/NO-GO)
and ``final_test`` (frozen, unused for tuning or decisions).

The script supports a ``--paired-recheck`` mode that runs only PhyTCA with
:math:`\\mu=0` versus :math:`\\mu=0.01` and asserts identical Task-1 trajectories.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legacy.baselines import LoRALinear
from data import JARVISCrystalDataset, build_protocol_b, cap_splits, collate_crystals
from legacy.phytca import PhyTCAModel
from train_utils import backward_transfer, compute_mad, forgetting, normalized_mae


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _repartition_dev_test(records: list[dict], dev_frac: float = 0.5, seed: int = 42) -> list[dict]:
    """Repartition records with split 'test' into 'continual_dev' and 'final_test'."""
    rng = np.random.default_rng(seed)
    test_recs = [r for r in records if r.get("split") == "test"]
    other_recs = [r for r in records if r.get("split") != "test"]
    n_dev = int(len(test_recs) * dev_frac)
    idx = np.arange(len(test_recs))
    rng.shuffle(idx)
    for i in idx[:n_dev]:
        test_recs[i]["split"] = "continual_dev"
    for i in idx[n_dev:]:
        test_recs[i]["split"] = "final_test"
    return other_recs + test_recs


def _make_loaders(
    recs: list[dict],
    batch_size: int,
    mean: float | None = None,
    std: float | None = None,
    splits: tuple[str, ...] = ("train", "val", "continual_dev", "final_test"),
    generator: torch.Generator | None = None,
) -> tuple[dict[str, DataLoader], torch.Tensor, torch.Tensor, float]:
    """Return loaders for requested splits plus normalization stats.

    The ``train`` loader uses the provided generator for deterministic shuffling.
    """
    loaders: dict[str, DataLoader] = {}
    datasets: dict[str, JARVISCrystalDataset] = {}
    for split in splits:
        ds = JARVISCrystalDataset(recs, split=split)
        datasets[split] = ds
        if split == "train":
            loaders[split] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_crystals,
                generator=generator,
            )
        else:
            loaders[split] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_crystals,
            )

    if mean is None or std is None:
        train_targets = torch.tensor(
            [r["target"] for r in recs if r.get("split") == "train"], dtype=torch.float32
        )
        mean = float(train_targets.mean())
        std = float(train_targets.std().clamp_min(1e-6))

    for ds in datasets.values():
        ds.target_mean = float(mean)
        ds.target_std = float(std)
        ds.normalize_target = True

    mad = compute_mad(
        torch.tensor([r["target"] for r in recs if r.get("split") == "train"], dtype=torch.float32)
    )
    return loaders, torch.tensor(mean), torch.tensor(std), mad


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    prop_id: int,
    fid_id: int,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    mad: float,
    device: torch.device,
) -> float:
    """Evaluate nMAE on a loader."""
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


def _total_grad_norm(model: nn.Module) -> float:
    """Total L2 gradient norm over all parameters with gradients."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def _canonical_base_state(
    seed: int,
    node_dim: int,
    hidden_dim: int,
    n_properties: int,
    n_fidelities: int,
    adapter_rank: int,
    num_nearest_neighbors: int,
    artifact_dir: Path,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build or load a canonical frozen-encoder base model state for the seed."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    init_path = artifact_dir / f"seed_{seed}_base.pt"
    if init_path.exists():
        state_dict = torch.load(init_path, map_location="cpu")
        print(f"  Loaded canonical base state from {init_path}")
        return state_dict

    # Deterministic canonical initialization.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = PhyTCAModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=n_properties,
        n_fidelities=n_fidelities,
        n_layers=3,
        adapter_rank=adapter_rank,
        num_nearest_neighbors=num_nearest_neighbors,
        freeze_encoder_weights=True,
    ).to(device)
    state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    torch.save(state_dict, init_path)
    print(f"  Saved canonical base state to {init_path}")
    return state_dict


def _detailed_parameter_stats(
    model: PhyTCAModel,
    method: str,
    replay_buffer: list | None = None,
    n_task_models: int = 1,
) -> dict[str, Any]:
    """Return detailed trainable/stored parameter and storage statistics."""
    group_counts = model.get_parameter_group_counts()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    stored = sum(p.numel() for p in model.parameters())

    # For methods that store multiple full models, scale base model counts.
    total_encoder = group_counts["encoder"] * n_task_models
    total_adapter = group_counts["adapter"] * n_task_models
    total_heads = group_counts["heads"] * n_task_models
    total_stored = stored * n_task_models
    total_trainable = trainable * n_task_models

    replay_bytes = 0
    replay_count = 0
    if replay_buffer:
        replay_count = len(replay_buffer)
        for item in replay_buffer:
            # Each stored replay sample is a tuple; tensor elements dominate storage.
            for tensor in item:
                if isinstance(tensor, torch.Tensor):
                    replay_bytes += tensor.numel() * tensor.element_size()

    return {
        "base_model_params": total_encoder,
        "adapter_params": total_adapter,
        "head_params": total_heads,
        "trainable_params": total_trainable,
        "stored_params": total_stored,
        "replay_sample_count": replay_count,
        "replay_storage_bytes": replay_bytes,
        "per_task_model_params": stored,
        "per_task_trainable_params": trainable,
    }


def _reset_rngs(seed: int) -> None:
    """Reset Python, NumPy, and PyTorch RNGs to a known seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _auto_batch_size(model: nn.Module, sample_batch: tuple, device: torch.device) -> int:
    """Try batch size 32, fall back to 16 on OOM."""
    for bs in (32, 16):
        try:
            node_feats, coords, mask, original_mask, _ = sample_batch
            node_feats = node_feats[:bs].to(device)
            coords = coords[:bs].to(device)
            mask = mask[:bs].to(device)
            original_mask = original_mask[:bs].to(device)
            with autocast('cuda'):
                _ = model(node_feats, coords, mask, original_mask, prop_id=0, fid_id=0)
            torch.cuda.empty_cache()
            return bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
    return 16


# ---------------------------------------------------------------------------
# Training core with per-epoch logging
# ---------------------------------------------------------------------------


def train_task_logged(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    probe_loaders: list[tuple[int, int, DataLoader, torch.Tensor, torch.Tensor, float]],
    prop_id: int,
    fid_id: int,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    patience: int = 3,
    stability_loss_fn: Callable | None = None,
    post_backward_fn: Callable[[nn.Module], None] | None = None,
    use_amp: bool = True,
) -> tuple[float, torch.Tensor, torch.Tensor, float, int, list[dict]]:
    """Train one task with early stopping and detailed per-epoch logging.

    Returns best val nMAE, mean, std, mad, best epoch, and epoch logs.
    """
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler('cuda') if use_amp and device.type == "cuda" else None

    all_targets = []
    for _, _, _, _, y in train_loader:
        all_targets.append(y)
    all_targets = torch.cat(all_targets)
    target_mean = all_targets.mean()
    target_std = all_targets.std().clamp_min(1e-6)
    mad = compute_mad(all_targets)

    best_nmae = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    epoch_logs: list[dict] = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_stab_loss = 0.0
        n_batches = 0
        for node_feats, coords, mask, original_mask, y in train_loader:
            node_feats = node_feats.to(device)
            coords = coords.to(device)
            mask = mask.to(device)
            original_mask = original_mask.to(device)
            y_norm = ((y.to(device) - target_mean) / target_std).float()

            optimizer.zero_grad()
            if scaler is not None:
                with autocast('cuda'):
                    pred = model(node_feats, coords, mask, original_mask, prop_id, fid_id)
                    loss = F.mse_loss(pred, y_norm)
                    stab = torch.tensor(0.0, device=device)
                    if stability_loss_fn is not None:
                        stab = stability_loss_fn(model)
                        loss += stab
                scaler.scale(loss).backward()
                if post_backward_fn is not None:
                    post_backward_fn(model)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(node_feats, coords, mask, original_mask, prop_id, fid_id)
                loss = F.mse_loss(pred, y_norm)
                stab = torch.tensor(0.0, device=device)
                if stability_loss_fn is not None:
                    stab = stability_loss_fn(model)
                    loss += stab
                loss.backward()
                if post_backward_fn is not None:
                    post_backward_fn(model)
                optimizer.step()

            epoch_loss += float(loss.detach())
            epoch_stab_loss += float(stab.detach())
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        avg_stab_loss = epoch_stab_loss / max(n_batches, 1)
        val_nmae = _evaluate(model, val_loader, prop_id, fid_id, target_mean, target_std, mad, device)
        probe_nmaes = {}
        for pid, fid, loader, mean_p, std_p, mad_p in probe_loaders:
            probe_nmaes[f"p{pid}_f{fid}"] = _evaluate(model, loader, pid, fid, mean_p, std_p, mad_p, device)
        grad_norm = _total_grad_norm(model)

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "stability_loss": avg_stab_loss,
            "val_nmae": val_nmae,
            "probe_nmaes": probe_nmaes,
            "lr": optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm,
        }
        epoch_logs.append(log_entry)

        if val_nmae < best_nmae:
            best_nmae = val_nmae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        scheduler.step()
        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return best_nmae, target_mean, target_std, mad, best_epoch, epoch_logs


# ---------------------------------------------------------------------------
# Continual experiment wrapper
# ---------------------------------------------------------------------------


def run_continual_method(
    method: str,
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
    base_state_dict: dict[str, torch.Tensor] | None = None,
    seed: int = 42,
    use_amp: bool = True,
    mu: float | None = None,
) -> dict[str, Any]:
    """Run a two-task continual method with fair initialization and return metrics."""
    from train_phytca import _name_to_id

    prop2id, fid2id = _name_to_id(tasks)
    n_props, n_fids = len(prop2id), len(fid2id)

    # Reset RNGs so every method starts from the same random state.
    _reset_rngs(seed)
    train_generator = torch.Generator().manual_seed(seed)

    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    start = time.time()

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
    if base_state_dict is not None:
        missing, unexpected = model.load_state_dict(copy.deepcopy(base_state_dict), strict=False)
        if missing or unexpected:
            print(f"  Warning: state dict mismatch missing={missing} unexpected={unexpected}")

    # Per-task loaders and stats.
    task_stats: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    test_loaders_per_task: list[list[tuple[int, int, DataLoader, torch.Tensor, torch.Tensor, float]]] = []
    train_loaders: list[DataLoader] = []
    val_loaders: list[DataLoader] = []

    for t, (_, prop_name, fid_name) in enumerate(tasks):
        pid = prop2id[prop_name]
        fid = fid2id[fid_name]
        loaders, mean, std, mad = _make_loaders(
            task_records[t],
            batch_size,
            splits=("train", "val", "continual_dev", "final_test"),
            generator=train_generator,
        )
        task_stats.append((mean, std, mad))
        train_loaders.append(loaders["train"])
        val_loaders.append(loaders["val"])
        test_loaders_per_task.append([
            (pid, fid, loaders["continual_dev"], mean, std, mad),
        ])

    # Auto batch size on first batch if requested.
    if batch_size == "auto":
        sample_batch = next(iter(train_loaders[0]))
        batch_size = _auto_batch_size(model, sample_batch, device)
        train_loaders = []
        for t in range(len(tasks)):
            loaders, mean, std, mad = _make_loaders(
                task_records[t],
                batch_size,
                splits=("train", "val", "continual_dev", "final_test"),
                generator=train_generator,
            )
            task_stats[t] = (mean, std, mad)
            train_loaders.append(loaders["train"])
            val_loaders[t] = loaders["val"]

    # Method-specific setup.
    replay_buffer: list[tuple] = []
    lora_models: list[PhyTCAModel] | None = None
    n_task_models = 1
    if method == "phytca_no_stability":
        mu = 0.0
    elif method == "phytca":
        mu = mu if mu is not None else 0.01
    else:
        mu = 0.0

    if method == "sequential":
        # Same architecture, but the encoder is trainable.
        for p in model.encoder_parameters():
            p.requires_grad = True
    elif method == "frozen_heads":
        for p in model.parameters():
            p.requires_grad = False
        for head in model.heads.values():
            for p in head.parameters():
                p.requires_grad = True
    elif method == "replay_1pct":
        pass
    elif method == "shared_lora":
        model.node_embed = LoRALinear(model.node_embed, rank=4)
        for key, head in model.heads.items():
            model.heads[key] = LoRALinear(head, rank=4)
        model.to(device)
    elif method in ("phytca", "phytca_no_stability"):
        pass
    else:
        raise ValueError(f"Unknown method: {method}")

    def post_backward_fn(m: nn.Module) -> None:
        if method in ("phytca", "phytca_no_stability"):
            for layer in m.layers:
                layer.adapter.zero_frozen_gradients()

    anchor: dict = {}
    nmaes: list[list[float]] = []
    per_task_logs: list[list[dict]] = []
    best_epochs: list[int] = []

    for t, (_, prop_name, fid_name) in enumerate(tasks):
        pid = prop2id[prop_name]
        fid = fid2id[fid_name]

        probe_loaders: list[tuple[int, int, DataLoader, torch.Tensor, torch.Tensor, float]] = []
        for prev_t in range(t + 1):
            probe_loaders.extend(test_loaders_per_task[prev_t])

        def stability_fn(m: nn.Module) -> torch.Tensor:
            if mu > 0 and hasattr(m, "stability_loss"):
                return m.stability_loss(mu, anchor)
            return torch.tensor(0.0, device=device)

        def extra_loss_fn(m: nn.Module) -> torch.Tensor:
            loss = stability_fn(m)
            if method == "replay_1pct" and replay_buffer and t > 0:
                n_replay = min(len(replay_buffer), batch_size)
                for rp, rf, r_nf, r_c, r_m, r_om, r_y in replay_buffer[:n_replay]:
                    r_pred = m(r_nf, r_c, r_m, r_om, rp, rf)
                    loss += F.mse_loss(r_pred, r_y)
            return loss

        _, mean, std, mad, best_epoch, logs = train_task_logged(
            model,
            train_loaders[t],
            val_loaders[t],
            probe_loaders,
            pid,
            fid,
            device,
            epochs=epochs,
            lr=lr,
            patience=patience,
            stability_loss_fn=extra_loss_fn,
            post_backward_fn=post_backward_fn,
            use_amp=use_amp,
        )
        per_task_logs.append(logs)
        best_epochs.append(best_epoch)

        # Evaluate on all seen tasks using the continual_dev split.
        row: list[float] = []
        for prev_t in range(t + 1):
            ppid, pfid, test_loader, pmean, pstd, pmad = test_loaders_per_task[prev_t][0]
            row.append(_evaluate(model, test_loader, ppid, pfid, pmean, pstd, pmad, device))
        nmaes.append(row)

        # Replay buffer: store 1% of previous task train samples.
        if method == "replay_1pct" and t == 0:
            train_count = sum(1 for r in task_records[t] if r.get("split") == "train")
            buffer_size = max(1, int(train_count * 0.01))
            stored = 0
            for node_feats, coords, mask, original_mask, y in train_loaders[t]:
                node_feats = node_feats.to(device)
                coords = coords.to(device)
                mask = mask.to(device)
                original_mask = original_mask.to(device)
                y_norm = ((y.to(device) - mean.to(device)) / std.to(device)).float()
                for i in range(node_feats.size(0)):
                    if stored >= buffer_size:
                        break
                    replay_buffer.append((
                        pid, fid,
                        node_feats[i:i + 1],
                        coords[i:i + 1],
                        mask[i:i + 1],
                        original_mask[i:i + 1],
                        y_norm[i:i + 1],
                    ))
                    stored += 1
                if stored >= buffer_size:
                    break

        # PhyTCA freezing.
        if method in ("phytca", "phytca_no_stability"):
            model.freeze_task(pid, fid)
        anchor = model.anchor_state()

    elapsed = time.time() - start
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0

    task1_after_t1 = nmaes[0][0]
    task1_after_t2 = nmaes[1][0]
    task2_final = nmaes[1][1]
    abs_forgetting = task1_after_t2 - task1_after_t1
    rel_forgetting = abs_forgetting / max(task1_after_t1, 1e-8)
    bwt = backward_transfer(nmaes)
    avg_final = sum(nmaes[-1]) / len(nmaes[-1])

    stats = _detailed_parameter_stats(model, method, replay_buffer, n_task_models)

    return {
        "method": method,
        "status": "ok",
        "batch_size": batch_size,
        "nmaes": nmaes,
        "task1_after_task1": task1_after_t1,
        "task1_after_task2": task1_after_t2,
        "task2_final_nmae": task2_final,
        "absolute_forgetting": abs_forgetting,
        "relative_forgetting": rel_forgetting,
        "bwt": bwt,
        "average_final_nmae": avg_final,
        "forgetting": forgetting(nmaes),
        **stats,
        "peak_gpu_mb": peak_mem,
        "wall_time_seconds": elapsed,
        "best_epochs": best_epochs,
        "per_task_epoch_logs": per_task_logs,
        "task1_final_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
    }


# ---------------------------------------------------------------------------
# Joint training upper bound (optional)
# ---------------------------------------------------------------------------


def run_joint_upper_bound(
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
    base_state_dict: dict[str, torch.Tensor] | None = None,
    seed: int = 42,
    use_amp: bool = True,
) -> dict[str, Any]:
    """Train jointly on all tasks and evaluate on each task."""
    from train_phytca import _name_to_id

    prop2id, fid2id = _name_to_id(tasks)
    n_props, n_fids = len(prop2id), len(fid2id)

    _reset_rngs(seed)
    train_generator = torch.Generator().manual_seed(seed)

    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    start = time.time()

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
    if base_state_dict is not None:
        model.load_state_dict(copy.deepcopy(base_state_dict), strict=False)
    # Joint training unfreezes the encoder.
    for p in model.encoder_parameters():
        p.requires_grad = True

    task_stats: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    loaders: list[dict[str, DataLoader]] = []
    for t, (_, prop_name, fid_name) in enumerate(tasks):
        task_loaders, mean, std, mad = _make_loaders(
            task_records[t],
            batch_size,
            splits=("train", "val", "continual_dev", "final_test"),
            generator=train_generator,
        )
        task_stats.append((mean, std, mad))
        loaders.append(task_loaders)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler('cuda') if use_amp and device.type == "cuda" else None

    best_nmae = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    epoch_logs: list[dict] = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for t, (_, prop_name, fid_name) in enumerate(tasks):
            pid = prop2id[prop_name]
            fid = fid2id[fid_name]
            mean, std, _ = task_stats[t]
            train_loader = loaders[t]["train"]
            for node_feats, coords, mask, original_mask, y in train_loader:
                node_feats = node_feats.to(device)
                coords = coords.to(device)
                mask = mask.to(device)
                original_mask = original_mask.to(device)
                y_norm = ((y.to(device) - mean.to(device)) / std.to(device)).float()

                optimizer.zero_grad()
                if scaler is not None:
                    with autocast('cuda'):
                        pred = model(node_feats, coords, mask, original_mask, pid, fid)
                        loss = F.mse_loss(pred, y_norm)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(node_feats, coords, mask, original_mask, pid, fid)
                    loss = F.mse_loss(pred, y_norm)
                    loss.backward()
                    optimizer.step()

                epoch_loss += float(loss.detach())
                n_batches += 1

        val_nmaes = []
        probe_nmaes = {}
        for t, (_, prop_name, fid_name) in enumerate(tasks):
            pid = prop2id[prop_name]
            fid = fid2id[fid_name]
            val_loader = loaders[t]["val"]
            mean, std, mad = task_stats[t]
            val_nmae = _evaluate(model, val_loader, pid, fid, mean, std, mad, device)
            val_nmaes.append(val_nmae)
            probe_nmaes[f"p{pid}_f{fid}"] = val_nmae

        avg_val = sum(val_nmaes) / len(val_nmaes)
        epoch_logs.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss / max(n_batches, 1),
            "val_nmae": avg_val,
            "probe_nmaes": probe_nmaes,
            "lr": optimizer.param_groups[0]["lr"],
            "grad_norm": _total_grad_norm(model),
        })

        if avg_val < best_nmae:
            best_nmae = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        scheduler.step()
        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Final evaluation on continual_dev split per task.
    nmaes: list[list[float]] = []
    for t, (_, prop_name, fid_name) in enumerate(tasks):
        row: list[float] = []
        for i in range(t + 1):
            pid = prop2id[tasks[i][1]]
            fid = fid2id[tasks[i][2]]
            dev_loader = loaders[i]["continual_dev"]
            mean, std, mad = task_stats[i]
            row.append(_evaluate(model, dev_loader, pid, fid, mean, std, mad, device))
        nmaes.append(row)

    elapsed = time.time() - start
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0

    stats = _detailed_parameter_stats(model, "joint")
    task1_after_t1 = nmaes[0][0]
    task1_after_t2 = nmaes[1][0]
    task2_final = nmaes[1][1]
    return {
        "method": "joint",
        "status": "ok",
        "batch_size": batch_size,
        "nmaes": nmaes,
        "task1_after_task1": task1_after_t1,
        "task1_after_task2": task1_after_t2,
        "task2_final_nmae": task2_final,
        "absolute_forgetting": task1_after_t2 - task1_after_t1,
        "relative_forgetting": (task1_after_t2 - task1_after_t1) / max(task1_after_t1, 1e-8),
        "bwt": backward_transfer(nmaes),
        "average_final_nmae": sum(nmaes[-1]) / len(nmaes[-1]),
        **stats,
        "peak_gpu_mb": peak_mem,
        "wall_time_seconds": elapsed,
        "best_epochs": [best_epoch],
        "per_task_epoch_logs": [epoch_logs],
    }


# ---------------------------------------------------------------------------
# Paired recheck assertions
# ---------------------------------------------------------------------------


def _assert_task1_trajectory_identity(mu0_result: dict, mu_pos_result: dict, tol: float = 1e-4) -> None:
    """Assert that PhyTCA mu=0 and mu>0 produce functionally identical Task-1 runs."""
    assert mu0_result["status"] == "ok", "mu=0 run failed"
    assert mu_pos_result["status"] == "ok", "mu>0 run failed"

    mu0_logs = mu0_result["per_task_epoch_logs"][0]
    mu_pos_logs = mu_pos_result["per_task_epoch_logs"][0]
    assert len(mu0_logs) == len(mu_pos_logs), "Task-1 epoch counts differ"

    for i, (l0, lp) in enumerate(zip(mu0_logs, mu_pos_logs)):
        assert abs(l0["train_loss"] - lp["train_loss"]) < tol, (
            f"Task-1 epoch {i + 1} train_loss differs: {l0['train_loss']} vs {lp['train_loss']}"
        )
        assert abs(l0["val_nmae"] - lp["val_nmae"]) < tol, (
            f"Task-1 epoch {i + 1} val_nmae differs: {l0['val_nmae']} vs {lp['val_nmae']}"
        )
        assert abs(l0["grad_norm"] - lp["grad_norm"]) < tol, (
            f"Task-1 epoch {i + 1} grad_norm differs: {l0['grad_norm']} vs {lp['grad_norm']}"
        )
        # Stability loss on Task 1 must be exactly zero (no anchor yet).
        if "stability_loss" in lp:
            assert abs(lp["stability_loss"]) < tol, (
                f"Task-1 epoch {i + 1} stability loss non-zero: {lp['stability_loss']}"
            )

    # Task-1 final performance must match.
    assert abs(mu0_result["task1_after_task1"] - mu_pos_result["task1_after_task1"]) < tol, (
        f"Task-1 final nMAE differs: {mu0_result['task1_after_task1']} vs {mu_pos_result['task1_after_task1']}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cap", type=int, default=2000)
    parser.add_argument("--val-cap", type=int, default=500)
    parser.add_argument("--test-cap", type=int, default=1000)
    parser.add_argument("--dev-frac", type=float, default=0.5,
                        help="Fraction of held-out test split to use as continual_dev (remainder is final_test)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--num-nearest-neighbors", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--methods", nargs="+", default=["sequential", "replay_1pct", "shared_lora", "phytca_no_stability", "phytca"])
    parser.add_argument("--with-joint", action="store_true")
    parser.add_argument("--paired-recheck", action="store_true",
                        help="Run only PhyTCA mu=0 vs mu=0.01 and assert identical Task-1 trajectory")
    parser.add_argument("--mu", type=float, default=0.01,
                        help="Stability coefficient for PhyTCA (default 0.01)")
    parser.add_argument("--mu-grid", action="store_true",
                        help="Run a grid search over PhyTCA stability coefficients on continual_dev")
    parser.add_argument("--output-dir", default="reports/phase0_b_screening")
    parser.add_argument("--artifact-dir", default="artifacts/init")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    use_amp = not args.no_amp
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(args.artifact_dir)

    # Load Protocol B and cap first two tasks deterministically.
    tasks_all, task_records_all, audit = build_protocol_b(seed=args.seed)
    tasks = tasks_all[:2]
    task_records = [
        cap_splits(recs, args.train_cap, args.val_cap, args.test_cap, seed=args.seed)
        for recs in task_records_all[:2]
    ]
    # Re-split held-out test into continual_dev and final_test.
    task_records = [
        _repartition_dev_test(recs, dev_frac=args.dev_frac, seed=args.seed + t)
        for t, recs in enumerate(task_records)
    ]

    print("=== Protocol B two-task screening ===")
    print(f"seed={args.seed}, train_cap={args.train_cap}, val_cap={args.val_cap}, test_cap={args.test_cap}")
    print(f"dev_frac={args.dev_frac}, paired_recheck={args.paired_recheck}")
    print(f"tasks: {tasks}")
    for t, desc in enumerate(tasks):
        counts = {"train": 0, "val": 0, "continual_dev": 0, "final_test": 0}
        for r in task_records[t]:
            counts[r["split"]] += 1
        print(f"  {desc}: {counts}")

    # Build canonical base initialization state.
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
        artifact_dir=artifact_dir,
        device=device,
    )

    results: list[dict] = []
    runs: list[tuple[str, str, float | None]] = []

    if args.mu_grid:
        mu_values = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
        runs.append(("sequential", "sequential", None))
        for mu_val in mu_values:
            runs.append((f"phytca_mu_{mu_val:g}", "phytca", mu_val))
    else:
        methods = list(args.methods)
        if args.with_joint:
            methods.append("joint")
        if args.paired_recheck:
            methods = ["phytca_no_stability", "phytca"]
        for method in methods:
            mu_val = args.mu if method == "phytca" else None
            runs.append((method, method, mu_val))

    for label, method, mu_val in runs:
        print(f"\n--- Running {label} ---")
        try:
            if method == "joint":
                result = run_joint_upper_bound(
                    tasks=tasks,
                    task_records=task_records,
                    node_dim=92,
                    hidden_dim=args.hidden_dim,
                    device=device,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    patience=args.patience,
                    adapter_rank=args.adapter_rank,
                    num_nearest_neighbors=args.num_nearest_neighbors,
                    base_state_dict=base_state,
                    seed=args.seed,
                    use_amp=use_amp,
                )
            else:
                result = run_continual_method(
                    method=method,
                    tasks=tasks,
                    task_records=task_records,
                    node_dim=92,
                    hidden_dim=args.hidden_dim,
                    device=device,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    patience=args.patience,
                    adapter_rank=args.adapter_rank,
                    num_nearest_neighbors=args.num_nearest_neighbors,
                    base_state_dict=base_state,
                    seed=args.seed,
                    use_amp=use_amp,
                    mu=mu_val,
                )
            result["run_label"] = label
            result["mu"] = mu_val
            results.append(result)
            print(
                f"T1@T1={result['task1_after_task1']:.3f} "
                f"T1@T2={result['task1_after_task2']:.3f} "
                f"T2={result['task2_final_nmae']:.3f} "
                f"abs_forget={result['absolute_forgetting']:.3f} "
                f"bwt={result['bwt']:.3f} "
                f"avg_final={result['average_final_nmae']:.3f} "
                f"trainable={result['trainable_params']:,} "
                f"stored={result['stored_params']:,} "
                f"time={result['wall_time_seconds']:.1f}s"
            )
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "method": method,
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    # Save results (drop heavy state dicts).
    summary_path = output_dir / ("mu_tuning_results.json" if args.mu_grid else "screening_results.json")
    json_safe_results = []
    for r in results:
        r_safe = {k: v for k, v in r.items() if k != "task1_final_state_dict"}
        json_safe_results.append(r_safe)
    with open(summary_path, "w") as f:
        json.dump(json_safe_results, f, indent=2)
    print(f"\nSaved results to {summary_path}")

    # Paired recheck assertions.
    if args.paired_recheck:
        mu0 = next((r for r in results if r["method"] == "phytca_no_stability"), None)
        mu_pos = next((r for r in results if r["method"] == "phytca"), None)
        print("\n=== Paired recheck assertions ===")
        if mu0 is None or mu_pos is None:
            print("NO_GO_MISSING_PAIRED_METHODS")
            return
        try:
            _assert_task1_trajectory_identity(mu0, mu_pos)
            print("PASS: PhyTCA mu=0 and mu=0.01 have identical Task-1 trajectories.")
            print("Decision: GO_TO_PAIRED_2K_RECHECK")
        except AssertionError as e:
            print(f"FAIL: {e}")
            print("Decision: NO_GO_TASK1_TRAJECTORY_MISMATCH")
        return

    # Mu-grid assessment.
    if args.mu_grid:
        print("\n=== Mu-grid assessment ===")
        seq = next((r for r in results if r["method"] == "sequential"), None)
        phytca_runs = [r for r in results if r["method"] == "phytca" and r.get("status") == "ok"]
        if seq is None or not phytca_runs:
            print("NO_GO_MISSING_BASELINES")
            return
        if seq.get("status") != "ok":
            print("NO_GO_BASELINE_ERROR")
            return

        print(f"Sequential FT: abs_forget={seq['absolute_forgetting']:.4f} avg_final={seq['average_final_nmae']:.4f}")
        for r in sorted(phytca_runs, key=lambda x: x.get("mu", -1)):
            print(
                f"  mu={r.get('mu')}: "
                f"abs_forget={r['absolute_forgetting']:.4f} "
                f"T2={r['task2_final_nmae']:.4f} "
                f"avg_final={r['average_final_nmae']:.4f}"
            )

        best = min(
            phytca_runs,
            key=lambda r: (r["absolute_forgetting"], r["average_final_nmae"]),
        )
        print(
            f"\nBest mu on continual_dev: {best.get('mu')} "
            f"(abs_forget={best['absolute_forgetting']:.4f}, "
            f"avg_final={best['average_final_nmae']:.4f})"
        )
        if best["absolute_forgetting"] < seq["absolute_forgetting"] and best["task2_final_nmae"] <= seq["task2_final_nmae"] * 1.10:
            print(f"Decision: GO_TO_PHASE0_5K_3SEEDS_WITH_MU_{best.get('mu')}")
        else:
            print("Decision: NO_GO_PHYTCA_NO_ADVANTAGE_AT_2K")
        return

    # GO/NO-GO assessment.
    print("\n=== GO/NO-GO assessment ===")
    seq = next((r for r in results if r["method"] == "sequential"), None)
    phytca = next((r for r in results if r["method"] == "phytca"), None)
    shared_lora = next((r for r in results if r["method"] == "shared_lora"), None)
    replay = next((r for r in results if r["method"] == "replay_1pct"), None)

    if seq is None or phytca is None:
        print("NO_GO_MISSING_BASELINES")
        return

    if seq.get("status") != "ok" or phytca.get("status") != "ok":
        print("NO_GO_BASELINE_ERROR")
        return

    a_pass = seq["absolute_forgetting"] > 0.005
    b_pass = phytca["absolute_forgetting"] < seq["absolute_forgetting"]
    c_pass = phytca["task2_final_nmae"] <= seq["task2_final_nmae"] * 1.10
    d_pass = False
    if shared_lora and shared_lora.get("status") == "ok":
        if phytca["absolute_forgetting"] < shared_lora["absolute_forgetting"]:
            d_pass = True
        if phytca["average_final_nmae"] < shared_lora["average_final_nmae"]:
            d_pass = True
        if phytca["stored_params"] < shared_lora["stored_params"]:
            d_pass = True
        if phytca["trainable_params"] < shared_lora["trainable_params"]:
            d_pass = True
    if replay and replay.get("status") == "ok":
        if phytca["absolute_forgetting"] < replay["absolute_forgetting"]:
            d_pass = True
        if phytca["average_final_nmae"] < replay["average_final_nmae"]:
            d_pass = True
        if phytca["stored_params"] < replay["stored_params"]:
            d_pass = True
        if phytca["trainable_params"] < replay["trainable_params"]:
            d_pass = True

    print(f"A (sequential forgetting > 0.005): {a_pass} ({seq['absolute_forgetting']:.4f})")
    print(f"B (phytca forget < sequential forget): {b_pass}")
    print(f"C (phytca T2 <= 1.10x sequential T2): {c_pass}")
    print(f"D (phytca advantage over shared_lora/replay): {d_pass}")

    if not a_pass:
        print("\nDecision: NO_GO_PROTOCOL_B_NO_FORGETTING_SIGNAL")
    elif not (b_pass and c_pass and d_pass):
        print("\nDecision: NO_GO_PHYTCA_NO_ADVANTAGE_AT_2K")
    else:
        print("\nDecision: GO_TO_PHASE0_5K_3SEEDS")


if __name__ == "__main__":
    main()
