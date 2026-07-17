"""Baseline comparison harness for the PCG three-axis benchmark.

Methods compared on the same protocol/tasks:
* pcg_proposed          - PersistentConsolidationGraph with novelty-gated basis expansion
* pcg_fixed_basis       - PCG that always reuses existing basis blocks (no expansion)
* pcg_always_expand     - PCG that always appends a new basis block
* cow_full_encoder      - Copy-on-write full encoder per endpoint (exact retention)
* per_endpoint_lora     - Shared frozen encoder + private LoRA per endpoint (exact retention)
* per_endpoint_head     - Shared frozen encoder + private head per endpoint (exact retention)
* independent           - Separate full encoder per endpoint, no retention constraint (upper bound)
* joint                 - Shared trainable encoder + heads, no freezing (upper bound)
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import JARVISCrystalDataset, collate_crystals
from persistent_consolidation_graph import PersistentConsolidationGraph
from pcg_runner import (
    build_pcg_encoder_and_graph_builder,
    cap_records,
    determine_parents_combined,
    filter_records_for_encoder,
    run_pcg_protocol,
)
from protocols import build_combined_protocol
from train_utils import (
    backward_transfer,
    compute_mad,
    forgetting,
    normalized_mae,
)


def _name_to_id(names: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(sorted(set(names)))}


def _endpoint_key(version: str, prop_id: int, fid_id: int) -> str:
    return f"v{version}_p{int(prop_id)}_f{int(fid_id)}"


# ---------------------------------------------------------------------------
# Simple encoder + per-endpoint-head baselines
# ---------------------------------------------------------------------------


class _LoRALayer(nn.Module):
    """Private LoRA applied to the pooled representation."""

    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.empty(dim, rank))
        self.b = nn.Parameter(torch.empty(rank, dim))
        nn.init.kaiming_uniform_(self.a, a=5 ** (1.0 / 3))
        nn.init.zeros_(self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.a @ self.b


class _SimpleEncoderHeadBaseline(nn.Module):
    """Shared or private encoder with per-endpoint heads and optional LoRA."""

    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int,
        mode: str,
        rank: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.rank = rank
        self.endpoints: nn.ModuleDict = nn.ModuleDict()
        self._route_order: list[str] = []

        # Shared encoder is frozen by default; joint mode unfreezes it.
        for p in self.encoder.parameters():
            p.requires_grad = False

    def add_endpoint(self, version: str, prop_id: int, fid_id: int) -> str:
        key = _endpoint_key(version, prop_id, fid_id)
        if key in self.endpoints:
            return key

        endpoint: dict[str, nn.Module] = {}
        if self.mode in ("cow_full_encoder", "independent"):
            if self.mode == "cow_full_encoder" and self._route_order:
                endpoint["encoder"] = copy.deepcopy(self.endpoints[self._route_order[-1]]["encoder"])
            else:
                endpoint["encoder"] = copy.deepcopy(self.encoder)
            for p in endpoint["encoder"].parameters():
                p.requires_grad = True
        if self.mode == "per_endpoint_lora":
            endpoint["lora"] = _LoRALayer(self.hidden_dim, self.rank)
        endpoint["head"] = nn.Linear(self.hidden_dim, 1)
        module_dict = nn.ModuleDict(endpoint)
        # New modules are not automatically moved to the parent device; sync here.
        ref = next(self.encoder.parameters())
        module_dict = module_dict.to(device=ref.device, dtype=ref.dtype)
        self.endpoints[key] = module_dict
        self._route_order.append(key)
        return key

    def freeze_endpoint(self, version: str, prop_id: int, fid_id: int) -> None:
        key = _endpoint_key(version, prop_id, fid_id)
        for p in self.endpoints[key].parameters():
            p.requires_grad = False

    def unfreeze_shared_encoder(self) -> None:
        """Used by joint mode so the shared encoder can be trained."""
        for p in self.encoder.parameters():
            p.requires_grad = True

    def current_trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def _encode(self, endpoint: nn.ModuleDict, node_feats: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if "encoder" in endpoint:
            h, _ = endpoint["encoder"].encode(node_feats, coords, mask)
        else:
            h, _ = self.encoder.encode(node_feats, coords, mask)
        return h

    def forward(
        self,
        node_feats: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        original_mask: torch.Tensor,
        version: str,
        prop_id: int,
        fid_id: int,
    ) -> torch.Tensor:
        key = _endpoint_key(version, prop_id, fid_id)
        endpoint = self.endpoints[key]
        h = self._encode(endpoint, node_feats, coords, mask)
        mask_exp = original_mask.unsqueeze(-1).float()
        pooled = (h * mask_exp).sum(dim=1) / (mask_exp.sum(dim=1).clamp_min(1.0))
        if "lora" in endpoint:
            pooled = pooled + endpoint["lora"](pooled)
        return endpoint["head"](pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# Training/evaluation helpers for simple baselines
# ---------------------------------------------------------------------------


def _make_loaders(
    recs: list[dict],
    batch_size: int,
    graph_builder: Any | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, torch.Tensor, float]:
    """Return train/val/test loaders plus normalization stats."""
    train_recs = [r for r in recs if r.get("split") == "train"]
    all_targets = torch.tensor([r["target"] for r in train_recs], dtype=torch.float32)
    target_mean = all_targets.mean()
    target_std = all_targets.std().clamp_min(1e-6)
    mad = compute_mad(all_targets)

    train_ds = JARVISCrystalDataset(recs, split="train", normalize_target=False, graph_builder=graph_builder)
    val_ds = JARVISCrystalDataset(recs, split="val", normalize_target=False, graph_builder=graph_builder)
    test_ds = JARVISCrystalDataset(recs, split="test", normalize_target=False, graph_builder=graph_builder)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_crystals)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals)
    return train_loader, val_loader, test_loader, target_mean, target_std, mad


def _evaluate_endpoint(
    model: nn.Module,
    recs: list[dict],
    version: str,
    prop_id: int,
    fid_id: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    mad: float,
    device: torch.device,
    batch_size: int,
    graph_builder: Any | None = None,
) -> dict[str, float]:
    """Evaluate a single endpoint on its test records."""
    ds = JARVISCrystalDataset(recs, split="test", normalize_target=False, graph_builder=graph_builder)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for node_feats, coords, mask, original_mask, y in loader:
            node_feats = node_feats.to(device)
            coords = coords.to(device)
            mask = mask.to(device)
            original_mask = original_mask.to(device)
            pred_norm = model(node_feats, coords, mask, original_mask, version, prop_id, fid_id)
            pred = pred_norm * std.to(device) + mean.to(device)
            preds.append(pred.cpu())
            targets.append(y)
    if not preds:
        return {"n": 0, "mae": float("nan"), "nmae": float("nan")}
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    mae = float(torch.abs(preds - targets).mean())
    return {"n": len(recs), "mae": mae, "nmae": float(normalized_mae(preds, targets, mad))}


def _evaluate_loader_nmae(
    model: nn.Module,
    loader: DataLoader,
    version: str,
    prop_id: int,
    fid_id: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    mad: float,
    device: torch.device,
) -> float:
    """Evaluate nMAE directly on a loader without building records."""
    if len(loader) == 0:
        return float("nan")
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for node_feats, coords, mask, original_mask, y in loader:
            node_feats = node_feats.to(device)
            coords = coords.to(device)
            mask = mask.to(device)
            original_mask = original_mask.to(device)
            pred_norm = model(node_feats, coords, mask, original_mask, version, prop_id, fid_id)
            pred = pred_norm * std.to(device) + mean.to(device)
            preds.append(pred.cpu())
            targets.append(y)
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    return float(normalized_mae(preds, targets, mad))


def _train_one_endpoint(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    version: str,
    prop_id: int,
    fid_id: int,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    patience: int = 5,
) -> tuple[float, torch.Tensor, torch.Tensor, float]:
    """Train a single endpoint with early stopping."""
    trainable = (
        model.current_trainable_parameters()
        if hasattr(model, "current_trainable_parameters")
        else [p for p in model.parameters() if p.requires_grad]
    )
    if not trainable:
        raise RuntimeError("No trainable parameters")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    all_targets = []
    for _, _, _, _, y in train_loader:
        all_targets.append(y)
    all_targets = torch.cat(all_targets)
    target_mean = all_targets.mean()
    target_std = all_targets.std().clamp_min(1e-6)
    mad = compute_mad(all_targets)

    best_nmae = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
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
            pred = model(node_feats, coords, mask, original_mask, version, prop_id, fid_id)
            loss = F.mse_loss(pred, y_norm)
            loss.backward()
            optimizer.step()

        val_nmae = _evaluate_loader_nmae(
            model, val_loader, version, prop_id, fid_id, target_mean, target_std, mad, device
        )
        if val_nmae != val_nmae:  # NaN
            continue
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
    tasks: list[tuple[str, str, str, str]],
    test_loaders: list[DataLoader],
    task_stats: list[tuple[torch.Tensor, torch.Tensor, float]],
    prop2id: dict[str, int],
    fid2id: dict[str, int],
    device: torch.device,
    t: int | None = None,
) -> list[float]:
    """Evaluate model on test sets of tasks 0..t (or all tasks if t is None)."""
    upper = len(tasks) if t is None else t + 1
    nmaes: list[float] = []
    for prev_t in range(upper):
        version, prop, fid, _ = tasks[prev_t]
        pid = prop2id[prop]
        pfid = fid2id[fid]
        mean, std, mad = task_stats[prev_t]
        nmae = _evaluate_loader_nmae(
            model, test_loaders[prev_t], version, pid, pfid, mean, std, mad, device
        )
        nmaes.append(nmae)
    return nmaes


# ---------------------------------------------------------------------------
# Baseline methods
# ---------------------------------------------------------------------------


def run_pcg_variant(
    tasks: list[tuple[str, str, str, str]],
    task_records: list[list[dict]],
    prop2id: dict[str, int],
    fid2id: dict[str, int],
    encoder: nn.Module,
    graph_builder: Any | None,
    device: torch.device,
    output_dir: Path,
    *,
    novelty_threshold: float,
    epochs_fast: int,
    epochs_cons: int,
    batch_size: int,
    lr: float,
    rank: int,
) -> dict[str, Any]:
    """Run a PCG variant with a fixed novelty threshold."""
    model = PersistentConsolidationGraph(
        encoder, encoder.hidden_dim if hasattr(encoder, "hidden_dim") else 64, rank=rank, novelty_threshold=novelty_threshold
    ).to(device)

    metrics = run_pcg_protocol(
        protocol_name="combined",
        tasks=tasks,
        task_records=task_records,
        model=model,
        prop2id=prop2id,
        fid2id=fid2id,
        device=device,
        batch_size=batch_size,
        epochs_fast=epochs_fast,
        epochs_cons=epochs_cons,
        lr=lr,
        output_dir=output_dir,
        parent_fn=determine_parents_combined,
        graph_builder=graph_builder,
    )
    return metrics


def run_simple_baseline(
    tasks: list[tuple[str, str, str, str]],
    task_records: list[list[dict]],
    prop2id: dict[str, int],
    fid2id: dict[str, int],
    encoder: nn.Module,
    graph_builder: Any | None,
    device: torch.device,
    *,
    mode: str,
    epochs: int,
    batch_size: int,
    lr: float,
    rank: int,
    hidden_dim: int,
) -> dict[str, Any]:
    """Run a simple encoder+head baseline in sequential (frozen) or joint mode."""
    if mode == "cow_full_encoder":
        model = _SimpleEncoderHeadBaseline(encoder, hidden_dim, mode, rank=rank).to(device)
    else:
        model = _SimpleEncoderHeadBaseline(encoder, hidden_dim, mode, rank=rank).to(device)

    task_stats: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    loaders: list[tuple[DataLoader, DataLoader, DataLoader]] = []
    for recs in task_records:
        train_loader, val_loader, test_loader, mean, std, mad = _make_loaders(recs, batch_size, graph_builder)
        task_stats.append((mean, std, mad))
        loaders.append((train_loader, val_loader, test_loader))
    test_loaders = [test_loader for _, _, test_loader in loaders]

    start_all = time.perf_counter()
    nmaes: list[list[float]] = []
    wall_times: list[float] = []

    if mode == "joint":
        model.unfreeze_shared_encoder()
        for t, (version, prop, fid, _) in enumerate(tasks):
            pid = prop2id[prop]
            pfid = fid2id[fid]
            model.add_endpoint(version, pid, pfid)
        trainable = model.current_trainable_parameters()
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for _ in range(epochs):
            model.train()
            for t, (train_loader, _, _) in enumerate(loaders):
                version, prop, fid, _ = tasks[t]
                pid = prop2id[prop]
                pfid = fid2id[fid]
                mean, std, _ = task_stats[t]
                for node_feats, coords, mask, original_mask, y in train_loader:
                    node_feats = node_feats.to(device)
                    coords = coords.to(device)
                    mask = mask.to(device)
                    original_mask = original_mask.to(device)
                    y_norm = ((y.to(device) - mean.to(device)) / std.to(device)).float()

                    optimizer.zero_grad()
                    pred = model(node_feats, coords, mask, original_mask, version, pid, pfid)
                    loss = F.mse_loss(pred, y_norm)
                    loss.backward()
                    optimizer.step()
            scheduler.step()

        final_nmaes = _evaluate_all_seen(
            model, tasks, test_loaders, task_stats, prop2id, fid2id, device
        )
        nmaes.append(final_nmaes)
        wall_times.append(time.perf_counter() - start_all)

    elif mode == "independent":
        for t, ((train_loader, val_loader, _), (version, prop, fid, _)) in enumerate(zip(loaders, tasks)):
            pid = prop2id[prop]
            pfid = fid2id[fid]
            train_start = time.perf_counter()
            model.add_endpoint(version, pid, pfid)
            _train_one_endpoint(model, train_loader, val_loader, version, pid, pfid, device, epochs=epochs, lr=lr)
            wall_times.append(time.perf_counter() - train_start)
            row = [float("nan")] * len(tasks)
            mean, std, mad = task_stats[t]
            row[t] = _evaluate_loader_nmae(
                model, test_loaders[t], version, pid, pfid, mean, std, mad, device
            )
            nmaes.append(row)

    else:
        # Sequential freezing baselines.
        for t, ((train_loader, val_loader, _), (version, prop, fid, _)) in enumerate(zip(loaders, tasks)):
            pid = prop2id[prop]
            pfid = fid2id[fid]
            train_start = time.perf_counter()
            model.add_endpoint(version, pid, pfid)
            _train_one_endpoint(model, train_loader, val_loader, version, pid, pfid, device, epochs=epochs, lr=lr)
            model.freeze_endpoint(version, pid, pfid)
            wall_times.append(time.perf_counter() - train_start)
            nmaes.append(
                _evaluate_all_seen(
                    model, tasks, test_loaders, task_stats, prop2id, fid2id, device, t=t
                )
            )

    total_time = time.perf_counter() - start_all
    final = nmaes[-1] if nmaes else []
    return {
        "method": mode,
        "tasks": [{"version": v, "property": p, "fidelity": f, "target_field": tf} for v, p, f, tf in tasks],
        "nmaes": nmaes,
        "average_final_nmae": sum(final) / len(final) if final else float("nan"),
        "forgetting": forgetting(nmaes),
        "backward_transfer": backward_transfer(nmaes),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "train_wall_times_seconds": wall_times,
        "total_wall_time_seconds": total_time,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="PCG baseline comparison harness")
    parser.add_argument("--properties", nargs="+", default=["band_gap"])
    parser.add_argument("--fidelities", nargs="+", default=["OptB88vdW", "TB-mBJ"])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--encoder-type", choices=["matgl", "egnn"], default="matgl")
    parser.add_argument("--epochs-fast", type=int, default=5)
    parser.add_argument("--epochs-cons", type=int, default=10)
    parser.add_argument("--epochs-baseline", type=int, default=20, help="Epochs for non-PCG baselines")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cap", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/pcg_baselines"))
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "pcg_proposed",
            "pcg_fixed_basis",
            "pcg_always_expand",
            "cow_full_encoder",
            "per_endpoint_lora",
            "per_endpoint_head",
            "independent",
            "joint",
        ],
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    tasks, task_records, audit = build_combined_protocol(
        properties=args.properties,
        fidelities=args.fidelities,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )

    encoder, graph_builder = build_pcg_encoder_and_graph_builder(args.encoder_type, args.hidden_dim)
    task_records = [filter_records_for_encoder(recs, args.encoder_type, encoder) for recs in task_records]

    if args.cap is not None:
        task_records = [cap_records(recs, args.cap) for recs in task_records]

    tasks = [t for t, recs in zip(tasks, task_records) if recs]
    task_records = [recs for recs in task_records if recs]

    prop2id = _name_to_id([p for _, p, _, _ in tasks])
    fid2id = _name_to_id([f for _, _, f, _ in tasks])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {"audit": audit, "methods": {}}

    for method in args.methods:
        print(f"\n{'='*60}\nRunning {method}\n{'='*60}")
        method_dir = args.output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        method_path = method_dir / "metrics.json"

        if method_path.exists():
            print(f"Skipping {method}: {method_path} already exists (resume).")
            with open(method_path, encoding="utf-8") as f:
                results["methods"][method] = json.load(f)
            continue

        if method.startswith("pcg_"):
            novelty_threshold = {
                "pcg_proposed": 0.2,
                "pcg_fixed_basis": 1.0,
                "pcg_always_expand": 0.0,
            }.get(method, 0.2)
            metrics = run_pcg_variant(
                tasks=tasks,
                task_records=task_records,
                prop2id=prop2id,
                fid2id=fid2id,
                encoder=encoder,
                graph_builder=graph_builder,
                device=device,
                output_dir=method_dir,
                novelty_threshold=novelty_threshold,
                epochs_fast=args.epochs_fast,
                epochs_cons=args.epochs_cons,
                batch_size=args.batch_size,
                lr=args.lr,
                rank=args.rank,
            )
        else:
            # Re-instantiate a fresh encoder for each non-PCG method.
            fresh_encoder, _ = build_pcg_encoder_and_graph_builder(args.encoder_type, args.hidden_dim)
            metrics = run_simple_baseline(
                tasks=tasks,
                task_records=task_records,
                prop2id=prop2id,
                fid2id=fid2id,
                encoder=fresh_encoder,
                graph_builder=graph_builder,
                device=device,
                mode=method,
                epochs=args.epochs_baseline,
                batch_size=args.batch_size,
                lr=args.lr,
                rank=args.rank,
                hidden_dim=args.hidden_dim,
            )

        results["methods"][method] = metrics
        with open(method_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved {method_path}")

    summary_path = args.output_dir / "baseline_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
