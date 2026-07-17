"""Pareto evaluation harness for backward-compatible model serving.

Provides reusable metrics for accuracy/retention/cost/utility trade-offs:
calibration error, inference latency, FLOP estimate, checkpoint size, top-k
recall, and Pareto-front extraction.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


class CalibrationError:
    """Regression Expected Calibration Error (ECE).

    Bins predictions by their predicted value and reports the weighted average
    absolute error within each bin.  A well-calibrated regressor has small ECE.
    """

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        """Return ECE and per-bin statistics."""
        pred = pred.detach().cpu().flatten()
        target = target.detach().cpu().flatten()
        if pred.numel() == 0:
            return {"ece": float("nan"), "max_cal_error": float("nan")}

        sorted_pred, order = torch.sort(pred)
        sorted_target = target[order]
        n = pred.numel()
        bin_size = max(1, n // self.n_bins)

        ece = 0.0
        max_err = 0.0
        total_weight = 0
        for start in range(0, n, bin_size):
            end = min(start + bin_size, n)
            if start >= n:
                break
            p_bin = sorted_pred[start:end]
            t_bin = sorted_target[start:end]
            weight = p_bin.numel()
            err = torch.abs(p_bin - t_bin).mean().item()
            ece += weight * err
            max_err = max(max_err, err)
            total_weight += weight

        return {
            "ece": ece / max(total_weight, 1),
            "max_cal_error": max_err,
        }


class LatencyMeter:
    """Measure average per-batch inference latency with warm-up."""

    def __init__(self, warmup: int = 3, repeats: int = 10) -> None:
        self.warmup = warmup
        self.repeats = repeats

    @torch.no_grad()
    def measure(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
        forward_args: tuple = (),
        forward_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Return mean and std of per-batch latency in milliseconds."""
        model.eval()
        latencies: list[float] = []
        forward_kwargs = forward_kwargs or {}

        for batch in loader:
            node_feats, coords, mask, original_mask, _ = batch
            node_feats = node_feats.to(device)
            coords = coords.to(device)
            mask = mask.to(device)
            original_mask = original_mask.to(device)
            args = (node_feats, coords, mask, original_mask) + forward_args

            # Warm-up
            for _ in range(self.warmup):
                _ = model(*args, **forward_kwargs)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            # Timed repeats on the same batch
            for _ in range(self.repeats):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(*args, **forward_kwargs)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000.0)

            # Only need one batch for latency characterization.
            break

        if not latencies:
            return {"latency_ms_mean": float("nan"), "latency_ms_std": float("nan")}

        lat_t = torch.tensor(latencies)
        return {
            "latency_ms_mean": float(lat_t.mean()),
            "latency_ms_std": float(lat_t.std()),
        }


class FLOPCounter:
    """Crude FLOP estimate via parameter count heuristics.

    For a linear layer ``y = x @ W^T + b`` we count ``2 * in_features *
    out_features * batch`` FLOPs.  This is an order-of-magnitude proxy; EGNN
    message-passing FLOPs are not captured precisely.
    """

    def __init__(self) -> None:
        self.total = 0

    def _hook(self, module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            batch = output.numel() // out_features
            self.total += 2 * in_features * out_features * batch

    def count_model(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
        forward_args: tuple = (),
        forward_kwargs: dict[str, Any] | None = None,
    ) -> int:
        """Return estimated forward FLOPs for one batch."""
        forward_kwargs = forward_kwargs or {}
        handles: list[Any] = []
        for m in model.modules():
            if isinstance(m, nn.Linear):
                handles.append(m.register_forward_hook(self._hook))

        model.eval()
        self.total = 0
        with torch.no_grad():
            for batch in loader:
                node_feats, coords, mask, original_mask, _ = batch
                node_feats = node_feats.to(device)
                coords = coords.to(device)
                mask = mask.to(device)
                original_mask = original_mask.to(device)
                _ = model(node_feats, coords, mask, original_mask, *forward_args, **forward_kwargs)
                break

        for h in handles:
            h.remove()
        return self.total


class CheckpointSize:
    """Estimate checkpoint size in bytes."""

    def __init__(self) -> None:
        pass

    def model_state_bytes(self, model: nn.Module) -> int:
        return sum(_tensor_bytes(p) for p in model.state_dict().values())

    def optimizer_state_bytes(self, optimizer: torch.optim.Optimizer) -> int:
        size = 0
        for state in optimizer.state.values():
            for v in state.values():
                if isinstance(v, torch.Tensor):
                    size += _tensor_bytes(v)
        return size

    def __call__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        extra_buffers: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, int]:
        model_bytes = self.model_state_bytes(model)
        opt_bytes = self.optimizer_state_bytes(optimizer) if optimizer is not None else 0
        extra_bytes = sum(_tensor_bytes(t) for t in (extra_buffers or {}).values())
        return {
            "model_bytes": model_bytes,
            "optimizer_bytes": opt_bytes,
            "extra_bytes": extra_bytes,
            "total_bytes": model_bytes + opt_bytes + extra_bytes,
        }


def evaluate_pareto_metrics(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    forward_args: tuple = (),
    forward_kwargs: dict[str, Any] | None = None,
    n_bins: int = 10,
    latency_warmup: int = 3,
    latency_repeats: int = 10,
) -> dict[str, float]:
    """Compute a bundle of accuracy/cost metrics for one endpoint."""
    forward_kwargs = forward_kwargs or {}
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for node_feats, coords, mask, original_mask, y in loader:
            node_feats = node_feats.to(device)
            coords = coords.to(device)
            mask = mask.to(device)
            original_mask = original_mask.to(device)
            pred = model(node_feats, coords, mask, original_mask, *forward_args, **forward_kwargs)
            preds.append(pred.cpu())
            targets.append(y)
    preds_t = torch.cat(preds)
    targets_t = torch.cat(targets)

    cal_metrics = CalibrationError(n_bins=n_bins)(preds_t, targets_t)
    latency_metrics = LatencyMeter(warmup=latency_warmup, repeats=latency_repeats).measure(
        model, loader, device, forward_args, forward_kwargs
    )
    flops = FLOPCounter().count_model(model, loader, device, forward_args, forward_kwargs)
    checkpoint_metrics = CheckpointSize()(model, optimizer)

    return {
        "calibration_ece": cal_metrics["ece"],
        "calibration_max_cal_error": cal_metrics["max_cal_error"],
        **latency_metrics,
        "estimated_flops": flops,
        **{f"checkpoint_{k}": v for k, v in checkpoint_metrics.items()},
    }


class TopKRecall:
    """Top-k recall for same-material retrieval across endpoints.

    Given pooled crystal embeddings and material IDs, compute for each record
    the fraction of same-material records that appear in the top-k nearest
    neighbors (by cosine similarity).
    """

    def __init__(self, k: int = 5) -> None:
        self.k = k

    def __call__(
        self,
        embeddings: torch.Tensor,
        material_ids: Sequence[str],
    ) -> dict[str, float]:
        if embeddings.numel() == 0 or len(material_ids) == 0:
            return {f"recall@{self.k}": float("nan")}

        embeddings = F.normalize(embeddings, dim=-1)
        sim = embeddings @ embeddings.t()
        # Exclude self from neighbors.
        sim.fill_diagonal_(-float("inf"))
        topk = torch.topk(sim, k=min(self.k, sim.size(0) - 1), dim=1).indices

        correct = 0
        total = 0
        for i, query_id in enumerate(material_ids):
            neighbors = [material_ids[j] for j in topk[i].tolist()]
            same = sum(1 for nid in neighbors if nid == query_id)
            # Count all other records with the same ID as potential positives.
            n_positives = sum(1 for mid in material_ids if mid == query_id) - 1
            if n_positives > 0:
                correct += same / n_positives
                total += 1

        return {f"recall@{self.k}": correct / max(total, 1)}


class ParetoFront:
    """Extract non-dominated points given multiple objectives.

    Objectives are assumed to be minimization targets (e.g. error, latency,
    parameter count).  A point dominates another if it is no worse in every
    objective and strictly better in at least one.
    """

    def __init__(self, objectives: Sequence[str]) -> None:
        self.objectives = objectives

    def __call__(self, points: list[dict[str, float]]) -> list[dict[str, float]]:
        """Return the non-dominated subset of ``points``."""
        nondominated: list[dict[str, float]] = []
        for p in points:
            dominated = False
            to_remove: list[dict[str, float]] = []
            for q in nondominated:
                p_better = all(p[o] <= q[o] for o in self.objectives)
                q_better = all(q[o] <= p[o] for o in self.objectives)
                if p_better and any(p[o] < q[o] for o in self.objectives):
                    to_remove.append(q)
                elif q_better and any(q[o] < p[o] for o in self.objectives):
                    dominated = True
                    break
            if not dominated:
                nondominated = [q for q in nondominated if q not in to_remove]
                nondominated.append(p)
        return nondominated
