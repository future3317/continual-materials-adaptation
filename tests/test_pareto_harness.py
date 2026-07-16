"""Tests for the Pareto evaluation harness."""

from __future__ import annotations

import torch

from pareto_harness import (
    CalibrationError,
    CheckpointSize,
    FLOPCounter,
    LatencyMeter,
    ParetoFront,
    TopKRecall,
)


def test_calibration_error_perfectly_calibrated():
    """ECE is zero when predictions equal targets."""
    pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    target = pred.clone()
    cal = CalibrationError(n_bins=2)
    metrics = cal(pred, target)
    assert metrics["ece"] == 0.0
    assert metrics["max_cal_error"] == 0.0


def test_calibration_error_detects_bias():
    """ECE is positive when predictions are shifted from targets."""
    pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    target = pred + 1.0
    cal = CalibrationError(n_bins=2)
    metrics = cal(pred, target)
    assert metrics["ece"] > 0.0
    assert metrics["max_cal_error"] == 1.0


class _DummyCrystalModel(torch.nn.Module):
    """Dummy model matching the crystal forward signature."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, node_feats, coords, mask, original_mask, *args):
        return self.module(node_feats)


def test_latency_meter_runs_on_simple_model():
    """LatencyMeter returns finite timings for a tiny model."""
    model = _DummyCrystalModel(torch.nn.Linear(3, 1))
    loader = torch.utils.data.DataLoader(
        [(torch.randn(3), torch.randn(3), torch.ones(3, dtype=torch.bool),
          torch.ones(3, dtype=torch.bool), torch.tensor(0.0))],
        batch_size=1,
    )
    meter = LatencyMeter(warmup=1, repeats=3)
    metrics = meter.measure(model, loader, torch.device("cpu"))
    assert metrics["latency_ms_mean"] >= 0.0
    assert metrics["latency_ms_std"] >= 0.0


def test_flop_counter_counts_linear_layers():
    """FLOPCounter estimates FLOPs for a simple MLP."""
    model = _DummyCrystalModel(torch.nn.Sequential(
        torch.nn.Linear(10, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 1),
    ))
    loader = torch.utils.data.DataLoader(
        [(torch.randn(10), torch.randn(3), torch.ones(5, dtype=torch.bool),
          torch.ones(5, dtype=torch.bool), torch.tensor(0.0))],
        batch_size=1,
    )
    counter = FLOPCounter()
    flops = counter.count_model(model, loader, torch.device("cpu"))
    # Two linear layers: 2*10*8 + 2*8*1 = 176 FLOPs for batch size 1.
    assert flops == 176


def test_checkpoint_size_accounts_for_optimizer():
    """CheckpointSize reports model + optimizer bytes."""
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Take one step to populate optimizer state.
    y = model(torch.randn(1, 10))
    y.backward()
    optimizer.step()
    size = CheckpointSize()(model, optimizer)
    assert size["model_bytes"] > 0
    assert size["optimizer_bytes"] > 0
    assert size["total_bytes"] == size["model_bytes"] + size["optimizer_bytes"]


def test_top_k_recall_perfect_for_duplicates():
    """All same-material duplicates are recalled."""
    embeddings = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
        [0.8, 0.2, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.9, 0.1, 0.0],
        [0.0, 0.8, 0.2, 0.0],
    ])
    material_ids = ["A", "A", "A", "B", "B", "B"]
    recall = TopKRecall(k=2)(embeddings, material_ids)
    assert recall["recall@2"] == 1.0


def test_top_k_recall_zero_for_uniques():
    """No same-material neighbors exist for unique IDs."""
    embeddings = torch.randn(4, 4)
    material_ids = ["A", "B", "C", "D"]
    recall = TopKRecall(k=1)(embeddings, material_ids)
    assert recall["recall@1"] == 0.0


def test_pareto_front_extracts_nondominated_points():
    """Only non-dominated points remain."""
    points = [
        {"error": 1.0, "params": 10.0},
        {"error": 0.9, "params": 9.0},
        {"error": 1.1, "params": 11.0},
        {"error": 0.5, "params": 5.0},
    ]
    front = ParetoFront(["error", "params"])(points)
    assert len(front) == 1
    assert front[0]["error"] == 0.5
    assert front[0]["params"] == 5.0
