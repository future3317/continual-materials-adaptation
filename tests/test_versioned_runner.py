"""Integration tests for the versioned benchmark runner."""

from __future__ import annotations

import pytest
import torch

from scripts.run_versioned_protocol import run_versioned_protocol


@pytest.mark.slow
@pytest.mark.parametrize("rank", [4, 8])
def test_versioned_runner_smoke(tmp_path, rank: int) -> None:
    """Run a tiny versioned benchmark and check exact retention."""
    device = torch.device("cpu")
    metrics = run_versioned_protocol(
        snapshots=["dft_3d_2021"],
        properties=["band_gap"],
        fidelities=["OptB88vdW", "TB-mBJ"],
        hidden_dim=16,
        rank=rank,
        n_layers=2,
        num_nearest_neighbors=4,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        weight_decay=1e-4,
        patience=1,
        device=device,
        seed=42,
        cap=8,
        output_dir=tmp_path,
    )

    assert len(metrics["tasks"]) == 2
    assert len(metrics["nmaes"]) == 2
    assert metrics["forgetting"] >= 0.0
    assert metrics["total_parameters"] > 0
    assert all(p > 0 for p in metrics["incremental_parameters"])

    # The metrics file was written.
    assert (tmp_path / "metrics.json").exists()


def test_versioned_runner_no_forgetting_on_published_routes(tmp_path) -> None:
    """Training the second route does not change predictions of the first."""
    device = torch.device("cpu")
    metrics = run_versioned_protocol(
        snapshots=["dft_3d_2021"],
        properties=["band_gap"],
        fidelities=["OptB88vdW", "TB-mBJ"],
        hidden_dim=16,
        rank=4,
        n_layers=2,
        num_nearest_neighbors=4,
        epochs=2,
        batch_size=4,
        lr=1e-3,
        weight_decay=1e-4,
        patience=1,
        device=device,
        seed=42,
        cap=6,
        output_dir=tmp_path,
    )

    # After task 2, task 1 nMAE should not be NaN and forgetting should be tiny.
    assert len(metrics["nmaes"][-1]) == 2
    assert all(x >= 0 for x in metrics["nmaes"][-1])
    # Exact retention by construction implies zero forgetting; allow numerical slack.
    assert metrics["forgetting"] <= 1e-3
