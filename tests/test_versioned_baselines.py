"""Tests for versioned-protocol baselines."""

from __future__ import annotations

import pytest
import torch

from models import CopyOnWriteFullChildModel
from scripts.run_versioned_baselines import METHODS, run_versioned_baselines


def test_copy_on_write_full_child_retains_published_routes():
    """Training a new child does not change a published child's predictions."""
    model = CopyOnWriteFullChildModel(
        node_dim=92, hidden_dim=16, n_layers=2, num_nearest_neighbors=4
    )
    model.add_route("2021", prop_id=0, fid_id=0)

    x = torch.randn(2, 5, 92)
    coords = torch.randn(2, 5, 3)
    mask = torch.ones(2, 5, dtype=torch.bool)
    original_mask = torch.ones(2, 5, dtype=torch.bool)

    # Train the first route.
    optimizer = torch.optim.AdamW(model.current_trainable_parameters(), lr=1e-2)
    target = torch.randn(2)
    for _ in range(5):
        optimizer.zero_grad()
        pred = model(x, coords, mask, original_mask, "2021", 0, 0)
        loss = (pred - target).pow(2).sum()
        loss.backward()
        optimizer.step()

    model.freeze_route("2021", 0, 0)

    # Snapshot predictions for the published route before training the next one.
    before = model(x, coords, mask, original_mask, "2021", 0, 0).detach().clone()

    model.add_route("2022", prop_id=0, fid_id=0)

    # Train only the new route.
    optimizer = torch.optim.AdamW(model.current_trainable_parameters(), lr=1e-2)
    target_new = torch.randn(2)
    for _ in range(5):
        optimizer.zero_grad()
        pred = model(x, coords, mask, original_mask, "2022", 0, 0)
        loss = (pred - target_new).pow(2).sum()
        loss.backward()
        optimizer.step()

    after = model(x, coords, mask, original_mask, "2021", 0, 0)
    assert torch.allclose(before, after, atol=1e-6)


def test_copy_on_write_full_child_increments_parameters():
    """Each new child adds a full encoder worth of parameters."""
    model = CopyOnWriteFullChildModel(
        node_dim=92, hidden_dim=16, n_layers=2, num_nearest_neighbors=4
    )
    model.add_route("2021", 0, 0)
    p1 = model.total_parameters()
    model.add_route("2022", 0, 0)
    p2 = model.total_parameters()
    assert p2 > p1
    # The incremental cost is roughly one full encoder plus one head.
    inc = model.incremental_parameters("2022", 0, 0)
    assert inc > 0


@pytest.mark.slow
@pytest.mark.parametrize("method", list(METHODS.keys()))
def test_versioned_baselines_smoke(method: str, tmp_path) -> None:
    """Each baseline runs on a tiny capped benchmark."""
    results = run_versioned_baselines(
        snapshots=["dft_3d_2021"],
        properties=["band_gap"],
        fidelities=["OptB88vdW", "TB-mBJ"],
        methods=[method],
        hidden_dim=16,
        rank=4,
        n_layers=2,
        num_nearest_neighbors=4,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        weight_decay=1e-4,
        patience=1,
        device=torch.device("cpu"),
        seed=42,
        cap=8,
        output_dir=tmp_path,
    )
    assert method in results["methods"]
    method_results = results["methods"][method]
    assert "nmaes" in method_results
    assert "total_parameters" in method_results
    assert method_results["total_parameters"] > 0
    assert (tmp_path / "baseline_results.json").exists()
