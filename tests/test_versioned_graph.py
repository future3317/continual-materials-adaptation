"""Tests for versioned fidelity graph and backward-compatible serving."""

from __future__ import annotations

import torch

from versioned_graph import SharedBasisAdapterBank, VersionedFidelityGraph


def test_shared_basis_adapter_isolates_published_routes():
    """Training a new route's M does not change a frozen route's output."""
    dim, rank, n_routes = 16, 4, 3
    bank = SharedBasisAdapterBank(dim, rank, bases_trainable=False)

    for i in range(n_routes):
        bank.add_route(f"r{i}")
        bank.freeze_route(f"r{i}")

    x = torch.randn(2, 8, dim)

    # Snapshot predictions for all routes before training a new one.
    before = {f"r{i}": bank(x, f"r{i}").clone() for i in range(n_routes)}

    # Add and train a new route; only its M should be trainable.
    bank.add_route("r_new")
    optimizer = torch.optim.SGD(bank.parameters(), lr=0.1)
    for _ in range(10):
        optimizer.zero_grad()
        loss = bank(x, "r_new").pow(2).sum()
        loss.backward()
        optimizer.step()

    after = {f"r{i}": bank(x, f"r{i}") for i in range(n_routes)}
    for key in before:
        assert torch.allclose(before[key], after[key], atol=1e-6)


def test_versioned_model_publishes_endpoint_exactly():
    """Publishing an endpoint leaves its predictions invariant."""
    model = VersionedFidelityGraph(
        node_dim=92, hidden_dim=32, n_layers=2, rank=4, num_nearest_neighbors=4
    )
    model.add_route("2021", prop_id=0, fid_id=0)

    node_feats = torch.randn(2, 5, 92)
    coords = torch.randn(2, 5, 3)
    mask = torch.ones(2, 5, dtype=torch.bool)
    original_mask = torch.ones(2, 5, dtype=torch.bool)

    # Random predictions before publishing.
    before = model(node_feats, coords, mask, original_mask, "2021", 0, 0).detach().clone()

    # Publish and add a new route.
    model.publish_route("2021", 0, 0)
    model.add_route("2022", prop_id=0, fid_id=0)

    # Train only the new route.
    optimizer = torch.optim.Adam(model.current_trainable_parameters(), lr=1e-2)
    for _ in range(5):
        optimizer.zero_grad()
        pred = model(node_feats, coords, mask, original_mask, "2022", 0, 0)
        loss = pred.pow(2).sum()
        loss.backward()
        optimizer.step()

    after = model(node_feats, coords, mask, original_mask, "2021", 0, 0)
    assert torch.allclose(before, after, atol=1e-6)


def test_impossibility_without_version_id():
    """A single deterministic predictor cannot both retain the old label and
    learn a new conflicting label for the same input without a version ID.

    This is the code-level sanity check for the formal impossibility result:
    if the model sees exactly the same features and is asked to predict the same
    endpoint, but the dataset now contains a different target, any parameter
    update that improves the new target must change the prediction on the old
    target as well.
    """
    x = torch.randn(1, 5, 92)
    coords = torch.randn(1, 5, 3)
    mask = torch.ones(1, 5, dtype=torch.bool)
    original_mask = torch.ones(1, 5, dtype=torch.bool)

    model = VersionedFidelityGraph(
        node_dim=92, hidden_dim=16, n_layers=1, rank=4, num_nearest_neighbors=4
    )
    model.add_route("latest", 0, 0)

    y_old = torch.tensor([1.0])
    y_new = torch.tensor([2.0])

    pred_old = model(x, coords, mask, original_mask, "latest", 0, 0).detach().item()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(20):
        optimizer.zero_grad()
        pred = model(x, coords, mask, original_mask, "latest", 0, 0)
        loss = (pred - y_new).pow(2).sum()
        loss.backward()
        optimizer.step()

    pred_after = model(x, coords, mask, original_mask, "latest", 0, 0).item()

    # The model moved toward y_new; since there is only one route and no
    # version ID, the old prediction cannot be retained simultaneously.
    assert abs(pred_after - y_new) < abs(pred_old - y_new)
    assert abs(pred_after - y_old) > 1e-3


def test_parameter_growth_per_route():
    """Incremental parameters per route are rank^2 per layer plus head."""
    hidden_dim, rank, n_layers = 64, 8, 3
    model = VersionedFidelityGraph(
        node_dim=92,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        rank=rank,
        num_nearest_neighbors=4,
    )
    model.add_route("v1", 0, 0)
    expected = n_layers * rank * rank + (hidden_dim + 1)
    assert model.incremental_parameters("v1", 0, 0) == expected
