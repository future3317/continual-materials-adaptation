"""Tests for PersistentConsolidationGraph core mechanics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from persistent_consolidation_graph import (
    BasisBank,
    EndpointRegistry,
    FastAdapter,
    PersistentConsolidationGraph,
    RouteSpec,
)


def _make_encoder(hidden_dim: int):
    """Tiny EGNN encoder for tests."""
    from models import CrystalEncoder

    return CrystalEncoder(
        node_dim=92,
        hidden_dim=hidden_dim,
        n_layers=1,
        num_nearest_neighbors=4,
        update_coors=False,
    )


def test_basis_bank_append_only_isolates_old_routes():
    dim, rank = 16, 4
    bank = BasisBank(dim, default_rank=rank)

    # Two routes share the initial empty bank; add a block for route 1.
    block_id = bank.add_block()
    route1 = RouteSpec("r1", [], [block_id], dim, (0.0, 1.0))
    route1.set_coefficient_shape(block_id, rank)

    x = torch.randn(2, 8, dim)
    coeffs1 = {block_id: route1.private_coefficients[block_id]}
    before = bank(x, [block_id], coeffs1).clone()

    # Add a second block and train route 2.
    block2 = bank.add_block()
    route2 = RouteSpec("r2", [], [block_id, block2], dim, (0.0, 1.0))
    route2.set_coefficient_shape(block_id, rank)
    route2.set_coefficient_shape(block2, rank)
    bank.freeze_block(block_id)
    coeffs2 = {bid: route2.private_coefficients[bid] for bid in route2.basis_block_ids}

    optimizer = torch.optim.Adam(bank.parameters(), lr=1e-2)
    for _ in range(10):
        optimizer.zero_grad()
        loss = bank(x, route2.basis_block_ids, coeffs2).pow(2).sum()
        loss.backward()
        optimizer.step()

    after = bank(x, [block_id], coeffs1)
    assert torch.allclose(before, after, atol=1e-6)


def test_fast_adapter_full_update_shape():
    dim, rank = 16, 4
    adapter = FastAdapter(dim, rank)
    assert adapter.full_update().shape == (dim, dim)


def test_pcg_adds_routes_and_forgets_none():
    hidden_dim, rank = 16, 4
    encoder = _make_encoder(hidden_dim)
    model = PersistentConsolidationGraph(encoder, hidden_dim, rank=rank)

    model.add_route("v1", prop_id=0, fid_id=0)
    x = torch.randn(2, 5, 92)
    coords = torch.randn(2, 5, 3)
    mask = torch.ones(2, 5, dtype=torch.bool)
    original_mask = torch.ones(2, 5, dtype=torch.bool)

    before = model(x, coords, mask, original_mask, "v1", 0, 0).detach().clone()

    model.publish_route("v1", 0, 0)
    model.add_route("v2", prop_id=0, fid_id=0)

    # Train the new route.
    optimizer = torch.optim.Adam(model.current_trainable_parameters(), lr=1e-2)
    for _ in range(5):
        optimizer.zero_grad()
        pred = model(x, coords, mask, original_mask, "v2", 0, 0)
        loss = pred.pow(2).sum()
        loss.backward()
        optimizer.step()

    after = model(x, coords, mask, original_mask, "v1", 0, 0)
    assert torch.allclose(before, after, atol=1e-6)


def test_endpoint_registry_detects_modification():
    route = RouteSpec("e1", [], [], 8, (0.0, 1.0))
    route.head.bias.data.fill_(1.0)
    registry = EndpointRegistry()
    registry.register(route)
    registry.publish("e1", torch.nn.Linear(8, 8), "g", "d")

    registry.assert_all_published_unchanged()

    route.head.bias.data.fill_(2.0)
    try:
        registry.assert_all_published_unchanged()
        raise AssertionError("Expected RuntimeError on modified head")
    except RuntimeError:
        pass


def test_parent_gate_reuses_parent_prediction():
    """A child endpoint with a parent should be able to learn a small residual."""
    hidden_dim, rank = 16, 4
    encoder = _make_encoder(hidden_dim)
    model = PersistentConsolidationGraph(encoder, hidden_dim, rank=rank)

    # Parent route: predict a constant.
    model.add_route("1", prop_id=0, fid_id=0, normalizer=(0.0, 1.0))
    parent_route = model.registry.routes["v1_p0_f0"]
    parent_route.head.bias.data.fill_(2.0)
    model.publish_route("1", 0, 0)

    # Child route: parent predicts 2.0; child target is 2.5.
    model.add_route("2", prop_id=0, fid_id=0, parent_ids=["v1_p0_f0"], normalizer=(0.0, 1.0))

    x = torch.randn(4, 5, 92)
    coords = torch.randn(4, 5, 3)
    mask = torch.ones(4, 5, dtype=torch.bool)
    original_mask = torch.ones(4, 5, dtype=torch.bool)

    # Before training child head is zero, so child prediction equals parent.
    pred_before = model(x, coords, mask, original_mask, "2", 0, 0, physical=True)
    assert torch.allclose(pred_before, torch.full_like(pred_before, 2.0), atol=1e-5)

    # Train child head to output residual 0.5.
    optimizer = torch.optim.Adam(model.current_trainable_parameters(), lr=1e-1)
    for _ in range(100):
        optimizer.zero_grad()
        pred = model(x, coords, mask, original_mask, "2", 0, 0, physical=True)
        loss = F.mse_loss(pred, torch.full_like(pred, 2.5))
        loss.backward()
        optimizer.step()

    pred_after = model(x, coords, mask, original_mask, "2", 0, 0, physical=True)
    assert torch.allclose(pred_after, torch.full_like(pred_after, 2.5), atol=5e-2)

    # Parent prediction must remain unchanged.
    pred_parent = model(x, coords, mask, original_mask, "1", 0, 0, physical=True)
    assert torch.allclose(pred_parent, torch.full_like(pred_parent, 2.0), atol=1e-5)


def test_middle_matrix_uses_full_matrix_not_diagonal():
    """An off-diagonal entry in the middle matrix must change the output."""
    dim, rank = 16, 4
    bank = BasisBank(dim, default_rank=rank)
    block_id = bank.add_block()
    route = RouteSpec("r1", [], [block_id], dim, (0.0, 1.0))
    route.set_coefficient_shape(block_id, rank)

    x = torch.randn(2, 8, dim)
    m = route.private_coefficients[block_id]
    m_init = m.detach().clone()

    out_before = bank(x, [block_id], {block_id: m_init})
    # Perturb an off-diagonal entry.
    m_changed = m_init.clone()
    m_changed[0, 1] += 1.0
    out_after = bank(x, [block_id], {block_id: m_changed})

    assert not torch.allclose(out_before, out_after, atol=1e-6)


def test_svd_block_reconstructs_fast_update():
    """A newly appended block should reconstruct the residual update."""
    dim, rank = 16, 8
    bank = BasisBank(dim, default_rank=rank)

    # Build a known low-rank update.
    a = torch.randn(dim, rank)
    b = torch.randn(rank, dim)
    delta_w = a @ b

    selected, new_ids = bank.reuse_or_expand(delta_w, novelty_threshold=0.0, svd_energy=0.9999)
    assert len(new_ids) == 1
    assert len(selected) == 0

    block = bank.blocks[new_ids[0]]
    reconstructed = block.u_in @ block.u_out.T
    # With high SVD energy the reconstruction should be close.
    assert torch.allclose(reconstructed, delta_w, atol=1e-3)


def test_new_block_id_appears_exactly_once():
    """Expansion must not duplicate the new block in the selected IDs."""
    dim, rank = 16, 4
    bank = BasisBank(dim, default_rank=rank)
    bank.add_block()

    delta_w = torch.randn(dim, dim)
    selected, new_ids = bank.reuse_or_expand(delta_w, novelty_threshold=0.0)
    assert len(new_ids) == 1
    # The new ID should appear only in new_ids, not twice in selected+new_ids.
    combined = selected + new_ids
    assert combined.count(new_ids[0]) == 1


def test_child_parent_prediction_equals_published_parent():
    """A child's parent prediction must equal querying the parent directly."""
    hidden_dim, rank = 16, 4
    encoder = _make_encoder(hidden_dim)
    model = PersistentConsolidationGraph(encoder, hidden_dim, rank=rank)

    model.add_route("1", prop_id=0, fid_id=0, normalizer=(0.0, 1.0))
    parent_route = model.registry.routes["v1_p0_f0"]
    parent_route.head.bias.data.fill_(2.0)
    model.publish_route("1", 0, 0)

    model.add_route("2", prop_id=0, fid_id=0, parent_ids=["v1_p0_f0"], normalizer=(0.0, 1.0))

    x = torch.randn(4, 5, 92)
    coords = torch.randn(4, 5, 3)
    mask = torch.ones(4, 5, dtype=torch.bool)
    original_mask = torch.ones(4, 5, dtype=torch.bool)

    # Direct parent prediction.
    parent_direct = model(x, coords, mask, original_mask, "1", 0, 0, physical=True)

    # Child's internal parent prediction (before child residual) must match.
    h = model._encode(x, coords, mask)
    parent_from_child = model._parent_predictions_h(h, original_mask, "v2_p0_f0", physical=True)

    assert torch.allclose(parent_direct, parent_from_child, atol=1e-5)


def test_encoder_is_frozen():
    """PCG must freeze the encoder in its constructor."""
    hidden_dim, rank = 16, 4
    encoder = _make_encoder(hidden_dim)
    model = PersistentConsolidationGraph(encoder, hidden_dim, rank=rank)
    assert all(not p.requires_grad for p in encoder.parameters())
    assert all(not p.requires_grad for p in model.encoder.parameters())
