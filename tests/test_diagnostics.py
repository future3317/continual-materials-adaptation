"""Unit tests for diagnostic model variants and replay accounting.

These tests are designed to be fast and self-contained: they use synthetic
inputs and do not require downloading JARVIS data.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from diagnostics import (
    FrozenOptCorrectionModel,
    ProgressiveAdapterCrystalGraphLayer,
    ProgressivePhyTCAModel,
    ProgressiveTuckerAdapter,
)
from phytca import PhyTCAModel, Tucker4DAdapter
from scripts.run_phase0_b_screening import _detailed_parameter_stats


def _random_model_state(node_dim=5, hidden_dim=8) -> dict[str, torch.Tensor]:
    """Return a tiny PhyTCA state dict for two property/fidelity ids."""
    model = PhyTCAModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=1,
        n_fidelities=2,
        n_layers=2,
        adapter_rank=4,
        num_nearest_neighbors=4,
        freeze_encoder_weights=True,
    )
    return {k: v.clone() for k, v in model.state_dict().items()}


def test_progressive_child_opt_slice_zero():
    """Only the parent OPT slice of the child is zero; MBJ slice is trainable."""
    adapter = ProgressiveTuckerAdapter(d_in=8, d_out=8, n_properties=1, n_fidelities=2, rank=4)
    opt_prop, opt_fid = 0, 0
    adapter.zero_and_freeze_child_slice(opt_prop, opt_fid)

    x = torch.randn(4, 8, requires_grad=True)
    out_opt = adapter(x, opt_prop, opt_fid)
    # With the child OPT slice zero, the OPT forward equals the parent forward.
    assert torch.allclose(out_opt, adapter.parent(x, opt_prop, opt_fid), atol=1e-6)

    # Simulate a training step on the OPT slice.
    out_opt.sum().backward()
    adapter.zero_child_gradients_for_parent()

    # Child OPT slice values and gradients remain zero.
    assert adapter.child.G[:, :, opt_prop, opt_fid].abs().max().item() == 0.0
    assert adapter.child.E_prop.weight[opt_prop].abs().max().item() == 0.0
    assert adapter.child.E_fid.weight[opt_fid].abs().max().item() == 0.0
    assert adapter.child.G.grad[:, :, opt_prop, opt_fid].abs().max().item() == 0.0

    # MBJ slice may have non-zero values and can be updated by an optimizer.
    mbj_fid = 1
    optimizer = torch.optim.SGD(adapter.child.parameters(), lr=1.0)
    out_mbj = adapter(x.detach(), opt_prop, mbj_fid)
    out_mbj.sum().backward()
    optimizer.step()
    adapter.zero_child_gradients_for_parent()
    assert adapter.child.G[:, :, opt_prop, mbj_fid].abs().max().item() > 0.0


def test_progressive_model_loads_parent_from_canonical():
    """Canonical adapter weights can be remapped into the progressive parent."""
    node_dim, hidden_dim = 5, 8
    base_state = _random_model_state(node_dim, hidden_dim)

    model = ProgressivePhyTCAModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=1,
        n_fidelities=2,
        n_layers=2,
        adapter_rank=4,
        num_nearest_neighbors=4,
    )

    encoder_state = {k: v for k, v in base_state.items() if "adapter" not in k}
    parent_state = {k.replace("adapter.", "adapter.parent.", 1): v for k, v in base_state.items() if "adapter" in k}
    mapped = {**encoder_state, **parent_state}
    model.load_state_dict(mapped, strict=False)

    # Parent adapters match the canonical base; child adapters are random-init
    # (not tied to the parent), so they are non-zero and distinct from parent.
    for name, p in model.named_parameters():
        if "adapter.parent" in name:
            canonical_name = name.replace("adapter.parent.", "adapter.")
            assert torch.allclose(p, base_state[canonical_name])
        elif "adapter.child" in name:
            assert p.abs().max().item() > 0.0
            parent_name = name.replace("adapter.child.", "adapter.parent.")
            parent_p = dict(model.named_parameters())[parent_name]
            assert not torch.allclose(p, parent_p)


def test_frozen_opt_correction_preserves_opt():
    """FrozenOptCorrectionModel returns the exact OPT prediction for OPT fidelity."""
    node_dim, hidden_dim = 5, 8
    base_state = _random_model_state(node_dim, hidden_dim)
    base = PhyTCAModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=1,
        n_fidelities=2,
        n_layers=2,
        adapter_rank=4,
        num_nearest_neighbors=4,
        freeze_encoder_weights=True,
    )
    base.load_state_dict(base_state)

    correction = FrozenOptCorrectionModel(base, opt_prop_id=0, opt_fid_id=0, affine=True)
    node_feats = torch.randn(2, 4, node_dim)
    coords = torch.randn(2, 4, 3)
    mask = torch.ones(2, 4, dtype=torch.bool)
    original_mask = torch.tensor([[True, True, False, False], [True, False, False, False]])

    y_opt = correction(node_feats, coords, mask, original_mask, prop_id=0, fid_id=0)
    expected = base(node_feats, coords, mask, original_mask, prop_id=0, fid_id=0)
    assert torch.allclose(y_opt, expected, atol=1e-6)


def test_frozen_opt_residual_initially_identity():
    """Residual correction starts as y_opt because delta MLP is zero-init."""
    node_dim, hidden_dim = 5, 8
    base_state = _random_model_state(node_dim, hidden_dim)
    base = PhyTCAModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=1,
        n_fidelities=2,
        n_layers=2,
        adapter_rank=4,
        num_nearest_neighbors=4,
        freeze_encoder_weights=True,
    )
    base.load_state_dict(base_state)

    correction = FrozenOptCorrectionModel(base, opt_prop_id=0, opt_fid_id=0, affine=False)
    node_feats = torch.randn(2, 4, node_dim)
    coords = torch.randn(2, 4, 3)
    mask = torch.ones(2, 4, dtype=torch.bool)
    original_mask = torch.tensor([[True, True, False, False], [True, False, False, False]])

    y_mbj = correction(node_feats, coords, mask, original_mask, prop_id=0, fid_id=1)
    y_opt = base(node_feats, coords, mask, original_mask, prop_id=0, fid_id=0)
    assert torch.allclose(y_mbj, y_opt, atol=1e-6)


def test_replay_storage_accounting():
    """_detailed_parameter_stats correctly counts replay buffer storage."""
    model = PhyTCAModel(
        node_dim=5,
        hidden_dim=8,
        n_properties=1,
        n_fidelities=2,
        n_layers=2,
        adapter_rank=4,
        num_nearest_neighbors=4,
    )

    # Fake replay buffer with two samples: each stores pid, fid, nf, c, m, om, y.
    nf = torch.randn(1, 4, 5)
    c = torch.randn(1, 4, 3)
    m = torch.ones(1, 4, dtype=torch.bool)
    om = torch.ones(1, 4, dtype=torch.bool)
    y = torch.randn(1)
    replay_buffer = [(0, 0, nf, c, m, om, y), (0, 0, nf, c, m, om, y)]

    stats = _detailed_parameter_stats(model, "replay_1pct", replay_buffer)
    expected_bytes = sum(
        tensor.numel() * tensor.element_size()
        for _, _, nf, c, m, om, y in replay_buffer
        for tensor in (nf, c, m, om, y)
    )
    assert stats["replay_sample_count"] == 2
    assert stats["replay_storage_bytes"] == expected_bytes


def test_tucker4d_freeze_slice():
    """Freezing a Tucker slice zeros its gradient while leaving other slices trainable."""
    adapter = Tucker4DAdapter(d_in=8, d_out=8, n_properties=2, n_fidelities=2, rank_out=4, rank_in=4, rank_prop=2, rank_fid=2)
    adapter.freeze_slice(prop_id=0, fid_id=0)

    x = torch.randn(4, 8, requires_grad=True)
    out = adapter(x, prop_id=0, fid_id=0)
    out.sum().backward()
    adapter.zero_frozen_gradients()

    assert adapter.G.grad[:, :, 0, 0].abs().max().item() == 0.0
    assert adapter.G.grad[:, :, 1, 1].abs().max().item() > 0.0
