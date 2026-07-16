"""Unit tests for diagnostic model variants and exact-retention guarantees.

These tests are fast and self-contained: they use synthetic inputs and do not
require downloading JARVIS data.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from adapters import ADAPTER_REGISTRY, LoRAABAAdapter, SingleChildTuckerAdapter
from diagnostics import (
    FrozenOptCorrectionModel,
    FrozenOptResidualModel,
    LowRankResidual,
    PhysicalResidualCorrection,
)
from models import ContinualCrystalModel, PredictionResidualHead
from scripts.run_phase0_b_screening import _detailed_parameter_stats


def _random_cont_model_state(
    node_dim: int = 5, hidden_dim: int = 8
) -> dict[str, torch.Tensor]:
    """Return a tiny ContinualCrystalModel state dict for two fidelity ids."""
    model = ContinualCrystalModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_properties=1,
        n_fidelities=2,
        adapter_name="single_child_tucker",
        adapter_rank=4,
        n_layers=2,
        num_nearest_neighbors=4,
    )
    model.add_task(0, 0)
    return {k: v.clone() for k, v in model.state_dict().items()}


def _make_tiny_batch(
    node_dim: int = 5, n_atoms: int = 4, batch_size: int = 2
) -> tuple[torch.Tensor, ...]:
    node_feats = torch.randn(batch_size, n_atoms, node_dim)
    coords = torch.randn(batch_size, n_atoms, 3)
    mask = torch.ones(batch_size, n_atoms, dtype=torch.bool)
    original_mask = torch.tensor(
        [[True, True, False, False], [True, False, False, False]]
    )
    return node_feats, coords, mask, original_mask


def test_single_child_tucker_matches_lora_aba():
    """Single-child Tucker is now a semantic alias for LoRA-ABA."""
    d_in, d_out, rank = 8, 8, 4
    tucker = SingleChildTuckerAdapter(d_in, d_out, rank)
    lora_aba = LoRAABAAdapter(d_in, d_out, rank)
    # Copy parameters so the two adapters compute the same map.
    lora_aba.u_in.data = tucker.u_in.data.clone()
    lora_aba.middle.data = tucker.middle.data.clone()
    lora_aba.u_out.data = tucker.u_out.data.clone()

    x = torch.randn(3, d_in)
    assert torch.allclose(tucker(x), lora_aba(x), atol=1e-6)
    assert isinstance(tucker, LoRAABAAdapter)


def test_adapter_registry_has_baselines():
    """All adapter names used by diagnostics are registered."""
    for name in ("lora_ab", "lora_aba", "single_child_tucker", "multi_axis_tucker"):
        assert name in ADAPTER_REGISTRY


def test_continual_model_exact_retention():
    """Freezing a task leaves its predictions unchanged after training a new task."""
    model = ContinualCrystalModel(
        node_dim=5,
        hidden_dim=8,
        n_properties=1,
        n_fidelities=2,
        adapter_name="single_child_tucker",
        adapter_rank=4,
        n_layers=2,
        num_nearest_neighbors=4,
    )
    model.add_task(0, 0)
    node_feats, coords, mask, original_mask = _make_tiny_batch()

    model.eval()
    with torch.no_grad():
        pred_before = model(node_feats, coords, mask, original_mask, 0, 0)

    # Train the MBJ task.
    model.add_task(0, 1)
    model.train()
    optimizer = torch.optim.AdamW(model.current_trainable_parameters(), lr=1e-2)
    target = torch.randn(2)
    for _ in range(3):
        optimizer.zero_grad()
        pred = model(node_feats, coords, mask, original_mask, 0, 1)
        loss = nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

    # Freeze MBJ and verify OPT predictions are unchanged.
    model.freeze_task(0, 1)
    model.eval()
    with torch.no_grad():
        pred_after = model(node_feats, coords, mask, original_mask, 0, 0)
    assert torch.allclose(pred_before, pred_after, atol=1e-6)


def test_continual_model_frozen_task_excluded_from_optimizer():
    """After freezing, the old task has no trainable parameters."""
    model = ContinualCrystalModel(
        node_dim=5,
        hidden_dim=8,
        n_properties=1,
        n_fidelities=2,
        adapter_name="single_child_tucker",
        adapter_rank=4,
        n_layers=2,
        num_nearest_neighbors=4,
    )
    model.add_task(0, 0)
    model.freeze_task(0, 0)
    model.add_task(0, 1)
    trainable = model.current_trainable_parameters()
    assert len(trainable) > 0
    # None of the trainable parameters should belong to the frozen task.
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert "p0_f0" not in name


def test_prediction_residual_head_normalization():
    """PredictionResidualHead converts parent-normalized -> physical -> child-normalized."""
    head = PredictionResidualHead(8)
    h = torch.randn(4, 8)
    parent_pred_norm = torch.randn(4)
    parent_mean, parent_std = 1.0, 2.0
    child_mean, child_std = -0.5, 0.5

    out = head(h, parent_pred_norm, parent_mean, parent_std, child_mean, child_std)
    expected_phys = parent_pred_norm * parent_std + parent_mean  # de-normalize
    expected_phys = expected_phys + head.residual_mlp(h).squeeze(-1)  # residual
    expected = (expected_phys - child_mean) / child_std
    assert torch.allclose(out, expected, atol=1e-6)


def test_frozen_opt_correction_preserves_opt():
    """FrozenOptCorrectionModel returns the exact OPT prediction for OPT fidelity."""
    model = ContinualCrystalModel(
        node_dim=5,
        hidden_dim=8,
        n_properties=1,
        n_fidelities=2,
        adapter_name="single_child_tucker",
        adapter_rank=4,
        n_layers=2,
        num_nearest_neighbors=4,
    )
    model.add_task(0, 0)
    model.load_state_dict(_random_cont_model_state(), strict=False)

    correction = FrozenOptCorrectionModel(model, opt_prop_id=0, opt_fid_id=0, affine=True)
    correction.set_normalizers(
        torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0), torch.tensor(1.0)
    )
    node_feats, coords, mask, original_mask = _make_tiny_batch()

    y_opt = correction(node_feats, coords, mask, original_mask, prop_id=0, fid_id=0)
    expected = model(node_feats, coords, mask, original_mask, 0, 0)
    assert torch.allclose(y_opt, expected, atol=1e-6)


def test_frozen_opt_residual_initially_identity():
    """Residual correction starts as y_opt because the residual MLP is zero-init."""
    model = ContinualCrystalModel(
        node_dim=5,
        hidden_dim=8,
        n_properties=1,
        n_fidelities=2,
        adapter_name="single_child_tucker",
        adapter_rank=4,
        n_layers=2,
        num_nearest_neighbors=4,
    )
    model.add_task(0, 0)
    model.load_state_dict(_random_cont_model_state(), strict=False)

    correction = FrozenOptCorrectionModel(model, opt_prop_id=0, opt_fid_id=0, affine=False)
    correction.set_normalizers(
        torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0), torch.tensor(1.0)
    )
    node_feats, coords, mask, original_mask = _make_tiny_batch()

    y_mbj = correction(node_feats, coords, mask, original_mask, prop_id=0, fid_id=1)
    y_opt = model(node_feats, coords, mask, original_mask, 0, 0)
    assert torch.allclose(y_mbj, y_opt, atol=1e-6)


def test_physical_residual_correction_units():
    """PhysicalResidualCorrection never adds quantities in different normalizations."""
    module = LowRankResidual(hidden_dim=8, rank=4)
    correction = PhysicalResidualCorrection(module)
    h = torch.randn(4, 8)
    parent_pred_norm = torch.randn(4)
    parent_mean, parent_std = 1.0, 2.0
    child_mean, child_std = -0.5, 0.5

    out = correction(h, parent_pred_norm, parent_mean, parent_std, child_mean, child_std)
    # All arithmetic is in physical units before the final child normalization.
    parent_phys = parent_pred_norm * parent_std + parent_mean
    residual_phys = module(h).squeeze(-1)
    expected = (parent_phys + residual_phys - child_mean) / child_std
    assert torch.allclose(out, expected, atol=1e-6)


def test_frozen_opt_residual_model_routing():
    """FrozenOptResidualModel returns parent for OPT and residual for child."""
    model = ContinualCrystalModel(
        node_dim=5,
        hidden_dim=8,
        n_properties=1,
        n_fidelities=2,
        adapter_name="single_child_tucker",
        adapter_rank=4,
        n_layers=2,
        num_nearest_neighbors=4,
    )
    model.add_task(0, 0)
    model.load_state_dict(_random_cont_model_state(), strict=False)

    correction_module = PredictionResidualHead(8).residual_mlp
    residual_model = FrozenOptResidualModel(model, 0, 0, correction_module)
    residual_model.set_normalizers(
        torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0), torch.tensor(1.0)
    )
    node_feats, coords, mask, original_mask = _make_tiny_batch()

    y_opt = residual_model(node_feats, coords, mask, original_mask, 0, 0)
    expected_opt = model(node_feats, coords, mask, original_mask, 0, 0)
    assert torch.allclose(y_opt, expected_opt, atol=1e-6)

    y_mbj = residual_model(node_feats, coords, mask, original_mask, 0, 1)
    # Residual MLP is zero-init, so output equals parent prediction.
    assert torch.allclose(y_mbj, expected_opt, atol=1e-6)


def test_replay_storage_accounting():
    """_detailed_parameter_stats correctly counts replay buffer storage."""
    from legacy.phytca import PhyTCAModel

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
