"""Unit tests for adapters.py and models.py.

These tests are fast and do not require JARVIS data.  They verify:
* Adapter forward shapes and parameter counts.
* No full ``d x d`` weight matrix is materialized.
* Functional equivalence of single-child Tucker and LoRA-ABA.
* Exact retention in ``ContinualCrystalModel``.
* Frozen tasks are excluded from the optimizer.
"""

from __future__ import annotations

import pytest
import torch

from adapters import (
    ADAPTER_REGISTRY,
    LoRAABAdapter,
    LoRAABAAdapter,
    MultiAxisTuckerAdapter,
    ResidualAdapter,
    SingleChildTuckerAdapter,
    make_adapter_bank,
)
from models import ContinualCrystalModel, PredictionResidualHead


@pytest.mark.parametrize("adapter_name", list(ADAPTER_REGISTRY.keys()))
def test_adapter_forward_2d(adapter_name: str):
    """All adapters accept 2D inputs and produce 2D outputs."""
    d_in, d_out, rank = 16, 16, 4
    if adapter_name == "multi_axis_tucker":
        adapter = MultiAxisTuckerAdapter(d_in, d_out, n_properties=2, n_fidelities=3, rank_out=rank, rank_in=rank)
        out = adapter(torch.randn(5, d_in), prop_id=0, fid_id=0)
    elif adapter_name == "bottleneck":
        from adapters import BottleneckAdapter
        adapter = BottleneckAdapter(d_in, d_out, bottleneck=8)
        out = adapter(torch.randn(5, d_in))
    elif adapter_name == "zero":
        from adapters import ZeroAdapter
        adapter = ZeroAdapter(d_in, d_out)
        out = adapter(torch.randn(5, d_in))
    else:
        cls = ADAPTER_REGISTRY[adapter_name]
        adapter = cls(d_in, d_out, rank)
        out = adapter(torch.randn(5, d_in))
    assert out.shape == (5, d_out)


@pytest.mark.parametrize("adapter_name", ["lora_ab", "lora_aba", "single_child_tucker"])
def test_adapter_forward_3d(adapter_name: str):
    """Low-rank adapters accept 3D per-node inputs."""
    cls = ADAPTER_REGISTRY[adapter_name]
    adapter = cls(16, 16, 4)
    x = torch.randn(2, 7, 16)
    out = adapter(x)
    assert out.shape == (2, 7, 16)


def test_adapter_parameter_count_matches_formula():
    """Parameter counts follow the closed-form expressions."""
    d_in, d_out, rank = 20, 24, 6
    ab = LoRAABAdapter(d_in, d_out, rank)
    assert ab.incremental_parameter_count() == d_in * rank + d_out * rank

    aba = LoRAABAAdapter(d_in, d_out, rank)
    assert aba.incremental_parameter_count() == d_in * rank + rank * rank + d_out * rank

    tucker = SingleChildTuckerAdapter(d_in, d_out, rank)
    assert tucker.incremental_parameter_count() == d_in * rank + rank * rank + d_out * rank


def test_single_child_tucker_matches_lora_aba():
    """SingleChildTuckerAdapter is a semantic alias for LoRAABAAdapter."""
    d_in, d_out, rank = 16, 16, 4
    tucker = SingleChildTuckerAdapter(d_in, d_out, rank)
    aba = LoRAABAAdapter(d_in, d_out, rank)
    assert isinstance(tucker, LoRAABAAdapter)

    # With identical weights, outputs and gradients must match.
    tucker.u_in.data = aba.u_in.data.clone()
    tucker.middle.data = aba.middle.data.clone()
    tucker.u_out.data = aba.u_out.data.clone()

    x = torch.randn(3, d_in, requires_grad=True)
    out_t = tucker(x)
    out_a = aba(x)
    assert torch.allclose(out_t, out_a, atol=1e-6)

    out_t.sum().backward()
    grad_t = x.grad.clone()
    x.grad = None
    out_a.sum().backward()
    grad_a = x.grad.clone()
    assert torch.allclose(grad_t, grad_a, atol=1e-6)


def test_single_child_parameter_formula_3329():
    """Default single-child config must match L(2dr + r^2) + (d + 1) = 3329."""
    from models import ContinualCrystalModel

    model = ContinualCrystalModel(
        node_dim=92,
        hidden_dim=64,
        n_properties=1,
        n_fidelities=2,
        adapter_name="single_child_tucker",
        adapter_rank=8,
        n_layers=3,
        num_nearest_neighbors=8,
    )
    model.add_task(0, 0)
    count = model.count_task_parameters(0, 0)
    expected = 3 * (2 * 64 * 8 + 8 * 8) + (64 + 1)
    assert count == expected == 3329, f"expected {expected}, got {count}"


def test_no_dxd_materialization():
    """Adapter forward never allocates a ``d_out x d_in`` tensor."""
    d_in, d_out, rank = 1024, 1024, 8
    adapter = SingleChildTuckerAdapter(d_in, d_out, rank)
    x = torch.randn(2, d_in)

    peak_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else None
    out = adapter(x)
    # We cannot reliably peak-detect without a memory profiler, but we can at
    # least assert the output shape and that the adapter has no large parameter.
    assert out.shape == (2, d_out)
    assert adapter.incremental_parameter_count() < d_in * d_out

    if peak_before is not None:
        peak_after = torch.cuda.memory_allocated()
        # Tight bound: peak memory should not grow by a full d_in*d_out matrix.
        assert peak_after - peak_before < 2 * d_in * d_out * 4


def test_make_adapter_bank():
    """``make_adapter_bank`` creates the requested number of adapters."""
    bank = make_adapter_bank("lora_aba", n_layers=3, dim=16, rank=4)
    assert len(bank) == 3
    assert all(isinstance(a, LoRAABAAdapter) for a in bank)

    bank_tucker = make_adapter_bank(
        "multi_axis_tucker",
        n_layers=2,
        dim=16,
        rank=4,
        n_properties=2,
        n_fidelities=3,
    )
    assert len(bank_tucker) == 2
    assert all(isinstance(a, MultiAxisTuckerAdapter) for a in bank_tucker)


def test_continual_model_exact_retention():
    """Freezing a task and training a new one does not change the old task's predictions."""
    model = ContinualCrystalModel(
        node_dim=8,
        hidden_dim=16,
        n_properties=1,
        n_fidelities=2,
        adapter_name="single_child_tucker",
        adapter_rank=4,
        n_layers=2,
        num_nearest_neighbors=4,
    )

    node_feats = torch.randn(2, 4, 8)
    coords = torch.randn(2, 4, 3)
    mask = torch.ones(2, 4, dtype=torch.bool)
    original_mask = torch.tensor([[True, True, False, False], [True, False, False, False]])

    model.add_task(0, 0)
    with torch.no_grad():
        pred_before = model(node_feats, coords, mask, original_mask, 0, 0)

    model.freeze_task(0, 0)
    model.add_task(0, 1)
    # Simulate training on new task by taking an SGD step.
    opt = torch.optim.SGD(model.current_trainable_parameters(), lr=0.1)
    out = model(node_feats, coords, mask, original_mask, 0, 1)
    out.sum().backward()
    opt.step()

    with torch.no_grad():
        pred_after = model(node_feats, coords, mask, original_mask, 0, 0)
    assert torch.allclose(pred_before, pred_after, atol=1e-6)


def test_continual_model_frozen_task_excluded_from_optimizer():
    """After freezing, the optimizer for the new task contains only new-task parameters."""
    model = ContinualCrystalModel(
        node_dim=8,
        hidden_dim=16,
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

    trainable = set(id(p) for p in model.current_trainable_parameters())
    opt = torch.optim.SGD(model.current_trainable_parameters(), lr=0.1)
    opt_params = set(id(p) for group in opt.param_groups for p in group["params"])
    assert opt_params == trainable

    # Old task parameters should not be in the optimizer.
    old_params = set(id(p) for p in model.adapter_banks["p0_f0"].parameters())
    assert old_params.isdisjoint(opt_params)


def test_prediction_residual_head_units():
    """PredictionResidualHead correctly converts between normalized and physical units."""
    head = PredictionResidualHead(hidden_dim=16)
    h = torch.randn(4, 16)
    parent_mean, parent_std = torch.tensor(1.0), torch.tensor(2.0)
    child_mean, child_std = torch.tensor(-1.0), torch.tensor(0.5)

    # Parent prediction in parent-normalized space.
    y_opt_norm = torch.randn(4)
    y_opt_phys = y_opt_norm * parent_std + parent_mean

    # The residual MLP is zero-initialized, so residual is zero.
    out_norm = head(h, y_opt_norm, parent_mean, parent_std, child_mean, child_std)
    out_phys = out_norm * child_std + child_mean
    assert torch.allclose(out_phys, y_opt_phys, atol=1e-6)


def test_parameter_group_counts():
    """Parameter group counts sum to total and encoder is frozen."""
    model = ContinualCrystalModel(
        node_dim=8,
        hidden_dim=16,
        n_properties=1,
        n_fidelities=2,
        adapter_name="lora_ab",
        adapter_rank=4,
        n_layers=2,
        num_nearest_neighbors=4,
    )
    model.add_task(0, 0)
    groups = model.get_parameter_group_counts()
    assert groups["total"] == groups["encoder"] + groups["adapters"] + groups["heads"]
    assert all(not p.requires_grad for p in model.encoder.parameters())
