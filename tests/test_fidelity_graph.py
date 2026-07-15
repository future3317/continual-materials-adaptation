"""Unit tests for fidelity_graph.py."""

from __future__ import annotations

import pytest
import torch

from fidelity_graph import (
    AdaptiveRankAllocator,
    FidelityGraph,
    FidelityGraphPredictor,
    ParentSelector,
    path_consistency_loss,
)


def test_fidelity_graph_topological_order():
    g = FidelityGraph()
    g.add_edge(0, 0, 0, 1)  # opt -> mbj
    g.add_edge(0, 1, 0, 2)  # mbj -> hse
    g.add_edge(0, 0, 0, 2)  # opt -> hse (direct)
    order = g.topological_order()
    assert order[0] == (0, 0)
    assert (0, 2) in order


def test_fidelity_graph_detects_cycle():
    g = FidelityGraph()
    g.add_edge(0, 0, 0, 1)
    g.add_edge(0, 1, 0, 0)
    with pytest.raises(ValueError):
        g.topological_order()


def test_parent_selector():
    selector = ParentSelector(lambda_param=0.0, lambda_cost=0.0)
    candidates = [(0, 0), (0, 1)]
    errors = [0.5, 0.3]
    params = [100, 200]
    assert selector.select(candidates, errors, params) == (0, 1)

    # Param penalty should flip the decision.
    selector2 = ParentSelector(lambda_param=1e-2, lambda_cost=0.0)
    assert selector2.select(candidates, errors, params) == (0, 0)


def test_adaptive_rank_allocator():
    allocator = AdaptiveRankAllocator(epsilon=0.05)
    # Low-rank residual: only first few singular values matter.
    torch.manual_seed(0)
    u = torch.randn(50, 5)
    s = torch.tensor([10.0, 5.0, 1.0, 0.1, 0.05])
    v = torch.randn(8, 5)
    r = u @ torch.diag(s) @ v.t()
    rank = allocator.allocate(r)
    assert 1 <= rank <= 5


def test_adaptive_rank_allocator_empty():
    allocator = AdaptiveRankAllocator(epsilon=0.05)
    assert allocator.allocate(torch.empty(0, 8)) == 1


def test_path_consistency_loss():
    p1 = torch.randn(4)
    p2 = torch.randn(4)
    p3 = p1.clone()
    loss = path_consistency_loss({"path1": p1, "path2": p2, "path3": p3})
    assert loss.item() > 0

    loss_same = path_consistency_loss({"path1": p1, "path2": p1})
    assert loss_same.item() == pytest.approx(0.0, abs=1e-6)


def test_fidelity_graph_predictor():
    g = FidelityGraph()
    g.add_edge(0, 0, 0, 1)
    edge_module = torch.nn.Linear(1, 1)
    torch.nn.init.zeros_(edge_module.weight)
    torch.nn.init.zeros_(edge_module.bias)
    predictor = FidelityGraphPredictor(g, {((0, 0), (0, 1)): edge_module})

    parent_preds = {(0, 0): torch.tensor([[1.0], [2.0], [3.0]])}
    out = predictor(parent_preds, (0, 1))
    assert torch.allclose(out, parent_preds[(0, 0)], atol=1e-6)


def test_fidelity_graph_freeze_edge():
    g = FidelityGraph()
    g.add_edge(0, 0, 0, 1)
    assert not g.is_frozen(0, 0, 0, 1)
    g.freeze_edge(0, 0, 0, 1)
    assert g.is_frozen(0, 0, 0, 1)
