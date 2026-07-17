"""Smoke tests for PCG with MatGL backbone."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from backbones import _MATGL_AVAILABLE, build_matgl_backbone
from persistent_consolidation_graph import PersistentConsolidationGraph


@pytest.mark.skipif(not _MATGL_AVAILABLE, reason="MatGL not installed")
def test_pcg_matgl_forward_shape():
    hidden_dim, rank = 16, 4
    encoder = build_matgl_backbone(None, hidden_dim=hidden_dim, freeze=True)
    model = PersistentConsolidationGraph(encoder, hidden_dim, rank=rank)

    # MatGL default element table has 89 elements but is not contiguous to Z=89;
    # restrict one-hot indices to the safe low-Z range.
    node_dim = 89
    atom_types = torch.randint(0, 83, (2, 5))
    x = F.one_hot(atom_types, num_classes=node_dim).float()
    coords = torch.randn(2, 5, 3)
    mask = torch.ones(2, 5, dtype=torch.bool)
    original_mask = mask.clone()

    model.add_route("v1", prop_id=0, fid_id=0, normalizer=(0.0, 1.0))
    pred = model(x, coords, mask, original_mask, "v1", 0, 0, physical=True)
    assert pred.shape == (2,)


@pytest.mark.skipif(not _MATGL_AVAILABLE, reason="MatGL not installed")
def test_pcg_matgl_forgets_none_after_new_route():
    hidden_dim, rank = 16, 4
    encoder = build_matgl_backbone(None, hidden_dim=hidden_dim, freeze=True)
    model = PersistentConsolidationGraph(encoder, hidden_dim, rank=rank)

    node_dim = 89
    atom_types = torch.randint(0, 83, (2, 5))
    x = F.one_hot(atom_types, num_classes=node_dim).float()
    coords = torch.randn(2, 5, 3)
    mask = torch.ones(2, 5, dtype=torch.bool)
    original_mask = mask.clone()

    model.add_route("v1", prop_id=0, fid_id=0, normalizer=(0.0, 1.0))
    before = model(x, coords, mask, original_mask, "v1", 0, 0, physical=True).detach().clone()

    model.publish_route("v1", 0, 0)
    model.add_route("v2", prop_id=0, fid_id=0, normalizer=(0.0, 1.0))

    optimizer = torch.optim.Adam(model.current_trainable_parameters(), lr=1e-2)
    for _ in range(3):
        optimizer.zero_grad()
        pred = model(x, coords, mask, original_mask, "v2", 0, 0, physical=True)
        loss = pred.pow(2).sum()
        loss.backward()
        optimizer.step()

    after = model(x, coords, mask, original_mask, "v1", 0, 0, physical=True)
    assert torch.allclose(before, after, atol=1e-6)
