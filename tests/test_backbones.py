"""Tests for MatGL-backed encoder interface in ``backbones.py``."""

from __future__ import annotations

import pytest
import torch
from pymatgen.core import Lattice, Structure

try:
    import matgl  # noqa: F401
    from backbones import MatGLBackbone, build_matgl_backbone

    _MATGL_AVAILABLE = True
    _matgl_skip_reason = ""
except Exception as exc:  # pragma: no cover - MatGL may be missing/broken
    _MATGL_AVAILABLE = False
    _matgl_skip_reason = str(exc)

from models import ContinualCrystalModel
from periodic_graph import build_periodic_edge_graph


pytestmark = pytest.mark.skipif(not _MATGL_AVAILABLE, reason=_matgl_skip_reason)


def _make_si_structure():
    lattice = Lattice.cubic(4.0)
    return Structure(lattice, ["Si", "Si"], [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])


def test_matgl_backbone_can_be_instantiated():
    backbone = build_matgl_backbone(model_name=None, hidden_dim=16, freeze=True)
    assert isinstance(backbone, MatGLBackbone)
    assert backbone.hidden_dim == 16


def test_matgl_backbone_forward_shape():
    backbone = build_matgl_backbone(model_name=None, hidden_dim=16, freeze=True)
    struct = _make_si_structure()
    graph = build_periodic_edge_graph(struct, cutoff=5.0)

    node_features = backbone(graph)
    assert node_features.shape == (len(struct), 16)
    assert node_features.dtype == graph["coords"].dtype


def test_matgl_backbone_frozen_has_no_trainable_parameters():
    backbone = build_matgl_backbone(model_name=None, hidden_dim=16, freeze=True)
    trainable = [p for p in backbone.matgl_model.parameters() if p.requires_grad]
    assert len(trainable) == 0
    # The projection head is the only potentially trainable part; for hidden_dim
    # equal to the native node embedding dimension it is an Identity.
    assert isinstance(backbone.projection, torch.nn.Identity)


def test_matgl_backbone_count_parameters():
    backbone = build_matgl_backbone(model_name=None, hidden_dim=16, freeze=True)
    count = backbone.count_parameters()
    assert count > 0
    assert count == sum(p.numel() for p in backbone.matgl_model.parameters())


def test_continual_model_accepts_matgl_backbone():
    backbone = build_matgl_backbone(model_name=None, hidden_dim=16, freeze=True)
    model = ContinualCrystalModel(
        node_dim=92,
        hidden_dim=16,
        n_properties=1,
        n_fidelities=1,
        adapter_name="lora_ab",
        adapter_rank=4,
        n_layers=1,
        encoder=backbone,
    )
    model.add_task(0, 0)

    node_feats = torch.randn(2, 4, 92)
    coords = torch.randn(2, 4, 3)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    original_mask = mask.clone()

    pred = model(node_feats, coords, mask, original_mask, 0, 0)
    assert pred.shape == (2,)


def test_continual_model_with_matgl_encoder_parameter_counts():
    backbone = build_matgl_backbone(model_name=None, hidden_dim=16, freeze=True)
    model = ContinualCrystalModel(
        node_dim=92,
        hidden_dim=16,
        n_properties=1,
        n_fidelities=1,
        adapter_name="lora_ab",
        adapter_rank=4,
        n_layers=1,
        encoder=backbone,
    )
    model.add_task(0, 0)
    groups = model.get_parameter_group_counts()
    assert groups["total"] == groups["encoder"] + groups["adapters"] + groups["heads"]
    assert groups["encoder"] == backbone.count_parameters()
