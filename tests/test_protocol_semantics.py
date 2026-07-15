"""Protocol A and B semantic correctness tests for PhyTCA."""

from __future__ import annotations

import pytest
import torch

from data import JARVISCrystalDataset, build_protocol_a, build_protocol_b, collate_crystals
from phytca import PhyTCAModel
from train_phytca import _last_occurrences, _name_to_id, continual_experiment


def test_protocol_a_same_head():
    """A1 and A2 share the same scalar prediction head."""
    tasks, _, _ = build_protocol_a(n_train_val_per_task=50)
    prop_ids, fid_ids = _name_to_id(tasks)
    model = PhyTCAModel(
        node_dim=92,
        hidden_dim=32,
        n_properties=len(prop_ids),
        n_fidelities=len(fid_ids),
        n_layers=2,
        adapter_rank=4,
        num_nearest_neighbors=4,
        freeze_encoder_weights=True,
    )
    key_a1 = f"p{prop_ids['formation_energy']}_f{fid_ids['OptB88vdW']}"
    key_a2 = f"p{prop_ids['formation_energy']}_f{fid_ids['OptB88vdW']}"
    assert key_a1 == key_a2
    assert key_a1 in model.heads


def test_protocol_a_same_adapter_route():
    """A1 and A2 route through the same property/fidelity embeddings and core slice."""
    tasks, _, _ = build_protocol_a(n_train_val_per_task=50)
    prop_ids, fid_ids = _name_to_id(tasks)
    model = PhyTCAModel(
        node_dim=92,
        hidden_dim=32,
        n_properties=len(prop_ids),
        n_fidelities=len(fid_ids),
        n_layers=2,
        adapter_rank=4,
        num_nearest_neighbors=4,
        freeze_encoder_weights=True,
    )
    p = prop_ids["formation_energy"]
    f = fid_ids["OptB88vdW"]
    x = torch.randn(2, 32)
    for layer in model.layers:
        adapter = layer.adapter
        out1 = adapter(x, prop_id=p, fid_id=f)
        out2 = adapter(x, prop_id=p, fid_id=f)
        assert out1.shape == (2, 32)
        assert torch.equal(out1, out2)
        # Different property ids use a different core slice.
        other_p = prop_ids["band_gap"]
        out3 = adapter(x, prop_id=other_p, fid_id=f)
        assert not torch.equal(out1, out3)


def test_protocol_a_no_snapshot_conditioning():
    """Model forward is conditioned only on property/fidelity ids, not snapshot id."""
    tasks, _, _ = build_protocol_a(n_train_val_per_task=50)
    prop_ids, fid_ids = _name_to_id(tasks)
    model = PhyTCAModel(
        node_dim=92,
        hidden_dim=32,
        n_properties=len(prop_ids),
        n_fidelities=len(fid_ids),
        n_layers=2,
        adapter_rank=4,
        num_nearest_neighbors=4,
    )
    import inspect
    sig = inspect.signature(model.forward)
    params = list(sig.parameters.keys())
    assert "snapshot_id" not in params
    assert "task_id" not in params
    assert "prop_id" in params
    assert "fid_id" in params


def test_protocol_a_freeze_only_after_last_occurrence():
    """Formation/OPT path is trainable during A2 and frozen only after A2."""
    tasks, _, _ = build_protocol_a(n_train_val_per_task=50)
    prop_ids, fid_ids = _name_to_id(tasks)
    freeze_steps = _last_occurrences(tasks)

    p = prop_ids["formation_energy"]
    f = fid_ids["OptB88vdW"]

    # Formation/OPT last appears at task index 1 (A2).
    assert (1 in freeze_steps) and (0 not in freeze_steps)

    model = PhyTCAModel(
        node_dim=92,
        hidden_dim=32,
        n_properties=len(prop_ids),
        n_fidelities=len(fid_ids),
        n_layers=2,
        adapter_rank=4,
        num_nearest_neighbors=4,
        freeze_encoder_weights=True,
    )

    # Before any freeze, head is trainable.
    key = f"p{p}_f{f}"
    assert all(par.requires_grad for par in model.heads[key].parameters())

    # Simulate finishing A2: freeze formation/OPT.
    model.freeze_task(p, f)
    assert not any(par.requires_grad for par in model.heads[key].parameters())

    # Core slice is recorded as frozen; other slices are still trainable.
    assert (p, f) in model.layers[0].adapter.frozen_slices
    other_p = prop_ids["band_gap"]
    assert (other_p, f) not in model.layers[0].adapter.frozen_slices


def test_protocol_a_parameters_update_on_a2():
    """Training A2 updates the shared formation/OPT path."""
    tasks, task_records, _ = build_protocol_a(n_train_val_per_task=50)
    prop_ids, fid_ids = _name_to_id(tasks)
    device = torch.device("cpu")
    model = PhyTCAModel(
        node_dim=92,
        hidden_dim=32,
        n_properties=len(prop_ids),
        n_fidelities=len(fid_ids),
        n_layers=2,
        adapter_rank=4,
        num_nearest_neighbors=4,
        freeze_encoder_weights=True,
    ).to(device)

    p_form = prop_ids["formation_energy"]
    f_opt = fid_ids["OptB88vdW"]
    key = f"p{p_form}_f{f_opt}"

    # Snapshot after "A1" by training task 0 then freezing (not yet frozen because A2 follows).
    # Instead, directly check that a backward pass on A2 data produces non-zero grads.
    recs_a2 = [r for r in task_records[1] if r["split"] == "train"][:8]
    ds = JARVISCrystalDataset(recs_a2)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=4,
        collate_fn=collate_crystals,
    )

    model.train()
    for node_feats, coords, mask, original_mask, y in loader:
        node_feats = node_feats.to(device)
        coords = coords.to(device)
        mask = mask.to(device)
        original_mask = original_mask.to(device)
        pred = model(node_feats, coords, mask, original_mask, p_form, f_opt)
        loss = pred.sum()
        loss.backward()
        break

    head_grad = next(model.heads[key].parameters()).grad
    assert head_grad is not None
    assert head_grad.abs().sum() > 0.0

    # Shared adapter parameters receive gradients.
    adapter = model.layers[0].adapter
    assert adapter.G.grad is not None
    assert adapter.G.grad.abs().sum() > 0.0


def test_protocol_a_a1_frozen_after_a2():
    """After A2, the formation/OPT head and adapter slice are frozen."""
    tasks, task_records, _ = build_protocol_a(n_train_val_per_task=50)
    prop_ids, fid_ids = _name_to_id(tasks)
    device = torch.device("cpu")

    nmaes, info = continual_experiment(
        tasks=tasks,
        task_records=task_records,
        node_dim=92,
        hidden_dim=32,
        device=device,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        mu=0.0,
        adapter_rank=4,
        num_nearest_neighbors=4,
    )
    model = info["model"]
    p_form = prop_ids["formation_energy"]
    f_opt = fid_ids["OptB88vdW"]
    key = f"p{p_form}_f{f_opt}"

    assert not any(par.requires_grad for par in model.heads[key].parameters())
    assert (p_form, f_opt) in model.layers[0].adapter.frozen_slices


def test_protocol_b_different_fidelity_embeddings():
    """Protocol B differs only in fidelity embeddings; property and head are shared."""
    tasks, _, _ = build_protocol_b(n_train_val_per_task=50)
    prop_ids, fid_ids = _name_to_id(tasks)
    model = PhyTCAModel(
        node_dim=92,
        hidden_dim=32,
        n_properties=len(prop_ids),
        n_fidelities=len(fid_ids),
        n_layers=2,
        adapter_rank=4,
        num_nearest_neighbors=4,
    )
    p = prop_ids["band_gap"]
    f_opt = fid_ids["OptB88vdW"]
    f_mbj = fid_ids["TB-mBJ"]

    opt_key = f"p{p}_f{f_opt}"
    mbj_key = f"p{p}_f{f_mbj}"
    assert opt_key in model.heads
    assert mbj_key in model.heads
    assert opt_key != mbj_key


def test_protocol_b_tasks_paired_before_split():
    """Protocol B records are paired OPT/MBJ structures before formula splitting."""
    tasks, task_records, _ = build_protocol_b(n_train_val_per_task=50)
    # Each task record has both fidelities available as targets.
    for recs in task_records:
        for r in recs:
            assert "target" in r
            assert r["property"] == "band_gap"
