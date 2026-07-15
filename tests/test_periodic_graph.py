"""Correctness tests for periodic graph construction and JARVIS data conversion."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

from data import (
    PeriodicGraphBuilder,
    build_periodic_graph,
    build_protocol_a,
    build_protocol_b,
    jarvis_record_to_structure,
    load_jarvis_dataset,
    parse_target,
)


def test_parse_target():
    """Robust target parser rejects sentinel values and non-finite numbers."""
    assert parse_target(1.5) == 1.5
    assert parse_target("1.5") == 1.5
    assert parse_target("na") is None
    assert parse_target("N/A") is None
    assert parse_target(None) is None
    assert parse_target("") is None
    assert parse_target(float("inf")) is None
    assert parse_target(float("nan")) is None


def test_jarvis_record_to_structure():
    """JARVIS atoms dict converts to a valid pymatgen Structure."""
    records = load_jarvis_dataset("dft_3d_2021")
    struct = jarvis_record_to_structure(records[0])
    assert isinstance(struct, Structure)
    assert struct.volume > 1e-6
    assert len(struct) > 0


def test_periodic_graph_supercell_size():
    """Supercell expansion multiplies the number of atoms correctly."""
    lattice = Lattice.cubic(3.0)
    struct = Structure(lattice, ["Si", "Si"], [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    graph = build_periodic_graph(struct, np.diag([2, 2, 2]))

    assert graph["node_feats"].shape == (16, 92)  # 2 * 2^3
    assert graph["coords"].shape == (16, 3)
    assert graph["original_mask"].sum() == 2
    assert graph["n_original"] == 2


def test_periodic_graph_original_mask_consistency():
    """Original atoms are exactly those with zero image offset."""
    lattice = Lattice.cubic(4.0)
    struct = Structure(lattice, ["Si", "Si"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    graph = build_periodic_graph(struct, np.diag([2, 2, 2]))

    zero_offset = (graph["image_offsets"] == 0).all(dim=1)
    assert torch.equal(zero_offset, graph["original_mask"])
    assert graph["original_mask"].sum() == len(struct)


def test_periodic_graph_invariance_to_unit_cell_translation():
    """Periodic graph is invariant under translating the unit cell origin."""
    lattice = Lattice.cubic(4.0)
    coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    struct1 = Structure(lattice, ["Si", "Si"], coords)
    struct2 = Structure(lattice, ["Si", "Si"], coords + np.array([0.1, 0.2, 0.3]))

    g1 = build_periodic_graph(struct1, np.diag([2, 2, 2]))
    g2 = build_periodic_graph(struct2, np.diag([2, 2, 2]))

    # Node features (one-hot by element) are identical.
    assert torch.equal(g1["node_feats"], g2["node_feats"])

    # The two supercells describe the same infinite lattice, so their point
    # clouds differ by a global translation equal to the unit-cell translation.
    delta = g2["coords"][0] - g1["coords"][0]
    assert torch.allclose(g2["coords"], g1["coords"] + delta, atol=1e-3)
    # The original-mask location may differ because pymatgen reorders atoms, but
    # the number of original atoms is preserved.
    assert g1["original_mask"].sum() == g2["original_mask"].sum()


def test_periodic_graph_replicas_have_correct_relative_positions():
    """Periodic image replicas are separated by integer lattice translations."""
    lattice = Lattice.cubic(4.0)
    struct = Structure(lattice, ["Si"], [[0.0, 0.0, 0.0]])
    graph = build_periodic_graph(struct, np.diag([2, 2, 2]))

    lattice_torch = torch.tensor(lattice.matrix, dtype=torch.float32)
    lattice_inv = torch.linalg.inv(lattice_torch)
    original_idx = torch.where(graph["original_mask"])[0][0]
    original_coord = graph["coords"][original_idx]
    original_frac = original_coord @ lattice_inv.T

    for idx in range(graph["coords"].size(0)):
        coord = graph["coords"][idx]
        frac = coord @ lattice_inv.T
        # Difference should be an integer offset times the lattice.
        diff = frac - original_frac
        offset = torch.round(diff)
        assert torch.allclose(diff, offset, atol=1e-3)


def test_periodic_graph_builder_egnn_forward():
    """A small EGNN can consume the periodic graph tensors."""
    pytest.importorskip("egnn_pytorch")
    from egnn_pytorch import EGNN

    records = load_jarvis_dataset("dft_3d_2021")
    struct = jarvis_record_to_structure(records[0])
    builder = PeriodicGraphBuilder(supercell_matrix=2)
    graph = builder(struct)

    feats = graph["node_feats"].unsqueeze(0)  # (1, N, F)
    coords = graph["coords"].unsqueeze(0)      # (1, N, 3)
    mask = torch.ones(feats.size(1), dtype=torch.bool).unsqueeze(0)

    egnn = EGNN(dim=92, m_dim=32, num_nearest_neighbors=8, update_coors=False, update_feats=True)
    out_feats, out_coords = egnn(feats, coords, mask=mask)
    assert out_feats.shape == feats.shape
    assert out_coords.shape == coords.shape


def test_protocol_a_jid_disjoint():
    """Protocol A data-incremental snapshots are JID-disjoint within each property."""
    tasks, task_records, _ = build_protocol_a(n_train_val_per_task=500)
    assert len(tasks) == 4
    # A1 and A2 are formation-energy snapshots; A3 and A4 are band-gap snapshots.
    a1_jids = {r["jid"] for r in task_records[0]}
    a2_jids = {r["jid"] for r in task_records[1]}
    a3_jids = {r["jid"] for r in task_records[2]}
    a4_jids = {r["jid"] for r in task_records[3]}
    assert not (a1_jids & a2_jids), "A1 and A2 share JIDs"
    assert not (a3_jids & a4_jids), "A3 and A4 share JIDs"

    # Train/val/test within each task are formula-disjoint.
    for recs in task_records:
        train_formulas = {r["formula"] for r in recs if r["split"] == "train"}
        val_formulas = {r["formula"] for r in recs if r["split"] == "val"}
        test_formulas = {r["formula"] for r in recs if r["split"] == "test"}
        assert not (train_formulas & val_formulas)
        assert not (train_formulas & test_formulas)
        assert not (val_formulas & test_formulas)


def test_protocol_b_paired_splits():
    """Protocol B OPT/MBJ records share JIDs and splits within each JARVIS version."""
    tasks, task_records, _ = build_protocol_b(n_train_val_per_task=500)
    assert len(tasks) == 4
    for opt_recs, mbj_recs in [(task_records[0], task_records[1]), (task_records[2], task_records[3])]:
        opt_map = {r["jid"]: r["split"] for r in opt_recs}
        for r in mbj_recs:
            assert r["jid"] in opt_map, f"MBJ record {r['jid']} missing from OPT task"
            assert opt_map[r["jid"]] == r["split"], f"MBJ record {r['jid']} has mismatched split"

    # Train/val/test within each task are formula-disjoint.
    for recs in task_records:
        train_formulas = {r["formula"] for r in recs if r["split"] == "train"}
        val_formulas = {r["formula"] for r in recs if r["split"] == "val"}
        test_formulas = {r["formula"] for r in recs if r["split"] == "test"}
        assert not (train_formulas & val_formulas)
        assert not (train_formulas & test_formulas)
        assert not (val_formulas & test_formulas)

    for recs in task_records:
        assert all(parse_target(r["target"]) is not None for r in recs)


def test_protocol_b_both_fidelities_exist_before_split():
    """Before formula-disjoint splitting, the two fidelities share structures."""
    d21 = load_jarvis_dataset("dft_3d_2021")
    d22 = load_jarvis_dataset("dft_3d")

    shared_2021 = 0
    for r in d21:
        opt = parse_target(r.get("optb88vdw_bandgap"))
        mbj = parse_target(r.get("mbj_bandgap"))
        if opt is not None and mbj is not None:
            shared_2021 += 1
    shared_2022 = 0
    for r in d22:
        opt = parse_target(r.get("optb88vdw_bandgap"))
        mbj = parse_target(r.get("mbj_bandgap"))
        if opt is not None and mbj is not None:
            shared_2022 += 1

    assert shared_2021 > 1000
    assert shared_2022 > 1000


# ---------------------------------------------------------------------------
# Multi-layer periodic boundary tests
# ---------------------------------------------------------------------------


def _make_si_structure():
    """Return a simple cubic Si structure for PBC tests."""
    lattice = Lattice.cubic(5.43)
    return Structure(lattice, ["Si", "Si"], [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])


def _run_egnn_through_model(graph: dict, n_layers: int) -> torch.Tensor:
    """Run a small PhyTCA-style EGNN stack and return pooled output."""
    pytest.importorskip("egnn_pytorch")
    from egnn_pytorch import EGNN

    feats = graph["node_feats"].unsqueeze(0)
    coords = graph["coords"].unsqueeze(0)
    mask = torch.ones(feats.size(1), dtype=torch.bool).unsqueeze(0)
    original_mask = graph["original_mask"].unsqueeze(0)

    h = feats
    egnn = EGNN(dim=92, m_dim=16, num_nearest_neighbors=8, update_coors=False, update_feats=True)
    for _ in range(n_layers):
        h, _ = egnn(h, coords, mask=mask)
    pooled = (h * original_mask.unsqueeze(-1).float()).sum(dim=1) / original_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return pooled


def test_pbc_invariance_depth_1():
    """Pooled EGNN output is invariant to unit-cell translation at depth 1."""
    struct = _make_si_structure()
    struct_translated = Structure(
        struct.lattice, ["Si", "Si"], np.array(struct.frac_coords) + np.array([0.1, 0.2, 0.3])
    )
    g1 = build_periodic_graph(struct, np.diag([2, 2, 2]))
    g2 = build_periodic_graph(struct_translated, np.diag([2, 2, 2]))
    torch.manual_seed(0)
    out1 = _run_egnn_through_model(g1, n_layers=1)
    torch.manual_seed(0)
    out2 = _run_egnn_through_model(g2, n_layers=1)
    assert torch.allclose(out1, out2, atol=1e-4)


def test_pbc_invariance_depth_2():
    """Pooled EGNN output is invariant to unit-cell translation at depth 2."""
    struct = _make_si_structure()
    struct_translated = Structure(
        struct.lattice, ["Si", "Si"], np.array(struct.frac_coords) + np.array([0.1, 0.2, 0.3])
    )
    g1 = build_periodic_graph(struct, np.diag([2, 2, 2]))
    g2 = build_periodic_graph(struct_translated, np.diag([2, 2, 2]))
    torch.manual_seed(0)
    out1 = _run_egnn_through_model(g1, n_layers=2)
    torch.manual_seed(0)
    out2 = _run_egnn_through_model(g2, n_layers=2)
    assert torch.allclose(out1, out2, atol=1e-4)


def test_pbc_invariance_depth_4():
    """Pooled EGNN output is invariant to unit-cell translation at depth 4."""
    struct = _make_si_structure()
    struct_translated = Structure(
        struct.lattice, ["Si", "Si"], np.array(struct.frac_coords) + np.array([0.1, 0.2, 0.3])
    )
    g1 = build_periodic_graph(struct, np.diag([2, 2, 2]))
    g2 = build_periodic_graph(struct_translated, np.diag([2, 2, 2]))
    torch.manual_seed(0)
    out1 = _run_egnn_through_model(g1, n_layers=4)
    torch.manual_seed(0)
    out2 = _run_egnn_through_model(g2, n_layers=4)
    assert torch.allclose(out1, out2, atol=1e-4)


def test_periodic_halo_convergence():
    """Pooled output converges as the supercell halo size increases."""
    struct = _make_si_structure()
    torch.manual_seed(0)
    out2 = _run_egnn_through_model(build_periodic_graph(struct, np.diag([2, 2, 2])), n_layers=4)
    torch.manual_seed(0)
    out3 = _run_egnn_through_model(build_periodic_graph(struct, np.diag([3, 3, 3])), n_layers=4)
    torch.manual_seed(0)
    out4 = _run_egnn_through_model(build_periodic_graph(struct, np.diag([4, 4, 4])), n_layers=4)

    # 3x3x3 and 4x4x4 should agree closely if the halo is converged.
    assert torch.allclose(out3, out4, atol=1e-3)
    # 2x2x2 is allowed to differ slightly; this records the gap.
    delta_23 = float(torch.norm((out2 - out3).detach()))
    print(f"halo convergence delta (2x2x2 vs 3x3x3): {delta_23:.6f}")


def test_primitive_vs_supercell_depth_4():
    """A primitive cell and its 2x2x2 supercell produce the same pooled output."""
    lattice = Lattice.cubic(2.715)
    primitive = Structure(lattice, ["Si"], [[0.0, 0.0, 0.0]])
    supercell = primitive * np.diag([2, 2, 2])

    torch.manual_seed(0)
    out_prim = _run_egnn_through_model(build_periodic_graph(primitive, np.diag([3, 3, 3])), n_layers=4)
    torch.manual_seed(0)
    out_super = _run_egnn_through_model(build_periodic_graph(supercell, np.diag([2, 2, 2])), n_layers=4)
    assert torch.allclose(out_prim, out_super, atol=1e-3)


def test_no_duplicate_periodic_edges():
    """Each (original_atom, image_offset) pair is unique in the supercell."""
    struct = _make_si_structure()
    graph = build_periodic_graph(struct, np.diag([2, 2, 2]))
    offsets = graph["image_offsets"].numpy()
    original_indices = graph["original_indices"].numpy()
    pairs = list(zip(original_indices, map(tuple, offsets)))
    assert len(pairs) == len(set(pairs))


def test_periodic_image_self_loop_handling():
    """Original atoms are not counted as their own periodic images."""
    struct = _make_si_structure()
    graph = build_periodic_graph(struct, np.diag([2, 2, 2]))
    original_mask = graph["original_mask"]
    offsets = graph["image_offsets"]
    # Original atoms have zero offset.
    assert torch.equal(original_mask, (offsets == 0).all(dim=1))
    # There are exactly n_original original atoms.
    assert original_mask.sum() == graph["n_original"]
