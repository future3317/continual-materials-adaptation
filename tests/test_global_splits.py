"""Tests for canonical material group splits and global split assignment."""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from data import (
    assign_global_splits,
    build_protocol_a,
    build_protocol_b,
    canonical_material_group_id,
)


def test_canonical_group_id_supercell_invariant():
    """Primitive cell and its 2x2x2 supercell share the same group id."""
    lattice = Lattice.cubic(2.715)
    primitive = Structure(lattice, ["Si"], [[0.0, 0.0, 0.0]])
    supercell = primitive * np.diag([2, 2, 2])

    gid_prim = canonical_material_group_id(primitive)
    gid_super = canonical_material_group_id(supercell)
    assert gid_prim == gid_super
    assert "Si" in gid_prim


def test_canonical_group_id_conventional_vs_primitive():
    """A conventional FCC cell and its primitive cell share the same group id."""
    conv = Structure(Lattice.cubic(4.0), ["Si", "Si", "Si", "Si"], [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ])
    prim = conv.get_primitive_structure()
    gid_conv = canonical_material_group_id(conv)
    gid_prim = canonical_material_group_id(prim)
    assert gid_conv == gid_prim


def test_assign_global_splits_no_group_leakage():
    """No canonical group id appears in more than one split."""
    lattice = Lattice.cubic(4.0)
    records = []
    for i in range(20):
        struct = Structure(lattice, ["Si"], [[0.0, 0.0, 0.1 * i]])
        records.append({"structure": struct})

    assign_global_splits(records, seed=123)

    groups_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for r in records:
        groups_by_split[r["split"]].add(r["group_id"])

    assert not (groups_by_split["train"] & groups_by_split["val"])
    assert not (groups_by_split["train"] & groups_by_split["test"])
    assert not (groups_by_split["val"] & groups_by_split["test"])
    assert all(r["split"] in {"train", "val", "test"} for r in records)


def test_protocol_b_global_split_same_jid_across_years():
    """A JID present in both 2021 and 2022 receives the same split."""
    tasks, task_records, audit = build_protocol_b(n_train_val_per_task=500)

    # Map JID -> split across all four tasks.
    jid_to_split: dict[str, str] = {}
    for recs in task_records:
        for r in recs:
            jid = r["jid"]
            if jid in jid_to_split:
                assert jid_to_split[jid] == r["split"], (
                    f"JID {jid} has mismatched splits across years/fidelities"
                )
            else:
                jid_to_split[jid] = r["split"]


def test_protocol_b_opt_mbj_same_split():
    """Paired OPT/MBJ records always share the same split."""
    tasks, task_records, audit = build_protocol_b(n_train_val_per_task=500)

    opt_21, mbj_21, opt_22, mbj_22 = task_records
    for opt_recs, mbj_recs in [(opt_21, mbj_21), (opt_22, mbj_22)]:
        opt_map = {r["jid"]: r["split"] for r in opt_recs}
        for r in mbj_recs:
            assert r["jid"] in opt_map
            assert opt_map[r["jid"]] == r["split"]


def test_protocol_b_no_group_id_in_multiple_splits():
    """No canonical group id appears in both train and test for Protocol B."""
    tasks, task_records, audit = build_protocol_b(n_train_val_per_task=500)

    groups_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for recs in task_records:
        for r in recs:
            groups_by_split[r["split"]].add(r["group_id"])

    assert not (groups_by_split["train"] & groups_by_split["val"])
    assert not (groups_by_split["train"] & groups_by_split["test"])
    assert not (groups_by_split["val"] & groups_by_split["test"])


def test_protocol_b_revision_audit_documented():
    """Protocol B audit reports revision counts between 2021 and 2022."""
    tasks, task_records, audit = build_protocol_b(n_train_val_per_task=500)

    required_keys = [
        "unchanged_jids",
        "new_jids",
        "revised_structures",
        "revised_targets",
    ]
    for key in required_keys:
        assert key in audit, f"Missing audit key: {key}"
        assert isinstance(audit[key], int)
        assert audit[key] >= 0


def test_protocol_a_global_split_same_jid_same_split():
    """With global_split=True, the same JID across A1/A3 keeps the same split."""
    tasks, task_records, audit = build_protocol_a(
        n_train_val_per_task=500, global_split=True
    )

    a1, a2, a3, a4 = task_records
    a1_split = {r["jid"]: r["split"] for r in a1}
    for r in a3:
        assert r["jid"] in a1_split
        assert a1_split[r["jid"]] == r["split"]

    # No group_id should leak across splits globally.
    groups_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for recs in task_records:
        for r in recs:
            groups_by_split[r["split"]].add(r["group_id"])
    assert not (groups_by_split["train"] & groups_by_split["val"])
    assert not (groups_by_split["train"] & groups_by_split["test"])
    assert not (groups_by_split["val"] & groups_by_split["test"])


def test_protocol_a_default_behavior_unchanged():
    """Default Protocol A still uses per-task formula-disjoint splits."""
    tasks, task_records, audit = build_protocol_a(n_train_val_per_task=500)

    a1, a2, a3, a4 = task_records
    # A2/A4 are added JIDs and therefore disjoint from 2021 JIDs.
    assert not {r["jid"] for r in a1} & {r["jid"] for r in a2}
    assert not {r["jid"] for r in a3} & {r["jid"] for r in a4}

    for recs in task_records:
        train_formulas = {r["formula"] for r in recs if r["split"] == "train"}
        val_formulas = {r["formula"] for r in recs if r["split"] == "val"}
        test_formulas = {r["formula"] for r in recs if r["split"] == "test"}
        assert not (train_formulas & val_formulas)
        assert not (train_formulas & test_formulas)
        assert not (val_formulas & test_formulas)
