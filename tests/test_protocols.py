"""Tests for protocols.py builders."""

from __future__ import annotations

import pytest

from protocols import (
    build_addition_protocol,
    build_combined_protocol,
    build_fidelity_expansion_protocol,
    build_revision_protocol,
)


@pytest.mark.slow
def test_build_revision_protocol_runs():
    tasks, task_records, audit = build_revision_protocol(
        properties=("band_gap",),
        fidelities=("OptB88vdW",),
        n_train_val_per_task=50,
    )
    assert audit["protocol"] == "revision"
    assert len(tasks) == len(task_records)
    assert len(tasks) > 0
    # Every task is a retained-material endpoint.
    for recs in task_records:
        assert len(recs) > 0


@pytest.mark.slow
def test_build_addition_protocol_added_only():
    tasks, task_records, audit = build_addition_protocol(
        properties=("band_gap",),
        fidelities=("OptB88vdW",),
        n_train_val_per_task=50,
    )
    assert audit["protocol"] == "addition"
    assert len(tasks) == len(task_records)
    # Second task per property/fidelity should be from 2022 added materials.
    for i, (version, _, _, _) in enumerate(tasks):
        if i % 2 == 1:
            assert version == "dft_3d"


@pytest.mark.slow
def test_build_fidelity_expansion_protocol_paired():
    tasks, task_records, audit = build_fidelity_expansion_protocol(
        version="dft_3d_2021",
        properties=("band_gap",),
        fidelities=("OptB88vdW", "TB-mBJ"),
        n_train_val_per_task=50,
    )
    assert audit["protocol"] == "fidelity_expansion"
    assert len(tasks) == 2
    assert len(task_records[0]) == len(task_records[1])


@pytest.mark.slow
def test_build_combined_protocol_uses_added_materials_for_2022():
    tasks, task_records, audit = build_combined_protocol(
        properties=("band_gap",),
        fidelities=("OptB88vdW", "TB-mBJ"),
        n_train_val_per_task=50,
    )
    # 2022 tasks should contain only added materials (no overlap with 2021 JIDs).
    jids_2021: set[str] = set()
    jids_2022: set[str] = set()
    for (version, _, _, _), recs in zip(tasks, task_records):
        if version == "dft_3d_2021":
            jids_2021.update(r["jid"] for r in recs)
        elif version == "dft_3d":
            jids_2022.update(r["jid"] for r in recs)
    assert jids_2021
    assert jids_2022
    assert jids_2021.isdisjoint(jids_2022)


@pytest.mark.slow
def test_build_revision_protocol_propagates_change_type():
    tasks, task_records, audit = build_revision_protocol(
        properties=("band_gap",),
        fidelities=("OptB88vdW",),
        n_train_val_per_task=None,
    )
    # 2022 tasks should have change_type annotations, not 'unknown'.
    for (version, _, _, _), recs in zip(tasks, task_records):
        if version == "dft_3d":
            types = {r.get("change_type", "unknown") for r in recs}
            assert "unknown" not in types, f"found unannotated records: {types}"
            assert "added" not in types, "revision protocol should not include added materials"
    tasks, task_records, audit = build_combined_protocol(
        properties=("band_gap",),
        fidelities=("OptB88vdW", "TB-mBJ"),
        n_train_val_per_task=50,
    )
    assert audit["protocol"] == "combined"
    assert len(tasks) == len(task_records)
    # Expect 4 tasks: 2021 OPT, 2021 MBJ, 2022 OPT, 2022 MBJ.
    assert len(tasks) == 4
    assert tasks[0][0] == "dft_3d_2021"
    assert tasks[1][0] == "dft_3d_2021"
    assert tasks[2][0] == "dft_3d"
    assert tasks[3][0] == "dft_3d"
