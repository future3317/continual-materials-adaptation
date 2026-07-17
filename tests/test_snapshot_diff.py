"""Tests for snapshot_diff.py."""

from __future__ import annotations

import pytest

from snapshot_diff import (
    build_jid_index,
    classify_jarvis_snapshots,
    classify_records,
    filter_by_change_type,
)


def _make_record(jid: str, elements: list[str], lattice_mat: list[list[float]], coords: list[list[float]], target: float | None = None, field: str = "optb88vdw_bandgap") -> dict:
    return {
        "jid": jid,
        "atoms": {
            "elements": elements,
            "lattice_mat": lattice_mat,
            "coords": coords,
            "cartesian": True,
        },
        field: target,
    }


def test_build_jid_index_keeps_first_occurrence():
    r1 = {"jid": "j1", "v": 1}
    r2 = {"jid": "j1", "v": 2}
    idx = build_jid_index([r1, r2])
    assert idx["j1"]["v"] == 1


def test_classify_added_and_removed():
    prev = [_make_record("j1", ["Si"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 0, 0]], 1.0)]
    next_records = [_make_record("j2", ["Ge"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 0, 0]], 2.0)]
    summary, annotated, removed = classify_records(prev, next_records, ["optb88vdw_bandgap"])
    assert summary["counts"]["added"] == 1
    assert summary["counts"]["removed"] == 1
    assert len(annotated) == 1
    assert annotated[0]["change_type"] == "added"
    assert len(removed) == 1


def test_classify_unchanged():
    rec = _make_record("j1", ["Si"], [[2, 0, 0], [0, 2, 0], [0, 0, 2]], [[0, 0, 0]], 1.0)
    summary, annotated, _ = classify_records([rec], [rec], ["optb88vdw_bandgap"])
    assert summary["counts"]["unchanged"] == 1
    assert annotated[0]["change_type"] == "unchanged"


def test_classify_label_revised():
    prev = [_make_record("j1", ["Si"], [[2, 0, 0], [0, 2, 0], [0, 0, 2]], [[0, 0, 0]], 1.0)]
    next_records = [_make_record("j1", ["Si"], [[2, 0, 0], [0, 2, 0], [0, 0, 2]], [[0, 0, 0]], 1.5)]
    summary, annotated, _ = classify_records(prev, next_records, ["optb88vdw_bandgap"], thresholds=(1e-4,))
    assert summary["counts"]["label_revised"] == 1
    assert annotated[0]["label_revisions"]["threshold_0.0001"] is True


def test_classify_structure_revised():
    prev = [_make_record("j1", ["Si"], [[2, 0, 0], [0, 2, 0], [0, 0, 2]], [[0, 0, 0]], 1.0)]
    next_records = [_make_record("j1", ["Si"], [[3, 0, 0], [0, 3, 0], [0, 0, 3]], [[0, 0, 0]], 1.0)]
    summary, annotated, _ = classify_records(prev, next_records, ["optb88vdw_bandgap"])
    assert summary["counts"]["structure_revised"] == 1
    assert annotated[0]["structure_match"] is False


def test_filter_by_change_type():
    prev = [
        _make_record("j1", ["Si"], [[2, 0, 0], [0, 2, 0], [0, 0, 2]], [[0, 0, 0]], 1.0),
        _make_record("j2", ["Ge"], [[2, 0, 0], [0, 2, 0], [0, 0, 2]], [[0, 0, 0]], 2.0),
    ]
    next_records = [
        _make_record("j1", ["Si"], [[2, 0, 0], [0, 2, 0], [0, 0, 2]], [[0, 0, 0]], 1.5),
        _make_record("j2", ["Ge"], [[2, 0, 0], [0, 2, 0], [0, 0, 2]], [[0, 0, 0]], 2.0),
    ]
    _, annotated, _ = classify_records(prev, next_records, ["optb88vdw_bandgap"], thresholds=(1e-4,))
    revised = filter_by_change_type(annotated, "label_revised")
    unchanged = filter_by_change_type(annotated, "unchanged")
    assert len(revised) == 1
    assert len(unchanged) == 1


@pytest.mark.slow
@pytest.mark.parametrize("thresholds", [(1e-4, 1e-3, 1e-2), (1e-6,)])
def test_classify_jarvis_snapshots_runs(thresholds):
    """Real-data smoke test: classification runs and counts are consistent."""
    summary, annotated, removed = classify_jarvis_snapshots(thresholds=thresholds)
    n_next = summary["n_next"]
    n_removed = len(removed)
    n_annotated = len(annotated)

    # Every next record must be annotated exactly once.
    assert n_annotated == n_next

    # Five-class consistency: retained + added == next, retained + removed == prev.
    counts = summary["counts"]
    retained = counts.get("unchanged", 0) + counts.get("label_revised", 0) + counts.get("structure_revised", 0) + counts.get("label_and_structure_revised", 0)
    assert retained + counts.get("added", 0) == summary["unique_next_jids"]
    assert retained + n_removed == summary["unique_prev_jids"]

    # Thresholds should be monotonic: smaller threshold detects at least as many revisions.
    if len(thresholds) > 1:
        rev_counts = summary["label_revisions_by_threshold"]
        vals = [rev_counts.get(f"threshold_{t}", 0) for t in thresholds]
        assert vals == sorted(vals, reverse=True)
