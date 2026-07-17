"""Snapshot diff for evolving materials databases.

This module classifies records between two database snapshots into five
semantic categories:

* ``added``      : present only in the newer snapshot;
* ``removed``    : present only in the older snapshot;
* ``unchanged``  : retained JID with matching structure and matching targets;
* ``label_revised``: retained JID with matching structure but revised targets;
* ``structure_revised``: retained JID whose structure changed.

For ``label_revised`` the magnitude threshold is configurable and the summary
reports multiple thresholds simultaneously, as requested in the ICLR upgrade.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher

from data import jarvis_record_to_structure, parse_target


ChangeType = str


def build_jid_index(records: Sequence[dict]) -> dict[str, dict]:
    """Return a JID-indexed dict, keeping the first occurrence for duplicates."""
    out: dict[str, dict] = {}
    for r in records:
        jid = r.get("jid")
        if jid is None:
            continue
        if jid not in out:
            out[jid] = r
    return out


def _coarse_structure_signature(record: dict) -> str:
    """Cheap structural signature used for pre-filtering.

    Combines reduced composition, number of sites, and volume per atom.
    """
    from math import gcd
    from functools import reduce

    atoms = record["atoms"]
    counts: dict[str, int] = {}
    for el in atoms["elements"]:
        counts[el] = counts.get(el, 0) + 1

    vals = list(counts.values())
    if vals:
        g = reduce(gcd, vals)
        reduced = {el: c // g for el, c in counts.items()}
    else:
        reduced = counts

    formula = "".join(f"{el}{reduced[el]}" for el in sorted(reduced))
    lattice = np.array(atoms["lattice_mat"], dtype=float)
    volume = abs(float(np.linalg.det(lattice)))
    n_sites = len(atoms["elements"])
    vpa = round(volume / n_sites, 3) if n_sites > 0 else 0.0
    return f"{formula}|n={n_sites}|vpa={vpa}"


def _label_revisions(
    r_prev: dict,
    r_next: dict,
    target_fields: Sequence[str],
    thresholds: Sequence[float],
) -> dict[str, bool]:
    """For each threshold, report whether any target changed by more than it."""
    revisions: dict[str, bool] = {}
    for thr in thresholds:
        revised = False
        for field in target_fields:
            v_prev = parse_target(r_prev.get(field))
            v_next = parse_target(r_next.get(field))
            if (v_prev is None) != (v_next is None):
                revised = True
                break
            if v_prev is not None and v_next is not None and abs(v_prev - v_next) > thr:
                revised = True
                break
        revisions[f"threshold_{thr}"] = revised
    return revisions


def _structures_match(r_prev: dict, r_next: dict, matcher: StructureMatcher) -> bool:
    """Return True iff the two records describe the same structure."""
    try:
        s_prev = jarvis_record_to_structure(r_prev)
        s_next = jarvis_record_to_structure(r_next)
    except Exception:  # malformed structure
        return False
    return matcher.fit(s_prev, s_next)


def classify_records(
    d_prev: Sequence[dict],
    d_next: Sequence[dict],
    target_fields: Sequence[str],
    thresholds: Sequence[float] = (1e-4, 1e-3, 1e-2),
    skip_structure_match: bool = False,
) -> tuple[dict[str, Any], list[dict], list[dict]]:
    """Classify records between two snapshots.

    Args:
        d_prev: Records from the older snapshot.
        d_next: Records from the newer snapshot.
        target_fields: Target fields used to detect label revisions.
        thresholds: Thresholds (in target units) for label revision.
        skip_structure_match: If True, skip expensive StructureMatcher calls and
            classify all retained records by label only. Useful when only added/
            removed counts are needed.

    Returns:
        summary: Dict with counts and per-threshold revision statistics.
        next_annotated: ``d_next`` records annotated with ``change_type``,
            ``matched_prev_jid``, ``label_revisions``, and ``structure_match``.
        removed: ``d_prev`` records whose JID does not appear in ``d_next``.
    """
    prev_by_jid = build_jid_index(d_prev)
    next_by_jid = build_jid_index(d_next)

    matcher = StructureMatcher()

    retained_jids = set(next_by_jid.keys()) & set(prev_by_jid.keys())
    added_jids = set(next_by_jid.keys()) - set(prev_by_jid.keys())
    removed_jids = set(prev_by_jid.keys()) - set(next_by_jid.keys())

    counts: dict[str, int] = defaultdict(int)
    per_threshold_counts: dict[str, int] = defaultdict(int)

    next_annotated: list[dict] = []

    # Process retained records (from the newer snapshot).
    for jid in retained_jids:
        r_prev = prev_by_jid[jid]
        r_next = next_by_jid[jid]

        label_revisions = _label_revisions(r_prev, r_next, target_fields, thresholds)

        # Pre-filter with coarse signature before expensive StructureMatcher.
        if skip_structure_match:
            struct_match = True
        else:
            coarse_match = _coarse_structure_signature(r_prev) == _coarse_structure_signature(r_next)
            struct_match = coarse_match and _structures_match(r_prev, r_next, matcher)

        # Determine change type.
        any_label_rev = any(label_revisions.values())
        if not struct_match:
            if any_label_rev:
                change_type = "label_and_structure_revised"
            else:
                change_type = "structure_revised"
        elif any_label_rev:
            change_type = "label_revised"
        else:
            change_type = "unchanged"

        counts[change_type] += 1
        for key, val in label_revisions.items():
            if val:
                per_threshold_counts[key] += 1

        annotated = dict(r_next)
        annotated["change_type"] = change_type
        annotated["matched_prev_jid"] = jid
        annotated["label_revisions"] = label_revisions
        annotated["structure_match"] = struct_match
        next_annotated.append(annotated)

    # Process added records.
    for jid in added_jids:
        annotated = dict(next_by_jid[jid])
        annotated["change_type"] = "added"
        annotated["matched_prev_jid"] = None
        annotated["label_revisions"] = {}
        annotated["structure_match"] = False
        next_annotated.append(annotated)
        counts["added"] += 1

    # Process removed records (from the older snapshot).
    removed: list[dict] = []
    for jid in removed_jids:
        rec = dict(prev_by_jid[jid])
        rec["change_type"] = "removed"
        rec["matched_prev_jid"] = None
        rec["label_revisions"] = {}
        rec["structure_match"] = False
        removed.append(rec)
        counts["removed"] += 1

    summary: dict[str, Any] = {
        "n_prev": len(d_prev),
        "n_next": len(d_next),
        "unique_prev_jids": len(prev_by_jid),
        "unique_next_jids": len(next_by_jid),
        "retained_jids": len(retained_jids),
        "added_jids": len(added_jids),
        "removed_jids": len(removed_jids),
        "counts": dict(counts),
        "label_revisions_by_threshold": dict(per_threshold_counts),
    }

    return summary, next_annotated, removed


def filter_by_change_type(records: Sequence[dict], change_type: str | Sequence[str]) -> list[dict]:
    """Return records whose ``change_type`` matches one of the requested types."""
    if isinstance(change_type, str):
        change_type = (change_type,)
    wanted = set(change_type)
    return [r for r in records if r.get("change_type") in wanted]


def classify_jarvis_snapshots(
    cache_dir: str | None = None,
    target_fields: Sequence[str] | None = None,
    thresholds: Sequence[float] = (1e-4, 1e-3, 1e-2),
    skip_structure_match: bool = False,
) -> tuple[dict[str, Any], list[dict], list[dict]]:
    """Convenience wrapper for JARVIS 2021 -> 2022 snapshot diff.

    Args:
        cache_dir: JARVIS cache directory.
        target_fields: Fields to check for label revision. Defaults to all scalar
            target fields used in ``data.TARGET_FIELDS``.
        thresholds: Label-revision thresholds.
        skip_structure_match: If True, skip StructureMatcher.

    Returns:
        Same tuple as :func:`classify_records`.
    """
    from data import TARGET_FIELDS, load_jarvis_dataset

    if target_fields is None:
        target_fields = sorted({field for field in TARGET_FIELDS.values()})

    d21 = load_jarvis_dataset("dft_3d_2021", cache_dir)
    d22 = load_jarvis_dataset("dft_3d", cache_dir)
    return classify_records(d21, d22, target_fields, thresholds, skip_structure_match)
