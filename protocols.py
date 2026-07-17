"""Protocol builders that consume snapshot_diff output.

These builders produce training/evaluation task sequences for the three
single-axis protocols (revision, addition, fidelity expansion) and the
combined graph protocol used in the PCG benchmark.
"""

from __future__ import annotations

from typing import Sequence

from data import TARGET_FIELDS, assign_global_splits, load_jarvis_dataset
from snapshot_diff import classify_records


def _records_for_property_fidelity(
    records: Sequence[dict],
    property_name: str,
    fidelity_name: str,
    target_field: str,
    dataset_tag: str,
) -> list[dict]:
    """Filter raw JARVIS records and attach versioned metadata."""
    from data import jarvis_record_to_structure, parse_target

    out: list[dict] = []
    for r in records:
        val = parse_target(r.get(target_field))
        if val is None:
            continue
        struct = jarvis_record_to_structure(r)
        out.append(
            {
                "jid": r.get("jid"),
                "structure": struct,
                "formula": struct.composition.reduced_formula,
                "dataset": dataset_tag,
                "version": dataset_tag,
                "property": property_name,
                "fidelity": fidelity_name,
                "target": val,
                "change_type": r.get("change_type", "unknown"),
            }
        )
    return out


def build_revision_protocol(
    properties: Sequence[str] = ("band_gap",),
    fidelities: Sequence[str] = ("OptB88vdW", "TB-mBJ"),
    cache_dir: str | None = None,
    seed: int = 42,
    n_train_val_per_task: int | None = None,
) -> tuple[list[tuple[str, str, str, str]], list[list[dict]], dict]:
    """Build a revision-only protocol.

    Sequence for each (property, fidelity):
      1. JARVIS-2021 endpoint trained on materials retained in 2022.
      2. JARVIS-2022 endpoint trained on the *same* retained materials,
         whose labels may have been revised.

    A global material-group split is shared across the two endpoints so that
    the same material always falls in the same partition.
    """
    d21 = load_jarvis_dataset("dft_3d_2021", cache_dir)
    d22 = load_jarvis_dataset("dft_3d", cache_dir)

    target_fields = [TARGET_FIELDS[(p, f)] for p in properties for f in fidelities if (p, f) in TARGET_FIELDS]
    summary, annotated_next, _ = classify_records(d21, d22, target_fields)

    retained_jids = {r["jid"] for r in annotated_next if r["change_type"] != "added"}
    retained_21 = [r for r in d21 if r.get("jid") in retained_jids]
    retained_22 = [r for r in d22 if r.get("jid") in retained_jids]

    tasks: list[tuple[str, str, str, str]] = []
    task_records: list[list[dict]] = []

    for prop in properties:
        for fid in fidelities:
            target_field = TARGET_FIELDS.get((prop, fid))
            if target_field is None:
                continue
            recs_21 = _records_for_property_fidelity(
                retained_21, prop, fid, target_field, "dft_3d_2021"
            )
            recs_22 = _records_for_property_fidelity(
                retained_22, prop, fid, target_field, "dft_3d"
            )
            if not recs_21 or not recs_22:
                continue

            all_recs = recs_21 + recs_22
            assign_global_splits(all_recs, seed=seed)

            if n_train_val_per_task is not None:
                recs_21 = recs_21[:n_train_val_per_task]
                recs_22 = recs_22[:n_train_val_per_task]

            tasks.append(("dft_3d_2021", prop, fid, target_field))
            task_records.append(recs_21)
            tasks.append(("dft_3d", prop, fid, target_field))
            task_records.append(recs_22)

    audit = {
        "protocol": "revision",
        "summary": summary,
        "n_tasks": len(tasks),
    }
    return tasks, task_records, audit


def build_addition_protocol(
    properties: Sequence[str] = ("band_gap",),
    fidelities: Sequence[str] = ("OptB88vdW", "TB-mBJ"),
    cache_dir: str | None = None,
    seed: int = 42,
    n_train_val_per_task: int | None = None,
) -> tuple[list[tuple[str, str, str, str]], list[list[dict]], dict]:
    """Build an addition-only protocol.

    Sequence for each (property, fidelity):
      1. JARVIS-2021 endpoint trained on all 2021 materials.
      2. JARVIS-2022 endpoint trained *only* on materials added in 2022.

    This isolates the data-incremental (addition) axis from label revision.
    """
    d21 = load_jarvis_dataset("dft_3d_2021", cache_dir)
    d22 = load_jarvis_dataset("dft_3d", cache_dir)

    target_fields = [TARGET_FIELDS[(p, f)] for p in properties for f in fidelities if (p, f) in TARGET_FIELDS]
    summary, annotated_next, _ = classify_records(
        d21, d22, target_fields, skip_structure_match=True
    )

    added_jids = {r["jid"] for r in annotated_next if r["change_type"] == "added"}
    added_22 = [r for r in d22 if r.get("jid") in added_jids]

    tasks: list[tuple[str, str, str, str]] = []
    task_records: list[list[dict]] = []

    for prop in properties:
        for fid in fidelities:
            target_field = TARGET_FIELDS.get((prop, fid))
            if target_field is None:
                continue
            recs_21 = _records_for_property_fidelity(
                d21, prop, fid, target_field, "dft_3d_2021"
            )
            recs_22_added = _records_for_property_fidelity(
                added_22, prop, fid, target_field, "dft_3d"
            )
            if not recs_21 or not recs_22_added:
                continue

            all_recs = recs_21 + recs_22_added
            assign_global_splits(all_recs, seed=seed)

            if n_train_val_per_task is not None:
                recs_21 = recs_21[:n_train_val_per_task]
                recs_22_added = recs_22_added[:n_train_val_per_task]

            tasks.append(("dft_3d_2021", prop, fid, target_field))
            task_records.append(recs_21)
            tasks.append(("dft_3d", prop, fid, target_field))
            task_records.append(recs_22_added)

    audit = {
        "protocol": "addition",
        "summary": summary,
        "n_tasks": len(tasks),
    }
    return tasks, task_records, audit


def build_fidelity_expansion_protocol(
    version: str = "dft_3d_2021",
    properties: Sequence[str] = ("band_gap",),
    fidelities: Sequence[str] = ("OptB88vdW", "TB-mBJ"),
    cache_dir: str | None = None,
    seed: int = 42,
    n_train_val_per_task: int | None = None,
) -> tuple[list[tuple[str, str, str, str]], list[list[dict]], dict]:
    """Build a fidelity-expansion protocol.

    Within a single snapshot, train a low-fidelity endpoint first, then a
    high-fidelity endpoint. Only materials with both fidelities are retained.
    """
    d = load_jarvis_dataset(version, cache_dir)

    tasks: list[tuple[str, str, str, str]] = []
    task_records: list[list[dict]] = []

    for prop in properties:
        recs_by_fid: dict[str, list[dict]] = {}
        for fid in fidelities:
            target_field = TARGET_FIELDS.get((prop, fid))
            if target_field is None:
                continue
            recs_by_fid[fid] = _records_for_property_fidelity(
                d, prop, fid, target_field, version
            )

        if len(recs_by_fid) < 2:
            continue

        # Keep only JIDs present in all selected fidelities.
        common_jids = None
        for fid, recs in recs_by_fid.items():
            jids = {r["jid"] for r in recs}
            common_jids = jids if common_jids is None else common_jids & jids

        filtered: dict[str, list[dict]] = {}
        for fid, recs in recs_by_fid.items():
            filtered[fid] = [r for r in recs if r["jid"] in common_jids]

        all_recs = [r for recs in filtered.values() for r in recs]
        assign_global_splits(all_recs, seed=seed)

        for fid in fidelities:
            if fid not in filtered:
                continue
            target_field = TARGET_FIELDS[(prop, fid)]
            recs = filtered[fid]
            if n_train_val_per_task is not None:
                recs = recs[:n_train_val_per_task]
            tasks.append((version, prop, fid, target_field))
            task_records.append(recs)

    audit = {
        "protocol": "fidelity_expansion",
        "version": version,
        "n_tasks": len(tasks),
    }
    return tasks, task_records, audit


def build_combined_protocol(
    properties: Sequence[str] = ("band_gap",),
    fidelities: Sequence[str] = ("OptB88vdW", "TB-mBJ"),
    cache_dir: str | None = None,
    seed: int = 42,
    n_train_val_per_task: int | None = None,
) -> tuple[list[tuple[str, str, str, str]], list[list[dict]], dict]:
    """Build the combined three-axis protocol used for the main benchmark.

    The canonical order is:
      1. 2021 low fidelity
      2. 2021 high fidelity
      3. 2022 low fidelity  (with parent edges to 2022 revision and 2021 high fidelity)
      4. 2022 high fidelity
    """
    tasks, task_records, audit = build_revision_protocol(
        properties=properties,
        fidelities=fidelities,
        cache_dir=cache_dir,
        seed=seed,
        n_train_val_per_task=n_train_val_per_task,
    )

    # Reorder to interleave fidelities within each version.
    ordered_tasks: list[tuple[str, str, str, str]] = []
    ordered_records: list[list[dict]] = []
    version_order = ["dft_3d_2021", "dft_3d"]
    for version in version_order:
        for fid in fidelities:
            for prop in properties:
                for i, (v, p, f, tf) in enumerate(tasks):
                    if v == version and p == prop and f == fid:
                        ordered_tasks.append((v, p, f, tf))
                        ordered_records.append(task_records[i])
                        break

    audit["protocol"] = "combined"
    return ordered_tasks, ordered_records, audit
