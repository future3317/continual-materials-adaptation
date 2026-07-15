"""Audit JARVIS data and emit GO/NO-GO gate for PhyTCA training.

Usage:
    python data_audit.py --protocol a --report-dir reports
    python data_audit.py --protocol b --report-dir reports

Outputs:
    * ``manifest.json``: record identifiers, sizes, split assignments.
    * ``audit_report.md``: human-readable audit summary.
    * ``gate.json``: machine-readable GO/NO-GO decision with criteria.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from data import (
    JARVIS_DATASETS,
    PeriodicGraphBuilder,
    build_protocol_a,
    build_protocol_b,
    jarvis_record_to_structure,
    load_jarvis_dataset,
    parse_target,
)


# ---------------------------------------------------------------------------
# Audit configuration
# ---------------------------------------------------------------------------

AUDIT_CONFIG = {
    "min_samples_per_task": 1000,
    "min_valid_target_rate": 0.80,
    "max_formula_overlap_rate": 0.0,
    "protocol_target_fields": {
        "a": ["formation_energy_peratom", "optb88vdw_bandgap"],
        "b": ["optb88vdw_bandgap", "mbj_bandgap"],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_cache_dir() -> str:
    from data import _default_cache_dir as data_cache_dir
    return data_cache_dir()


def _dataset_snapshot_info(name: str, cache_dir: str | None = None) -> dict:
    """Verify local JARVIS snapshot filename/content matches the expected dataset.

    Returns a dict with the resolved filename, absolute path, size, SHA256,
    raw JSON length, and unique JID count.  If the cached file does not contain
    the expected JSON member, marks the snapshot as mismatched.
    """
    from data import _default_cache_dir

    cache_dir = cache_dir or _default_cache_dir()
    expected_zip = JARVIS_DATASETS.get(name)
    expected_json = expected_zip.replace(".zip", "") if expected_zip else f"{name}.json"
    zip_path = os.path.join(cache_dir, expected_zip) if expected_zip else None

    info = {
        "dataset_key": name,
        "expected_filename": expected_zip,
        "expected_json_member": expected_json,
        "absolute_cache_path": zip_path,
        "file_size_bytes": None,
        "sha256": None,
        "raw_json_length": None,
        "unique_jid_count": None,
        "snapshot_match": False,
    }

    if zip_path and os.path.exists(zip_path):
        info["file_size_bytes"] = os.path.getsize(zip_path)
        with open(zip_path, "rb") as f:
            data = f.read()
            info["sha256"] = hashlib.sha256(data).hexdigest()
        try:
            with zipfile.ZipFile(zip_path) as zf:
                members = zf.namelist()
                if expected_json in members:
                    info["snapshot_match"] = True
                    raw = zf.read(expected_json)
                    info["raw_json_length"] = len(raw)
                    records = json.loads(raw)
                    info["unique_jid_count"] = len({r.get("jid") for r in records})
        except (zipfile.BadZipFile, ValueError):
            pass

    return info


def _audit_dataset(records: list[dict], dataset_name: str, fields: list[str]) -> dict:
    """Compute per-dataset validity and target statistics for required fields."""
    n = len(records)

    # Structure validity is inherently record-wise; keep a compact loop.
    invalid_structures = 0
    for r in records:
        try:
            struct = jarvis_record_to_structure(r)
            if struct.volume <= 1e-6:
                invalid_structures += 1
        except Exception:
            invalid_structures += 1

    # Vectorized target-field counting.
    field_counts: dict[str, dict[str, int]] = {}
    for f in fields:
        values = np.array([r.get(f) for r in records], dtype=object)
        missing_mask = np.array(
            [v is None or (isinstance(v, str) and v.strip() == "") for v in values]
        )
        non_missing = values[~missing_mask]
        parsed = np.array([parse_target(v) for v in non_missing], dtype=float)
        nonfinite_mask = ~np.isfinite(parsed)
        field_counts[f] = {
            "valid": int((~missing_mask).sum() - nonfinite_mask.sum()),
            "missing": int(missing_mask.sum()),
            "nonfinite": int(nonfinite_mask.sum()),
        }

    overall_valid = all(
        (field_counts[f]["valid"] / max(n, 1)) >= AUDIT_CONFIG["min_valid_target_rate"]
        for f in fields
    )

    return {
        "dataset": dataset_name,
        "n_records": n,
        "invalid_structures": invalid_structures,
        "target_field_counts": field_counts,
        "overall_valid": bool(overall_valid),
    }


def _target_distribution(records: list[dict], field: str) -> dict[str, float] | None:
    """Compute mean/std/min/max of a target field, ignoring missing values."""
    vals = [parse_target(r.get(field)) for r in records]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    arr = np.array(vals)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
    }


def _build_manifest(tasks: list[tuple], task_records: list[list[dict]]) -> dict:
    """Create a manifest of task membership."""
    manifest: dict[str, Any] = {"tasks": [], "records": defaultdict(list)}
    for t, task_desc in enumerate(tasks):
        task_key = "_".join(str(x) for x in task_desc)
        manifest["tasks"].append(
            {
                "id": t,
                "key": task_key,
                "descriptor": task_desc,
                "n_records": len(task_records[t]),
            }
        )
        for r in task_records[t]:
            manifest["records"][task_key].append(
                {
                    "jid": r.get("jid"),
                    "formula": r.get("formula"),
                    "target": r.get("target"),
                    "dataset": r.get("dataset"),
                    "property": r.get("property"),
                    "fidelity": r.get("fidelity"),
                }
            )
    manifest["records"] = dict(manifest["records"])
    return manifest


def _check_formula_disjointness(task_records: list[list[dict]]) -> tuple[bool, list[dict]]:
    """Verify pairwise formula disjointness across tasks."""
    formulas = [set(r.get("formula", "") for r in recs) for recs in task_records]
    issues = []
    for i in range(len(formulas)):
        for j in range(i + 1, len(formulas)):
            inter = formulas[i] & formulas[j]
            if inter:
                issues.append(
                    {
                        "task_i": i,
                        "task_j": j,
                        "shared_formulas": sorted(inter)[:20],
                        "n_shared": len(inter),
                    }
                )
    return (not issues), issues


def _check_graph_builder(task_records: list[list[dict]]) -> dict:
    """Smoke-test the periodic graph builder on the first few records."""
    builder = PeriodicGraphBuilder(supercell_matrix=2)
    successes = 0
    failures = 0
    errors: list[str] = []
    n_originals = []
    n_totals = []

    for recs in task_records:
        for r in recs[:5]:
            try:
                graph = builder(r["structure"])
                n_orig = int(graph["original_mask"].sum())
                n_total = graph["node_feats"].size(0)
                n_originals.append(n_orig)
                n_totals.append(n_total)
                successes += 1
            except Exception as exc:  # noqa: BLE001
                failures += 1
                errors.append(str(exc)[:200])
                if failures >= 3:
                    break
        if failures >= 3:
            break

    return {
        "successes": successes,
        "failures": failures,
        "errors": errors[:5],
        "sample_n_original": n_originals[:10],
        "sample_n_total": n_totals[:10],
        "ok": failures == 0 and successes > 0,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _markdown_report(audit: dict, gate: dict) -> str:
    lines = [
        "# JARVIS Data Audit Report",
        "",
        f"**Protocol:** {audit['protocol']}",
        f"**Generated:** {audit['timestamp']}",
        f"**Decision:** {'GO' if gate['go'] else 'NO-GO'}",
        "",
        "## Dataset-level audit",
        "",
    ]
    for ds in audit["dataset_audits"]:
        lines.append(f"### {ds['dataset']}")
        lines.append(f"- Records: {ds['n_records']:,}")
        lines.append(f"- Invalid structures: {ds['invalid_structures']}")
        lines.append("- Target fields:")
        for field, counts in ds["target_field_counts"].items():
            valid_rate = counts['valid'] / max(ds['n_records'], 1)
            lines.append(
                f"  - `{field}`: valid={counts['valid']:,} ({valid_rate:.2%}), "
                f"missing={counts['missing']:,}, nonfinite={counts['nonfinite']:,}"
            )
        lines.append("")

    lines.append("## Task summary")
    lines.append("")
    lines.append("| Task | Dataset | Property | Fidelity | Records | Target mean | Target std |")
    lines.append("|------|---------|----------|----------|--------:|------------:|-----------:|")
    for task in audit["task_summary"]:
        lines.append(
            f"| {task['id']} | {task['dataset']} | {task['property']} | "
            f"{task['fidelity']} | {task['n_records']:,} | "
            f"{task['target_mean']:.4f} | {task['target_std']:.4f} |"
        )
    lines.append("")

    lines.append("## Formula disjointness")
    lines.append(f"- Disjoint: {audit['disjoint']}")
    if audit["overlap_issues"]:
        lines.append("- Overlaps detected:")
        for issue in audit["overlap_issues"]:
            lines.append(
                f"  - Tasks {issue['task_i']} & {issue['task_j']}: "
                f"{issue['n_shared']} shared formulas"
            )
    lines.append("")

    lines.append("## Periodic graph builder smoke test")
    lines.append(f"- Successes: {audit['graph_builder']['successes']}")
    lines.append(f"- Failures: {audit['graph_builder']['failures']}")
    if audit["graph_builder"]["errors"]:
        lines.append("- Errors:")
        for err in audit["graph_builder"]["errors"]:
            lines.append(f"  - {err}")
    lines.append("")

    lines.append("## GO/NO-GO criteria")
    lines.append("")
    lines.append("| Criterion | Status | Detail |")
    lines.append("|-----------|--------|--------|")
    for c in gate["criteria"]:
        status = "PASS" if c["passed"] else "FAIL"
        lines.append(f"| {c['name']} | {status} | {c['detail']} |")
    lines.append("")

    if not gate["go"]:
        lines.append("## Blockers")
        lines.append("")
        for c in gate["criteria"]:
            if not c["passed"]:
                lines.append(f"- {c['name']}: {c['detail']}")
        lines.append("")

    lines.append("## Artifacts")
    lines.append(f"- Manifest: `{audit['manifest_path']}`")
    lines.append(f"- Gate: `{audit['gate_path']}`")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main audit routine
# ---------------------------------------------------------------------------


def _protocol_a_markdown_report(audit: dict, gate: dict) -> str:
    """Human-readable Protocol A audit report with exact counts."""
    snap = audit["snapshot_info"]
    counts = audit["protocol_a_counts"]
    lines = [
        "# Protocol A Data Audit Report",
        "",
        f"**Generated:** {audit['timestamp']}",
        f"**Decision:** {'GO' if gate['go'] else 'NO-GO'}",
        "",
        "## Snapshot integrity",
        "",
    ]
    for info in snap.values():
        lines.append(f"### {info['dataset_key']}")
        lines.append(f"- Expected file: `{info['expected_filename']}`")
        lines.append(f"- Resolved path: `{info['absolute_cache_path']}`")
        lines.append(f"- File size (bytes): {info['file_size_bytes']:,}")
        lines.append(f"- SHA256: `{info['sha256']}`")
        lines.append(f"- Raw JSON length: {info['raw_json_length']:,}")
        lines.append(f"- Unique JID count: {info['unique_jid_count']:,}")
        lines.append(f"- Snapshot match: {info['snapshot_match']}")
        lines.append("")

    lines.append("## Protocol A exact counts")
    lines.append("")
    lines.append(f"- raw_2021_records: {counts['raw_2021_records']:,}")
    lines.append(f"- raw_2022_records: {counts['raw_2022_records']:,}")
    lines.append(f"- unique_2021_jids: {counts['unique_2021_jids']:,}")
    lines.append(f"- unique_2022_jids: {counts['unique_2022_jids']:,}")
    lines.append(f"- duplicate_2021_jids: {counts['duplicate_2021_jids']}")
    lines.append(f"- duplicate_2022_jids: {counts['duplicate_2022_jids']}")
    lines.append(f"- retained_jids: {counts['retained_jids']:,}")
    lines.append(f"- added_jids: {counts['added_jids']:,}")
    lines.append(f"- removed_jids: {counts['removed_jids']:,}")
    lines.append(f"- valid_old_formation_records: {counts['valid_old_formation_records']:,}")
    lines.append(f"- valid_added_formation_records: {counts['valid_added_formation_records']:,}")
    lines.append("")

    lines.append("## Task split counts")
    lines.append("")
    lines.append("| Task | train | val | test | total |")
    lines.append("|------|------:|----:|-----:|------:|")
    for key in ["task_a1", "task_a2", "task_a3", "task_a4"]:
        c = counts[key]
        total = c["train"] + c["val"] + c["test"]
        lines.append(f"| {key} | {c['train']:,} | {c['val']:,} | {c['test']:,} | {total:,} |")
    lines.append("")

    lines.append("## Periodic graph builder smoke test")
    lines.append(f"- Successes: {audit['graph_builder']['successes']}")
    lines.append(f"- Failures: {audit['graph_builder']['failures']}")
    lines.append("")

    lines.append("## GO/NO-GO criteria")
    lines.append("")
    lines.append("| Criterion | Status | Detail |")
    lines.append("|-----------|--------|--------|")
    for c in gate["criteria"]:
        status = "PASS" if c["passed"] else "FAIL"
        lines.append(f"| {c['name']} | {status} | {c['detail']} |")
    lines.append("")

    if not gate["go"]:
        lines.append("## Blockers")
        lines.append("")
        for c in gate["criteria"]:
            if not c["passed"]:
                lines.append(f"- {c['name']}: {c['detail']}")
        lines.append("")

    lines.append("## Artifacts")
    lines.append(f"- Manifest: `{audit['manifest_path']}`")
    lines.append(f"- Gate: `{audit['gate_path']}`")
    lines.append("")
    return "\n".join(lines)


def _protocol_b_markdown_report(audit: dict, gate: dict) -> str:
    """Human-readable Protocol B audit report with paired-fidelity partitions."""
    counts = audit["protocol_b_counts"]
    lines = [
        "# Protocol B Data Audit Report",
        "",
        f"**Generated:** {audit['timestamp']}",
        f"**Decision:** {'GO' if gate['go'] else 'NO-GO'}",
        "",
        "## Paired-fidelity counts",
        f"- matched_jids_2021: {counts['matched_jids_2021']:,}",
        f"- matched_jids_2022: {counts['matched_jids_2022']:,}",
        "",
        "## Task split counts",
        "",
        "| Task | train | val | test | total |",
        "|------|------:|----:|-----:|------:|",
    ]
    for key in ["task_b1", "task_b2", "task_b3", "task_b4"]:
        c = counts[key]
        total = c["train"] + c["val"] + c["test"]
        lines.append(f"| {key} | {c['train']:,} | {c['val']:,} | {c['test']:,} | {total:,} |")
    lines.append("")

    lines.append("## Periodic graph builder smoke test")
    lines.append(f"- Successes: {audit['graph_builder']['successes']}")
    lines.append(f"- Failures: {audit['graph_builder']['failures']}")
    lines.append("")

    lines.append("## GO/NO-GO criteria")
    lines.append("")
    lines.append("| Criterion | Status | Detail |")
    lines.append("|-----------|--------|--------|")
    for c in gate["criteria"]:
        status = "PASS" if c["passed"] else "FAIL"
        lines.append(f"| {c['name']} | {status} | {c['detail']} |")
    lines.append("")

    if not gate["go"]:
        lines.append("## Blockers")
        lines.append("")
        for c in gate["criteria"]:
            if not c["passed"]:
                lines.append(f"- {c['name']}: {c['detail']}")
        lines.append("")

    lines.append("## Artifacts")
    lines.append(f"- Manifest: `{audit['manifest_path']}`")
    lines.append(f"- Gate: `{audit['gate_path']}`")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main audit routine
# ---------------------------------------------------------------------------


def run_audit(protocol: str, report_dir: str, cap: int | None = None) -> dict:
    """Run the full audit for a protocol and write artifacts."""
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    print(f"Auditing protocol {protocol.upper()}...")
    if protocol == "a":
        tasks, task_records, protocol_counts = build_protocol_a(
            seed=42, n_train_val_per_task=cap
        )
    else:
        tasks, task_records, protocol_counts = build_protocol_b(
            seed=42, n_train_val_per_task=cap
        )

    # Dataset-level audits (only fields required by this protocol).
    required_fields = AUDIT_CONFIG["protocol_target_fields"][protocol]
    dataset_names = sorted({r.get("dataset") for recs in task_records for r in recs})
    dataset_audits = []
    for ds_name in dataset_names:
        records = load_jarvis_dataset(ds_name)
        dataset_audits.append(_audit_dataset(records, ds_name, required_fields))

    # Task-level summaries.
    task_summary = []
    for t, (ds, prop, fid) in enumerate(tasks):
        recs = task_records[t]
        targets = [r["target"] for r in recs]
        arr = np.array(targets)
        task_summary.append(
            {
                "id": t,
                "dataset": ds,
                "property": prop,
                "fidelity": fid,
                "n_records": len(recs),
                "target_mean": float(arr.mean()),
                "target_std": float(arr.std()),
                "target_min": float(arr.min()),
                "target_max": float(arr.max()),
                "target_median": float(np.median(arr)),
            }
        )

    graph_builder_audit = _check_graph_builder(task_records)

    # GO/NO-GO criteria.
    criteria = []

    min_size_ok = all(t["n_records"] >= AUDIT_CONFIG["min_samples_per_task"] for t in task_summary)
    min_size_detail = (
        f"all tasks >= {AUDIT_CONFIG['min_samples_per_task']}" if min_size_ok
        else "tasks below minimum: "
        + ", ".join(str(t["id"]) for t in task_summary if t["n_records"] < AUDIT_CONFIG["min_samples_per_task"])
    )
    criteria.append({"name": "min_samples_per_task", "passed": min_size_ok, "detail": min_size_detail})

    # Validity is checked on the actual task records (post protocol filtering).
    task_valid_ok = all(all(parse_target(r.get("target")) is not None for r in recs) for recs in task_records)
    criteria.append(
        {
            "name": "valid_target_rate",
            "passed": task_valid_ok,
            "detail": "all task targets finite"
            if task_valid_ok
            else "some task targets missing or nonfinite",
        }
    )

    criteria.append(
        {
            "name": "periodic_graph_builder",
            "passed": graph_builder_audit["ok"],
            "detail": f"{graph_builder_audit['successes']} successes, {graph_builder_audit['failures']} failures",
        }
    )

    audit: dict[str, Any] = {
        "protocol": protocol,
        "timestamp": _now(),
        "dataset_audits": dataset_audits,
        "task_summary": task_summary,
        "graph_builder": graph_builder_audit,
        "manifest_path": str(report_path / f"manifest_protocol_{protocol}.json"),
        "gate_path": str(report_path / f"gate_protocol_{protocol}.json"),
    }

    if protocol == "a":
        # Snapshot-level integrity checks.
        snap_info = {
            "dft_3d_2021": _dataset_snapshot_info("dft_3d_2021"),
            "dft_3d": _dataset_snapshot_info("dft_3d"),
        }
        snapshot_match_ok = all(info["snapshot_match"] for info in snap_info.values())
        criteria.append(
            {
                "name": "jarvis_snapshot_match",
                "passed": snapshot_match_ok,
                "detail": "cached zip contains expected JSON member"
                if snapshot_match_ok
                else "NO_GO_WRONG_JARVIS_SNAPSHOT",
            }
        )
        audit["snapshot_info"] = snap_info
        audit["protocol_a_counts"] = protocol_counts
        report_md = _protocol_a_markdown_report(audit, gate={"go": False, "criteria": []})
    else:
        audit["protocol_b_counts"] = protocol_counts
        report_md = _protocol_b_markdown_report(audit, gate={"go": False, "criteria": []})

    go = all(c["passed"] for c in criteria)

    gate = {
        "go": go,
        "protocol": protocol,
        "timestamp": _now(),
        "criteria": criteria,
    }

    # Re-render report with the real gate.
    if protocol == "a":
        report_md = _protocol_a_markdown_report(audit, gate)
    else:
        report_md = _protocol_b_markdown_report(audit, gate)

    manifest = _build_manifest(tasks, task_records)
    if protocol == "a":
        manifest["snapshot_info"] = audit["snapshot_info"]
        manifest["protocol_a_counts"] = audit["protocol_a_counts"]
    else:
        manifest["protocol_b_counts"] = audit["protocol_b_counts"]

    # Write artifacts.
    with open(audit["manifest_path"], "w") as f:
        json.dump(manifest, f, indent=2)
    with open(audit["gate_path"], "w") as f:
        json.dump(gate, f, indent=2)
    with open(report_path / f"audit_protocol_{protocol}.md", "w") as f:
        f.write(report_md)

    print(report_md)
    print(f"\nGate: {'GO' if go else 'NO-GO'}")
    return gate


def main():
    parser = argparse.ArgumentParser(description="Audit JARVIS data for PhyTCA.")
    parser.add_argument("--protocol", choices=["a", "b"], required=True)
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--cap", type=int, default=None, help="Per-task cap for smoke tests")
    args = parser.parse_args()

    run_audit(args.protocol, args.report_dir, cap=args.cap)


if __name__ == "__main__":
    main()
