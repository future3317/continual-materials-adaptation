"""Command-line tool to analyze snapshot diffs between JARVIS releases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from snapshot_diff import classify_jarvis_snapshots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze JARVIS 2021 -> 2022 snapshot diff"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="JARVIS cache directory",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[1e-4, 1e-3, 1e-2],
        help="Label-revision thresholds in target units",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/snapshot_diff"),
        help="Directory for the JSON report",
    )
    args = parser.parse_args()

    summary, next_annotated, removed = classify_jarvis_snapshots(
        cache_dir=args.cache_dir,
        thresholds=tuple(args.thresholds),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "snapshot_diff_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "n_next_annotated": len(next_annotated),
                "n_removed": len(removed),
            },
            f,
            indent=2,
        )

    print("Snapshot diff summary")
    print("=" * 40)
    print(f"Previous snapshot records : {summary['n_prev']}")
    print(f"Next snapshot records     : {summary['n_next']}")
    print(f"Unique previous JIDs      : {summary['unique_prev_jids']}")
    print(f"Unique next JIDs          : {summary['unique_next_jids']}")
    print(f"Retained JIDs             : {summary['retained_jids']}")
    print(f"Added JIDs                : {summary['added_jids']}")
    print(f"Removed JIDs              : {summary['removed_jids']}")
    print("-" * 40)
    for ctype, count in summary["counts"].items():
        print(f"  {ctype:30s}: {count}")
    print("-" * 40)
    print("Label revisions by threshold:")
    for key, count in summary["label_revisions_by_threshold"].items():
        print(f"  {key:30s}: {count}")
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
