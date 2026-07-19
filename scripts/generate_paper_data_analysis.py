"""Generate data-analysis figures and tables for the FR-PhyTCA paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import build_protocol_b, load_jarvis_dataset, parse_target


def _stats(x: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "median": float(np.median(x)),
        "q25": float(np.percentile(x, 25)),
        "q75": float(np.percentile(x, 75)),
    }


def _protocol_b_analysis(cache_dir: str | None = None) -> dict:
    tasks, task_records, audit = build_protocol_b(
        cache_dir=cache_dir, seed=42, n_train_val_per_task=None
    )
    opt_vals = np.array([r["target"] for r in task_records[0]])
    mbj_vals = np.array([r["target"] for r in task_records[1]])
    residual = mbj_vals - opt_vals

    nonzero_mask = (opt_vals > 0) & (mbj_vals > 0)

    return {
        "n_paired_2021": int(len(opt_vals)),
        "n_zero_opt": int((opt_vals == 0).sum()),
        "n_zero_mbj": int((mbj_vals == 0).sum()),
        "n_nonzero_both": int(nonzero_mask.sum()),
        "all": {
            "opt": _stats(opt_vals),
            "mbj": _stats(mbj_vals),
            "residual_mbj_minus_opt": _stats(residual),
        },
        "nonzero_both": {
            "opt": _stats(opt_vals[nonzero_mask]),
            "mbj": _stats(mbj_vals[nonzero_mask]),
            "residual_mbj_minus_opt": _stats(residual[nonzero_mask]),
        },
        "correlations": {
            "pearson_opt_mbj_all": float(np.corrcoef(opt_vals, mbj_vals)[0, 1]),
            "pearson_opt_mbj_nonzero": float(np.corrcoef(opt_vals[nonzero_mask], mbj_vals[nonzero_mask])[0, 1]),
            "pearson_opt_residual_nonzero": float(np.corrcoef(opt_vals[nonzero_mask], residual[nonzero_mask])[0, 1]),
        },
    }


def _snapshot_overlap_analysis(cache_dir: str | None = None) -> dict:
    d21 = load_jarvis_dataset("dft_3d_2021", cache_dir)
    d22 = load_jarvis_dataset("dft_3d", cache_dir)

    jids21 = {r["jid"]: r for r in d21}
    jids22 = {r["jid"]: r for r in d22}

    retained = set(jids21) & set(jids22)
    added = set(jids22) - set(jids21)
    removed = set(jids21) - set(jids22)

    # Check label revisions for band-gap fields.
    target_changed = 0
    target_fields = ("optb88vdw_bandgap", "mbj_bandgap")
    for jid in retained:
        for field in target_fields:
            v1 = parse_target(jids21[jid].get(field))
            v2 = parse_target(jids22[jid].get(field))
            if (v1 is None) != (v2 is None):
                target_changed += 1
                break
            if v1 is not None and v2 is not None and abs(v1 - v2) > 1e-6:
                target_changed += 1
                break

    return {
        "n_2021": len(d21),
        "n_2022": len(d22),
        "unique_2021": len(jids21),
        "unique_2022": len(jids22),
        "retained_jids": len(retained),
        "added_jids": len(added),
        "removed_jids": len(removed),
        "bandgap_target_revisions": target_changed,
    }


def _make_figures(opt_vals: np.ndarray, mbj_vals: np.ndarray, out_dir: Path) -> None:
    nonzero_mask = (opt_vals > 0) & (mbj_vals > 0)
    opt_nz = opt_vals[nonzero_mask]
    mbj_nz = mbj_vals[nonzero_mask]
    res_nz = mbj_nz - opt_nz

    # Scatter OPT vs MBJ.
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.scatter(opt_nz, mbj_nz, s=2, alpha=0.3, rasterized=True)
    ax.plot([0, 10], [0, 10], "k--", lw=1, label="$y=x$")
    ax.set_xlabel("OptB88vdW band gap (eV)")
    ax.set_ylabel("TB-mBJ band gap (eV)")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "opt_vs_mbj_scatter.pdf", dpi=300)
    plt.close(fig)

    # Histogram of residuals.
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.hist(res_nz, bins=100, range=(-0.5, 4.0), color="steelblue", edgecolor="white")
    ax.axvline(float(np.median(res_nz)), color="darkred", ls="--", lw=1.5, label=f"median={np.median(res_nz):.2f} eV")
    ax.set_xlabel("TB-mBJ $-$ OptB88vdW (eV)")
    ax.set_ylabel("Count")
    ax.set_xlim(-0.5, 4.0)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "mbj_minus_opt_residual_hist.pdf", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/paper_data_analysis"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pb = _protocol_b_analysis(args.cache_dir)
    snap = _snapshot_overlap_analysis(args.cache_dir)

    report = {"protocol_b": pb, "snapshot_overlap": snap}
    with open(args.output_dir / "data_analysis_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Build arrays for figures from the 2021 paired set.
    tasks, task_records, _ = build_protocol_b(
        cache_dir=args.cache_dir, seed=42, n_train_val_per_task=None
    )
    opt_vals = np.array([r["target"] for r in task_records[0]])
    mbj_vals = np.array([r["target"] for r in task_records[1]])
    _make_figures(opt_vals, mbj_vals, args.output_dir)

    print(f"Report saved to {args.output_dir / 'data_analysis_report.json'}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
