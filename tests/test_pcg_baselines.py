"""Smoke tests for the PCG baseline comparison harness."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from scripts.run_pcg_baselines import main


METHODS = [
    "pcg_proposed",
    "pcg_fixed_basis",
    "pcg_always_expand",
    "cow_full_encoder",
    "per_endpoint_lora",
    "per_endpoint_head",
    "independent",
    "joint",
]


def test_all_baselines_smoke() -> None:
    """All baseline methods should run on a tiny capped protocol in one pass."""
    output_dir = Path(tempfile.mkdtemp())
    old_argv = sys.argv
    try:
        sys.argv = [
            "run_pcg_baselines.py",
            "--properties",
            "band_gap",
            "--fidelities",
            "OptB88vdW",
            "--encoder-type",
            "egnn",
            "--hidden-dim",
            "16",
            "--rank",
            "4",
            "--epochs-fast",
            "1",
            "--epochs-cons",
            "1",
            "--epochs-baseline",
            "1",
            "--batch-size",
            "8",
            "--cap",
            "10",
            "--device",
            "cpu",
            "--output-dir",
            str(output_dir),
            "--methods",
            *METHODS,
        ]
        main()

        summary_path = output_dir / "baseline_results.json"
        assert summary_path.exists()
        with open(summary_path, encoding="utf-8") as f:
            results = json.load(f)
        for method in METHODS:
            assert method in results["methods"]
            metrics = results["methods"][method]
            assert "average_final_nmae" in metrics
            assert "total_parameters" in metrics
            assert metrics["total_parameters"] > 0
    finally:
        sys.argv = old_argv
        shutil.rmtree(output_dir, ignore_errors=True)
