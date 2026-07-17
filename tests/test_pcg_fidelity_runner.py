"""Smoke test for the PCG fidelity-expansion runner."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models import CrystalEncoder
from persistent_consolidation_graph import PersistentConsolidationGraph
from pcg_runner import (
    cap_records,
    determine_parents_fidelity_expansion,
    run_pcg_protocol,
)
from protocols import build_fidelity_expansion_protocol


@pytest.mark.slow
def test_fidelity_expansion_runner_smoke(tmp_path: Path):
    tasks, task_records, audit = build_fidelity_expansion_protocol(
        version="dft_3d_2021",
        properties=("band_gap",),
        fidelities=("OptB88vdW", "TB-mBJ"),
        n_train_val_per_task=10,
    )
    task_records = [cap_records(recs, 10) for recs in task_records]

    prop2id = {"band_gap": 0}
    fid2id = {"OptB88vdW": 0, "TB-mBJ": 1}

    encoder = CrystalEncoder(
        node_dim=92, hidden_dim=16, n_layers=1, num_nearest_neighbors=4
    )
    model = PersistentConsolidationGraph(encoder, hidden_dim=16, rank=4)

    metrics = run_pcg_protocol(
        protocol_name="fidelity_expansion",
        tasks=tasks,
        task_records=task_records,
        model=model,
        prop2id=prop2id,
        fid2id=fid2id,
        device=torch.device("cpu"),
        batch_size=4,
        epochs_fast=1,
        epochs_cons=1,
        lr=1e-3,
        output_dir=tmp_path,
        parent_fn=determine_parents_fidelity_expansion,
    )

    assert metrics["protocol"] == "fidelity_expansion"
    assert len(metrics["route_keys"]) == 2
    assert "average_final_nmae" in metrics
