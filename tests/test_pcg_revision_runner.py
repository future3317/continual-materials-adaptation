"""Smoke test for the PCG revision runner."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models import CrystalEncoder
from persistent_consolidation_graph import PersistentConsolidationGraph
from pcg_runner import cap_records, determine_parents_revision, run_pcg_protocol
from protocols import build_revision_protocol


@pytest.mark.slow
def test_revision_runner_smoke(tmp_path: Path):
    tasks, task_records, audit = build_revision_protocol(
        properties=("band_gap",),
        fidelities=("OptB88vdW",),
        n_train_val_per_task=10,
    )
    task_records = [cap_records(recs, 10) for recs in task_records]

    prop2id = {"band_gap": 0}
    fid2id = {"OptB88vdW": 0}

    encoder = CrystalEncoder(
        node_dim=92, hidden_dim=16, n_layers=1, num_nearest_neighbors=4
    )
    model = PersistentConsolidationGraph(encoder, hidden_dim=16, rank=4)

    eval_subsets = {1: ["label_revised", "unchanged", "structure_revised"]}
    metrics = run_pcg_protocol(
        protocol_name="revision",
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
        parent_fn=determine_parents_revision,
        eval_subsets=eval_subsets,
    )

    assert metrics["protocol"] == "revision"
    assert len(metrics["route_keys"]) == len(tasks)
    assert "average_final_nmae" in metrics
    assert "subset_metrics" in metrics
