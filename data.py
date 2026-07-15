"""JARVIS data loading and periodic graph construction for PhyTCA.

This module replaces the synthetic-only data pipeline with real JARVIS data
loaded via ``jarvis-tools`` (with a robust zip fallback) and builds periodic
graphs suitable for crystal graph encoders.

Supported protocols:
  * Protocol A: database evolution across JARVIS-2021 and JARVIS-2022.
  * Protocol B: multi-fidelity band-gap learning (OptB88vdW -> TB-mBJ).

All scalar targets are validated and missing values are excluded at the sample
level. Splits are formula-disjoint to prevent leakage.
"""

from __future__ import annotations

import json
import math
import os
import zipfile
from typing import Any, Callable, Sequence

import numpy as np
import torch
from pymatgen.core import Lattice, Structure
from torch.utils.data import Dataset

from periodic_graph import build_periodic_edge_graph


# ---------------------------------------------------------------------------
# JARVIS cache helpers
# ---------------------------------------------------------------------------

JARVIS_DATASETS: dict[str, str] = {
    "dft_3d_2021": "jdft_3d-8-18-2021.json.zip",
    "dft_3d": "jdft_3d-12-12-2022.json.zip",
}


def _default_cache_dir() -> str:
    """Return the default JARVIS cache directory relative to this file."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache", "jarvis")


def load_jarvis_dataset(name: str, cache_dir: str | None = None) -> list[dict]:
    """Load a JARVIS dataset, preferring a local zip cache.

    ``jarvis-tools`` sometimes re-downloads even when a cache exists; this
    function loads the JSON directly from a cached zip if present, otherwise
    falls back to the official ``jarvis.db.figshare.data`` loader.

    Args:
        name: Dataset name, e.g. ``"dft_3d_2021"`` or ``"dft_3d"``.
        cache_dir: Directory containing cached zip files. Defaults to
            ``<project>/data_cache/jarvis``.

    Returns:
        List of JARVIS records (dicts).
    """
    cache_dir = cache_dir or _default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    zip_name = JARVIS_DATASETS.get(name, f"{name}.json.zip")
    zip_path = os.path.join(cache_dir, zip_name)

    if os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path) as zf:
                members = zf.namelist()
                if not members:
                    raise ValueError(f"Empty zip: {zip_path}")
                return json.loads(zf.read(members[0]))
        except (zipfile.BadZipFile, ValueError):
            # Corrupt zip: remove and fall back to downloader.
            os.remove(zip_path)

    # Fallback to official loader.
    from jarvis.db.figshare import data as jdata

    old_cache = os.environ.get("JARVIS_DB_CACHE")
    os.environ["JARVIS_DB_CACHE"] = cache_dir
    try:
        return jdata(name)
    finally:
        if old_cache is None:
            os.environ.pop("JARVIS_DB_CACHE", None)
        else:
            os.environ["JARVIS_DB_CACHE"] = old_cache


# ---------------------------------------------------------------------------
# Record conversion and target parsing
# ---------------------------------------------------------------------------


def jarvis_record_to_structure(record: dict) -> Structure:
    """Convert a JARVIS ``atoms`` dict to a pymatgen ``Structure``.

    Handles both cartesian and fractional coordinates and validates that the
    resulting structure has a positive volume.
    """
    atoms = record["atoms"]
    lattice = Lattice(atoms["lattice_mat"])
    elements = atoms["elements"]
    coords = atoms["coords"]
    cartesian = atoms.get("cartesian", False)
    struct = Structure(lattice, elements, coords, coords_are_cartesian=cartesian)
    if struct.volume <= 1e-6:
        raise ValueError(f"Degenerate structure with volume {struct.volume}")
    return struct


def parse_target(value: Any) -> float | None:
    """Return a valid float target or ``None`` if the value is missing.

    Rejects ``None``, ``NaN``, ``inf``, empty strings, and string markers such
    as ``"na"`` or ``"None"``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"", "na", "n/a", "none", "nan", "inf", "-inf"}:
            return None
        try:
            value = float(value)
        except ValueError:
            return None
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fv):
        return None
    return fv


# ---------------------------------------------------------------------------
# Protocol builders
# ---------------------------------------------------------------------------


def _has_targets(record: dict, fields: Sequence[str]) -> bool:
    """Check whether a record has finite values for all target fields."""
    return all(parse_target(record.get(f)) is not None for f in fields)


def _select_and_tag(
    records: list[dict],
    dataset_tag: str,
    property_name: str,
    fidelity_name: str,
    target_field: str,
) -> list[dict]:
    """Filter records with a valid target and attach protocol metadata."""
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
                "property": property_name,
                "fidelity": fidelity_name,
                "target": val,
            }
        )
    return out


def _assign_splits(
    records: list[dict],
    seed: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> list[dict]:
    """Assign formula-disjoint train/val/test splits to records in place."""
    rng = np.random.default_rng(seed)
    formulas = list({r["formula"] for r in records})
    rng.shuffle(formulas)

    n = len(formulas)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    train_formulas = set(formulas[:n_train])
    val_formulas = set(formulas[n_train : n_train + n_val])
    test_formulas = set(formulas[n_train + n_val :])

    for r in records:
        f = r["formula"]
        if f in train_formulas:
            r["split"] = "train"
        elif f in val_formulas:
            r["split"] = "val"
        else:
            r["split"] = "test"
    return records


def build_protocol_a(
    cache_dir: str | None = None,
    seed: int = 42,
    n_train_val_per_task: int | None = None,
) -> tuple[list[tuple[str, str, str]], list[list[dict]], dict]:
    """Build Protocol A: data-incremental JARVIS database evolution.

    Task sequence:
      1. A1: JARVIS-2021 formation energy (OptB88vdW)
      2. A2: newly added JARVIS-2022 formation energy (OptB88vdW)
      3. A3: JARVIS-2021 band gap (OptB88vdW)
      4. A4: newly added JARVIS-2022 band gap (OptB88vdW)

    A2 and A4 are restricted to JIDs that do not appear in the 2021 snapshot,
    so A1 -> A2 is a true data-incremental expansion of the formation-energy
    task, and A3 -> A4 is the same for band gap.  Each task is split
    formula-disjointly into train/val/test.

    Args:
        cache_dir: JARVIS cache directory.
        seed: Random seed for splitting.
        n_train_val_per_task: If given, cap each task to this many records
            (useful for fast smoke tests).  Cap applies after the "added"
            filter and before splitting.

    Returns:
        tasks: List of ``(dataset, property, fidelity)`` task descriptors.
        task_records: List of record lists, one per task, each with a
            ``split`` field set to ``"train"``, ``"val"``, or ``"test"``.
        audit: Dict of exact snapshot counts for gate reporting.
    """
    d21 = load_jarvis_dataset("dft_3d_2021", cache_dir)
    d22 = load_jarvis_dataset("dft_3d", cache_dir)

    raw_2021_records = len(d21)
    raw_2022_records = len(d22)
    jids_2021 = [r["jid"] for r in d21]
    jids_2022 = [r["jid"] for r in d22]
    unique_2021_jids = set(jids_2021)
    unique_2022_jids = set(jids_2022)
    duplicate_2021_jids = len(jids_2021) - len(unique_2021_jids)
    duplicate_2022_jids = len(jids_2022) - len(unique_2022_jids)
    retained_jids = unique_2021_jids & unique_2022_jids
    added_jids = unique_2022_jids - unique_2021_jids
    removed_jids = unique_2021_jids - unique_2022_jids

    def filter_added(records: list[dict]) -> list[dict]:
        return [r for r in records if r["jid"] in added_jids]

    a1 = _select_and_tag(d21, "dft_3d_2021", "formation_energy", "OptB88vdW", "formation_energy_peratom")
    a2 = filter_added(
        _select_and_tag(d22, "dft_3d", "formation_energy", "OptB88vdW", "formation_energy_peratom")
    )
    a3 = _select_and_tag(d21, "dft_3d_2021", "band_gap", "OptB88vdW", "optb88vdw_bandgap")
    a4 = filter_added(
        _select_and_tag(d22, "dft_3d", "band_gap", "OptB88vdW", "optb88vdw_bandgap")
    )

    if n_train_val_per_task is not None:
        a1 = a1[:n_train_val_per_task]
        a2 = a2[:n_train_val_per_task]
        a3 = a3[:n_train_val_per_task]
        a4 = a4[:n_train_val_per_task]

    # Formula-disjoint splits per task.
    a1 = _assign_splits(a1, seed=seed)
    a2 = _assign_splits(a2, seed=seed + 1)
    a3 = _assign_splits(a3, seed=seed + 2)
    a4 = _assign_splits(a4, seed=seed + 3)

    tasks = [
        ("dft_3d_2021", "formation_energy", "OptB88vdW"),
        ("dft_3d", "formation_energy", "OptB88vdW"),
        ("dft_3d_2021", "band_gap", "OptB88vdW"),
        ("dft_3d", "band_gap", "OptB88vdW"),
    ]
    task_records = [a1, a2, a3, a4]

    def split_counts(recs: list[dict]) -> dict[str, int]:
        return {
            "train": sum(1 for r in recs if r["split"] == "train"),
            "val": sum(1 for r in recs if r["split"] == "val"),
            "test": sum(1 for r in recs if r["split"] == "test"),
        }

    audit = {
        "raw_2021_records": raw_2021_records,
        "raw_2022_records": raw_2022_records,
        "unique_2021_jids": len(unique_2021_jids),
        "unique_2022_jids": len(unique_2022_jids),
        "duplicate_2021_jids": duplicate_2021_jids,
        "duplicate_2022_jids": duplicate_2022_jids,
        "retained_jids": len(retained_jids),
        "added_jids": len(added_jids),
        "removed_jids": len(removed_jids),
        "valid_old_formation_records": len(a1),
        "valid_added_formation_records": len(a2),
        "task_a1": split_counts(a1),
        "task_a2": split_counts(a2),
        "task_a3": split_counts(a3),
        "task_a4": split_counts(a4),
    }

    # Hard assertions required by the audit gate.
    assert raw_2021_records == 55723
    assert raw_2022_records == 75993
    assert len(a1) <= raw_2021_records
    assert set(r["jid"] for r in a2).isdisjoint(unique_2021_jids)
    assert len(a1) == audit["task_a1"]["train"] + audit["task_a1"]["val"] + audit["task_a1"]["test"]
    assert len(a2) == audit["task_a2"]["train"] + audit["task_a2"]["val"] + audit["task_a2"]["test"]

    return tasks, task_records, audit


def build_protocol_b(
    cache_dir: str | None = None,
    seed: int = 42,
    n_train_val_per_task: int | None = None,
) -> tuple[list[tuple[str, str, str]], list[list[dict]], dict]:
    """Build Protocol B: multi-fidelity band-gap learning with paired splits.

    Task sequence:
      1. JARVIS-2021 band gap / OptB88vdW
      2. JARVIS-2021 band gap / TB-mBJ
      3. JARVIS-2022 band gap / OptB88vdW
      4. JARVIS-2022 band gap / TB-mBJ

    Only structures with both band-gap fidelities are retained, and the two
    fidelity records for the same structure are assigned to the same
    train/val/test partition.

    Args:
        cache_dir: JARVIS cache directory.
        seed: Random seed for splitting.
        n_train_val_per_task: Optional per-task cap.

    Returns:
        tasks, task_records, audit.
    """
    d21 = load_jarvis_dataset("dft_3d_2021", cache_dir)
    d22 = load_jarvis_dataset("dft_3d", cache_dir)

    def pair_bandgaps(records: list[dict], ds_tag: str) -> tuple[list[dict], list[dict]]:
        """Return paired OPT and MBJ records with shared metadata."""
        opt_recs, mbj_recs = [], []
        for r in records:
            opt = parse_target(r.get("optb88vdw_bandgap"))
            mbj = parse_target(r.get("mbj_bandgap"))
            if opt is None or mbj is None:
                continue
            struct = jarvis_record_to_structure(r)
            base = {
                "jid": r.get("jid"),
                "structure": struct,
                "formula": struct.composition.reduced_formula,
                "dataset": ds_tag,
                "property": "band_gap",
            }
            opt_recs.append({**base, "fidelity": "OptB88vdW", "target": opt})
            mbj_recs.append({**base, "fidelity": "TB-mBJ", "target": mbj})
        return opt_recs, mbj_recs

    opt_21, mbj_21 = pair_bandgaps(d21, "dft_3d_2021")
    opt_22, mbj_22 = pair_bandgaps(d22, "dft_3d")

    # Assign splits jointly by formula so OPT/MBJ pairs stay together.
    def assign_paired_splits(opt_recs: list[dict], mbj_recs: list[dict], seed: int) -> None:
        rng = np.random.default_rng(seed)
        formulas = list({r["formula"] for r in opt_recs})
        rng.shuffle(formulas)
        n = len(formulas)
        n_train = max(1, int(n * 0.70))
        n_val = max(1, int(n * 0.15))
        train_formulas = set(formulas[:n_train])
        val_formulas = set(formulas[n_train : n_train + n_val])
        test_formulas = set(formulas[n_train + n_val :])

        for r in opt_recs + mbj_recs:
            f = r["formula"]
            if f in train_formulas:
                r["split"] = "train"
            elif f in val_formulas:
                r["split"] = "val"
            else:
                r["split"] = "test"

    assign_paired_splits(opt_21, mbj_21, seed=seed)
    assign_paired_splits(opt_22, mbj_22, seed=seed + 1)

    # Optional cap applied per fidelity.
    if n_train_val_per_task is not None:
        opt_21 = opt_21[:n_train_val_per_task]
        mbj_21 = mbj_21[:n_train_val_per_task]
        opt_22 = opt_22[:n_train_val_per_task]
        mbj_22 = mbj_22[:n_train_val_per_task]

    tasks = [
        ("dft_3d_2021", "band_gap", "OptB88vdW"),
        ("dft_3d_2021", "band_gap", "TB-mBJ"),
        ("dft_3d", "band_gap", "OptB88vdW"),
        ("dft_3d", "band_gap", "TB-mBJ"),
    ]
    task_records = [opt_21, mbj_21, opt_22, mbj_22]

    def split_counts(recs: list[dict]) -> dict[str, int]:
        return {
            "train": sum(1 for r in recs if r["split"] == "train"),
            "val": sum(1 for r in recs if r["split"] == "val"),
            "test": sum(1 for r in recs if r["split"] == "test"),
        }

    # Verify OPT/MBJ pairing is preserved after splitting.
    def matched_jid_count(opt_recs: list[dict], mbj_recs: list[dict]) -> int:
        opt_jids = {r["jid"] for r in opt_recs}
        return sum(1 for r in mbj_recs if r["jid"] in opt_jids)

    audit = {
        "matched_jids_2021": matched_jid_count(opt_21, mbj_21),
        "matched_jids_2022": matched_jid_count(opt_22, mbj_22),
        "task_b1": split_counts(opt_21),
        "task_b2": split_counts(mbj_21),
        "task_b3": split_counts(opt_22),
        "task_b4": split_counts(mbj_22),
    }

    # Assert shared partitions for paired records.
    for opt_recs, mbj_recs in [(opt_21, mbj_21), (opt_22, mbj_22)]:
        opt_map = {r["jid"]: r["split"] for r in opt_recs}
        for r in mbj_recs:
            assert r["jid"] in opt_map
            assert opt_map[r["jid"]] == r["split"]

    return tasks, task_records, audit


# ---------------------------------------------------------------------------
# Formula-disjoint splitting
# ---------------------------------------------------------------------------


def formula_disjoint_split(
    task_record_lists: Sequence[list[dict]],
    seed: int = 42,
) -> list[list[dict]]:
    """Reorder records so that tasks share no reduced formulas.

    The input is a list where each inner list already belongs to one task and
    is ordered by the protocol builder. This function assigns formulas
    round-robin to tasks and then redistributes records so that every task
    contains only records whose formula was assigned to it.

    Args:
        task_record_lists: Lists of records per task.
        seed: Random seed for formula shuffling.

    Returns:
        Records per task with formula-disjoint guarantees.
    """
    rng = np.random.default_rng(seed)
    n_tasks = len(task_record_lists)

    # Collect all formulas and assign round-robin to tasks.
    all_formulas: set[str] = set()
    for recs in task_record_lists:
        all_formulas.update(r.get("formula", "") for r in recs)
    all_formulas.discard("")
    formulas = np.array(list(all_formulas))
    rng.shuffle(formulas)

    owner_for_formula = {
        f: i % n_tasks for i, f in enumerate(formulas)
    }

    # Redistribute records based on formula ownership.
    out: list[list[dict]] = [[] for _ in range(n_tasks)]
    for recs in task_record_lists:
        for r in recs:
            owner = owner_for_formula.get(r.get("formula", ""))
            if owner is not None:
                out[owner].append(r)
    return out


def cap_splits(
    records: list[dict],
    train_cap: int | None = None,
    val_cap: int | None = None,
    test_cap: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Cap each split to a maximum number of records deterministically.

    The original split assignment is preserved; if a split has more records
    than the cap, a random subset (with ``seed``) is retained.  This is useful
    for small-scale Phase 0 screens where train/val/test sizes must be fixed
    across methods.
    """
    rng = np.random.default_rng(seed)
    split_caps = {"train": train_cap, "val": val_cap, "test": test_cap}
    out: list[dict] = []
    for split in ("train", "val", "test"):
        recs = [r for r in records if r.get("split") == split]
        cap = split_caps[split]
        if cap is not None and len(recs) > cap:
            idx = np.arange(len(recs))
            rng.shuffle(idx)
            recs = [recs[i] for i in idx[:cap]]
        out.extend(recs)
    return out


# ---------------------------------------------------------------------------
# Periodic graph builder
# ---------------------------------------------------------------------------


class PeriodicGraphBuilder:
    """Build a periodic graph compatible with crystal graph encoders from a pymatgen ``Structure``.

    The builder expands the unit cell into a supercell (default ``2x2x2``) so
    that atoms near the boundary can interact with their periodic images. The
    returned tensors include both original and image atoms; callers should use
    ``original_mask`` to pool only over the original unit cell.

    Attributes:
        supercell_matrix: Integer matrix describing the supercell expansion.
        node_feature_dim: Dimensionality of one-hot element node features.
        max_neighbors: Soft cap on the number of periodic neighbors considered
            by the crystal graph encoder (via ``num_nearest_neighbors``).
    """

    def __init__(
        self,
        supercell_matrix: int | Sequence[int] | np.ndarray = 2,
        node_feature_dim: int = 92,
        max_neighbors: int = 16,
    ) -> None:
        if isinstance(supercell_matrix, int):
            self.supercell_matrix = np.diag([supercell_matrix, supercell_matrix, supercell_matrix])
        else:
            self.supercell_matrix = np.array(supercell_matrix, dtype=int).reshape(3, 3)
        self.node_feature_dim = node_feature_dim
        self.max_neighbors = max_neighbors

    def __call__(self, structure: Structure) -> dict[str, torch.Tensor]:
        """Build periodic graph tensors for the input structure.

        Returns:
            Dictionary with keys:
              * ``node_feats``: (N_total, node_feature_dim) one-hot features.
              * ``coords``: (N_total, 3) Cartesian coordinates.
              * ``original_mask``: (N_total,) bool tensor marking original atoms.
              * ``image_offsets``: (N_total, 3) integer supercell offsets.
        """
        return build_periodic_graph(
            structure,
            supercell_matrix=self.supercell_matrix,
            node_feature_dim=self.node_feature_dim,
        )


class PeriodicEdgeGraphBuilder:
    """Build an explicit periodic-edge graph from a pymatgen ``Structure``.

    Unlike ``PeriodicGraphBuilder``, this builder keeps only the unit-cell atoms
    as nodes and stores periodicity as integer lattice shifts on the edges.  See
    ``periodic_graph.build_periodic_edge_graph`` for details.

    Attributes:
        cutoff: Real-space cutoff radius for edges.
        max_neighbors: Optional per-atom cap on the number of closest neighbors.
        node_feature_dim: Dimensionality of one-hot element node features.
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        max_neighbors: int | None = None,
        node_feature_dim: int = 92,
    ) -> None:
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.node_feature_dim = node_feature_dim

    def __call__(self, structure: Structure) -> dict[str, torch.Tensor]:
        """Build explicit periodic-edge graph tensors for the input structure."""
        return build_periodic_edge_graph(
            structure,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            node_feature_dim=self.node_feature_dim,
        )


def _element_one_hot(element: str, dim: int = 92) -> torch.Tensor:
    """One-hot encode an element by atomic number (Z <= dim)."""
    from pymatgen.core import Element

    z = int(Element(element).Z)
    vec = torch.zeros(dim)
    vec[min(z, dim) - 1] = 1.0
    return vec


def build_periodic_graph(
    structure: Structure,
    supercell_matrix: np.ndarray,
    node_feature_dim: int = 92,
) -> dict[str, torch.Tensor]:
    """Expand a structure to a supercell and build graph tensors.

    Args:
        structure: Input crystal structure.
        supercell_matrix: 3x3 integer matrix; for a simple NxNxN expansion use
            ``np.diag([N, N, N])``.
        node_feature_dim: One-hot feature dimension.

    Returns:
        Dict with ``node_feats``, ``coords``, ``original_mask``, ``image_offsets``.
    """
    sc = structure * supercell_matrix
    n_orig = len(structure)
    n_sc = len(sc)

    # Map each supercell atom to its original atom index and integer lattice offset.
    # For a supercell atom at cartesian position C, there is a unique original
    # atom i and integer offset k (row vector) such that
    #     C = cart_orig[i] + k @ lattice_orig
    lattice_inv = np.linalg.inv(structure.lattice.matrix)
    cart_orig = structure.cart_coords
    cart_sc = sc.cart_coords

    # Vectorized over all supercell atoms and all original atoms.
    delta = (cart_sc[:, None, :] - cart_orig[None, :, :]) @ lattice_inv  # (n_sc, n_orig, 3)
    k = np.rint(delta).astype(np.int64)                                 # (n_sc, n_orig, 3)
    errs = np.linalg.norm(delta - k, axis=2)                            # (n_sc, n_orig)
    original_indices = np.argmin(errs, axis=1)                          # (n_sc,)
    offsets_arr = k[np.arange(n_sc), original_indices]                  # (n_sc, 3)

    node_feats = torch.stack(
        [_element_one_hot(str(site.specie), node_feature_dim) for site in sc]
    )
    coords = torch.tensor(sc.cart_coords, dtype=torch.float32)
    image_offsets = torch.tensor(offsets_arr, dtype=torch.long)
    original_mask = torch.tensor((offsets_arr == 0).all(axis=1), dtype=torch.bool)

    return {
        "node_feats": node_feats,
        "coords": coords,
        "original_mask": original_mask,
        "image_offsets": image_offsets,
        "n_original": n_orig,
        "original_indices": torch.tensor(original_indices, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Dataset and collation
# ---------------------------------------------------------------------------


class JARVISCrystalDataset(Dataset):
    """PyTorch Dataset for JARVIS records with periodic graph expansion."""

    def __init__(
        self,
        records: list[dict],
        graph_builder: Callable[[Structure], dict[str, torch.Tensor]] | None = None,
        normalize_target: bool = True,
        split: str | None = None,
        use_explicit_edges: bool = False,
    ) -> None:
        if split is not None:
            records = [r for r in records if r.get("split") == split]
        self.records = records
        self.normalize_target = normalize_target
        self.use_explicit_edges = use_explicit_edges

        if graph_builder is not None:
            self.graph_builder = graph_builder
        elif use_explicit_edges:
            self.graph_builder = PeriodicEdgeGraphBuilder()
        else:
            self.graph_builder = PeriodicGraphBuilder()

        if records:
            targets = torch.tensor([r["target"] for r in records], dtype=torch.float32)
            self.target_mean = float(targets.mean())
            self.target_std = float(targets.std().clamp_min(1e-8))
        else:
            self.target_mean = 0.0
            self.target_std = 1.0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[dict[str, torch.Tensor], torch.Tensor]:
        rec = self.records[idx]
        graph = self.graph_builder(rec["structure"])
        target = torch.tensor(rec["target"], dtype=torch.float32)
        if self.normalize_target:
            target = (target - self.target_mean) / self.target_std

        if self.use_explicit_edges:
            return graph, target

        return (
            graph["node_feats"],
            graph["coords"],
            graph["original_mask"],
            target,
        )


def collate_crystals(batch: list) -> tuple:
    """Collate periodic crystal graphs into padded batched tensors.

    Returns:
        ``(node_feats, coords, mask, original_mask, targets)``.
    """
    node_feats_list, coords_list, original_mask_list, targets = [], [], [], []
    max_n = max(feats.size(0) for feats, _, _, _ in batch)
    for feats, coords, orig_mask, target in batch:
        n = feats.size(0)
        pad = max_n - n
        if pad > 0:
            feats = torch.cat([feats, torch.zeros(pad, feats.size(1))], dim=0)
            coords = torch.cat([coords, torch.zeros(pad, 3)], dim=0)
            orig_mask = torch.cat(
                [orig_mask, torch.zeros(pad, dtype=torch.bool)], dim=0
            )
        node_feats_list.append(feats)
        coords_list.append(coords)
        original_mask_list.append(orig_mask)
        targets.append(target)

    batched_feats = torch.stack(node_feats_list, dim=0)  # (B, N, F)
    batched_coords = torch.stack(coords_list, dim=0)     # (B, N, 3)
    original_mask = torch.stack(original_mask_list, dim=0)  # (B, N)
    mask = torch.stack(
        [
            torch.cat(
                [
                    torch.ones(n, dtype=torch.bool),
                    torch.zeros(max_n - n, dtype=torch.bool),
                ]
            )
            for n in (feats.size(0) for feats, _, _, _ in batch)
        ],
        dim=0,
    )  # (B, N)
    targets = torch.stack(targets)  # (B,)
    return batched_feats, batched_coords, mask, original_mask, targets
