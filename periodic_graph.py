"""Explicit periodic-edge graph builder for crystal structures.

This module builds a sparse periodic graph whose nodes are exactly the atoms in
the input unit cell.  Edges are created by a real-space cutoff and carry an
integer lattice shift ``n_ij`` so that the relative displacement is

    r_ij = x_j + L @ n_ij - x_i,

where ``x`` are Cartesian coordinates and ``L`` is the lattice matrix.  This
avoids the supercell expansion used by ``data.PeriodicGraphBuilder`` and keeps
pooling over the original unit-cell atoms only.

The returned dictionaries are compatible with PyTorch Geometric ``Data`` and
``Batch`` objects, and a helper produces the dense padded tensor format consumed
by ``models.ContinualCrystalModel`` for backward compatibility.
"""

from __future__ import annotations

import numpy as np
import torch
from pymatgen.core import Structure

try:
    from torch_geometric.data import Batch, Data

    _HAS_PYG = True
except Exception:  # pragma: no cover - PyG is a hard dependency elsewhere
    _HAS_PYG = False
    Batch = None  # type: ignore[misc, assignment]
    Data = None  # type: ignore[misc, assignment]


def _element_one_hot(element: str, dim: int = 92) -> torch.Tensor:
    """One-hot encode an element by atomic number (Z <= dim)."""
    from pymatgen.core import Element

    z = int(Element(element).Z)
    vec = torch.zeros(dim)
    vec[min(z, dim) - 1] = 1.0
    return vec


def _compute_edge_vectors(
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    edge_shifts: torch.Tensor,
    lattice: torch.Tensor,
) -> torch.Tensor:
    """Return r_ij = x_j + L @ n_ij - x_i for each edge."""
    src = edge_index[0]
    dst = edge_index[1]
    shift = edge_shifts.to(dtype=coords.dtype)
    # (E, 3) Cartesian displacement of the periodic image of dst.
    return coords[dst] + shift @ lattice.T - coords[src]


def _cap_neighbors(
    center: np.ndarray,
    point: np.ndarray,
    offsets: np.ndarray,
    distances: np.ndarray,
    max_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep the closest ``max_neighbors`` edges per center atom."""
    order = np.lexsort((distances, center))
    center = center[order]
    point = point[order]
    offsets = offsets[order]
    distances = distances[order]

    keep = np.ones(len(center), dtype=bool)
    current_center = -1
    count = 0
    for i, c in enumerate(center):
        if c != current_center:
            current_center = c
            count = 0
        if count >= max_neighbors:
            keep[i] = False
        count += 1

    return center[keep], point[keep], offsets[keep], distances[keep]


def build_periodic_edge_graph(
    structure: Structure,
    cutoff: float,
    max_neighbors: int | None = None,
    node_feature_dim: int = 92,
    exclude_self: bool = True,
) -> dict[str, torch.Tensor]:
    """Build an explicit periodic-edge graph for ``structure``.

    Args:
        structure: Input crystal structure (unit cell).
        cutoff: Real-space cutoff radius for edges.
        max_neighbors: Optional per-atom cap on the number of neighbors kept.
            When given, only the closest ``max_neighbors`` neighbors inside the
            cutoff are retained.  The graph is still cutoff-based; this is only
            a budget cap.
        node_feature_dim: Dimensionality of one-hot element node features.
        exclude_self: If ``True`` (default), exclude self-edges where a periodic
            image of an atom coincides with itself.

    Returns:
        Dictionary with keys:
          * ``node_feats``: (N, F) one-hot features for unit-cell atoms.
          * ``coords``: (N, 3) Cartesian coordinates.
          * ``edge_index``: (2, E) LongTensor, ``[src, dst]``.
          * ``edge_shifts``: (E, 3) integer lattice shifts ``n_ij``.
          * ``edge_vectors``: (E, 3) relative displacement vectors ``r_ij``.
          * ``batch``: (N,) zero-filled batch assignment for a single graph.
          * ``n_original``: number of unit-cell atoms (equal to ``N``).
    """
    n_orig = len(structure)
    node_feats = torch.stack(
        [_element_one_hot(str(site.specie), node_feature_dim) for site in structure]
    )
    coords = torch.tensor(structure.cart_coords, dtype=torch.float32)
    lattice = torch.tensor(structure.lattice.matrix, dtype=torch.float32)

    if n_orig == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_shifts = torch.empty((0, 3), dtype=torch.long)
        edge_vectors = torch.empty((0, 3), dtype=torch.float32)
    else:
        center, point, offsets, distances = structure.get_neighbor_list(
            r=cutoff, exclude_self=exclude_self
        )

        if max_neighbors is not None and max_neighbors > 0 and len(center) > 0:
            center, point, offsets, distances = _cap_neighbors(
                np.asarray(center),
                np.asarray(point),
                np.asarray(offsets),
                np.asarray(distances),
                int(max_neighbors),
            )

        edge_index = torch.tensor(
            np.stack([center, point], axis=0), dtype=torch.long
        )
        edge_shifts = torch.tensor(offsets, dtype=torch.long)
        edge_vectors = _compute_edge_vectors(coords, edge_index, edge_shifts, lattice)

    return {
        "node_feats": node_feats,
        "coords": coords,
        "edge_index": edge_index,
        "edge_shifts": edge_shifts,
        "edge_vectors": edge_vectors,
        "lattice": lattice,
        "batch": torch.zeros(n_orig, dtype=torch.long),
        "n_original": n_orig,
    }


def structure_to_periodic_edge_graph(
    structure: Structure,
    cutoff: float,
    max_neighbors: int | None = None,
    node_feature_dim: int = 92,
    exclude_self: bool = True,
) -> dict[str, torch.Tensor]:
    """Thin wrapper around ``build_periodic_edge_graph`` for a pymatgen ``Structure``."""
    return build_periodic_edge_graph(
        structure, cutoff, max_neighbors, node_feature_dim, exclude_self
    )


def to_dense_tensors(
    graph: dict[str, torch.Tensor],
    max_atoms: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a sparse periodic graph to the dense padded format.

    .. warning::

        This function exists only for backward compatibility with models that
        consume dense padded tensors (e.g. the default EGNN encoder).  The
        returned ``coords`` matrix contains only the unit-cell atom positions;
        it does **not** encode the periodic lattice shifts stored in
        ``graph['edge_shifts']``.  Consequently, a dense EGNN that recomputes
        distances from ``coords`` will *not* see the correct periodic
        displacements.  For crystal-aware encoding, use a sparse-graph backbone
        (MatGL/ALIGNN) with :meth:`forward` on the sparse graph dict.

    Returns:
        ``(node_feats, coords, mask, original_mask)`` compatible with
        ``models.ContinualCrystalModel``.  Because the sparse graph only stores
        unit-cell atoms, ``original_mask`` is all-``True``.
    """
    import warnings

    warnings.warn(
        "to_dense_tensors drops periodic lattice shifts; the dense EGNN path "
        "does not enforce periodic boundary conditions. Use a sparse-graph "
        "backbone (MatGL/ALIGNN) for crystal-aware encoding.",
        UserWarning,
        stacklevel=2,
    )

    node_feats = graph["node_feats"]
    coords = graph["coords"]
    n = node_feats.size(0)

    if max_atoms is None:
        max_atoms = n
    if max_atoms < n:
        raise ValueError(
            f"max_atoms ({max_atoms}) is smaller than the number of atoms ({n})."
        )

    pad = max_atoms - n
    if pad > 0:
        node_feats = torch.cat(
            [node_feats, torch.zeros(pad, node_feats.size(1), dtype=node_feats.dtype)],
            dim=0,
        )
        coords = torch.cat([coords, torch.zeros(pad, 3, dtype=coords.dtype)], dim=0)

    mask = torch.cat([torch.ones(n, dtype=torch.bool), torch.zeros(pad, dtype=torch.bool)])
    original_mask = torch.ones(max_atoms, dtype=torch.bool)
    return node_feats, coords, mask, original_mask


def collate_periodic_graphs(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Collate a list of sparse periodic graphs into a batched dictionary.

    Returns:
        Dictionary with keys:
          * ``node_feats``, ``coords``: concatenated node tensors.
          * ``edge_index``, ``edge_shifts``, ``edge_vectors``: concatenated and
            offset edge tensors.
          * ``batch``: (N_total,) LongTensor assigning each node to its graph.
          * ``dense_node_feats``, ``dense_coords``, ``dense_mask``,
            ``dense_original_mask``: dense padded tensors compatible with the
            existing model interface.
          * ``pyg_batch``: PyTorch Geometric ``Batch`` object (if PyG is
            available).
    """
    if not batch:
        raise ValueError("Cannot collate an empty list of graphs.")

    node_feats = torch.cat([g["node_feats"] for g in batch], dim=0)
    coords = torch.cat([g["coords"] for g in batch], dim=0)

    batch_vec_list: list[torch.Tensor] = []
    edge_index_list: list[torch.Tensor] = []
    edge_shifts_list: list[torch.Tensor] = []
    edge_vectors_list: list[torch.Tensor] = []
    cumsum = 0

    for i, g in enumerate(batch):
        n = g["node_feats"].size(0)
        batch_vec_list.append(torch.full((n,), i, dtype=torch.long))
        edge_index_list.append(g["edge_index"] + cumsum)
        edge_shifts_list.append(g["edge_shifts"])
        edge_vectors_list.append(g["edge_vectors"])
        cumsum += n

    batched: dict[str, torch.Tensor] = {
        "node_feats": node_feats,
        "coords": coords,
        "edge_index": (
            torch.cat(edge_index_list, dim=1)
            if edge_index_list
            else torch.empty((2, 0), dtype=torch.long)
        ),
        "edge_shifts": (
            torch.cat(edge_shifts_list, dim=0)
            if edge_shifts_list
            else torch.empty((0, 3), dtype=torch.long)
        ),
        "edge_vectors": (
            torch.cat(edge_vectors_list, dim=0)
            if edge_vectors_list
            else torch.empty((0, 3), dtype=torch.float32)
        ),
        "batch": torch.cat(batch_vec_list),
    }

    # Dense padded tensors for backward compatibility with ContinualCrystalModel.
    max_atoms = max(g["node_feats"].size(0) for g in batch)
    dense_node_feats, dense_coords, dense_mask, dense_original_mask = [], [], [], []
    for g in batch:
        nf, c, m, om = to_dense_tensors(g, max_atoms=max_atoms)
        dense_node_feats.append(nf)
        dense_coords.append(c)
        dense_mask.append(m)
        dense_original_mask.append(om)

    batched["dense_node_feats"] = torch.stack(dense_node_feats, dim=0)
    batched["dense_coords"] = torch.stack(dense_coords, dim=0)
    batched["dense_mask"] = torch.stack(dense_mask, dim=0)
    batched["dense_original_mask"] = torch.stack(dense_original_mask, dim=0)

    if _HAS_PYG:
        data_list = []
        for g in batch:
            data = Data(
                x=g["node_feats"],
                pos=g["coords"],
                edge_index=g["edge_index"],
                edge_attr=g["edge_vectors"],
            )
            data.edge_shifts = g["edge_shifts"]
            data_list.append(data)
        batched["pyg_batch"] = Batch.from_data_list(data_list)

    return batched
