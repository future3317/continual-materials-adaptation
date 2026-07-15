"""Optional stronger backbones for ``models.ContinualCrystalModel``.

This module provides a MatGL-backed encoder interface that can be swapped in as
a drop-in replacement for the default EGNN ``CrystalEncoder``.  The primary
interface is the sparse periodic graph produced by ``periodic_graph.py``; a
dense padded-tensor path is also supported for backward compatibility by
building a minimal kNN graph from the non-padded atoms (molecule-like, no PBC).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from pymatgen.core import Element
from torch_geometric.data import Data

try:
    from matgl.config import DEFAULT_ELEMENTS
    from matgl.models import M3GNet
    from matgl.utils.io import load_model

    _MATGL_AVAILABLE = True
except Exception:  # pragma: no cover - MatGL is an optional backbone dependency
    _MATGL_AVAILABLE = False
    DEFAULT_ELEMENTS = ()  # type: ignore[misc, assignment]
    M3GNet = None  # type: ignore[misc, assignment]
    load_model = None  # type: ignore[misc, assignment]


class MatGLBackbone(nn.Module):
    """Wrap a MatGL model (e.g. M3GNet) as a node-level feature encoder.

    The wrapped model is permanently frozen by default so it can serve as a
    shared backbone in the exact-retention continual-learning setup.  Node
    features from the last MatGL graph-convolution block are projected to
    ``hidden_dim``.

    Args:
        model_name: Either a pre-instantiated MatGL ``nn.Module`` (useful for
            tests with random weights), a string identifier such as
            ``"M3GNet-MP-2021.2.8-PES"`` that ``matgl.load_model`` understands,
            or a local path to a saved MatGL model directory.
        hidden_dim: Dimension of the node features returned by this encoder.
        freeze: If ``True`` (default), set ``requires_grad=False`` on all
            backbone parameters.
    """

    def __init__(
        self,
        model_name: Union[str, nn.Module],
        hidden_dim: int,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if not _MATGL_AVAILABLE:
            raise ImportError(
                "MatGL is required for MatGLBackbone but could not be imported."
            )

        self.hidden_dim = hidden_dim
        self.freeze = freeze
        self.num_nearest_neighbors = 8

        if isinstance(model_name, nn.Module):
            self.matgl_model = model_name
        else:
            self.matgl_model = load_model(model_name)

        self.element_types = tuple(getattr(self.matgl_model, "element_types", DEFAULT_ELEMENTS))
        self._max_element_z = max((Element(elem).Z for elem in self.element_types), default=0)
        self._z_to_index = torch.full(
            (self._max_element_z + 1,), -1, dtype=torch.long
        )
        for idx, elem in enumerate(self.element_types):
            self._z_to_index[Element(elem).Z] = idx

        self.node_dim = getattr(
            self.matgl_model, "dim_node_embedding", self.hidden_dim
        )
        if self.node_dim != self.hidden_dim:
            self.projection = nn.Linear(self.node_dim, self.hidden_dim)
        else:
            self.projection = nn.Identity()

        if freeze:
            for p in self.matgl_model.parameters():
                p.requires_grad = False

    def _node_feats_to_node_type(self, node_feats: torch.Tensor) -> torch.Tensor:
        """Convert one-hot element vectors to MatGL ``node_type`` indices."""
        device = node_feats.device
        z = node_feats.argmax(dim=-1) + 1  # atomic number
        z_to_index = self._z_to_index.to(device)
        node_type = z_to_index[z]
        if (node_type < 0).any():
            invalid = z[node_type < 0].unique().tolist()
            raise ValueError(
                f"Found elements with Z={invalid} that are not in the model's element_types."
            )
        return node_type

    def _build_knn_edges(
        self, pos: torch.Tensor, k: int
    ) -> torch.Tensor:
        """Return directed kNN edge indices for a set of positions."""
        n = pos.size(0)
        if n <= 1:
            return torch.zeros((2, 0), dtype=torch.long, device=pos.device)
        k = min(k, n - 1)
        dist = torch.cdist(pos, pos)
        dist.fill_diagonal_(float("inf"))
        knn = torch.topk(dist, k, largest=False, dim=1).indices
        src = torch.arange(n, device=pos.device).unsqueeze(1).expand(-1, k).reshape(-1)
        dst = knn.reshape(-1)
        return torch.stack([src, dst], dim=0)

    def _graph_dict_to_pyg(self, graph_dict: dict[str, torch.Tensor]) -> Data:
        """Convert a ``periodic_graph.py`` sparse graph dict to a PyG ``Data``."""
        pos = graph_dict["coords"]
        edge_index = graph_dict["edge_index"]
        lattice = graph_dict["lattice"]
        shifts = graph_dict["edge_shifts"].to(dtype=pos.dtype)
        pbc_offshift = shifts @ lattice.T
        node_type = self._node_feats_to_node_type(graph_dict["node_feats"])
        batch = graph_dict.get(
            "batch",
            torch.zeros(pos.size(0), dtype=torch.long, device=pos.device),
        )
        return Data(
            pos=pos,
            node_type=node_type,
            edge_index=edge_index,
            pbc_offshift=pbc_offshift,
            batch=batch,
        )

    def _run_matgl(self, data: Data) -> torch.Tensor:
        """Run the MatGL model on a PyG ``Data`` and return projected node features."""
        self.matgl_model(data)
        n_blocks = getattr(self.matgl_model, "n_blocks", None)
        if n_blocks is None:
            # Fallback: some MatGL models store the block count differently.
            n_blocks = getattr(self.matgl_model, "nblocks", 1)
        node_feat = self.matgl_model.feature_dict[f"gc_{n_blocks}"]["node_feat"]
        return self.projection(node_feat)

    def forward(self, graph_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode a sparse periodic graph to node features.

        Args:
            graph_dict: Dictionary from ``periodic_graph.py`` with keys
                ``node_feats``, ``coords``, ``edge_index``, ``edge_shifts``,
                ``lattice``, and optionally ``batch``.

        Returns:
            Node features of shape ``(N, hidden_dim)`` where ``N`` is the total
            number of nodes in the sparse graph.
        """
        data = self._graph_dict_to_pyg(graph_dict)
        return self._run_matgl(data)

    def encode(
        self,
        node_feats: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        adapter_bank: Optional[Sequence[nn.Module]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode dense padded tensors to node features.

        This path builds a minimal molecule-like kNN graph from the non-padded
        atoms in each batch item.  Periodic boundary conditions are *not*
        enforced here; use :meth:`forward` with a sparse periodic graph for
        crystal-aware encoding.

        Args:
            node_feats: ``(B, N, node_dim)`` one-hot / embedding node features.
            coords: ``(B, N, 3)`` Cartesian coordinates.
            mask: ``(B, N)`` bool padding mask; ``True`` marks real atoms.
            adapter_bank: Ignored; retained for signature compatibility with
                ``CrystalEncoder.forward``.

        Returns:
            ``(h, coords)`` where ``h`` has shape ``(B, N, hidden_dim)``.
        """
        B, N, _ = node_feats.shape
        device = node_feats.device
        dtype = coords.dtype

        all_pos: list[torch.Tensor] = []
        all_node_type: list[torch.Tensor] = []
        all_edge_index: list[torch.Tensor] = []
        all_batch: list[torch.Tensor] = []
        all_pbc_offshift: list[torch.Tensor] = []
        valid_indices: list[torch.Tensor] = []

        cumsum = 0
        for b in range(B):
            valid = mask[b]
            n_real = int(valid.sum().item())
            if n_real == 0:
                continue
            nf_b = node_feats[b, valid]
            pos_b = coords[b, valid]
            node_type_b = self._node_feats_to_node_type(nf_b)
            k = min(self.num_nearest_neighbors, max(1, n_real - 1))
            edge_index_b = self._build_knn_edges(pos_b, k=k)

            all_pos.append(pos_b)
            all_node_type.append(node_type_b)
            all_edge_index.append(edge_index_b + cumsum)
            all_batch.append(torch.full((n_real,), b, dtype=torch.long, device=device))
            all_pbc_offshift.append(
                torch.zeros(edge_index_b.size(1), 3, dtype=dtype, device=device)
            )
            valid_indices.append(torch.where(valid)[0])
            cumsum += n_real

        h_dense = torch.zeros(B, N, self.hidden_dim, device=device, dtype=dtype)
        if cumsum == 0:
            return h_dense, coords

        data = Data(
            pos=torch.cat(all_pos, dim=0),
            node_type=torch.cat(all_node_type, dim=0),
            edge_index=torch.cat(all_edge_index, dim=1),
            batch=torch.cat(all_batch, dim=0),
            pbc_offshift=torch.cat(all_pbc_offshift, dim=0),
        )
        h_sparse = self._run_matgl(data)  # (cumsum, hidden_dim)

        offset = 0
        for b, idx in enumerate(valid_indices):
            n_real = idx.size(0)
            h_dense[b, idx] = h_sparse[offset : offset + n_real]
            offset += n_real

        return h_dense, coords

    def count_parameters(self) -> int:
        """Return the number of backbone parameters (including projection)."""
        return sum(p.numel() for p in self.parameters())


def build_matgl_backbone(
    model_name: Optional[Union[str, nn.Module]] = None,
    hidden_dim: Optional[int] = None,
    freeze: bool = True,
) -> MatGLBackbone:
    """Create a ``MatGLBackbone``.

    If ``model_name`` is ``None``, a tiny randomly-initialized M3GNet is built
    for fast tests that do not download pre-trained weights.

    Args:
        model_name: Pre-instantiated MatGL model, model identifier, or local path.
        hidden_dim: Output node-feature dimension.  Defaults to the MatGL model's
            node embedding dimension when ``model_name`` is provided, otherwise 16.
        freeze: Whether to freeze the backbone.

    Returns:
        A ``MatGLBackbone`` instance.
    """
    if not _MATGL_AVAILABLE:
        raise ImportError(
            "MatGL is required for build_matgl_backbone but could not be imported."
        )

    if model_name is None:
        # Tiny architecture for fast, weight-free tests.
        matgl_hidden = 16 if hidden_dim is None else max(hidden_dim, 8)
        model = M3GNet(
            element_types=tuple(DEFAULT_ELEMENTS),
            dim_node_embedding=matgl_hidden,
            dim_edge_embedding=8,
            ntypes_state=None,
            dim_state_embedding=0,
            max_n=2,
            max_l=2,
            nblocks=1,
            rbf_type="SphericalBessel",
            is_intensive=True,
            readout_type="weighted_atom",
            cutoff=4.0,
            threebody_cutoff=3.0,
            units=matgl_hidden,
            ntargets=1,
        )
        return MatGLBackbone(
            model_name=model,
            hidden_dim=hidden_dim if hidden_dim is not None else matgl_hidden,
            freeze=freeze,
        )

    if isinstance(model_name, nn.Module):
        model = model_name
    else:
        model = load_model(model_name)

    native_dim = getattr(model, "dim_node_embedding", hidden_dim or 64)
    return MatGLBackbone(
        model_name=model,
        hidden_dim=hidden_dim if hidden_dim is not None else native_dim,
        freeze=freeze,
    )
