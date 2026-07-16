"""Versioned fidelity graph for backward-compatible model serving.

This module reframes the continual-learning problem as serving a set of
immutable, versioned prediction endpoints.  Each endpoint is identified by
``(version, property, fidelity)``.  Shared low-rank bases can be trained during
a warm-up phase, but once an endpoint is published the bases that it uses are
frozen; subsequent endpoints add new coefficient matrices on top of the frozen
bases.  This gives exact retention of published endpoints by construction.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import CrystalEncoder


class SharedBasisAdapterBank(nn.Module):
    """A bank of route-specific residuals that share a common low-rank basis.

    For each layer the shared basis is ``U_in`` and ``U_out``.  A route ``r``
    owns a private middle matrix ``M_r``.  The residual for route ``r`` is

        delta_r(x) = x @ U_in @ M_r^T @ U_out^T.

    When a route is published, its ``M_r`` is frozen.  If ``U_in`` and ``U_out``
    are also frozen, published routes are structurally isolated from future
    updates.
    """

    def __init__(self, dim: int, rank: int, bases_trainable: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.u_in = nn.Parameter(torch.empty(dim, rank))
        self.u_out = nn.Parameter(torch.empty(dim, rank))
        self.route_m = nn.ParameterDict()

        self.bases_trainable = bases_trainable
        self.reset_parameters()
        self._set_bases_trainable(bases_trainable)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.u_in, a=5 ** (1.0 / 3))
        nn.init.kaiming_uniform_(self.u_out, a=5 ** (1.0 / 3))

    def _set_bases_trainable(self, trainable: bool) -> None:
        self.bases_trainable = trainable
        self.u_in.requires_grad = trainable
        self.u_out.requires_grad = trainable

    def add_route(self, key: str) -> None:
        """Allocate a new private middle matrix for ``key``."""
        if key in self.route_m:
            return
        self.route_m[key] = nn.Parameter(torch.empty(self.rank, self.rank))
        nn.init.orthogonal_(self.route_m[key])

    def freeze_route(self, key: str) -> None:
        """Freeze the coefficient matrix for a published route."""
        if key in self.route_m:
            self.route_m[key].requires_grad = False

    def forward(self, x: torch.Tensor, key: str) -> torch.Tensor:
        """Apply the residual for route ``key``."""
        if key not in self.route_m:
            raise KeyError(f"Route {key} not allocated in adapter bank")
        m = self.route_m[key]
        # Chain: x -> U_in -> M^T -> U_out^T, without materializing full matrix.
        h = F.linear(x, self.u_in.t())
        h = F.linear(h, m)
        return F.linear(h, self.u_out)


class VersionedFidelityGraph(nn.Module):
    """Backward-compatible crystal-property predictor with versioned endpoints.

    Args:
        node_dim: Input one-hot node feature dimension.
        hidden_dim: Hidden dimension of the crystal encoder and adapters.
        n_layers: Number of crystal-graph encoder layers.
        rank: Rank of the shared low-rank basis.
        num_nearest_neighbors: EGNN kNN parameter.
        bases_trainable: Whether shared bases are trainable before the first
            endpoint is published.  Defaults to True.
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        n_layers: int = 3,
        rank: int = 8,
        num_nearest_neighbors: int = 8,
        bases_trainable: bool = True,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rank = rank

        self.encoder = CrystalEncoder(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            num_nearest_neighbors=num_nearest_neighbors,
            update_coors=False,
        )
        # The encoder is permanently frozen for exact retention.
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.adapter_banks = nn.ModuleList(
            [
                SharedBasisAdapterBank(hidden_dim, rank, bases_trainable=bases_trainable)
                for _ in range(n_layers)
            ]
        )
        self.heads: nn.ModuleDict = nn.ModuleDict()
        self._route_order: list[str] = []
        self._published_routes: set[str] = set()

    def _route_key(self, version: str, prop_id: int, fid_id: int) -> str:
        return f"v{version}_p{int(prop_id)}_f{int(fid_id)}"

    def add_route(self, version: str, prop_id: int, fid_id: int) -> str:
        """Allocate a new versioned endpoint."""
        key = self._route_key(version, prop_id, fid_id)
        if key in self.heads:
            return key
        for bank in self.adapter_banks:
            bank.add_route(key)
        self.heads[key] = nn.Linear(self.hidden_dim, 1)
        self._route_order.append(key)
        return key

    def publish_route(self, version: str, prop_id: int, fid_id: int) -> None:
        """Publish an endpoint: freeze its parameters and the shared bases it uses.

        Once published, the endpoint's predictions cannot change, because the
        shared bases and the route's private coefficients are excluded from all
        future optimizers.
        """
        key = self._route_key(version, prop_id, fid_id)
        if key not in self.heads:
            raise KeyError(f"Route {key} does not exist")
        self._published_routes.add(key)
        for bank in self.adapter_banks:
            bank.freeze_route(key)
            bank._set_bases_trainable(False)
        self.heads[key].requires_grad_(False)

    def is_published(self, version: str, prop_id: int, fid_id: int) -> bool:
        return self._route_key(version, prop_id, fid_id) in self._published_routes

    def forward(
        self,
        node_feats: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        original_mask: torch.Tensor,
        version: str,
        prop_id: int,
        fid_id: int,
    ) -> torch.Tensor:
        """Predict for the requested versioned endpoint."""
        key = self._route_key(version, prop_id, fid_id)
        if key not in self.heads:
            raise KeyError(f"Route {key} not allocated")

        h = self.encoder.node_embed(node_feats)
        c = coords
        for i, layer in enumerate(self.encoder.layers):
            h, c = layer(h, c, mask=mask)
            h = h + self.adapter_banks[i](h, key)

        pooled = (
            h.sum(dim=1)
            if original_mask is None
            else (h * original_mask.unsqueeze(-1)).sum(dim=1)
        )
        return self.heads[key](pooled).squeeze(-1)

    def incremental_parameters(self, version: str, prop_id: int, fid_id: int) -> int:
        """Parameters added for one endpoint (excluding shared bases)."""
        key = self._route_key(version, prop_id, fid_id)
        total = sum(bank.route_m[key].numel() for bank in self.adapter_banks)
        total += sum(p.numel() for p in self.heads[key].parameters())
        return total

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def current_trainable_parameters(self) -> list[nn.Parameter]:
        """Parameters that are currently trainable."""
        return [p for p in self.parameters() if p.requires_grad]
