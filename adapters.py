"""Unified residual adapter interface for continual multi-fidelity models.

Design principles (from 反馈_2.md audit):
* Structural isolation instead of gradient zeroing: each task owns a separate
  adapter bank; the parent bank is frozen by ``requires_grad=False`` and by
  excluding it from the child optimizer.
* No materialization of a full ``d_out x d_in`` weight matrix in forward.
* Exact parameter accounting via ``incremental_parameter_count``.
* All adapters support both 2D (B, d_in) and 3D (B, N, d_in) inputs so they can
  be inserted at any layer of a crystal graph encoder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_linear_chain(x: torch.Tensor, *weights: torch.Tensor) -> torch.Tensor:
    """Apply a chain of linear maps without materializing any full weight matrix.

    Each ``weight`` has shape ``(out_dim, in_dim)`` and implements
    ``z -> z @ weight.t()``.
    """
    for w in weights:
        x = F.linear(x, w)
    return x


class ResidualAdapter(nn.Module, ABC):
    """Base class for a residual adapter that maps features to features.

    The forward signature is ``forward(x) -> y`` where ``x`` and ``y`` have the
    same leading dimensions.  Property/fidelity routing is handled *outside* the
    adapter by selecting the appropriate adapter instance; this keeps the adapter
    itself simple and makes exact parameter accounting trivial.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return residual features with the same leading dims as ``x``."""
        ...

    @abstractmethod
    def incremental_parameter_count(self) -> int:
        """Number of trainable parameters added by this adapter."""
        ...


class LoRAABAdapter(ResidualAdapter):
    """LoRA-AB: low-rank residual ``U_out @ U_in^T``.

    For an input ``x`` the output is ``x @ U_in @ U_out^T``.
    """

    def __init__(self, d_in: int, d_out: int, rank: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.u_in = nn.Parameter(torch.empty(d_in, rank))
        self.u_out = nn.Parameter(torch.empty(d_out, rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.u_in, a=5 ** (1.0 / 3))
        nn.init.zeros_(self.u_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _apply_linear_chain(x, self.u_in.t(), self.u_out)

    def incremental_parameter_count(self) -> int:
        return self.d_in * self.rank + self.d_out * self.rank


class LoRAABAAdapter(ResidualAdapter):
    """LoRA-ABA: low-rank residual with trainable middle matrix.

    Output is ``x @ U_in @ M^T @ U_out^T``.

    This is functionally equivalent to ``SingleChildTuckerAdapter`` and serves
    as a fair architecture-matched baseline: same placement, same parent
    features, same parameter budget.
    """

    def __init__(self, d_in: int, d_out: int, rank: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.u_in = nn.Parameter(torch.empty(d_in, rank))
        self.middle = nn.Parameter(torch.empty(rank, rank))
        self.u_out = nn.Parameter(torch.empty(d_out, rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.u_in, a=5 ** (1.0 / 3))
        nn.init.orthogonal_(self.middle)
        nn.init.zeros_(self.u_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _apply_linear_chain(x, self.u_in.t(), self.middle, self.u_out)

    def incremental_parameter_count(self) -> int:
        return self.d_in * self.rank + self.rank * self.rank + self.d_out * self.rank


class SingleChildTuckerAdapter(ResidualAdapter):
    """Tucker-style residual for a *single* (property, fidelity) child.

    When there is only one new fidelity the property and fidelity Tucker modes
    do not provide cross-task sharing; keeping a full 4D Tucker core is
    redundant and makes parameter accounting inconsistent with the theory.  This
    adapter keeps the Tucker *name* but implements the minimal form
    ``U_out @ M @ U_in^T``, identical in function to ``LoRAABAAdapter`` but
    initialized with a Tucker-style core.

    This directly addresses 反馈_2.md 2.3 (Tucker degeneration) and 2.2
    (parameter-count mismatch).
    """

    def __init__(self, d_in: int, d_out: int, rank: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.u_in = nn.Parameter(torch.empty(d_in, rank))
        self.core = nn.Parameter(torch.empty(rank, rank))
        self.u_out = nn.Parameter(torch.empty(d_out, rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.u_in, a=5 ** (1.0 / 3))
        nn.init.orthogonal_(self.core)
        nn.init.zeros_(self.u_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _apply_linear_chain(x, self.u_in.t(), self.core, self.u_out)

    def incremental_parameter_count(self) -> int:
        return self.d_in * self.rank + self.rank * self.rank + self.d_out * self.rank


class MultiAxisTuckerAdapter(ResidualAdapter):
    """Full Tucker adapter with shared property and fidelity embeddings.

    The full update tensor is
        A = G x_1 U_out x_2 U_in x_3 E_prop x_4 E_fid
    and for a concrete ``(property p, fidelity f)`` the slice is
        Delta W_{p,f} = U_out @ (G x_3 e_p x_4 e_f) @ U_in^T.

    The forward avoids materializing ``Delta W_{p,f}`` by applying
    ``U_in^T``, the contracted core slice, and ``U_out^T`` sequentially.

    This adapter is only meaningful when ``n_properties >= 2`` or
    ``n_fidelities >= 3`` so that the property/fidelity modes are actually
    shared across tasks.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_properties: int,
        n_fidelities: int,
        rank_out: int = 8,
        rank_in: int = 8,
        rank_prop: int = 4,
        rank_fid: int = 4,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n_properties = n_properties
        self.n_fidelities = n_fidelities
        self.rank_out = rank_out
        self.rank_in = rank_in
        self.rank_prop = rank_prop
        self.rank_fid = rank_fid

        self.u_in = nn.Parameter(torch.empty(d_in, rank_in))
        self.u_out = nn.Parameter(torch.empty(d_out, rank_out))
        self.e_prop = nn.Embedding(n_properties, rank_prop)
        self.e_fid = nn.Embedding(n_fidelities, rank_fid)
        self.g = nn.Parameter(torch.empty(rank_out, rank_in, rank_prop, rank_fid))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.u_in, a=5 ** (1.0 / 3))
        nn.init.kaiming_uniform_(self.u_out, a=5 ** (1.0 / 3))
        nn.init.normal_(self.e_prop.weight, std=0.1)
        nn.init.normal_(self.e_fid.weight, std=0.1)
        nn.init.normal_(self.g, std=0.1)

    def _core_slice(self, prop_id: int, fid_id: int) -> torch.Tensor:
        e_p = self.e_prop(torch.tensor(prop_id, device=self.g.device))
        e_f = self.e_fid(torch.tensor(fid_id, device=self.g.device))
        # (rank_out, rank_in)
        return torch.einsum("oipf,p,f->oi", self.g, e_p, e_f)

    def forward(self, x: torch.Tensor, prop_id: int, fid_id: int) -> torch.Tensor:
        core = self._core_slice(prop_id, fid_id)
        return _apply_linear_chain(x, self.u_in.t(), core.t(), self.u_out)

    def incremental_parameter_count(self) -> int:
        return (
            self.d_in * self.rank_in
            + self.d_out * self.rank_out
            + self.rank_out * self.rank_in * self.rank_prop * self.rank_fid
            + self.n_properties * self.rank_prop
            + self.n_fidelities * self.rank_fid
        )

    def expand_property_axis(self, new_n_properties: int) -> None:
        """Add rows to the property embedding (used when a new property arrives)."""
        if new_n_properties <= self.n_properties:
            return
        old = self.n_properties
        old_weight = self.e_prop.weight.data
        new_rows = torch.randn(
            new_n_properties - old, self.rank_prop, device=old_weight.device
        ) * 0.1
        new_weight = torch.cat([old_weight, new_rows], dim=0)
        self.e_prop = nn.Embedding(new_n_properties, self.rank_prop)
        self.e_prop.weight.data.copy_(new_weight)
        self.n_properties = new_n_properties

    def expand_fidelity_axis(self, new_n_fidelities: int) -> None:
        """Add rows to the fidelity embedding (used when a new fidelity arrives)."""
        if new_n_fidelities <= self.n_fidelities:
            return
        old = self.n_fidelities
        old_weight = self.e_fid.weight.data
        new_rows = torch.randn(
            new_n_fidelities - old, self.rank_fid, device=old_weight.device
        ) * 0.1
        new_weight = torch.cat([old_weight, new_rows], dim=0)
        self.e_fid = nn.Embedding(new_n_fidelities, self.rank_fid)
        self.e_fid.weight.data.copy_(new_weight)
        self.n_fidelities = new_n_fidelities


class BottleneckAdapter(ResidualAdapter):
    """Simple two-layer bottleneck MLP residual for baseline comparisons."""

    def __init__(self, d_in: int, d_out: int, bottleneck: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.bottleneck = bottleneck
        self.net = nn.Sequential(
            nn.Linear(d_in, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, d_out),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** (1.0 / 3))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def incremental_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ZeroAdapter(ResidualAdapter):
    """Identity/zero adapter: adds nothing.  Useful for the parent path."""

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(*x.shape[:-1], self.d_out)

    def incremental_parameter_count(self) -> int:
        return 0


ADAPTER_REGISTRY: dict[str, type[ResidualAdapter]] = {
    "lora_ab": LoRAABAdapter,
    "lora_aba": LoRAABAAdapter,
    "single_child_tucker": SingleChildTuckerAdapter,
    "multi_axis_tucker": MultiAxisTuckerAdapter,
    "bottleneck": BottleneckAdapter,
    "zero": ZeroAdapter,
}


def make_adapter_bank(
    adapter_name: str,
    n_layers: int,
    dim: int,
    rank: int,
    n_properties: Optional[int] = None,
    n_fidelities: Optional[int] = None,
) -> nn.ModuleList:
    """Create a bank of ``n_layers`` identical adapters.

    For ``multi_axis_tucker`` the property/fidelity counts must be provided.
    For per-layer heterogeneous ranks, callers can build the list manually.
    """
    cls = ADAPTER_REGISTRY[adapter_name]
    bank = nn.ModuleList()
    for _ in range(n_layers):
        if adapter_name == "multi_axis_tucker":
            if n_properties is None or n_fidelities is None:
                raise ValueError(
                    "multi_axis_tucker requires n_properties and n_fidelities"
                )
            bank.append(
                cls(
                    dim,
                    dim,
                    n_properties,
                    n_fidelities,
                    rank_out=rank,
                    rank_in=rank,
                    rank_prop=max(2, n_properties),
                    rank_fid=max(2, n_fidelities),
                )
            )
        else:
            bank.append(cls(dim, dim, rank))
    return bank
