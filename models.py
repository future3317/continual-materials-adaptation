"""Exact-retention continual crystal property predictor.

This module replaces the monolithic PhyTCAModel with a design that enforces
exact retention by structural isolation:

* A shared crystal-graph encoder is permanently frozen.
* Each (property, fidelity) task owns a private adapter bank and head.
* When a task is frozen, its adapter bank and head are excluded from all
  future optimizers; no gradient hook or gradient-zeroing is needed.
* New tasks are added by allocating a new adapter bank + head, leaving old
  parameters physically untouched.

Adapters are taken from ``adapters.py`` and share a uniform interface so that
LoRA-AB, LoRA-ABA, single-child Tucker, multi-axis Tucker, and bottleneck MLP
baselines can be compared with identical placement, rank budget, and training
recipe.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from egnn_pytorch import EGNN

from adapters import ResidualAdapter, make_adapter_bank


class CopyOnWriteTopBlock(nn.Module):
    """Private child copy of the top encoder layer.

    Implements the copy-on-write top block baseline from 反馈_2.md 4.3:
    the parent route keeps using the original frozen top EGNN layer, while a
    new high-fidelity child receives a deep copy of that layer that is trained
    independently.  Because the parent layer is never updated, exact retention
    of the parent route is preserved.
    """

    def __init__(self, top_layer: EGNN) -> None:
        super().__init__()
        self.child_layer = copy.deepcopy(top_layer)

    def forward(
        self, h: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.child_layer(h, coords, mask=mask)


class CrystalEncoder(nn.Module):
    """Frozen crystal graph encoder that optionally accepts per-task adapters.

    The encoder is a stack of EGNN layers.  After each EGNN layer, if an
    adapter is provided it is added to the node features as a residual.  The
    encoder itself is always frozen; only the injected adapters are trainable.
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        n_layers: int = 3,
        num_nearest_neighbors: int = 8,
        update_coors: bool = False,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.num_nearest_neighbors = num_nearest_neighbors

        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                EGNN(
                    dim=hidden_dim,
                    edge_dim=0,
                    m_dim=max(16, hidden_dim),
                    num_nearest_neighbors=num_nearest_neighbors,
                    update_coors=update_coors,
                    update_feats=True,
                )
                for _ in range(n_layers)
            ]
        )

        # Encoder is permanently frozen in continual mode.
        for p in self.parameters():
            p.requires_grad = False

    def forward(
        self,
        node_feats: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        adapter_bank: Optional[Sequence[ResidualAdapter]] = None,
        private_top_block: Optional[CopyOnWriteTopBlock] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode crystal to node features.

        Args:
            node_feats: (B, N, node_dim)
            coords: (B, N, 3)
            mask: (B, N) bool padding mask.
            adapter_bank: optional list of adapters, one per layer.
            private_top_block: optional child-private copy of the top EGNN layer.

        Returns:
            (h, coords) where ``h`` has shape (B, N, hidden_dim).
        """
        h = self.node_embed(node_feats)
        n_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            if i == n_layers - 1 and private_top_block is not None:
                h, coords = private_top_block(h, coords, mask)
            else:
                h, coords = layer(h, coords, mask=mask)
            if adapter_bank is not None:
                adapter = adapter_bank[i]
                h = h + adapter(h)
        return h, coords

    def encode(
        self,
        node_feats: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        adapter_bank: Optional[Sequence[ResidualAdapter]] = None,
        private_top_block: Optional[CopyOnWriteTopBlock] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for ``forward``; kept for a uniform encoder interface."""
        return self.forward(node_feats, coords, mask, adapter_bank, private_top_block)


class ContinualCrystalModel(nn.Module):
    """Exact-retention model for continually arriving (property, fidelity) tasks.

    Args:
        node_dim: dimension of one-hot / embedding node features.
        hidden_dim: hidden dimension of the crystal encoder.
        n_properties: total number of distinct properties (used only for
            multi-axis Tucker adapters).
        n_fidelities: total number of distinct fidelities.
        adapter_name: key into ``adapters.ADAPTER_REGISTRY``.
        adapter_rank: rank used by all adapter types.
        n_layers: number of crystal-graph encoder layers.
        num_nearest_neighbors: EGNN kNN parameter.
        update_coors: whether EGNN updates coordinates (default False, see
            反馈_2.md 6.4).
        encoder: Optional pre-instantiated encoder module (e.g. ``MatGLBackbone``).
            If ``None``, the default ``CrystalEncoder`` is built.
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        n_properties: int,
        n_fidelities: int,
        adapter_name: str = "single_child_tucker",
        adapter_rank: int = 8,
        n_layers: int = 3,
        num_nearest_neighbors: int = 8,
        update_coors: bool = False,
        encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.n_properties = n_properties
        self.n_fidelities = n_fidelities
        self.adapter_name = adapter_name
        self.adapter_rank = adapter_rank
        self.n_layers = n_layers

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = CrystalEncoder(
                node_dim=node_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                num_nearest_neighbors=num_nearest_neighbors,
                update_coors=update_coors,
            )

        # One prediction head per (property, fidelity) task.
        self.heads: Dict[str, nn.Linear] = nn.ModuleDict()
        # Per-task adapter banks.
        self.adapter_banks: Dict[str, nn.ModuleList] = nn.ModuleDict()
        # Per-task private top encoder blocks (copy-on-write baseline).
        self.private_top_blocks: Dict[str, CopyOnWriteTopBlock] = nn.ModuleDict()
        self._task_order: List[Tuple[int, int]] = []
        self._frozen_tasks: set[str] = set()

    # -----------------------------------------------------------------------
    # Task lifecycle
    # -----------------------------------------------------------------------

    def _task_key(self, prop_id: int, fid_id: int) -> str:
        return f"p{int(prop_id)}_f{int(fid_id)}"

    def add_task(self, prop_id: int, fid_id: int) -> str:
        """Allocate a new adapter bank + head for ``(prop_id, fid_id)``.

        Returns the task key.  If the task already exists, no new parameters are
        added (data-incremental snapshots reuse the same route).
        """
        key = self._task_key(prop_id, fid_id)
        if key in self.heads:
            return key

        bank = make_adapter_bank(
            adapter_name=self.adapter_name,
            n_layers=self.n_layers,
            dim=self.hidden_dim,
            rank=self.adapter_rank,
            n_properties=self.n_properties,
            n_fidelities=self.n_fidelities,
        )
        self.adapter_banks[key] = bank
        self.heads[key] = nn.Linear(self.hidden_dim, 1)
        self._task_order.append((int(prop_id), int(fid_id)))
        return key

    def freeze_task(self, prop_id: int, fid_id: int) -> None:
        """Freeze the adapter bank, head, and optional private top block for a completed task."""
        key = self._task_key(prop_id, fid_id)
        self._frozen_tasks.add(key)
        if key in self.heads:
            for p in self.heads[key].parameters():
                p.requires_grad = False
        if key in self.adapter_banks:
            for adapter in self.adapter_banks[key]:
                for p in adapter.parameters():
                    p.requires_grad = False
        if key in self.private_top_blocks:
            for p in self.private_top_blocks[key].parameters():
                p.requires_grad = False

    def add_private_top_block(self, prop_id: int, fid_id: int) -> str:
        """Attach a child-private copy of the top encoder layer to a task.

        The task must already have an adapter bank/head allocated.  The private
        top block is trainable by default and is frozen together with the task
        when ``freeze_task`` is called.
        """
        key = self._task_key(prop_id, fid_id)
        if key not in self.heads:
            raise RuntimeError(f"Task {key} does not exist; call add_task first.")
        if key not in self.private_top_blocks:
            if not hasattr(self.encoder, "layers") or not self.encoder.layers:
                raise RuntimeError("Encoder has no layers to copy for private top block.")
            top_layer = self.encoder.layers[-1]
            self.private_top_blocks[key] = CopyOnWriteTopBlock(top_layer)
        return key

    def is_frozen(self, prop_id: int, fid_id: int) -> bool:
        return self._task_key(prop_id, fid_id) in self._frozen_tasks

    def current_trainable_parameters(self) -> List[nn.Parameter]:
        """Return parameters that are currently trainable.

        Because old tasks are frozen via ``requires_grad=False`` and are not in
        any child optimizer, this implements exact retention without hooks.
        """
        return [p for p in self.parameters() if p.requires_grad]

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def encode(
        self,
        node_feats: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        original_mask: torch.Tensor,
        prop_id: int,
        fid_id: int,
    ) -> torch.Tensor:
        """Return pooled crystal-level representation for a task."""
        key = self._task_key(prop_id, fid_id)
        bank = self.adapter_banks[key] if key in self.adapter_banks else None
        top_block = self.private_top_blocks[key] if key in self.private_top_blocks else None
        h, _ = self.encoder.encode(node_feats, coords, mask, adapter_bank=bank, private_top_block=top_block)
        mask_exp = original_mask.unsqueeze(-1).float()
        pooled = (h * mask_exp).sum(dim=1) / (mask_exp.sum(dim=1).clamp_min(1.0))
        return pooled

    def forward(
        self,
        node_feats: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        original_mask: torch.Tensor,
        prop_id: int,
        fid_id: int,
    ) -> torch.Tensor:
        """Predict normalized targets for ``(prop_id, fid_id)``.

        The returned value is in the *normalized* coordinate system of the
        target fidelity.  Callers must de-normalize with the task-specific
        ``target_mean`` / ``target_std`` to obtain physical units.
        """
        pooled = self.encode(node_feats, coords, mask, original_mask, prop_id, fid_id)
        key = self._task_key(prop_id, fid_id)
        return self.heads[key](pooled).squeeze(-1)

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def count_encoder_parameters(self) -> int:
        return sum(p.numel() for p in self.encoder.parameters())

    def count_task_parameters(self, prop_id: int, fid_id: int) -> int:
        """Parameters belonging to one task (head + its adapter bank + private top block)."""
        key = self._task_key(prop_id, fid_id)
        total = 0
        if key in self.heads:
            total += sum(p.numel() for p in self.heads[key].parameters())
        if key in self.adapter_banks:
            for adapter in self.adapter_banks[key]:
                total += adapter.incremental_parameter_count()
        if key in self.private_top_blocks:
            total += sum(p.numel() for p in self.private_top_blocks[key].parameters())
        return total

    def count_incremental_parameters(self, prop_id: int, fid_id: int) -> int:
        """Alias for ``count_task_parameters`` for the new task."""
        return self.count_task_parameters(prop_id, fid_id)

    def count_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_parameter_group_counts(self) -> Dict[str, int]:
        """Return encoder / adapters / heads / private top blocks breakdown."""
        encoder = self.count_encoder_parameters()
        heads = sum(sum(p.numel() for p in h.parameters()) for h in self.heads.values())
        adapters = 0
        for bank in self.adapter_banks.values():
            for adapter in bank:
                adapters += adapter.incremental_parameter_count()
        private_tops = sum(
            sum(p.numel() for p in block.parameters())
            for block in self.private_top_blocks.values()
        )
        return {
            "total": self.count_total_parameters(),
            "encoder": encoder,
            "adapters": adapters,
            "heads": heads,
            "private_top_blocks": private_tops,
        }

    def load_parent_checkpoint(
        self,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = False,
    ) -> None:
        """Load a checkpoint into the encoder and the first task bank/head.

        This is intended for warm-starting from a pre-trained parent route.
        """
        self.load_state_dict(state_dict, strict=strict)


# ---------------------------------------------------------------------------
# Copy-on-write full child baseline
# ---------------------------------------------------------------------------


class CopyOnWriteFullChildModel(nn.Module):
    """Each versioned endpoint owns a deep copy of the full encoder + a head.

    This is the strongest exact-retention baseline: every route has an
    independent network, so there is no cross-route interference at all.  The
    cost is a parameter count that grows linearly with the number of routes.
    New children are initialized by copying the most recently trained child,
    giving a warm-start while keeping older children frozen.
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        n_layers: int = 3,
        num_nearest_neighbors: int = 8,
        update_coors: bool = False,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.num_nearest_neighbors = num_nearest_neighbors

        # The template encoder is also the first child's encoder.
        self.template = CrystalEncoder(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            num_nearest_neighbors=num_nearest_neighbors,
            update_coors=update_coors,
        )
        self.child_encoders: Dict[str, CrystalEncoder] = nn.ModuleDict()
        self.heads: Dict[str, nn.Linear] = nn.ModuleDict()
        self._route_order: List[str] = []
        self._frozen_routes: set[str] = set()

    def _route_key(self, version: str, prop_id: int, fid_id: int) -> str:
        return f"v{version}_p{int(prop_id)}_f{int(fid_id)}"

    def add_route(self, version: str, prop_id: int, fid_id: int) -> str:
        """Allocate a new full child encoder and head.

        The first route uses the template encoder; later routes copy the most
        recent child before it was frozen, then unfreeze the copy for training.
        """
        key = self._route_key(version, prop_id, fid_id)
        if key in self.heads:
            return key

        if not self._route_order:
            child = self.template
        else:
            latest_key = self._route_order[-1]
            child = copy.deepcopy(self.child_encoders[latest_key])

        # Ensure the new child is trainable even if copied from a frozen parent.
        for p in child.parameters():
            p.requires_grad = True

        self.child_encoders[key] = child
        self.heads[key] = nn.Linear(self.hidden_dim, 1)
        self._route_order.append(key)
        return key

    def freeze_route(self, version: str, prop_id: int, fid_id: int) -> None:
        """Freeze a published endpoint."""
        key = self._route_key(version, prop_id, fid_id)
        self._frozen_routes.add(key)
        if key in self.child_encoders:
            for p in self.child_encoders[key].parameters():
                p.requires_grad = False
        if key in self.heads:
            for p in self.heads[key].parameters():
                p.requires_grad = False

    def is_frozen(self, version: str, prop_id: int, fid_id: int) -> bool:
        return self._route_key(version, prop_id, fid_id) in self._frozen_routes

    def current_trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

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
        """Predict for the requested endpoint using its private encoder."""
        key = self._route_key(version, prop_id, fid_id)
        if key not in self.heads:
            raise KeyError(f"Route {key} not allocated")

        child = self.child_encoders[key]
        h, _ = child(node_feats, coords, mask)
        if original_mask is None:
            pooled = h.mean(dim=1)
        else:
            mask_exp = original_mask.unsqueeze(-1).float()
            pooled = (h * mask_exp).sum(dim=1) / (mask_exp.sum(dim=1).clamp_min(1.0))
        return self.heads[key](pooled).squeeze(-1)

    def incremental_parameters(self, version: str, prop_id: int, fid_id: int) -> int:
        key = self._route_key(version, prop_id, fid_id)
        total = sum(p.numel() for p in self.child_encoders[key].parameters())
        total += sum(p.numel() for p in self.heads[key].parameters())
        return total

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Prediction-residual helpers (addresses 反馈_2.md 2.1)
# ---------------------------------------------------------------------------


class PredictionResidualHead(nn.Module):
    """Output head that explicitly learns the prediction residual in physical units.

    Given a parent prediction (in normalized parent space) and a child latent
    representation, this module de-normalizes the parent prediction, adds a
    learned physical residual, and re-normalizes to the child space.

    This avoids the cross-fidelity normalization bug where ``y_L^norm`` and
    ``delta^norm`` are added in different coordinate systems (反馈_2.md 5.1).

    The module is intended for single-fidelity correction baselines; the main
    ContinualCrystalModel uses the equivalent implicit formulation through a
    child head trained on child-normalized targets.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.residual_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize so the residual starts at zero.
        for m in self.residual_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        h: torch.Tensor,
        parent_pred_norm: torch.Tensor,
        parent_mean: torch.Tensor,
        parent_std: torch.Tensor,
        child_mean: torch.Tensor,
        child_std: torch.Tensor,
    ) -> torch.Tensor:
        """Return child prediction in child-normalized space.

        Args:
            h: pooled representation, shape (B, hidden_dim).
            parent_pred_norm: parent prediction in parent-normalized space.
            parent_mean, parent_std: parent target normalizers (physical units).
            child_mean, child_std: child target normalizers (physical units).

        Returns:
            child prediction in child-normalized space.
        """
        parent_pred_phys = parent_pred_norm * parent_std + parent_mean
        residual_phys = self.residual_mlp(h).squeeze(-1)
        child_pred_phys = parent_pred_phys + residual_phys
        return (child_pred_phys - child_mean) / child_std


# ---------------------------------------------------------------------------
# Legacy metric helpers (kept here to simplify migration from phytca.py)
# ---------------------------------------------------------------------------


def normalized_mae(pred: torch.Tensor, target: torch.Tensor, mad: float) -> torch.Tensor:
    """Normalized MAE by mean absolute deviation."""
    return torch.abs(pred - target).mean() / max(mad, 1e-8)


def compute_mad(targets: torch.Tensor) -> float:
    """Mean absolute deviation of a target tensor."""
    return float(torch.abs(targets - targets.mean()).mean())


def forgetting(nmaes: List[List[float]]) -> float:
    """Average per-task forgetting across a continual run."""
    T = len(nmaes)
    if T <= 1:
        return 0.0
    vals = []
    for i in range(T):
        best = min(nmaes[t][i] for t in range(i, T))
        final = nmaes[T - 1][i]
        vals.append(max(0.0, final - best))
    return sum(vals) / len(vals)


def backward_transfer(nmaes: List[List[float]]) -> float:
    """Average backward transfer for an error metric (lower is better)."""
    T = len(nmaes)
    if T <= 1:
        return 0.0
    vals = []
    for i in range(T - 1):
        best = nmaes[i][i]
        final = nmaes[T - 1][i]
        vals.append(best - final)
    return sum(vals) / len(vals)


def forward_transfer(nmaes: List[List[float]], scratch_nmaes: List[float]) -> float:
    """Average forward transfer vs training each task from scratch."""
    if not nmaes:
        return 0.0
    return sum(scratch_nmaes[t] - nmaes[t][t] for t in range(len(nmaes))) / len(nmaes)
