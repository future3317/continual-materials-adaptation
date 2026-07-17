"""Persistent Consolidation Graph (PCG) for backward-compatible model serving.

A published endpoint is immutable.  New endpoints first acquire knowledge through
a fast private adapter, then project that update onto an append-only basis bank.
When the update is too novel to be represented by existing bases, new basis
blocks are appended.  Old blocks are never modified, so every published endpoint
remains structurally invariant.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasisBlock(nn.Module):
    """One append-only block of shared low-rank bases."""

    def __init__(
        self,
        block_id: str,
        dim: int,
        rank: int,
        created_by_version: str,
        plastic: bool = True,
    ) -> None:
        super().__init__()
        self.block_id = block_id
        self.dim = dim
        self.rank = rank
        self.created_by_version = created_by_version
        self.plastic = plastic

        self.u_in = nn.Parameter(torch.empty(dim, rank))
        self.u_out = nn.Parameter(torch.empty(dim, rank))
        self.reset_parameters()
        self._set_plastic(plastic)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.u_in, a=5 ** (1.0 / 3))
        nn.init.kaiming_uniform_(self.u_out, a=5 ** (1.0 / 3))

    def _set_plastic(self, plastic: bool) -> None:
        self.plastic = plastic
        self.u_in.requires_grad = plastic
        self.u_out.requires_grad = plastic

    def freeze(self) -> None:
        self._set_plastic(False)

    def parameter_hash(self) -> str:
        """Return a SHA256 hex digest of the block parameters."""
        h = hashlib.sha256()
        for p in (self.u_in, self.u_out):
            h.update(p.detach().cpu().numpy().tobytes())
        return h.hexdigest()[:16]


class BasisBank(nn.Module):
    """Append-only bank of low-rank basis blocks.

    Old blocks are frozen once they are referenced by a published route.  New
    knowledge can reuse existing blocks, append new blocks extracted from a fast
    adapter residual, or create a private copy-on-write block.
    """

    def __init__(self, dim: int, default_rank: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.default_rank = default_rank
        self.blocks: nn.ModuleDict = nn.ModuleDict()
        self._block_counter = 0

    def add_block(
        self,
        u_in: torch.Tensor | None = None,
        u_out: torch.Tensor | None = None,
        rank: int | None = None,
        created_by_version: str = "",
        plastic: bool = True,
    ) -> str:
        """Append a new basis block and return its id."""
        rank = rank or self.default_rank
        block_id = f"b{self._block_counter}"
        self._block_counter += 1
        block = BasisBlock(block_id, self.dim, rank, created_by_version, plastic=plastic)
        if u_in is not None:
            block.u_in.data.copy_(u_in)
        if u_out is not None:
            block.u_out.data.copy_(u_out)
        self.blocks[block_id] = block
        return block_id

    def freeze_block(self, block_id: str) -> None:
        if block_id in self.blocks:
            self.blocks[block_id].freeze()

    def freeze_blocks(self, block_ids: Sequence[str]) -> None:
        for bid in block_ids:
            self.freeze_block(bid)

    def get_block(self, block_id: str) -> BasisBlock:
        return self.blocks[block_id]

    def measure_novelty(
        self,
        fast_update: torch.Tensor,
        existing_blocks: Sequence[BasisBlock],
        eps: float = 1e-12,
    ) -> float:
        """Novelty ratio of a full-rank update versus existing basis blocks.

        Args:
            fast_update: Full update matrix of shape ``(dim, dim)``.
            existing_blocks: Blocks currently in the bank.

        Returns:
            Ratio ``|R|_F^2 / (|ΔW|_F^2 + eps)`` where ``R`` is the component of
            ``ΔW`` orthogonal to the existing in/out subspaces.
        """
        if not existing_blocks:
            return 1.0

        # Concatenate existing bases on the same device as the update.
        device = fast_update.device
        u_in = torch.cat([b.u_in.detach().to(device) for b in existing_blocks], dim=1)  # (dim, R)
        u_out = torch.cat([b.u_out.detach().to(device) for b in existing_blocks], dim=1)

        # Projection matrices.
        p_in = u_in @ torch.linalg.pinv(u_in.T @ u_in) @ u_in.T
        p_out = u_out @ torch.linalg.pinv(u_out.T @ u_out) @ u_out.T

        residual = fast_update - p_out @ fast_update @ p_in
        numer = float(residual.norm("fro") ** 2)
        denom = float(fast_update.norm("fro") ** 2) + eps
        return numer / denom

    def reuse_or_expand(
        self,
        fast_update: torch.Tensor,
        novelty_threshold: float = 0.2,
        svd_energy: float = 0.9,
        created_by_version: str = "",
    ) -> tuple[list[str], list[str]]:
        """Decide whether to reuse existing blocks or append new ones.

        For simplicity the current implementation assumes a single-layer residual
        update.  Multi-layer residuals are handled by calling this per layer.

        Returns:
            (selected_existing_block_ids, new_block_ids)
        """
        existing_blocks = list(self.blocks.values())
        novelty = self.measure_novelty(fast_update, existing_blocks)

        if novelty <= novelty_threshold and existing_blocks:
            # Reuse all existing blocks.
            return list(self.blocks.keys()), []

        # Expand: extract new basis from the residual via truncated SVD.
        if existing_blocks:
            device = fast_update.device
            u_in = torch.cat([b.u_in.detach().to(device) for b in existing_blocks], dim=1)
            u_out = torch.cat([b.u_out.detach().to(device) for b in existing_blocks], dim=1)
            p_in = u_in @ torch.linalg.pinv(u_in.T @ u_in) @ u_in.T
            p_out = u_out @ torch.linalg.pinv(u_out.T @ u_out) @ u_out.T
            residual = fast_update - p_out @ fast_update @ p_in
        else:
            residual = fast_update

        u_out_new, s, vh = torch.linalg.svd(residual, full_matrices=False)
        energies = torch.cumsum(s**2, dim=0)
        total_energy = energies[-1].clamp_min(1e-12)
        rank = int((energies / total_energy >= svd_energy).nonzero(as_tuple=True)[0][0].item()) + 1

        u_in_new = vh[:rank].T  # (dim, rank)
        u_out_new = u_out_new[:, :rank]  # (dim, rank)

        new_block_id = self.add_block(
            u_in=u_in_new,
            u_out=u_out_new,
            rank=rank,
            created_by_version=created_by_version,
            plastic=True,
        )
        return list(self.blocks.keys()), [new_block_id]

    def orthogonality_loss(self, new_block_ids: Sequence[str]) -> torch.Tensor:
        """Penalty encouraging new blocks to be orthogonal to old ones."""
        loss = torch.tensor(0.0)
        device = next(iter(self.blocks.parameters())).device
        loss = loss.to(device)

        new_blocks = [self.blocks[bid] for bid in new_block_ids]
        old_blocks = [b for bid, b in self.blocks.items() if bid not in new_block_ids]
        if not new_blocks or not old_blocks:
            return loss

        for nb in new_blocks:
            for ob in old_blocks:
                loss = loss + (nb.u_in.T @ ob.u_in).norm("fro") ** 2
                loss = loss + (nb.u_out.T @ ob.u_out).norm("fro") ** 2
        return loss

    def forward(self, x: torch.Tensor, block_ids: Sequence[str], coefficients: dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply selected basis blocks with route-private coefficients.

        Args:
            x: Hidden features of shape ``(..., dim)``.
            block_ids: Blocks to use for this route.
            coefficients: Mapping from block_id to middle matrix ``M`` of shape
                ``(rank, rank)``.

        Returns:
            Residual features of shape ``(..., dim)``.
        """
        if not block_ids:
            return torch.zeros_like(x)

        # Group blocks by rank so we can vectorize each group with einsum.
        by_rank: dict[int, list[tuple[str, BasisBlock]]] = {}
        for bid in block_ids:
            block = self.blocks[bid]
            by_rank.setdefault(block.rank, []).append((bid, block))

        out = torch.zeros_like(x)
        for _, items in by_rank.items():
            bids, blocks = zip(*items)
            u_in = torch.stack([b.u_in for b in blocks], dim=0)  # (G, dim, rank)
            u_out = torch.stack([b.u_out for b in blocks], dim=0)  # (G, dim, rank)
            m = torch.stack([coefficients[bid] for bid in bids], dim=0)  # (G, rank, rank)
            h = torch.einsum("...d,gdr->...gr", x, u_in)
            h = torch.einsum("...gr,grr->...gr", h, m)
            out = out + torch.einsum("...gr,gdr->...d", h, u_out)
        return out


class FastAdapter(nn.Module):
    """Temporary LoRA used only during the fast learning stage."""

    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.a = nn.Parameter(torch.empty(dim, rank))
        self.b = nn.Parameter(torch.empty(rank, dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.a, a=5 ** (1.0 / 3))
        nn.init.zeros_(self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.a @ self.b

    def full_update(self) -> torch.Tensor:
        """Return the full-rank update matrix ``A B^T``."""
        return self.a @ self.b


class ParentGate(nn.Module):
    """Learnable scalar weights over parent endpoint predictions."""

    def __init__(self, parent_ids: Sequence[str]) -> None:
        super().__init__()
        self.parent_ids = list(parent_ids)
        self.logits = nn.Parameter(torch.zeros(len(parent_ids)))

    def forward(self, parent_preds: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        weights = {pid: w for pid, w in zip(self.parent_ids, F.softmax(self.logits, dim=0))}
        return weights


class RouteSpec(nn.Module):
    """All parameters and metadata for one published endpoint."""

    def __init__(
        self,
        endpoint_id: str,
        parent_ids: Sequence[str],
        basis_block_ids: Sequence[str],
        dim: int,
        normalizer: tuple[float, float],
    ) -> None:
        super().__init__()
        self.endpoint_id = endpoint_id
        self.parent_ids = list(parent_ids)
        self.basis_block_ids = list(basis_block_ids)
        self.dim = dim
        self.normalizer = normalizer

        self.parent_gate = ParentGate(parent_ids) if parent_ids else None
        self.private_coefficients: nn.ParameterDict = nn.ParameterDict()
        self.head = nn.Linear(dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for p in self.private_coefficients.parameters():
            nn.init.orthogonal_(p)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def set_coefficient_shape(self, block_id: str, rank: int) -> None:
        """Allocate (or replace) the private middle matrix with the correct rank."""
        if block_id in self.private_coefficients:
            param = self.private_coefficients[block_id]
            if param.shape == (rank, rank):
                return
            # Replace with correctly shaped parameter.
            del self.private_coefficients[block_id]
        self.private_coefficients[block_id] = nn.Parameter(torch.empty(rank, rank))
        nn.init.orthogonal_(self.private_coefficients[block_id])


class EndpointRegistry(nn.Module):
    """Track published endpoints and their dependency manifests."""

    def __init__(self) -> None:
        super().__init__()
        self.routes: nn.ModuleDict = nn.ModuleDict()
        self.published_manifests: dict[str, dict[str, Any]] = {}

    def register(self, route: RouteSpec) -> None:
        self.routes[route.endpoint_id] = route

    def publish(self, endpoint_id: str, encoder: nn.Module, graph_builder_hash: str, dataset_hash: str) -> None:
        route = self.routes[endpoint_id]
        block_hashes = {bid: route.private_coefficients[bid].detach().cpu().numpy().tobytes().hex()[:16] for bid in route.basis_block_ids}
        manifest = {
            "endpoint_id": endpoint_id,
            "parent_endpoint_ids": route.parent_ids,
            "basis_block_ids": route.basis_block_ids,
            "block_hashes": block_hashes,
            "head_hash": hashlib.sha256(
                torch.cat([p.detach().cpu().flatten() for p in route.head.parameters()]).numpy().tobytes()
            ).hexdigest()[:16],
            "normalizer": route.normalizer,
            "graph_builder_hash": graph_builder_hash,
            "dataset_hash": dataset_hash,
        }
        self.published_manifests[endpoint_id] = manifest

    def assert_all_published_unchanged(self) -> None:
        for endpoint_id, manifest in self.published_manifests.items():
            route = self.routes[endpoint_id]
            current_block_hashes = {
                bid: route.private_coefficients[bid].detach().cpu().numpy().tobytes().hex()[:16]
                for bid in route.basis_block_ids
            }
            if current_block_hashes != manifest["block_hashes"]:
                raise RuntimeError(f"Published endpoint {endpoint_id} private coefficients changed")
            current_head = hashlib.sha256(
                torch.cat([p.detach().cpu().flatten() for p in route.head.parameters()]).numpy().tobytes()
            ).hexdigest()[:16]
            if current_head != manifest["head_hash"]:
                raise RuntimeError(f"Published endpoint {endpoint_id} head changed")


class PersistentConsolidationGraph(nn.Module):
    """PCG with frozen encoder, append-only basis bank, and immutable endpoints."""

    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int,
        rank: int = 8,
        novelty_threshold: float = 0.2,
        svd_energy: float = 0.9,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.novelty_threshold = novelty_threshold
        self.svd_energy = svd_energy

        self.basis_bank = BasisBank(hidden_dim, default_rank=rank)
        self.registry = EndpointRegistry()
        self._route_order: list[str] = []

    def _endpoint_key(self, version: str, prop_id: int, fid_id: int) -> str:
        return f"v{version}_p{int(prop_id)}_f{int(fid_id)}"

    def add_route(
        self,
        version: str,
        prop_id: int,
        fid_id: int,
        parent_ids: Sequence[str] | None = None,
        normalizer: tuple[float, float] = (0.0, 1.0),
    ) -> str:
        """Allocate a new endpoint route."""
        key = self._endpoint_key(version, prop_id, fid_id)
        if key in self.registry.routes:
            return key

        # The first route needs at least one basis block to start from.
        if not self.basis_bank.blocks:
            self.basis_bank.add_block(created_by_version=version)

        # Newly added blocks may be on CPU even if the model was moved to GPU.
        ref = next(self.encoder.parameters())
        self.basis_bank.to(device=ref.device, dtype=ref.dtype)

        route = RouteSpec(
            endpoint_id=key,
            parent_ids=parent_ids or [],
            basis_block_ids=list(self.basis_bank.blocks.keys()),
            dim=self.hidden_dim,
            normalizer=normalizer,
        )
        # Move to the encoder's device/dtype so forward works before training.
        ref = next(self.encoder.parameters())
        route = route.to(device=ref.device, dtype=ref.dtype)

        # Initialize coefficient shapes for existing blocks.
        for bid in route.basis_block_ids:
            rank = self.basis_bank.blocks[bid].rank
            route.set_coefficient_shape(bid, rank)

        self.registry.register(route)
        self._route_order.append(key)
        return key

    def learn_endpoint(
        self,
        version: str,
        prop_id: int,
        fid_id: int,
        train_loader: Any,
        val_loader: Any,
        device: torch.device,
        epochs_fast: int = 10,
        epochs_cons: int = 15,
        lr: float = 1e-3,
        beta: float = 1.0,
        gamma: float = 1e-3,
    ) -> dict[str, Any]:
        """Two-stage endpoint learning: fast adapter then consolidation.

        Loaders are expected to return *physical* targets; normalization to the
        endpoint's own coordinate system happens inside the model.
        """
        key = self._endpoint_key(version, prop_id, fid_id)
        route = self.registry.routes[key]
        mean_e, std_e = route.normalizer

        # Fast stage: train temporary adapter + the route head on the residual
        # target after accounting for parent predictions.
        fast = FastAdapter(self.hidden_dim, self.rank).to(device)
        route.to(device)
        trainable_fast = list(fast.parameters()) + list(route.head.parameters())
        if route.parent_gate is not None:
            trainable_fast += list(route.parent_gate.parameters())
        # Pre-compute encoder features once; the backbone is frozen.
        train_features = self._precompute_features(train_loader, device)
        val_features = self._precompute_features(val_loader, device)

        # Fast stage: train temporary adapter + the route head on the residual
        # target after accounting for parent predictions.
        fast = FastAdapter(self.hidden_dim, self.rank).to(device)
        route.to(device)
        trainable_fast = list(fast.parameters()) + list(route.head.parameters())
        if route.parent_gate is not None:
            trainable_fast += list(route.parent_gate.parameters())
        optimizer_fast = torch.optim.AdamW(trainable_fast, lr=lr, weight_decay=1e-4)
        for _ in range(epochs_fast):
            fast.train()
            route.head.train()
            if route.parent_gate is not None:
                route.parent_gate.train()
            for h, original_mask, y_phys in train_features:
                optimizer_fast.zero_grad()
                pred_phys = self._forward_with_adapter_h(h, original_mask, key, fast, physical=True)
                loss = F.mse_loss(pred_phys, y_phys)
                loss.backward()
                optimizer_fast.step()

        # Novelty gate uses the full fast update matrix.
        fast_update = fast.full_update().detach()
        selected_ids, new_ids = self.basis_bank.reuse_or_expand(
            fast_update,
            novelty_threshold=self.novelty_threshold,
            svd_energy=self.svd_energy,
            created_by_version=version,
        )

        # Update route to use selected + new blocks.
        route.basis_block_ids = selected_ids + new_ids
        for bid in route.basis_block_ids:
            rank = self.basis_bank.blocks[bid].rank
            route.set_coefficient_shape(bid, rank)
        # Ensure new blocks live on the same device as the encoder.
        self.basis_bank.to(device=device, dtype=next(self.encoder.parameters()).dtype)
        route.to(device)

        # Consolidation stage.
        trainable = [p for p in route.parameters() if p.requires_grad]
        for block in self.basis_bank.blocks.values():
            if block.plastic:
                trainable.extend(block.parameters())
        optimizer_cons = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)

        best_val_loss = float("inf")
        best_state = None
        for _ in range(epochs_cons):
            route.train()
            self.basis_bank.train()
            for h, original_mask, y_phys in train_features:
                optimizer_cons.zero_grad()

                pred_fast = self._forward_with_adapter_h(h, original_mask, key, fast, physical=True).detach()
                pred_persistent = self._forward_route_h(h, original_mask, key, physical=True)

                loss_task = F.mse_loss(pred_persistent, y_phys)
                loss_distill = F.mse_loss(pred_persistent, pred_fast)
                loss_orth = self.basis_bank.orthogonality_loss(new_ids) if new_ids else torch.tensor(0.0, device=device)
                loss = loss_task + beta * loss_distill + gamma * loss_orth
                loss.backward()
                optimizer_cons.step()

            val_loss = self._eval_loss(val_features, key, physical=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}

        if best_state is not None:
            self.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        return {
            "endpoint_id": key,
            "selected_basis_blocks": selected_ids,
            "new_basis_blocks": new_ids,
            "best_val_loss": best_val_loss,
        }

    def publish_route(self, version: str, prop_id: int, fid_id: int) -> None:
        key = self._endpoint_key(version, prop_id, fid_id)
        route = self.registry.routes[key]
        self.basis_bank.freeze_blocks(route.basis_block_ids)
        for p in route.parameters():
            p.requires_grad = False
        self.registry.publish(key, self.encoder, "graph_builder_v1", "dataset_v1")

    def _encode(self, node_feats: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return node features from the frozen encoder."""
        h, _ = self.encoder.encode(node_feats, coords, mask)
        return h

    def _pool(self, h: torch.Tensor, original_mask: torch.Tensor) -> torch.Tensor:
        mask_exp = original_mask.unsqueeze(-1).float()
        return (h * mask_exp).sum(dim=1) / (mask_exp.sum(dim=1).clamp_min(1.0))

    @torch.no_grad()
    def _precompute_features(
        self, loader: Any, device: torch.device
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Cache encoder outputs for a loader to avoid repeated backbone passes."""
        self.encoder.eval()
        features: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for batch in loader:
            node_feats, coords, mask, original_mask, y_phys = [b.to(device) for b in batch]
            h = self._encode(node_feats, coords, mask)
            features.append((h.detach(), original_mask.to(device), y_phys.to(device)))
        return features

    def _parent_predictions_h(
        self,
        h: torch.Tensor,
        original_mask: torch.Tensor,
        endpoint_id: str,
        physical: bool = True,
    ) -> torch.Tensor:
        """Return the weighted parent prediction for ``endpoint_id``.

        If ``physical`` is True the returned tensor is in physical units; otherwise
        it is normalized in the parent endpoint's coordinate system.
        """
        route = self.registry.routes[endpoint_id]
        if not route.parent_ids:
            return torch.zeros(h.size(0), device=h.device, dtype=h.dtype)

        parent_preds: dict[str, torch.Tensor] = {}
        for pid in route.parent_ids:
            parent_preds[pid] = self._forward_route_h(
                h, original_mask, pid, physical=True
            )

        weights = route.parent_gate(parent_preds) if route.parent_gate is not None else {pid: 1.0 / len(route.parent_ids) for pid in route.parent_ids}
        weighted = sum(weights[pid] * parent_preds[pid] for pid in route.parent_ids)
        return weighted

    def _forward_route_h(
        self,
        h: torch.Tensor,
        original_mask: torch.Tensor,
        endpoint_id: str,
        physical: bool = False,
    ) -> torch.Tensor:
        """Route forward given pre-computed encoder features ``h``."""
        route = self.registry.routes[endpoint_id]
        coeffs = {bid: route.private_coefficients[bid] for bid in route.basis_block_ids}
        h = h + self.basis_bank(h, route.basis_block_ids, coeffs)
        pooled = self._pool(h, original_mask)
        residual_phys = route.head(pooled).squeeze(-1)

        parent_pred_phys = self._parent_predictions_h(h, original_mask, endpoint_id, physical=True)
        pred_phys = parent_pred_phys + residual_phys

        if physical:
            return pred_phys
        mean, std = route.normalizer
        return (pred_phys - mean) / std

    def _forward_with_adapter_h(
        self,
        h: torch.Tensor,
        original_mask: torch.Tensor,
        endpoint_id: str,
        adapter: FastAdapter,
        physical: bool = False,
    ) -> torch.Tensor:
        """Fast-adapter forward given pre-computed encoder features ``h``."""
        route = self.registry.routes[endpoint_id]
        h = h + adapter(h)
        pooled = self._pool(h, original_mask)
        residual_phys = route.head(pooled).squeeze(-1)

        parent_pred_phys = self._parent_predictions_h(h, original_mask, endpoint_id, physical=True)
        pred_phys = parent_pred_phys + residual_phys

        if physical:
            return pred_phys
        mean, std = route.normalizer
        return (pred_phys - mean) / std

    def _parent_predictions(
        self,
        node_feats: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        original_mask: torch.Tensor,
        endpoint_id: str,
        physical: bool = True,
    ) -> torch.Tensor:
        """Wrapper that encodes inputs before computing parent predictions."""
        h = self._encode(node_feats, coords, mask)
        return self._parent_predictions_h(h, original_mask, endpoint_id, physical=physical)

    def _forward_route(
        self,
        node_feats: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        original_mask: torch.Tensor,
        endpoint_id: str,
        physical: bool = False,
    ) -> torch.Tensor:
        """Wrapper that encodes inputs before the route forward."""
        h = self._encode(node_feats, coords, mask)
        return self._forward_route_h(h, original_mask, endpoint_id, physical=physical)

    def _forward_with_adapter(
        self,
        node_feats: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        original_mask: torch.Tensor,
        endpoint_id: str,
        adapter: FastAdapter,
        physical: bool = False,
    ) -> torch.Tensor:
        """Wrapper that encodes inputs before the fast-adapter forward."""
        h = self._encode(node_feats, coords, mask)
        return self._forward_with_adapter_h(h, original_mask, endpoint_id, adapter, physical=physical)

    def _eval_loss(
        self,
        features: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        endpoint_id: str,
        physical: bool = True,
    ) -> float:
        self.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for h, original_mask, y in features:
                pred = self._forward_route_h(h, original_mask, endpoint_id, physical=physical)
                total_loss += F.mse_loss(pred, y, reduction="sum").item()
                n += y.numel()
        return total_loss / max(n, 1)

    def forward(
        self,
        node_feats: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        original_mask: torch.Tensor,
        version: str,
        prop_id: int,
        fid_id: int,
        physical: bool = False,
    ) -> torch.Tensor:
        key = self._endpoint_key(version, prop_id, fid_id)
        if key not in self.registry.routes:
            raise KeyError(f"Endpoint {key} not allocated")
        return self._forward_route(node_feats, coords, mask, original_mask, key, physical=physical)

    def current_trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def incremental_parameters(self, version: str, prop_id: int, fid_id: int) -> int:
        key = self._endpoint_key(version, prop_id, fid_id)
        route = self.registry.routes[key]
        total = sum(route.private_coefficients[bid].numel() for bid in route.basis_block_ids)
        total += sum(p.numel() for p in route.head.parameters())
        return total
