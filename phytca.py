"""PhyTCA: Physics-Structured Tensor Component Adaptation for evolving
materials databases.

Core components:
- Tucker4DAdapter: multi-axis Tucker adapter over
  (output_channel, input_channel, property, fidelity).
- Periodic crystal graph encoder with inserted Tucker adapters.
- Continual learning utilities (stability loss, freezing, expansion).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from egnn_pytorch import EGNN


class Tucker4DAdapter(nn.Module):
    """Adapter via Tucker decomposition of a 4D weight-update tensor.

    The full adapter tensor is
        A = G x_1 U_out x_2 U_in x_3 E_prop x_4 E_fid
    with shapes
        G: (R_out, R_in, R_p, R_f)
        U_out: (d_out, R_out)
        U_in:  (d_in,  R_in)
        E_prop: (N_p, R_p)
        E_fid:  (N_f, R_f)

    For a specific (property p, fidelity f) the adapter is
        Delta W_{p,f} = U_out @ (G x_3 e_p x_4 e_f) @ U_in^T,
    where e_p, e_f are the p-th/f-th rows of the embedding matrices.
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

        self.U_out = nn.Parameter(torch.randn(d_out, rank_out) * 0.1)
        self.U_in = nn.Parameter(torch.randn(d_in, rank_in) * 0.1)
        self.E_prop = nn.Embedding(n_properties, rank_prop)
        self.E_fid = nn.Embedding(n_fidelities, rank_fid)
        self.G = nn.Parameter(
            torch.randn(rank_out, rank_in, rank_prop, rank_fid) * 0.1
        )

        # Frozen (property, fidelity) slices.
        self.frozen_slices: Set[Tuple[int, int]] = set()

    def forward(self, x: torch.Tensor, prop_id: int, fid_id: int) -> torch.Tensor:
        """Apply adapter Delta W_{p,f} to input x.

        Args:
            x: (B, d_in) or (B, N, d_in).
            prop_id: property index.
            fid_id: fidelity index.
        Returns:
            y = x @ Delta W^T, same leading dims as x, last dim d_out.
        """
        e_p = self.E_prop(torch.tensor(prop_id, device=self.G.device))   # (R_p,)
        e_f = self.E_fid(torch.tensor(fid_id, device=self.G.device))     # (R_f,)
        # Contract property/fidelity axes of the core.
        core_slice = torch.einsum("oipf,p,f->oi", self.G, e_p, e_f)       # (R_out, R_in)
        delta_w = self.U_out @ core_slice @ self.U_in.t()                 # (d_out, d_in)

        if x.dim() == 2:
            return x @ delta_w.t()
        if x.dim() == 3:
            return torch.einsum("bni,io->bno", x, delta_w.t())
        raise ValueError(f"Unsupported input dim {x.dim()}")

    def freeze_slice(self, prop_id: int, fid_id: int) -> None:
        self.frozen_slices.add((int(prop_id), int(fid_id)))

    def expand_property_axis(self, new_n_properties: int) -> None:
        """Add new property rows to the embedding matrix."""
        if new_n_properties <= self.n_properties:
            return
        old = self.n_properties
        old_weight = self.E_prop.weight.data
        new_weight = torch.cat(
            [
                old_weight,
                torch.randn(new_n_properties - old, self.rank_prop, device=self.G.device) * 0.1,
            ],
            dim=0,
        )
        self.E_prop = nn.Embedding(new_n_properties, self.rank_prop)
        self.E_prop.weight.data.copy_(new_weight)
        self.n_properties = new_n_properties

    def expand_fidelity_axis(self, new_n_fidelities: int) -> None:
        """Add new fidelity rows to the embedding matrix."""
        if new_n_fidelities <= self.n_fidelities:
            return
        old = self.n_fidelities
        old_weight = self.E_fid.weight.data
        new_weight = torch.cat(
            [
                old_weight,
                torch.randn(new_n_fidelities - old, self.rank_fid, device=self.G.device) * 0.1,
            ],
            dim=0,
        )
        self.E_fid = nn.Embedding(new_n_fidelities, self.rank_fid)
        self.E_fid.weight.data.copy_(new_weight)
        self.n_fidelities = new_n_fidelities

    def zero_frozen_gradients(self) -> None:
        """Zero out gradients of frozen (property, fidelity) core slices."""
        if not self.frozen_slices or self.G.grad is None:
            return
        for p, f in self.frozen_slices:
            self.G.grad[:, :, p, f].zero_()
        # Also freeze corresponding embedding rows.
        if self.E_prop.weight.grad is not None:
            for p, _ in self.frozen_slices:
                self.E_prop.weight.grad[p].zero_()
        if self.E_fid.weight.grad is not None:
            for _, f in self.frozen_slices:
                self.E_fid.weight.grad[f].zero_()

    def count_parameters(self) -> int:
        return (
            self.U_out.numel()
            + self.U_in.numel()
            + self.E_prop.weight.numel()
            + self.E_fid.weight.numel()
            + self.G.numel()
        )


class AdapterCrystalGraphLayer(nn.Module):
    """Periodic crystal graph message-passing layer with a residual Tucker adapter on node features."""

    def __init__(
        self,
        dim: int,
        n_properties: int,
        n_fidelities: int,
        rank: int = 8,
        num_nearest_neighbors: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = EGNN(
            dim=dim,
            edge_dim=0,
            m_dim=max(16, dim),
            num_nearest_neighbors=num_nearest_neighbors,
            update_coors=True,
            update_feats=True,
        )
        self.adapter = Tucker4DAdapter(
            dim,
            dim,
            n_properties,
            n_fidelities,
            rank_out=rank,
            rank_in=rank,
            rank_prop=max(2, n_properties),
            rank_fid=max(2, n_fidelities),
        )

    def forward(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        prop_id: int,
        fid_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        new_feats, new_coords = self.encoder(feats, coords, mask=mask)
        delta = self.adapter(new_feats, prop_id, fid_id)
        return new_feats + delta, new_coords


class PhyTCAModel(nn.Module):
    """Full crystal property predictor with multi-axis Tucker adapters."""

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        n_properties: int,
        n_fidelities: int,
        n_layers: int = 3,
        adapter_rank: int = 8,
        num_nearest_neighbors: int = 8,
        freeze_encoder_weights: bool = True,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.n_properties = n_properties
        self.n_fidelities = n_fidelities
        self.n_layers = n_layers
        self.freeze_encoder_weights = freeze_encoder_weights

        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                AdapterCrystalGraphLayer(
                    hidden_dim,
                    n_properties,
                    n_fidelities,
                    rank=adapter_rank,
                    num_nearest_neighbors=num_nearest_neighbors,
                )
                for _ in range(n_layers)
            ]
        )

        # One prediction head per (property, fidelity).
        self.heads = nn.ModuleDict()
        for p in range(n_properties):
            for f in range(n_fidelities):
                self.heads[f"p{p}_f{f}"] = nn.Linear(hidden_dim, 1)

        if freeze_encoder_weights:
            for p in self.encoder_parameters():
                p.requires_grad = False

    def encoder_parameters(self) -> List[nn.Parameter]:
        """Parameters of the raw crystal graph encoder layers (no adapters, no heads)."""
        params = list(self.node_embed.parameters())
        for layer in self.layers:
            params.extend(layer.encoder.parameters())
        return params

    def encode(
        self,
        node_feats: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        original_mask: torch.Tensor,
        prop_id: int,
        fid_id: int,
    ) -> torch.Tensor:
        """Return the pooled crystal-level representation for a given task."""
        h = self.node_embed(node_feats)
        for layer in self.layers:
            h, coords = layer(h, coords, mask, prop_id, fid_id)

        # Masked mean pooling over original atoms only.
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
        """Forward pass for a specific (property, fidelity) task.

        Args:
            node_feats: (B, N_total, node_dim) including periodic images.
            coords:     (B, N_total, 3)
            mask:       (B, N_total) padding mask (True for real atoms/images).
            original_mask: (B, N_total) True only for atoms in the original
                unit cell; used for final pooling.
            prop_id: property index.
            fid_id: fidelity index.
        Returns:
            predictions: (B,)
        """
        pooled = self.encode(node_feats, coords, mask, original_mask, prop_id, fid_id)
        key = f"p{prop_id}_f{fid_id}"
        return self.heads[key](pooled).squeeze(-1)

    def freeze_task(self, prop_id: int, fid_id: int) -> None:
        """Freeze adapter slices and head for a completed task."""
        for layer in self.layers:
            layer.adapter.freeze_slice(prop_id, fid_id)
        key = f"p{prop_id}_f{fid_id}"
        for p in self.heads[key].parameters():
            p.requires_grad = False

    def anchor_state(self) -> Dict[str, torch.Tensor]:
        """Snapshot of currently trainable adapter parameters."""
        state = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                state[name] = p.data.clone()
        return state

    def stability_loss(self, mu: float, anchor: Dict[str, torch.Tensor]) -> torch.Tensor:
        """L2 penalty on trainable parameters to anchor."""
        if mu <= 0.0 or not anchor:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for name, p in self.named_parameters():
            if p.requires_grad and name in anchor:
                loss += (p - anchor[name].to(p.device)).pow(2).sum()
        return mu * loss

    def count_adapter_parameters(self) -> int:
        """Count parameters in adapters + heads (excluding crystal graph encoder)."""
        total = sum(p.numel() for p in self.node_embed.parameters())
        for layer in self.layers:
            total += layer.adapter.count_parameters()
        for head in self.heads.values():
            total += sum(p.numel() for p in head.parameters())
        return total

    def count_total_parameters(self) -> int:
        """Count all model parameters."""
        return sum(p.numel() for p in self.parameters())

    def count_encoder_parameters(self) -> int:
        """Count parameters in the crystal graph encoder (including node embed)."""
        return sum(p.numel() for p in self.encoder_parameters())

    def count_head_parameters(self) -> int:
        """Count parameters in all prediction heads."""
        return sum(sum(p.numel() for p in head.parameters()) for head in self.heads.values())

    def get_parameter_group_counts(self) -> Dict[str, int]:
        """Return detailed parameter counts by role."""
        encoder = self.count_encoder_parameters()
        adapter = self.count_adapter_parameters() - sum(
            sum(p.numel() for p in head.parameters()) for head in self.heads.values()
        )
        heads = self.count_head_parameters()
        return {
            "total": self.count_total_parameters(),
            "encoder": encoder,
            "adapter": adapter,
            "heads": heads,
        }


def normalized_mae(pred: torch.Tensor, target: torch.Tensor, mad: float) -> torch.Tensor:
    """Normalized MAE by mean absolute deviation."""
    return torch.abs(pred - target).mean() / max(mad, 1e-8)


def compute_mad(targets: torch.Tensor) -> float:
    """Mean absolute deviation of a target tensor."""
    return float(torch.abs(targets - targets.mean()).mean())


def forgetting(nmaes: List[List[float]]) -> float:
    """Average per-task forgetting across a continual run.

    nmaes[t][i] is the nMAE on task i after training task t.
    """
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
    """Average backward transfer for an error metric (lower is better).

    Positive values indicate improvement on old tasks after learning new ones;
    negative values indicate forgetting. For two tasks this equals
    -absolute_forgetting.
    """
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
