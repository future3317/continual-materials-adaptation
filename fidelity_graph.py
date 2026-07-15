"""Fidelity graph utilities for continually evolving multi-fidelity models.

This module provides the building blocks for the ICLR upgrade route described
in `反馈_2.md` section 4:

* ``FidelityGraph`` — a DAG whose nodes are (property, fidelity) pairs and whose
  edges represent learnable residual corrections.
* ``ParentSelector`` — scores candidate parents by validation residual error,
  incremental parameter cost, and compute cost.
* ``AdaptiveRankAllocator`` — chooses the smallest rank whose residual operator
  tail energy is below a threshold.
* ``path_consistency_loss`` — aligns predictions from multiple paths through the
  fidelity DAG.

These components are intentionally decoupled from ``ContinualCrystalModel`` so
that they can be plugged in as the project scales to three+ fidelities and
multi-property experiments.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FidelityGraph:
    """Directed acyclic graph of fidelity nodes and residual edges.

    A node is identified by ``(prop_id, fid_id)``.  An edge ``u -> v`` means
    that fidelity ``v`` can be predicted from fidelity ``u`` via a residual
    correction.  Multiple incoming edges are allowed; the final prediction can
    be a weighted combination of parent predictions.

    The graph is kept as an explicit adjacency list so that exact-retention
    bookkeeping (which edges are frozen) is transparent.
    """

    def __init__(self) -> None:
        self.nodes: Set[Tuple[int, int]] = set()
        self.edges: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self.frozen_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()

    def add_node(self, prop_id: int, fid_id: int) -> None:
        self.nodes.add((int(prop_id), int(fid_id)))

    def add_edge(
        self,
        parent_prop: int,
        parent_fid: int,
        child_prop: int,
        child_fid: int,
    ) -> None:
        parent = (int(parent_prop), int(parent_fid))
        child = (int(child_prop), int(child_fid))
        self.add_node(*parent)
        self.add_node(*child)
        self.edges.setdefault(child, []).append(parent)

    def parents(self, prop_id: int, fid_id: int) -> List[Tuple[int, int]]:
        return self.edges.get((int(prop_id), int(fid_id)), [])

    def freeze_edge(
        self,
        parent_prop: int,
        parent_fid: int,
        child_prop: int,
        child_fid: int,
    ) -> None:
        self.frozen_edges.add(
            ((int(parent_prop), int(parent_fid)), (int(child_prop), int(child_fid)))
        )

    def is_frozen(
        self,
        parent_prop: int,
        parent_fid: int,
        child_prop: int,
        child_fid: int,
    ) -> bool:
        return (
            (int(parent_prop), int(parent_fid)),
            (int(child_prop), int(child_fid)),
        ) in self.frozen_edges

    def topological_order(self) -> List[Tuple[int, int]]:
        """Return a topological ordering of the fidelity DAG."""
        in_degree = {node: 0 for node in self.nodes}
        for child, parents in self.edges.items():
            in_degree[child] = len(parents)

        order: List[Tuple[int, int]] = []
        queue = [n for n, d in in_degree.items() if d == 0]
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child, parents in self.edges.items():
                if node in parents:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

        if len(order) != len(self.nodes):
            raise ValueError("Fidelity graph contains a cycle")
        return order


class ParentSelector:
    """Score candidate parents for a new fidelity node.

    Score (lower is better):
        S(u -> v) = val_residual_error + lambda_P * P_uv + lambda_C * C_u
    where ``P_uv`` is the incremental parameter cost of the edge and ``C_u`` is
    the compute/data cost of acquiring fidelity ``u``.
    """

    def __init__(
        self,
        lambda_param: float = 1e-4,
        lambda_cost: float = 1e-6,
    ) -> None:
        self.lambda_param = lambda_param
        self.lambda_cost = lambda_cost

    def score(
        self,
        val_residual_error: float,
        incremental_params: int,
        parent_compute_cost: float = 0.0,
    ) -> float:
        return (
            val_residual_error
            + self.lambda_param * incremental_params
            + self.lambda_cost * parent_compute_cost
        )

    def select(
        self,
        candidates: Sequence[Tuple[int, int]],
        val_errors: Sequence[float],
        param_counts: Sequence[int],
        compute_costs: Optional[Sequence[float]] = None,
    ) -> Tuple[int, int]:
        """Return the candidate parent with the lowest score."""
        if compute_costs is None:
            compute_costs = [0.0] * len(candidates)
        best_idx = min(
            range(len(candidates)),
            key=lambda i: self.score(val_errors[i], param_counts[i], compute_costs[i]),
        )
        return candidates[best_idx]


class AdaptiveRankAllocator:
    """Choose the smallest rank whose residual-operator tail energy is below a threshold.

    Given a target residual matrix ``R`` (e.g. the least-squares residual operator
    on a frozen representation), compute its singular values ``sigma`` and select
        r* = min { r : sum_{i > r} sigma_i^2 / sum_i sigma_i^2 <= epsilon }.

    This addresses `反馈_2.md` section 4.2 (adaptive rank).
    """

    def __init__(self, epsilon: float = 0.05, max_rank: int = 64) -> None:
        if not 0 < epsilon < 1:
            raise ValueError("epsilon must be in (0, 1)")
        self.epsilon = epsilon
        self.max_rank = max_rank

    def allocate(self, residual_matrix: torch.Tensor) -> int:
        """Return recommended rank for the residual operator.

        Args:
            residual_matrix: shape (N, d) where rows are samples and columns are
                hidden features of the residual mapping.

        Returns:
            Recommended rank (at least 1).
        """
        if residual_matrix.numel() == 0:
            return 1
        # Use SVD on the residual operator.
        _, s, _ = torch.svd(residual_matrix)
        total_energy = (s ** 2).sum()
        if total_energy <= 0:
            return 1

        cumulative_tail = torch.cumsum(s ** 2, dim=0)
        # cumulative_tail[r] = sum_{i <= r} sigma_i^2
        tail_energy = total_energy - cumulative_tail
        # tail_energy[r] = sum_{i > r} sigma_i^2
        within_budget = tail_energy / total_energy <= self.epsilon
        ranks = torch.where(within_budget)[0]
        if ranks.numel() == 0:
            return min(self.max_rank, len(s))
        best_rank = int(ranks[0].item()) + 1  # ranks are 0-indexed
        return max(1, min(best_rank, self.max_rank, len(s)))


def path_consistency_loss(
    predictions: Dict[Tuple[Tuple[int, int], ...], torch.Tensor],
) -> torch.Tensor:
    """Align predictions that reach the same fidelity through different paths.

    Args:
        predictions: mapping from path (tuple of (prop_id, fid_id) nodes) to
            predicted values.  All paths in the dict must end at the same target
            fidelity.

    Returns:
        MSE between all pairs of path predictions.
    """
    if len(predictions) < 2:
        return torch.tensor(0.0)

    vals = list(predictions.values())
    loss = torch.tensor(0.0, device=vals[0].device)
    count = 0
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            loss = loss + F.mse_loss(vals[i], vals[j])
            count += 1
    return loss / max(count, 1)


class FidelityGraphPredictor(nn.Module):
    """Skeleton predictor that routes through a fidelity graph.

    This is a placeholder integration point.  It assumes each edge has an
    associated residual module and combines parent predictions as a weighted sum.
    In a full implementation, the residual modules would be stored in
    ``ContinualCrystalModel.adapter_banks`` and the weights would be learned or
    set by ``ParentSelector``.
    """

    def __init__(
        self,
        graph: FidelityGraph,
        edge_modules: Dict[
            Tuple[Tuple[int, int], Tuple[int, int]], nn.Module
        ],
    ) -> None:
        super().__init__()
        self.graph = graph
        self.edge_modules = nn.ModuleDict()
        for (u, v), module in edge_modules.items():
            key = f"p{u[0]}_f{u[1]}__p{v[0]}_f{v[1]}"
            self.edge_modules[key] = module

    def forward(
        self,
        parent_predictions: Dict[Tuple[int, int], torch.Tensor],
        target: Tuple[int, int],
    ) -> torch.Tensor:
        """Combine parent predictions to predict ``target``.

        Args:
            parent_predictions: mapping from parent node to its prediction.
            target: (prop_id, fid_id) to predict.

        Returns:
            Combined prediction.
        """
        parents = self.graph.parents(*target)
        if not parents:
            raise ValueError(f"Target {target} has no parents in the fidelity graph")

        preds = []
        for p in parents:
            key = f"p{p[0]}_f{p[1]}__p{target[0]}_f{target[1]}"
            residual = self.edge_modules[key](parent_predictions[p])
            preds.append(parent_predictions[p] + residual)

        # Uniform combination; could be learned weights.
        return torch.stack(preds, dim=0).mean(dim=0)
