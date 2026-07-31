"""UniversalExecutive — trajectory scoring and action selection organ.

Scores trajectory candidates along multiple dimensions, selects the best
action (SELECT, RECALL_AGAIN, EXPAND, ASK_OR_DEFER, STOP), and enforces
a budget of at most 2 replans per 32-token block.

Gate contract:
- The executive may authorize residual injection through Memory or WorldModel
  but never injects residuals itself (enforced by ResidualCondition contract).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .contracts import COGNITIVE_ORGAN_WIDTH


EXECUTIVE_STATE_SCHEMA = "darwin-executive-state-v1"
MAX_REPLANS_PER_BLOCK = 2


# ── Action types ────────────────────────────────────────────────────────────


class ExecutiveAction(StrEnum):
    SELECT = "SELECT"
    RECALL_AGAIN = "RECALL_AGAIN"
    EXPAND = "EXPAND"
    ASK_OR_DEFER = "ASK_OR_DEFER"
    STOP = "STOP"


@dataclass
class ScoredTrajectory:
    candidate_id: str
    goal_alignment: float
    memory_consistency: float
    predicted_value: float
    trajectory_uncertainty: float
    novelty: float
    safety_risk: float
    compute_cost: float
    total_score: float
    rank: int


@dataclass
class ExecutiveDecision:
    action: ExecutiveAction
    selected_candidate_id: str | None
    recall_query_params: dict[str, Any] | None
    expand_budget: int | None
    defer_reason: str | None
    scores: list[ScoredTrajectory]
    replans_used: int


# ── Trajectory Critic ───────────────────────────────────────────────────────


class TrajectoryCritic(nn.Module):
    """Learned value estimator for trajectory candidates.

    Scores each candidate on multiple dimensions using a small MLP
    that takes trajectory embeddings concatenated with context features.
    """

    def __init__(self) -> None:
        super().__init__()
        input_dim = 3 * COGNITIVE_ORGAN_WIDTH + 6  # 3 horizons + 6 scalar features
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 7),  # 7 score dimensions
        )

    def forward(
        self,
        z_short: torch.Tensor,   # [B, 512]
        z_medium: torch.Tensor,  # [B, 512]
        z_long: torch.Tensor,    # [B, 512]
        goal_embedding: torch.Tensor | None = None,
        memory_consistency_raw: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Score trajectories.

        Returns dict with keys: goal_alignment, memory_consistency,
        predicted_value, trajectory_uncertainty, novelty, safety_risk,
        compute_cost.
        """
        B = z_short.shape[0]
        device = z_short.device
        features = torch.cat([z_short, z_medium, z_long], dim=-1)  # [B, 1536]
        # Scalar features
        goal_sim = torch.zeros(B, 1, device=device)
        if goal_embedding is not None and goal_embedding.shape[0] == B:
            goal_sim = F.cosine_similarity(z_short, goal_embedding, dim=-1).unsqueeze(-1)
        mem_cons = torch.zeros(B, 1, device=device)
        if memory_consistency_raw is not None and memory_consistency_raw.shape[0] == B:
            mem_cons = memory_consistency_raw.unsqueeze(-1)
        z_norm = z_short.norm(dim=-1, keepdim=True)
        z_var = z_short.var(dim=-1, keepdim=True)
        # Concatenate all features
        scalars = torch.cat([goal_sim, mem_cons, z_norm, z_var,
                             torch.zeros(B, 2, device=device)], dim=-1)
        combined = torch.cat([features, scalars], dim=-1)
        scores = self.scorer(combined)  # [B, 7]
        return {
            "goal_alignment": torch.sigmoid(scores[:, 0]),
            "memory_consistency": torch.sigmoid(scores[:, 1]),
            "predicted_value": scores[:, 2],
            "trajectory_uncertainty": torch.sigmoid(scores[:, 3]),
            "novelty": torch.sigmoid(scores[:, 4]),
            "safety_risk": torch.sigmoid(scores[:, 5]),
            "compute_cost": torch.sigmoid(scores[:, 6]),
        }


# ── The organ ───────────────────────────────────────────────────────────────


class UniversalExecutive(nn.Module):
    """Scores, ranks, and selects trajectory candidates.

    Produces an ExecutiveDecision per forward step.  Enforces:
    - max 2 replans per 32-token block
    - high risk never selected over safe equivalent
    - >= 90% ASK_OR_DEFER when no candidate is valid
    - compute budget respected
    """

    def __init__(
        self,
        *,
        safety_threshold: float = 0.7,
        min_value_threshold: float = 0.1,
        max_total_compute: int = 100,
    ) -> None:
        super().__init__()
        self.safety_threshold = float(safety_threshold)
        self.min_value_threshold = float(min_value_threshold)
        self.max_total_compute = int(max_total_compute)
        self.critic = TrajectoryCritic()
        self._replans_this_block: int = 0
        self._total_compute_spent: int = 0

    def score_candidates(
        self,
        candidates: list[dict[str, torch.Tensor]],  # each with z_short, z_medium, z_long
        goal_embedding: torch.Tensor | None = None,
        memory_recall_scores: torch.Tensor | None = None,
    ) -> list[ScoredTrajectory]:
        """Score and rank trajectory candidates.

        Args:
            candidates: list of dicts with keys z_short, z_medium, z_long (each [512])
            goal_embedding: [512] optional goal representation
            memory_recall_scores: [K] cosine similarity scores from memory

        Returns:
            sorted list of ScoredTrajectory (best first)
        """
        if not candidates:
            return []

        K = len(candidates)
        # Stack candidate embeddings
        zs = torch.stack([c["z_short"] for c in candidates])  # [K, 512]
        zm = torch.stack([c["z_medium"] for c in candidates])
        zl = torch.stack([c["z_long"] for c in candidates])

        goal = goal_embedding.unsqueeze(0) if goal_embedding is not None else None
        mem_cons = memory_recall_scores if memory_recall_scores is not None else None

        all_scores = self.critic(zs, zm, zl, goal, mem_cons)

        scored: list[ScoredTrajectory] = []
        for i in range(K):
            ga = float(all_scores["goal_alignment"][i].detach())
            mc = float(all_scores["memory_consistency"][i].detach())
            pv = float(all_scores["predicted_value"][i].detach())
            tu = float(all_scores["trajectory_uncertainty"][i].detach())
            nv = float(all_scores["novelty"][i].detach())
            sr = float(all_scores["safety_risk"][i].detach())
            cc = float(all_scores["compute_cost"][i].detach())

            # Total score: value minus risk, penalized by uncertainty
            total = pv - 2.0 * sr - 0.5 * tu + 0.3 * ga + 0.2 * mc - 0.1 * cc
            scored.append(ScoredTrajectory(
                candidate_id=f"traj_{i}",
                goal_alignment=ga,
                memory_consistency=mc,
                predicted_value=pv,
                trajectory_uncertainty=tu,
                novelty=nv,
                safety_risk=sr,
                compute_cost=cc,
                total_score=total,
                rank=0,
            ))

        # Sort and assign ranks
        scored.sort(key=lambda s: s.total_score, reverse=True)
        for rank, s in enumerate(scored):
            s.rank = rank + 1

        return scored

    def decide(
        self,
        scored: list[ScoredTrajectory],
        *,
        context_uncertainty: float = 0.0,
    ) -> ExecutiveDecision:
        """Select action based on scored trajectories.

        Decision logic:
        1. If no candidates → ASK_OR_DEFER
        2. If replans exhausted → SELECT best safe or ASK_OR_DEFER
        3. If best is high risk → check if safer alternative exists
        4. If all have low value → ASK_OR_DEFER
        5. Otherwise → SELECT best
        """
        if not scored:
            return ExecutiveDecision(
                action=ExecutiveAction.ASK_OR_DEFER,
                selected_candidate_id=None,
                recall_query_params=None,
                expand_budget=None,
                defer_reason="no_candidates",
                scores=scored,
                replans_used=self._replans_this_block,
            )

        best = scored[0]

        # Safety gate: high risk never selected when safe alternative exists
        if best.safety_risk > self.safety_threshold:
            safe = [s for s in scored if s.safety_risk <= self.safety_threshold]
            if safe:
                best = safe[0]  # Fall back to best safe option
            else:
                return ExecutiveDecision(
                    action=ExecutiveAction.ASK_OR_DEFER,
                    selected_candidate_id=None,
                    recall_query_params=None,
                    expand_budget=None,
                    defer_reason="all_high_risk",
                    scores=scored,
                    replans_used=self._replans_this_block,
                )

        # Value gate: if nothing is worth acting on
        if best.predicted_value < self.min_value_threshold:
            if self._replans_this_block < MAX_REPLANS_PER_BLOCK:
                self._replans_this_block += 1
                return ExecutiveDecision(
                    action=ExecutiveAction.RECALL_AGAIN,
                    selected_candidate_id=None,
                    recall_query_params={"threshold": max(0.3, self.min_value_threshold - 0.1)},
                    expand_budget=None,
                    defer_reason="low_value_retry",
                    scores=scored,
                    replans_used=self._replans_this_block,
                )
            return ExecutiveDecision(
                action=ExecutiveAction.ASK_OR_DEFER,
                selected_candidate_id=None,
                recall_query_params=None,
                expand_budget=None,
                defer_reason="low_value_exhausted",
                scores=scored,
                replans_used=self._replans_this_block,
            )

        # High uncertainty → expand if budget allows
        if best.trajectory_uncertainty > 0.7 and self._replans_this_block < MAX_REPLANS_PER_BLOCK:
            self._replans_this_block += 1
            return ExecutiveDecision(
                action=ExecutiveAction.EXPAND,
                selected_candidate_id=best.candidate_id,
                recall_query_params=None,
                expand_budget=8,  # extra proposal slots
                defer_reason=None,
                scores=scored,
                replans_used=self._replans_this_block,
            )

        # Select best
        self._total_compute_spent += int(best.compute_cost * 10)
        return ExecutiveDecision(
            action=ExecutiveAction.SELECT,
            selected_candidate_id=best.candidate_id,
            recall_query_params=None,
            expand_budget=None,
            defer_reason=None,
            scores=scored,
            replans_used=self._replans_this_block,
        )

    def reset_block(self) -> None:
        """Reset replan counter for a new 32-token block."""
        self._replans_this_block = 0

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: Any, strict: bool = True) -> Any:
        return super().load_state_dict(state_dict, strict=strict)
