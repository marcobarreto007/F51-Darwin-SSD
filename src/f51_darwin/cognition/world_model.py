"""HierarchicalWorldModel — semantic future prediction organ.

Generates multiple trajectory proposals at three temporal horizons
(short/32, medium/128, long/300 tokens) using a learned context encoder
and an EMA-updated target encoder.  Anti-collapse losses maintain
diversity across proposals.

Gate contract:
- gate=0.0 → zero residual, backbone untouched
- gate>0.0 → bounded plan conditioning via PlanAdapter
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .contracts import COGNITIVE_ORGAN_WIDTH


WORLD_MODEL_STATE_SCHEMA = "darwin-world-model-state-v1"

# Horizon token counts
HORIZON_SHORT = 32
HORIZON_MEDIUM = 128
HORIZON_LONG = 300
NUM_PROPOSAL_SLOTS = 8


# ── Target Encoder (EMA) ────────────────────────────────────────────────────


class TargetEncoder(nn.Module):
    """EMA-updated encoder that never sees the future directly during training.

    Produces pooled representations for three horizon blocks from raw hidden states.
    """

    def __init__(self, d_model: int, ema_decay: float = 0.996) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_model, COGNITIVE_ORGAN_WIDTH),
            nn.GELU(),
            nn.Linear(COGNITIVE_ORGAN_WIDTH, COGNITIVE_ORGAN_WIDTH),
        )
        self.ema_decay = float(ema_decay)
        # EMA copy
        self.ema_encoder = copy.deepcopy(self.encoder)
        for p in self.ema_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_ema(self) -> None:
        for online, ema in zip(self.encoder.parameters(), self.ema_encoder.parameters()):
            ema.data.mul_(self.ema_decay).add_(online.data, alpha=1.0 - self.ema_decay)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Pool and encode hidden states.

        Args:
            hidden: [B, T, d_model] — raw backbone hidden states

        Returns:
            [B, 512] — pooled representation
        """
        pooled = hidden.mean(dim=1)
        return self.encoder(pooled)

    def encode_target(self, hidden: torch.Tensor) -> torch.Tensor:
        """Encode with EMA (no grad) for target computation."""
        pooled = hidden.mean(dim=1)
        return self.ema_encoder(pooled)


# ── Trajectory Proposer ─────────────────────────────────────────────────────


class TrajectoryProposer(nn.Module):
    """Generates K trajectory proposals from context + goal + memory recalls.

    Uses learned proposal queries that attend to context representations.
    """

    def __init__(self, horizon_dim: int = COGNITIVE_ORGAN_WIDTH) -> None:
        super().__init__()
        self.horizon_dim = int(horizon_dim)
        # Learned proposal queries [K, 512]
        self.proposal_queries = nn.Parameter(
            torch.randn(NUM_PROPOSAL_SLOTS, self.horizon_dim) * 0.02
        )
        # Cross-attention: queries attend to context
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.horizon_dim,
            num_heads=8,
            batch_first=True,
            bias=False,
        )
        # Horizon heads: one per horizon, shared across proposals
        self.short_head = nn.Linear(self.horizon_dim, self.horizon_dim)
        self.medium_head = nn.Linear(self.horizon_dim, self.horizon_dim)
        self.long_head = nn.Linear(self.horizon_dim, self.horizon_dim)

    def forward(
        self,
        context: torch.Tensor,  # [B, T_ctx, 512]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate K trajectory proposals.

        Args:
            context: encoded context + goal + memory

        Returns:
            z_short: [B, K, 512]
            z_medium: [B, K, 512]
            z_long: [B, K, 512]
        """
        B = context.shape[0]
        # Attend proposal queries to context
        queries = self.proposal_queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, 512]
        attended, _ = self.cross_attn(
            query=queries,
            key=context,
            value=context,
        )  # [B, K, 512]
        # Horizon-specific projections
        z_short = self.short_head(attended)
        z_medium = self.medium_head(attended)
        z_long = self.long_head(attended)
        return z_short, z_medium, z_long


# ── Plan Adapter ────────────────────────────────────────────────────────────


class PlanAdapter(nn.Module):
    """Converts a selected trajectory into decoder conditioning.

    Bounded, gate-zero-initialized residual injection at declared positions.
    """

    def __init__(self, d_model: int, max_scale: float) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.max_scale = float(max_scale)
        # Short horizon → immediate conditioning
        self.short_adapter = nn.Linear(COGNITIVE_ORGAN_WIDTH, d_model, bias=False)
        # Medium + Long → directional anchor
        self.anchor_adapter = nn.Linear(2 * COGNITIVE_ORGAN_WIDTH, d_model, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))

    def effective_scale(self) -> torch.Tensor:
        return self.max_scale * torch.tanh(self.gate)

    def condition(
        self,
        hidden: torch.Tensor,
        z_short: torch.Tensor,     # [B, 512]
        z_medium: torch.Tensor,    # [B, 512]
        z_long: torch.Tensor,      # [B, 512]
        positions: torch.Tensor,   # [B, 1]
    ) -> torch.Tensor:
        gate_val = self.effective_scale().to(device=hidden.device, dtype=hidden.dtype)
        if bool(self.gate.detach().eq(0).item()):
            return hidden
        short_proj = self.short_adapter(z_short.to(device=hidden.device, dtype=hidden.dtype))
        anchor = torch.cat([z_medium, z_long], dim=-1).to(device=hidden.device, dtype=hidden.dtype)
        anchor_proj = self.anchor_adapter(anchor)
        combined = short_proj + 0.3 * anchor_proj  # short dominates, anchor guides
        combined = combined.unsqueeze(1)  # [B, 1, d_model]
        reference = hidden.gather(
            dim=1,
            index=positions.unsqueeze(-1).expand(-1, -1, self.d_model),
        )
        combined_norm = combined.float().norm(dim=-1, keepdim=True)
        reference_norm = reference.float().norm(dim=-1, keepdim=True)
        factor = (reference_norm / combined_norm.clamp_min(1e-6)).clamp(max=1.0)
        bounded = combined * factor.to(dtype=combined.dtype)
        delta = gate_val * bounded
        result = hidden.clone()
        result.scatter_add_(
            dim=1,
            index=positions.unsqueeze(-1).expand(-1, -1, self.d_model),
            src=delta,
        )
        return result


# ── The organ ───────────────────────────────────────────────────────────────


@dataclass
class TrajectoryCandidate:
    candidate_id: str
    z_short: torch.Tensor   # [512]
    z_medium: torch.Tensor   # [512]
    z_long: torch.Tensor     # [512]
    diversity_score: float
    slot_index: int


class HierarchicalWorldModel(nn.Module):
    """Generates semantic future trajectories at three horizons.

    Uses a context encoder (trained), a target encoder (EMA, frozen),
    and a trajectory proposer with 8 learned proposal slots.  Produces
    K alternative futures with explicit uncertainty and anti-collapse
    diversity losses.
    """

    def __init__(
        self,
        d_model: int,
        max_scale: float = 0.15,
        *,
        ema_decay: float = 0.996,
        diversity_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.max_scale = float(max_scale)
        self.diversity_weight = float(diversity_weight)

        # Context encoder: d_model → 512
        self.context_encoder = nn.Sequential(
            nn.Linear(d_model, COGNITIVE_ORGAN_WIDTH),
            nn.GELU(),
            nn.Linear(COGNITIVE_ORGAN_WIDTH, COGNITIVE_ORGAN_WIDTH),
        )

        # Target encoder (EMA)
        self.target_encoder = TargetEncoder(d_model, ema_decay)

        # Trajectory proposer
        self.proposer = TrajectoryProposer()

        # Plan adapter for decoder conditioning
        self.plan_adapter = PlanAdapter(d_model, max_scale)

    def encode_context(
        self,
        hidden: torch.Tensor,
        goal_embedding: torch.Tensor | None = None,
        memory_embeddings: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Build context representation from hidden states + optional signals.

        Args:
            hidden: [B, T, d_model]
            goal_embedding: [B, 512] optional
            memory_embeddings: list of [512] optional

        Returns:
            [B, T_ctx, 512] context sequence for proposer
        """
        ctx = self.context_encoder(hidden)  # [B, T, 512]
        # Append goal as an extra token if provided
        if goal_embedding is not None:
            goal = goal_embedding.unsqueeze(1).to(device=ctx.device, dtype=ctx.dtype)
            ctx = torch.cat([ctx, goal], dim=1)
        # Append memory embeddings as extra tokens
        if memory_embeddings:
            for mem in memory_embeddings[:4]:  # max 4 memories to bound compute
                mem_token = mem.to(device=ctx.device, dtype=ctx.dtype)
                if mem_token.ndim == 1:
                    mem_token = mem_token.unsqueeze(0).unsqueeze(0)
                elif mem_token.ndim == 2:
                    mem_token = mem_token.unsqueeze(1)
                ctx = torch.cat([ctx, mem_token.expand(ctx.shape[0], -1, -1)], dim=1)
        return ctx

    def propose_trajectories(
        self,
        hidden: torch.Tensor,
        goal_embedding: torch.Tensor | None = None,
        memory_embeddings: list[torch.Tensor] | None = None,
    ) -> list[TrajectoryCandidate]:
        """Generate K trajectory proposals.

        Returns:
            list of TrajectoryCandidate, one per proposal slot
        """
        context = self.encode_context(hidden, goal_embedding, memory_embeddings)
        zs, zm, zl = self.proposer(context)  # each [B, K, 512]
        B = zs.shape[0]
        candidates: list[TrajectoryCandidate] = []
        for k in range(NUM_PROPOSAL_SLOTS):
            # Compute diversity: avg distance to all other proposals
            all_slots = torch.stack([zs[:, k, :], zm[:, k, :], zl[:, k, :]], dim=1)
            diversity = 0.0
            for j in range(NUM_PROPOSAL_SLOTS):
                if j != k:
                    other = torch.stack([zs[:, j, :], zm[:, j, :], zl[:, j, :]], dim=1)
                    diversity += float((1.0 - F.cosine_similarity(
                        all_slots.reshape(B, -1), other.reshape(B, -1), dim=-1
                    )).mean().detach())
            diversity /= (NUM_PROPOSAL_SLOTS - 1) if NUM_PROPOSAL_SLOTS > 1 else 1.0
            candidates.append(TrajectoryCandidate(
                candidate_id=f"traj_{k}",
                z_short=zs[:, k, :].mean(dim=0).detach(),
                z_medium=zm[:, k, :].mean(dim=0).detach(),
                z_long=zl[:, k, :].mean(dim=0).detach(),
                diversity_score=diversity,
                slot_index=k,
            ))
        return candidates

    def compute_losses(
        self,
        hidden: torch.Tensor,
        future_hidden_short: torch.Tensor | None,
        future_hidden_medium: torch.Tensor | None,
        future_hidden_long: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Compute training losses against observed futures.

        Args:
            hidden: [B, T, d_model] context hidden states
            future_hidden_short: [B, 32, d_model] or None
            future_hidden_medium: [B, 128, d_model] or None
            future_hidden_long: [B, 300, d_model] or None

        Returns:
            dict with 'prediction_error', 'best_of_k', 'diversity', 'anti_collapse'
        """
        losses: dict[str, torch.Tensor] = {}
        device = hidden.device

        context = self.encode_context(hidden)
        zs, zm, zl = self.proposer(context)  # [B, K, 512]

        # Prediction error against EMA targets
        pred_error = torch.tensor(0.0, device=device)
        if future_hidden_short is not None:
            target_s = self.target_encoder.encode_target(future_hidden_short)
            best_s = torch.stack([
                F.cosine_similarity(zs[:, k, :], target_s, dim=-1)
                for k in range(NUM_PROPOSAL_SLOTS)
            ]).max(dim=0).values
            pred_error = pred_error + (1.0 - best_s).mean()

        if future_hidden_medium is not None:
            target_m = self.target_encoder.encode_target(future_hidden_medium)
            best_m = torch.stack([
                F.cosine_similarity(zm[:, k, :], target_m, dim=-1)
                for k in range(NUM_PROPOSAL_SLOTS)
            ]).max(dim=0).values
            pred_error = pred_error + (1.0 - best_m).mean()

        if future_hidden_long is not None:
            target_l = self.target_encoder.encode_target(future_hidden_long)
            best_l = torch.stack([
                F.cosine_similarity(zl[:, k, :], target_l, dim=-1)
                for k in range(NUM_PROPOSAL_SLOTS)
            ]).max(dim=0).values
            pred_error = pred_error + (1.0 - best_l).mean()

        losses["prediction_error"] = pred_error

        # Diversity: encourage proposals to differ
        all_proposals = torch.cat([
            zs.reshape(-1, NUM_PROPOSAL_SLOTS, COGNITIVE_ORGAN_WIDTH),
            zm.reshape(-1, NUM_PROPOSAL_SLOTS, COGNITIVE_ORGAN_WIDTH),
            zl.reshape(-1, NUM_PROPOSAL_SLOTS, COGNITIVE_ORGAN_WIDTH),
        ], dim=0)  # [3*B, K, 512]
        sim_matrix = F.cosine_similarity(
            all_proposals.unsqueeze(2), all_proposals.unsqueeze(1), dim=-1
        )  # [3*B, K, K]
        mask = 1.0 - torch.eye(NUM_PROPOSAL_SLOTS, device=device).unsqueeze(0)
        diversity_loss = (sim_matrix * mask).sum() / (mask.sum() + 1e-8)
        losses["diversity"] = diversity_loss * self.diversity_weight

        # Anti-collapse: variance across proposals should stay above threshold
        variance = all_proposals.var(dim=1).mean()
        anti_collapse = F.relu(0.01 - variance)
        losses["anti_collapse"] = anti_collapse * 0.5

        return losses

    def update_target_encoder(self) -> None:
        self.target_encoder.update_ema()

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: Any, strict: bool = True) -> Any:
        return super().load_state_dict(state_dict, strict=strict)
