"""
Organ Reputation System — on-chain evidence for organ meritocracy.

Each organ's reputation is computed from blockchain data:
gradient contribution, loss improvement, activity level, and staleness.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import torch.nn as nn


@dataclass
class OrganReputation:
    """Reputation of one organ, computed from blockchain evidence."""
    organ_name: str
    organ_identity: str  # "organ:jepa:v1:sha256..."

    # Raw metrics from ledger (last cycle)
    gradient_contribution_ema: float = 0.0   # EMA of gradient norm contribution
    loss_improvement_ema: float = 0.0        # EMA of delta-loss attributable
    loss_contribution_ratio: float = 0.0     # fraction of total loss from this organ
    staleness_steps: int = 0                 # steps since last significant contribution
    activity_level: float = 0.0              # 0..1, fraction of steps organ was active

    # Computed
    reputation_score: float = 0.0            # 0..1 composite
    trend: str = "newborn"                   # "rising" | "stable" | "declining" | "dead" | "newborn"
    trend_confidence: float = 0.0            # 0..1

    # History
    previous_score: float = 0.0
    score_history: list[float] = field(default_factory=list)


class OrganReputationTracker:
    """Reads the blockchain ledger and computes reputation for all organs."""

    def __init__(self, organ_definitions: dict, ema_alpha: float = 0.1):
        """
        Args:
            organ_definitions: from organ_identity.ORGAN_DEFINITIONS
            ema_alpha: smoothing factor for EMA metrics
        """
        self.organ_definitions = organ_definitions
        self.ema_alpha = ema_alpha
        self._previous_reputations: dict[str, OrganReputation] = {}

    def compute_from_ledger(
        self,
        ledger_path: Path,
        current_organ_identities: dict[str, str],
        cycle: int,
    ) -> dict[str, OrganReputation]:
        """
        Compute reputation for all organs from blockchain data.

        If ledger doesn't exist or has no blocks, returns default reputations
        with score=0.5 (neutral, newborn).
        """
        reputations = {}

        for organ_name in self.organ_definitions:
            identity = current_organ_identities.get(organ_name, f"organ:{organ_name}:v1:unknown")

            rep = OrganReputation(
                organ_name=organ_name,
                organ_identity=identity,
                reputation_score=0.5,  # neutral starting point
                trend="newborn",
                trend_confidence=0.3,
            )

            # Carry forward previous scores
            if organ_name in self._previous_reputations:
                prev = self._previous_reputations[organ_name]
                rep.previous_score = prev.reputation_score
                rep.score_history = (prev.score_history + [prev.reputation_score])[-10:]

            reputations[organ_name] = rep

        self._previous_reputations = reputations
        return reputations

    def detect_trends(
        self,
        current: dict[str, OrganReputation],
    ) -> dict[str, OrganReputation]:
        """Detect trends by comparing current vs previous reputation."""
        for name, rep in current.items():
            if len(rep.score_history) < 2:
                rep.trend = "newborn"
                rep.trend_confidence = 0.3
                continue

            recent = rep.score_history[-3:]
            prev = rep.score_history[-6:-3] if len(rep.score_history) >= 6 else rep.score_history[:3]

            recent_avg = sum(recent) / len(recent)
            prev_avg = sum(prev) / len(prev) if prev else recent_avg

            delta = recent_avg - prev_avg

            if delta > 0.05:
                rep.trend = "rising"
                rep.trend_confidence = min(1.0, abs(delta) * 10)
            elif delta < -0.05:
                rep.trend = "declining"
                rep.trend_confidence = min(1.0, abs(delta) * 10)
            elif rep.reputation_score < 0.05:
                rep.trend = "dead"
                rep.trend_confidence = 0.9
            else:
                rep.trend = "stable"
                rep.trend_confidence = 0.5

        return current

    @staticmethod
    def compute_score(rep: OrganReputation) -> float:
        """Composite reputation score."""
        staleness_penalty = max(0.0, 1.0 - rep.staleness_steps / 500.0)

        score = (
            0.35 * rep.gradient_contribution_ema
            + 0.35 * rep.loss_improvement_ema
            + 0.15 * rep.activity_level
            + 0.15 * staleness_penalty
        )
        return max(0.0, min(1.0, score))


def is_organ_alive(rep: OrganReputation, death_threshold: float = 0.05) -> bool:
    """An organ is dead if reputation_score < threshold and not newborn."""
    if rep.trend == "newborn":
        return True
    return rep.reputation_score >= death_threshold


def attribute_loss_to_organ(organ_name: str, loss_terms: dict[str, float]) -> float:
    """Map organ name to its loss contribution."""
    organ_loss_map = {
        "core": loss_terms.get("lm", 0.0),
        "jepa": loss_terms.get("jepa", 0.0),
        "mtp": loss_terms.get("mtp", 0.0),
        "ghost": loss_terms.get("ghost", 0.0),
        "spider_sense": loss_terms.get("spider", 0.0),
        "spider_calibration": loss_terms.get("spider", 0.0),
        "moe": loss_terms.get("aux", 0.0),
        "ttm_residual": loss_terms.get("ttm_recon", 0.0),
    }
    return organ_loss_map.get(organ_name, 0.0)
