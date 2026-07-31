from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvolutionMetrics:
    new_gain: float
    retention: float
    forgetting: float
    compute_cost: float
    redundancy: float


def calculate_evolution_score(metrics: EvolutionMetrics) -> float:
    return (
        float(metrics.new_gain)
        + float(metrics.retention)
        - float(metrics.forgetting)
        - float(metrics.compute_cost)
        - float(metrics.redundancy)
    )


def should_grow(
    *,
    new_loss: float,
    replay_loss_improved: bool,
    local_adaptation_failed: bool,
    evolution_score: float,
    new_loss_threshold: float = 2.0,
    min_evolution_score: float = 0.15,
) -> bool:
    return (
        new_loss >= new_loss_threshold
        and not replay_loss_improved
        and local_adaptation_failed
        and evolution_score >= min_evolution_score
    )

