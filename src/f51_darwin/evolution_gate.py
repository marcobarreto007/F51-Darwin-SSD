from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvolutionGateThresholds:
    min_loss_delta: float = 0.01
    max_replay_regression: float = 0.02
    min_verified_generated: int = 1
    min_verification_rate: float = 0.5
    require_replay: bool = True


@dataclass(frozen=True)
class EvolutionGateMetrics:
    baseline_loss: float
    candidate_loss: float
    generated_total: int
    verified_generated: int
    baseline_replay_loss: float | None = None
    candidate_replay_loss: float | None = None
    train_steps: int = 0
    tokens_seen: int = 0

    @property
    def loss_delta(self) -> float:
        return float(self.baseline_loss) - float(self.candidate_loss)

    @property
    def verification_rate(self) -> float:
        return self.verified_generated / max(self.generated_total, 1)

    @property
    def replay_delta(self) -> float | None:
        if self.baseline_replay_loss is None or self.candidate_replay_loss is None:
            return None
        return float(self.baseline_replay_loss) - float(self.candidate_replay_loss)


@dataclass(frozen=True)
class EvolutionGateDecision:
    action: str
    passed: bool
    gates: dict[str, bool] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    score: float = 0.0


def decide_promotion(
    metrics: EvolutionGateMetrics,
    thresholds: EvolutionGateThresholds | None = None,
) -> EvolutionGateDecision:
    """Decide whether a candidate checkpoint/module is promotable.

    Promotion requires measured learning, replay retention, and verified model
    output. This prevents loss-only improvements from being mislabeled as
    self-evolution.
    """

    thresholds = thresholds or EvolutionGateThresholds()
    replay_available = metrics.replay_delta is not None
    replay_ok = (
        replay_available
        and metrics.candidate_replay_loss is not None
        and metrics.baseline_replay_loss is not None
        and metrics.candidate_replay_loss <= metrics.baseline_replay_loss + thresholds.max_replay_regression
    )
    if not thresholds.require_replay and not replay_available:
        replay_ok = True

    gates = {
        "loss_improved": metrics.loss_delta >= thresholds.min_loss_delta,
        "replay_retained": replay_ok,
        "verified_generation": metrics.verified_generated >= thresholds.min_verified_generated,
        "verification_rate": metrics.verification_rate >= thresholds.min_verification_rate,
    }

    reasons: list[str] = []
    if not gates["loss_improved"]:
        reasons.append(
            f"loss_delta {metrics.loss_delta:.6f} < required {thresholds.min_loss_delta:.6f}"
        )
    if not replay_available and thresholds.require_replay:
        reasons.append("missing replay loss before/after")
    elif not gates["replay_retained"]:
        reasons.append(
            "replay regression exceeds tolerance "
            f"{thresholds.max_replay_regression:.6f}"
        )
    if not gates["verified_generation"]:
        reasons.append(
            f"verified_generated {metrics.verified_generated} < required {thresholds.min_verified_generated}"
        )
    if not gates["verification_rate"]:
        reasons.append(
            f"verification_rate {metrics.verification_rate:.3f} < required {thresholds.min_verification_rate:.3f}"
        )

    score = (
        metrics.loss_delta
        + (metrics.replay_delta or 0.0)
        + metrics.verification_rate
    )
    passed = all(gates.values())
    if passed:
        action = "promote"
    elif gates["loss_improved"] and gates["replay_retained"]:
        action = "quarantine"
    else:
        action = "reject"

    return EvolutionGateDecision(
        action=action,
        passed=passed,
        gates=gates,
        reasons=reasons,
        score=score,
    )
