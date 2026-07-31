from __future__ import annotations

from f51_darwin.evolution_gate import (
    EvolutionGateMetrics,
    EvolutionGateThresholds,
    decide_promotion,
)


def test_promotion_requires_loss_replay_and_verified_generation() -> None:
    decision = decide_promotion(
        EvolutionGateMetrics(
            baseline_loss=2.0,
            candidate_loss=1.7,
            baseline_replay_loss=1.5,
            candidate_replay_loss=1.49,
            generated_total=4,
            verified_generated=3,
        )
    )
    assert decision.passed
    assert decision.action == "promote"
    assert all(decision.gates.values())


def test_loss_improvement_without_verification_quarantines() -> None:
    decision = decide_promotion(
        EvolutionGateMetrics(
            baseline_loss=2.0,
            candidate_loss=1.7,
            baseline_replay_loss=1.5,
            candidate_replay_loss=1.49,
            generated_total=4,
            verified_generated=0,
        )
    )
    assert not decision.passed
    assert decision.action == "quarantine"
    assert not decision.gates["verified_generation"]


def test_replay_regression_rejects_candidate() -> None:
    decision = decide_promotion(
        EvolutionGateMetrics(
            baseline_loss=2.0,
            candidate_loss=1.7,
            baseline_replay_loss=1.5,
            candidate_replay_loss=1.9,
            generated_total=4,
            verified_generated=3,
        )
    )
    assert not decision.passed
    assert decision.action == "reject"
    assert not decision.gates["replay_retained"]


def test_missing_replay_can_be_allowed_for_smoke_only() -> None:
    decision = decide_promotion(
        EvolutionGateMetrics(
            baseline_loss=2.0,
            candidate_loss=1.7,
            generated_total=2,
            verified_generated=1,
        ),
        EvolutionGateThresholds(require_replay=False),
    )
    assert decision.passed
