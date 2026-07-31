import pytest

from f51_darwin.evolution_score import EvolutionMetrics, calculate_evolution_score, should_grow


def test_evolution_score_formula() -> None:
    score = calculate_evolution_score(
        EvolutionMetrics(
            new_gain=0.4,
            retention=0.8,
            forgetting=0.1,
            compute_cost=0.2,
            redundancy=0.05,
        )
    )
    assert score == pytest.approx(0.85)


def test_growth_requires_all_gates() -> None:
    assert should_grow(
        new_loss=2.5,
        replay_loss_improved=False,
        local_adaptation_failed=True,
        evolution_score=0.3,
    )
    assert not should_grow(
        new_loss=2.5,
        replay_loss_improved=True,
        local_adaptation_failed=True,
        evolution_score=0.3,
    )
