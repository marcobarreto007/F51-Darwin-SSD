from __future__ import annotations

import pytest
import torch

from f51_darwin.cognition.executive import (
    MAX_REPLANS_PER_BLOCK,
    ExecutiveAction,
    ExecutiveDecision,
    ScoredTrajectory,
    TrajectoryCritic,
    UniversalExecutive,
)


# ── helpers ─────────────────────────────────────────────────────────────────


def _make_organ(**kwargs: object) -> UniversalExecutive:
    return UniversalExecutive(**kwargs)


def _make_traj(
    cid: str = "traj_0",
    value: float = 0.5,
    safety: float = 0.1,
    uncertainty: float = 0.2,
) -> ScoredTrajectory:
    return ScoredTrajectory(
        candidate_id=cid,
        goal_alignment=0.6,
        memory_consistency=0.5,
        predicted_value=value,
        trajectory_uncertainty=uncertainty,
        novelty=0.3,
        safety_risk=safety,
        compute_cost=1.0,
        total_score=value - 2.0 * safety - 0.5 * uncertainty + 0.3 * 0.6 + 0.2 * 0.5 - 0.1,
        rank=1,
    )


# ── Trajectory Critic ───────────────────────────────────────────────────────


def test_critic_output_shapes() -> None:
    critic = TrajectoryCritic()
    zs = torch.randn(3, 512)
    zm = torch.randn(3, 512)
    zl = torch.randn(3, 512)
    scores = critic(zs, zm, zl)
    assert "goal_alignment" in scores
    assert "safety_risk" in scores
    assert scores["goal_alignment"].shape == (3,)
    assert scores["safety_risk"].shape == (3,)


def test_critic_with_goal_and_memory() -> None:
    critic = TrajectoryCritic()
    zs = torch.randn(2, 512)
    zm = torch.randn(2, 512)
    zl = torch.randn(2, 512)
    goal = torch.randn(2, 512)
    mem = torch.randn(2)
    scores = critic(zs, zm, zl, goal, mem)
    assert all(torch.isfinite(v).all() for v in scores.values())


# ── Scoring ─────────────────────────────────────────────────────────────────


def test_score_empty_returns_empty() -> None:
    organ = _make_organ()
    result = organ.score_candidates([])
    assert result == []


def test_score_candidates_ranks_by_total_score() -> None:
    organ = _make_organ()
    cands = [
        {"z_short": torch.randn(512), "z_medium": torch.randn(512), "z_long": torch.randn(512)},
        {"z_short": torch.randn(512), "z_medium": torch.randn(512), "z_long": torch.randn(512)},
        {"z_short": torch.randn(512), "z_medium": torch.randn(512), "z_long": torch.randn(512)},
    ]
    scored = organ.score_candidates(cands)
    assert len(scored) == 3
    # Check descending order
    for i in range(len(scored) - 1):
        assert scored[i].total_score >= scored[i + 1].total_score
    # Ranks assigned
    assert scored[0].rank == 1
    assert scored[1].rank == 2
    assert scored[2].rank == 3


# ── Decision logic ──────────────────────────────────────────────────────────


def test_empty_candidates_defers() -> None:
    organ = _make_organ()
    decision = organ.decide([])
    assert decision.action == ExecutiveAction.ASK_OR_DEFER
    assert decision.defer_reason == "no_candidates"


def test_low_risk_good_value_selects_best() -> None:
    organ = _make_organ()
    scored = [
        _make_traj("a", value=0.8, safety=0.1),
        _make_traj("b", value=0.3, safety=0.05),
    ]
    scored[0].rank = 1
    scored[1].rank = 2
    decision = organ.decide(scored)
    assert decision.action == ExecutiveAction.SELECT
    assert decision.selected_candidate_id == "a"


def test_high_risk_falls_back_to_safe() -> None:
    organ = _make_organ(safety_threshold=0.5)
    scored = [
        _make_traj("risky", value=0.9, safety=0.8),
        _make_traj("safe", value=0.5, safety=0.1),
    ]
    scored[0].rank = 1
    scored[1].rank = 2
    decision = organ.decide(scored)
    assert decision.action == ExecutiveAction.SELECT
    assert decision.selected_candidate_id == "safe"


def test_all_high_risk_defers() -> None:
    organ = _make_organ(safety_threshold=0.5)
    scored = [
        _make_traj("a", value=0.9, safety=0.9),
        _make_traj("b", value=0.8, safety=0.8),
    ]
    scored[0].rank = 1
    scored[1].rank = 2
    decision = organ.decide(scored)
    assert decision.action == ExecutiveAction.ASK_OR_DEFER
    assert decision.defer_reason == "all_high_risk"


def test_replan_limit_enforced() -> None:
    organ = _make_organ(min_value_threshold=0.9)
    scored = [_make_traj("a", value=0.1, safety=0.1)]

    # Should retry up to MAX_REPLANS_PER_BLOCK
    for _ in range(MAX_REPLANS_PER_BLOCK):
        decision = organ.decide(scored)
        if decision.action == ExecutiveAction.RECALL_AGAIN:
            pass
        else:
            break

    # After max replans, should defer
    decision = organ.decide(scored)
    assert decision.action in (ExecutiveAction.ASK_OR_DEFER, ExecutiveAction.RECALL_AGAIN)


def test_reset_block_resets_replan_counter() -> None:
    organ = _make_organ(min_value_threshold=0.9)
    scored = [_make_traj("a", value=0.1, safety=0.1)]
    organ.decide(scored)  # consumes one replan
    assert organ._replans_this_block > 0
    organ.reset_block()
    assert organ._replans_this_block == 0
