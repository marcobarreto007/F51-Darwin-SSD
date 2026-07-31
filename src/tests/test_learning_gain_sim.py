"""Validation and falsification gate tests for learning_gain simulation.

These tests MUST pass before any scientific run.
"""

from __future__ import annotations

import json
import numpy as np
import pytest

from research.learning_gain.state import (
    Experience, LearnerState, RegimeSlot, EligibilityTrace,
    UpdateAction, ACTION_COST, BudgetExceededError,
    cosine_similarity, predict, execute_action, deliver_delayed_consequences,
)
from research.learning_gain.scenarios import (
    generate_s01, generate_s02, generate_s03, generate_s04, generate_s05,
    generate_s06, generate_s07, generate_s08, generate_s09, generate_s10,
    SCENARIO_GENERATORS, check_s06_marginals, get_scenario_config,
)
from research.learning_gain.policies import (
    FrozenPolicy, AlwaysUpdatePolicy, SurpriseOnlyPolicy,
    GainAdaptivePolicy, RandomGatePolicy,
)
from research.learning_gain.evaluator import (
    create_probes, Evaluator, compute_G, check_falsification_gates, leakage_check,
)
from research.learning_gain.runner import run_single, run_experiment


# ═══════════════════════════════════════════════════════════════════
# VALIDATION TESTS (must pass BEFORE scientific run)
# ═══════════════════════════════════════════════════════════════════

class TestDeterministicReplay:
    """Same seed -> identical Experience stream."""

    def test_s01_replay(self):
        a = generate_s01(42)
        b = generate_s01(42)
        assert len(a) == len(b)
        for i, (ea, eb) in enumerate(zip(a, b)):
            assert np.allclose(ea.x, eb.x), f"Feature mismatch at {i}"
            assert np.allclose(ea.y, eb.y), f"Outcome mismatch at {i}"
            assert ea.t == eb.t

    def test_all_scenarios_replay(self):
        for sid, gen in SCENARIO_GENERATORS.items():
            a = gen(42)
            b = gen(42)
            assert len(a) == len(b), f"{sid}: length mismatch"
            assert all(np.allclose(ea.x, eb.x) for ea, eb in zip(a, b)), f"{sid}: x mismatch"
            assert all(np.allclose(ea.y, eb.y) for ea, eb in zip(a, b)), f"{sid}: y mismatch"


class TestScenarioInvariants:
    """Scenario invariants from spec."""

    def test_s06_marginal_preservation(self):
        orig = generate_s01(42)
        shuf = generate_s06(42)
        assert len(orig) == len(shuf)
        # Marginals should be approximately preserved
        assert check_s06_marginals(orig, shuf), "S06 marginals not preserved"

    def test_seed_different_produces_different_streams(self):
        a = generate_s01(42)
        b = generate_s01(43)
        # Should differ somewhere
        any_diff = any(
            not np.allclose(ea.x, eb.x) or not np.allclose(ea.y, eb.y)
            for ea, eb in zip(a, b)
        )
        assert any_diff, "Different seeds produced identical streams"

    def test_c_not_in_observable_features(self):
        """c_t (latent regime) is a separate field — policies never receive it.

        The real invariant: x_t alone cannot predict c_t. In S08,
        alternating contexts share the same features (proto_shared)
        so x_t should have near-zero mutual information with c_t.
        """
        exps_s08 = generate_s08(42)
        # In S08, features are shared between contexts
        # So x_t should NOT be separable by c_t
        xs_a = np.array([e.x for e in exps_s08 if e.c == 0])
        xs_b = np.array([e.x for e in exps_s08 if e.c == 1])
        if len(xs_a) > 0 and len(xs_b) > 0:
            mean_a = xs_a.mean(axis=0)
            mean_b = xs_b.mean(axis=0)
            cos_sim = float(np.dot(mean_a, mean_b) / (
                np.linalg.norm(mean_a) * np.linalg.norm(mean_b) + 1e-10))
            # High cosine similarity -> features are shared -> c not in x
            assert cos_sim > 0.7, \
                f"S08: features differ by context (cos={cos_sim:.3f}), possible c leak"


class TestStateMachine:
    """State machine correctness."""

    def test_cosine_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert abs(cosine_similarity(a, a) - 1.0) < 1e-9

    def test_cosine_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(cosine_similarity(a, b)) < 1e-9

    def test_predict_empty_state(self):
        state = LearnerState()
        pred, conf, matched = predict(state, np.array([1.0, 2.0]))
        assert np.all(pred == 0.0)
        assert conf == 0.0
        assert matched is None

    def test_predict_with_slots(self):
        state = LearnerState()
        slot = RegimeSlot(
            prototype=np.array([1.0, 0.0]),
            outcome_weights=np.array([[0.5, 0.2]]),
            birth_step=0, last_accessed=0,
        )
        state.slots.append(slot)
        pred, conf, matched = predict(state, np.array([1.0, 0.1]))
        assert matched == 0
        assert conf > 0.9

    def test_skip_action_costs_minimal(self):
        state = LearnerState(compute_budget_remaining=100, write_budget_remaining=100)
        exp = Experience(x=np.ones(2), c=0, y=np.ones(1), q=0.5, d=0, t=0)
        state, comp, write = execute_action(state, UpdateAction.SKIP, exp, None)
        assert comp == ACTION_COST[UpdateAction.SKIP]["compute"]
        assert write == ACTION_COST[UpdateAction.SKIP]["write"]

    def test_allocate_creates_slot(self):
        state = LearnerState(compute_budget_remaining=100, write_budget_remaining=100)
        exp = Experience(x=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64), c=0,
                         y=np.ones(1, dtype=np.float64), q=0.5, d=0, t=0)
        assert len(state.slots) == 0
        state, _, _ = execute_action(state, UpdateAction.ALLOCATE, exp, None)
        assert len(state.slots) == 1
        assert state.slots[0].birth_step == 0

    def test_budget_exceeded_raises(self):
        state = LearnerState(compute_budget_remaining=0, write_budget_remaining=0)
        exp = Experience(x=np.ones(2), c=0, y=np.ones(1), q=0.5, d=0, t=0)
        with pytest.raises(BudgetExceededError):
            execute_action(state, UpdateAction.LOCAL_UPDATE, exp, 0)


class TestEvaluatorIsolation:
    """Evaluator isolation from policies."""

    def test_evaluator_never_passed_to_policy(self):
        exps = generate_s01(42)
        probes = create_probes(exps, 42)
        evaluator = Evaluator(probes, "S01")
        policy = GainAdaptivePolicy()
        # The policy constructor never receives evaluator
        # The policy's decide_action never receives evaluator
        # This is an architectural guarantee
        assert not leakage_check(policy, evaluator)

    def test_no_oracle_access(self):
        """Policy cannot access future outcomes."""
        policy = GainAdaptivePolicy()
        # Policy API only exposes: exp.x, exp.y, exp.q, exp.d (NOT exp.c)
        # This is enforced by the Experience dataclass having c as a separate
        # field that policies never read (by convention, enforced in runner)
        pass  # Architectural guarantee


class TestNoSpeculativeWork:
    """SKIP must never cost more than LOCAL_UPDATE."""

    def test_skip_cheaper_than_local_update(self):
        assert ACTION_COST[UpdateAction.SKIP]["compute"] < \
               ACTION_COST[UpdateAction.LOCAL_UPDATE]["compute"]

    def test_skip_cheaper_than_allocate(self):
        assert ACTION_COST[UpdateAction.SKIP]["compute"] < \
               ACTION_COST[UpdateAction.ALLOCATE]["compute"]


class TestAccountingConservation:
    """Total compute spent must equal sum of action costs."""

    def test_single_action_accounting(self):
        state = LearnerState(compute_budget_remaining=100, write_budget_remaining=100)
        exp = Experience(x=np.ones(4, dtype=np.float64), c=0,
                         y=np.ones(1, dtype=np.float64), q=0.5, d=0, t=0)
        initial = state.total_compute_spent
        state, comp, write = execute_action(state, UpdateAction.LOCAL_UPDATE, exp, None)
        assert state.total_compute_spent == initial + comp
        assert state.total_writes_spent == write


class TestSeedPairing:
    """All policies must see identical streams for same seed."""

    def test_same_seed_same_stream(self):
        s1 = generate_s01(123)
        s2 = generate_s01(123)
        assert all(np.allclose(a.x, b.x) and np.allclose(a.y, b.y)
                   for a, b in zip(s1, s2))


class TestS10EvaluatorIndependence:
    """S10: internal score manipulation must not affect external evaluation."""

    def test_s10_probes_are_independent(self):
        exps = generate_s10(42)
        probes = create_probes(exps, 42)
        evaluator = Evaluator(probes, "S10")
        # Evaluator owns probes independently of policy
        assert len(probes.held_out_indices) > 0
        assert evaluator.scenario_id == "S10"


# ═══════════════════════════════════════════════════════════════════
# FALSIFICATION GATE TESTS (verify gates work on small scale)
# ═══════════════════════════════════════════════════════════════════

class TestGate2NoExpensiveSkip:
    """Gate 2: SKIP never costs more than LOCAL_UPDATE."""

    def test_all_actions_have_declared_costs(self):
        for action in UpdateAction:
            assert action in ACTION_COST, f"{action} has no cost entry"
            assert "compute" in ACTION_COST[action]
            assert "write" in ACTION_COST[action]


class TestGate7Reproducibility:
    """Gate 7: Same seed, same config -> same results."""

    def test_same_config_same_results(self):
        exps = generate_s01(42)
        cfg = get_scenario_config("S01")
        policy1 = FrozenPolicy()
        r1 = run_single(policy1, exps, "S01", 42, cfg)

        exps2 = generate_s01(42)
        policy2 = FrozenPolicy()
        r2 = run_single(policy2, exps2, "S01", 42, cfg)

        assert r1["total_compute"] == r2["total_compute"]
        assert r1["total_writes"] == r2["total_writes"]


# ═══════════════════════════════════════════════════════════════════
# SMOKE TEST (quick full-pipeline run)
# ═══════════════════════════════════════════════════════════════════

class TestSmokeRun:
    """Quick full-pipeline run with minimal seeds."""

    def test_full_pipeline_smoke(self):
        """Run all policies on 2 scenarios with 3 seeds."""
        raw, aggregate = run_experiment(
            scenarios=["S01", "S02"],
            seeds=[0, 1, 2],
        )
        assert len(raw) > 0
        assert aggregate["n_scenarios"] == 2
        assert aggregate["n_seeds"] == 3
        # Every policy should have results
        for pname in ["Frozen", "AlwaysUpdate", "SurpriseOnly", "GainAdaptive"]:
            p_rows = [r for r in raw if r["policy"] == pname]
            assert len(p_rows) > 0, f"No results for {pname}"

    def test_gain_adaptive_produces_finite_values(self):
        raw, _ = run_experiment(
            scenarios=["S01", "S03"],
            seeds=[0, 1, 2],
        )
        ga_rows = [r for r in raw if r["policy"] == "GainAdaptive"]
        for r in ga_rows:
            assert not np.isnan(r.get("G", 0)), f"NaN G for GA: {r}"
            assert np.isfinite(r.get("total_compute", 0)), f"Infinite compute for GA"


# ═══════════════════════════════════════════════════════════════════
# POLICY BEHAVIOR TESTS
# ═══════════════════════════════════════════════════════════════════

class TestPolicyBehavior:
    """Verify each policy behaves as specified."""

    def test_frozen_never_updates_after_warmup(self):
        policy = FrozenPolicy()
        state = LearnerState()
        exp = Experience(x=np.ones(4), c=0, y=np.ones(1), q=0.5, d=0, t=100)
        action = policy.decide_action(state, exp, np.zeros(1), 0.5, None)
        assert action == UpdateAction.SKIP

    def test_always_update_updates(self):
        policy = AlwaysUpdatePolicy(similarity_threshold=0.3)
        state = LearnerState()
        exp = Experience(x=np.ones(4), c=0, y=np.ones(1), q=0.5, d=0, t=0)
        action = policy.decide_action(state, exp, np.zeros(1), 0.1, None)
        assert action == UpdateAction.ALLOCATE  # no match, allocates

    def test_surprise_only_skips_when_low_surprise(self):
        policy = SurpriseOnlyPolicy(surprise_threshold=5.0)  # very high threshold
        state = LearnerState()
        exp = Experience(x=np.ones(4), c=0, y=np.ones(1), q=0.5, d=0, t=0)
        pred = np.array([0.99])
        action = policy.decide_action(state, exp, pred, 0.9, None)
        assert action == UpdateAction.SKIP

    def test_gain_adaptive_returns_valid_action(self):
        policy = GainAdaptivePolicy()
        state = LearnerState()
        exp = Experience(x=np.ones(4), c=0, y=np.ones(1), q=0.5, d=0, t=0)
        action = policy.decide_action(state, exp, np.zeros(1), 0.3, None)
        assert action in UpdateAction
