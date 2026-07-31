"""Learning policies: Frozen, AlwaysUpdate, SurpriseOnly, GainAdaptive, RandomGate.

Each policy observes (x_t, y_t, q_t, d_t) and decides on an UpdateAction.
Policies NEVER receive c_t (latent regime), future outcomes, or evaluator probes.

Per spec: thresholds are constructor args, fixed before evaluation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from research.learning_gain.state import (
    EPSILON,
    LearnerState,
    Experience,
    UpdateAction,
    RegimeSlot,
    EligibilityTrace,
    cosine_similarity,
    predict,
    execute_action,
)

# ── Base Policy Interface ──────────────────────────────────────────

class Policy:
    """Abstract policy. All policies must implement decide_action()."""

    def __init__(self, name: str):
        self.name = name
        self.n_updates = 0
        self.n_skips = 0
        self.n_allocates = 0
        self.n_consolidates = 0

    def warm_up(
        self, state: LearnerState, warmup_experiences: list[Experience]
    ) -> LearnerState:
        """Shared warm-up: populate initial slots from warmup data."""
        for exp in warmup_experiences:
            _, _, matched = predict(state, exp.x)
            if matched is None or cosine_similarity(
                exp.x, state.slots[matched].prototype
            ) < 0.5:
                state, _, _ = execute_action(
                    state, UpdateAction.ALLOCATE, exp, None, lr=0.05,
                )
            else:
                state, _, _ = execute_action(
                    state, UpdateAction.LOCAL_UPDATE, exp, matched, lr=0.05,
                )
        return state

    def decide_action(
        self, state: LearnerState, exp: Experience,
        prediction: np.ndarray, confidence: float,
        matched_slot: int | None,
    ) -> UpdateAction:
        raise NotImplementedError

    def record_action(self, action: UpdateAction) -> None:
        if action == UpdateAction.SKIP:
            self.n_skips += 1
        elif action == UpdateAction.LOCAL_UPDATE:
            self.n_updates += 1
        elif action == UpdateAction.ALLOCATE:
            self.n_allocates += 1
        elif action == UpdateAction.CONSOLIDATE:
            self.n_consolidates += 1

    def get_stats(self) -> dict[str, Any]:
        return {
            "policy": self.name,
            "n_skips": self.n_skips,
            "n_updates": self.n_updates,
            "n_allocates": self.n_allocates,
            "n_consolidates": self.n_consolidates,
        }


# ── Frozen ─────────────────────────────────────────────────────────

class FrozenPolicy(Policy):
    """Never changes persistent state after warm-up.

    Supplies the no-learning baseline.
    """

    def __init__(self):
        super().__init__("Frozen")

    def decide_action(
        self, state: LearnerState, exp: Experience,
        prediction: np.ndarray, confidence: float,
        matched_slot: int | None,
    ) -> UpdateAction:
        return UpdateAction.SKIP


# ── Always Update ──────────────────────────────────────────────────

class AlwaysUpdatePolicy(Policy):
    """Applies a local update to EVERY eligible experience.

    Allocates when no prototype matches. Tests whether more
    plasticity is automatically better.
    """

    def __init__(self, similarity_threshold: float = 0.5):
        super().__init__("AlwaysUpdate")
        self.similarity_threshold = similarity_threshold

    def decide_action(
        self, state: LearnerState, exp: Experience,
        prediction: np.ndarray, confidence: float,
        matched_slot: int | None,
    ) -> UpdateAction:
        if matched_slot is not None and confidence >= self.similarity_threshold:
            return UpdateAction.LOCAL_UPDATE
        else:
            return UpdateAction.ALLOCATE


# ── Surprise Only ──────────────────────────────────────────────────

class SurpriseOnlyPolicy(Policy):
    """Updates when predictive surprisal exceeds a threshold.

    Tests whether prediction error alone distinguishes useful
    novelty from noise.
    """

    def __init__(self, surprise_threshold: float = 0.5):
        super().__init__("SurpriseOnly")
        self.surprise_threshold = surprise_threshold

    def decide_action(
        self, state: LearnerState, exp: Experience,
        prediction: np.ndarray, confidence: float,
        matched_slot: int | None,
    ) -> UpdateAction:
        # Surprisal: negative log probability of outcome given prediction
        error = float(np.sum((exp.y - prediction) ** 2))
        surprisal = -math.log(max(EPSILON, 1.0 - min(0.99, error)))

        state.recent_surprisals.append(surprisal)
        if len(state.recent_surprisals) > state.max_recent_surprisals:
            state.recent_surprisals.pop(0)

        if surprisal > self.surprise_threshold:
            if matched_slot is not None and confidence >= 0.3:
                return UpdateAction.LOCAL_UPDATE
            else:
                return UpdateAction.ALLOCATE
        else:
            return UpdateAction.SKIP


# ── Gain Adaptive ──────────────────────────────────────────────────

class GainAdaptivePolicy(Policy):
    """Estimates future value of an update; acts on positive expected gain.

    g_t = P(recur | h_t) · I_t · q_t - λ_w·C_write - λ_i·Ĉ_interference - λ_c·C_compute

    Selects the cheapest action with positive expected gain, or SKIP
    if all negative. Cannot inspect future stream or evaluator.

    Anchored in:
    - Hebb (1949): P(recur) via recurrence_count
    - Shannon (1948): I_t as information gain
    - Ashby (1949): bounded actions with cost tradeoff
    - Wiener (1948): prediction error as feedback signal
    """

    def __init__(
        self,
        similarity_threshold: float = 0.4,
        lambda_write: float = 0.05,
        lambda_interference: float = 0.1,
        lambda_compute: float = 0.01,
        recurrence_decay: float = 0.95,
    ):
        super().__init__("GainAdaptive")
        self.similarity_threshold = similarity_threshold
        self.lambda_write = lambda_write
        self.lambda_interference = lambda_interference
        self.lambda_compute = lambda_compute
        self.recurrence_decay = recurrence_decay

    def decide_action(
        self, state: LearnerState, exp: Experience,
        prediction: np.ndarray, confidence: float,
        matched_slot: int | None,
    ) -> UpdateAction:
        # ── Estimate I_t: information gain ──
        # How much would this experience reduce uncertainty?
        # Approximated as: 1 - confidence (what the model doesn't know)
        I_t = max(0.0, 1.0 - confidence)

        # ── Estimate P(recur | h_t): probability of recurrence ──
        if matched_slot is not None and matched_slot < len(state.slots):
            slot = state.slots[matched_slot]
            # Recurrence probability from history
            total_access = max(1, slot.access_count)
            P_recur = slot.recurrence_count / total_access
            # Decay for old slots
            age = max(0, exp.t - slot.birth_step)
            P_recur *= self.recurrence_decay ** (age / 100.0)

            # ── SUSTAINED FAILURE DETECTION ──
            # Track rolling error for this slot (last 8 predictions)
            error = float(np.sum((exp.y - prediction) ** 2))
            slot.recent_errors.append(error)
            if len(slot.recent_errors) > 8:
                slot.recent_errors.pop(0)
            # Sustained failure: error increased DRAMATICALLY vs historical
            # Requires: absolute error > 0.03 AND 5x worse than baseline
            if len(slot.recent_errors) >= 6 and slot.access_count >= 10:
                recent = slot.recent_errors[-4:]
                older = slot.recent_errors[:4]
                median_recent = float(np.median(recent))
                median_older = float(np.median(older))
                slot.sustained_failure = (
                    median_recent > 0.03  # must be genuinely bad
                    and median_older > 0  # have a baseline
                    and median_recent > median_older * 5.0  # 5x worse
                )
        else:
            P_recur = 0.1  # unknown -> low prior

        # ── Estimate Ĉ_interference ──
        if matched_slot is not None and matched_slot < len(state.slots):
            C_interference_est = state.slots[matched_slot].interference_score
        else:
            C_interference_est = 0.0

        # ── Check sustained failure: force explore new slot ──
        slot_failing = (
            matched_slot is not None
            and matched_slot < len(state.slots)
            and state.slots[matched_slot].sustained_failure
        )

        # ── Evaluate actions ──
        candidates: list[tuple[float, UpdateAction]] = []

        # SKIP: zero gain, zero cost (but compute cost of decision itself)
        g_skip = 0.0
        candidates.append((g_skip, UpdateAction.SKIP))

        # LOCAL_UPDATE (only if slot is NOT failing)
        if (matched_slot is not None
            and confidence >= self.similarity_threshold
            and not slot_failing):
            g_local = (
                P_recur * I_t * exp.q
                - self.lambda_write * 1    # 1 persistent write
                - self.lambda_interference * C_interference_est
                - self.lambda_compute * 3   # 3 compute units
            )
            candidates.append((g_local, UpdateAction.LOCAL_UPDATE))

        # ALLOCATE (boosted priority when slot is failing)
        alloc_boost = 0.3 if slot_failing else 0.0
        g_alloc = (
            P_recur * I_t * exp.q
            + alloc_boost  # boost for regime change detection
            - self.lambda_write * 2    # 2 persistent writes
            - self.lambda_interference * 0.0  # new slot, no interference yet
            - self.lambda_compute * 5   # 5 compute units
        )
        candidates.append((g_alloc, UpdateAction.ALLOCATE))

        # CONSOLIDATE (only if slot has recurred enough)
        if (matched_slot is not None
            and matched_slot < len(state.slots)
            and state.slots[matched_slot].recurrence_count >= 3
            and state.slots[matched_slot].access_count >= 5):
            g_cons = (
                P_recur * I_t * exp.q * 1.2  # consolidation bonus
                - self.lambda_write * 2
                - self.lambda_interference * C_interference_est * 0.5  # reduced future interference
                - self.lambda_compute * 4
            )
            candidates.append((g_cons, UpdateAction.CONSOLIDATE))

        # Select action with max gain
        best_gain, best_action = max(candidates, key=lambda x: x[0])

        if best_gain <= 0:
            return UpdateAction.SKIP
        return best_action


# ── Random Gate (matched control) ──────────────────────────────────

class RandomGatePolicy(Policy):
    """Matches GainAdaptive's update rate but selects randomly.

    Used as secondary routing control. Update rate is matched AFTER
    the GainAdaptive run (set externally).
    """

    def __init__(self, update_probability: float = 0.3, seed: int = 0):
        super().__init__("RandomGate")
        self.update_probability = update_probability
        self._rng = np.random.RandomState(seed)

    def decide_action(
        self, state: LearnerState, exp: Experience,
        prediction: np.ndarray, confidence: float,
        matched_slot: int | None,
    ) -> UpdateAction:
        if self._rng.random() < self.update_probability:
            if matched_slot is not None and confidence >= 0.3:
                return UpdateAction.LOCAL_UPDATE
            else:
                return UpdateAction.ALLOCATE
        return UpdateAction.SKIP


# ── Policy Registry ────────────────────────────────────────────────

POLICY_REGISTRY: dict[str, type[Policy]] = {
    "Frozen": FrozenPolicy,
    "AlwaysUpdate": AlwaysUpdatePolicy,
    "SurpriseOnly": SurpriseOnlyPolicy,
    "GainAdaptive": GainAdaptivePolicy,
    "RandomGate": RandomGatePolicy,
}
