"""Learner state, bounded memory, update actions, and cost accounting.

No Darwin-X dependency. Pure Python + numpy.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ── Constants ──────────────────────────────────────────────────────
EPSILON = 1e-10


# ── Update Actions ─────────────────────────────────────────────────
class UpdateAction(enum.Enum):
    SKIP = "skip"                    # no persistent change
    LOCAL_UPDATE = "local_update"    # bounded update to existing slot
    ALLOCATE = "allocate"           # create new regime slot
    CONSOLIDATE = "consolidate"     # merge or protect repeatedly useful state


# ── Cost constants per action ──────────────────────────────────────
ACTION_COST = {
    UpdateAction.SKIP:         {"compute": 1, "write": 0},
    UpdateAction.LOCAL_UPDATE: {"compute": 3, "write": 1},
    UpdateAction.ALLOCATE:     {"compute": 5, "write": 2},
    UpdateAction.CONSOLIDATE:  {"compute": 4, "write": 2},
}


# ── Experience ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class Experience:
    """Immutable experience. c_t is latent — policies NEVER receive it."""
    x: np.ndarray          # observable feature vector [feature_dim]
    c: int                 # latent regime (evaluator-only)
    y: np.ndarray          # outcome [outcome_dim]
    q: float               # consequence importance [0, 1]
    d: int                 # delay before consequence observable
    t: int                 # absolute timestep

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x.tolist(), "c": self.c,
            "y": self.y.tolist(), "q": self.q,
            "d": self.d, "t": self.t,
        }


# ── Regime Slot ────────────────────────────────────────────────────
@dataclass
class RegimeSlot:
    """Persistent prototype for a learned regime.

    Anchored in Hebb (1949): cell assembly — a distributed pattern
    that is strengthened by recurrence and weakened by disuse.
    """
    prototype: np.ndarray         # [feature_dim] — centroid of regime
    outcome_weights: np.ndarray   # [outcome_dim, feature_dim] — linear mapping
    birth_step: int
    last_accessed: int
    access_count: int = 0
    recurrence_count: int = 0     # times this slot was the best match
    protected: bool = False       # consolidated = immune to eviction
    interference_score: float = 0.0  # cumulative damage from overlapping updates
    recent_errors: list[float] = field(default_factory=list)  # rolling prediction errors
    sustained_failure: bool = False   # true when errors consistently high

    def to_dict(self) -> dict[str, Any]:
        return {
            "prototype": self.prototype.tolist(),
            "outcome_weights": self.outcome_weights.tolist(),
            "birth_step": self.birth_step,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "recurrence_count": self.recurrence_count,
            "protected": self.protected,
            "interference_score": float(self.interference_score),
        }


# ── Eligibility Trace ──────────────────────────────────────────────
@dataclass
class EligibilityTrace:
    """Bounded trace linking a past experience to a pending consequence.

    Anchored in Wiener (1948): temporal credit assignment.
    """
    experience_ref: int           # timestep of the original experience
    remaining_delay: int          # steps until consequence arrives
    initial_delay: int            # original d_t for normalization
    feature_snapshot: np.ndarray  # [feature_dim] at time of experience

    def to_dict(self) -> dict[str, Any]:
        return {
            "experience_ref": self.experience_ref,
            "remaining_delay": self.remaining_delay,
            "initial_delay": self.initial_delay,
            "feature_snapshot": self.feature_snapshot.tolist(),
        }


# ── Learner State ──────────────────────────────────────────────────
class BudgetExceededError(Exception):
    """Raised when a policy exceeds its compute, write, or memory budget."""


@dataclass
class LearnerState:
    """Full learner state with bounded capacity on all dimensions.

    Anchored in von Neumann (1948, Hixon): 3 timescales —
    fast (prediction/confidence), episodic (traces), slow (slots).
    """
    # Slow state: persistent regime prototypes
    slots: list[RegimeSlot] = field(default_factory=list)
    max_slots: int = 16

    # Episodic state: eligibility traces
    traces: list[EligibilityTrace] = field(default_factory=list)
    max_traces: int = 64

    # Budget tracking
    compute_budget_remaining: float = float("inf")
    write_budget_remaining: int = 10_000
    total_compute_spent: float = 0.0
    total_writes_spent: int = 0
    total_interference_events: int = 0

    # Fast state accumulators
    recent_surprisals: list[float] = field(default_factory=list)  # rolling window
    max_recent_surprisals: int = 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_slots": len(self.slots),
            "slots": [s.to_dict() for s in self.slots],
            "num_traces": len(self.traces),
            "traces": [t.to_dict() for t in self.traces],
            "compute_budget_remaining": self.compute_budget_remaining,
            "write_budget_remaining": self.write_budget_remaining,
            "total_compute_spent": self.total_compute_spent,
            "total_writes_spent": self.total_writes_spent,
            "total_interference_events": self.total_interference_events,
        }


# ── Pure Functions ─────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < EPSILON or nb < EPSILON:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def predict(
    state: LearnerState, x: np.ndarray
) -> tuple[np.ndarray, float, int | None]:
    """Predict outcome from current state.

    Returns:
        outcome_prediction: [outcome_dim]
        confidence: float in [0, 1]
        matched_slot_idx: int or None (which slot was the best match)
    """
    if not state.slots:
        return np.zeros(1, dtype=np.float64), 0.0, None

    # Find best matching slot by prototype similarity
    best_sim = -1.0
    best_idx = 0
    for i, slot in enumerate(state.slots):
        sim = cosine_similarity(x, slot.prototype)
        if sim > best_sim:
            best_sim = sim
            best_idx = i

    slot = state.slots[best_idx]
    prediction = slot.outcome_weights @ x
    confidence = max(0.0, min(1.0, float(best_sim)))

    return prediction, confidence, best_idx


def execute_action(
    state: LearnerState,
    action: UpdateAction,
    exp: Experience,
    matched_slot_idx: int | None,
    *,
    lr: float = 0.1,
) -> tuple[LearnerState, float, int]:
    """Execute an update action and return (new_state, compute_cost, write_cost).

    IMPORTANT: Returns ACTUAL costs, not estimated. A SKIP still costs
    compute (the decision itself). Cannot be called with an action whose
    cost would exceed the remaining budget.

    Anchored in Ashby (1949): the 4 levels of homeostat response.
    """
    cost = ACTION_COST[action]
    compute_cost = float(cost["compute"])
    write_cost = int(cost["write"])

    # Budget check
    if state.compute_budget_remaining < compute_cost:
        raise BudgetExceededError(
            f"Compute budget exceeded: need {compute_cost}, "
            f"have {state.compute_budget_remaining}"
        )
    if state.write_budget_remaining < write_cost:
        raise BudgetExceededError(
            f"Write budget exceeded: need {write_cost}, "
            f"have {state.write_budget_remaining}"
        )

    state.total_compute_spent += compute_cost
    state.total_writes_spent += write_cost
    state.compute_budget_remaining -= compute_cost
    state.write_budget_remaining -= write_cost

    if action == UpdateAction.SKIP:
        pass  # no persistent change

    elif action == UpdateAction.LOCAL_UPDATE:
        if matched_slot_idx is not None and matched_slot_idx < len(state.slots):
            slot = state.slots[matched_slot_idx]
            # Bounded update to prototype
            alpha = lr * 0.5  # bounded, per spec
            slot.prototype = (1.0 - alpha) * slot.prototype + alpha * exp.x
            slot.prototype /= np.linalg.norm(slot.prototype) + EPSILON
            # Update outcome mapping
            pred = slot.outcome_weights @ exp.x
            error = exp.y - pred
            slot.outcome_weights += alpha * np.outer(error, exp.x)
            # Track interference
            for j, other in enumerate(state.slots):
                if j != matched_slot_idx:
                    overlap = cosine_similarity(slot.prototype, other.prototype)
                    if overlap > 0.3:
                        slot.interference_score += overlap * 0.01
                        state.total_interference_events += 1
            slot.last_accessed = exp.t
            slot.access_count += 1

    elif action == UpdateAction.ALLOCATE:
        if len(state.slots) >= state.max_slots:
            _evict_lru(state)
        prototype = exp.x.copy()
        prototype /= np.linalg.norm(prototype) + EPSILON
        outcome_weights = np.zeros((len(exp.y), len(exp.x)), dtype=np.float64)
        state.slots.append(RegimeSlot(
            prototype=prototype,
            outcome_weights=outcome_weights,
            birth_step=exp.t,
            last_accessed=exp.t,
            access_count=1,
        ))

    elif action == UpdateAction.CONSOLIDATE:
        if matched_slot_idx is not None and matched_slot_idx < len(state.slots):
            slot = state.slots[matched_slot_idx]
            slot.protected = True
            slot.recurrence_count += 1
            # Merge nearby unprotected slots into this one
            for j in range(len(state.slots) - 1, -1, -1):
                if j != matched_slot_idx and not state.slots[j].protected:
                    sim = cosine_similarity(slot.prototype, state.slots[j].prototype)
                    if sim > 0.8:
                        # Weighted merge
                        w1 = slot.access_count
                        w2 = state.slots[j].access_count
                        total_w = w1 + w2
                        slot.prototype = (
                            (w1 * slot.prototype + w2 * state.slots[j].prototype) / total_w
                        )
                        slot.prototype /= np.linalg.norm(slot.prototype) + EPSILON
                        slot.outcome_weights = (
                            (w1 * slot.outcome_weights + w2 * state.slots[j].outcome_weights)
                            / total_w
                        )
                        slot.access_count += state.slots[j].access_count
                        del state.slots[j]

    return state, compute_cost, write_cost


def deliver_delayed_consequences(
    state: LearnerState, current_step: int,
    experience_log: dict[int, Experience],
) -> list[tuple[Experience, np.ndarray | None]]:
    """Process pending traces. Returns list of (experience, outcome_now_available).

    Anchored in Wiener (1948): delayed feedback delivery.
    """
    results: list[tuple[Experience, np.ndarray | None]] = []
    new_traces: list[EligibilityTrace] = []

    for trace in state.traces:
        trace.remaining_delay -= 1
        if trace.remaining_delay <= 0:
            exp = experience_log.get(trace.experience_ref)
            if exp is not None:
                results.append((exp, exp.y))
        else:
            new_traces.append(trace)

    state.traces = new_traces[:state.max_traces]
    return results


# ── Helpers ─────────────────────────────────────────────────────────

def _evict_lru(state: LearnerState) -> None:
    """Evict the least-recently-accessed non-protected slot."""
    candidates = [s for s in state.slots if not s.protected]
    if not candidates:
        raise BudgetExceededError("All slots protected — cannot evict")
    victim = min(candidates, key=lambda s: s.last_accessed)
    state.slots.remove(victim)
