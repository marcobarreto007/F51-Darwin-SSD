"""Deterministic simulation of recurrent Ghost memory.

This is a research harness, not a live Darwin-X organ.  It compares the
semantics of the current Heartbeat surprise slots with a DenStream-inspired
Ghost lifecycle:

    surprise -> transient trace -> recurrence -> real memory
                    |                 |
                    +-- decay/prune   +-- contradiction/quarantine

The proposed candidate deliberately adds two safety gates beyond DenStream:
independent-source recurrence and contradiction quarantine.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable


Vector = tuple[float, ...]


def cosine(a: Vector, b: Vector) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def normalized(values: Iterable[float]) -> Vector:
    values = tuple(float(value) for value in values)
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return values
    return tuple(value / norm for value in values)


@dataclass
class Event:
    vector: Vector
    step: int
    source: str
    claim: str = ""


@dataclass
class BaselineSlot:
    key: Vector
    source: str
    claim: str
    access_count: int = 0


class CurrentHeartbeatSemantics:
    """Small deterministic model of heartbeat.py::TestTimeMemory.

    The live implementation appends one slot for every surprise, retrieves by
    cosine similarity, increments access_count on retrieved slots, and evicts
    the least-accessed slot at capacity.  It has no recurrence merge, decay,
    quarantine, or promotion state.
    """

    def __init__(self, capacity: int = 8, similarity_threshold: float = 0.5):
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.slots: list[BaselineSlot] = []

    def ingest(self, event: Event) -> None:
        if len(self.slots) >= self.capacity:
            victim = min(range(len(self.slots)), key=lambda i: self.slots[i].access_count)
            self.slots.pop(victim)
        self.slots.append(BaselineSlot(event.vector, event.source, event.claim))

        ranked = sorted(
            ((cosine(event.vector, slot.key), index) for index, slot in enumerate(self.slots)),
            reverse=True,
        )[:4]
        if ranked and ranked[0][0] >= self.similarity_threshold:
            for similarity, index in ranked:
                if similarity > 0.0:
                    self.slots[index].access_count += 1

    def advance(self, _step: int) -> None:
        # The current Heartbeat slots do not decay with time.
        return

    def state(self) -> dict[str, object]:
        return {
            "slots": len(self.slots),
            "real": 0,
            "ghost": len(self.slots),
            "quarantine": 0,
            "keys": [list(slot.key) for slot in self.slots],
        }

    def dumps(self) -> str:
        return json.dumps({"capacity": self.capacity, "slots": [asdict(slot) for slot in self.slots]})

    @classmethod
    def loads(cls, payload: str) -> "CurrentHeartbeatSemantics":
        raw = json.loads(payload)
        memory = cls(capacity=int(raw["capacity"]))
        for item in raw["slots"]:
            memory.slots.append(
                BaselineSlot(
                    key=tuple(item["key"]),
                    source=str(item["source"]),
                    claim=str(item["claim"]),
                    access_count=int(item["access_count"]),
                )
            )
        return memory


@dataclass
class GhostTrace:
    trace_id: int
    centroid: Vector
    weight: float
    first_step: int
    last_step: int
    recurrence_count: int
    sources: set[str] = field(default_factory=set)
    claim: str = ""
    state: str = "ghost"

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["centroid"] = list(self.centroid)
        data["sources"] = sorted(self.sources)
        return data


class GhostRecurrenceMemory:
    """DenStream-inspired transient-to-real memory prototype."""

    def __init__(
        self,
        *,
        capacity: int = 8,
        similarity_threshold: float = 0.97,
        decay_lambda: float = 0.12,
        prune_weight: float = 0.20,
        promotion_weight: float = 2.50,
        min_recurrences: int = 3,
        min_independent_sources: int = 2,
    ) -> None:
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.decay_lambda = decay_lambda
        self.prune_weight = prune_weight
        self.promotion_weight = promotion_weight
        self.min_recurrences = min_recurrences
        self.min_independent_sources = min_independent_sources
        self.traces: list[GhostTrace] = []
        self._next_id = 1

    def _decay_to(self, step: int) -> None:
        survivors: list[GhostTrace] = []
        for trace in self.traces:
            if trace.state == "real":
                survivors.append(trace)
                continue
            elapsed = max(0, step - trace.last_step)
            if elapsed:
                trace.weight *= 2.0 ** (-self.decay_lambda * elapsed)
                trace.last_step = step
            if trace.state == "quarantine" or trace.weight >= self.prune_weight:
                survivors.append(trace)
        self.traces = survivors

    def _nearest(self, vector: Vector, states: set[str]) -> tuple[float, GhostTrace | None]:
        candidates = [trace for trace in self.traces if trace.state in states]
        if not candidates:
            return -1.0, None
        scored = [(cosine(vector, trace.centroid), trace) for trace in candidates]
        return max(scored, key=lambda pair: pair[0])

    @staticmethod
    def _contradicts(trace: GhostTrace, event: Event) -> bool:
        return bool(trace.claim and event.claim and trace.claim != event.claim)

    def _new_trace(self, event: Event, state: str = "ghost") -> GhostTrace:
        trace = GhostTrace(
            trace_id=self._next_id,
            centroid=normalized(event.vector),
            weight=1.0,
            first_step=event.step,
            last_step=event.step,
            recurrence_count=1,
            sources={event.source},
            claim=event.claim,
            state=state,
        )
        self._next_id += 1
        self.traces.append(trace)
        self._enforce_capacity()
        return trace

    def _merge(self, trace: GhostTrace, event: Event) -> None:
        old_weight = trace.weight
        combined = tuple(
            (old_weight * old + new) / (old_weight + 1.0)
            for old, new in zip(trace.centroid, normalized(event.vector))
        )
        trace.centroid = normalized(combined)
        trace.weight += 1.0
        trace.last_step = event.step
        trace.recurrence_count += 1
        trace.sources.add(event.source)
        if not trace.claim:
            trace.claim = event.claim

    def _maybe_promote(self, trace: GhostTrace) -> None:
        if (
            trace.state == "ghost"
            and trace.weight >= self.promotion_weight
            and trace.recurrence_count >= self.min_recurrences
            and len(trace.sources) >= self.min_independent_sources
        ):
            trace.state = "real"

    def _enforce_capacity(self) -> None:
        while len(self.traces) > self.capacity:
            removable = [trace for trace in self.traces if trace.state != "real"]
            if not removable:
                break
            victim = min(removable, key=lambda trace: (trace.weight, trace.recurrence_count, trace.last_step))
            self.traces.remove(victim)

    def seed_real(self, event: Event) -> GhostTrace:
        trace = self._new_trace(event, state="real")
        trace.weight = self.promotion_weight
        trace.recurrence_count = self.min_recurrences
        return trace

    def ingest(self, event: Event) -> str:
        self._decay_to(event.step)

        real_similarity, real = self._nearest(event.vector, {"real"})
        if real is not None and real_similarity >= self.similarity_threshold:
            if self._contradicts(real, event):
                self._new_trace(event, state="quarantine")
                return "quarantined_contradiction"
            self._merge(real, event)
            return "reinforced_real"

        ghost_similarity, ghost = self._nearest(event.vector, {"ghost"})
        if ghost is not None and ghost_similarity >= self.similarity_threshold:
            if self._contradicts(ghost, event):
                ghost.state = "quarantine"
                return "quarantined_contradiction"
            self._merge(ghost, event)
            self._maybe_promote(ghost)
            return "promoted" if ghost.state == "real" else "reinforced_ghost"

        self._new_trace(event)
        return "created_ghost"

    def advance(self, step: int) -> None:
        self._decay_to(step)
        self._enforce_capacity()

    def state(self) -> dict[str, object]:
        return {
            "slots": len(self.traces),
            "real": sum(trace.state == "real" for trace in self.traces),
            "ghost": sum(trace.state == "ghost" for trace in self.traces),
            "quarantine": sum(trace.state == "quarantine" for trace in self.traces),
            "keys": [list(trace.centroid) for trace in self.traces],
            "weights": [round(trace.weight, 4) for trace in self.traces],
            "recurrences": [trace.recurrence_count for trace in self.traces],
        }

    def dumps(self) -> str:
        return json.dumps(
            {
                "config": {
                    "capacity": self.capacity,
                    "similarity_threshold": self.similarity_threshold,
                    "decay_lambda": self.decay_lambda,
                    "prune_weight": self.prune_weight,
                    "promotion_weight": self.promotion_weight,
                    "min_recurrences": self.min_recurrences,
                    "min_independent_sources": self.min_independent_sources,
                },
                "next_id": self._next_id,
                "traces": [trace.to_dict() for trace in self.traces],
            },
            sort_keys=True,
        )

    @classmethod
    def loads(cls, payload: str) -> "GhostRecurrenceMemory":
        raw = json.loads(payload)
        memory = cls(**raw["config"])
        memory._next_id = int(raw["next_id"])
        for item in raw["traces"]:
            memory.traces.append(
                GhostTrace(
                    trace_id=int(item["trace_id"]),
                    centroid=tuple(item["centroid"]),
                    weight=float(item["weight"]),
                    first_step=int(item["first_step"]),
                    last_step=int(item["last_step"]),
                    recurrence_count=int(item["recurrence_count"]),
                    sources=set(item["sources"]),
                    claim=str(item["claim"]),
                    state=str(item["state"]),
                )
            )
        return memory


@dataclass
class ScenarioResult:
    scenario: str
    expected: str
    baseline_passed: bool
    proposed_passed: bool
    baseline_state: dict[str, object]
    proposed_state: dict[str, object]


def _run_both(
    name: str,
    expected: str,
    exercise: Callable[[CurrentHeartbeatSemantics, GhostRecurrenceMemory], None],
    baseline_check: Callable[[dict[str, object]], bool],
    proposed_check: Callable[[dict[str, object]], bool],
    *,
    capacity: int = 8,
) -> ScenarioResult:
    baseline = CurrentHeartbeatSemantics(capacity=capacity)
    proposed = GhostRecurrenceMemory(capacity=capacity)
    exercise(baseline, proposed)
    baseline_state = baseline.state()
    proposed_state = proposed.state()
    return ScenarioResult(
        scenario=name,
        expected=expected,
        baseline_passed=baseline_check(baseline_state),
        proposed_passed=proposed_check(proposed_state),
        baseline_state=baseline_state,
        proposed_state=proposed_state,
    )


def run_scenarios() -> list[ScenarioResult]:
    results: list[ScenarioResult] = []

    def one_off(baseline, proposed):
        event = Event((1.0, 0.0), 0, "source-a", "fact-a")
        baseline.ingest(event); proposed.ingest(event)
        baseline.advance(100); proposed.advance(100)

    results.append(_run_both(
        "01_one_off_noise",
        "A non-recurring surprise decays completely.",
        one_off,
        lambda state: state["slots"] == 0,
        lambda state: state["slots"] == 0,
    ))

    def exact_recurrence(baseline, proposed):
        for step, source in enumerate(("a", "b", "c")):
            event = Event((1.0, 0.0), step, source, "fact-a")
            baseline.ingest(event); proposed.ingest(event)

    results.append(_run_both(
        "02_exact_recurrence",
        "Three independent exact recurrences become one real memory.",
        exact_recurrence,
        lambda state: state["real"] == 1 and state["slots"] == 1,
        lambda state: state["real"] == 1 and state["slots"] == 1,
    ))

    def semantic_recurrence(baseline, proposed):
        vectors = ((1.0, 0.0), (0.995, 0.05), (0.99, -0.04))
        for step, (vector, source) in enumerate(zip(vectors, ("a", "b", "c"))):
            event = Event(vector, step, source, "same-concept")
            baseline.ingest(event); proposed.ingest(event)

    results.append(_run_both(
        "03_semantic_variants",
        "Nearby latent paraphrases merge and promote without duplication.",
        semantic_recurrence,
        lambda state: state["real"] == 1 and state["slots"] == 1,
        lambda state: state["real"] == 1 and state["slots"] == 1,
    ))

    def distinct_concepts(baseline, proposed):
        for step, source in enumerate(("a", "b", "c")):
            for vector, claim, suffix in (((1.0, 0.0), "alpha", "x"), ((0.0, 1.0), "beta", "y")):
                event = Event(vector, step, source + suffix, claim)
                baseline.ingest(event); proposed.ingest(event)

    results.append(_run_both(
        "04_distinct_concepts",
        "Two dissimilar recurrent concepts form two memories, not one or six.",
        distinct_concepts,
        lambda state: state["real"] == 2 and state["slots"] == 2,
        lambda state: state["real"] == 2 and state["slots"] == 2,
    ))

    def delayed_recurrence(baseline, proposed):
        first = Event((1.0, 0.0), 0, "a", "fact-a")
        baseline.ingest(first); proposed.ingest(first)
        baseline.advance(100); proposed.advance(100)
        second = Event((1.0, 0.0), 100, "b", "fact-a")
        baseline.ingest(second); proposed.ingest(second)

    results.append(_run_both(
        "05_delayed_recurrence",
        "A recurrence after the decay horizon starts a new ghost, not a promotion.",
        delayed_recurrence,
        lambda state: state["real"] == 0 and state["ghost"] == 1 and state["slots"] == 1,
        lambda state: state["real"] == 0 and state["ghost"] == 1 and state["slots"] == 1,
    ))

    def historical_match(baseline, proposed):
        seed = Event((1.0, 0.0), 0, "archive", "fact-a")
        baseline.ingest(seed)
        proposed.seed_real(seed)
        event = Event((0.999, 0.02), 1, "new-source", "fact-a")
        baseline.ingest(event); proposed.ingest(event)

    results.append(_run_both(
        "06_match_existing_history",
        "A match reinforces one existing real memory and creates no duplicate.",
        historical_match,
        lambda state: state["real"] == 1 and state["slots"] == 1,
        lambda state: state["real"] == 1 and state["ghost"] == 0 and state["slots"] == 1,
    ))

    def contradiction(baseline, proposed):
        seed = Event((1.0, 0.0), 0, "archive", "earth-round")
        baseline.ingest(seed)
        proposed.seed_real(seed)
        event = Event((1.0, 0.0), 1, "hostile", "earth-flat")
        baseline.ingest(event); proposed.ingest(event)

    results.append(_run_both(
        "07_contradiction",
        "A latent match with contradictory content is quarantined.",
        contradiction,
        lambda state: state["real"] == 1 and state["quarantine"] == 1,
        lambda state: state["real"] == 1 and state["quarantine"] == 1,
    ))

    def source_burst(baseline, proposed):
        for step in range(6):
            event = Event((1.0, 0.0), step, "same-source", "fact-a")
            baseline.ingest(event); proposed.ingest(event)

    results.append(_run_both(
        "08_single_source_burst",
        "Repeated copies from one source remain one ghost and cannot self-confirm.",
        source_burst,
        lambda state: state["real"] == 0 and state["ghost"] == 1 and state["slots"] == 1,
        lambda state: state["real"] == 0 and state["ghost"] == 1 and state["slots"] == 1,
    ))

    def capacity_pressure(baseline, proposed):
        signal_a = Event((1, 0, 0, 0, 0, 0), 0, "a", "signal")
        signal_b = Event((1, 0, 0, 0, 0, 0), 1, "b", "signal")
        for event in (signal_a, signal_b):
            baseline.ingest(event); proposed.ingest(event)
        noises = (
            (0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0),
            (0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 1),
        )
        for offset, vector in enumerate(noises, 2):
            event = Event(vector, offset, f"noise-{offset}", f"noise-{offset}")
            baseline.ingest(event); proposed.ingest(event)

    def proposed_kept_signal(state):
        return state["slots"] <= 4 and any(key[0] > 0.9 for key in state["keys"])

    def baseline_kept_signal(state):
        return state["slots"] <= 4 and any(key[0] > 0.9 for key in state["keys"])

    results.append(_run_both(
        "09_capacity_pressure",
        "Bounded memory prunes one-off noise while preserving recurrent signal.",
        capacity_pressure,
        baseline_kept_signal,
        proposed_kept_signal,
        capacity=4,
    ))

    def checkpoint_roundtrip(baseline, proposed):
        for step, source in enumerate(("a", "b")):
            event = Event((1.0, 0.0), step, source, "fact-a")
            baseline.ingest(event); proposed.ingest(event)
        restored_baseline = CurrentHeartbeatSemantics.loads(baseline.dumps())
        restored_proposed = GhostRecurrenceMemory.loads(proposed.dumps())
        event = Event((1.0, 0.0), 2, "c", "fact-a")
        restored_baseline.ingest(event); restored_proposed.ingest(event)
        baseline.slots = restored_baseline.slots
        proposed.traces = restored_proposed.traces
        proposed._next_id = restored_proposed._next_id

    results.append(_run_both(
        "10_checkpoint_roundtrip",
        "A two-hit ghost survives restart and the third independent hit promotes it.",
        checkpoint_roundtrip,
        lambda state: state["real"] == 1 and state["slots"] == 1,
        lambda state: state["real"] == 1 and state["slots"] == 1,
    ))

    return results


def report(results: list[ScenarioResult]) -> dict[str, object]:
    baseline_passes = sum(result.baseline_passed for result in results)
    proposed_passes = sum(result.proposed_passed for result in results)
    return {
        "benchmark": "Ghost recurrence lifecycle",
        "candidates": {
            "baseline": "Current Heartbeat surprise-slot semantics",
            "proposed": "DenStream-inspired recurrent Ghost lifecycle",
        },
        "scenario_count": len(results),
        "baseline_passes": baseline_passes,
        "proposed_passes": proposed_passes,
        "results": [asdict(result) for result in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Write the complete deterministic result as JSON.")
    args = parser.parse_args()

    payload = report(run_scenarios())
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Ghost recurrence benchmark")
    print(f"Current Heartbeat: {payload['baseline_passes']}/{payload['scenario_count']}")
    print(f"Proposed lifecycle: {payload['proposed_passes']}/{payload['scenario_count']}")
    for result in payload["results"]:
        print(
            f"  {result['scenario']}: current={'PASS' if result['baseline_passed'] else 'FAIL'} "
            f"proposed={'PASS' if result['proposed_passed'] else 'FAIL'}"
        )
    return 0 if payload["proposed_passes"] == payload["scenario_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
