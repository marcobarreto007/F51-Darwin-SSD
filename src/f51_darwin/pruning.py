from __future__ import annotations

from dataclasses import dataclass

from f51_darwin.expert_pool import ExpertPool, ExpertRecord, ModuleState


@dataclass(frozen=True)
class AblationDecision:
    action: str
    delta: float
    reason: str


def quarantine_if_low_score(
    pool: ExpertPool,
    expert_id: str,
    threshold: float = -0.10,
) -> ExpertRecord:
    record = pool.records[expert_id]
    if record.state == ModuleState.ACTIVE and record.score <= threshold:
        return pool.set_state(expert_id, ModuleState.QUARANTINE)
    return record


def ablation_decision(
    baseline_score: float,
    ablated_score: float,
    threshold: float = 0.01,
) -> AblationDecision:
    delta = float(baseline_score) - float(ablated_score)
    if delta <= threshold:
        return AblationDecision("quarantine", delta, "ablation impact below threshold")
    return AblationDecision("keep", delta, "module still protects capability")


def mark_dead(pool: ExpertPool, expert_id: str) -> ExpertRecord:
    record = pool.records[expert_id]
    if record.state != ModuleState.QUARANTINE:
        raise ValueError("No component is deleted before quarantine and ablation.")
    return pool.set_state(expert_id, ModuleState.DEAD)


def mark_merged(pool: ExpertPool, expert_id: str) -> ExpertRecord:
    record = pool.records[expert_id]
    if record.state not in {ModuleState.QUARANTINE, ModuleState.ACTIVE, ModuleState.FROZEN}:
        raise ValueError("Only live or quarantined modules can be merged.")
    return pool.set_state(expert_id, ModuleState.MERGED)

