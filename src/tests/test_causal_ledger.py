from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from f51_darwin.organism.causal_bus import (
    AblationArm,
    Intervention,
    InterventionOperation,
    InterventionTarget,
    OrganCausalBus,
    Phase,
    Signal,
    StepIdentity,
    StepOutcome,
)
from f51_darwin.organism.causal_ledger import (
    CausalLedger,
    LedgerTamperError,
    verify_ledger,
)


def step(*, optimizer_step: int = 12) -> StepIdentity:
    return StepIdentity(
        run_id="ledger-run",
        cycle=4,
        optimizer_step=optimizer_step,
        accumulation_window=1,
        batch_digest="batch:ledger",
        rng_digest="rng:ledger",
        base_checkpoint_id="base:ledger",
        training_contract_id="contract:ledger",
        ablation_plan_id="plan:ledger",
    )


class LedgerAdapter:
    adapter_id = "ledger-adapter"
    version = "v1"

    def observe(self, identity, phase, context):
        return (
            Signal(
                signal_id=f"lm/{identity.optimizer_step}",
                source="ledger-adapter:v1",
                phase=phase,
                step_key=identity.key,
                name="lm_loss",
                value=context.get("lm_loss", 1.0),
                unit="nats/token",
            ),
        )

    def propose(self, identity, phase, signals, context):
        return (
            Intervention(
                intervention_id=f"aux-zero/{identity.optimizer_step}",
                source="ledger-adapter:v1",
                phase=phase,
                target=InterventionTarget.LOSS_TERM,
                operation=InterventionOperation.SET_SCALE,
                subject="aux",
                value=0.0,
                valid_from_step=identity.optimizer_step,
                valid_through_step=identity.optimizer_step,
                evidence_ids=(signals[0].signal_id,),
            ),
        )

    def feedback(self, outcome):
        return None


def outcome(
    identity: StepIdentity, ids: tuple[str, ...] = ()
) -> StepOutcome:
    return StepOutcome(
        identity=identity,
        optimizer_step_applied=True,
        accepted_intervention_ids=ids,
        raw_losses={"lm": 1.0, "aux": 0.02},
        effective_losses={"lm": 1.0, "aux": 0.0},
        gradient_norm_before=2.0,
        gradient_norm_after=1.0,
        parameter_delta_norm=0.01,
    )


def test_ledger_writes_canonical_hash_chain_and_resolves_intent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "causal.jsonl"
    ledger = CausalLedger(path, fsync=False)
    identity = step()
    bus = OrganCausalBus(
        arm=AblationArm.SHADOW,
        adapters=(LedgerAdapter(),),
        intent_recorder=ledger,
    )

    decision = bus.decide(identity, Phase.PRE_LOSS, {"lm_loss": 1.0})
    bus.record_intent(identity, (decision,))
    assert decision.accepted
    first_verification = ledger.verify()
    assert first_verification.valid
    assert first_verification.event_count == 1
    assert first_verification.orphan_intent_step_keys == (identity.key,)

    ledger.record_outcome(outcome(identity))
    verification = ledger.verify()
    assert verification.valid
    assert verification.event_count == 2
    assert verification.orphan_intent_step_keys == ()

    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert lines[0] == json.dumps(
        first,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert first["event_type"] == "STEP_INTENT"
    assert first["sequence"] == 0
    assert first["prev_hash"] == "0" * 64
    assert second["event_type"] == "STEP_OUTCOME"
    assert second["sequence"] == 1
    assert second["prev_hash"] == first["event_hash"]
    assert verification.head_hash == second["event_hash"]


def test_multiple_phases_are_one_aggregate_intent_until_step_outcome(
    tmp_path: Path,
) -> None:
    ledger = CausalLedger(tmp_path / "causal.jsonl", fsync=False)
    identity = step()
    empty_bus = OrganCausalBus(
        arm=AblationArm.SHADOW,
        adapters=(),
        intent_recorder=ledger,
    )

    decisions = (
        empty_bus.decide(identity, Phase.PRE_LOSS, {}),
        empty_bus.decide(identity, Phase.PRE_BACKWARD, {}),
    )
    empty_bus.record_intent(identity, decisions)

    verification = ledger.verify()
    assert verification.orphan_intent_step_keys == (identity.key,)
    assert verification.orphan_intent_count == 1
    assert verification.event_count == 1

    ledger.record_outcome(outcome(identity))
    assert ledger.verify().orphan_intent_count == 0


def test_tampering_is_detected_and_reopen_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "causal.jsonl"
    ledger = CausalLedger(path, fsync=False)
    identity = step()
    bus = OrganCausalBus(
        arm=AblationArm.SHADOW,
        adapters=(LedgerAdapter(),),
        intent_recorder=ledger,
    )
    decision = bus.decide(identity, Phase.PRE_LOSS, {"lm_loss": 1.0})
    bus.record_intent(identity, (decision,))

    event = json.loads(path.read_text(encoding="utf-8"))
    event["payload"]["decisions"][0]["context"]["lm_loss"] = 999.0
    path.write_text(
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    verification = verify_ledger(path)
    assert not verification.valid
    assert any("event_hash" in error for error in verification.errors)
    with pytest.raises(LedgerTamperError):
        CausalLedger(path, fsync=False)


def test_truncated_or_invalid_tail_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "causal.jsonl"
    path.write_text('{"schema_version":1', encoding="utf-8")

    verification = verify_ledger(path)

    assert not verification.valid
    assert verification.event_count == 0
    assert any("invalid_json" in error for error in verification.errors)


def test_nonfinite_outcome_is_rejected_without_advancing_chain(
    tmp_path: Path,
) -> None:
    path = tmp_path / "causal.jsonl"
    ledger = CausalLedger(path, fsync=False)
    identity = step()
    with pytest.raises(TypeError, match="gradient_norm_after"):
        replace(outcome(identity), gradient_norm_after=float("nan"))

    verification = ledger.verify()
    assert verification.valid
    assert verification.event_count == 0
    assert verification.head_hash == "0" * 64


def test_outcome_without_intent_is_gracefully_skipped(
    tmp_path: Path,
) -> None:
    """Outcome without a prior INTENT returns an adapted sentinel (no crash).

    Since the adaptive-tolerance change (commit 3e0121a), the ledger
    silently skips orphan outcomes instead of raising — the organism
    survives and the next cycle corrects naturally.
    """
    ledger = CausalLedger(tmp_path / "causal.jsonl", fsync=False)

    result = ledger.record_outcome(outcome(step()))
    assert result.get("_adapted") is True
    assert result.get("_reason") == "intent lost — outcome skipped"

    # Chain remains untouched — no event was written.
    assert ledger.verify().valid
    assert ledger.verify().event_count == 0


def test_public_aliases_and_recover_head_follow_an_external_valid_append(
    tmp_path: Path,
) -> None:
    path = tmp_path / "causal.jsonl"
    first_writer = CausalLedger(path, fsync=False)
    second_writer = CausalLedger(path, fsync=False)
    identity = step()
    bus = OrganCausalBus(
        arm=AblationArm.SHADOW,
        adapters=(LedgerAdapter(),),
    )
    decision = bus.decide(identity, Phase.PRE_LOSS, {"lm_loss": 1.0})

    second_writer.append_intent(identity, (decision,))
    assert first_writer.recover_head() == second_writer.head_hash

    first_writer.append_outcome(outcome(identity))
    verification = first_writer.verify_chain()
    assert verification.valid
    assert verification.event_count == 2


def test_duplicate_intent_after_completed_step_is_gracefully_skipped(
    tmp_path: Path,
) -> None:
    """Duplicate INTENT after a completed step returns an adapted sentinel.

    Since the dedup fix, a fully-recorded step (INTENT + OUTCOME) silently
    skips duplicate INTENTs instead of raising — the organism recovers
    naturally on the next cycle.
    """
    ledger = CausalLedger(tmp_path / "causal.jsonl", fsync=False)
    identity = step()
    bus = OrganCausalBus(
        arm=AblationArm.CONTROL, intent_recorder=ledger
    )
    decision = bus.decide(identity, Phase.PRE_LOSS, {})
    bus.record_intent(identity, (decision,))
    ledger.record_outcome(outcome(identity))

    # Duplicate INTENT for an already-completed step → adapted, not raised.
    result = bus.record_intent(identity, (decision,))
    assert result is not None
    assert result.get("_adapted") is True
    assert result.get("_reason") == "step already recorded — intent skipped"
    # Verify the ledger directly: duplicate intent was skipped.
    verification = ledger.verify()
    assert verification.valid
    assert verification.event_count == 2  # only original INTENT + OUTCOME
    assert verification.orphan_intent_count == 0

    # A retry with a new attempt_id still works.
    retry = replace(identity, attempt_id="attempt-1")
    retry_decision = bus.decide(retry, Phase.PRE_LOSS, {})
    bus.record_intent(retry, (retry_decision,))
    verification2 = ledger.verify()
    assert verification2.valid
    assert verification2.event_count == 3  # +new INTENT
    assert verification2.orphan_intent_count == 1  # retry INTENT has no OUTCOME yet


def test_outcome_ids_must_be_effective_unique_and_match_arm(
    tmp_path: Path,
) -> None:
    ledger = CausalLedger(tmp_path / "causal.jsonl", fsync=False)
    identity = step()
    bus = OrganCausalBus(
        arm=AblationArm.SHADOW,
        adapters=(LedgerAdapter(),),
        intent_recorder=ledger,
    )
    decision = bus.decide(identity, Phase.PRE_LOSS, {"lm_loss": 1.0})
    bus.record_intent(identity, (decision,))

    with pytest.raises(ValueError, match="not_effective"):
        ledger.record_outcome(
            outcome(identity, (f"aux-zero/{identity.optimizer_step}",))
        )
    assert ledger.verify().valid
    assert ledger.verify().orphan_intent_count == 1


def test_apply_outcome_requires_exact_effective_ids_without_error(
    tmp_path: Path,
) -> None:
    ledger = CausalLedger(tmp_path / "causal.jsonl", fsync=False)
    identity = step()
    bus = OrganCausalBus(
        arm=AblationArm.APPLY,
        adapters=(LedgerAdapter(),),
        intent_recorder=ledger,
    )
    decision = bus.decide(identity, Phase.PRE_LOSS, {"lm_loss": 1.0})
    bus.record_intent(identity, (decision,))

    with pytest.raises(ValueError, match="missing_effective"):
        ledger.record_outcome(outcome(identity))
    exact_id = f"aux-zero/{identity.optimizer_step}"
    ledger.record_outcome(outcome(identity, (exact_id,)))

    verification = ledger.verify()
    assert verification.valid
    assert verification.orphan_intent_count == 0


def test_apply_error_outcome_may_report_applied_subset(
    tmp_path: Path,
) -> None:
    ledger = CausalLedger(tmp_path / "causal.jsonl", fsync=False)
    identity = step()
    bus = OrganCausalBus(
        arm=AblationArm.APPLY,
        adapters=(LedgerAdapter(),),
        intent_recorder=ledger,
    )
    decision = bus.decide(identity, Phase.PRE_LOSS, {"lm_loss": 1.0})
    bus.record_intent(identity, (decision,))
    failed = replace(outcome(identity), error="executor_failed")

    ledger.record_outcome(failed)

    assert ledger.verify().valid
    assert ledger.verify().orphan_intent_count == 0


def test_step_anchor_type_and_key_are_verified_even_after_rehash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "causal.jsonl"
    ledger = CausalLedger(path, fsync=False)
    identity = step()
    bus = OrganCausalBus(
        arm=AblationArm.CONTROL, intent_recorder=ledger
    )
    decision = bus.decide(identity, Phase.PRE_LOSS, {})
    bus.record_intent(identity, (decision,))
    event = json.loads(path.read_text(encoding="utf-8"))
    event["step"]["optimizer_step"] = "12"
    body = {key: value for key, value in event.items() if key != "event_hash"}
    encoded = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    import hashlib

    event["event_hash"] = hashlib.sha256(encoded).hexdigest()
    path.write_text(
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    verification = verify_ledger(path)
    assert not verification.valid
    assert any("optimizer_step" in error for error in verification.errors)
