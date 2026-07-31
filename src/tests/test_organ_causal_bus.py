from __future__ import annotations

from dataclasses import replace
import math
import random

import numpy as np
import pytest
import torch

from f51_darwin.organism.causal_bus import (
    AblationArm,
    BusDecision,
    CausalBus,
    Intervention,
    InterventionOperation,
    InterventionTarget,
    OrganCausalBus,
    Phase,
    Signal,
    StepIdentity,
    StepOutcome,
    Decision,
)


def step(*, optimizer_step: int = 7) -> StepIdentity:
    return StepIdentity(
        run_id="run-51",
        cycle=3,
        optimizer_step=optimizer_step,
        accumulation_window=2,
        batch_digest="batch:abc",
        rng_digest="rng:def",
        base_checkpoint_id="base:123",
        training_contract_id="contract:456",
        ablation_plan_id="plan:control-vs-apply",
    )


def valid_signal(identity: StepIdentity, *, value: float = 1.25) -> Signal:
    return Signal(
        signal_id="loss/lm",
        source="loss-observer:v1",
        phase=Phase.PRE_LOSS,
        step_key=identity.key,
        name="lm_loss",
        value=value,
        unit="nats/token",
        evidence_ids=("forward:7",),
    )


def valid_intervention(
    identity: StepIdentity,
    *,
    intervention_id: str = "aux-scale",
    source: str = "router-guard:v1",
    priority: int = 10,
) -> Intervention:
    return Intervention(
        intervention_id=intervention_id,
        source=source,
        phase=Phase.PRE_LOSS,
        target=InterventionTarget.LOSS_TERM,
        operation=InterventionOperation.SET_SCALE,
        subject="aux",
        value=0.0,
        valid_from_step=identity.optimizer_step,
        valid_through_step=identity.optimizer_step,
        priority=priority,
        evidence_ids=(),
    )


class RecordingAdapter:
    version = "v1"

    def __init__(
        self,
        *,
        adapter_id: str = "router-guard",
        signal_value: float = 1.25,
        proposals: tuple[Intervention, ...] = (),
    ) -> None:
        self.adapter_id = adapter_id
        self.signal_value = signal_value
        self.proposals = proposals
    def observe(self, identity, phase, context):
        assert type(context) is not dict
        return (
            replace(
                valid_signal(identity, value=self.signal_value),
                signal_id=f"{self.adapter_id}/lm",
                source=f"{self.adapter_id}:v1",
            ),
        )

    def propose(self, identity, phase, signals, context):
        return self.proposals

    def feedback(self, outcome):
        return None


class IntentRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[StepIdentity, tuple[BusDecision, ...]]] = []

    def record_intent(self, identity, decisions):
        self.calls.append((identity, tuple(decisions)))


def test_phase_and_ablation_contract_is_explicit() -> None:
    assert CausalBus is OrganCausalBus
    assert Decision is BusDecision
    assert [phase.value for phase in Phase] == [
        "PRE_LOSS",
        "PRE_BACKWARD",
        "PRE_OPTIMIZER",
        "POST_STEP",
        "CYCLE_BOUNDARY",
    ]
    assert [arm.value for arm in AblationArm] == ["CONTROL", "SHADOW", "APPLY"]
    assert step().key == step().key
    assert step().key != step(optimizer_step=8).key


def test_intervention_rejects_an_inverted_validity_window() -> None:
    identity = step()

    with pytest.raises(ValueError, match="valid_through_step"):
        replace(
            valid_intervention(identity),
            valid_from_step=8,
            valid_through_step=7,
        )


def test_control_does_not_call_adapters() -> None:
    identity = step()
    adapter = RecordingAdapter(proposals=(valid_intervention(identity),))
    recorder = IntentRecorder()
    bus = OrganCausalBus(
        arm=AblationArm.CONTROL,
        adapters=(adapter,),
        intent_recorder=recorder,
    )

    decision = bus.decide(identity, Phase.PRE_LOSS, {"lm_loss": 1.25})
    bus.record_intent(identity, (decision,))

    assert decision.arm is AblationArm.CONTROL
    assert decision.accepted == ()
    assert decision.effective == ()
    assert len(recorder.calls) == 1


def test_shadow_computes_and_records_but_has_no_effective_interventions() -> None:
    identity = step()
    proposal = valid_intervention(identity)
    adapter = RecordingAdapter(proposals=(proposal,))
    recorder = IntentRecorder()
    bus = OrganCausalBus(
        arm=AblationArm.SHADOW,
        adapters=(adapter,),
        intent_recorder=recorder,
    )

    decision = bus.decide(identity, Phase.PRE_LOSS, {"lm_loss": 1.25})
    bus.record_intent(identity, (decision,))

    assert decision.accepted == (proposal,)
    assert decision.effective == ()
    assert not decision.blocked
    assert len(recorder.calls) == 1


def test_apply_returns_only_deterministically_ordered_decisions() -> None:
    identity = step()
    earlier = replace(
        valid_intervention(
            identity,
            intervention_id="a-earlier",
            source="a-organ:v1",
            priority=5,
        ),
    )
    bus = OrganCausalBus(
        arm=AblationArm.APPLY,
        adapters=(
            RecordingAdapter(adapter_id="a-organ", proposals=(earlier,)),
        ),
    )

    decision = bus.decide(identity, Phase.PRE_LOSS, {})

    assert decision.accepted == (earlier,)
    assert decision.effective == decision.accepted
    assert decision.context == {}
    assert not hasattr(decision, "model")
    assert not hasattr(decision, "optimizer")


def test_conflicting_interventions_are_all_rejected_fail_closed() -> None:
    identity = step()
    left = valid_intervention(
        identity, intervention_id="left", source="router-guard:v1"
    )
    right = valid_intervention(
        identity, intervention_id="right", source="router-guard:v1"
    )
    bus = OrganCausalBus(
        arm=AblationArm.APPLY,
        adapters=(RecordingAdapter(proposals=(left, right)),),
    )

    decision = bus.decide(identity, Phase.PRE_LOSS, {})

    assert decision.accepted == ()
    assert decision.effective == ()
    assert {item.item_id for item in decision.rejected} == {"left", "right"}
    assert {item.reason for item in decision.rejected} == {"conflict"}


def test_duplicate_intervention_ids_are_rejected_even_for_different_targets() -> None:
    identity = step()
    left = valid_intervention(identity, intervention_id="duplicate")
    right = replace(left)
    bus = OrganCausalBus(
        arm=AblationArm.APPLY,
        adapters=(RecordingAdapter(proposals=(left, right)),),
    )

    decision = bus.decide(identity, Phase.PRE_LOSS, {})

    assert decision.accepted == ()
    assert len(decision.rejected) == 2
    assert {item.reason for item in decision.rejected} == {
        "duplicate_intervention_id"
    }


@pytest.mark.parametrize(
    ("proposal", "reason"),
    [
        (
            lambda identity: replace(
                valid_intervention(identity),
                target="MODEL_OBJECT",
            ),
            "target_not_allowed",
        ),
        (
            lambda identity: replace(
                valid_intervention(identity),
                operation="CALLBACK",
            ),
            "operation_not_allowed",
        ),
        (
            lambda identity: replace(
                valid_intervention(identity),
                value=math.inf,
            ),
            "nonfinite_value",
        ),
        (
                lambda identity: replace(
                    valid_intervention(identity),
                    valid_from_step=identity.optimizer_step - 1,
                    valid_through_step=identity.optimizer_step - 1,
                ),
            "expired",
        ),
    ],
)
def test_invalid_interventions_are_rejected(proposal, reason) -> None:
    identity = step()
    item = proposal(identity)
    bus = OrganCausalBus(
        arm=AblationArm.APPLY,
        adapters=(RecordingAdapter(proposals=(item,)),),
    )

    decision = bus.decide(identity, Phase.PRE_LOSS, {})

    assert decision.accepted == ()
    assert decision.effective == ()
    assert len(decision.rejected) == 1
    assert decision.rejected[0].reason == reason


def test_nonfinite_signal_blocks_proposal_generation() -> None:
    identity = step()
    adapter = RecordingAdapter(
        signal_value=float("nan"),
        proposals=(valid_intervention(identity),),
    )
    bus = OrganCausalBus(arm=AblationArm.APPLY, adapters=(adapter,))

    decision = bus.decide(identity, Phase.PRE_LOSS, {})

    assert decision.blocked
    assert decision.accepted == ()
    assert decision.effective == ()
    assert decision.rejected[0].reason == "nonfinite_signal"


def test_context_rejects_mutable_runtime_objects_and_is_frozen_for_adapters() -> None:
    identity = step()
    adapter = RecordingAdapter()
    bus = OrganCausalBus(arm=AblationArm.SHADOW, adapters=(adapter,))

    bus.decide(identity, Phase.PRE_LOSS, {"nested": {"value": 1.0}})

    with pytest.raises(TypeError, match="JSON-compatible"):
        bus.decide(identity, Phase.PRE_LOSS, {"model": object()})


def test_feedback_runs_only_outside_control() -> None:
    identity = step()
    outcome = StepOutcome(
        identity=identity,
        optimizer_step_applied=True,
        accepted_intervention_ids=("aux-scale",),
        raw_losses={"lm": 1.25, "aux": 0.02},
        effective_losses={"lm": 1.25, "aux": 0.0},
        gradient_norm_before=2.0,
        gradient_norm_after=1.0,
        parameter_delta_norm=0.01,
    )
    control_adapter = RecordingAdapter()
    shadow_adapter = RecordingAdapter()
    control = OrganCausalBus(
        arm=AblationArm.CONTROL,
        adapters=(control_adapter,),
    )
    shadow = OrganCausalBus(
        arm=AblationArm.SHADOW,
        adapters=(shadow_adapter,),
    )

    assert control.feedback(outcome) == ()
    assert shadow.feedback(outcome) == ()


def test_shadow_restores_all_rng_streams_around_adapter_calls() -> None:
    class RandomAdapter:
        adapter_id = "random"
        version = "v1"

        def observe(self, identity, phase, context):
            random.random()
            np.random.random()
            torch.rand(())
            return ()

        def propose(self, identity, phase, signals, context):
            random.random()
            np.random.random()
            torch.rand(())
            return ()

        def feedback(self, outcome):
            return None

    random.seed(51)
    np.random.seed(51)
    torch.manual_seed(51)
    expected = (random.random(), np.random.random(), torch.rand(()))
    random.seed(51)
    np.random.seed(51)
    torch.manual_seed(51)

    OrganCausalBus(
        arm=AblationArm.SHADOW, adapters=(RandomAdapter(),)
    ).decide(step(), Phase.PRE_LOSS, {})
    actual = (random.random(), np.random.random(), torch.rand(()))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_adapter_state_mutation_is_restored_and_blocks_decision() -> None:
    class MutatingAdapter:
        adapter_id = "mutating"
        version = "v1"

        def __init__(self):
            self.counter = 0

        def observe(self, identity, phase, context):
            self.counter += 1
            return ()

        def propose(self, identity, phase, signals, context):
            return ()

        def feedback(self, outcome):
            return None

    adapter = MutatingAdapter()
    decision = OrganCausalBus(
        arm=AblationArm.SHADOW, adapters=(adapter,)
    ).decide(step(), Phase.PRE_LOSS, {})

    assert decision.blocked
    assert decision.rejected[0].reason == "adapter_state_mutation"
    assert adapter.counter == 0


@pytest.mark.parametrize("failure_point", ["observe", "propose", "feedback"])
def test_shadow_restores_dict_slots_and_rng_when_adapter_raises(
    failure_point: str,
) -> None:
    class SlottedBase:
        __slots__ = ("base_state",)

        def __init__(self):
            self.base_state = ["base"]

    class RaisingSlottedAdapter(SlottedBase):
        adapter_id = "raising-slotted"
        version = "v1"

        def __init__(self, point):
            super().__init__()
            self.dict_state = {"value": 1}
            self.point = point

        def _touch(self, point):
            if self.point != point:
                return
            self.base_state.append(point)
            self.dict_state["value"] = 2
            random.random()
            np.random.random()
            torch.rand(())
            raise RuntimeError(point)

        def observe(self, identity, phase, context):
            self._touch("observe")
            return ()

        def propose(self, identity, phase, signals, context):
            self._touch("propose")
            return ()

        def feedback(self, outcome):
            self._touch("feedback")

    random.seed(73)
    np.random.seed(73)
    torch.manual_seed(73)
    expected = (random.random(), np.random.random(), torch.rand(()))
    random.seed(73)
    np.random.seed(73)
    torch.manual_seed(73)
    adapter = RaisingSlottedAdapter(failure_point)
    bus = OrganCausalBus(
        arm=AblationArm.SHADOW, adapters=(adapter,)
    )

    if failure_point == "feedback":
        decision = bus.decide(step(), Phase.PRE_LOSS, {})
        assert not decision.blocked
        rejected = bus.feedback(
            StepOutcome(
                identity=step(),
                optimizer_step_applied=False,
            )
        )
        assert rejected[0].reason == "adapter_feedback_error:RuntimeError"
    else:
        decision = bus.decide(step(), Phase.PRE_LOSS, {})
        assert decision.blocked
        assert failure_point in decision.rejected[0].reason

    assert adapter.base_state == ["base"]
    assert adapter.dict_state == {"value": 1}
    actual = (random.random(), np.random.random(), torch.rand(()))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


@pytest.mark.parametrize("failure_point", ["observe", "propose", "feedback"])
def test_apply_exception_restores_dict_slots_and_all_rng(
    failure_point: str,
) -> None:
    class ApplyBase:
        __slots__ = ("slot_state",)

        def __init__(self):
            self.slot_state = ["clean"]

    class ApplyRaisingAdapter(ApplyBase):
        adapter_id = "apply-raising"
        version = "v1"

        def __init__(self, point):
            super().__init__()
            self.dict_state = {"clean": True}
            self.point = point

        def _fail(self, point):
            if self.point != point:
                return
            self.slot_state.append("dirty")
            self.dict_state["clean"] = False
            random.random()
            np.random.random()
            torch.rand(())
            raise RuntimeError(point)

        def observe(self, identity, phase, context):
            self._fail("observe")
            return ()

        def propose(self, identity, phase, signals, context):
            self._fail("propose")
            return ()

        def feedback(self, outcome):
            self._fail("feedback")

    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    expected = (random.random(), np.random.random(), torch.rand(()))
    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    adapter = ApplyRaisingAdapter(failure_point)
    bus = OrganCausalBus(arm=AblationArm.APPLY, adapters=(adapter,))

    decision = bus.decide(step(), Phase.PRE_LOSS, {})
    if failure_point == "feedback":
        assert not decision.blocked
        rejected = bus.feedback(
            StepOutcome(
                identity=step(),
                optimizer_step_applied=False,
            )
        )
        assert rejected[0].reason == "adapter_feedback_error:RuntimeError"
    else:
        assert decision.blocked
        assert failure_point in decision.rejected[0].reason

    assert adapter.slot_state == ["clean"]
    assert adapter.dict_state == {"clean": True}
    actual = (random.random(), np.random.random(), torch.rand(()))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_apply_feedback_may_persist_state_but_never_rng_side_effects() -> None:
    class StatefulFeedbackAdapter:
        __slots__ = ("feedback_count",)
        adapter_id = "stateful-feedback"
        version = "v1"

        def __init__(self):
            self.feedback_count = 0

        def observe(self, identity, phase, context):
            return ()

        def propose(self, identity, phase, signals, context):
            return ()

        def feedback(self, outcome):
            self.feedback_count += 1
            random.random()
            np.random.random()
            torch.rand(())

    random.seed(92)
    np.random.seed(92)
    torch.manual_seed(92)
    expected = (random.random(), np.random.random(), torch.rand(()))
    random.seed(92)
    np.random.seed(92)
    torch.manual_seed(92)
    adapter = StatefulFeedbackAdapter()
    bus = OrganCausalBus(arm=AblationArm.APPLY, adapters=(adapter,))
    identity = step()

    bus.decide(identity, Phase.PRE_LOSS, {})
    assert bus.feedback(
        StepOutcome(identity=identity, optimizer_step_applied=False)
    ) == ()

    assert adapter.feedback_count == 1
    actual = (random.random(), np.random.random(), torch.rand(()))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


@pytest.mark.parametrize(
    ("replacement", "reason"),
    [
        ({"subject": "../all"}, "subject_not_allowed"),
        ({"value": -1.0}, "invalid_operation_value"),
        ({"evidence_ids": ("missing",)}, "unknown_evidence_id"),
        ({"source": "forged:v1"}, "intervention_source_mismatch"),
    ],
)
def test_operation_domain_evidence_and_source_are_validated(
    replacement, reason
) -> None:
    identity = step()
    proposal = replace(valid_intervention(identity), **replacement)
    decision = OrganCausalBus(
        arm=AblationArm.APPLY,
        adapters=(RecordingAdapter(proposals=(proposal,)),),
    ).decide(identity, Phase.PRE_LOSS, {})

    assert decision.accepted == ()
    assert decision.rejected[0].reason == reason


@pytest.mark.parametrize("subject", ["aux", "ghost", "jepa", "spider"])
def test_loss_scale_accepts_only_the_four_exact_subjects(subject: str) -> None:
    identity = step()
    proposal = replace(
        valid_intervention(identity),
        intervention_id=f"{subject}-scale",
        subject=subject,
        value=0.25,
    )
    decision = OrganCausalBus(
        arm=AblationArm.APPLY,
        adapters=(RecordingAdapter(proposals=(proposal,)),),
    ).decide(identity, Phase.PRE_LOSS, {})

    assert decision.accepted == (proposal,)
    assert decision.rejected == ()


@pytest.mark.parametrize(
    ("subject", "value", "reason"),
    [
        ("unknown", 0.25, "subject_not_allowed"),
        ("*", 0.25, "subject_not_allowed"),
        ("ghost.*", 0.25, "subject_not_allowed"),
        ("ghost", -0.1, "invalid_operation_value"),
        ("ghost", True, "invalid_operation_value"),
        ("ghost", float("inf"), "nonfinite_value"),
        ("ghost", float("nan"), "nonfinite_value"),
    ],
)
def test_loss_scale_rejects_unknown_wildcard_or_invalid_values(
    subject: str,
    value: object,
    reason: str,
) -> None:
    identity = step()
    proposal = replace(
        valid_intervention(identity),
        intervention_id="invalid-loss-scale",
        subject=subject,
        value=value,
    )
    decision = OrganCausalBus(
        arm=AblationArm.APPLY,
        adapters=(RecordingAdapter(proposals=(proposal,)),),
    ).decide(identity, Phase.PRE_LOSS, {})

    assert decision.accepted == ()
    assert decision.rejected[0].reason == reason
