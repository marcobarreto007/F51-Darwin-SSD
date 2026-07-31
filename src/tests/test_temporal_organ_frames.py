from dataclasses import replace

import pytest
import torch

from f51_darwin.organism.causal_bus import StepIdentity
from f51_darwin.organism.organ_frames import OrganObservationFrame


def identity(step: int) -> StepIdentity:
    return StepIdentity(
        run_id="run-v2",
        cycle=3,
        optimizer_step=step,
        accumulation_window=0,
        batch_digest=f"batch-{step}",
        rng_digest=f"rng-{step}",
        base_checkpoint_id="base-v2",
        training_contract_id="contract-v2",
    )


def test_frame_is_valid_only_after_source_step() -> None:
    frame = OrganObservationFrame.create(
        source=identity(10),
        signals={"fresh_lm": 2.5, "heldout_lm": 2.7},
        valid_from_step=11,
        valid_through_step=11,
    )
    with pytest.raises(ValueError, match="temporal_barrier"):
        frame.validate_for(identity(10))
    frame.validate_for(identity(11))
    with pytest.raises(ValueError, match="expired"):
        frame.validate_for(identity(12))


def test_frame_rejects_nonfinite_and_tamper() -> None:
    with pytest.raises(ValueError, match="finite"):
        OrganObservationFrame.create(
            source=identity(1),
            signals={"fresh_lm": float("nan")},
            valid_from_step=2,
            valid_through_step=2,
        )
    frame = OrganObservationFrame.create(
        source=identity(1),
        signals={"fresh_lm": 3.0},
        valid_from_step=2,
        valid_through_step=2,
    )
    state = frame.to_state()
    state["signals"]["fresh_lm"] = 1.0
    with pytest.raises(ValueError, match="digest"):
        OrganObservationFrame.from_state(state)


def test_frame_roundtrip_is_exact_and_immutable() -> None:
    frame = OrganObservationFrame.create(
        source=identity(7),
        signals={"per_sample_novelty": (0.1, 0.9)},
        valid_from_step=8,
        valid_through_step=9,
    )
    restored = OrganObservationFrame.from_state(frame.to_state())
    assert restored == frame
    with pytest.raises(TypeError):
        restored.signals["x"] = 1


def test_step_identity_state_roundtrip_uses_every_field() -> None:
    source = replace(
        identity(7),
        ablation_plan_id="plan-v2",
        attempt_id="attempt-v2",
    )

    assert StepIdentity.from_state(source.to_state()) == source
    assert set(source.to_state()) == {
        "run_id",
        "cycle",
        "optimizer_step",
        "accumulation_window",
        "batch_digest",
        "rng_digest",
        "base_checkpoint_id",
        "training_contract_id",
        "ablation_plan_id",
        "attempt_id",
    }


@pytest.mark.parametrize(
    "signals",
    [
        {"boolean": True},
        {"nested": {"boolean": False}},
        {"tensor": torch.tensor(1.0)},
        {"object": object()},
        {1: 2.0},
    ],
)
def test_frame_rejects_noncanonical_evidence(signals: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        OrganObservationFrame.create(
            source=identity(1),
            signals=signals,
            valid_from_step=2,
            valid_through_step=2,
        )


def test_frame_freezes_nested_containers_and_thaws_independent_state() -> None:
    nested = {"series": [1.0, {"score": 2.0}]}
    frame = OrganObservationFrame.create(
        source=identity(1),
        signals=nested,
        valid_from_step=2,
        valid_through_step=3,
    )
    nested["series"][1]["score"] = 99.0

    assert frame.signals["series"][1]["score"] == 2.0
    state = frame.to_state()
    state["signals"]["series"][1]["score"] = 88.0
    assert frame.signals["series"][1]["score"] == 2.0


def test_frame_rejects_invalid_window_and_training_contract() -> None:
    with pytest.raises(ValueError, match="temporal_barrier"):
        OrganObservationFrame.create(
            source=identity(2),
            signals={"fresh_lm": 1.0},
            valid_from_step=2,
            valid_through_step=3,
        )
    with pytest.raises(ValueError, match="valid_through_step"):
        OrganObservationFrame.create(
            source=identity(2),
            signals={"fresh_lm": 1.0},
            valid_from_step=4,
            valid_through_step=3,
        )
    frame = OrganObservationFrame.create(
        source=identity(2),
        signals={"fresh_lm": 1.0},
        valid_from_step=3,
        valid_through_step=3,
    )
    with pytest.raises(ValueError, match="training_contract_mismatch"):
        frame.validate_for(
            replace(identity(3), training_contract_id="other-contract")
        )
