from __future__ import annotations

import copy

import pytest

from f51_darwin.cognition.contracts import (
    CognitiveForwardMetadata,
    CognitivePulseEvent,
)
from f51_darwin.cognition.pulse import CognitivePulse


def _event(step: int) -> CognitivePulseEvent:
    return CognitivePulseEvent.shadow(
        CognitiveForwardMetadata(
            step_id=step,
            checkpoint_id="sha256:" + "a" * 64,
            context_digest="sha256:" + f"{step:064x}",
        )
    )


def test_pulse_is_append_only_and_hash_chained() -> None:
    pulse = CognitivePulse()
    first = pulse.append(_event(0))
    second = pulse.append(_event(1))
    assert first.previous_sha256 is None
    assert second.previous_sha256 == first.sha256
    assert pulse.head_sha256 == second.sha256

    restored = CognitivePulse.from_state_dict(pulse.state_dict())
    assert restored.state_dict() == pulse.state_dict()


def test_pulse_rejects_duplicate_or_reordered_steps() -> None:
    pulse = CognitivePulse()
    pulse.append(_event(2))
    with pytest.raises(ValueError, match="strictly increase"):
        pulse.append(_event(2))


def test_pulse_rejects_tampered_state() -> None:
    pulse = CognitivePulse()
    pulse.append(_event(0))
    payload = copy.deepcopy(pulse.state_dict())
    payload["records"][0]["event"]["context_digest"] = "sha256:" + "f" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        CognitivePulse.from_state_dict(payload)
