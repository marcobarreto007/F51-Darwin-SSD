from __future__ import annotations

import torch

from f51_darwin.cognition import (
    CANONICAL_ORGAN_IDS,
    CognitiveForwardMetadata,
    CognitiveRuntime,
)


def _metadata(step: int = 0) -> CognitiveForwardMetadata:
    return CognitiveForwardMetadata(
        step_id=step,
        checkpoint_id="sha256:" + "a" * 64,
        context_digest="sha256:" + "b" * 64,
    )


def test_shadow_runtime_executes_paths_without_changing_hidden() -> None:
    torch.manual_seed(51)
    runtime = CognitiveRuntime(d_model=16, max_scale=0.15)
    hidden = torch.randn(2, 4, 16)
    result, record = runtime.observe_shadow(hidden, _metadata())
    assert torch.equal(result, hidden)
    assert record.event.mode == "shadow"
    assert runtime.manifest()["organ_ids"] == list(CANONICAL_ORGAN_IDS)
    assert runtime.memory_adapter.gate.item() == 0.0
    assert runtime.world_model_adapter.gate.item() == 0.0


def test_shadow_runtime_pulse_steps_must_increase() -> None:
    runtime = CognitiveRuntime(d_model=16, max_scale=0.15)
    hidden = torch.randn(1, 2, 16)
    runtime.observe_shadow(hidden, _metadata(1))
    try:
        runtime.observe_shadow(hidden, _metadata(1))
    except ValueError as error:
        assert "strictly increase" in str(error)
    else:
        raise AssertionError("duplicate pulse step was accepted")
