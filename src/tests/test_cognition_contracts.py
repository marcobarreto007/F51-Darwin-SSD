from __future__ import annotations

import pytest
import torch

from f51_darwin.cognition.contracts import (
    CANONICAL_ORGAN_IDS,
    CognitiveForwardMetadata,
    CognitivePulseEvent,
    OrganKind,
    ResidualCondition,
)


def test_exactly_three_canonical_organs() -> None:
    assert CANONICAL_ORGAN_IDS == (
        "organ:memory:v1",
        "organ:world_model:v1",
        "organ:executive:v1",
    )
    assert tuple(item.value for item in OrganKind) == CANONICAL_ORGAN_IDS


def test_residual_condition_is_shape_strict() -> None:
    condition = ResidualCondition(
        condition_id="memory-001",
        organ=OrganKind.MEMORY,
        values=torch.zeros(2, 3, 512),
        positions=torch.tensor([[0, 2, 4], [1, 3, 5]]),
    )
    assert condition.batch_size == 2
    assert condition.count == 3

    with pytest.raises(ValueError, match="organ_width=512"):
        ResidualCondition(
            condition_id="bad",
            organ=OrganKind.MEMORY,
            values=torch.zeros(1, 1, 16),
            positions=torch.zeros(1, 1, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="Executive"):
        ResidualCondition(
            condition_id="bad",
            organ=OrganKind.EXECUTIVE,
            values=torch.zeros(1, 1, 512),
            positions=torch.zeros(1, 1, dtype=torch.long),
        )


def test_metadata_and_pulse_reject_invalid_identity() -> None:
    metadata = CognitiveForwardMetadata(
        step_id=7,
        checkpoint_id="sha256:" + "a" * 64,
        context_digest="sha256:" + "b" * 64,
    )
    event = CognitivePulseEvent.shadow(metadata)
    assert event.mode == "shadow"
    assert event.selected_memory_ids == ()
    assert event.candidate_ids == ()

    with pytest.raises(ValueError, match="step_id"):
        CognitiveForwardMetadata(
            step_id=-1,
            checkpoint_id="sha256:" + "a" * 64,
            context_digest="sha256:" + "b" * 64,
        )
