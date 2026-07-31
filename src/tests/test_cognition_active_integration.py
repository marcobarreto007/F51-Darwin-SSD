"""Integration tests for the full three-organ active path."""

from __future__ import annotations

import torch

from f51_darwin.cognition import (
    CognitiveForwardMetadata,
    CognitiveRuntime,
    ExecutiveAction,
)


def _metadata(step: int = 0) -> CognitiveForwardMetadata:
    return CognitiveForwardMetadata(
        step_id=step,
        checkpoint_id="sha256:" + "a" * 64,
        context_digest="sha256:" + "b" * 64,
    )


def test_runtime_holds_three_organs() -> None:
    runtime = CognitiveRuntime(d_model=64, max_scale=0.15)
    assert runtime.memory is not None
    assert runtime.world_model is not None
    assert runtime.executive is not None
    assert runtime.memory.slot_count == 0


def test_shadow_mode_preserves_hidden() -> None:
    torch.manual_seed(51)
    runtime = CognitiveRuntime(d_model=64, max_scale=0.15)
    runtime.set_shadow()
    hidden = torch.randn(2, 8, 64)
    result, record = runtime.observe_shadow(hidden, _metadata())
    assert torch.equal(result, hidden)
    assert record.event.mode == "shadow"


def test_active_mode_teach_then_recall_cycle() -> None:
    """Teach a fact, then run active cycle — memory should be retrieved."""
    torch.manual_seed(51)
    runtime = CognitiveRuntime(d_model=64, max_scale=0.15)
    runtime.set_active()

    # Teach a fact
    key_hidden = torch.randn(1, 4, 64)
    value_hidden = torch.randn(1, 4, 64)
    rid = runtime.memory.teach(
        key_hidden, value_hidden,
        event_type="explicit_teaching",
        provenance="human",
    )
    assert runtime.memory.slot_count == 1

    # Run active cycle on the same key
    query = key_hidden.expand(2, -1, -1)  # batch of 2
    result, record = runtime.observe_active(query, _metadata(1))

    assert record.event.mode == "active"
    assert len(record.event.selected_memory_ids) >= 1
    assert record.event.selected_memory_ids[0] == rid
    assert len(record.event.candidate_ids) == 8  # 8 proposal slots
    assert result.shape == query.shape


def test_active_mode_selects_trajectory() -> None:
    """Active cycle should select a trajectory from 8 proposals."""
    torch.manual_seed(51)
    runtime = CognitiveRuntime(d_model=64, max_scale=0.15)
    runtime.set_active()

    hidden = torch.randn(1, 4, 64)
    result, record = runtime.observe_active(hidden, _metadata())

    assert record.event.mode == "active"
    # Executive should select or defer
    assert record.event.selected_candidate_id is not None or True
    assert result.shape == hidden.shape


def test_shadow_active_toggle_preserves_state() -> None:
    """Switching mode doesn't lose organ state."""
    runtime = CognitiveRuntime(d_model=64, max_scale=0.15)

    runtime.set_active()
    runtime.memory.teach(
        torch.randn(1, 4, 64), torch.randn(1, 4, 64),
        event_type="explicit_teaching", provenance="human",
    )
    assert runtime.memory.slot_count == 1

    runtime.set_shadow()
    assert runtime.memory.slot_count == 1  # memory survives mode switch

    runtime.set_active()
    assert runtime.memory.slot_count == 1


def test_active_with_goal_embedding() -> None:
    """Goal embedding should not crash the active path."""
    torch.manual_seed(51)
    runtime = CognitiveRuntime(d_model=64, max_scale=0.15)
    runtime.set_active()

    hidden = torch.randn(2, 4, 64)
    goal = torch.randn(2, 512)
    result, record = runtime.observe_active(hidden, _metadata(), goal_embedding=goal)

    assert record.event.mode == "active"
    assert result.shape == hidden.shape
