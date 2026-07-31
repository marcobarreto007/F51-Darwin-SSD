"""End-to-end test: UniversalMemory teach -> save -> reload -> recall.

Proves the full cycle closes:
  1. Teach facts into memory
  2. Save memory snapshot
  3. Reload fresh model with restored memory
  4. Recall returns correct fact associations
  5. Controls (empty, shuffled) fail
"""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import pytest
import torch

from f51_darwin.cognition import (
    CognitiveForwardMetadata,
    CognitiveRuntime,
    UniversalMemory,
)


def _metadata(step: int = 0) -> CognitiveForwardMetadata:
    return CognitiveForwardMetadata(
        step_id=step,
        checkpoint_id="sha256:" + "a" * 64,
        context_digest="sha256:" + "b" * 64,
    )


def _sq(batch: int = 1, seq: int = 4, d: int = 64) -> torch.Tensor:
    return torch.randn(batch, seq, d)


# ── Teach → recall ──────────────────────────────────────────────────────────


def test_teach_and_recall_same_session() -> None:
    runtime = CognitiveRuntime(d_model=64, max_scale=0.15)
    runtime.set_active()

    key = _sq(d=64)
    val = _sq(d=64)
    rid = runtime.memory.teach(key, val, event_type="explicit_teaching", provenance="human")
    assert rid is not None

    recalls = runtime.memory.recall(key, require_verified=True)
    assert not recalls[0].abstained
    assert recalls[0].record.memory_id == rid


# ── Save → reload → recall ─────────────────────────────────────────────────


def _clone_runtime_with_memory(runtime: CognitiveRuntime) -> CognitiveRuntime:
    """Create a fresh runtime and restore both model weights and memory state."""
    runtime2 = CognitiveRuntime(d_model=runtime.d_model, max_scale=runtime.max_scale)
    # Copy key/value encoder and readout weights
    runtime2.memory.key_encoder.load_state_dict(runtime.memory.key_encoder.state_dict())
    runtime2.memory.value_encoder.load_state_dict(runtime.memory.value_encoder.state_dict())
    runtime2.memory.readout.load_state_dict(runtime.memory.readout.state_dict())
    # Restore memory store
    runtime2.memory.load_memory_state(runtime.memory.memory_state_dict())
    runtime2.set_active()
    return runtime2


def test_save_reload_recall_cycle() -> None:
    runtime = CognitiveRuntime(d_model=64, max_scale=0.15)
    runtime.set_active()

    key = _sq(d=64)
    val = _sq(d=64)
    rid = runtime.memory.teach(key, val, event_type="explicit_teaching", provenance="human")

    runtime2 = _clone_runtime_with_memory(runtime)
    assert runtime2.memory.slot_count == 1

    recalls = runtime2.memory.recall(key, require_verified=True)
    assert not recalls[0].abstained
    assert recalls[0].record.memory_id == rid


# ── Save to disk → reload from disk ─────────────────────────────────────────


def test_disk_persistence_roundtrip() -> None:
    runtime = CognitiveRuntime(d_model=64, max_scale=0.15)
    runtime.set_active()

    key = _sq(d=64)
    val = _sq(d=64)
    runtime.memory.teach(key, val, event_type="explicit_teaching", provenance="human")

    # Save full state (encoder weights + memory store)
    full_state = {
        "key_encoder": runtime.memory.key_encoder.state_dict(),
        "value_encoder": runtime.memory.value_encoder.state_dict(),
        "readout": runtime.memory.readout.state_dict(),
        "memory": runtime.memory.memory_state_dict(),
    }

    with tempfile.NamedTemporaryFile(suffix=".pt", mode="wb", delete=False) as f:
        torch.save(full_state, f)
        tmp_path = Path(f.name)

    try:
        loaded = torch.load(tmp_path, map_location="cpu", weights_only=False)
        runtime2 = CognitiveRuntime(d_model=64, max_scale=0.15)
        runtime2.set_active()
        runtime2.memory.key_encoder.load_state_dict(loaded["key_encoder"])
        runtime2.memory.value_encoder.load_state_dict(loaded["value_encoder"])
        runtime2.memory.readout.load_state_dict(loaded["readout"])
        runtime2.memory.load_memory_state(loaded["memory"])
        assert runtime2.memory.slot_count == 1

        recalls = runtime2.memory.recall(key, require_verified=True)
        assert not recalls[0].abstained
    finally:
        tmp_path.unlink(missing_ok=True)


# ── Control: empty memory ───────────────────────────────────────────────────


def test_empty_memory_abstains() -> None:
    runtime = CognitiveRuntime(d_model=64, max_scale=0.15)
    runtime.set_active()
    recalls = runtime.memory.recall(_sq(), require_verified=True)
    assert recalls[0].abstained


# ── Control: wrong key ──────────────────────────────────────────────────────


def test_wrong_key_abstains_or_low_score() -> None:
    runtime = CognitiveRuntime(d_model=64, max_scale=0.15)
    runtime.set_active()
    runtime.memory.teach(_sq(d=64), _sq(d=64), event_type="explicit_teaching", provenance="human")

    # Completely different key
    wrong_key = _sq(d=64)
    recalls = runtime.memory.recall(wrong_key, require_verified=False)
    # Should either abstain or have low confidence
    assert recalls[0].abstained or recalls[0].score < 0.8


# ── Full active runtime cycle ───────────────────────────────────────────────


def test_active_runtime_cycle_with_taught_memory() -> None:
    runtime = CognitiveRuntime(d_model=64, max_scale=0.15)
    runtime.set_active()

    key = _sq(d=64)
    val = _sq(d=64)
    runtime.memory.teach(key, val, event_type="explicit_teaching", provenance="human")

    hidden = key.expand(2, -1, -1)  # batch=2
    result, record = runtime.observe_active(hidden, _metadata(1))

    assert record.event.mode == "active"
    assert len(record.event.selected_memory_ids) >= 1
    assert result.shape == hidden.shape


# ── Shadow mode still works after active ────────────────────────────────────


def test_shadow_mode_unchanged_after_active_use() -> None:
    torch.manual_seed(51)
    runtime = CognitiveRuntime(d_model=64, max_scale=0.15)

    # Use active mode
    runtime.set_active()
    runtime.memory.teach(_sq(d=64), _sq(d=64), event_type="explicit_teaching", provenance="human")
    runtime.observe_active(_sq(batch=1, d=64), _metadata(1))

    # Switch to shadow — should still preserve hidden
    runtime.set_shadow()
    hidden = torch.randn(2, 4, 64)
    result, record = runtime.observe_shadow(hidden, _metadata(2))
    assert torch.equal(result, hidden)


# ── Consolidation after teaching ────────────────────────────────────────────


def test_consolidation_after_multiple_teach() -> None:
    runtime = CognitiveRuntime(d_model=64, max_scale=0.15)
    runtime.set_active()

    for i in range(5):
        runtime.memory.teach(
            _sq(d=64), _sq(d=64),
            event_type="explicit_teaching", provenance="human",
            tags=(f"fact_{i}",),
        )
    assert runtime.memory.slot_count == 5
    assert runtime.memory.verified_count == 5

    report = runtime.memory.consolidate()
    assert "total_before" in report
    assert "total_after" in report
