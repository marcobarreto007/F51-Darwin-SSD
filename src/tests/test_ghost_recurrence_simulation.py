from research.simulate_ghost_recurrence import (
    GhostRecurrenceMemory,
    run_scenarios,
)
import torch

from f51_darwin.heartbeat import TestTimeMemory as HeartbeatTestTimeMemory


def test_proposed_ghost_lifecycle_passes_all_ten_scenarios() -> None:
    results = run_scenarios()

    assert len(results) == 10
    assert all(result.proposed_passed for result in results)


def test_current_heartbeat_semantics_are_not_mislabeled_as_consolidation() -> None:
    results = run_scenarios()

    assert sum(result.baseline_passed for result in results) < len(results)
    exact = next(result for result in results if result.scenario == "02_exact_recurrence")
    assert exact.baseline_state["slots"] == 3
    assert exact.baseline_state["real"] == 0


def test_checkpoint_state_preserves_transient_evidence_before_promotion() -> None:
    checkpoint = next(
        result for result in run_scenarios()
        if result.scenario == "10_checkpoint_roundtrip"
    )

    assert checkpoint.proposed_state["slots"] == 1
    assert checkpoint.proposed_state["real"] == 1
    assert checkpoint.proposed_state["recurrences"] == [3]


def test_serialized_state_contains_no_model_weights_or_training_side_effects() -> None:
    state = GhostRecurrenceMemory().dumps()

    assert "optimizer" not in state
    assert "model_state_dict" not in state


def test_baseline_matches_live_heartbeat_duplicate_slot_behavior() -> None:
    torch.manual_seed(51)
    memory = HeartbeatTestTimeMemory(d_model=8, capacity=8)
    hidden = torch.randn(1, 3, 8)

    for _ in range(3):
        assert memory.write_if_surprised(hidden, jepa_error=0.9, domain="papers")

    assert len(memory.slots) == 3
    assert memory.retrieve(hidden) is not None
    assert sum(slot.access_count for slot in memory.slots) > 0
