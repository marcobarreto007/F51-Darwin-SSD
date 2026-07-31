from __future__ import annotations

import copy
import random

import pytest

from f51_darwin.replay_buffer import ReplayBuffer, ReplayExample


def test_priority_is_clamped_and_only_attached_to_future_rows() -> None:
    buffer = ReplayBuffer(capacity=8, seed=51)
    buffer.add([1], label="before")
    already_sampled = buffer.sample(1)

    buffer.add([2], label="low", priority=-10.0)
    buffer.add([3], label="high", priority=10.0)

    assert already_sampled == [
        ReplayExample(input_ids=(1,), label="before", priority=1.0)
    ]
    assert [record["priority"] for record in buffer.to_records()] == [
        1.0,
        0.5,
        2.0,
    ]


def test_priority_changes_weighted_future_sampling() -> None:
    buffer = ReplayBuffer(capacity=4, seed=7)
    buffer.add([1], label="low", priority=0.5)
    buffer.add([2], label="high", priority=2.0)

    counts = {"low": 0, "high": 0}
    for _ in range(2_000):
        counts[buffer.sample(1)[0].label] += 1

    assert counts["high"] > counts["low"] * 3


def test_priority_and_private_rng_survive_state_roundtrip() -> None:
    source = ReplayBuffer(capacity=4, seed=51)
    for index, priority in enumerate((0.5, 0.75, 1.5, 2.0)):
        source.add(
            [index, index + 1],
            label=f"sample-{index}",
            priority=priority,
        )
    source.sample(2)
    state = copy.deepcopy(source.state_dict())

    restored = ReplayBuffer(capacity=4, seed=999)
    restored.load_state_dict(state)

    assert restored.state_dict() == state
    assert restored.sample(3) == source.sample(3)


def test_equal_priorities_use_exact_legacy_random_sample_path() -> None:
    seed = 93
    records = [
        ReplayExample((index,), label=f"sample-{index}")
        for index in range(6)
    ]
    expected_rng = random.Random(seed)
    expected = expected_rng.sample(records, 4)
    buffer = ReplayBuffer(capacity=8, seed=seed)
    for record in records:
        buffer.add(record.input_ids, label=record.label, priority=1.0)

    assert buffer.sample(4) == expected


def test_replay_sampling_does_not_consume_global_rng() -> None:
    random.seed(777)
    expected = random.Random(777).getstate()
    buffer = ReplayBuffer(capacity=4, seed=51)
    buffer.add([1], priority=0.5)
    buffer.add([2], priority=2.0)

    buffer.sample(1)

    assert random.getstate() == expected


def test_legacy_records_restore_with_unit_priority() -> None:
    buffer = ReplayBuffer(capacity=4, seed=51)
    buffer.load_records(
        [
            {"input_ids": [1, 2], "label": "old"},
            {"ids": [3, 4]},
        ]
    )

    assert buffer.to_records() == [
        {"input_ids": [1, 2], "label": "old", "priority": 1.0},
        {"input_ids": [3, 4], "label": "ancestral", "priority": 1.0},
    ]


@pytest.mark.parametrize("priority", [float("nan"), float("inf"), True, "1"])
def test_replay_rejects_invalid_priority(priority: object) -> None:
    buffer = ReplayBuffer(capacity=4, seed=51)
    with pytest.raises((TypeError, ValueError), match="priority"):
        buffer.add([1, 2], priority=priority)
    assert len(buffer) == 0
