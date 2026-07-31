from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class ReplayExample:
    input_ids: tuple[int, ...]
    label: str = "ancestral"
    priority: float = 1.0


class ReplayBuffer:
    def __init__(self, capacity: int = 1024, seed: int = 51) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = capacity
        self._items: deque[ReplayExample] = deque(maxlen=capacity)
        self._rng = random.Random(seed)
        self._seen_count = 0

    def __len__(self) -> int:
        return len(self._items)

    def add(
        self,
        input_ids: Iterable[int],
        label: str = "ancestral",
        priority: float = 1.0,
    ) -> None:
        """Keep a bounded reservoir so old examples survive long training runs."""
        example = ReplayExample(
            tuple(int(token) for token in input_ids),
            label,
            _bounded_priority(priority),
        )
        self._seen_count += 1
        if len(self._items) < self.capacity:
            self._items.append(example)
            return

        replacement = self._rng.randrange(self._seen_count)
        if replacement < self.capacity:
            self._items[replacement] = example

    def sample(self, size: int) -> list[ReplayExample]:
        if size <= 0 or not self._items:
            return []
        size = min(size, len(self._items))
        items = list(self._items)
        if all(example.priority == 1.0 for example in items):
            return self._rng.sample(items, size)
        weighted = [
            (
                self._rng.random() ** (1.0 / example.priority),
                index,
                example,
            )
            for index, example in enumerate(items)
        ]
        weighted.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        return [example for _, _, example in weighted[:size]]

    def mix_with_new(self, new_examples: list[ReplayExample], replay_size: int) -> list[ReplayExample]:
        return list(new_examples) + self.sample(replay_size)

    def to_records(self) -> list[dict[str, object]]:
        return [
            {
                "input_ids": list(example.input_ids),
                "label": example.label,
                "priority": example.priority,
            }
            for example in self._items
        ]

    def load_records(self, records: list[dict[str, object]]) -> None:
        self._items.clear()
        for record in records[-self.capacity:]:
            input_ids = record.get("input_ids", record.get("ids", []))
            self._items.append(
                ReplayExample(
                    tuple(int(token) for token in input_ids),
                    str(record.get("label", "ancestral")),
                    _bounded_priority(record.get("priority", 1.0)),
                )
            )
        self._seen_count = len(self._items)

    @property
    def seen_count(self) -> int:
        return self._seen_count

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 2,
            "seen_count": self._seen_count,
            "records": self.to_records(),
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        records = state.get("records", [])
        if not isinstance(records, list):
            raise TypeError("ReplayBuffer records must be a list.")
        self.load_records(records)
        self._seen_count = max(int(state.get("seen_count", len(self._items))), len(self._items))
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self._rng.setstate(_nested_tuple(rng_state))


def _bounded_priority(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("replay priority must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("replay priority must be finite")
    return min(2.0, max(0.5, result))


def _nested_tuple(value):
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_nested_tuple(item) for item in value)
    return value


def forgetting_proxy(previous_loss: float, current_replay_loss: float) -> float:
    return max(0.0, float(current_replay_loss) - float(previous_loss))
