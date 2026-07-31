from __future__ import annotations

import pytest

from f51_darwin.organism.cli import (
    remaining_budget_steps,
    validate_train_budget,
)


def test_budget_requires_positive_integer_for_train_budget() -> None:
    with pytest.raises(ValueError, match="positive"):
        validate_train_budget("train-budget", None)
    with pytest.raises(ValueError, match="positive"):
        validate_train_budget("train-budget", 0)
    validate_train_budget("train-budget", 65_000_000_000)
    validate_train_budget("cycle", None)


def test_remaining_budget_steps_never_exceeds_whole_batch_budget() -> None:
    assert (
        remaining_budget_steps(
            train_tokens_seen=0,
            max_train_tokens=65_000_000_000,
            batch_tokens=4096,
            max_steps_per_cycle=500,
        )
        == 500
    )
    assert (
        remaining_budget_steps(
            train_tokens_seen=64_999_990_000,
            max_train_tokens=65_000_000_000,
            batch_tokens=4096,
            max_steps_per_cycle=500,
        )
        == 2
    )
    assert (
        remaining_budget_steps(
            train_tokens_seen=64_999_997_440,
            max_train_tokens=65_000_000_000,
            batch_tokens=4096,
            max_steps_per_cycle=500,
        )
        == 0
    )


def test_remaining_budget_steps_rejects_invalid_accounting() -> None:
    with pytest.raises(ValueError, match="train_tokens_seen"):
        remaining_budget_steps(-1, 100, 10, 10)
    with pytest.raises(ValueError, match="batch_tokens"):
        remaining_budget_steps(0, 100, 0, 10)
    with pytest.raises(ValueError, match="max_steps_per_cycle"):
        remaining_budget_steps(0, 100, 10, 0)
