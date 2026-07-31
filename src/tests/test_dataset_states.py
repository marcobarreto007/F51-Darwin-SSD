import pytest

from f51_darwin.dataset_states import DatasetStatus, SourceType, can_transition


def test_valid_transitions() -> None:
    assert can_transition(DatasetStatus.CANDIDATE, DatasetStatus.QUARANTINE)
    assert not can_transition(DatasetStatus.CANDIDATE, DatasetStatus.APPROVED)
    assert can_transition(DatasetStatus.QUARANTINE, DatasetStatus.APPROVED)
    assert not can_transition(DatasetStatus.REJECTED, DatasetStatus.APPROVED)


def test_source_types_exist() -> None:
    assert SourceType.SYNTHETIC.value == "synthetic"
    assert DatasetStatus.RETIRED.value == "retired"
