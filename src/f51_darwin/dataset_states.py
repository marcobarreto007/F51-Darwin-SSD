from __future__ import annotations

from enum import Enum


class DatasetStatus(str, Enum):
    CANDIDATE = "candidate"
    QUARANTINE = "quarantine"
    APPROVED = "approved"
    REJECTED = "rejected"
    RETIRED = "retired"


class SourceType(str, Enum):
    REAL = "real"
    SYNTHETIC = "synthetic"
    EDITED = "edited"
    IMPORTED = "imported"


VALID_TRANSITIONS: dict[DatasetStatus, set[DatasetStatus]] = {
    DatasetStatus.CANDIDATE: {
        DatasetStatus.QUARANTINE,
        DatasetStatus.REJECTED,
    },
    DatasetStatus.QUARANTINE: {
        DatasetStatus.APPROVED,
        DatasetStatus.REJECTED,
        DatasetStatus.RETIRED,
    },
    DatasetStatus.APPROVED: {DatasetStatus.RETIRED},
    DatasetStatus.REJECTED: {DatasetStatus.RETIRED},
    DatasetStatus.RETIRED: set(),
}


def can_transition(current: DatasetStatus, target: DatasetStatus) -> bool:
    return target in VALID_TRANSITIONS.get(current, set())
