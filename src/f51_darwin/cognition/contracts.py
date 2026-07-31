from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch


COGNITIVE_ARCHITECTURE_VERSION = "three_organs_v1"
COGNITIVE_ORGAN_WIDTH = 512
PULSE_SCHEMA = "darwin-cognitive-pulse-v1"


class OrganKind(StrEnum):
    MEMORY = "organ:memory:v1"
    WORLD_MODEL = "organ:world_model:v1"
    EXECUTIVE = "organ:executive:v1"


CANONICAL_ORGAN_IDS = tuple(item.value for item in OrganKind)


def _require_digest(name: str, value: str) -> None:
    prefix = "sha256:"
    suffix = value.removeprefix(prefix)
    if not value.startswith(prefix) or len(suffix) != 64:
        raise ValueError(f"{name} must be sha256:<64 lowercase hex>")
    if any(char not in "0123456789abcdef" for char in suffix):
        raise ValueError(f"{name} must be sha256:<64 lowercase hex>")


@dataclass(frozen=True)
class CognitiveForwardMetadata:
    step_id: int
    checkpoint_id: str
    context_digest: str

    def __post_init__(self) -> None:
        if self.step_id < 0:
            raise ValueError("step_id must be non-negative")
        _require_digest("checkpoint_id", self.checkpoint_id)
        _require_digest("context_digest", self.context_digest)


@dataclass(frozen=True)
class ResidualCondition:
    condition_id: str
    organ: OrganKind
    values: torch.Tensor
    positions: torch.Tensor

    def __post_init__(self) -> None:
        if not self.condition_id:
            raise ValueError("condition_id must be non-empty")
        if self.organ is OrganKind.EXECUTIVE:
            raise ValueError("Executive may authorize but not inject residuals")
        if self.values.ndim != 3 or self.values.shape[-1] != COGNITIVE_ORGAN_WIDTH:
            raise ValueError("values must have shape [batch, count, organ_width=512]")
        if self.positions.ndim != 2:
            raise ValueError("positions must have shape [batch, count]")
        if self.positions.dtype != torch.long:
            raise ValueError("positions must use torch.long")
        if tuple(self.positions.shape) != tuple(self.values.shape[:2]):
            raise ValueError("positions and values must share batch/count")
        if not bool(torch.isfinite(self.values).all()):
            raise ValueError("condition values must be finite")

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

    @property
    def count(self) -> int:
        return int(self.values.shape[1])


@dataclass(frozen=True)
class CognitivePulseEvent:
    schema: str
    mode: str
    step_id: int
    checkpoint_id: str
    context_digest: str
    selected_memory_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    selected_candidate_id: str | None
    prediction_error: float | None
    uncertainty: float | None
    compute_spent: int

    def __post_init__(self) -> None:
        if self.schema != PULSE_SCHEMA:
            raise ValueError(f"unsupported pulse schema: {self.schema}")
        if self.mode not in {"shadow", "active"}:
            raise ValueError("pulse mode must be shadow or active")
        if self.step_id < 0 or self.compute_spent < 0:
            raise ValueError("pulse counters must be non-negative")
        _require_digest("checkpoint_id", self.checkpoint_id)
        _require_digest("context_digest", self.context_digest)
        for name, value in (
            ("prediction_error", self.prediction_error),
            ("uncertainty", self.uncertainty),
        ):
            if value is not None and not float("-inf") < float(value) < float("inf"):
                raise ValueError(f"{name} must be finite")

    @classmethod
    def shadow(cls, metadata: CognitiveForwardMetadata) -> "CognitivePulseEvent":
        return cls(
            schema=PULSE_SCHEMA,
            mode="shadow",
            step_id=metadata.step_id,
            checkpoint_id=metadata.checkpoint_id,
            context_digest=metadata.context_digest,
            selected_memory_ids=(),
            candidate_ids=(),
            selected_candidate_id=None,
            prediction_error=None,
            uncertainty=None,
            compute_spent=0,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mode": self.mode,
            "step_id": self.step_id,
            "checkpoint_id": self.checkpoint_id,
            "context_digest": self.context_digest,
            "selected_memory_ids": list(self.selected_memory_ids),
            "candidate_ids": list(self.candidate_ids),
            "selected_candidate_id": self.selected_candidate_id,
            "prediction_error": self.prediction_error,
            "uncertainty": self.uncertainty,
            "compute_spent": self.compute_spent,
        }
