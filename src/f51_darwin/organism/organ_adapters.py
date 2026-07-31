"""Pure adapters for lagged causal-organ observation frames."""

from __future__ import annotations

from collections.abc import Mapping
import math
from types import MappingProxyType

from .causal_bus import StepIdentity
from .organ_frames import OrganObservationFrame


class CuriosityPriorityAdapter:
    """Extract bounded future-replay priorities from an exact t+1 frame.

    The adapter is deliberately side-effect free and has no loss-scaling
    surface. The caller may attach the returned values only to replay rows
    created after the successful source outcome.
    """

    adapter_id = "curiosity-priority"
    version = "v1"

    def consume(
        self,
        *,
        frame: OrganObservationFrame,
        target: StepIdentity,
    ) -> Mapping[str, float]:
        if not isinstance(frame, OrganObservationFrame):
            raise TypeError("frame must be an OrganObservationFrame")
        if not isinstance(target, StepIdentity):
            raise TypeError("target must be a StepIdentity")
        frame.validate_for(target)
        if target.optimizer_step != frame.source.optimizer_step + 1:
            raise ValueError(
                "exact_next_step curiosity priority frame required"
            )

        payload = frame.signals.get("curiosity")
        if not isinstance(payload, Mapping):
            raise ValueError("curiosity priority frame is missing curiosity")
        if set(payload) != {
            "sample_ids",
            "priorities",
            "observation_id",
        }:
            raise ValueError("curiosity priority frame schema mismatch")
        sample_ids = _strings(payload["sample_ids"], "sample_ids")
        priorities = _priorities(payload["priorities"])
        if len(sample_ids) != len(priorities):
            raise ValueError(
                "curiosity sample_ids and priorities length mismatch"
            )
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("curiosity sample_ids must be unique")
        observation_id = payload["observation_id"]
        if not isinstance(observation_id, str) or not observation_id.startswith(
            "curiosity-observation-v1:"
        ):
            raise ValueError("curiosity observation_id is invalid")
        return MappingProxyType(dict(zip(sample_ids, priorities, strict=True)))

    def replay_priorities(
        self,
        *,
        frame: OrganObservationFrame,
        target: StepIdentity,
    ) -> Mapping[str, float]:
        return self.consume(frame=frame, target=target)

    def priorities_for(
        self,
        *,
        frame: OrganObservationFrame,
        target: StepIdentity,
    ) -> Mapping[str, float]:
        return self.consume(frame=frame, target=target)

    def __call__(
        self,
        *,
        frame: OrganObservationFrame,
        target: StepIdentity,
    ) -> Mapping[str, float]:
        return self.consume(frame=frame, target=target)


def _strings(value: object, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise TypeError(f"curiosity {label} must be a sequence")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"curiosity {label} must contain non-empty strings")
    return result


def _priorities(value: object) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise TypeError("curiosity priorities must be a sequence")
    priorities: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError("curiosity priority must be numeric")
        priority = float(item)
        if not math.isfinite(priority) or not 0.5 <= priority <= 2.0:
            raise ValueError(
                "curiosity priority must be finite and within [0.5, 2.0]"
            )
        priorities.append(priority)
    return tuple(priorities)


__all__ = ["CuriosityPriorityAdapter"]
