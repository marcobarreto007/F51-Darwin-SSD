"""Immutable, delayed observation frames for causal organism feedback."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Any, Mapping

from .causal_bus import StepIdentity


@dataclass(frozen=True)
class OrganObservationFrame:
    source: StepIdentity
    signals: Mapping[str, Any]
    valid_from_step: int
    valid_through_step: int
    digest: str

    @classmethod
    def create(
        cls,
        *,
        source: StepIdentity,
        signals: Mapping[str, Any],
        valid_from_step: int,
        valid_through_step: int,
    ) -> "OrganObservationFrame":
        if not isinstance(source, StepIdentity):
            raise TypeError("source must be a StepIdentity")
        for name, value in (
            ("valid_from_step", valid_from_step),
            ("valid_through_step", valid_through_step),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if valid_from_step <= source.optimizer_step:
            raise ValueError("temporal_barrier requires a later target step")
        if valid_through_step < valid_from_step:
            raise ValueError("valid_through_step precedes valid_from_step")
        frozen = _freeze_finite(signals, path="signals")
        if not isinstance(frozen, Mapping):
            raise TypeError("signals must be a mapping")
        payload = _payload(source, frozen, valid_from_step, valid_through_step)
        digest = "organ-frame-v1:" + sha256(_canonical(payload)).hexdigest()
        return cls(source, frozen, valid_from_step, valid_through_step, digest)

    def validate_for(self, target: StepIdentity) -> None:
        if not isinstance(target, StepIdentity):
            raise TypeError("target must be a StepIdentity")
        if target.training_contract_id != self.source.training_contract_id:
            raise ValueError("training_contract_mismatch")
        if target.optimizer_step < self.valid_from_step:
            raise ValueError("temporal_barrier")
        if target.optimizer_step > self.valid_through_step:
            raise ValueError("expired")
        expected = self.from_state(self.to_state()).digest
        if expected != self.digest:
            raise ValueError("digest mismatch")

    def to_state(self) -> dict[str, object]:
        return _thaw(
            _payload(
                self.source,
                self.signals,
                self.valid_from_step,
                self.valid_through_step,
            )
        ) | {"digest": self.digest}

    @classmethod
    def from_state(
        cls, state: Mapping[str, Any]
    ) -> "OrganObservationFrame":
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        source = StepIdentity.from_state(state["source"])
        frame = cls.create(
            source=source,
            signals=state["signals"],
            valid_from_step=state["valid_from_step"],
            valid_through_step=state["valid_through_step"],
        )
        if state.get("digest") != frame.digest:
            raise ValueError("digest mismatch")
        return frame


def _freeze_finite(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        raise TypeError(f"{path} must not contain boolean numeric evidence")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite values")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} must use string keys")
            frozen[key] = _freeze_finite(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_finite(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{path} must contain canonical JSON-compatible values, "
        f"got {type(value).__name__}"
    )


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if value is None or isinstance(value, (str, int, float)):
        return value
    raise TypeError(
        "value must contain canonical JSON-compatible immutable values"
    )


def _payload(
    source: StepIdentity,
    signals: Mapping[str, Any],
    valid_from_step: int,
    valid_through_step: int,
) -> Mapping[str, Any]:
    return {
        "source": source.to_state(),
        "signals": signals,
        "valid_from_step": valid_from_step,
        "valid_through_step": valid_through_step,
    }


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _thaw(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
