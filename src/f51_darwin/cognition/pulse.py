from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from f51_darwin.hashing import canonical_sha256

from .contracts import CognitivePulseEvent


PULSE_STATE_SCHEMA = "darwin-cognitive-pulse-state-v1"


@dataclass(frozen=True)
class CognitivePulseRecord:
    event: CognitivePulseEvent
    previous_sha256: str | None
    sha256: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "event": self.event.to_payload(),
            "previous_sha256": self.previous_sha256,
            "sha256": self.sha256,
        }


def _event_from_payload(raw: Mapping[str, Any]) -> CognitivePulseEvent:
    return CognitivePulseEvent(
        schema=str(raw["schema"]),
        mode=str(raw["mode"]),
        step_id=int(raw["step_id"]),
        checkpoint_id=str(raw["checkpoint_id"]),
        context_digest=str(raw["context_digest"]),
        selected_memory_ids=tuple(raw["selected_memory_ids"]),
        candidate_ids=tuple(raw["candidate_ids"]),
        selected_candidate_id=raw.get("selected_candidate_id"),
        prediction_error=raw.get("prediction_error"),
        uncertainty=raw.get("uncertainty"),
        compute_spent=int(raw["compute_spent"]),
    )


class CognitivePulse:
    def __init__(self) -> None:
        self._records: list[CognitivePulseRecord] = []

    @property
    def records(self) -> tuple[CognitivePulseRecord, ...]:
        return tuple(self._records)

    @property
    def head_sha256(self) -> str | None:
        return self._records[-1].sha256 if self._records else None

    def append(self, event: CognitivePulseEvent) -> CognitivePulseRecord:
        if self._records and event.step_id <= self._records[-1].event.step_id:
            raise ValueError("CognitivePulse step_id must strictly increase")
        previous = self.head_sha256
        digest = canonical_sha256(
            {"event": event.to_payload(), "previous_sha256": previous}
        )
        record = CognitivePulseRecord(
            event=event,
            previous_sha256=previous,
            sha256=digest,
        )
        self._records.append(record)
        return record

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": PULSE_STATE_SCHEMA,
            "records": [record.to_payload() for record in self._records],
            "head_sha256": self.head_sha256,
        }

    @classmethod
    def from_state_dict(cls, raw: Mapping[str, Any]) -> "CognitivePulse":
        if raw.get("schema") != PULSE_STATE_SCHEMA:
            raise ValueError("unsupported CognitivePulse state schema")
        pulse = cls()
        for stored in raw.get("records", []):
            event = _event_from_payload(stored["event"])
            record = pulse.append(event)
            if record.previous_sha256 != stored.get("previous_sha256"):
                raise ValueError("CognitivePulse previous hash mismatch")
            if record.sha256 != stored.get("sha256"):
                raise ValueError("CognitivePulse record hash mismatch")
        if pulse.head_sha256 != raw.get("head_sha256"):
            raise ValueError("CognitivePulse head hash mismatch")
        return pulse
