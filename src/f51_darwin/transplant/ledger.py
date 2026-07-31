from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .experiment import ArmBudget


@dataclass(frozen=True)
class OrganEvidence:
    control_nll: float
    living_nll: float
    future_gain: float
    forgetting: float
    compute_cost: float
    instability: float
    compute_saving: float = 0.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("organ evidence must be finite")


def organ_utility(evidence: OrganEvidence) -> float:
    quality_gain = evidence.control_nll - evidence.living_nll
    return (
        quality_gain
        + evidence.future_gain
        + evidence.compute_saving
        - evidence.forgetting
        - evidence.compute_cost
        - evidence.instability
    )


@dataclass(frozen=True)
class LedgerRecord:
    schema_version: int
    organ_name: str
    language_donor_sha256: str
    organ_donor_sha256: str
    recipient_parent_sha256: str | None
    arm: str
    budget: ArmBudget
    budget_identity: str
    evidence: OrganEvidence
    utility: float
    lifecycle_decision: str
    evidence_id: str

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "organ_name": self.organ_name,
            "language_donor_sha256": self.language_donor_sha256,
            "organ_donor_sha256": self.organ_donor_sha256,
            "recipient_parent_sha256": self.recipient_parent_sha256,
            "arm": self.arm,
            "budget": asdict(self.budget),
            "budget_identity": self.budget_identity,
            "evidence": asdict(self.evidence),
            "utility": self.utility,
            "lifecycle_decision": self.lifecycle_decision,
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self.unsigned_payload(), "evidence_id": self.evidence_id}


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _evidence_id(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _validate_sha256(value: str | None, *, field: str) -> None:
    if value is None:
        return
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"invalid {field}")


class OrganLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records = self._load_records()

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "OrganLedger":
        return cls(path)

    @property
    def records(self) -> tuple[LedgerRecord, ...]:
        return tuple(self._records)

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        if self.path.exists():
            with self.path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()

    def _load_records(self) -> list[LedgerRecord]:
        if not self.path.exists():
            return []
        records: list[LedgerRecord] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                budget_payload = payload["budget"]
                budget = ArmBudget(
                    tokens=int(budget_payload["tokens"]),
                    steps=int(budget_payload["steps"]),
                    seed=int(budget_payload["seed"]),
                    starts=tuple(int(value) for value in budget_payload["starts"]),
                )
                evidence = OrganEvidence(**payload["evidence"])
                record = LedgerRecord(
                    schema_version=int(payload["schema_version"]),
                    organ_name=str(payload["organ_name"]),
                    language_donor_sha256=str(payload["language_donor_sha256"]),
                    organ_donor_sha256=str(payload["organ_donor_sha256"]),
                    recipient_parent_sha256=payload["recipient_parent_sha256"],
                    arm=str(payload["arm"]),
                    budget=budget,
                    budget_identity=str(payload["budget_identity"]),
                    evidence=evidence,
                    utility=float(payload["utility"]),
                    lifecycle_decision=str(payload["lifecycle_decision"]),
                    evidence_id=str(payload["evidence_id"]),
                )
                expected = _evidence_id(record.unsigned_payload())
                if record.evidence_id != expected:
                    raise ValueError(
                        f"evidence hash mismatch at JSONL line {line_number}"
                    )
                if record.budget_identity != budget.identity:
                    raise ValueError(
                        f"budget identity mismatch at JSONL line {line_number}"
                    )
                records.append(record)
        return records

    def append(
        self,
        *,
        organ_name: str,
        language_donor_sha256: str,
        organ_donor_sha256: str,
        recipient_parent_sha256: str | None,
        arm: str,
        budget: ArmBudget,
        evidence: OrganEvidence,
        lifecycle_decision: str,
    ) -> LedgerRecord:
        _validate_sha256(language_donor_sha256, field="language donor sha256")
        _validate_sha256(organ_donor_sha256, field="organ donor sha256")
        _validate_sha256(recipient_parent_sha256, field="recipient parent sha256")
        if arm not in {"A", "B", "C", "D"}:
            raise ValueError("ledger arm must be A, B, C or D")

        unsigned = {
            "schema_version": 1,
            "organ_name": organ_name,
            "language_donor_sha256": language_donor_sha256,
            "organ_donor_sha256": organ_donor_sha256,
            "recipient_parent_sha256": recipient_parent_sha256,
            "arm": arm,
            "budget": asdict(budget),
            "budget_identity": budget.identity,
            "evidence": asdict(evidence),
            "utility": organ_utility(evidence),
            "lifecycle_decision": lifecycle_decision,
        }
        record = LedgerRecord(
            schema_version=1,
            organ_name=organ_name,
            language_donor_sha256=language_donor_sha256,
            organ_donor_sha256=organ_donor_sha256,
            recipient_parent_sha256=recipient_parent_sha256,
            arm=arm,
            budget=budget,
            budget_identity=budget.identity,
            evidence=evidence,
            utility=organ_utility(evidence),
            lifecycle_decision=lifecycle_decision,
            evidence_id=_evidence_id(unsigned),
        )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = _canonical_bytes(record.to_payload()) + b"\n"
        descriptor = os.open(
            self.path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY | os.O_BINARY,
        )
        try:
            written = os.write(descriptor, encoded)
            if written != len(encoded):
                raise OSError("short ledger append")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._records.append(record)
        return record

