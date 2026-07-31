from __future__ import annotations

from pathlib import Path

import pytest

from f51_darwin.transplant_16b.ledger import (
    CoverageLedger,
    CoverageRecord,
    DriftRecord,
)


def test_ledger_resume_requires_same_plan_and_head(tmp_path: Path) -> None:
    path = tmp_path / "coverage.jsonl"
    ledger = CoverageLedger.open(path, plan_id="a" * 64)
    ledger.append(
        CoverageRecord(
            "donor.x",
            ("target.x",),
            "projection",
            "complete",
            {},
            drift=DriftRecord("kl", 0.267),
        )
    )

    resumed = CoverageLedger.open(path, plan_id="a" * 64)
    assert resumed.head == ledger.head
    assert resumed.records == ledger.records

    with pytest.raises(ValueError, match="plan identity"):
        CoverageLedger.open(path, plan_id="b" * 64)


def test_complete_rejects_unclassified_and_random_targets(
    tmp_path: Path,
) -> None:
    ledger = CoverageLedger.open(
        tmp_path / "coverage.jsonl",
        plan_id="a" * 64,
    )
    with pytest.raises(ValueError, match="donor coverage"):
        ledger.assert_complete({"donor.a"}, {"target.a"})

    ledger.append(
        CoverageRecord(
            "donor.a",
            ("target.a",),
            "random",
            "complete",
            {},
            drift=DriftRecord("l2", 12.5),
        )
    )
    with pytest.raises(ValueError, match="random target"):
        ledger.assert_complete({"donor.a"}, {"target.a"})


def test_tampered_chain_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "coverage.jsonl"
    ledger = CoverageLedger.open(path, plan_id="a" * 64)
    ledger.append(
        CoverageRecord(
            "donor.a",
            ("target.a",),
            "projection",
            "complete",
            {"mse": 0.0},
            drift=DriftRecord("mse", 0.0),
        )
    )
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('"mse":0.0', '"mse":1.0'), encoding="utf-8")

    with pytest.raises(ValueError, match="coverage ledger digest"):
        CoverageLedger.open(path, plan_id="a" * 64)
