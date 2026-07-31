from __future__ import annotations

import json
from pathlib import Path

import pytest

from f51_darwin.transplant_16b.contracts import (
    SourceFile,
    TransplantPlan,
)
from f51_darwin.transplant_16b.sources import sha256_file, verify_source_file


def test_source_file_fails_closed_after_tamper(tmp_path: Path) -> None:
    path = tmp_path / "donor.bin"
    path.write_bytes(b"valid")
    source = SourceFile(
        path=str(path),
        sha256=sha256_file(path),
        size_bytes=5,
    )

    assert verify_source_file(source).sha256 == source.sha256
    path.write_bytes(b"bad")

    with pytest.raises(ValueError, match="source identity mismatch"):
        verify_source_file(source)


def test_plan_hash_is_canonical(tmp_path: Path) -> None:
    plan = TransplantPlan.testing(tmp_path)
    first = plan.identity()
    payload = json.loads(plan.to_json())

    assert TransplantPlan.from_mapping(payload).identity() == first
    assert len(first) == 64


def test_plan_identity_tracks_source_and_calibration(tmp_path: Path) -> None:
    plan = TransplantPlan.testing(tmp_path)
    changed = TransplantPlan.from_mapping(
        {
            **json.loads(plan.to_json()),
            "calibration_digest": "f" * 64,
        }
    )

    assert changed.identity() != plan.identity()
