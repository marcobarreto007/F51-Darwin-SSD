from __future__ import annotations

from pathlib import Path

from tools.find_gold_checkpoint import qualify_candidate, search_gold


def test_candidate_requires_exact_size_and_sha(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"wrong")
    result = qualify_candidate(
        candidate,
        expected_size=10_876_850_383,
        expected_sha256="239fdcf175ac35d9402d664e2a2b40252ec9c73aea8ee458250d426adde9241b",
    )
    assert result.status == "rejected"
    assert "size_mismatch" in result.reasons


def test_exact_hash_candidate_is_found(tmp_path: Path) -> None:
    candidate = tmp_path / "renamed.pt"
    candidate.write_bytes(b"gold")
    import hashlib

    expected = {
        "size_bytes": 4,
        "sha256": hashlib.sha256(b"gold").hexdigest(),
    }
    report = search_gold([tmp_path], expected)
    assert report["status"] == "candidate_found"
    assert report["recovered_path"] == str(candidate.resolve())


def test_missing_gold_is_not_reported_as_recovered(tmp_path: Path) -> None:
    report = search_gold(
        [tmp_path],
        {"size_bytes": 100, "sha256": "0" * 64},
    )
    assert report["status"] == "missing"
    assert report["recovered_path"] is None
