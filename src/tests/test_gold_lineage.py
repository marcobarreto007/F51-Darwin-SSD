from __future__ import annotations

import json
from pathlib import Path

import pytest

from f51_darwin.artifact_manifest import (
    EXPECTED_CORPUS_BYTES,
    EXPECTED_CORPUS_SHA256,
    EXPECTED_CORPUS_TOKENS,
    publish_pointer_atomic,
    select_canonical_checkpoint,
    validate_checkpoint_report,
    verify_corpus_manifest,
)


def corpus_fixture(tmp_path: Path) -> tuple[Path, Path]:
    corpus = tmp_path / "feast_v2.bin"
    corpus.write_bytes(b"\x01\x00\x00\x00" * 3)
    manifest = tmp_path / "feast_v2.bin.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "bytes": 12,
                "tokens": 3,
                "sha256": "placeholder",
                "dtype": "int32",
                "little_endian": True,
                "tokenizer": "f51_bpe_80k",
                "composition": [
                    {"filename": "base.bin", "bytes": 8, "tokens": 2},
                    {"filename": "synthetic.bin", "bytes": 4, "tokens": 1},
                ],
            }
        ),
        encoding="utf-8",
    )
    return corpus, manifest


def verified_report(*, cycle: int = 71, identity_ok: bool = True) -> dict[str, object]:
    return {
        "checkpoint": f"organism_cycle_{cycle:03d}.pt",
        "checkpoint_version": 7,
        "embedded_config_matches_file": True,
        "training_state": {"cycle": cycle, "step": 40751 if cycle == 71 else 40251},
        "resume_shape_compatible": True,
        "topology_manifest_valid": True,
        "optimizer_type": "AdamW",
        "optimizer_resume_compatible": True,
        "identity_verified": identity_ok,
        "base_checkpoint_id": "lineage-070",
        "strict_resume_compatible": identity_ok,
        "gold_valid": identity_ok and cycle == 71,
    }


@pytest.mark.parametrize(
    ("field", "value", "expected_issue"),
    [
        ("checkpoint_version", 6, "checkpoint_version"),
        ("embedded_config_matches_file", False, "config"),
        ("training_state.cycle", 70, "cycle"),
        ("training_state.step", 40750, "step"),
        ("resume_shape_compatible", False, "shapes"),
        ("topology_manifest_valid", False, "topology"),
        ("optimizer_type", "SGD", "adamw"),
        ("optimizer_resume_compatible", False, "adamw_resume"),
        ("identity_verified", False, "identity"),
        ("base_checkpoint_id", "wrong", "base_checkpoint_id"),
    ],
)
def test_checkpoint_invariants_fail_closed(
    field: str, value: object, expected_issue: str
) -> None:
    report = verified_report()
    if field.startswith("training_state."):
        report["training_state"][field.split(".", 1)[1]] = value  # type: ignore[index]
    else:
        report[field] = value

    issues = validate_checkpoint_report(
        report,
        expected_cycle=71,
        expected_step=40751,
        expected_base_checkpoint_id="lineage-070",
    )

    assert expected_issue in issues


def test_cycle_number_alone_never_selects_checkpoint() -> None:
    reports = [
        verified_report(cycle=71, identity_ok=True),
        verified_report(cycle=77, identity_ok=False),
    ]

    assert select_canonical_checkpoint(reports) == reports[0]


def test_cycle_071_does_not_require_cycle_070_identity() -> None:
    cycle_070 = verified_report(cycle=70)
    cycle_071 = verified_report(cycle=71)
    cycle_070["base_checkpoint_id"] = "same-checkpoint-070"
    cycle_071["base_checkpoint_id"] = "same-checkpoint-071"

    assert validate_checkpoint_report(
        cycle_071,
        expected_cycle=71,
        expected_step=40751,
    ) == []


def test_operational_verifier_does_not_rewrite_tracked_summary() -> None:
    text = Path("src/tools/verify_gold_lineage.py").read_text(encoding="utf-8")

    assert "governance/audit/provenance/gold-lineage-summary.json" not in text
    assert '"tracked_worktree_clean": True' in text


def test_corpus_verifier_rejects_wrong_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, manifest = corpus_fixture(tmp_path)
    monkeypatch.setattr("f51_darwin.artifact_manifest.EXPECTED_CORPUS_BYTES", 12)
    monkeypatch.setattr("f51_darwin.artifact_manifest.EXPECTED_CORPUS_TOKENS", 3)
    monkeypatch.setattr("f51_darwin.artifact_manifest.EXPECTED_CORPUS_SHA256", "placeholder")

    assert verify_corpus_manifest(corpus, manifest, verify_hash=False)["status"] == "ok"
    for field, bad in (
        ("bytes", 8),
        ("tokens", 2),
        ("dtype", "uint32"),
        ("little_endian", False),
        ("tokenizer", "wrong"),
        ("composition", []),
    ):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload[field] = bad
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=field if field != "little_endian" else "endianness"):
            verify_corpus_manifest(corpus, manifest, verify_hash=False)
        corpus, manifest = corpus_fixture(tmp_path)


def test_publication_is_atomic_and_complete(tmp_path: Path) -> None:
    checkpoint = tmp_path / "organism_cycle_071.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.yaml"
    config.write_text("model_name: nitro\n", encoding="utf-8")
    pointer = tmp_path / "organism_latest.json"
    report = verified_report()

    publish_pointer_atomic(
        pointer,
        checkpoint=checkpoint,
        report=report,
        config=config,
        source_commit="a" * 40,
        base_checkpoint_id="lineage-070",
    )

    payload = json.loads(pointer.read_text(encoding="utf-8"))
    assert payload["filename"] == checkpoint.name
    assert payload["cycle"] == 71
    assert payload["step"] == 40751
    assert payload["checkpoint_version"] == 7
    assert len(payload["sha256"]) == 64
    assert len(payload["config_sha256"]) == 64
    assert payload["source_commit"] == "a" * 40
    assert not pointer.with_name(f"{pointer.name}.tmp").exists()


def test_constants_match_approved_corpus_contract() -> None:
    assert EXPECTED_CORPUS_BYTES == 74195890756
    assert EXPECTED_CORPUS_TOKENS == 18548972689
    assert EXPECTED_CORPUS_SHA256 == "9677e9f22f4d78efa7b25c77b2da3cbdb5fb2a499926ac641a75c13b65e25cff"
