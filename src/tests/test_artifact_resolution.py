from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from f51_darwin import artifacts
from f51_darwin.artifacts import (
    audit_checkpoint_files,
    audit_checkpoint_pointers,
    audit_token_bins,
    extract_step,
    find_latest_checkpoint,
    inspect_checkpoint_file,
    inspect_checkpoint_pointer,
    inspect_token_bin,
    resolve_checkpoint_pointer,
    resolve_latest_checkpoint,
    resolve_latest_organism_checkpoint,
    scan_latest_organism_checkpoint,
    resolve_token_bin,
)
from f51_darwin.dataset_layout import FEAST_TOKEN_RELATIVE


def test_supported_artifact_resolver_has_no_sibling_fallback() -> None:
    text = Path("src/f51_darwin/artifacts.py").read_text(encoding="utf-8")

    assert "F51-Dataset-Organizado" not in text
    assert "external dataset workspace" not in text


def test_resolve_token_bin_skips_empty_and_prefers_existing_order(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "tokens.bin").write_bytes(b"")
    (data / "tokens_full.bin").write_bytes(b"123")
    chunk = data / "tokens_chunk_aa"
    chunk.write_bytes((1).to_bytes(4, "little", signed=True))

    assert resolve_token_bin(tmp_path, allow_repo_fallback=True) == chunk


def test_resolve_token_bin_prefers_canonical_external_corpus(tmp_path: Path) -> None:
    project = tmp_path / "F51-Darwin-SSD"
    project.mkdir()
    local = project / "data" / "tokens.bin"
    local.parent.mkdir()
    local.write_bytes((7).to_bytes(4, "little", signed=True))
    feast = project / "workspace" / FEAST_TOKEN_RELATIVE
    feast.parent.mkdir(parents=True)
    feast.write_bytes((51).to_bytes(4, "little", signed=True))

    assert resolve_token_bin(project) == feast


def test_inspect_token_bin_reports_empty_unaligned_and_ok(tmp_path: Path) -> None:
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    unaligned = tmp_path / "bad.bin"
    unaligned.write_bytes(b"123")
    good = tmp_path / "good.bin"
    good.write_bytes((1).to_bytes(4, "little", signed=True) * 3)

    assert inspect_token_bin(tmp_path, empty).status == "empty"
    assert inspect_token_bin(tmp_path, unaligned).status == "unaligned"
    health = inspect_token_bin(tmp_path, good)
    assert health.status == "ok"
    assert health.token_count == 3


def test_audit_token_bins_reports_candidate_health(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "tokens.bin").write_bytes(b"")
    (data / "tokens_chunk_aa").write_bytes((7).to_bytes(4, "little", signed=True))

    health = audit_token_bins(tmp_path, ["data/tokens.bin", "data/tokens_chunk_aa", "missing.bin"])

    assert [item.status for item in health] == ["empty", "ok", "missing"]


def test_checkpoint_pointer_must_point_to_existing_file(tmp_path: Path) -> None:
    pointer = tmp_path / "latest.json"
    pointer.write_text(json.dumps({"path": "missing.pt"}), encoding="utf-8")
    assert resolve_checkpoint_pointer(tmp_path, pointer) is None

    checkpoint = tmp_path / "step_0000007.pt"
    checkpoint.write_bytes(b"pt")
    pointer.write_text(json.dumps({"path": checkpoint.name}), encoding="utf-8")
    assert resolve_checkpoint_pointer(tmp_path, pointer) == checkpoint


def test_external_checkpoint_pointer_resolves_target_beside_pointer(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    external = tmp_path / "external-checkpoints"
    external.mkdir()
    checkpoint = external / "organism_cycle_064.pt"
    checkpoint.write_bytes(b"pt")
    pointer = external / "organism_latest.json"
    pointer.write_text(json.dumps({"path": checkpoint.name, "step": 32000}), encoding="utf-8")

    assert resolve_checkpoint_pointer(project, pointer) == checkpoint
    health = inspect_checkpoint_pointer(project, pointer)
    assert health.status == "ok"
    assert health.target == checkpoint


def test_inspect_checkpoint_pointer_reports_broken_target(tmp_path: Path) -> None:
    pointer = tmp_path / "latest.json"
    pointer.write_text(json.dumps({"path": "missing.pt", "step": 42}), encoding="utf-8")

    health = inspect_checkpoint_pointer(tmp_path, pointer)

    assert health.status == "broken_target"
    assert health.exists
    assert health.valid_json
    assert health.target == tmp_path / "missing.pt"
    assert not health.target_exists
    assert health.step == 42


def test_audit_checkpoint_pointers_reports_missing_and_ok(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step_0000007.pt"
    checkpoint.write_bytes(b"pt")
    pointer = tmp_path / "latest.json"
    pointer.write_text(json.dumps({"path": checkpoint.name}), encoding="utf-8")

    health = audit_checkpoint_pointers(tmp_path, [pointer, "missing.json"])

    assert [item.status for item in health] == ["ok", "missing"]


def test_inspect_checkpoint_file_reports_corrupt_zip_header(tmp_path: Path) -> None:
    checkpoint = tmp_path / "organism_cycle_001.pt"
    checkpoint.write_bytes(b"PK\x03\x04truncated")

    health = inspect_checkpoint_file(checkpoint)

    assert health.status == "corrupt_zip"
    assert health.exists
    assert health.size_bytes > 0
    assert health.step == 1


def test_audit_checkpoint_files_reports_zip_and_legacy_formats(tmp_path: Path) -> None:
    import zipfile

    root = tmp_path / "checkpoints"
    root.mkdir()
    zipped = root / "step_0000002.pt"
    with zipfile.ZipFile(zipped, "w") as archive:
        archive.writestr("data.pkl", b"payload")
    legacy = root / "step_0000003.pt"
    legacy.write_bytes(b"\x80\x04legacy")

    health = audit_checkpoint_files(tmp_path, ["checkpoints"])

    assert [item.status for item in health] == ["torch_zip", "legacy_pickle_or_raw"]


def test_latest_checkpoint_falls_back_to_highest_step(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints" / "unified"
    root.mkdir(parents=True)
    (root / "step_0000002.pt").write_bytes(b"old")
    newest = root / "step_0000010.pt"
    newest.write_bytes(b"new")

    found = find_latest_checkpoint(tmp_path, ["checkpoints/unified"])
    assert found is not None
    assert found.path == newest
    assert extract_step(newest) == 10


def test_resolve_latest_checkpoint_prefers_valid_pointer_before_scan(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints" / "base"
    root.mkdir(parents=True)
    scan_latest = root / "step_0000010.pt"
    scan_latest.write_bytes(b"scan")
    pointed = root / "step_0000004.pt"
    pointed.write_bytes(b"pointer")
    pointer = root / "latest.json"
    pointer.write_text(json.dumps({"path": "checkpoints/base/step_0000004.pt"}), encoding="utf-8")

    assert resolve_latest_checkpoint(
        tmp_path,
        pointers=["checkpoints/base/latest.json"],
        search_roots=["checkpoints/base"],
    ) == pointed


def write_operational_lineage_contract(
    project: Path,
    checkpoint: Path,
    pointer_payload: dict[str, object],
    *,
    report_base_checkpoint_id: str | None = None,
) -> None:
    cycle = int(pointer_payload["cycle"])
    base_id = report_base_checkpoint_id or str(pointer_payload["base_checkpoint_id"])
    manifests = project / "workspace" / "04_MANIFESTOS"
    manifests.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "status": "ok",
        "source_commit": pointer_payload["source_commit"],
        "tracked_worktree_clean": True,
        "blockers": [],
        "pointer": {**pointer_payload, "status": "ok"},
        "checkpoints": {
            str(cycle): {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_version": pointer_payload["checkpoint_version"],
                "file_sha256": pointer_payload["sha256"],
                "base_checkpoint_id": base_id,
                "gold_valid": True,
                "strict_resume_compatible": True,
                "identity_verified": True,
                "identity": {
                    "declared": base_id,
                    "with_raw_embedded_config": base_id,
                },
                "training_state": {
                    "cycle": cycle,
                    "step": pointer_payload["step"],
                },
            }
        },
    }
    (manifests / "gold_local_lineage.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_latest_organism_checkpoint_uses_validated_canonical_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "F51-Darwin-SSD"
    project.mkdir()
    checkpoints = project / "workspace" / "03_CHECKPOINTS"
    checkpoints.mkdir(parents=True)
    (checkpoints / "organism_cycle_009.pt").write_bytes(b"old")
    noncanonical_newer = checkpoints / "organism_cycle_239.pt"
    noncanonical_newer.write_bytes(b"new")
    canonical = checkpoints / "organism_cycle_071.pt"
    canonical.write_bytes(b"gold")
    historical = checkpoints / "historical_git_lfs"
    historical.mkdir()
    (historical / "organism_cycle_999.pt").write_bytes(b"archive")

    source_commit = "b" * 40
    base_id = "darwin-model-core-v1:" + "a" * 64
    pointer_payload = {
        "version": 1,
        "checkpoint_version": 7,
        "path": canonical.name,
        "filename": canonical.name,
        "cycle": 71,
        "step": 40751,
        "size_bytes": canonical.stat().st_size,
        "sha256": hashlib.sha256(canonical.read_bytes()).hexdigest(),
        "base_checkpoint_id": base_id,
        "source_commit": source_commit,
        "health": "ok",
    }
    (checkpoints / "organism_latest.json").write_text(
        json.dumps(pointer_payload), encoding="utf-8"
    )
    write_operational_lineage_contract(project, canonical, pointer_payload)
    monkeypatch.setattr(artifacts, "_current_source_commit", lambda _root: source_commit)
    monkeypatch.setattr(artifacts, "_tracked_worktree_clean", lambda _root: True)

    assert resolve_latest_organism_checkpoint(project, require=True) == canonical
    assert scan_latest_organism_checkpoint(project, require=True) == noncanonical_newer


def test_latest_organism_checkpoint_rejects_missing_or_invalid_pointer(tmp_path: Path) -> None:
    project = tmp_path / "F51-Darwin-SSD"
    checkpoints = project / "workspace" / "03_CHECKPOINTS"
    checkpoints.mkdir(parents=True)
    checkpoint = checkpoints / "organism_cycle_077.pt"
    checkpoint.write_bytes(b"noncanonical")

    assert resolve_latest_organism_checkpoint(project) is None
    with pytest.raises(FileNotFoundError, match="canonical organism pointer"):
        resolve_latest_organism_checkpoint(project, require=True)

    (checkpoints / "organism_latest.json").write_text(
        json.dumps(
            {
                "path": checkpoint.name,
                "size_bytes": checkpoint.stat().st_size,
                "sha256": "not-a-sha256",
                "health": "ok",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sha256"):
        resolve_latest_organism_checkpoint(project, require=True)


@pytest.mark.parametrize(
    "base_id",
    [
        "a" * 64,
        "wrong-prefix:" + "a" * 64,
        "darwin-model-core-v1:not-hex",
        "darwin-model-core-v1:" + "a" * 63,
    ],
)
def test_canonical_pointer_rejects_invalid_integral_identity_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    base_id: str,
) -> None:
    project = tmp_path / "project"
    checkpoint = project / "workspace/03_CHECKPOINTS/organism_cycle_071.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"gold")
    source_commit = "b" * 40
    payload = {
        "version": 1,
        "checkpoint_version": 7,
        "path": checkpoint.name,
        "filename": checkpoint.name,
        "cycle": 71,
        "step": 40751,
        "size_bytes": checkpoint.stat().st_size,
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "base_checkpoint_id": base_id,
        "source_commit": source_commit,
        "health": "ok",
    }
    checkpoint.with_name("organism_latest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    write_operational_lineage_contract(project, checkpoint, payload)
    monkeypatch.setattr(artifacts, "_current_source_commit", lambda _root: source_commit)
    monkeypatch.setattr(artifacts, "_tracked_worktree_clean", lambda _root: True)

    with pytest.raises(ValueError, match="base_checkpoint_id"):
        resolve_latest_organism_checkpoint(project, require=True)


def test_canonical_pointer_rejects_manifest_identity_or_source_commit_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    checkpoint = project / "workspace/03_CHECKPOINTS/organism_cycle_071.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"gold")
    source_commit = "b" * 40
    base_id = "darwin-model-core-v1:" + "a" * 64
    payload = {
        "version": 1,
        "checkpoint_version": 7,
        "path": checkpoint.name,
        "filename": checkpoint.name,
        "cycle": 71,
        "step": 40751,
        "size_bytes": checkpoint.stat().st_size,
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "base_checkpoint_id": base_id,
        "source_commit": source_commit,
        "health": "ok",
    }
    checkpoint.with_name("organism_latest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    write_operational_lineage_contract(
        project,
        checkpoint,
        payload,
        report_base_checkpoint_id="darwin-model-core-v1:" + "c" * 64,
    )
    monkeypatch.setattr(artifacts, "_current_source_commit", lambda _root: source_commit)
    monkeypatch.setattr(artifacts, "_tracked_worktree_clean", lambda _root: True)

    with pytest.raises(ValueError, match="base_checkpoint_id"):
        resolve_latest_organism_checkpoint(project, require=True)

    write_operational_lineage_contract(project, checkpoint, payload)
    monkeypatch.setattr(artifacts, "_current_source_commit", lambda _root: "d" * 40)
    with pytest.raises(ValueError, match="source_commit"):
        resolve_latest_organism_checkpoint(project, require=True)
