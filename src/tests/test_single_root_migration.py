from __future__ import annotations

import json
from pathlib import Path

import pytest

from f51_darwin import workspace_migration as migration


def make_project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "F51-Darwin-SSD"
    source = project / "F51-Dataset-Organizado"
    source.mkdir(parents=True)
    return project, source


def test_plan_uses_actual_nested_source(tmp_path: Path) -> None:
    project, source = make_project(tmp_path)

    plan = migration.build_migration_plan(project, source)

    assert plan.source_root == source.resolve()
    assert plan.target_root == (project / "workspace").resolve()


def test_apply_and_rollback_are_renames_not_copies(tmp_path: Path) -> None:
    project, source = make_project(tmp_path)
    artifact = source / "01_TOKENIZADOS" / "sample.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"lineage-bytes")
    plan = migration.build_migration_plan(project, source)
    manifest = source / "04_MANIFESTOS" / "single_root_migration.json"
    manifest.parent.mkdir(parents=True)
    migration.write_migration_manifest(plan, manifest)

    applied_manifest = migration.apply_plan(plan, manifest)

    assert not source.exists()
    assert (project / "workspace/01_TOKENIZADOS/sample.bin").read_bytes() == b"lineage-bytes"
    migration.rollback_plan(applied_manifest)
    assert not (project / "workspace").exists()
    assert artifact.read_bytes() == b"lineage-bytes"


def test_plan_rejects_different_volumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project, source = make_project(tmp_path)
    monkeypatch.setattr(
        migration,
        "_volume_identity",
        lambda path: "source-volume" if Path(path).name == source.name else "target-volume",
    )

    with pytest.raises(ValueError, match="same_volume"):
        migration.build_migration_plan(project, source)


def test_plan_rejects_existing_target(tmp_path: Path) -> None:
    project, source = make_project(tmp_path)
    (project / "workspace").mkdir()

    with pytest.raises(FileExistsError, match="target_conflict"):
        migration.build_migration_plan(project, source)


def test_plan_rejects_reparse_point_escaping_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, source = make_project(tmp_path)
    monkeypatch.setattr(
        migration,
        "find_reparse_points",
        lambda _source: [{"path": str(source / "escape"), "target": str(tmp_path.parent)}],
    )

    with pytest.raises(ValueError, match="reparse_boundary"):
        migration.build_migration_plan(project, source)


def test_plan_rejects_active_process_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, source = make_project(tmp_path)
    monkeypatch.setattr(
        migration,
        "find_conflicting_processes",
        lambda: [{"pid": 51, "command_line": "python darwin_organism.py run247"}],
    )

    with pytest.raises(RuntimeError, match="active_process"):
        migration.build_migration_plan(project, source)


def test_apply_rejects_altered_manifest(tmp_path: Path) -> None:
    project, source = make_project(tmp_path)
    plan = migration.build_migration_plan(project, source)
    manifest = source / "04_MANIFESTOS" / "single_root_migration.json"
    migration.write_migration_manifest(plan, manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["plan"]["source_root"] = str(project / "altered")
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest_integrity"):
        migration.apply_plan(plan, manifest)


def test_rollback_refuses_after_runtime_use(tmp_path: Path) -> None:
    project, source = make_project(tmp_path)
    plan = migration.build_migration_plan(project, source)
    manifest = source / "04_MANIFESTOS" / "single_root_migration.json"
    migration.write_migration_manifest(plan, manifest)
    applied_manifest = migration.apply_plan(plan, manifest)
    migration.update_manifest_state(applied_manifest, runtime_use_detected=True)

    with pytest.raises(RuntimeError, match="rollback_runtime_use"):
        migration.rollback_plan(applied_manifest)


def test_process_matcher_distinguishes_entrypoints_from_test_arguments() -> None:
    assert migration.is_conflicting_process_command_line(
        r'C:\Python\python.exe C:\repo\src\\scripts\\darwin_organism.py run247'
    )
    assert migration.is_conflicting_process_command_line(
        r'powershell.exe -NoProfile -File C:\repo\src\\scripts\\start_overnight_16b.ps1'
    )
    assert not migration.is_conflicting_process_command_line(
        r'C:\Python\python.exe -m pytest src\\tests\\test_serve_davi.py -k darwin_organism.py'
    )
    assert not migration.is_conflicting_process_command_line(
        r'pytest.exe -q src\\tests\\test_single_root_migration.py -k start_overnight_16b.ps1'
    )


def test_apply_failure_after_primary_rename_remains_rollbackable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, source = make_project(tmp_path)
    (source / "payload.bin").write_bytes(b"payload")
    root_runs = project / "runs"
    root_runs.mkdir()
    (root_runs / "state.json").write_text("{}", encoding="utf-8")
    plan = migration.build_migration_plan(project, source)
    manifest = source / "04_MANIFESTOS" / "single_root_migration.json"
    migration.write_migration_manifest(plan, manifest)
    original_replace = migration.os.replace

    def fail_first_root_move(old, new):
        if Path(old) == root_runs:
            raise OSError("injected root move failure")
        return original_replace(old, new)

    monkeypatch.setattr(migration.os, "replace", fail_first_root_move)
    with pytest.raises(OSError, match="injected root move failure"):
        migration.apply_plan(plan, manifest)

    failed_manifest = project / "workspace" / migration.MANIFEST_RELATIVE
    state = json.loads(failed_manifest.read_text(encoding="utf-8"))["state"]
    assert state["status"] == "apply_failed_recoverable"
    monkeypatch.setattr(migration.os, "replace", original_replace)
    rolled_back = migration.rollback_plan(failed_manifest)
    assert rolled_back == source / migration.MANIFEST_RELATIVE
    assert source.joinpath("payload.bin").read_bytes() == b"payload"
    assert root_runs.joinpath("state.json").is_file()


def test_crash_immediately_after_primary_rename_rolls_back_from_source_manifest_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, source = make_project(tmp_path)
    artifact = source / "payload.bin"
    artifact.write_bytes(b"crash-safe")
    plan = migration.build_migration_plan(project, source)
    source_manifest = source / migration.MANIFEST_RELATIVE
    migration.write_migration_manifest(plan, source_manifest)
    original_replace = migration.os.replace

    class SimulatedProcessCrash(BaseException):
        pass

    def crash_after_primary_rename(old, new):
        original_replace(old, new)
        if Path(old) == source and Path(new) == project / "workspace":
            raise SimulatedProcessCrash("crash after rename syscall")

    monkeypatch.setattr(migration.os, "replace", crash_after_primary_rename)
    with pytest.raises(SimulatedProcessCrash, match="crash after rename syscall"):
        migration.apply_plan(plan, source_manifest)

    target_manifest = project / "workspace" / migration.MANIFEST_RELATIVE
    state = json.loads(target_manifest.read_text(encoding="utf-8"))["state"]
    assert state["status"] == "primary_rename_intent"
    assert state["primary_rename_completed"] is False
    assert not source.exists()

    monkeypatch.setattr(migration.os, "replace", original_replace)
    rolled_back = migration.rollback_plan(source_manifest)

    assert rolled_back == source_manifest
    assert artifact.read_bytes() == b"crash-safe"
    assert not (project / "workspace").exists()


def test_ingest_marks_runtime_use_before_first_state_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from f51_darwin.ingestion import runtime as ingest_pipeline

    project, source = make_project(tmp_path)
    plan = migration.build_migration_plan(project, source)
    manifest = source / migration.MANIFEST_RELATIVE
    migration.write_migration_manifest(plan, manifest)
    applied_manifest = migration.apply_plan(plan, manifest)
    observed: list[bool] = []

    class ProbeDataFactory:
        def __init__(self, paths, firewall=None):
            del paths, firewall
            payload = json.loads(applied_manifest.read_text(encoding="utf-8"))
            observed.append(payload["state"]["runtime_use_detected"])

    monkeypatch.setattr(ingest_pipeline, "DataFactory", ProbeDataFactory)
    ingest_pipeline.IngestPipeline(project)

    assert observed == [True]
    state = json.loads(applied_manifest.read_text(encoding="utf-8"))["state"]
    assert state["runtime_use_actor"] == "ingest_pipeline"


def test_runtime_state_writers_call_automatic_marker() -> None:
    ingest = Path("src/f51_darwin/ingestion/runtime.py").read_text(encoding="utf-8")
    organism = Path("src/f51_darwin/organism/bootstrap.py").read_text(encoding="utf-8")
    ghost = Path("research/ghost_stream.py").read_text(encoding="utf-8")

    assert 'mark_runtime_use(self.root, actor="ingest_pipeline")' in ingest
    assert 'mark_runtime_use(self.root, actor="darwin_organism")' in organism
    assert 'mark_runtime_use(Path(project_root).resolve(), actor="ghost_stream")' in ghost
