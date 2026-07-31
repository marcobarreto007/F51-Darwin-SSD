from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SOURCE_DIR_NAME = "F51-Dataset-Organizado"
TARGET_DIR_NAME = "workspace"
MANIFEST_RELATIVE = Path("04_MANIFESTOS/single_root_migration.json")
ARTIFACT_RELATIVES = (
    Path("01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast_v2.bin"),
    Path("01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast_v2.bin.manifest.json"),
    Path("03_CHECKPOINTS/organism_cycle_070.pt"),
    Path("03_CHECKPOINTS/organism_cycle_071.pt"),
    Path("03_CHECKPOINTS/organism_cycle_077.pt"),
)
ROOT_MOVES = (
    (Path("tokenizer"), Path("tokenizer")),
    (Path("runs"), Path("runtime/runs")),
    (Path("logs"), Path("runtime/logs")),
    (Path("eval_results"), Path("runtime/evaluations")),
)
CONFLICTING_ENTRYPOINTS = {
    "_autocommit.ps1",
    "darwin_organism.py",
    "serve_davi.py",
    "ghost_stream.py",
    "ingest_pipeline.py",
    "start_overnight_16b.ps1",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _volume_identity(path: Path) -> str:
    resolved = path.resolve(strict=False)
    anchor = resolved.anchor
    if not anchor:
        raise ValueError(f"volume_identity: path has no volume anchor: {resolved}")
    return anchor.casefold()


def _tree_inventory(root: Path) -> tuple[int, int, dict[str, int]]:
    file_count = 0
    total_bytes = 0
    top_level: dict[str, int] = {}
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        dirnames[:] = [
            name
            for name in dirnames
            if not _is_reparse_point(base / name)
        ]
        for name in filenames:
            path = base / name
            try:
                size = path.stat().st_size
            except OSError:
                continue
            file_count += 1
            total_bytes += size
            relative = path.relative_to(root)
            key = relative.parts[0] if relative.parts else "."
            top_level[key] = top_level.get(key, 0) + size
    return file_count, total_bytes, dict(sorted(top_level.items()))


def _is_reparse_point(path: Path) -> bool:
    try:
        stat = path.lstat()
    except OSError:
        return False
    attributes = getattr(stat, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def find_reparse_points(root: Path) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in [*dirnames, *filenames]:
            path = base / name
            if not _is_reparse_point(path):
                continue
            try:
                target = os.readlink(path)
            except OSError:
                target = "unreadable"
            found.append({"path": str(path), "target": str(target)})
        dirnames[:] = [name for name in dirnames if not _is_reparse_point(base / name)]
    return found


def is_conflicting_process_command_line(command_line: str) -> bool:
    """Return true only when the process is actually running a Darwin entrypoint."""
    try:
        tokens = [token.strip('"') for token in shlex.split(command_line, posix=False)]
    except ValueError:
        return False
    if not tokens:
        return False
    executable = Path(tokens[0]).name.casefold()
    if executable in CONFLICTING_ENTRYPOINTS:
        return True
    if executable in {"powershell", "powershell.exe", "pwsh", "pwsh.exe"}:
        for index, token in enumerate(tokens[:-1]):
            if token.casefold() == "-file":
                return Path(tokens[index + 1]).name.casefold() in CONFLICTING_ENTRYPOINTS
        return False
    if executable in {"python", "python.exe", "pythonw.exe", "py", "py.exe"}:
        for index, token in enumerate(tokens[1:], start=1):
            lowered = token.casefold()
            if lowered == "-m":
                if index + 1 >= len(tokens):
                    return False
                module = tokens[index + 1].casefold().replace(".", "/")
                return Path(module).name in {
                    Path(name).stem for name in CONFLICTING_ENTRYPOINTS if name.endswith(".py")
                }
            if lowered in {"-c", "--command"}:
                return False
            if token.startswith("-"):
                continue
            return Path(token).name.casefold() in CONFLICTING_ENTRYPOINTS
    return False


def find_conflicting_processes() -> list[dict[str, Any]]:
    if os.name != "nt":
        return []
    command = (
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("active_process_probe_failed")
    payload = json.loads(result.stdout)
    rows = payload if isinstance(payload, list) else [payload]
    conflicts = []
    for row in rows:
        pid = int(row.get("ProcessId") or 0)
        line = str(row.get("CommandLine") or "")
        if pid != os.getpid() and is_conflicting_process_command_line(line):
            conflicts.append({"pid": pid, "command_line": line})
    return conflicts


@dataclass(frozen=True)
class MigrationPlan:
    project_root: Path
    source_root: Path
    target_root: Path
    source_commit: str
    created_at_utc: str
    source_volume: str
    target_volume: str
    disk_free_bytes: int
    file_count: int
    total_bytes: int
    top_level_bytes: dict[str, int]
    reparse_points: list[dict[str, str]]
    artifact_hashes: dict[str, dict[str, Any]]
    root_moves: list[dict[str, str]]
    broken_data_junction: dict[str, Any] | None
    active_processes: list[dict[str, Any]]
    rollback_command: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("project_root", "source_root", "target_root"):
            payload[key] = str(payload[key])
        return payload


def _path_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True


def _broken_data_junction(project_root: Path) -> dict[str, Any] | None:
    path = project_root / "data"
    if not os.path.lexists(path) or not _is_reparse_point(path):
        return None
    try:
        target = os.readlink(path)
    except OSError as exc:
        raise ValueError(f"data_junction_unreadable: {path}") from exc
    return {
        "path": str(path),
        "target": str(target),
        "target_exists": Path(target).exists(),
    }


def _artifact_inventory(project_root: Path, source_root: Path) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    paths: Iterable[tuple[str, Path]] = [
        (relative.as_posix(), source_root / relative) for relative in ARTIFACT_RELATIVES
    ]
    config = project_root / "src/configs/darwin_x_100m.yaml"  # 1.6B-Nitro archived
    paths = [*paths, ("src/configs/darwin_x_100m.yaml", config)]
    for label, path in paths:
        if not path.is_file():
            artifacts[label] = {"path": str(path), "exists": False}
            continue
        artifacts[label] = {
            "path": str(path),
            "exists": True,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return artifacts


def build_migration_plan(project_root: str | Path, source_root: str | Path) -> MigrationPlan:
    project = Path(project_root).resolve()
    source = Path(source_root).resolve()
    expected_source = (project / SOURCE_DIR_NAME).resolve()
    target = (project / TARGET_DIR_NAME).resolve()
    if source != expected_source or not _path_inside(source, project):
        raise ValueError(f"source_boundary: expected {expected_source}, observed {source}")
    if not source.is_dir():
        raise FileNotFoundError(f"source_missing: {source}")
    if os.path.lexists(target):
        raise FileExistsError(f"target_conflict: {target}")

    source_volume = _volume_identity(source)
    target_volume = _volume_identity(target)
    if source_volume != target_volume:
        raise ValueError(
            f"same_volume: source={source_volume!r}, target={target_volume!r}"
        )

    active = find_conflicting_processes()
    if active:
        raise RuntimeError(f"active_process: {active}")

    reparses = find_reparse_points(source)
    for item in reparses:
        target_text = item.get("target", "")
        target_path = Path(target_text)
        if not target_path.is_absolute():
            target_path = Path(item["path"]).parent / target_path
        if not _path_inside(target_path, project):
            raise ValueError(f"reparse_boundary: {item}")

    root_moves: list[dict[str, str]] = []
    for source_relative, target_relative in ROOT_MOVES:
        extra_source = project / source_relative
        future_conflict = source / target_relative
        if not os.path.lexists(extra_source):
            continue
        if os.path.lexists(future_conflict):
            raise FileExistsError(f"target_conflict: {future_conflict}")
        root_moves.append(
            {"source": str(extra_source), "target": str(target / target_relative)}
        )

    file_count, total_bytes, top_level = _tree_inventory(source)
    disk_free = shutil.disk_usage(project).free
    manifest_path = target / MANIFEST_RELATIVE
    rollback = (
        f"& src/tools/migrate_single_root_workspace.ps1 -ProjectRoot '{project}' "
        f"-Rollback -Manifest '{manifest_path}'"
    )
    return MigrationPlan(
        project_root=project,
        source_root=source,
        target_root=target,
        source_commit=_source_commit(project),
        created_at_utc=_utc_now(),
        source_volume=source_volume,
        target_volume=target_volume,
        disk_free_bytes=disk_free,
        file_count=file_count,
        total_bytes=total_bytes,
        top_level_bytes=top_level,
        reparse_points=reparses,
        artifact_hashes=_artifact_inventory(project, source),
        root_moves=root_moves,
        broken_data_junction=_broken_data_junction(project),
        active_processes=active,
        rollback_command=rollback,
    )


def validate_migration_plan(plan: MigrationPlan) -> None:
    if plan.source_volume != plan.target_volume:
        raise ValueError("same_volume: plan volumes differ")
    if plan.source_root != plan.project_root / SOURCE_DIR_NAME:
        raise ValueError("source_boundary: plan source is not the nested source")
    if plan.target_root != plan.project_root / TARGET_DIR_NAME:
        raise ValueError("target_boundary: plan target is not workspace")
    if plan.active_processes:
        raise RuntimeError(f"active_process: {plan.active_processes}")


def _manifest_digest(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("integrity_sha256", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_migration_manifest(plan: MigrationPlan, path: str | Path) -> Path:
    validate_migration_plan(plan)
    manifest = Path(path)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "plan": plan.to_dict(),
        "state": {
            "status": "planned",
            "runtime_use_detected": False,
            "destructive_cleanup_executed": False,
        },
    }
    payload["integrity_sha256"] = _manifest_digest(payload)
    _atomic_write_json(manifest, payload)
    return manifest


def _load_verified_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("integrity_sha256") != _manifest_digest(payload):
        raise ValueError(f"manifest_integrity: {path}")
    return payload


def update_manifest_state(path: str | Path, **updates: Any) -> Path:
    manifest = Path(path)
    payload = _load_verified_manifest(manifest)
    payload.setdefault("state", {}).update(updates)
    payload["integrity_sha256"] = _manifest_digest(payload)
    _atomic_write_json(manifest, payload)
    return manifest


def discover_migration_manifest(path: str | Path) -> Path:
    """Find the manifest on either side of the atomic primary-root rename."""
    manifest = Path(path)
    candidates = [manifest]
    dataset_root = manifest.parent.parent
    if manifest.name == MANIFEST_RELATIVE.name and manifest.parent.name == MANIFEST_RELATIVE.parent.name:
        if dataset_root.name == SOURCE_DIR_NAME:
            candidates.append(dataset_root.parent / TARGET_DIR_NAME / MANIFEST_RELATIVE)
        elif dataset_root.name == TARGET_DIR_NAME:
            candidates.append(dataset_root.parent / SOURCE_DIR_NAME / MANIFEST_RELATIVE)
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) > 1:
        raise RuntimeError(f"manifest_location_conflict: {existing}")
    if existing:
        return existing[0]
    raise FileNotFoundError(f"migration_manifest_missing: {candidates}")


def mark_runtime_use(project_root: str | Path, *, actor: str) -> Path | None:
    """Atomically close the rollback window before a migrated runtime writes state."""
    manifest = Path(project_root) / TARGET_DIR_NAME / MANIFEST_RELATIVE
    if not manifest.is_file():
        return None
    payload = _load_verified_manifest(manifest)
    state = payload.get("state", {})
    if state.get("status") != "applied":
        raise RuntimeError(
            f"runtime_use_state: expected applied, observed {state.get('status')}"
        )
    if state.get("runtime_use_detected"):
        return manifest
    return update_manifest_state(
        manifest,
        runtime_use_detected=True,
        runtime_use_actor=actor,
        runtime_use_first_at_utc=_utc_now(),
    )


def _assert_plan_matches_manifest(plan: MigrationPlan, payload: dict[str, Any]) -> None:
    if payload.get("plan") != plan.to_dict():
        raise ValueError("manifest_integrity: manifest plan does not match in-memory plan")


def _remove_validated_broken_data_junction(plan: MigrationPlan) -> None:
    recorded = plan.broken_data_junction
    if not recorded:
        return
    path = Path(recorded["path"])
    if not os.path.lexists(path):
        return
    if not _is_reparse_point(path):
        raise ValueError(f"data_junction_changed: {path} is no longer a reparse point")
    observed = os.readlink(path)
    if str(observed) != str(recorded["target"]):
        raise ValueError(
            f"data_junction_changed: expected={recorded['target']}, observed={observed}"
        )
    os.rmdir(path)


def apply_plan(plan: MigrationPlan, manifest_path: str | Path) -> Path:
    validate_migration_plan(plan)
    manifest = Path(manifest_path)
    payload = _load_verified_manifest(manifest)
    _assert_plan_matches_manifest(plan, payload)
    if find_conflicting_processes():
        raise RuntimeError("active_process: conflict appeared after planning")
    if not plan.source_root.is_dir():
        raise FileNotFoundError(f"source_missing: {plan.source_root}")
    if os.path.lexists(plan.target_root):
        raise FileExistsError(f"target_conflict: {plan.target_root}")

    relative_manifest = manifest.relative_to(plan.source_root)
    completed: list[dict[str, str]] = []
    update_manifest_state(
        manifest,
        status="primary_rename_intent",
        primary_rename_completed=False,
        completed_root_moves=completed,
        pending_root_move=None,
        primary_rename_intent_at_utc=_utc_now(),
    )
    os.replace(plan.source_root, plan.target_root)
    applied_manifest = plan.target_root / relative_manifest
    update_manifest_state(
        applied_manifest,
        status="applying",
        primary_rename_completed=True,
        completed_root_moves=completed,
    )
    try:
        for move in plan.root_moves:
            source = Path(move["source"])
            target = Path(move["target"])
            if not os.path.lexists(source):
                raise FileNotFoundError(f"root_move_source_missing: {source}")
            if os.path.lexists(target):
                raise FileExistsError(f"root_move_target_conflict: {target}")
            update_manifest_state(applied_manifest, pending_root_move=move)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
            completed.append(move)
            update_manifest_state(
                applied_manifest,
                completed_root_moves=completed,
                pending_root_move=None,
            )
        _remove_validated_broken_data_junction(plan)
        return update_manifest_state(
            applied_manifest,
            status="applied",
            applied_at_utc=_utc_now(),
            operation="same_volume_directory_rename",
            data_junction_removed=bool(plan.broken_data_junction),
            pending_root_move=None,
        )
    except Exception as exc:
        try:
            update_manifest_state(
                applied_manifest,
                status="apply_failed_recoverable",
                apply_error=f"{type(exc).__name__}: {exc}",
                completed_root_moves=completed,
            )
        except Exception:
            pass
        raise


def rollback_plan(manifest_path: str | Path) -> Path:
    manifest = discover_migration_manifest(manifest_path)
    payload = _load_verified_manifest(manifest)
    state = payload.get("state", {})
    allowed_states = {
        "primary_rename_intent",
        "applied",
        "applying",
        "apply_failed_recoverable",
    }
    if state.get("status") not in allowed_states:
        raise RuntimeError(
            f"rollback_state: expected recoverable state, observed {state.get('status')}"
        )
    if state.get("runtime_use_detected"):
        raise RuntimeError("rollback_runtime_use: runtime use was recorded after migration")

    plan_data = payload["plan"]
    project = Path(plan_data["project_root"])
    source = Path(plan_data["source_root"])
    target = Path(plan_data["target_root"])
    source_exists = os.path.lexists(source)
    target_exists = os.path.lexists(target)
    if (
        state.get("status") == "primary_rename_intent"
        and source_exists
        and not target_exists
    ):
        return update_manifest_state(
            manifest,
            status="rolled_back",
            rolled_back_at_utc=_utc_now(),
            rollback_note="primary rename was not observed",
        )
    if source_exists:
        raise FileExistsError(f"rollback_source_conflict: {source}")
    if not target_exists or not target.is_dir():
        raise FileNotFoundError(f"rollback_target_missing: {target}")
    if find_conflicting_processes():
        raise RuntimeError("active_process: rollback refused")

    if state.get("status") == "applied":
        recoverable_moves = list(plan_data.get("root_moves", []))
    else:
        recoverable_moves = list(state.get("completed_root_moves", []))
        pending = state.get("pending_root_move")
        if isinstance(pending, dict) and pending not in recoverable_moves:
            recoverable_moves.append(pending)
    for move in reversed(recoverable_moves):
        moved_source = Path(move["source"])
        moved_target = Path(move["target"])
        source_exists = os.path.lexists(moved_source)
        target_exists = os.path.lexists(moved_target)
        if source_exists and not target_exists:
            continue
        if source_exists:
            raise FileExistsError(f"rollback_root_move_conflict: {moved_source}")
        if not target_exists:
            raise FileNotFoundError(f"rollback_root_move_missing: {moved_target}")
        moved_source.parent.mkdir(parents=True, exist_ok=True)
        os.replace(moved_target, moved_source)
    runtime = target / "runtime"
    if runtime.is_dir() and not any(runtime.iterdir()):
        runtime.rmdir()

    relative_manifest = manifest.relative_to(target)
    os.replace(target, source)
    rolled_back_manifest = source / relative_manifest
    return update_manifest_state(
        rolled_back_manifest,
        status="rolled_back",
        rolled_back_at_utc=_utc_now(),
    )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reversible Darwin-X single-root migration")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--manifest", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--rollback", action="store_true")
    args = parser.parse_args(argv)

    project = args.project_root.resolve()
    if args.rollback:
        if args.manifest is None:
            parser.error("--rollback requires --manifest")
        result = rollback_plan(args.manifest)
        print(json.dumps({"status": "rolled_back", "manifest": str(result)}, indent=2))
        return 0

    source = (args.source or project / SOURCE_DIR_NAME).resolve()
    plan = build_migration_plan(project, source)
    if not args.apply:
        print(json.dumps({"status": "dry_run", "plan": plan.to_dict()}, indent=2))
        return 0

    manifest = args.manifest or source / MANIFEST_RELATIVE
    write_migration_manifest(plan, manifest)
    result = apply_plan(plan, manifest)
    print(json.dumps({"status": "applied", "manifest": str(result)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
