from __future__ import annotations

import json
import re
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from f51_darwin.dataset_layout import (
    resolve_feast_token_bin,
    resolve_organism_checkpoints_root,
)


TOKEN_BIN_CANDIDATES = (
    "data/tokens.bin",
    "data/tokens_full.bin",
    "data/tokens_chunk_aa",
    "tokens_full.bin",
)

TOKEN_BIN_AUDIT_CANDIDATES = TOKEN_BIN_CANDIDATES + (
    "data/tokens_final.bin",
    "data/tokens_unified.bin",
    "data/tokens_fast_test.bin",
)

CHECKPOINT_PATTERNS = ("step_*.pt", "cloud_step_*.pt", "organism_cycle_*.pt", "moe_step_*.pt", "*.pt")


@dataclass(frozen=True)
class CheckpointCandidate:
    path: Path
    step: int
    mtime: float
    source: str


@dataclass(frozen=True)
class CheckpointPointerHealth:
    pointer: Path
    exists: bool
    valid_json: bool
    target: Path | None
    target_exists: bool
    error: str = ""
    step: int = -1

    @property
    def status(self) -> str:
        if not self.exists:
            return "missing"
        if not self.valid_json:
            return "invalid_json"
        if self.target is None:
            return "missing_target_field"
        if not self.target_exists:
            return "broken_target"
        return "ok"

    def to_dict(self) -> dict[str, object]:
        return {
            "pointer": str(self.pointer),
            "exists": self.exists,
            "valid_json": self.valid_json,
            "target": None if self.target is None else str(self.target),
            "target_exists": self.target_exists,
            "error": self.error,
            "step": self.step,
            "status": self.status,
        }


@dataclass(frozen=True)
class CheckpointFileHealth:
    path: Path
    exists: bool
    size_bytes: int
    header_hex: str
    is_zip: bool
    status: str
    step: int = -1
    zip_entries: int | None = None
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "exists": self.exists,
            "size_bytes": self.size_bytes,
            "header_hex": self.header_hex,
            "is_zip": self.is_zip,
            "status": self.status,
            "step": self.step,
            "zip_entries": self.zip_entries,
            "error": self.error,
        }


@dataclass(frozen=True)
class TokenBinHealth:
    path: Path
    exists: bool
    size_bytes: int
    int32_aligned: bool
    token_count: int
    status: str
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "exists": self.exists,
            "size_bytes": self.size_bytes,
            "int32_aligned": self.int32_aligned,
            "token_count": self.token_count,
            "status": self.status,
            "error": self.error,
        }


def _as_project_path(project_root: str | Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(project_root) / path


def _as_pointer_target(
    project_root: str | Path, pointer: Path, value: str | Path
) -> Path:
    """Resolve modern pointer-relative targets with a legacy project fallback."""
    path = Path(value)
    if path.is_absolute():
        return path
    beside_pointer = pointer.parent / path
    if beside_pointer.exists():
        return beside_pointer
    legacy_project_path = Path(project_root) / path
    if legacy_project_path.exists():
        return legacy_project_path
    return beside_pointer


def resolve_token_bin(
    project_root: str | Path,
    candidates: Iterable[str | Path] = TOKEN_BIN_CANDIDATES,
    *,
    min_bytes: int = 4,
    require_int32_alignment: bool = True,
    allow_repo_fallback: bool = False,
) -> Path | None:
    """Resolve the canonical workspace token bin or an explicitly allowed fallback.

    Repository-local candidates are compatibility fixtures, not a production
    corpus. Callers must opt in to that fallback explicitly.
    """

    root = Path(project_root)
    if candidates is TOKEN_BIN_CANDIDATES:
        external = resolve_feast_token_bin(root)
        health = inspect_token_bin(root, external, min_bytes=min_bytes)
        if health.exists and health.size_bytes >= min_bytes:
            if not require_int32_alignment or health.int32_aligned:
                return external
        if not allow_repo_fallback:
            return None

    for candidate in candidates:
        path = _as_project_path(root, candidate)
        health = inspect_token_bin(root, path, min_bytes=min_bytes)
        if not health.exists:
            continue
        if health.size_bytes < min_bytes:
            continue
        if require_int32_alignment and not health.int32_aligned:
            continue
        return path
    return None


def inspect_token_bin(
    project_root: str | Path,
    path: str | Path,
    *,
    min_bytes: int = 4,
) -> TokenBinHealth:
    token_path = _as_project_path(project_root, path)
    if not token_path.exists() or not token_path.is_file():
        return TokenBinHealth(
            path=token_path,
            exists=False,
            size_bytes=0,
            int32_aligned=False,
            token_count=0,
            status="missing",
            error="file_missing",
        )

    size = token_path.stat().st_size
    aligned = size % 4 == 0
    if size == 0:
        status = "empty"
        error = "zero_bytes"
    elif not aligned:
        status = "unaligned"
        error = "not_int32_aligned"
    elif size < min_bytes:
        status = "too_small"
        error = f"size_below_{min_bytes}_bytes"
    else:
        status = "ok"
        error = ""
    return TokenBinHealth(
        path=token_path,
        exists=True,
        size_bytes=size,
        int32_aligned=aligned,
        token_count=size // 4 if aligned else 0,
        status=status,
        error=error,
    )


def audit_token_bins(
    project_root: str | Path,
    candidates: Iterable[str | Path] = TOKEN_BIN_CANDIDATES,
    *,
    min_bytes: int = 4,
) -> list[TokenBinHealth]:
    root = Path(project_root)
    seen: set[Path] = set()
    health: list[TokenBinHealth] = []
    for candidate in candidates:
        path = _as_project_path(root, candidate)
        resolved = path.resolve() if path.exists() else path
        if resolved in seen:
            continue
        seen.add(resolved)
        health.append(inspect_token_bin(root, path, min_bytes=min_bytes))
    return health


def resolve_checkpoint_pointer(project_root: str | Path, pointer_path: str | Path) -> Path | None:
    """Resolve a latest*.json pointer only when it points to an existing file."""

    pointer = _as_project_path(project_root, pointer_path)
    if not pointer.exists() or not pointer.is_file():
        return None
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for key in ("path", "checkpoint", "checkpoint_path"):
        raw = payload.get(key)
        if not raw:
            continue
        checkpoint = _as_pointer_target(project_root, pointer, raw)
        if checkpoint.exists() and checkpoint.is_file():
            return checkpoint
    return None


def inspect_checkpoint_pointer(project_root: str | Path, pointer_path: str | Path) -> CheckpointPointerHealth:
    """Inspect a latest*.json pointer without trusting or mutating it."""

    root = Path(project_root)
    pointer = _as_project_path(root, pointer_path)
    if not pointer.exists() or not pointer.is_file():
        return CheckpointPointerHealth(
            pointer=pointer,
            exists=False,
            valid_json=False,
            target=None,
            target_exists=False,
            error="pointer_missing",
        )
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return CheckpointPointerHealth(
            pointer=pointer,
            exists=True,
            valid_json=False,
            target=None,
            target_exists=False,
            error=type(exc).__name__,
        )
    target: Path | None = None
    for key in ("path", "checkpoint", "checkpoint_path"):
        raw = payload.get(key)
        if raw:
            target = _as_pointer_target(root, pointer, raw)
            break
    if target is None:
        return CheckpointPointerHealth(
            pointer=pointer,
            exists=True,
            valid_json=True,
            target=None,
            target_exists=False,
            error="missing_target_field",
            step=int(payload.get("step", -1) or -1),
        )
    return CheckpointPointerHealth(
        pointer=pointer,
        exists=True,
        valid_json=True,
        target=target,
        target_exists=target.exists() and target.is_file(),
        error="" if target.exists() and target.is_file() else "target_missing",
        step=int(payload.get("step", extract_step(target)) or -1),
    )


def audit_checkpoint_pointers(
    project_root: str | Path,
    pointers: Iterable[str | Path],
) -> list[CheckpointPointerHealth]:
    return [inspect_checkpoint_pointer(project_root, pointer) for pointer in pointers]


def inspect_checkpoint_file(
    path: str | Path,
    *,
    include_zip_entries: bool = False,
) -> CheckpointFileHealth:
    checkpoint = Path(path)
    if not checkpoint.exists() or not checkpoint.is_file():
        return CheckpointFileHealth(
            path=checkpoint,
            exists=False,
            size_bytes=0,
            header_hex="",
            is_zip=False,
            status="missing",
            error="file_missing",
        )

    size = checkpoint.stat().st_size
    try:
        with checkpoint.open("rb") as handle:
            header = handle.read(8)
    except OSError as exc:
        return CheckpointFileHealth(
            path=checkpoint,
            exists=True,
            size_bytes=size,
            header_hex="",
            is_zip=False,
            status="read_error",
            step=extract_step(checkpoint),
            error=type(exc).__name__,
        )

    if size == 0:
        return CheckpointFileHealth(
            path=checkpoint,
            exists=True,
            size_bytes=size,
            header_hex=header.hex(),
            is_zip=False,
            status="empty",
            step=extract_step(checkpoint),
        )

    is_zip = zipfile.is_zipfile(checkpoint)
    if is_zip:
        entries: int | None = None
        if include_zip_entries:
            try:
                with zipfile.ZipFile(checkpoint) as archive:
                    entries = len(archive.namelist())
            except (OSError, zipfile.BadZipFile) as exc:
                return CheckpointFileHealth(
                    path=checkpoint,
                    exists=True,
                    size_bytes=size,
                    header_hex=header.hex(),
                    is_zip=True,
                    status="zip_read_error",
                    step=extract_step(checkpoint),
                    error=type(exc).__name__,
                )
        return CheckpointFileHealth(
            path=checkpoint,
            exists=True,
            size_bytes=size,
            header_hex=header.hex(),
            is_zip=True,
            status="torch_zip",
            step=extract_step(checkpoint),
            zip_entries=entries,
        )

    if header.startswith(b"PK"):
        status = "corrupt_zip"
        error = "zip_header_without_central_directory"
    elif header.startswith(b"\x80"):
        status = "legacy_pickle_or_raw"
        error = ""
    else:
        status = "unknown_format"
        error = "unrecognized_header"
    return CheckpointFileHealth(
        path=checkpoint,
        exists=True,
        size_bytes=size,
        header_hex=header.hex(),
        is_zip=False,
        status=status,
        step=extract_step(checkpoint),
        error=error,
    )


def audit_checkpoint_files(
    project_root: str | Path,
    roots: Iterable[str | Path],
    *,
    patterns: Iterable[str] = CHECKPOINT_PATTERNS,
) -> list[CheckpointFileHealth]:
    project = Path(project_root)
    seen: set[Path] = set()
    health: list[CheckpointFileHealth] = []
    for root_value in roots:
        root = _as_project_path(project, root_value)
        if not root.exists() or not root.is_dir():
            continue
        for pattern in patterns:
            for path in root.rglob(pattern):
                resolved = path.resolve()
                if resolved in seen or not path.is_file():
                    continue
                seen.add(resolved)
                health.append(inspect_checkpoint_file(path))
    return sorted(health, key=lambda item: str(item.path))


def extract_step(path: str | Path) -> int:
    name = Path(path).stem
    matches = re.findall(r"(\d+)", name)
    return int(matches[-1]) if matches else -1


def find_latest_checkpoint(
    project_root: str | Path,
    roots: Iterable[str | Path],
    *,
    patterns: Iterable[str] = CHECKPOINT_PATTERNS,
) -> CheckpointCandidate | None:
    """Find the newest checkpoint in known checkpoint roots by step, then mtime."""

    project = Path(project_root)
    seen: set[Path] = set()
    candidates: list[CheckpointCandidate] = []
    for root_value in roots:
        root = _as_project_path(project, root_value)
        if not root.exists() or not root.is_dir():
            continue
        for pattern in patterns:
            for path in root.rglob(pattern):
                resolved = path.resolve()
                if resolved in seen or not path.is_file():
                    continue
                seen.add(resolved)
                candidates.append(
                    CheckpointCandidate(
                        path=path,
                        step=extract_step(path),
                        mtime=path.stat().st_mtime,
                        source=str(root),
                    )
                )
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.step, item.mtime))


def resolve_latest_checkpoint(
    project_root: str | Path,
    *,
    pointers: Iterable[str | Path] = (),
    search_roots: Iterable[str | Path] = (),
) -> Path | None:
    """Resolve valid latest pointers first, then fall back to scanning roots."""

    root = Path(project_root)
    for pointer in pointers:
        checkpoint = resolve_checkpoint_pointer(root, pointer)
        if checkpoint is not None:
            return checkpoint
    latest = find_latest_checkpoint(root, search_roots)
    return None if latest is None else latest.path


def _current_source_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip().lower() if result.returncode == 0 else ""


def _tracked_worktree_clean(project_root: Path) -> bool:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and not result.stdout.strip()


def resolve_latest_organism_checkpoint(
    project_root: str | Path, *, require: bool = False
) -> Path | None:
    """Resolve only the checkpoint published by the canonical validated pointer.

    Runtime consumers must not infer lineage from cycle number or modification
    time.  An unpublished checkpoint can still be inspected through
    :func:`scan_latest_organism_checkpoint`, but it is never selected here.
    """
    checkpoint_root = resolve_organism_checkpoints_root(project_root, require=require)
    pointer = checkpoint_root / "organism_latest.json"
    if not pointer.is_file():
        if require:
            raise FileNotFoundError(f"Missing canonical organism pointer: {pointer}")
        return None
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if require:
            raise ValueError(f"Invalid canonical organism pointer JSON: {pointer}") from exc
        return None

    raw_path = payload.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        if require:
            raise ValueError("Canonical organism pointer has no path")
        return None
    checkpoint = (pointer.parent / raw_path).resolve()
    canonical_root = checkpoint_root.resolve()
    if checkpoint.parent != canonical_root or not checkpoint.is_file():
        if require:
            raise ValueError(
                "Canonical organism pointer target must be an existing checkpoint "
                f"directly under {canonical_root}"
            )
        return None

    errors: list[str] = []
    if payload.get("health") != "ok":
        errors.append("health")
    if int(payload.get("checkpoint_version", -1) or -1) < 7:
        errors.append("checkpoint_version")
    if int(payload.get("size_bytes", -1) or -1) != checkpoint.stat().st_size:
        errors.append("size_bytes")
    expected_sha = str(payload.get("sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        errors.append("sha256")
    base_checkpoint_id = str(payload.get("base_checkpoint_id", "")).lower()
    if not re.fullmatch(r"darwin-model-core-v1:[0-9a-f]{64}", base_checkpoint_id):
        errors.append("base_checkpoint_id")
    pointer_commit = str(payload.get("source_commit", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{40}", pointer_commit):
        errors.append("source_commit")
    project = Path(project_root).resolve()
    if pointer_commit != _current_source_commit(project) or not _tracked_worktree_clean(
        project
    ):
        errors.append("source_commit")
    cycle_match = re.fullmatch(r"organism_cycle_(\d+)(?:_step_\d+)?\.pt", checkpoint.name)
    if cycle_match is None or int(payload.get("cycle", -1) or -1) != int(
        cycle_match.group(1)
    ):
        errors.append("cycle")
    if int(payload.get("step", -1) or -1) < 0:
        errors.append("step")

    # The heavyweight verifier already hashed and inspected the checkpoint.
    # Runtime startup binds the pointer to that clean-HEAD operational proof
    # instead of re-reading a 10+ GB file every time Davi starts.
    lineage_path = checkpoint_root.parent / "04_MANIFESTOS" / "gold_local_lineage.json"
    try:
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        lineage = {}
        errors.append("gold_local_lineage")
    if lineage:
        if (
            lineage.get("status") != "ok"
            or lineage.get("tracked_worktree_clean") is not True
            or lineage.get("blockers")
            or str(lineage.get("source_commit", "")).lower() != pointer_commit
        ):
            errors.append("source_commit")

        published = lineage.get("pointer") or {}
        pointer_fields = (
            "path",
            "filename",
            "checkpoint_version",
            "cycle",
            "step",
            "size_bytes",
            "sha256",
            "base_checkpoint_id",
            "source_commit",
            "health",
        )
        if published.get("status") != "ok" or any(
            published.get(field) != payload.get(field) for field in pointer_fields
        ):
            errors.append("gold_pointer_contract")

        cycle_key = str(int(payload.get("cycle", -1) or -1))
        checkpoint_report = (lineage.get("checkpoints") or {}).get(cycle_key) or {}
        report_identity = checkpoint_report.get("identity") or {}
        report_training = checkpoint_report.get("training_state") or {}
        try:
            reported_checkpoint = Path(checkpoint_report.get("checkpoint", "")).resolve()
        except (OSError, TypeError):
            reported_checkpoint = Path()
        report_contract_valid = (
            checkpoint_report.get("gold_valid") is True
            and checkpoint_report.get("strict_resume_compatible") is True
            and checkpoint_report.get("identity_verified") is True
            and checkpoint_report.get("base_checkpoint_id") == base_checkpoint_id
            and report_identity.get("declared") == base_checkpoint_id
            and report_identity.get("with_raw_embedded_config") == base_checkpoint_id
            and checkpoint_report.get("file_sha256") == expected_sha
            and int(checkpoint_report.get("checkpoint_bytes", -1) or -1)
            == checkpoint.stat().st_size
            and int(checkpoint_report.get("checkpoint_version", -1) or -1)
            == int(payload.get("checkpoint_version", -1) or -1)
            and int(report_training.get("cycle", -1) or -1)
            == int(payload.get("cycle", -1) or -1)
            and int(report_training.get("step", -1) or -1)
            == int(payload.get("step", -1) or -1)
            and reported_checkpoint == checkpoint
        )
        if not report_contract_valid:
            errors.append("base_checkpoint_id")
    if errors:
        message = "Invalid canonical organism pointer fields: " + ", ".join(
            sorted(set(errors))
        )
        if require:
            raise ValueError(message)
        return None
    return checkpoint


def scan_latest_organism_checkpoint(
    project_root: str | Path, *, require: bool = False
) -> Path | None:
    """Inventory-only scan for the highest numbered organism checkpoint."""
    checkpoint_root = resolve_organism_checkpoints_root(project_root, require=require)
    candidates: list[tuple[int, int, float, Path]] = []
    for path in checkpoint_root.glob("organism_*.pt"):
        # Match both organism_cycle_N.pt and organism_cycle_N_step_SSSSSS.pt
        match = re.fullmatch(r"organism_cycle_(\d+)(?:_step_(\d+))?", path.stem)
        if match and path.is_file():
            cycle = int(match.group(1))
            step = int(match.group(2)) if match.group(2) is not None else 0
            candidates.append((cycle, step, path.stat().st_mtime, path))
    if candidates:
        # Prefer highest cycle, then highest step within that cycle
        return max(candidates, key=lambda item: (item[0], item[1], item[2]))[3]
    if require:
        raise FileNotFoundError(
            f"No organism_*.pt checkpoint found in workspace root: {checkpoint_root}"
        )
    return None
