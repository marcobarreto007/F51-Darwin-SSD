"""Crash-safe compare-and-swap publication for the active circuit checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from f51_darwin.circuits.identity import canonical_json_bytes
from f51_darwin.circuits.ledger import (
    CircuitLedger,
    LedgerError,
    LedgerRecord,
    _exclusive_lock,
    _exclusive_locks,
    _file_identity_from_descriptor,
    _fsync_directory,
    _reject_reparse_ancestors,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GENESIS_SHA256 = "0" * 64
_POINTER_NAME = "active.json"


class PointerError(ValueError):
    """The active pointer or a publication candidate is invalid."""


class PointerConflict(PointerError):
    """The active checkpoint no longer matches the caller's expected parent."""


@dataclass(frozen=True)
class CheckpointCandidate:
    """A checkpoint whose file must remain immutable after successful publication."""

    path: Path
    sha256: str
    ledger_path: Path
    ledger_sha256: str
    ledger_count: int


@dataclass(frozen=True)
class ActivePointer:
    checkpoint: str
    checkpoint_sha256: str
    parent_checkpoint_sha256: str
    ledger_sha256: str
    schema_version: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
            "ledger_sha256": self.ledger_sha256,
        }


@dataclass(frozen=True)
class _CheckpointBinding:
    identity: tuple[int, int]
    size: int
    mtime_ns: int
    ctime_ns: int
    mode: int


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PointerError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PointerError(f"cannot hash checkpoint: {exc}") from exc
    return digest.hexdigest()


def _validate_checkpoint(path: Path, expected_sha256: str) -> Path:
    _require_sha256(expected_sha256, "checkpoint SHA-256")
    try:
        safe_path = _reject_reparse_ancestors(path, label="checkpoint path")
    except LedgerError as exc:
        raise PointerError(str(exc)) from exc
    if not safe_path.is_file():
        raise PointerError("checkpoint must be a regular file")
    if _sha256_file(safe_path) != expected_sha256:
        raise PointerError("checkpoint SHA-256 mismatch")
    return safe_path


def _checkpoint_binding(path: Path, expected_sha256: str) -> _CheckpointBinding:
    """Hash an opened checkpoint and bind its identity plus mutation metadata."""

    _require_sha256(expected_sha256, "checkpoint SHA-256")
    try:
        safe_path = _reject_reparse_ancestors(path, label="checkpoint path")
    except LedgerError as exc:
        raise PointerError(str(exc)) from exc
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(safe_path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or bool(
            getattr(before, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise PointerError("checkpoint must be a non-reparse regular file")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        before_binding = _CheckpointBinding(
            identity=_file_identity_from_descriptor(descriptor),
            size=before.st_size,
            mtime_ns=before.st_mtime_ns,
            ctime_ns=before.st_ctime_ns,
            mode=before.st_mode,
        )
        after_binding = _CheckpointBinding(
            identity=_file_identity_from_descriptor(descriptor),
            size=after.st_size,
            mtime_ns=after.st_mtime_ns,
            ctime_ns=after.st_ctime_ns,
            mode=after.st_mode,
        )
        if before_binding != after_binding:
            raise PointerError("checkpoint changed while it was being validated")
        if digest.hexdigest() != expected_sha256:
            raise PointerError("checkpoint SHA-256 mismatch")
        return after_binding
    except PointerError:
        raise
    except OSError as exc:
        raise PointerError(f"cannot bind checkpoint identity: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _decode_pointer(content: bytes) -> ActivePointer:
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PointerError("active pointer is malformed") from exc
    if (
        not isinstance(payload, dict)
        or canonical_json_bytes(payload) != content
        or set(payload)
        != {
            "schema_version",
            "checkpoint",
            "checkpoint_sha256",
            "parent_checkpoint_sha256",
            "ledger_sha256",
        }
    ):
        raise PointerError("active pointer is not canonical or has invalid fields")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise PointerError("unsupported active pointer schema")
    if not isinstance(payload["checkpoint"], str) or not Path(
        payload["checkpoint"]
    ).is_absolute():
        raise PointerError("active pointer checkpoint must be an absolute path")
    _require_sha256(payload["checkpoint_sha256"], "checkpoint SHA-256")
    _require_sha256(payload["parent_checkpoint_sha256"], "parent checkpoint SHA-256")
    _require_sha256(payload["ledger_sha256"], "ledger SHA-256")
    pointer = ActivePointer(
        checkpoint=payload["checkpoint"],
        checkpoint_sha256=payload["checkpoint_sha256"],
        parent_checkpoint_sha256=payload["parent_checkpoint_sha256"],
        ledger_sha256=payload["ledger_sha256"],
    )
    if str(_validate_checkpoint(Path(pointer.checkpoint), pointer.checkpoint_sha256)) != (
        pointer.checkpoint
    ):
        raise PointerError("active pointer checkpoint path is not canonical")
    return pointer


def _validate_root(root: Path) -> Path:
    try:
        safe_root = _reject_reparse_ancestors(root, label="pointer root")
    except LedgerError as exc:
        raise PointerError(str(exc)) from exc
    if not safe_root.is_dir():
        raise PointerError("pointer root must be an existing real directory")
    return safe_root


@contextmanager
def _publication_lock(
    root: Path,
    ledger_lock: Path | None = None,
    on_release_integrity_error: Callable[[], None] | None = None,
) -> Iterator[None]:
    """Serialize validation, compare and replace using a non-redirectable lock."""

    lock_paths = [root / ".active.lock"]
    if ledger_lock is not None:
        lock_paths.append(ledger_lock)
    try:
        with _exclusive_locks(
            lock_paths,
            on_release_integrity_error=on_release_integrity_error,
        ):
            yield
    except LedgerError as exc:
        raise PointerError(f"active pointer lock rejected: {exc}") from exc


def _read_pointer_bytes_unlocked(root: Path) -> bytes:
    pointer_path = root / _POINTER_NAME
    try:
        safe_pointer = _reject_reparse_ancestors(
            pointer_path,
            label="active pointer path",
        )
    except LedgerError as exc:
        raise PointerError(str(exc)) from exc
    if not safe_pointer.is_file():
        raise PointerError("active pointer is missing or is not a regular file")
    try:
        return safe_pointer.read_bytes()
    except OSError as exc:
        raise PointerError(f"cannot read active pointer: {exc}") from exc


def _read_active_pointer_unlocked(root: Path) -> tuple[ActivePointer, bytes]:
    content = _read_pointer_bytes_unlocked(root)
    return _decode_pointer(content), content


def read_active_pointer(root: str | Path) -> ActivePointer:
    root_path = _validate_root(Path(root))
    with _publication_lock(root_path):
        pointer, _ = _read_active_pointer_unlocked(root_path)
        return pointer


def _verified_candidate_records_unlocked(
    ledger: CircuitLedger,
    ledger_sha256: str,
    ledger_count: int,
) -> tuple[LedgerRecord, ...]:
    if type(ledger_count) is not int or ledger_count < 1:
        raise PointerError("candidate ledger count must be a positive integer")
    records = ledger._records_unlocked()
    ledger._verify_records(records, require_anchor=True)
    if len(records) != ledger_count or records[-1].record_sha256 != ledger_sha256:
        raise PointerError("candidate ledger tail or count mismatch")
    return records


def _require_ledger_ancestor(
    current_tail_sha256: str,
    candidate_records: tuple[LedgerRecord, ...],
) -> None:
    candidate_hashes = [
        getattr(record, "record_sha256", None) for record in candidate_records
    ]
    if current_tail_sha256 not in candidate_hashes:
        raise PointerError(
            "current ledger tail is not an ancestor/prefix of candidate history"
        )


def _replace_pointer_bytes(root: Path, content: bytes) -> None:
    pointer_path = root / _POINTER_NAME
    temporary = root / f".{_POINTER_NAME}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, pointer_path)
        _fsync_directory(root)
    finally:
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:
            pass


def _restore_previous_pointer(
    root: Path,
    content: bytes | None,
    expected: ActivePointer | None,
) -> None:
    pointer_path = root / _POINTER_NAME
    if content is None:
        if os.path.lexists(pointer_path):
            pointer_path.unlink()
            _fsync_directory(root)
        if os.path.lexists(pointer_path):
            raise PointerError("failed to restore absent active pointer")
        return
    _replace_pointer_bytes(root, content)
    restored, restored_bytes = _read_active_pointer_unlocked(root)
    if restored_bytes != content or restored != expected:
        raise PointerError("restored active pointer failed exact validation")


def publish_active_pointer(
    root: str | Path,
    expected_parent_sha256: str,
    candidate: CheckpointCandidate,
) -> ActivePointer:
    """Publish a candidate that remains immutable for the pointer's lifetime.

    Publication is compare-and-swap under both locks. Callers must never mutate,
    replace or delete a checkpoint after successful publication.
    """

    root_path = _validate_root(Path(root))
    expected_parent = _require_sha256(
        expected_parent_sha256,
        "expected parent checkpoint SHA-256",
    )
    try:
        candidate_path = Path(candidate.path)
        candidate_sha256 = candidate.sha256
        ledger_path = Path(candidate.ledger_path)
        ledger_sha256 = candidate.ledger_sha256
        ledger_count = candidate.ledger_count
    except (AttributeError, TypeError) as exc:
        raise PointerError(
            "candidate does not satisfy the checkpoint contract"
        ) from exc
    _require_sha256(ledger_sha256, "ledger SHA-256")
    candidate_ledger = CircuitLedger(
        ledger_path,
        expected_tail_sha256=ledger_sha256,
        expected_record_count=ledger_count,
    )
    ledger_lock = candidate_ledger._lock_path()
    pointer_replaced = False
    initial_pointer_bytes: bytes | None = None
    current: ActivePointer | None = None
    temporary: Path | None = None

    def restore_on_release_integrity_error() -> None:
        if pointer_replaced:
            failures: list[BaseException] = []
            for _ in range(2):
                try:
                    _restore_previous_pointer(
                        root_path,
                        initial_pointer_bytes,
                        current,
                    )
                    return
                except BaseException as restore_exc:
                    failures.append(restore_exc)
            raise PointerError(
                "exact prior pointer restoration failed twice: "
                + "; ".join(str(item) for item in failures)
            )

    with _publication_lock(
        root_path,
        ledger_lock,
        restore_on_release_integrity_error,
    ):
        try:
            with nullcontext():
                resolved_candidate = _validate_checkpoint(
                    candidate_path,
                    candidate_sha256,
                )
                initial_candidate_binding = _checkpoint_binding(
                    resolved_candidate,
                    candidate_sha256,
                )
                candidate_records = _verified_candidate_records_unlocked(
                    candidate_ledger,
                    ledger_sha256,
                    ledger_count,
                )

                pointer_path = root_path / _POINTER_NAME
                if os.path.lexists(pointer_path):
                    current, initial_pointer_bytes = _read_active_pointer_unlocked(
                        root_path
                    )
                    current_sha256 = current.checkpoint_sha256
                    _require_ledger_ancestor(
                        current.ledger_sha256,
                        candidate_records,
                    )
                else:
                    current = None
                    initial_pointer_bytes = None
                    current_sha256 = _GENESIS_SHA256
                if current_sha256 != expected_parent:
                    raise PointerConflict(
                        "active checkpoint does not match expected parent SHA-256"
                    )

                pointer = ActivePointer(
                    checkpoint=str(resolved_candidate),
                    checkpoint_sha256=candidate_sha256,
                    parent_checkpoint_sha256=expected_parent,
                    ledger_sha256=ledger_sha256,
                )
                temporary = root_path / (
                    f".{_POINTER_NAME}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
                )
                with temporary.open("xb") as handle:
                    handle.write(canonical_json_bytes(pointer.to_dict()))
                    handle.flush()
                    os.fsync(handle.fileno())

                final_records = _verified_candidate_records_unlocked(
                    candidate_ledger,
                    ledger_sha256,
                    ledger_count,
                )
                if tuple(
                    record.record_sha256 for record in final_records
                ) != tuple(record.record_sha256 for record in candidate_records):
                    raise PointerError(
                        "candidate ledger bytes changed before publication"
                    )
                if initial_pointer_bytes is None:
                    if os.path.lexists(pointer_path):
                        raise PointerConflict(
                            "active pointer appeared during publication"
                        )
                else:
                    final_current, final_pointer_bytes = (
                        _read_active_pointer_unlocked(root_path)
                    )
                    if (
                        final_pointer_bytes != initial_pointer_bytes
                        or final_current != current
                    ):
                        raise PointerConflict(
                            "active pointer changed during publication"
                        )

                # Bind path identity and bytes immediately before pointer replacement.
                final_candidate = _validate_checkpoint(
                    candidate_path,
                    candidate_sha256,
                )
                final_candidate_binding = _checkpoint_binding(
                    final_candidate,
                    candidate_sha256,
                )
                if final_candidate_binding != initial_candidate_binding:
                    raise PointerError(
                        "candidate checkpoint identity or stat changed"
                    )
                os.replace(temporary, pointer_path)
                pointer_replaced = True
                try:
                    _fsync_directory(root_path)
                except (LedgerError, OSError) as exc:
                    raise PointerError(
                        f"cannot fsync active pointer directory: {exc}"
                    ) from exc

                _validate_checkpoint(candidate_path, candidate_sha256)
                published_binding = _checkpoint_binding(
                    candidate_path,
                    candidate_sha256,
                )
                if published_binding != final_candidate_binding:
                    raise PointerError(
                        "candidate checkpoint changed during pointer publication"
                    )
                published, published_bytes = _read_active_pointer_unlocked(root_path)
                expected_pointer_bytes = canonical_json_bytes(pointer.to_dict())
                if published != pointer or published_bytes != expected_pointer_bytes:
                    raise PointerError("published active pointer failed validation")
        except BaseException as exc:
            if pointer_replaced:
                try:
                    _restore_previous_pointer(
                        root_path,
                        initial_pointer_bytes,
                        current,
                    )
                    pointer_replaced = False
                except BaseException as restore_exc:
                    raise PointerError(
                        "publication failed and prior pointer restoration failed: "
                        f"{restore_exc}"
                    ) from restore_exc
            try:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
            except OSError:
                pass
            if isinstance(exc, LedgerError):
                raise PointerError(f"ledger verification failed: {exc}") from exc
            if isinstance(
                exc,
                (KeyboardInterrupt, SystemExit, PointerError),
            ):
                raise
            raise PointerError(f"cannot replace active pointer: {exc}") from exc
        assert published is not None
        return published


__all__ = [
    "ActivePointer",
    "CheckpointCandidate",
    "PointerConflict",
    "PointerError",
    "publish_active_pointer",
    "read_active_pointer",
]
