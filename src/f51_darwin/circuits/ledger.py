"""Canonical append-only circuit ledger with externally anchored history."""

from __future__ import annotations

import ctypes
import json
import os
import re
import stat
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable

from f51_darwin.circuits.identity import canonical_json_bytes, canonical_sha256


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GENESIS_SHA256 = "0" * 64
_TRANSACTION_ANCHOR_FIELDS = frozenset(
    ("event_count", "event_tail_sha256", "transaction_manifest_sha256")
)
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class LedgerError(ValueError):
    """The append-only ledger or one of its external anchors is invalid."""


@dataclass(frozen=True)
class LedgerRecord:
    schema_version: int
    sequence: int
    previous_record_sha256: str
    event: Mapping[str, Any]
    record_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "previous_record_sha256": self.previous_record_sha256,
            "event": dict(self.event),
            "record_sha256": self.record_sha256,
        }


@dataclass
class _HeldNamespace:
    path: Path
    identity: tuple[int, int]
    descriptor: int
    lease: _ResourceLease | None = None
    locked: bool = False


@dataclass
class _HeldFileLock:
    path: Path
    identity: tuple[int, int]
    handle: BinaryIO
    lease: _ResourceLease | None = None
    locked: bool = False


@dataclass
class _PendingDescriptor:
    descriptor: int
    lease: _ResourceLease | None = None


@dataclass
class _PendingHandle:
    handle: Any | None
    closer: Callable[[Any], object]
    fallback_descriptor: int | None = None
    lease: _ResourceLease | None = None


@dataclass
class _ResourceLease:
    resource: (
        _PendingDescriptor
        | _PendingHandle
        | _HeldNamespace
        | _HeldFileLock
    )
    released: bool = False


_COORDINATOR_GUARD = threading.RLock()
_ACTIVE_COORDINATORS: set[tuple[int, int]] = set()
_FORK_GUARD = threading.RLock()
_HELD_RESOURCES: dict[int, _ResourceLease] = {}


def _before_fork() -> None:
    _FORK_GUARD.acquire()


def _after_fork_parent() -> None:
    _FORK_GUARD.release()


def _after_fork_child() -> None:
    global _COORDINATOR_GUARD, _FORK_GUARD
    for lease in tuple(_HELD_RESOURCES.values()):
        try:
            _release_lease(lease)
        except BaseException:
            pass
    _HELD_RESOURCES.clear()
    _COORDINATOR_GUARD = threading.RLock()
    _FORK_GUARD = threading.RLock()
    _ACTIVE_COORDINATORS.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_before_fork,
        after_in_parent=_after_fork_parent,
        after_in_child=_after_fork_child,
    )


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise LedgerError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _is_reparse_or_symlink(path: Path) -> bool:
    if not os.path.lexists(path):
        return False
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise LedgerError(f"cannot inspect path safety for {path}: {exc}") from exc
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT
    )


def _reject_reparse_ancestors(path: str | Path, *, label: str) -> Path:
    """Return an absolute lexical path only when no existing component is redirected."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    chain = [absolute, *absolute.parents]
    for component in reversed(chain):
        if _is_reparse_or_symlink(component):
            raise LedgerError(
                f"{label} contains a symlink or reparse-point ancestor: {component}"
            )
    return absolute


def _fsync_directory(path: str | Path) -> None:
    """Flush directory metadata or fail when the platform cannot provide durability."""

    directory = _reject_reparse_ancestors(path, label="fsync directory")
    if not directory.is_dir():
        raise OSError(f"fsync target is not a directory: {directory}")
    if os.name == "nt":
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        create_file.restype = wintypes.HANDLE
        with _FORK_GUARD:
            handle = create_file(
                str(directory),
                0x40000000,  # GENERIC_WRITE
                0x00000001 | 0x00000002 | 0x00000004,
                None,
                3,  # OPEN_EXISTING
                0x02000000 | 0x00200000,  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
                None,
            )
            invalid = wintypes.HANDLE(-1).value
            if handle == invalid:
                error = ctypes.get_last_error()
                raise OSError(error, f"cannot open directory for fsync: {directory}")
            pending_handle = _register_pending_handle(
                handle,
                kernel32.CloseHandle,
            )
        try:
            if not kernel32.FlushFileBuffers(handle):
                error = ctypes.get_last_error()
                raise OSError(error, f"cannot fsync directory: {directory}")
        finally:
            _release_lease(_require_lease(pending_handle))
        return

    pending = _open_fsync_directory_descriptor(directory)
    try:
        os.fsync(pending.descriptor)
    finally:
        _release_lease(_require_lease(pending))


def _validate_event(event: object) -> dict[str, Any]:
    if not isinstance(event, Mapping):
        raise LedgerError("ledger event must be a mapping")
    if any(not isinstance(key, str) for key in event):
        raise LedgerError("ledger event keys must be strings")
    copied = dict(event)
    try:
        canonical_json_bytes(copied)
    except (TypeError, ValueError) as exc:
        raise LedgerError("ledger event is not canonical JSON serializable") from exc

    present = _TRANSACTION_ANCHOR_FIELDS.intersection(copied)
    if present and present != _TRANSACTION_ANCHOR_FIELDS:
        raise LedgerError("transaction tail anchor is incomplete")
    if present:
        count = copied["event_count"]
        if type(count) is not int or count < 1:
            raise LedgerError("event_count must be a positive integer")
        _require_sha256(copied["event_tail_sha256"], "event_tail_sha256")
        _require_sha256(
            copied["transaction_manifest_sha256"],
            "transaction_manifest_sha256",
        )
    return copied


# Supported fork contract: each helper holds _FORK_GUARD from before the normal
# stdlib open/CreateFile call until it returns an already-registered lease.
# Python cannot recover a descriptor that arbitrary monkeypatched or signal
# code forks on inside os.open/CreateFile before that syscall wrapper returns;
# such hostile in-process execution is intentionally outside this contract.
def _open_lock_descriptor(
    path: Path,
    *,
    create: bool = True,
) -> _PendingDescriptor:
    if os.name != "nt":
        flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        if create:
            flags |= os.O_CREAT
        with _FORK_GUARD:
            descriptor = os.open(path, flags, 0o600)
            return _register_pending_descriptor(descriptor)

    from ctypes import wintypes
    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    with _FORK_GUARD:
        handle = create_file(
            str(path),
            0x80000000 | 0x40000000,  # GENERIC_READ | GENERIC_WRITE
            0x00000001 | 0x00000002,  # FILE_SHARE_READ | FILE_SHARE_WRITE
            None,
            4 if create else 3,  # OPEN_ALWAYS or OPEN_EXISTING
            0x00000080 | 0x00200000,  # NORMAL | OPEN_REPARSE_POINT
            None,
        )
        invalid = wintypes.HANDLE(-1).value
        if handle == invalid:
            error = ctypes.get_last_error()
            raise OSError(
                error,
                f"cannot open lock without following reparse point: {path}",
            )
        try:
            descriptor = msvcrt.open_osfhandle(
                handle,
                os.O_RDWR | getattr(os, "O_BINARY", 0),
            )
        except BaseException:
            kernel32.CloseHandle(handle)
            raise
        return _register_pending_descriptor(descriptor)


def _open_namespace_descriptor(path: Path) -> _PendingDescriptor:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    with _FORK_GUARD:
        descriptor = os.open(path, flags)
        return _register_pending_descriptor(descriptor)


def _open_append_descriptor(path: Path, flags: int) -> _PendingDescriptor:
    with _FORK_GUARD:
        descriptor = os.open(path, flags, 0o600)
        return _register_pending_descriptor(descriptor)


def _open_fsync_directory_descriptor(path: Path) -> _PendingDescriptor:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    with _FORK_GUARD:
        descriptor = os.open(path, flags)
        return _register_pending_descriptor(descriptor)


def _file_identity_from_descriptor(descriptor: int) -> tuple[int, int]:
    """Return the stable underlying file identity for an open descriptor."""

    if os.name != "nt":
        metadata = os.fstat(descriptor)
        return metadata.st_dev, metadata.st_ino

    from ctypes import wintypes
    import msvcrt

    class _ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_information = kernel32.GetFileInformationByHandle
    get_information.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ByHandleFileInformation),
    ]
    get_information.restype = wintypes.BOOL
    information = _ByHandleFileInformation()
    os_handle = wintypes.HANDLE(msvcrt.get_osfhandle(descriptor))
    if not get_information(os_handle, ctypes.byref(information)):
        error = ctypes.get_last_error()
        raise OSError(error, "cannot inspect lock file identity")
    file_index = (
        int(information.nFileIndexHigh) << 32
    ) | int(information.nFileIndexLow)
    return int(information.dwVolumeSerialNumber), file_index


def _verify_lock_path_identity(
    path: Path,
    expected_identity: tuple[int, int],
) -> None:
    """Verify that a lock pathname still names the held non-reparse file."""

    safe_path = _reject_reparse_ancestors(path, label="ledger lock")
    pending: _PendingDescriptor | None = None
    try:
        pending = _open_lock_descriptor(safe_path, create=False)
        descriptor = pending.descriptor
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or bool(
            getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT
        ):
            raise LedgerError("ledger lock must be a non-reparse regular file")
        if _file_identity_from_descriptor(descriptor) != expected_identity:
            raise LedgerError("ledger lock path identity changed")
    except LedgerError:
        raise
    except OSError as exc:
        raise LedgerError(
            f"cannot verify ledger lock path identity: {exc}"
        ) from exc
    finally:
        if pending is not None and pending.lease is not None:
            _release_lease(pending.lease)


def _verify_namespace_path(path: Path, identity: tuple[int, int]) -> None:
    safe_path = _reject_reparse_ancestors(
        path,
        label="ledger lock namespace",
    )
    try:
        metadata = os.stat(safe_path, follow_symlinks=False)
    except OSError as exc:
        raise LedgerError(f"cannot verify lock namespace: {exc}") from exc
    if (metadata.st_dev, metadata.st_ino) != identity:
        raise LedgerError("ledger lock namespace identity changed")


def _merge_release_errors(
    first: LedgerError | None,
    second: LedgerError | None,
) -> LedgerError | None:
    if first is None:
        return second
    if second is None:
        return first
    return LedgerError(f"{first}; additionally: {second}")


def _notify_release_integrity_error(
    error: LedgerError,
    callback: Callable[[], None] | None,
) -> LedgerError:
    if callback is None:
        return error
    try:
        callback()
    except BaseException as callback_exc:
        return LedgerError(
            f"{error}; release integrity callback also failed: {callback_exc}"
        )
    return error


def _enter_coordinator() -> tuple[int, int]:
    key = (os.getpid(), threading.get_ident())
    with _COORDINATOR_GUARD:
        if key in _ACTIVE_COORDINATORS:
            raise LedgerError("nested or reentrant lock coordinator is forbidden")
        _ACTIVE_COORDINATORS.add(key)
    return key


def _leave_coordinator(key: tuple[int, int]) -> None:
    with _COORDINATOR_GUARD:
        _ACTIVE_COORDINATORS.discard(key)


# These registrars take rollback ownership before mutating the registry. Their
# callers deliberately have no raw-close fallback: a failed __setitem__ may
# already have published the lease, so only lease-aware rollback is safe.
def _register_pending_descriptor(descriptor: int) -> _PendingDescriptor:
    with _FORK_GUARD:
        lease: _ResourceLease | None = None
        try:
            pending = _PendingDescriptor(descriptor)
            lease = _ResourceLease(pending)
            pending.lease = lease
            _HELD_RESOURCES[id(lease)] = lease
            return pending
        except BaseException:
            if lease is None:
                os.close(descriptor)
            else:
                _release_lease(lease)
            raise


def _register_pending_handle(
    handle: Any,
    closer: Callable[[Any], object],
) -> _PendingHandle:
    with _FORK_GUARD:
        lease: _ResourceLease | None = None
        try:
            pending = _PendingHandle(handle, closer)
            lease = _ResourceLease(pending)
            pending.lease = lease
            _HELD_RESOURCES[id(lease)] = lease
            return pending
        except BaseException:
            if lease is None:
                closer(handle)
            else:
                _release_lease(lease)
            raise


def _require_lease(
    resource: _PendingDescriptor | _PendingHandle | _HeldNamespace | _HeldFileLock,
) -> _ResourceLease:
    lease = resource.lease
    if lease is None or lease.released:
        raise LedgerError("resource ownership lease is not active")
    return lease


def _resource_is_owned(resource: _HeldNamespace | _HeldFileLock) -> bool:
    lease = resource.lease
    return (
        lease is not None
        and not lease.released
        and lease.resource is resource
    )


def _release_lease(lease: _ResourceLease) -> None:
    with _FORK_GUARD:
        if lease.released:
            return
        resource = lease.resource
        try:
            if isinstance(resource, _HeldFileLock):
                resource.handle.close()
                resource.locked = False
            elif isinstance(resource, _PendingHandle):
                if resource.handle is None:
                    if resource.fallback_descriptor is not None:
                        os.close(resource.fallback_descriptor)
                else:
                    resource.closer(resource.handle)
            else:
                os.close(resource.descriptor)
                if isinstance(resource, _HeldNamespace):
                    resource.locked = False
        finally:
            lease.released = True
            _HELD_RESOURCES.pop(id(lease), None)


def _after_lease_transfer(
    lease: _ResourceLease,
    resource: _HeldNamespace | _HeldFileLock,
) -> None:
    """Test hook at the exact point where the lease changes owners."""


def _close_binary_handle(handle: BinaryIO) -> None:
    handle.close()


def _after_fdopen_adoption(
    lease: _ResourceLease,
    pending: _PendingHandle,
) -> None:
    """Test hook after fdopen ownership is adopted but before promotion."""


def _fdopen_registered_handle(
    pending: _PendingDescriptor,
) -> _PendingHandle:
    with _FORK_GUARD:
        lease = _require_lease(pending)
        if lease.resource is not pending:
            raise LedgerError("pending descriptor registration was lost")
        adopted = _PendingHandle(
            None,
            _close_binary_handle,
            fallback_descriptor=pending.descriptor,
            lease=lease,
        )
        lease.resource = adopted
        try:
            handle = os.fdopen(
                pending.descriptor,
                "r+b",
                buffering=0,
            )
            adopted.handle = handle
            adopted.fallback_descriptor = None
            _after_fdopen_adoption(lease, adopted)
            return adopted
        except BaseException:
            _release_lease(lease)
            raise


def _transfer_pending_descriptor(
    pending: _PendingDescriptor | _PendingHandle,
    resource: _HeldNamespace | _HeldFileLock,
) -> None:
    with _FORK_GUARD:
        lease = _require_lease(pending)
        if lease.resource is not pending:
            raise LedgerError("pending descriptor registration was lost")
        resource.lease = lease
        lease.resource = resource
        _after_lease_transfer(lease, resource)


def _close_file_resource(resource: _HeldFileLock) -> None:
    lease = resource.lease
    if lease is not None:
        _release_lease(lease)


def _close_namespace_resource(resource: _HeldNamespace) -> None:
    lease = resource.lease
    if lease is not None:
        _release_lease(lease)


def _unlock_file(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _lock_file(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _acquire_parent_namespaces(parents: tuple[Path, ...]) -> list[_HeldNamespace]:
    if os.name == "nt":
        return []
    import fcntl

    held: list[_HeldNamespace] = []
    try:
        for parent in parents:
            resource: _HeldNamespace | None = None
            pending: _PendingDescriptor | None = None
            try:
                pending = _open_namespace_descriptor(parent)
                descriptor = pending.descriptor
                metadata = os.fstat(descriptor)
                if not stat.S_ISDIR(metadata.st_mode):
                    raise LedgerError(
                        "ledger lock namespace must be a directory"
                    )
                identity = (metadata.st_dev, metadata.st_ino)
                resource = _HeldNamespace(parent, identity, descriptor)
                _transfer_pending_descriptor(pending, resource)
                held.append(resource)
            except BaseException:
                if pending is not None and pending.lease is not None:
                    _release_lease(pending.lease)
                raise
        identities: set[tuple[int, int]] = set()
        unique: list[_HeldNamespace] = []
        for resource in sorted(held, key=lambda item: item.identity):
            if resource.identity in identities:
                _close_namespace_resource(resource)
                continue
            identities.add(resource.identity)
            unique.append(resource)
        held = unique
        for resource in held:
            fcntl.flock(resource.descriptor, fcntl.LOCK_EX)
            resource.locked = True
            _verify_namespace_path(resource.path, resource.identity)
        return held
    except BaseException:
        for namespace in reversed(held):
            try:
                if namespace.locked:
                    fcntl.flock(namespace.descriptor, fcntl.LOCK_UN)
                    namespace.locked = False
            except OSError:
                pass
            try:
                _close_namespace_resource(namespace)
            except OSError:
                pass
        raise


def _acquire_file_locks(paths: tuple[Path, ...]) -> list[_HeldFileLock]:
    opened: list[_HeldFileLock] = []
    identities: set[tuple[int, int]] = set()
    try:
        for path in paths:
            resource: _HeldFileLock | None = None
            pending: _PendingDescriptor | None = None
            try:
                pending = _open_lock_descriptor(path)
                descriptor = pending.descriptor
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or bool(
                    getattr(metadata, "st_file_attributes", 0)
                    & _REPARSE_POINT
                ):
                    raise LedgerError(
                        "ledger lock must be a non-reparse regular file"
                    )
                identity = _file_identity_from_descriptor(descriptor)
                if identity in identities:
                    raise LedgerError(
                        "coordinated lock paths alias the same file"
                    )
                identities.add(identity)
                if metadata.st_size == 0:
                    if os.write(descriptor, b"\0") != 1:
                        raise LedgerError(
                            "ledger lock initialization was partial"
                        )
                    os.fsync(descriptor)
                pending_handle = _fdopen_registered_handle(pending)
                handle = pending_handle.handle
                if handle is None:
                    raise LedgerError("fdopen ownership adoption failed")
                resource = _HeldFileLock(path, identity, handle)
                _transfer_pending_descriptor(pending_handle, resource)
                opened.append(resource)
            except BaseException:
                if pending is not None and pending.lease is not None:
                    _release_lease(pending.lease)
                raise
        opened.sort(key=lambda item: item.identity)
        for item in opened:
            _lock_file(item.handle)
            item.locked = True
            _verify_lock_path_identity(item.path, item.identity)
        return opened
    except BaseException:
        for item in reversed(opened):
            try:
                if item.locked:
                    _unlock_file(item.handle)
                    item.locked = False
            except OSError:
                pass
            try:
                _close_file_resource(item)
            except OSError:
                pass
        raise


@contextmanager
def _exclusive_locks(
    lock_paths: Iterator[Path] | tuple[Path, ...] | list[Path],
    *,
    on_release_integrity_error: Callable[[], None] | None = None,
) -> Iterator[None]:
    key = _enter_coordinator()
    files: list[_HeldFileLock] = []
    namespaces: list[_HeldNamespace] = []
    try:
        normalized_paths = tuple(
            _reject_reparse_ancestors(path, label="ledger lock")
            for path in lock_paths
        )
        if len(set(normalized_paths)) != len(normalized_paths):
            raise LedgerError("coordinated lock paths alias the same file")
        safe_paths = tuple(
            sorted(
                normalized_paths,
                key=lambda path: os.fsencode(str(path)),
            )
        )
        if not safe_paths:
            raise LedgerError("lock coordinator requires at least one path")
        if any(not path.parent.is_dir() for path in safe_paths):
            raise LedgerError("ledger lock parent must be an existing directory")
        parents = tuple(
            sorted(
                {path.parent for path in safe_paths},
                key=lambda path: os.fsencode(str(path)),
            )
        )
        namespaces = _acquire_parent_namespaces(parents)
        files = _acquire_file_locks(safe_paths)
    except BaseException as exc:
        for namespace in reversed(namespaces):
            try:
                if namespace.locked and os.name != "nt":
                    import fcntl

                    fcntl.flock(namespace.descriptor, fcntl.LOCK_UN)
                    namespace.locked = False
            except OSError:
                pass
            try:
                _close_namespace_resource(namespace)
            except OSError:
                pass
        _leave_coordinator(key)
        if isinstance(exc, OSError):
            raise LedgerError(f"cannot acquire coordinated locks: {exc}") from exc
        raise

    release_error: LedgerError | None = None
    try:
        yield
    finally:
        for item in files:
            try:
                _verify_lock_path_identity(item.path, item.identity)
            except LedgerError as exc:
                release_error = _merge_release_errors(release_error, exc)
        for namespace in namespaces:
            try:
                _verify_namespace_path(namespace.path, namespace.identity)
            except LedgerError as exc:
                release_error = _merge_release_errors(release_error, exc)
        if release_error is not None:
            release_error = _notify_release_integrity_error(
                release_error,
                on_release_integrity_error,
            )

        for item in reversed(files):
            try:
                if _resource_is_owned(item) and item.locked:
                    _unlock_file(item.handle)
                    item.locked = False
            except (OSError, ValueError) as exc:
                error = LedgerError(f"cannot unlock ledger lock: {exc}")
                error = _notify_release_integrity_error(
                    error,
                    on_release_integrity_error,
                )
                release_error = _merge_release_errors(release_error, error)
        for item in reversed(files):
            try:
                _close_file_resource(item)
            except OSError as exc:
                error = LedgerError(f"cannot close ledger lock: {exc}")
                error = _notify_release_integrity_error(
                    error,
                    on_release_integrity_error,
                )
                release_error = _merge_release_errors(release_error, error)
        if os.name != "nt":
            import fcntl

            for namespace in reversed(namespaces):
                try:
                    if _resource_is_owned(namespace) and namespace.locked:
                        fcntl.flock(namespace.descriptor, fcntl.LOCK_UN)
                        namespace.locked = False
                except OSError as exc:
                    error = LedgerError(f"cannot unlock lock namespace: {exc}")
                    error = _notify_release_integrity_error(
                        error,
                        on_release_integrity_error,
                    )
                    release_error = _merge_release_errors(release_error, error)
                try:
                    _close_namespace_resource(namespace)
                except OSError as exc:
                    error = LedgerError(f"cannot close lock namespace: {exc}")
                    release_error = _merge_release_errors(release_error, error)
        else:
            for namespace in reversed(namespaces):
                try:
                    _close_namespace_resource(namespace)
                except OSError as exc:
                    error = LedgerError(f"cannot close lock namespace: {exc}")
                    release_error = _merge_release_errors(release_error, error)
        _leave_coordinator(key)
        if release_error is not None:
            raise release_error


@contextmanager
def _exclusive_lock(lock_path: Path) -> Iterator[None]:
    with _exclusive_locks((lock_path,)):
        yield


class CircuitLedger:
    """A canonical JSONL hash chain whose terminal digest and count are trusted."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_tail_sha256: str | None = None,
        expected_record_count: int | None = None,
    ) -> None:
        self.path = Path(path)
        if (expected_tail_sha256 is None) != (expected_record_count is None):
            raise LedgerError("trusted ledger tail and count must be provided together")
        if expected_tail_sha256 is not None:
            _require_sha256(expected_tail_sha256, "expected ledger tail")
            if type(expected_record_count) is not int or expected_record_count < 0:
                raise LedgerError("expected ledger record count must be non-negative")
            if expected_record_count == 0 and expected_tail_sha256 != _GENESIS_SHA256:
                raise LedgerError("empty ledger must use the genesis tail")
            if expected_record_count > 0 and expected_tail_sha256 == _GENESIS_SHA256:
                raise LedgerError("nonempty ledger cannot use the genesis tail")
        self._trusted_tail_sha256 = expected_tail_sha256
        self._trusted_record_count = expected_record_count

    @property
    def trusted_record_count(self) -> int | None:
        return self._trusted_record_count

    @staticmethod
    def _decode_line(line: bytes, sequence: int) -> LedgerRecord:
        if not line.endswith(b"\n") or line == b"\n":
            raise LedgerError("ledger contains a truncated or empty record")
        content = line[:-1]
        try:
            payload = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LedgerError("ledger record is malformed") from exc
        if (
            not isinstance(payload, dict)
            or canonical_json_bytes(payload) != content
            or set(payload)
            != {
                "schema_version",
                "sequence",
                "previous_record_sha256",
                "event",
                "record_sha256",
            }
        ):
            raise LedgerError("ledger record is not canonical or has invalid fields")
        event = _validate_event(payload["event"])
        unsigned = {
            "schema_version": 1,
            "sequence": sequence,
            "previous_record_sha256": payload["previous_record_sha256"],
            "event": event,
        }
        expected_record_sha256 = canonical_sha256(unsigned)
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or type(payload["sequence"]) is not int
            or payload["sequence"] != sequence
            or not isinstance(payload["previous_record_sha256"], str)
            or _SHA256.fullmatch(payload["previous_record_sha256"]) is None
            or not isinstance(payload["record_sha256"], str)
            or _SHA256.fullmatch(payload["record_sha256"]) is None
            or payload["record_sha256"] != expected_record_sha256
        ):
            raise LedgerError("ledger record identity is invalid")
        return LedgerRecord(
            schema_version=1,
            sequence=sequence,
            previous_record_sha256=payload["previous_record_sha256"],
            event=event,
            record_sha256=payload["record_sha256"],
        )

    def _lock_path(self) -> Path:
        return self.path.with_name(f".{self.path.name}.lock")

    def _records_unlocked(self) -> tuple[LedgerRecord, ...]:
        safe_path = _reject_reparse_ancestors(self.path, label="ledger path")
        if not os.path.lexists(safe_path):
            return ()
        if not safe_path.is_file():
            raise LedgerError("ledger path must be a regular file")
        try:
            lines = safe_path.read_bytes().splitlines(keepends=True)
        except OSError as exc:
            raise LedgerError(f"cannot read ledger: {exc}") from exc
        records: list[LedgerRecord] = []
        previous = _GENESIS_SHA256
        for sequence, line in enumerate(lines):
            record = self._decode_line(line, sequence)
            if record.previous_record_sha256 != previous:
                raise LedgerError("ledger hash chain is broken")
            records.append(record)
            previous = record.record_sha256
        return tuple(records)

    def _verify_records(
        self,
        records: tuple[LedgerRecord, ...],
        *,
        require_anchor: bool,
    ) -> str:
        tail = records[-1].record_sha256 if records else _GENESIS_SHA256
        if records and require_anchor and self._trusted_tail_sha256 is None:
            raise LedgerError(
                "nonempty ledger requires a trusted external tail and count anchor"
            )
        if self._trusted_tail_sha256 is not None and (
            tail != self._trusted_tail_sha256
            or len(records) != self._trusted_record_count
        ):
            raise LedgerError(
                "ledger tail or count does not match the external trusted anchor"
            )
        return tail

    def records(self) -> tuple[LedgerRecord, ...]:
        with _exclusive_lock(self._lock_path()):
            return self._records_unlocked()

    def verified_records(self) -> tuple[LedgerRecord, ...]:
        with _exclusive_lock(self._lock_path()):
            records = self._records_unlocked()
            self._verify_records(records, require_anchor=True)
            return records

    def verify(self) -> str:
        records = self.verified_records()
        return records[-1].record_sha256 if records else _GENESIS_SHA256

    def append(self, event: Mapping[str, Any]) -> LedgerRecord:
        validated_event = _validate_event(event)
        with _exclusive_lock(self._lock_path()):
            records = self._records_unlocked()
            previous = self._verify_records(records, require_anchor=True)
            sequence = len(records)
            unsigned = {
                "schema_version": 1,
                "sequence": sequence,
                "previous_record_sha256": previous,
                "event": validated_event,
            }
            record = LedgerRecord(
                schema_version=1,
                sequence=sequence,
                previous_record_sha256=previous,
                event=validated_event,
                record_sha256=canonical_sha256(unsigned),
            )
            safe_path = _reject_reparse_ancestors(self.path, label="ledger path")
            if not safe_path.parent.is_dir():
                raise LedgerError("ledger parent must be an existing real directory")
            flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
            flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
            pending: _PendingDescriptor | None = None
            try:
                pending = _open_append_descriptor(safe_path, flags)
                descriptor = pending.descriptor
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or bool(
                    getattr(metadata, "st_file_attributes", 0)
                    & _REPARSE_POINT
                ):
                    raise LedgerError(
                        "ledger path must be a non-reparse regular file"
                    )
                content = canonical_json_bytes(record.to_dict()) + b"\n"
                if os.write(descriptor, content) != len(content):
                    raise LedgerError("ledger append was partial")
                os.fsync(descriptor)
            except LedgerError:
                raise
            except OSError as exc:
                raise LedgerError(
                    f"cannot append ledger record: {exc}"
                ) from exc
            finally:
                if pending is not None and pending.lease is not None:
                    _release_lease(pending.lease)
            try:
                _fsync_directory(safe_path.parent)
            except (LedgerError, OSError) as exc:
                raise LedgerError(f"cannot fsync ledger parent directory: {exc}") from exc
            self._trusted_tail_sha256 = record.record_sha256
            self._trusted_record_count = sequence + 1
            observed = self._records_unlocked()
            if (
                self._verify_records(observed, require_anchor=True)
                != record.record_sha256
            ):
                raise LedgerError("ledger append read-back failed")
            return record


__all__ = ["CircuitLedger", "LedgerError", "LedgerRecord"]
