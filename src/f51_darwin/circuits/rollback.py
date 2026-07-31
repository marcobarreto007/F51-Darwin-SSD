"""Immutable, identity-checked selective circuit rollback checkpoints."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import io
import json
import os
import re
import sys
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from f51_darwin.circuits.identity import canonical_json_bytes, canonical_sha256
from f51_darwin.circuits.ledger import (
    CircuitLedger,
    LedgerError,
    _fsync_directory,
    _reject_reparse_ancestors,
)
from f51_darwin.circuits.package import CircuitPackageError, verify_circuit_package
from f51_darwin.circuits.pointer import CheckpointCandidate


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CIRCUIT_ID = re.compile(r"^f51-circuit-v1:[0-9a-f]{64}$")
_GENESIS_SHA256 = "0" * 64
_TRANSACTION_TRANSITIONS = {
    "candidate": frozenset(("shadow",)),
    "shadow": frozenset(("adapter_active",)),
    "adapter_active": frozenset(("organ_unfrozen", "quarantine")),
    "organ_unfrozen": frozenset(("active", "quarantine")),
    "active": frozenset(("frozen", "quarantine")),
    "frozen": frozenset(),
    "quarantine": frozenset(("rollback",)),
    "rollback": frozenset(),
}
_REGISTRY_FIELDS = frozenset(
    (
        "state_prefix",
        "package_path",
        "package_sha256",
        "tap_contract_path",
        "tap_contract_sha256",
        "preinstall_snapshot_sha256",
        "transaction_root",
        "transaction_manifest_sha256",
        "event_count",
        "event_tail_sha256",
    )
)
_SNAPSHOT_FIELDS = frozenset(
    (
        "schema_version",
        "circuit_id",
        "state_prefix",
        "circuit_state",
        "package_path",
        "package_sha256",
        "tap_contract_path",
        "tap_contract_sha256",
    )
)


class CircuitRollbackError(ValueError):
    """Rollback evidence or the resulting checkpoint failed strict verification."""


@dataclass(frozen=True)
class CircuitSnapshot:
    path: Path
    sha256: str


@dataclass(frozen=True)
class CircuitCheckpoint(CheckpointCandidate):
    model_factory: Callable[[], nn.Module]
    probe: object


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CircuitRollbackError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _sha256_file(path: Path, label: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CircuitRollbackError(f"cannot hash {label}: {exc}") from exc
    return digest.hexdigest()


def _verified_regular_file(path: Path, expected_sha256: str, label: str) -> Path:
    _require_sha256(expected_sha256, f"{label} SHA-256")
    try:
        resolved = _reject_reparse_ancestors(path, label=f"{label} path")
    except LedgerError as exc:
        raise CircuitRollbackError(str(exc)) from exc
    if not resolved.is_file():
        raise CircuitRollbackError(f"{label} must be a regular file")
    if _sha256_file(resolved, label) != expected_sha256:
        raise CircuitRollbackError(f"{label} SHA-256 mismatch")
    return resolved


def _load_torch_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CircuitRollbackError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise CircuitRollbackError(f"{label} payload must be a dictionary")
    return payload


def _load_verified_torch_mapping(
    path: Path,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise CircuitRollbackError(f"{label} is unreadable: {exc}") from exc
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise CircuitRollbackError(f"{label} SHA-256 mismatch during load")
    try:
        payload = torch.load(
            io.BytesIO(content),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise CircuitRollbackError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise CircuitRollbackError(f"{label} payload must be a dictionary")
    return payload


def _canonical_mapping(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        safe_path = _reject_reparse_ancestors(path, label=f"{label} path")
    except LedgerError as exc:
        raise CircuitRollbackError(str(exc)) from exc
    if not safe_path.is_file():
        raise CircuitRollbackError(f"{label} must be a regular file")
    try:
        content = safe_path.read_bytes()
        payload = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CircuitRollbackError(f"{label} is unreadable or malformed") from exc
    if not isinstance(payload, dict) or canonical_json_bytes(payload) != content:
        raise CircuitRollbackError(f"{label} is not canonical")
    return payload, content


def _verify_artifact(
    path_value: object,
    sha_value: object,
    label: str,
    *,
    expected_circuit_id: str | None = None,
) -> Path:
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise CircuitRollbackError(f"{label} path must be absolute")
    path = Path(path_value)
    try:
        safe_path = _reject_reparse_ancestors(path, label=f"{label} path")
    except LedgerError as exc:
        raise CircuitRollbackError(str(exc)) from exc
    if str(safe_path) != path_value:
        raise CircuitRollbackError(f"{label} path is not canonical")
    expected = _require_sha256(sha_value, f"{label} SHA-256")
    if label == "package":
        if not safe_path.is_dir():
            raise CircuitRollbackError("package must be a canonical package directory")
        try:
            package = verify_circuit_package(safe_path, expected)
        except CircuitPackageError as exc:
            raise CircuitRollbackError(f"package verification failed: {exc}") from exc
        if (
            expected_circuit_id is None
            or package.manifest.identity != expected_circuit_id
        ):
            raise CircuitRollbackError(
                "package circuit identity does not match rollback circuit"
            )
        return safe_path
    return _verified_regular_file(safe_path, expected, label)


def _verify_transaction(
    registry: Mapping[str, Any],
    *,
    circuit_id: str,
    snapshot_sha256: str,
) -> None:
    root_value = registry["transaction_root"]
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise CircuitRollbackError("transaction root must be absolute")
    try:
        root = _reject_reparse_ancestors(
            Path(root_value),
            label="transaction root",
        )
    except LedgerError as exc:
        raise CircuitRollbackError(str(exc)) from exc
    if str(root) != root_value or not root.is_dir():
        raise CircuitRollbackError("transaction root is invalid")
    if {entry.name for entry in root.iterdir()} != {"transaction.json", "events"}:
        raise CircuitRollbackError("transaction root artifacts are invalid")
    manifest, manifest_bytes = _canonical_mapping(
        root / "transaction.json", "transaction manifest"
    )
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != registry["transaction_manifest_sha256"]:
        raise CircuitRollbackError("transaction manifest SHA-256 mismatch")
    expected_manifest_fields = {
        "schema_version",
        "transaction_id",
        "operation",
        "circuit_id",
        "package_sha256",
        "recipient_parent_sha256",
        "tap_contract_sha256",
        "preinstall_snapshot_sha256",
        "lifecycle",
    }
    if (
        set(manifest) != expected_manifest_fields
        or manifest.get("schema_version") != 1
        or manifest.get("operation") not in {"inject", "replace"}
        or manifest.get("circuit_id") != circuit_id
        or manifest.get("package_sha256") != registry["package_sha256"]
        or manifest.get("tap_contract_sha256") != registry["tap_contract_sha256"]
        or manifest.get("preinstall_snapshot_sha256") != snapshot_sha256
        or manifest.get("lifecycle") != "candidate"
    ):
        raise CircuitRollbackError("transaction manifest identity mismatch")
    _require_sha256(
        manifest.get("recipient_parent_sha256"),
        "transaction recipient parent SHA-256",
    )
    transaction_id = manifest.get("transaction_id")
    if not isinstance(transaction_id, str) or not transaction_id:
        raise CircuitRollbackError("transaction identity is invalid")

    events_root = root / "events"
    if not events_root.is_dir():
        raise CircuitRollbackError("transaction events root is invalid")
    event_paths = sorted(events_root.iterdir(), key=lambda item: item.name)
    expected_count = registry["event_count"]
    if type(expected_count) is not int or expected_count < 1:
        raise CircuitRollbackError("transaction event_count is invalid")
    if len(event_paths) != expected_count:
        raise CircuitRollbackError("transaction event count mismatch")
    previous_sha256 = _GENESIS_SHA256
    previous_lifecycle: str | None = None
    for sequence, event_path in enumerate(event_paths):
        event, _ = _canonical_mapping(event_path, "transaction event")
        expected_fields = {
            "schema_version",
            "transaction_id",
            "sequence",
            "lifecycle",
            "manifest_sha256",
            "previous_event_sha256",
            "event_sha256",
        }
        if sequence:
            expected_fields.add("from_lifecycle")
        lifecycle = event.get("lifecycle")
        unsigned = {key: value for key, value in event.items() if key != "event_sha256"}
        expected_event_sha256 = canonical_sha256(unsigned)
        if (
            set(event) != expected_fields
            or type(event.get("schema_version")) is not int
            or event.get("schema_version") != 1
            or event.get("transaction_id") != transaction_id
            or type(event.get("sequence")) is not int
            or event.get("sequence") != sequence
            or not isinstance(lifecycle, str)
            or event_path.name != f"{sequence:04d}-{lifecycle}.json"
            or event.get("manifest_sha256") != manifest_sha256
            or event.get("previous_event_sha256") != previous_sha256
            or event.get("event_sha256") != expected_event_sha256
            or (sequence == 0 and lifecycle != "candidate")
            or (sequence > 0 and event.get("from_lifecycle") != previous_lifecycle)
            or (
                sequence > 0
                and lifecycle
                not in _TRANSACTION_TRANSITIONS.get(
                    previous_lifecycle or "", frozenset()
                )
            )
        ):
            raise CircuitRollbackError("transaction event identity mismatch")
        previous_lifecycle = lifecycle
        previous_sha256 = expected_event_sha256
    if previous_sha256 != registry["event_tail_sha256"]:
        raise CircuitRollbackError("transaction event tail SHA-256 mismatch")


def _validate_tensor_state(
    incoming: object,
    expected: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(incoming, Mapping):
        raise CircuitRollbackError(f"{label} must be a tensor mapping")
    if set(incoming) != set(expected):
        missing = sorted(set(expected) - set(incoming))
        extra = sorted(set(incoming) - set(expected))
        raise CircuitRollbackError(
            f"{label} keys mismatch: missing={missing}, extra={extra}"
        )
    validated: dict[str, torch.Tensor] = {}
    for key, target in expected.items():
        value = incoming[key]
        if not isinstance(value, torch.Tensor):
            raise CircuitRollbackError(f"{label} value is not a tensor: {key}")
        if value.shape != target.shape or value.dtype != target.dtype:
            raise CircuitRollbackError(f"{label} tensor contract mismatch: {key}")
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all().item()
        ):
            raise CircuitRollbackError(f"non-finite {label} tensor: {key}")
        validated[key] = value.detach().cpu()
    return validated


def _reject_cross_prefix_aliases(model: nn.Module, prefix: str) -> None:
    registrations = [
        *model.named_parameters(remove_duplicate=False),
        *model.named_buffers(remove_duplicate=False),
    ]
    by_object: dict[int, set[str]] = {}
    by_storage: dict[int, set[str]] = {}
    for name, tensor in registrations:
        by_object.setdefault(id(tensor), set()).add(name)
        try:
            storage = tensor.untyped_storage()
            storage_id = storage._cdata
        except (AttributeError, RuntimeError) as exc:
            raise CircuitRollbackError(
                f"cannot inspect registered tensor storage alias: {name}"
            ) from exc
        if storage.nbytes() > 0:
            by_storage.setdefault(storage_id, set()).add(name)

    for alias_kind, groups in (("object", by_object), ("storage", by_storage)):
        for names in groups.values():
            target = sorted(name for name in names if name.startswith(prefix))
            non_target = sorted(name for name in names if not name.startswith(prefix))
            if target and non_target:
                raise CircuitRollbackError(
                    f"cross-prefix {alias_kind} alias detected: "
                    f"target={target}, non_target={non_target}"
                )


def _verify_loaded_selective_state(
    model: nn.Module,
    expected_state: Mapping[str, torch.Tensor],
    *,
    prefix: str,
) -> None:
    observed = model.state_dict()
    if set(observed) != set(expected_state):
        raise CircuitRollbackError("fresh rollback state keys changed after load")
    for key, expected in expected_state.items():
        if not torch.equal(observed[key].detach().cpu(), expected.detach().cpu()):
            category = "target" if key.startswith(prefix) else "non-target"
            raise CircuitRollbackError(
                f"fresh rollback {category} state mismatch after load: {key}"
            )


def _fresh_model(
    factory: object,
    state: Mapping[str, torch.Tensor],
) -> nn.Module:
    if not callable(factory):
        raise CircuitRollbackError("model_factory must be callable")
    try:
        model = factory()
    except Exception as exc:
        raise CircuitRollbackError(f"fresh model factory failed: {exc}") from exc
    if not isinstance(model, nn.Module):
        raise CircuitRollbackError("model_factory must return a torch module")
    expected = model.state_dict()
    validated = _validate_tensor_state(state, expected, label="checkpoint model state")
    try:
        model.load_state_dict(validated, strict=True)
    except Exception as exc:
        raise CircuitRollbackError(f"fresh model state load failed: {exc}") from exc
    model.eval()
    return model


def _output_tensors(output: object) -> list[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return [output]
    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return [logits]
    if isinstance(output, Mapping):
        tensors: list[torch.Tensor] = []
        for value in output.values():
            tensors.extend(_output_tensors(value))
        return tensors
    if isinstance(output, (tuple, list)):
        tensors = []
        for value in output:
            tensors.extend(_output_tensors(value))
        return tensors
    return []


def _run_probe(model: nn.Module, probe: object) -> None:
    try:
        with torch.no_grad():
            if callable(probe) and not isinstance(probe, torch.Tensor):
                output = probe(model)
            elif isinstance(probe, Mapping):
                output = model(**probe)
            elif isinstance(probe, (tuple, list)):
                output = model(*probe)
            else:
                output = model(probe)
    except Exception as exc:
        raise CircuitRollbackError(f"fresh-model probe failed: {exc}") from exc
    tensors = _output_tensors(output)
    if not tensors:
        raise CircuitRollbackError("fresh-model probe produced no tensor output")
    if any(
        (tensor.is_floating_point() or tensor.is_complex())
        and not bool(torch.isfinite(tensor).all().item())
        for tensor in tensors
    ):
        raise CircuitRollbackError("fresh-model probe produced non-finite output")


def _verify_ledger_binding(
    active: CircuitCheckpoint,
    metadata: Mapping[str, Any],
    registry: Mapping[str, Any],
    circuit_id: str,
) -> None:
    metadata_ledger_path = metadata.get("ledger_path")
    try:
        active_ledger_path = _reject_reparse_ancestors(
            active.ledger_path,
            label="ledger path",
        )
    except LedgerError as exc:
        raise CircuitRollbackError(str(exc)) from exc
    if (
        not isinstance(metadata_ledger_path, str)
        or not Path(metadata_ledger_path).is_absolute()
        or metadata_ledger_path != str(active_ledger_path)
    ):
        raise CircuitRollbackError("checkpoint ledger path mismatch")
    metadata_ledger_sha256 = metadata.get("ledger_sha256")
    if metadata_ledger_sha256 != active.ledger_sha256:
        raise CircuitRollbackError("checkpoint ledger SHA-256 mismatch")
    metadata_ledger_count = metadata.get("ledger_count")
    if (
        type(metadata_ledger_count) is not int
        or metadata_ledger_count != active.ledger_count
        or metadata_ledger_count < 1
    ):
        raise CircuitRollbackError("checkpoint ledger count mismatch")
    _require_sha256(active.ledger_sha256, "ledger SHA-256")
    try:
        ledger = CircuitLedger(
            active_ledger_path,
            expected_tail_sha256=active.ledger_sha256,
            expected_record_count=active.ledger_count,
        )
        if ledger.verify() != active.ledger_sha256:
            raise CircuitRollbackError("ledger tail SHA-256 mismatch")
        records = ledger.verified_records()
    except LedgerError as exc:
        raise CircuitRollbackError(f"ledger verification failed: {exc}") from exc
    matching = [
        record
        for record in records
        if record.event.get("kind") == "circuit_activated"
        and record.event.get("circuit_id") == circuit_id
        and record.event.get("package_sha256") == registry["package_sha256"]
        and record.event.get("tap_contract_sha256")
        == registry["tap_contract_sha256"]
        and record.event.get("preinstall_snapshot_sha256")
        == registry["preinstall_snapshot_sha256"]
        and record.event.get("event_count") == registry["event_count"]
        and record.event.get("event_tail_sha256") == registry["event_tail_sha256"]
        and record.event.get("transaction_manifest_sha256")
        == registry["transaction_manifest_sha256"]
    ]
    if len(matching) != 1:
        raise CircuitRollbackError(
            "ledger does not contain exactly one matching transaction tail anchor"
        )


def _publish_staging_no_replace(staging: Path, destination: Path) -> None:
    """Atomically publish one verified file without replacing a competitor."""

    if os.name == "nt":
        try:
            os.rename(staging, destination)
            return
        except FileExistsError as exc:
            raise CircuitRollbackError(
                "rollback output appeared during publication"
            ) from exc
        except OSError as exc:
            if getattr(exc, "winerror", None) in {80, 183}:
                raise CircuitRollbackError(
                    "rollback output appeared during publication"
                ) from exc
            raise CircuitRollbackError(
                f"cannot publish rollback checkpoint: {exc}"
            ) from exc

    if not sys.platform.startswith("linux"):
        raise CircuitRollbackError(
            "atomic no-replace checkpoint publication is unavailable on this platform"
        )
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise CircuitRollbackError(
            "atomic no-replace checkpoint publication is unavailable on this Linux host"
        ) from exc
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(
        -100,
        os.fsencode(staging),
        -100,
        os.fsencode(destination),
        1,
    ) == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise CircuitRollbackError("rollback output appeared during publication")
    if error_number in {errno.ENOSYS, errno.EINVAL}:
        raise CircuitRollbackError(
            "atomic no-replace checkpoint publication is unavailable on this Linux host"
        )
    raise CircuitRollbackError(
        f"cannot publish rollback checkpoint: {os.strerror(error_number)}"
    )


def _atomic_verified_save(
    payload: dict[str, Any],
    output: Path,
    *,
    model_factory: Callable[[], nn.Module],
    probe: object,
    expected_state: Mapping[str, torch.Tensor],
    prefix: str,
) -> str:
    try:
        safe_output = _reject_reparse_ancestors(
            output,
            label="rollback output path",
        )
    except LedgerError as exc:
        raise CircuitRollbackError(str(exc)) from exc
    if os.path.lexists(safe_output):
        raise CircuitRollbackError("rollback output already exists")
    if not safe_output.parent.is_dir():
        raise CircuitRollbackError("rollback output parent must be an existing real directory")
    temporary = safe_output.with_name(
        f".{safe_output.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        staging_sha256 = _sha256_file(temporary, "rollback staging checkpoint")
        reloaded = _load_torch_mapping(temporary, "rollback staging checkpoint")
        state = reloaded.get("model_state")
        if not isinstance(state, Mapping):
            raise CircuitRollbackError("rollback staging checkpoint model state is missing")
        fresh = _fresh_model(model_factory, state)
        _reject_cross_prefix_aliases(fresh, prefix)
        _verify_loaded_selective_state(
            fresh,
            expected_state,
            prefix=prefix,
        )
        _run_probe(fresh, probe)
        _publish_staging_no_replace(temporary, safe_output)
        try:
            _fsync_directory(safe_output.parent)
        except (LedgerError, OSError) as exc:
            raise CircuitRollbackError(
                f"cannot fsync rollback checkpoint directory: {exc}"
            ) from exc
        published_sha256 = _sha256_file(safe_output, "rollback checkpoint")
        if published_sha256 != staging_sha256:
            raise CircuitRollbackError(
                "published rollback checkpoint bytes differ from verified staging"
            )
    except BaseException as exc:
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:
            pass
        if isinstance(exc, (KeyboardInterrupt, SystemExit, CircuitRollbackError)):
            raise
        raise CircuitRollbackError(f"cannot publish rollback checkpoint: {exc}") from exc
    return published_sha256


def create_rollback_child(
    active_checkpoint: CircuitCheckpoint,
    snapshot: CircuitSnapshot,
    circuit_id: str,
    output: str | Path,
) -> CircuitCheckpoint:
    """Create a verified child that restores only ``circuit_id`` from its snapshot."""

    if not isinstance(circuit_id, str) or _CIRCUIT_ID.fullmatch(circuit_id) is None:
        raise CircuitRollbackError("circuit identity is invalid")
    try:
        active_path = Path(active_checkpoint.path)
        active_sha256 = active_checkpoint.sha256
        snapshot_path = Path(snapshot.path)
        snapshot_sha256 = snapshot.sha256
        model_factory = active_checkpoint.model_factory
        probe = active_checkpoint.probe
        ledger_count = active_checkpoint.ledger_count
    except (AttributeError, TypeError) as exc:
        raise CircuitRollbackError("rollback descriptors are invalid") from exc
    active_path = _verified_regular_file(
        active_path, active_sha256, "active checkpoint"
    )
    snapshot_path = _verified_regular_file(
        snapshot_path, snapshot_sha256, "snapshot"
    )
    destination = Path(output)

    active = _load_verified_torch_mapping(
        active_path,
        active_sha256,
        "active checkpoint",
    )
    if (
        set(active) != {"schema_version", "metadata", "model_state"}
        or active.get("schema_version") != 1
        or not isinstance(active.get("metadata"), Mapping)
        or not isinstance(active.get("model_state"), Mapping)
    ):
        raise CircuitRollbackError("active checkpoint schema mismatch")
    metadata = dict(active["metadata"])
    if metadata.get("schema_version") != 1:
        raise CircuitRollbackError("active checkpoint metadata schema mismatch")
    circuits = metadata.get("circuits")
    if not isinstance(circuits, Mapping) or circuit_id not in circuits:
        raise CircuitRollbackError("circuit is not registered in active checkpoint")
    registry_value = circuits[circuit_id]
    if not isinstance(registry_value, Mapping) or set(registry_value) != _REGISTRY_FIELDS:
        raise CircuitRollbackError("circuit registry fields mismatch")
    registry = dict(registry_value)
    if (
        not isinstance(registry["state_prefix"], str)
        or not registry["state_prefix"]
        or not registry["state_prefix"].endswith(".")
    ):
        raise CircuitRollbackError("registered circuit state prefix is invalid")
    for field in (
        "package_sha256",
        "tap_contract_sha256",
        "preinstall_snapshot_sha256",
        "transaction_manifest_sha256",
        "event_tail_sha256",
    ):
        _require_sha256(registry[field], field)
    if snapshot_sha256 != registry["preinstall_snapshot_sha256"]:
        raise CircuitRollbackError("snapshot SHA-256 mismatch against checkpoint registry")

    snapshot_payload = _load_verified_torch_mapping(
        snapshot_path,
        snapshot_sha256,
        "snapshot",
    )
    if (
        set(snapshot_payload) != _SNAPSHOT_FIELDS
        or snapshot_payload.get("schema_version") != 1
        or snapshot_payload.get("circuit_id") != circuit_id
    ):
        raise CircuitRollbackError("snapshot identity or fields mismatch")
    for field in (
        "state_prefix",
        "package_path",
        "package_sha256",
        "tap_contract_path",
        "tap_contract_sha256",
    ):
        if snapshot_payload.get(field) != registry[field]:
            raise CircuitRollbackError(f"snapshot {field} mismatch")

    _verify_artifact(
        registry["package_path"],
        registry["package_sha256"],
        "package",
        expected_circuit_id=circuit_id,
    )
    _verify_artifact(
        registry["tap_contract_path"], registry["tap_contract_sha256"], "tap"
    )
    _verify_transaction(
        registry,
        circuit_id=circuit_id,
        snapshot_sha256=snapshot_sha256,
    )
    _verify_ledger_binding(active_checkpoint, metadata, registry, circuit_id)

    fresh_active = _fresh_model(model_factory, active["model_state"])
    prefix = registry["state_prefix"]
    _reject_cross_prefix_aliases(fresh_active, prefix)
    expected_full_state = fresh_active.state_dict()
    expected_circuit_state = {
        key: tensor
        for key, tensor in expected_full_state.items()
        if key.startswith(prefix)
    }
    if not expected_circuit_state:
        raise CircuitRollbackError("registered circuit state prefix matched no model state")
    restored_state = _validate_tensor_state(
        snapshot_payload["circuit_state"],
        expected_circuit_state,
        label="rollback snapshot state",
    )
    child_state = {
        key: tensor.detach().cpu().clone()
        for key, tensor in active["model_state"].items()
    }
    before_non_target = {
        key: tensor
        for key, tensor in child_state.items()
        if not key.startswith(prefix)
    }
    child_state.update(restored_state)
    if any(
        not torch.equal(child_state[key], value)
        for key, value in before_non_target.items()
    ):
        raise CircuitRollbackError("rollback mutated state outside registered prefix")

    child_metadata = dict(metadata)
    child_metadata.update(
        {
            "parent_checkpoint_sha256": active_sha256,
            "rolled_back_circuit_id": circuit_id,
            "rollback_snapshot_sha256": snapshot_sha256,
        }
    )
    child_payload = {
        "schema_version": 1,
        "metadata": child_metadata,
        "model_state": child_state,
    }
    child_sha256 = _atomic_verified_save(
        child_payload,
        destination,
        model_factory=model_factory,
        probe=probe,
        expected_state=child_state,
        prefix=prefix,
    )
    return CircuitCheckpoint(
        path=destination,
        sha256=child_sha256,
        ledger_path=Path(active_checkpoint.ledger_path),
        ledger_sha256=active_checkpoint.ledger_sha256,
        ledger_count=ledger_count,
        model_factory=model_factory,
        probe=probe,
    )


__all__ = [
    "CircuitCheckpoint",
    "CircuitRollbackError",
    "CircuitSnapshot",
    "create_rollback_child",
]
