from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from collections.abc import Mapping, Set
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .recipient import TwoDonorRecipient


@dataclass(frozen=True)
class RecipientCheckpoint:
    path: Path
    sha256: str


@dataclass(frozen=True)
class OrganSnapshot:
    path: Path
    sha256: str
    organ_name: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(
    value: str | None,
    *,
    field: str,
    allow_none: bool = False,
) -> None:
    if value is None and allow_none:
        return
    if (
        value is None
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"invalid {field}")


def recipient_config_identity(recipient: TwoDonorRecipient) -> str:
    config = recipient.config.to_dict()
    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cpu_state_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {
            deepcopy(key): _cpu_state_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_cpu_state_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state_value(item) for item in value)
    return deepcopy(value)


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, Any]:
    return {
        name: _cpu_state_value(value)
        for name, value in module.state_dict().items()
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _validate_state_dict(
    expected: Mapping[str, Any],
    incoming: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    if set(incoming) != set(expected):
        missing = sorted(set(expected) - set(incoming))
        unexpected = sorted(set(incoming) - set(expected))
        raise ValueError(
            f"{label} state keys mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    validated: dict[str, Any] = {}
    for name, target in expected.items():
        value = incoming[name]
        if name.endswith("_extra_state"):
            if not isinstance(target, Mapping) or not isinstance(value, Mapping):
                raise TypeError(f"{label} extra state must be a mapping: {name}")
            validated[name] = _cpu_state_value(value)
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{label} state is not a tensor: {name}")
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f"{label} tensor contract mismatch: {name}")
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise FloatingPointError(f"non-finite {label} tensor: {name}")
        validated[name] = value
    return validated


def save_recipient_checkpoint(
    recipient: TwoDonorRecipient,
    path: str | Path,
    *,
    language_donor_sha256: str,
    organ_donor_sha256: str,
    parent_checkpoint_sha256: str | None,
    ledger_sha256: str,
    organ_states: Mapping[str, str],
    declared_optimizer_parameters: Set[str] = frozenset(),
) -> RecipientCheckpoint:
    _validate_sha256(language_donor_sha256, field="language donor sha256")
    _validate_sha256(organ_donor_sha256, field="organ donor sha256")
    _validate_sha256(
        parent_checkpoint_sha256,
        field="parent checkpoint sha256",
        allow_none=True,
    )
    _validate_sha256(ledger_sha256, field="ledger sha256")
    known_parameters = {name for name, _ in recipient.named_parameters()}
    unknown = set(declared_optimizer_parameters) - known_parameters
    if unknown:
        raise ValueError(f"unknown optimizer parameters: {sorted(unknown)}")
    for organ_name in organ_states:
        recipient.organ_slot(organ_name)

    metadata = {
        "schema_version": 1,
        "language_donor_sha256": language_donor_sha256,
        "organ_donor_sha256": organ_donor_sha256,
        "recipient_config_identity": recipient_config_identity(recipient),
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "ledger_sha256": ledger_sha256,
        "organ_states": dict(sorted(organ_states.items())),
        "declared_optimizer_parameters": sorted(declared_optimizer_parameters),
    }
    destination = Path(path)
    _atomic_torch_save(
        {
            "metadata": metadata,
            "model_state": _cpu_state_dict(recipient),
        },
        destination,
    )
    return RecipientCheckpoint(destination, _sha256_file(destination))


def load_recipient_checkpoint(
    recipient: TwoDonorRecipient,
    path: str | Path,
    *,
    expected_language_donor_sha256: str,
    expected_organ_donor_sha256: str,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("recipient checkpoint payload must be a dictionary")
    metadata = payload.get("metadata")
    state = payload.get("model_state")
    if not isinstance(metadata, dict) or not isinstance(state, dict):
        raise ValueError("recipient checkpoint payload is incomplete")
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported recipient checkpoint schema")
    if metadata.get("language_donor_sha256") != expected_language_donor_sha256:
        raise ValueError("language donor identity mismatch")
    if metadata.get("organ_donor_sha256") != expected_organ_donor_sha256:
        raise ValueError("organ donor identity mismatch")
    if metadata.get("recipient_config_identity") != recipient_config_identity(recipient):
        raise ValueError("recipient config identity mismatch")

    validated = _validate_state_dict(
        recipient.state_dict(),
        state,
        label="recipient",
    )
    recipient.load_state_dict(validated, strict=True)
    return metadata


def save_organ_snapshot(
    recipient: TwoDonorRecipient,
    organ_name: str,
    path: str | Path,
) -> OrganSnapshot:
    slot = recipient.organ_slot(organ_name)
    destination = Path(path)
    _atomic_torch_save(
        {
            "schema_version": 1,
            "organ_name": organ_name,
            "slot_state": _cpu_state_dict(slot),
        },
        destination,
    )
    return OrganSnapshot(
        path=destination,
        sha256=_sha256_file(destination),
        organ_name=organ_name,
    )


def rollback_organ(
    recipient: TwoDonorRecipient,
    snapshot: OrganSnapshot,
) -> None:
    if _sha256_file(snapshot.path) != snapshot.sha256:
        raise ValueError("snapshot hash mismatch")
    payload = torch.load(
        snapshot.path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported organ snapshot schema")
    if payload.get("organ_name") != snapshot.organ_name:
        raise ValueError("organ snapshot identity mismatch")
    state = payload.get("slot_state")
    if not isinstance(state, dict):
        raise ValueError("organ snapshot state is missing")
    slot = recipient.organ_slot(snapshot.organ_name)
    validated = _validate_state_dict(
        slot.state_dict(),
        state,
        label="organ snapshot",
    )
    slot.load_state_dict(validated, strict=True)
