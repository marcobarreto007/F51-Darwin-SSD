from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from f51_darwin.circuits.identity import canonical_sha256


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_non_empty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: object, name: str = "sha256") -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _require_int(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_exact_keys(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    observed = set(payload)
    unknown = observed - expected
    missing = expected - observed
    if unknown:
        raise ValueError(f"{name} has unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{name} has missing fields: {sorted(missing)}")


@dataclass(frozen=True)
class TensorIdentity:
    key: str
    role: str
    shape: tuple[int, ...]
    dtype: str
    sha256: str

    _FIELDS: ClassVar[set[str]] = {"key", "role", "shape", "dtype", "sha256"}

    def __post_init__(self) -> None:
        _require_non_empty_text(self.key, "key")
        _require_non_empty_text(self.role, "role")
        _require_non_empty_text(self.dtype, "dtype")
        _require_sha256(self.sha256)
        if not isinstance(self.shape, tuple) or not self.shape:
            raise ValueError("shape must be a non-empty tuple of non-negative integers")
        for dimension in self.shape:
            _require_int(dimension, "shape dimension", minimum=0)

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "role": self.role,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TensorIdentity:
        if not isinstance(payload, Mapping):
            raise ValueError("tensor identity must be a mapping")
        cls._validate_keys(payload)
        shape = payload["shape"]
        if not isinstance(shape, list):
            raise ValueError("shape must be a list")
        return cls(
            key=payload["key"],
            role=payload["role"],
            shape=tuple(shape),
            dtype=payload["dtype"],
            sha256=payload["sha256"],
        )

    @classmethod
    def _validate_keys(cls, payload: Mapping[str, Any]) -> None:
        _require_exact_keys(payload, cls._FIELDS, "tensor identity")


@dataclass(frozen=True)
class TapContract:
    provider: str
    module_path: str
    position: str
    output_selector: str
    rank: int
    width: int
    minimum_sequence_length: int

    _FIELDS: ClassVar[set[str]] = {
        "provider",
        "module_path",
        "position",
        "output_selector",
        "rank",
        "width",
        "minimum_sequence_length",
    }
    _POSITIONS: ClassVar[frozenset[str]] = frozenset({"pre", "post"})

    def __post_init__(self) -> None:
        _require_non_empty_text(self.provider, "provider")
        _require_non_empty_text(self.module_path, "module_path")
        if self.position not in self._POSITIONS:
            raise ValueError(f"position must be one of {sorted(self._POSITIONS)}")
        _require_non_empty_text(self.output_selector, "output_selector")
        _require_int(self.rank, "rank", minimum=1)
        _require_int(self.width, "width", minimum=1)
        _require_int(self.minimum_sequence_length, "minimum_sequence_length", minimum=1)

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "module_path": self.module_path,
            "position": self.position,
            "output_selector": self.output_selector,
            "rank": self.rank,
            "width": self.width,
            "minimum_sequence_length": self.minimum_sequence_length,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TapContract:
        if not isinstance(payload, Mapping):
            raise ValueError("tap contract must be a mapping")
        _require_exact_keys(payload, cls._FIELDS, "tap contract")
        return cls(**dict(payload))


@dataclass(frozen=True)
class CircuitManifest:
    source_checkpoint_sha256: str
    source_base_checkpoint_id: str
    source_config_identity: str
    source_topology_identity: str
    circuit_kind: str
    constructor_id: str
    tensors: tuple[TensorIdentity, ...]
    tap: TapContract
    accepted_recipient_families: tuple[str, ...]
    source_checkpoint_path_hint: str | None = None
    schema_version: int = 1

    _FIELDS: ClassVar[set[str]] = {
        "schema_version",
        "source_checkpoint_sha256",
        "source_base_checkpoint_id",
        "source_config_identity",
        "source_topology_identity",
        "circuit_kind",
        "constructor_id",
        "tensors",
        "tap",
        "accepted_recipient_families",
        "source_checkpoint_path_hint",
    }

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        _require_sha256(self.source_checkpoint_sha256, "source_checkpoint_sha256")
        _require_non_empty_text(self.source_base_checkpoint_id, "source_base_checkpoint_id")
        _require_non_empty_text(self.source_config_identity, "source_config_identity")
        _require_non_empty_text(self.source_topology_identity, "source_topology_identity")
        _require_non_empty_text(self.circuit_kind, "circuit_kind")
        _require_non_empty_text(self.constructor_id, "constructor_id")
        if self.source_checkpoint_path_hint is not None:
            _require_non_empty_text(self.source_checkpoint_path_hint, "source_checkpoint_path_hint")
        if not isinstance(self.tensors, tuple) or not self.tensors:
            raise ValueError("tensors must be a non-empty tuple")
        if not all(isinstance(item, TensorIdentity) for item in self.tensors):
            raise ValueError("tensors must contain TensorIdentity values")
        tensor_keys = [item.key for item in self.tensors]
        if len(set(tensor_keys)) != len(tensor_keys):
            raise ValueError("tensors must have unique keys")
        if not isinstance(self.tap, TapContract):
            raise ValueError("tap must be a TapContract")
        if not isinstance(self.accepted_recipient_families, tuple) or not self.accepted_recipient_families:
            raise ValueError("accepted_recipient_families must be a non-empty tuple")
        if any(not isinstance(item, str) or not item.strip() for item in self.accepted_recipient_families):
            raise ValueError("accepted_recipient_families must contain non-empty strings")
        if len(set(self.accepted_recipient_families)) != len(self.accepted_recipient_families):
            raise ValueError("accepted_recipient_families must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "source_base_checkpoint_id": self.source_base_checkpoint_id,
            "source_config_identity": self.source_config_identity,
            "source_topology_identity": self.source_topology_identity,
            "circuit_kind": self.circuit_kind,
            "constructor_id": self.constructor_id,
            "tensors": [item.to_dict() for item in self.tensors],
            "tap": self.tap.to_dict(),
            "accepted_recipient_families": list(self.accepted_recipient_families),
            "source_checkpoint_path_hint": self.source_checkpoint_path_hint,
        }

    def identity_core(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "source_base_checkpoint_id": self.source_base_checkpoint_id,
            "source_config_identity": self.source_config_identity,
            "source_topology_identity": self.source_topology_identity,
            "circuit_kind": self.circuit_kind,
            "constructor_id": self.constructor_id,
            "tensors": [item.to_dict() for item in self.tensors],
            "tap": self.tap.to_dict(),
            "accepted_recipient_families": list(self.accepted_recipient_families),
        }

    @property
    def identity(self) -> str:
        return f"f51-circuit-v1:{canonical_sha256(self.identity_core())}"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CircuitManifest:
        if not isinstance(payload, Mapping):
            raise ValueError("circuit manifest must be a mapping")
        _require_exact_keys(payload, cls._FIELDS, "circuit manifest")
        tensors = payload["tensors"]
        families = payload["accepted_recipient_families"]
        if not isinstance(tensors, list):
            raise ValueError("tensors must be a list")
        if not isinstance(families, list):
            raise ValueError("accepted_recipient_families must be a list")
        return cls(
            schema_version=payload["schema_version"],
            source_checkpoint_sha256=payload["source_checkpoint_sha256"],
            source_base_checkpoint_id=payload["source_base_checkpoint_id"],
            source_config_identity=payload["source_config_identity"],
            source_topology_identity=payload["source_topology_identity"],
            circuit_kind=payload["circuit_kind"],
            constructor_id=payload["constructor_id"],
            tensors=tuple(TensorIdentity.from_dict(item) for item in tensors),
            tap=TapContract.from_dict(payload["tap"]),
            accepted_recipient_families=tuple(families),
            source_checkpoint_path_hint=payload["source_checkpoint_path_hint"],
        )
