"""Gate-zero circuit installation wrappers and immutable lifecycle records."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
from torch import nn

from f51_darwin.circuits.compatibility import (
    CircuitCompatibilityError,
    _mutation_count,
    _restore_model,
    _state_snapshot,
)
from f51_darwin.circuits.identity import canonical_json_bytes


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CIRCUIT_ID = re.compile(r"^f51-circuit-v1:[0-9a-f]{64}$")
_OPERATIONS = frozenset(("inject", "replace"))
_CIRCUIT_MODES = frozenset(("shadow", "active"))
_GENESIS_EVENT_SHA256 = "0" * 64
_TRANSITIONS = {
    "candidate": frozenset(("shadow",)),
    "shadow": frozenset(("adapter_active",)),
    "adapter_active": frozenset(("organ_unfrozen", "quarantine")),
    "organ_unfrozen": frozenset(("active", "quarantine")),
    "active": frozenset(("frozen", "quarantine")),
    "frozen": frozenset(),
    "quarantine": frozenset(("rollback",)),
    "rollback": frozenset(),
}


class CircuitTransactionError(ValueError):
    """A circuit replacement cannot meet its strict installation contract."""


class GatedCircuitWrapper(nn.Module):
    """Run a candidate circuit without changing output until its gate is opened."""

    def __init__(
        self,
        original: nn.Module,
        candidate: nn.Module,
        mode: str,
        max_residual_ratio: float = 0.25,
        output_selector: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(original, nn.Module) or not isinstance(candidate, nn.Module):
            raise CircuitTransactionError("original and candidate must be nn.Module instances")
        if mode not in _OPERATIONS:
            raise CircuitTransactionError("mode must be inject or replace")
        if type(max_residual_ratio) not in (float, int):
            raise CircuitTransactionError("max_residual_ratio must be a finite positive number")
        ratio = float(max_residual_ratio)
        if not math.isfinite(ratio) or ratio <= 0.0:
            raise CircuitTransactionError("max_residual_ratio must be a finite positive number")
        if output_selector is not None and type(output_selector) is not int:
            raise CircuitTransactionError("output_selector must be an integer or None")
        self.original = original
        self.candidate = candidate
        self.mode = mode
        self.max_residual_ratio = ratio
        self.output_selector = output_selector
        self.external_gate = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def _selected_output(self, output: object) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            if self.output_selector is not None:
                raise CircuitTransactionError("output_selector requires a tuple output")
            return output
        if not isinstance(output, tuple):
            raise CircuitTransactionError("original output must be a tensor or tuple")
        if self.output_selector is None:
            raise CircuitTransactionError("tuple output requires an explicit output_selector")
        if not 0 <= self.output_selector < len(output):
            raise CircuitTransactionError("output_selector is outside the tuple output")
        selected = output[self.output_selector]
        if not isinstance(selected, torch.Tensor):
            raise CircuitTransactionError("selected original output must be a tensor")
        return selected

    def _candidate_tensor(self, output: object, original: torch.Tensor) -> torch.Tensor:
        if isinstance(output, tuple):
            if self.output_selector is None:
                raise CircuitTransactionError("tuple candidate output requires output_selector")
            if not 0 <= self.output_selector < len(output):
                raise CircuitTransactionError("output_selector is outside the candidate tuple")
            output = output[self.output_selector]
        if not isinstance(output, torch.Tensor):
            raise CircuitTransactionError("candidate output must be a tensor")
        if output.shape != original.shape:
            raise CircuitTransactionError(
                "candidate output shape is incompatible with original output"
            )
        if output.dtype != original.dtype or output.device != original.device:
            raise CircuitTransactionError("candidate output dtype or device is incompatible")
        if not output.is_floating_point():
            raise CircuitTransactionError("candidate and original outputs must be floating tensors")
        if not bool(torch.isfinite(output).all().item()):
            raise CircuitTransactionError("candidate output is non-finite")
        if not bool(torch.isfinite(original).all().item()):
            raise CircuitTransactionError("original output is non-finite")
        return output

    def _bounded(self, original: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Clip residual magnitude per output element to the declared finite bound."""
        reference = original.detach().abs().clamp_min(1.0)
        limit = reference * self.max_residual_ratio
        bounded = torch.clamp(residual, min=-limit, max=limit)
        if not bool(torch.isfinite(bounded).all().item()):
            raise CircuitTransactionError("bounded residual is non-finite")
        return bounded

    def _replace_selected(self, output: object, selected: torch.Tensor) -> object:
        if isinstance(output, torch.Tensor):
            return selected
        assert isinstance(output, tuple)
        assert self.output_selector is not None
        values = list(output)
        values[self.output_selector] = selected
        return tuple(values)

    def forward(self, *args: object, circuit_mode: str, **kwargs: object) -> object:
        if circuit_mode not in _CIRCUIT_MODES:
            raise CircuitTransactionError("circuit_mode must be shadow or active")
        original_output = self.original(*args, **kwargs)
        selected_original = self._selected_output(original_output)
        if circuit_mode == "shadow":
            with torch.no_grad():
                self._candidate_tensor(self.candidate(*args, **kwargs), selected_original)
            return original_output
        candidate_output = self._candidate_tensor(
            self.candidate(*args, **kwargs), selected_original
        )
        residual = (
            candidate_output - selected_original
            if self.mode == "replace"
            else candidate_output
        )
        bounded = self._bounded(selected_original, residual)
        gate = torch.tanh(self.external_gate).to(
            device=selected_original.device, dtype=selected_original.dtype
        )
        combined = selected_original + gate * bounded
        if not bool(torch.isfinite(combined).all().item()):
            raise CircuitTransactionError("gated output is non-finite")
        return self._replace_selected(original_output, combined)


@dataclass(frozen=True)
class CircuitVerification:
    mode: str
    probes_run: int
    max_abs_error: float


@dataclass(frozen=True)
class _TransactionIdentity:
    root: Path
    transaction_id: str
    operation: str
    circuit_id: str
    package_sha256: str
    recipient_parent_sha256: str
    tap_contract_sha256: str
    preinstall_snapshot_sha256: str

    def manifest_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "transaction_id": self.transaction_id,
            "operation": self.operation,
            "circuit_id": self.circuit_id,
            "package_sha256": self.package_sha256,
            "recipient_parent_sha256": self.recipient_parent_sha256,
            "tap_contract_sha256": self.tap_contract_sha256,
            "preinstall_snapshot_sha256": self.preinstall_snapshot_sha256,
            "lifecycle": "candidate",
        }


@dataclass(frozen=True)
class _LedgerState:
    lifecycle: str
    sequence: int
    event_sha256: str


@dataclass(frozen=True)
class _TrustedTail:
    count: int
    event_sha256: str


class CircuitTransaction:
    """An exclusive transaction whose identity and lifecycle derive from disk."""

    __slots__ = (
        "_identity",
        "_manifest_bytes",
        "_manifest_sha256",
        "_trusted_tail",
        "_wrapper",
    )

    def __init__(
        self,
        identity: _TransactionIdentity,
        manifest_bytes: bytes,
        trusted_tail: _TrustedTail,
        wrapper: GatedCircuitWrapper,
    ) -> None:
        self._identity = identity
        self._manifest_bytes = manifest_bytes
        self._manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        self._trusted_tail = trusted_tail
        self._wrapper = wrapper

    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(self, name):
            raise AttributeError(f"{name} is immutable")
        object.__setattr__(self, name, value)

    @property
    def root(self) -> Path:
        return self._identity.root

    @property
    def transaction_id(self) -> str:
        return self._identity.transaction_id

    @property
    def operation(self) -> str:
        return self._identity.operation

    @property
    def circuit_id(self) -> str:
        return self._identity.circuit_id

    @property
    def package_sha256(self) -> str:
        return self._identity.package_sha256

    @property
    def recipient_parent_sha256(self) -> str:
        return self._identity.recipient_parent_sha256

    @property
    def tap_contract_sha256(self) -> str:
        return self._identity.tap_contract_sha256

    @property
    def preinstall_snapshot_sha256(self) -> str:
        return self._identity.preinstall_snapshot_sha256

    @property
    def wrapper(self) -> GatedCircuitWrapper:
        return self._wrapper

    @property
    def lifecycle(self) -> str:
        return self._audit_integrity().lifecycle

    @classmethod
    def create(
        cls,
        root: Path | str,
        transaction_id: str,
        operation: str,
        circuit_id: str,
        package_sha256: str,
        recipient_parent_sha256: str,
        tap_contract_sha256: str,
        preinstall_snapshot_sha256: str,
        wrapper: GatedCircuitWrapper | None = None,
        *,
        original: nn.Module | None = None,
        candidate: nn.Module | None = None,
        max_residual_ratio: float = 0.25,
        output_selector: int | None = None,
    ) -> CircuitTransaction:
        root_path = Path(root)
        cls._validate_identity_values(
            transaction_id,
            operation,
            circuit_id,
            package_sha256,
            recipient_parent_sha256,
            tap_contract_sha256,
            preinstall_snapshot_sha256,
        )
        if wrapper is None:
            if original is None or candidate is None:
                raise CircuitTransactionError("wrapper or original and candidate are required")
            wrapper = GatedCircuitWrapper(
                original,
                candidate,
                operation,
                max_residual_ratio=max_residual_ratio,
                output_selector=output_selector,
            )
        if not isinstance(wrapper, GatedCircuitWrapper):
            raise CircuitTransactionError("wrapper must be a GatedCircuitWrapper")
        if wrapper.mode != operation:
            raise CircuitTransactionError("wrapper mode must match transaction operation")
        try:
            os.mkdir(root_path)
        except FileExistsError as exc:
            raise CircuitTransactionError("transaction root already exists") from exc
        except OSError as exc:
            raise CircuitTransactionError(f"cannot create transaction root: {exc}") from exc
        identity = _TransactionIdentity(
            root=root_path,
            transaction_id=transaction_id,
            operation=operation,
            circuit_id=circuit_id,
            package_sha256=package_sha256,
            recipient_parent_sha256=recipient_parent_sha256,
            tap_contract_sha256=tap_contract_sha256,
            preinstall_snapshot_sha256=preinstall_snapshot_sha256,
        )
        manifest_bytes = canonical_json_bytes(identity.manifest_payload())
        transaction = cls(
            identity,
            manifest_bytes,
            _TrustedTail(count=0, event_sha256=_GENESIS_EVENT_SHA256),
            wrapper,
        )
        try:
            transaction._write_exclusive_bytes("transaction.json", manifest_bytes)
            (root_path / "events").mkdir()
            tail = transaction._write_event(
                0,
                "candidate",
                None,
                _GENESIS_EVENT_SHA256,
            )
            object.__setattr__(transaction, "_trusted_tail", tail)
            transaction._audit_integrity()
        except Exception:
            # A partial root remains exclusive evidence of a failed create; it is never reused.
            raise
        return transaction

    @staticmethod
    def _validate_identity_values(
        transaction_id: str,
        operation: str,
        circuit_id: str,
        package_sha256: str,
        recipient_parent_sha256: str,
        tap_contract_sha256: str,
        preinstall_snapshot_sha256: str,
    ) -> None:
        if not isinstance(transaction_id, str) or not transaction_id.strip():
            raise CircuitTransactionError("transaction identity is invalid")
        if operation not in _OPERATIONS:
            raise CircuitTransactionError("operation identity is invalid")
        if not isinstance(circuit_id, str) or _CIRCUIT_ID.fullmatch(circuit_id) is None:
            raise CircuitTransactionError("circuit identity is invalid")
        for field_name, value in (
            ("package", package_sha256),
            ("recipient parent", recipient_parent_sha256),
            ("tap contract", tap_contract_sha256),
            ("preinstall snapshot", preinstall_snapshot_sha256),
        ):
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise CircuitTransactionError(f"{field_name} identity is invalid")

    def _write_exclusive_bytes(self, relative_path: str, content: bytes) -> None:
        target = self.root / relative_path
        try:
            with target.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            raise CircuitTransactionError(
                f"immutable artifact already exists: {relative_path}"
            ) from exc

    def _write_event(
        self,
        sequence: int,
        lifecycle: str,
        previous: str | None,
        previous_event_sha256: str,
    ) -> _TrustedTail:
        payload: dict[str, object] = {
            "schema_version": 1,
            "transaction_id": self.transaction_id,
            "sequence": sequence,
            "lifecycle": lifecycle,
            "manifest_sha256": self._manifest_sha256,
            "previous_event_sha256": previous_event_sha256,
        }
        if previous is not None:
            payload["from_lifecycle"] = previous
        payload["event_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        relative_path = f"events/{sequence:04d}-{lifecycle}.json"
        self._write_exclusive_bytes(
            relative_path,
            canonical_json_bytes(payload),
        )
        event_path = self.root / relative_path
        try:
            observed = event_path.read_bytes()
        except OSError as exc:
            raise CircuitTransactionError(f"event read-back failed: {exc}") from exc
        if observed != canonical_json_bytes(payload):
            raise CircuitTransactionError("event read-back does not match exclusive write")
        self._validate_event_payload(
            self._canonical_mapping(observed, "event artifact"),
            sequence,
            previous,
            previous_event_sha256,
            lifecycle,
        )
        return _TrustedTail(count=sequence + 1, event_sha256=payload["event_sha256"])

    def transition(self, lifecycle: str) -> None:
        current = self._audit_integrity()
        if not isinstance(lifecycle, str) or lifecycle not in _TRANSITIONS.get(
            current.lifecycle, frozenset()
        ):
            raise CircuitTransactionError(
                f"invalid lifecycle transition: {current.lifecycle!r} -> {lifecycle!r}"
            )
        tail = self._write_event(
            current.sequence + 1,
            lifecycle,
            current.lifecycle,
            current.event_sha256,
        )
        object.__setattr__(self, "_trusted_tail", tail)
        self._audit_integrity()

    def _audit_integrity(self) -> _LedgerState:
        """Authenticate immutable identity and the complete append-only event trail."""
        identity = self._identity
        if not isinstance(identity, _TransactionIdentity):
            raise CircuitTransactionError("transaction identity state is invalid")
        self._validate_identity_values(
            identity.transaction_id,
            identity.operation,
            identity.circuit_id,
            identity.package_sha256,
            identity.recipient_parent_sha256,
            identity.tap_contract_sha256,
            identity.preinstall_snapshot_sha256,
        )
        if not isinstance(self._wrapper, GatedCircuitWrapper):
            raise CircuitTransactionError("transaction wrapper state is invalid")
        if self._wrapper.mode != identity.operation:
            raise CircuitTransactionError("wrapper mode does not match transaction identity")
        trusted_tail = self._trusted_tail
        if (
            not isinstance(trusted_tail, _TrustedTail)
            or type(trusted_tail.count) is not int
            or trusted_tail.count < 1
            or not isinstance(trusted_tail.event_sha256, str)
            or _SHA256.fullmatch(trusted_tail.event_sha256) is None
        ):
            raise CircuitTransactionError("trusted tail anchor is invalid")
        if not isinstance(self._manifest_bytes, bytes) or not isinstance(
            self._manifest_sha256, str
        ):
            raise CircuitTransactionError("transaction manifest state is invalid")
        captured_digest = hashlib.sha256(self._manifest_bytes).hexdigest()
        if self._manifest_sha256 != captured_digest:
            raise CircuitTransactionError("captured transaction manifest digest changed")
        root = identity.root
        if not isinstance(root, Path) or not root.is_dir() or root.is_symlink():
            raise CircuitTransactionError("transaction root is invalid")
        root_entries = {path.name for path in root.iterdir()}
        if root_entries != {"transaction.json", "events"}:
            raise CircuitTransactionError("transaction root artifacts are invalid")
        manifest_path = root / "transaction.json"
        events_path = root / "events"
        if (
            not manifest_path.is_file()
            or manifest_path.is_symlink()
            or not events_path.is_dir()
            or events_path.is_symlink()
        ):
            raise CircuitTransactionError("transaction manifest or events path is invalid")
        try:
            manifest_bytes = manifest_path.read_bytes()
        except OSError as exc:
            raise CircuitTransactionError(f"transaction manifest is unreadable: {exc}") from exc
        if manifest_bytes != self._manifest_bytes:
            raise CircuitTransactionError("transaction manifest bytes changed")
        if hashlib.sha256(manifest_bytes).hexdigest() != self._manifest_sha256:
            raise CircuitTransactionError("transaction manifest digest mismatch")
        manifest = self._canonical_mapping(manifest_bytes, "transaction manifest")
        if manifest != identity.manifest_payload():
            raise CircuitTransactionError("transaction manifest does not match identity")

        event_paths = sorted(events_path.iterdir(), key=lambda path: path.name)
        if not event_paths:
            raise CircuitTransactionError("event history is missing candidate")
        lifecycle = ""
        previous_event_sha256 = _GENESIS_EVENT_SHA256
        for sequence, event_path in enumerate(event_paths):
            if not event_path.is_file() or event_path.is_symlink():
                raise CircuitTransactionError("event artifact is invalid")
            try:
                event_bytes = event_path.read_bytes()
            except OSError as exc:
                raise CircuitTransactionError(f"event artifact is unreadable: {exc}") from exc
            payload = self._canonical_mapping(event_bytes, "event artifact")
            event_lifecycle = self._validate_event_payload(
                payload,
                sequence,
                None if sequence == 0 else lifecycle,
                previous_event_sha256,
            )
            if event_path.name != f"{sequence:04d}-{event_lifecycle}.json":
                raise CircuitTransactionError("event artifact identity is invalid")
            lifecycle = event_lifecycle
            previous_event_sha256 = payload["event_sha256"]
        if (
            trusted_tail.count != len(event_paths)
            or trusted_tail.event_sha256 != previous_event_sha256
        ):
            raise CircuitTransactionError("trusted tail anchor does not match event history")
        return _LedgerState(
            lifecycle=lifecycle,
            sequence=len(event_paths) - 1,
            event_sha256=previous_event_sha256,
        )

    def _validate_event_payload(
        self,
        payload: Mapping[str, object],
        sequence: int,
        previous_lifecycle: str | None,
        previous_event_sha256: str,
        expected_lifecycle: str | None = None,
    ) -> str:
        expected_keys = {
            "schema_version",
            "transaction_id",
            "sequence",
            "lifecycle",
            "manifest_sha256",
            "previous_event_sha256",
            "event_sha256",
        }
        if sequence:
            expected_keys.add("from_lifecycle")
        if set(payload) != expected_keys:
            raise CircuitTransactionError("event artifact fields are invalid")
        lifecycle = payload["lifecycle"]
        event_sha256 = payload["event_sha256"]
        unsigned = {key: value for key, value in payload.items() if key != "event_sha256"}
        expected_event_sha256 = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or payload["transaction_id"] != self.transaction_id
            or type(payload["sequence"]) is not int
            or payload["sequence"] != sequence
            or payload["manifest_sha256"] != self._manifest_sha256
            or payload["previous_event_sha256"] != previous_event_sha256
            or not isinstance(lifecycle, str)
            or not isinstance(event_sha256, str)
            or _SHA256.fullmatch(event_sha256) is None
            or event_sha256 != expected_event_sha256
            or (expected_lifecycle is not None and lifecycle != expected_lifecycle)
        ):
            raise CircuitTransactionError("event artifact identity is invalid")
        if sequence == 0:
            if lifecycle != "candidate":
                raise CircuitTransactionError("event history must begin at candidate")
        elif (
            payload["from_lifecycle"] != previous_lifecycle
            or lifecycle not in _TRANSITIONS.get(previous_lifecycle, frozenset())
        ):
            raise CircuitTransactionError("event lifecycle transition is invalid")
        return lifecycle

    @staticmethod
    def _canonical_mapping(content: bytes, artifact_name: str) -> dict[str, object]:
        try:
            payload = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CircuitTransactionError(f"{artifact_name} is malformed") from exc
        if not isinstance(payload, dict) or canonical_json_bytes(payload) != content:
            raise CircuitTransactionError(f"{artifact_name} is not canonical")
        return payload

    def _verify(
        self, probes: Iterable[object], *, mode: str, tolerance: float
    ) -> CircuitVerification:
        self._audit_integrity()
        if type(tolerance) not in (float, int) or not math.isfinite(float(tolerance)):
            raise CircuitTransactionError("tolerance must be finite")
        if float(tolerance) < 0.0:
            raise CircuitTransactionError("tolerance must be non-negative")
        materialized = list(probes)
        if not materialized:
            raise CircuitTransactionError("at least one probe is required")
        baseline = _state_snapshot(self.wrapper.original)
        rng_state = (
            random.getstate(),
            torch.get_rng_state().clone(),
            tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None,
            np.random.get_state(),
        )
        errors: list[float] = []

        def restore_rng() -> None:
            random.setstate(rng_state[0])
            torch.set_rng_state(rng_state[1])
            if rng_state[2] is not None:
                torch.cuda.set_rng_state_all(list(rng_state[2]))
            np.random.set_state(rng_state[3])

        def require_clean_backbone() -> None:
            mutations = _mutation_count(self.wrapper.original, baseline)
            try:
                _restore_model(self.wrapper.original, baseline)
            except CircuitCompatibilityError as exc:
                raise CircuitTransactionError(f"backbone restoration failed: {exc}") from exc
            if mutations:
                raise CircuitTransactionError("backbone mutation detected during probe")

        try:
            for probe in materialized:
                restore_rng()
                original_output = self._call(self.wrapper.original, probe)
                require_clean_backbone()
                restore_rng()
                circuit_output = self._call(self.wrapper, probe, circuit_mode=mode)
                require_clean_backbone()
                original_tensor = self.wrapper._selected_output(original_output)
                circuit_tensor = self.wrapper._selected_output(circuit_output)
                if not bool(torch.isfinite(original_tensor).all().item()) or not bool(
                    torch.isfinite(circuit_tensor).all().item()
                ):
                    raise CircuitTransactionError("probe output is non-finite")
                if original_tensor.shape != circuit_tensor.shape:
                    raise CircuitTransactionError("probe output shape changed")
                errors.append(float((original_tensor - circuit_tensor).abs().max().item()))
        except CircuitCompatibilityError as exc:
            raise CircuitTransactionError(f"backbone state contract failed: {exc}") from exc
        finally:
            try:
                _restore_model(self.wrapper.original, baseline)
            except CircuitCompatibilityError as exc:
                raise CircuitTransactionError(f"backbone restoration failed: {exc}") from exc
            restore_rng()
        max_abs_error = max(errors, default=float("inf"))
        if max_abs_error > float(tolerance):
            raise CircuitTransactionError(
                f"{mode} equivalence failed: {max_abs_error} exceeds {float(tolerance)}"
            )
        return CircuitVerification(
            mode="gate_zero" if mode == "active" else "shadow",
            probes_run=len(materialized),
            max_abs_error=max_abs_error,
        )

    @staticmethod
    def _call(module: nn.Module, probe: object, **kwargs: object) -> object:
        if isinstance(probe, Mapping):
            return module(**probe, **kwargs)
        if isinstance(probe, (tuple, list)):
            return module(*probe, **kwargs)
        return module(probe, **kwargs)

    def verify_shadow(
        self, probes: Iterable[object], tolerance: float = 1e-5
    ) -> CircuitVerification:
        return self._verify(probes, mode="shadow", tolerance=tolerance)

    def verify_gate_zero(
        self, probes: Iterable[object], tolerance: float = 1e-5
    ) -> CircuitVerification:
        self._audit_integrity()
        if self.wrapper.external_gate.item() != 0.0:
            raise CircuitTransactionError("gate-zero verification requires an exact zero gate")
        return self._verify(probes, mode="active", tolerance=tolerance)


__all__ = [
    "CircuitTransaction",
    "CircuitTransactionError",
    "CircuitVerification",
    "GatedCircuitWrapper",
]
