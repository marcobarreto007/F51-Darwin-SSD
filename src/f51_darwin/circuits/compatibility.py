"""Fail-closed static and executable compatibility preflight."""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import numpy as np
import torch
from torch import nn

from f51_darwin.circuits.identity import canonical_sha256
from f51_darwin.circuits.package import LoadedCircuitPackage, tensor_sha256
from f51_darwin.circuits.taps import CircuitTapError, TapCapture, resolve_module


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CircuitCompatibilityError(ValueError):
    """Static package/recipient compatibility could not be proven."""


@dataclass(frozen=True)
class RecipientIdentity:
    sha256: str
    family: str
    tap_rank: int
    tap_width: int
    minimum_sequence_length: int
    supports_mask: bool
    supports_cache: bool
    expected_package_sha256: str
    expected_package_content_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.family, str) or not self.family:
            raise ValueError("family must be non-empty")
        for field_name in ("tap_rank", "tap_width", "minimum_sequence_length"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if type(self.supports_mask) is not bool or type(self.supports_cache) is not bool:
            raise ValueError("supports_mask and supports_cache must be booleans")
        if (
            not isinstance(self.expected_package_sha256, str)
            or _SHA256.fullmatch(self.expected_package_sha256) is None
        ):
            raise ValueError(
                "expected_package_sha256 must be a lowercase SHA-256 digest"
            )
        if (
            not isinstance(self.expected_package_content_sha256, str)
            or _SHA256.fullmatch(self.expected_package_content_sha256) is None
        ):
            raise ValueError(
                "expected_package_content_sha256 must be a lowercase SHA-256 digest"
            )


@dataclass(frozen=True)
class PreflightReport:
    compatible: bool
    launch: bool
    max_abs_shadow_error: float
    gate_zero_max_abs_logit_error: float
    state_mutations: int
    outputs_finite: bool
    probes_run: int
    recipient_sha256: str
    package_sha256: str
    constructor_id: str
    candidate_outputs_finite: bool


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for record in _state_snapshot(model):
        for value in (
            record.kind,
            record.name,
            str(record.alias_index),
            str(record.storage_alias_index),
            str(record.persistent),
            record.layout,
            str(record.shape),
            str(record.stride),
            str(record.storage_offset),
            str(record.storage_nbytes),
            record.dtype,
            record.device,
            str(record.requires_grad),
            _tensor_digest(record.content),
        ):
            digest.update(value.encode())
    return digest.hexdigest()


def _coerce_identity(value: RecipientIdentity | Mapping[str, object]) -> RecipientIdentity:
    if isinstance(value, RecipientIdentity):
        return value
    if not isinstance(value, Mapping):
        raise CircuitCompatibilityError("recipient identity must be a mapping")
    try:
        return RecipientIdentity(**dict(value))
    except (TypeError, ValueError) as exc:
        raise CircuitCompatibilityError(f"invalid recipient identity: {exc}") from exc


def _forward(model: nn.Module, probe: object) -> torch.Tensor:
    if isinstance(probe, Mapping):
        output = model(**probe)
    elif isinstance(probe, (tuple, list)):
        output = model(*probe)
    else:
        output = model(probe)
    logits = getattr(output, "logits", output)
    if not isinstance(logits, torch.Tensor):
        raise CircuitCompatibilityError("recipient output must be a tensor or expose .logits")
    return logits


@dataclass(frozen=True)
class _RegisteredObjectSnapshot:
    kind: str
    name: str
    owner: nn.Module
    local_name: str
    original: torch.Tensor
    alias_index: int
    storage: object
    storage_cdata: int
    storage_data_ptr: int
    data_ptr: int
    storage_alias_index: int
    storage_nbytes: int
    storage_offset: int
    stride: tuple[int, ...]
    persistent: bool | None
    layout: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    requires_grad: bool
    content: torch.Tensor


def _registration_slots(
    model: nn.Module,
) -> list[tuple[str, str, nn.Module, str, torch.Tensor, bool | None]]:
    slots = []
    for module_path, module in model.named_modules(remove_duplicate=False):
        prefix = f"{module_path}." if module_path else ""
        slots.extend(
            ("parameter", prefix + name, module, name, value, None)
            for name, value in module._parameters.items()
            if isinstance(value, torch.Tensor)
        )
        slots.extend(
            (
                "buffer",
                prefix + name,
                module,
                name,
                value,
                name not in module._non_persistent_buffers_set,
            )
            for name, value in module._buffers.items()
            if isinstance(value, torch.Tensor)
        )
    slots.sort(key=lambda item: (item[0], item[1]))
    keys = [(kind, name) for kind, name, *_rest in slots]
    if len(keys) != len(set(keys)):
        raise CircuitCompatibilityError("recipient has duplicate registration paths")
    return slots


def _state_snapshot(model: nn.Module) -> tuple[_RegisteredObjectSnapshot, ...]:
    aliases: dict[int, int] = {}
    storage_aliases: dict[int, int] = {}
    records = []
    for kind, name, owner, local_name, value, persistent in _registration_slots(
        model
    ):
        if value.layout != torch.strided:
            raise CircuitCompatibilityError(
                f"recipient registered tensor must use strided layout: {name}"
            )
        storage = value.untyped_storage()
        storage_cdata = int(storage._cdata)
        records.append(
            _RegisteredObjectSnapshot(
                kind=kind,
                name=name,
                owner=owner,
                local_name=local_name,
                original=value,
                alias_index=aliases.setdefault(id(value), len(aliases)),
                storage=storage,
                storage_cdata=storage_cdata,
                storage_data_ptr=int(storage.data_ptr()),
                data_ptr=int(value.data_ptr()),
                storage_alias_index=storage_aliases.setdefault(
                    storage_cdata, len(storage_aliases)
                ),
                storage_nbytes=int(storage.nbytes()),
                storage_offset=int(value.storage_offset()),
                stride=tuple(value.stride()),
                persistent=persistent,
                layout=str(value.layout),
                shape=tuple(value.shape),
                dtype=str(value.dtype),
                device=str(value.device),
                requires_grad=bool(value.requires_grad),
                content=value.detach().clone(),
            )
        )
    return tuple(records)


def _storage_topology(
    value: torch.Tensor,
) -> tuple[int, int, int, int, int, tuple[int, ...], str]:
    if value.layout != torch.strided:
        return (-1, -1, -1, -1, -1, (), str(value.layout))
    storage = value.untyped_storage()
    return (
        int(storage._cdata),
        int(storage.data_ptr()),
        int(value.data_ptr()),
        int(storage.nbytes()),
        int(value.storage_offset()),
        tuple(value.stride()),
        str(value.layout),
    )


def _mutation_count(
    model: nn.Module,
    baseline: tuple[_RegisteredObjectSnapshot, ...],
) -> int:
    current = _registration_slots(model)
    if [(kind, name) for kind, name, *_rest in current] != [
        (record.kind, record.name) for record in baseline
    ]:
        return max(len(current), len(baseline))
    mutations = 0
    aliases: dict[int, int] = {}
    storage_aliases: dict[int, int] = {}
    for record, (kind, name, _owner, _local, value, persistent) in zip(
        baseline, current
    ):
        alias_index = aliases.setdefault(id(value), len(aliases))
        (
            storage_cdata,
            storage_data_ptr,
            data_ptr,
            storage_nbytes,
            storage_offset,
            stride,
            layout,
        ) = _storage_topology(value)
        storage_alias_index = storage_aliases.setdefault(
            storage_cdata, len(storage_aliases)
        )
        if (
            kind != record.kind
            or name != record.name
            or value is not record.original
            or alias_index != record.alias_index
            or storage_cdata != record.storage_cdata
            or storage_data_ptr != record.storage_data_ptr
            or data_ptr != record.data_ptr
            or storage_alias_index != record.storage_alias_index
            or storage_nbytes != record.storage_nbytes
            or storage_offset != record.storage_offset
            or stride != record.stride
            or layout != record.layout
            or persistent != record.persistent
            or tuple(value.shape) != record.shape
            or str(value.dtype) != record.dtype
            or str(value.device) != record.device
            or bool(value.requires_grad) != record.requires_grad
            or not torch.equal(value, record.content)
        ):
            mutations += 1
    return mutations


def _restore_model(
    model: nn.Module,
    state: tuple[_RegisteredObjectSnapshot, ...],
) -> None:
    with torch.no_grad():
        restored_objects: set[int] = set()
        object_records: list[_RegisteredObjectSnapshot] = []
        for record in state:
            owner = record.owner
            owner._parameters.pop(record.local_name, None)
            owner._buffers.pop(record.local_name, None)
            if record.kind == "parameter":
                owner._parameters[record.local_name] = record.original
                owner._non_persistent_buffers_set.discard(record.local_name)
            else:
                owner._buffers[record.local_name] = record.original
                if record.persistent:
                    owner._non_persistent_buffers_set.discard(record.local_name)
                else:
                    owner._non_persistent_buffers_set.add(record.local_name)
            if id(record.original) not in restored_objects:
                object_records.append(record)
                restored_objects.add(id(record.original))
        for record in object_records:
            topology = _storage_topology(record.original)
            expected_topology = (
                record.storage_cdata,
                record.storage_data_ptr,
                record.data_ptr,
                record.storage_nbytes,
                record.storage_offset,
                record.stride,
                record.layout,
            )
            if topology != expected_topology:
                restored_view = torch.empty(
                    0,
                    dtype=record.content.dtype,
                    device=record.content.device,
                )
                restored_view.set_(
                    record.storage,
                    record.storage_offset,
                    record.shape,
                    record.stride,
                )
                record.original.data = restored_view
            record.original.requires_grad_(record.requires_grad)
        for record in object_records:
            if not torch.equal(record.original, record.content):
                record.original.copy_(record.content)
    if _mutation_count(model, state) != 0:
        raise CircuitCompatibilityError(
            "recipient exact registered state could not be restored"
        )


def package_content_sha256(package: LoadedCircuitPackage) -> str:
    """Bind the loaded manifest, runtime contract, and exact current tensor identities."""
    if not isinstance(package, LoadedCircuitPackage):
        raise CircuitCompatibilityError("package must be a LoadedCircuitPackage")
    if not isinstance(package.runtime_state, Mapping):
        raise CircuitCompatibilityError("runtime_state must be a mapping")
    runtime_state = dict(package.runtime_state)
    runtime_state.pop("package_content_sha256", None)
    tensor_identities = []
    for key, tensor in sorted(package.tensors.items()):
        if not isinstance(tensor, torch.Tensor):
            raise CircuitCompatibilityError(f"package tensor is not a tensor: {key}")
        tensor_identities.append(
            {
                "key": key,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": tensor_sha256(tensor),
            }
        )
    try:
        return canonical_sha256(
            {
                "manifest": package.manifest.to_dict(),
                "runtime_state": runtime_state,
                "tensors": tensor_identities,
            }
        )
    except (TypeError, ValueError) as exc:
        raise CircuitCompatibilityError(
            f"package content proof is not canonical: {exc}"
        ) from exc


def _validate_package(
    package: LoadedCircuitPackage,
    expected_package_sha256: str,
    expected_package_content_sha256: str,
) -> None:
    if not isinstance(package, LoadedCircuitPackage):
        raise CircuitCompatibilityError("package must be a verified LoadedCircuitPackage")
    if (
        not isinstance(package.identity.sha256, str)
        or _SHA256.fullmatch(package.identity.sha256) is None
    ):
        raise CircuitCompatibilityError("package SHA-256 identity is invalid")
    if package.identity.sha256 != expected_package_sha256:
        raise CircuitCompatibilityError("package SHA-256 does not match immutable proof")
    identities = {item.key: item for item in package.manifest.tensors}
    if set(identities) != set(package.tensors):
        raise CircuitCompatibilityError("package tensor keys do not match the manifest")
    for key, tensor in package.tensors.items():
        identity = identities[key]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tuple(tensor.shape) != identity.shape
            or str(tensor.dtype) != identity.dtype
            or (
                (tensor.is_floating_point() or tensor.is_complex())
                and not bool(torch.isfinite(tensor).all().item())
            )
            or tensor_sha256(tensor) != identity.sha256
        ):
            raise CircuitCompatibilityError(f"package tensor identity mismatch: {key}")
    if not isinstance(package.runtime_state, Mapping):
        raise CircuitCompatibilityError("runtime_state must be a mapping")
    proof = package.runtime_state.get("package_content_sha256")
    current_content_sha256 = package_content_sha256(package)
    if (
        not isinstance(proof, str)
        or _SHA256.fullmatch(proof) is None
        or proof != current_content_sha256
        or current_content_sha256 != expected_package_content_sha256
    ):
        raise CircuitCompatibilityError("package content proof mismatch")


class _ResidualScaleCandidate(nn.Module):
    def __init__(self, scale: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("scale", scale.detach().clone())

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        scale = self.scale.to(device=activation.device, dtype=activation.dtype)
        return activation * scale


def _build_residual_v1(package: LoadedCircuitPackage) -> nn.Module:
    if set(package.tensors) != {"candidate.scale"}:
        raise CircuitCompatibilityError(
            "residual-v1 constructor requires only candidate.scale"
        )
    scale = package.tensors["candidate.scale"]
    if tuple(scale.shape) != (1,):
        raise CircuitCompatibilityError(
            "residual-v1 candidate.scale must have shape (1,)"
        )
    dependencies = package.runtime_state.get("dependencies", ["activation"])
    if dependencies != ["activation"]:
        raise CircuitCompatibilityError("unsupported constructor dependencies")
    if package.runtime_state.get("requires_mask") or package.runtime_state.get(
        "requires_cache"
    ):
        raise CircuitCompatibilityError("unsupported constructor dependencies")
    return _ResidualScaleCandidate(scale)


_CONSTRUCTOR_REGISTRY: Mapping[
    str,
    Callable[[LoadedCircuitPackage], nn.Module],
] = {
    "residual-v1": _build_residual_v1,
}


def _construct_candidate(package: LoadedCircuitPackage) -> nn.Module:
    constructor = _CONSTRUCTOR_REGISTRY.get(package.manifest.constructor_id)
    if constructor is None:
        raise CircuitCompatibilityError(
            f"unsupported circuit constructor: {package.manifest.constructor_id!r}"
        )
    candidate = constructor(package)
    candidate.eval()
    return candidate


def _max_error(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    if len(left) != len(right):
        return float("inf")
    errors = []
    for actual, expected in zip(left, right):
        if actual.shape != expected.shape:
            return float("inf")
        errors.append(float((actual - expected).abs().max().item()))
    return max(errors, default=float("inf"))


def preflight_circuit(
    package: LoadedCircuitPackage,
    recipient: nn.Module,
    recipient_identity: RecipientIdentity | Mapping[str, object],
    probes: Iterable[object],
) -> PreflightReport:
    identity = _coerce_identity(recipient_identity)
    _validate_package(
        package,
        identity.expected_package_sha256,
        identity.expected_package_content_sha256,
    )
    actual_sha256 = model_state_sha256(recipient)
    if actual_sha256 != identity.sha256:
        raise CircuitCompatibilityError("recipient SHA-256 mismatch")
    manifest = package.manifest
    if identity.family not in manifest.accepted_recipient_families:
        raise CircuitCompatibilityError("recipient family is not declared by the manifest")
    tap = manifest.tap
    if tap.provider != "module_path_v1" or tap.output_selector != "tensor":
        raise CircuitCompatibilityError("unsupported tap provider or output selector")
    try:
        module = resolve_module(recipient, tap.module_path)
    except CircuitTapError as exc:
        raise CircuitCompatibilityError(f"unresolved tap: {exc}") from exc
    if getattr(module, "_f51_circuit_tap_occupied", False):
        raise CircuitCompatibilityError("tap is occupied")
    module_width = getattr(module, "out_features", tap.width)
    if type(module_width) is int and module_width != tap.width:
        raise CircuitCompatibilityError("tap module width mismatch")
    if identity.tap_width != tap.width:
        raise CircuitCompatibilityError("tap width mismatch")
    if identity.tap_rank != tap.rank:
        raise CircuitCompatibilityError("tap rank mismatch")
    if identity.minimum_sequence_length < tap.minimum_sequence_length:
        raise CircuitCompatibilityError("minimum sequence length mismatch")
    runtime_state = package.runtime_state
    if not isinstance(runtime_state, Mapping):
        raise CircuitCompatibilityError("runtime_state must be a mapping")
    if runtime_state.get("requires_mask") and not identity.supports_mask:
        raise CircuitCompatibilityError("unsupported mask requirement")
    if runtime_state.get("requires_cache") and not identity.supports_cache:
        raise CircuitCompatibilityError("unsupported cache requirement")
    candidate = _construct_candidate(package)

    materialized = list(probes)
    if not materialized:
        raise CircuitCompatibilityError("at least one registered probe is required")
    original_state = _state_snapshot(recipient)
    training_modes = {module: module.training for module in recipient.modules()}
    rng_state = (
        random.getstate(),
        torch.get_rng_state().clone(),
        tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None,
        np.random.get_state(),
    )
    state_mutations = 0
    baseline: list[torch.Tensor] = []
    shadow: list[torch.Tensor] = []
    gate_zero: list[torch.Tensor] = []
    outputs_finite = True
    candidate_outputs_finite = True

    def execute(destination: list[torch.Tensor], capture: TapCapture | None = None) -> None:
        nonlocal state_mutations, outputs_finite
        random.setstate(rng_state[0])
        torch.set_rng_state(rng_state[1])
        if rng_state[2] is not None:
            torch.cuda.set_rng_state_all(list(rng_state[2]))
        np.random.set_state(rng_state[3])
        context = capture if capture is not None else _NullContext()
        with context:
            with torch.no_grad():
                for probe in materialized:
                    logits = _forward(recipient, probe)
                    outputs_finite = outputs_finite and bool(torch.isfinite(logits).all().item())
                    destination.append(logits.detach().cpu().clone())
        state_mutations += _mutation_count(recipient, original_state)
        _restore_model(recipient, original_state)

    try:
        recipient.eval()
        execute(baseline)
        execute(shadow, TapCapture(recipient, tap))

        def gate_zero_intervention(activation: torch.Tensor) -> torch.Tensor:
            nonlocal candidate_outputs_finite, outputs_finite
            candidate_output = candidate(activation)
            if not isinstance(candidate_output, torch.Tensor):
                raise CircuitCompatibilityError(
                    "candidate constructor did not return a tensor"
                )
            if candidate_output.shape != activation.shape:
                raise CircuitCompatibilityError("candidate output shape mismatch")
            finite = bool(torch.isfinite(candidate_output).all().item())
            candidate_outputs_finite = candidate_outputs_finite and finite
            outputs_finite = outputs_finite and finite
            return activation + candidate_output * 0.0

        execute(
            gate_zero,
            TapCapture(recipient, tap, intervention=gate_zero_intervention),
        )
    except CircuitTapError as exc:
        raise CircuitCompatibilityError(f"executable tap rejected: {exc}") from exc
    finally:
        _restore_model(recipient, original_state)
        random.setstate(rng_state[0])
        torch.set_rng_state(rng_state[1])
        if rng_state[2] is not None:
            torch.cuda.set_rng_state_all(list(rng_state[2]))
        np.random.set_state(rng_state[3])
        for module, training in training_modes.items():
            module.training = training

    shadow_error = _max_error(shadow, baseline)
    gate_zero_error = _max_error(gate_zero, baseline)
    compatible = (
        shadow_error <= 1e-5
        and gate_zero_error <= 1e-5
        and state_mutations == 0
        and outputs_finite
    )
    return PreflightReport(
        compatible=compatible,
        launch=False,
        max_abs_shadow_error=shadow_error,
        gate_zero_max_abs_logit_error=gate_zero_error,
        state_mutations=state_mutations,
        outputs_finite=outputs_finite,
        probes_run=len(materialized),
        recipient_sha256=actual_sha256,
        package_sha256=package.identity.sha256,
        constructor_id=manifest.constructor_id,
        candidate_outputs_finite=candidate_outputs_finite,
    )


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> None:
        return None


__all__ = [
    "CircuitCompatibilityError",
    "PreflightReport",
    "RecipientIdentity",
    "model_state_sha256",
    "package_content_sha256",
    "preflight_circuit",
]
