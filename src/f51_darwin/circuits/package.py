"""Verified, deterministic on-disk packages for portable Darwin-X circuits."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import ctypes
import errno
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file

from f51_darwin.circuits.identity import canonical_json_bytes, canonical_sha256
from f51_darwin.circuits.manifest import CircuitManifest, TensorIdentity


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PACKAGE_FILES = frozenset(
    {
        "manifest.json",
        "weights.safetensors",
        "runtime_state.json",
        "checksums.json",
        "package.sha256",
    }
)
_CHECKSUMMED_FILES = ("manifest.json", "weights.safetensors", "runtime_state.json")


class CircuitPackageError(ValueError):
    """The circuit package is malformed, unsafe, or does not match its identity."""


@dataclass(frozen=True)
class PackageIdentity:
    sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.sha256, "package sha256")


@dataclass(frozen=True)
class LoadedCircuitPackage:
    manifest: CircuitManifest
    tensors: dict[str, torch.Tensor]
    runtime_state: Any
    identity: PackageIdentity


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CircuitPackageError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash the exact CPU tensor content, independent of autograd metadata."""
    if not isinstance(tensor, torch.Tensor):
        raise CircuitPackageError("tensor must be a torch.Tensor")
    if tensor.device.type != "cpu":
        raise CircuitPackageError("tensor must be on CPU")
    if tensor.layout != torch.strided:
        raise CircuitPackageError("tensor must use strided layout")
    raw = tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _is_finite(tensor: torch.Tensor) -> bool:
    return not (tensor.is_floating_point() or tensor.is_complex()) or bool(
        torch.isfinite(tensor).all().item()
    )


def _validate_tensor(identity: TensorIdentity, tensor: object) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise CircuitPackageError(f"tensor {identity.key!r} must be a torch.Tensor")
    if tensor.device.type != "cpu":
        raise CircuitPackageError(f"tensor {identity.key!r} must be on CPU")
    if tensor.layout != torch.strided:
        raise CircuitPackageError(f"tensor {identity.key!r} must use strided layout")
    if not tensor.is_contiguous():
        raise CircuitPackageError(f"tensor {identity.key!r} must be contiguous")
    if tuple(tensor.shape) != identity.shape:
        raise CircuitPackageError(f"tensor shape mismatch for {identity.key!r}")
    if str(tensor.dtype) != identity.dtype:
        raise CircuitPackageError(f"tensor dtype mismatch for {identity.key!r}")
    if not _is_finite(tensor):
        raise CircuitPackageError(f"non-finite tensor {identity.key!r}")
    if tensor_sha256(tensor) != identity.sha256:
        raise CircuitPackageError(f"tensor content hash mismatch for {identity.key!r}")
    return tensor


def _validate_tensors(
    manifest: CircuitManifest, tensors: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    if not isinstance(tensors, Mapping):
        raise CircuitPackageError("tensors must be a mapping")
    expected = {identity.key for identity in manifest.tensors}
    actual = set(tensors)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise CircuitPackageError(f"tensor keys do not match manifest (missing={missing}, extra={extra})")
    return {identity.key: _validate_tensor(identity, tensors[identity.key]) for identity in manifest.tensors}


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(path, canonical_json_bytes(value))


def _strict_json(path: Path) -> Any:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")

        def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key {key!r}")
                result[key] = value
            return result

        value = json.loads(
            text,
            object_pairs_hook=no_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(f"invalid constant {item}")),
        )
        if canonical_json_bytes(value) != raw:
            raise ValueError("JSON is not canonical")
        return value
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise CircuitPackageError(f"malformed canonical JSON in {path.name}: {exc}") from exc


def _validate_checksums(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(_CHECKSUMMED_FILES):
        raise CircuitPackageError("checksums.json must contain exactly the package payload checksums")
    checksums: dict[str, str] = {}
    for name in _CHECKSUMMED_FILES:
        checksums[name] = _require_sha256(value[name], f"checksum for {name}")
    return checksums


def _verify_file_set(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise CircuitPackageError("package root must be a real directory")
    observed = {entry.name for entry in root.iterdir()}
    if observed != _PACKAGE_FILES:
        missing = sorted(_PACKAGE_FILES - observed)
        extra = sorted(observed - _PACKAGE_FILES)
        if extra:
            raise CircuitPackageError(f"unexpected files in package root: {extra}")
        raise CircuitPackageError(f"missing required package files: {missing}")
    for name in _PACKAGE_FILES:
        entry = root / name
        if entry.is_symlink() or not entry.is_file():
            raise CircuitPackageError(f"package member {name!r} must be a regular file")


def _read_manifest(root: Path) -> CircuitManifest:
    payload = _strict_json(root / "manifest.json")
    if not isinstance(payload, dict):
        raise CircuitPackageError("manifest.json must contain an object")
    try:
        return CircuitManifest.from_dict(payload)
    except ValueError as exc:
        raise CircuitPackageError(f"invalid circuit manifest: {exc}") from exc


def _read_package_sha256(root: Path) -> str:
    try:
        value = (root / "package.sha256").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CircuitPackageError(f"cannot read package.sha256: {exc}") from exc
    return _require_sha256(value, "package.sha256")


def _publish_staging_no_replace(staging: Path, destination: Path) -> None:
    """Atomically publish a staging directory without ever replacing a destination."""
    if os.name == "nt":
        try:
            # Windows MoveFile/rename fails when the destination already exists.
            os.rename(staging, destination)
            return
        except FileExistsError as exc:
            raise CircuitPackageError("destination appeared during package publication") from exc
        except OSError as exc:
            raise CircuitPackageError(f"cannot publish circuit package: {exc}") from exc

    if not sys.platform.startswith("linux"):
        raise CircuitPackageError("atomic no-replace package publication is unavailable on this platform")

    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise CircuitPackageError("atomic no-replace package publication is unavailable on this Linux host") from exc
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(staging), -100, os.fsencode(destination), 1) == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise CircuitPackageError("destination appeared during package publication")
    if error_number in {errno.ENOSYS, errno.EINVAL}:
        raise CircuitPackageError("atomic no-replace package publication is unavailable on this Linux host")
    raise CircuitPackageError(
        f"cannot publish circuit package: {os.strerror(error_number)}"
    )


def verify_circuit_package(root: str | Path, expected_sha256: str) -> LoadedCircuitPackage:
    """Strictly load a package only after every file and tensor is authenticated."""
    expected_sha256 = _require_sha256(expected_sha256, "expected_sha256")
    destination = Path(root)
    _verify_file_set(destination)

    checksum_payload = _validate_checksums(_strict_json(destination / "checksums.json"))
    for name, expected_checksum in checksum_payload.items():
        actual_checksum = sha256_file(destination / name)
        if actual_checksum != expected_checksum:
            raise CircuitPackageError(f"checksum mismatch for {name}")

    package_sha256 = _read_package_sha256(destination)
    computed_sha256 = canonical_sha256(checksum_payload)
    if package_sha256 != computed_sha256:
        raise CircuitPackageError("package identity mismatch in package.sha256")
    if expected_sha256 != computed_sha256:
        raise CircuitPackageError("package identity mismatch against expected_sha256")

    manifest = _read_manifest(destination)
    runtime_state = _strict_json(destination / "runtime_state.json")
    try:
        tensors = load_file(str(destination / "weights.safetensors"), device="cpu")
    except Exception as exc:
        raise CircuitPackageError(f"cannot load weights.safetensors: {exc}") from exc
    validated_tensors = _validate_tensors(manifest, tensors)
    return LoadedCircuitPackage(
        manifest=manifest,
        tensors=validated_tensors,
        runtime_state=runtime_state,
        identity=PackageIdentity(computed_sha256),
    )


def write_circuit_package(
    root: str | Path,
    manifest: CircuitManifest,
    tensors: Mapping[str, torch.Tensor],
    runtime_state: Any,
) -> PackageIdentity:
    """Atomically write a complete package to an absent destination directory."""
    if not isinstance(manifest, CircuitManifest):
        raise CircuitPackageError("manifest must be a CircuitManifest")
    destination = Path(root)
    if destination.exists() or destination.is_symlink():
        raise CircuitPackageError("destination already exists; circuit package destinations must be absent")
    if not destination.name or not destination.parent.is_dir():
        raise CircuitPackageError("destination parent must be an existing directory")
    validated_tensors = _validate_tensors(manifest, tensors)
    try:
        runtime_bytes = canonical_json_bytes(runtime_state)
    except (TypeError, ValueError) as exc:
        raise CircuitPackageError(f"runtime_state is not canonical JSON serializable: {exc}") from exc

    staging = destination.parent / f".{destination.name}.staging-{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise CircuitPackageError(f"staging destination already exists: {staging.name}")
    staging.mkdir()
    try:
        _write_json(staging / "manifest.json", manifest.to_dict())
        save_file(validated_tensors, str(staging / "weights.safetensors"))
        with (staging / "weights.safetensors").open("r+b") as handle:
            os.fsync(handle.fileno())
        _write_bytes(staging / "runtime_state.json", runtime_bytes)
        checksums = {name: sha256_file(staging / name) for name in _CHECKSUMMED_FILES}
        _write_json(staging / "checksums.json", checksums)
        identity = PackageIdentity(canonical_sha256(checksums))
        _write_bytes(staging / "package.sha256", identity.sha256.encode("utf-8"))
        verify_circuit_package(staging, identity.sha256)
        _publish_staging_no_replace(staging, destination)
        return identity
    except Exception:
        if staging.exists() and staging.is_dir():
            shutil.rmtree(staging)
        raise
