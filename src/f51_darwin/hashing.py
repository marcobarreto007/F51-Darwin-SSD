"""Canonical hashing utilities for F51 Darwin-X.

Single source of truth for sha256_file, tensor_sha256, canonical_sha256,
and canonical_json. Every other module MUST import from here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024**2) -> str:
    """SHA-256 hash of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    """SHA-256 hash of a tensor's data, dtype, and shape."""
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(
        memoryview(value.reshape(-1).view(torch.uint8).numpy()).cast("B")
    )
    return digest.hexdigest()


def canonical_json_bytes(payload: Mapping[str, Any] | Any) -> bytes:
    """Canonical JSON serialization for hashing."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(payload: Mapping[str, Any] | Any) -> str:
    """SHA-256 of a canonical JSON representation."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def atomic_json_write(payload: dict[str, Any], path: Path) -> None:
    """Atomically write JSON to disk via temp file + os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import os
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
