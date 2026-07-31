from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .contracts import SourceFile, SourceIdentity


@dataclass(frozen=True)
class DonorTensorRecord:
    name: str
    shape: tuple[int, ...]
    dtype: str
    numel: int


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024**2) -> str:
    source = Path(path)
    hasher = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_source_file(source: SourceFile) -> SourceFile:
    path = Path(source.path).resolve()
    reasons: list[str] = []
    if not path.is_file():
        reasons.append("missing")
    else:
        if path.stat().st_size != source.size_bytes:
            reasons.append(
                f"size expected={source.size_bytes} actual={path.stat().st_size}"
            )
        actual_hash = sha256_file(path)
        if actual_hash != source.sha256:
            reasons.append(
                f"sha256 expected={source.sha256} actual={actual_hash}"
            )
    if reasons:
        raise ValueError(
            f"source identity mismatch for {path}: {'; '.join(reasons)}"
        )
    return source


def verify_sources(sources: SourceIdentity) -> tuple[SourceFile, ...]:
    return tuple(verify_source_file(source) for source in sources.files())


def inventory_safetensor(path: str | Path) -> tuple[DonorTensorRecord, ...]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "safetensors==0.8.0 is required for donor inventory"
        ) from exc

    records: list[DonorTensorRecord] = []
    with safe_open(str(Path(path).resolve()), framework="pt", device="cpu") as handle:
        for name in sorted(handle.keys()):
            view = handle.get_slice(name)
            shape = tuple(int(size) for size in view.get_shape())
            numel = 1
            for size in shape:
                numel *= size
            records.append(
                DonorTensorRecord(
                    name=name,
                    shape=shape,
                    dtype=str(view.get_dtype()),
                    numel=numel,
                )
            )
    return tuple(records)
