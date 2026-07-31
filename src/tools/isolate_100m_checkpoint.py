#!/usr/bin/env python3
"""Copy one verified 100M checkpoint into an isolated lineage root."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from f51_darwin.artifact_manifest import write_json_atomic
from f51_darwin.organism.checkpoint_root import (
    assert_new_checkpoint_target,
    build_lineage_root_identity,
    checkpoint_metadata,
    model_config_identity,
    write_lineage_root_identity,
)


@dataclass(frozen=True)
class IsolationResult:
    checkpoint: Path
    pointer: Path
    sha256: str
    size_bytes: int


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def isolate_checkpoint(
    source: str | Path,
    destination_root: str | Path,
    config_path: str | Path,
) -> IsolationResult:
    source_path = Path(source).resolve()
    destination = Path(destination_root).resolve()
    config_raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config_raw, dict):
        raise ValueError("model config must be a YAML mapping")
    metadata = checkpoint_metadata(source_path)
    if metadata["model_name"] != "F51-Darwin-X-100M":
        raise ValueError(f"source is not the 100M lineage: {source_path}")
    if metadata["config_identity"] != model_config_identity(config_raw):
        raise ValueError("source checkpoint config does not match 100M YAML")
    cycle = int(metadata["training_state"].get("cycle", -1))
    step = int(metadata["training_state"].get("step", -1))
    if cycle < 0 or step < 0:
        raise ValueError("source checkpoint has invalid training state")
    destination.mkdir(parents=True, exist_ok=True)
    final_path = assert_new_checkpoint_target(destination, cycle)
    copying_path = final_path.with_suffix(final_path.suffix + ".copying")
    if copying_path.exists():
        raise FileExistsError(f"refusing to overwrite incomplete copy: {copying_path}")
    source_hash = sha256_file(source_path)
    try:
        with source_path.open("rb") as source_stream, copying_path.open("xb") as target:
            shutil.copyfileobj(source_stream, target, length=8 * 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        if copying_path.stat().st_size != source_path.stat().st_size:
            raise RuntimeError("isolated checkpoint size mismatch")
        if sha256_file(copying_path) != source_hash:
            raise RuntimeError("isolated checkpoint SHA-256 mismatch")
        if final_path.exists():
            raise FileExistsError(f"refusing to overwrite existing checkpoint: {final_path}")
        os.rename(copying_path, final_path)
    except BaseException:
        copying_path.unlink(missing_ok=True)
        raise
    identity = build_lineage_root_identity(
        model_config=config_raw,
        tokenizer_id=metadata["tokenizer_id"],
        creation_mode="resume",
        base_checkpoint_id=metadata["base_checkpoint_id"],
        source_checkpoint_sha256=source_hash,
    )
    write_lineage_root_identity(destination, identity)
    pointer = destination / "organism_latest.json"
    write_json_atomic(
        pointer,
        {
            "version": 1,
            "checkpoint_version": metadata["checkpoint_version"],
            "path": final_path.name,
            "cycle": cycle,
            "step": step,
            "size_bytes": final_path.stat().st_size,
            "base_checkpoint_id": metadata["base_checkpoint_id"],
            "source_checkpoint_sha256": source_hash,
        },
    )
    return IsolationResult(
        checkpoint=final_path,
        pointer=pointer,
        sha256=source_hash,
        size_bytes=final_path.stat().st_size,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = isolate_checkpoint(args.source, args.destination_root, args.config)
    print(
        f"isolated={result.checkpoint} bytes={result.size_bytes} "
        f"sha256={result.sha256} pointer={result.pointer}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
