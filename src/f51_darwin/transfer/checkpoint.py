from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .student import DarwinTransferStudent


CHECKPOINT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class TransferCheckpoint:
    schema_version: int
    donor_manifest_sha256: str
    replaced_layers: tuple[int, ...]
    stage: str
    step: int
    generation: int = 1
    stage_step: int = 0


def save_transfer_checkpoint(
    student: DarwinTransferStudent,
    path: Path,
    *,
    donor_manifest_sha256: str,
    stage: str,
    step: int,
    generation: int = 1,
    stage_step: int = 0,
    optimizer_state: dict | None = None,
) -> TransferCheckpoint:
    metadata = TransferCheckpoint(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        donor_manifest_sha256=str(donor_manifest_sha256),
        replaced_layers=student.replaced_layers,
        stage=str(stage),
        step=int(step),
        generation=int(generation),
        stage_step=int(stage_step),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(
        {
            "metadata": asdict(metadata),
            "model": student.state_dict(),
            "optimizer": optimizer_state,
        },
        temporary,
    )
    os.replace(temporary, path)
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(asdict(metadata), indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def load_transfer_checkpoint(
    student: DarwinTransferStudent,
    path: Path,
    *,
    expected_donor_manifest_sha256: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> TransferCheckpoint:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    raw = payload["metadata"]
    raw["replaced_layers"] = tuple(raw["replaced_layers"])
    raw.setdefault("generation", 1)
    raw.setdefault("stage_step", raw.get("step", 0))
    metadata = TransferCheckpoint(**raw)
    if metadata.schema_version not in {1, CHECKPOINT_SCHEMA_VERSION}:
        raise ValueError("unsupported transfer checkpoint schema")
    if metadata.donor_manifest_sha256 != expected_donor_manifest_sha256:
        raise ValueError("donor manifest hash mismatch")
    if metadata.replaced_layers != student.replaced_layers:
        raise ValueError("replaced layer identity mismatch")
    student.load_state_dict(payload["model"], strict=True)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return metadata


def inherit_matching_weights(
    student: DarwinTransferStudent,
    path: Path,
    *,
    expected_donor_manifest_sha256: str,
) -> int:
    """Overlay shape-compatible weights from an earlier progressive generation."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    raw = payload["metadata"]
    if raw["donor_manifest_sha256"] != expected_donor_manifest_sha256:
        raise ValueError("donor manifest hash mismatch")
    current = student.state_dict()
    inherited = {
        name: tensor
        for name, tensor in payload["model"].items()
        if name in current and current[name].shape == tensor.shape
    }
    current.update(inherited)
    student.load_state_dict(current, strict=True)
    return len(inherited)
