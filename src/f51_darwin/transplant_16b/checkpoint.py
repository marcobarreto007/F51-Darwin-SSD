from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.organism.checkpoint_root import (
    build_lineage_root_identity,
    write_lineage_root_identity,
)
from f51_darwin.state_identity import backbone_identity

from .sources import sha256_file


@dataclass(frozen=True)
class TransplantCheckpointMetadata:
    tokenizer_id: str
    plan_id: str
    source_identity: Mapping[str, Any]
    projection_identity: str
    organ_projection_identity: str
    coverage_head: str
    calibration_digest: str

    @classmethod
    def testing(cls) -> "TransplantCheckpointMetadata":
        return cls(
            tokenizer_id="smol-tokenizer-contract-v1:" + "a" * 64,
            plan_id="b" * 64,
            source_identity={
                "smol_weights": "c" * 64,
                "organ": "d" * 64,
            },
            projection_identity="e" * 64,
            organ_projection_identity="f" * 64,
            coverage_head="1" * 64,
            calibration_digest="2" * 64,
        )


@dataclass(frozen=True)
class TransplantCheckpointResult:
    checkpoint: Path
    manifest: Path
    checkpoint_sha256: str
    base_checkpoint_id: str


def build_v9_payload(
    model: torch.nn.Module,
    config: DarwinXConfig,
    metadata: TransplantCheckpointMetadata,
) -> dict[str, Any]:
    model_state = {
        key: tensor.detach().cpu()
        for key, tensor in model.state_dict().items()
    }
    base_checkpoint_id = backbone_identity(model_state, config)
    heartbeat_state = (
        model.heartbeat_state_dict()
        if hasattr(model, "heartbeat_state_dict")
        else None
    )
    if heartbeat_state is not None:
        heartbeat_state = {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in heartbeat_state.items()
        }
    return {
        "version": 9,
        "base_checkpoint_id": base_checkpoint_id,
        "tokenizer_id": metadata.tokenizer_id,
        "model_state_dict": model_state,
        "topology_manifest": model.topology_manifest(),
        "config": dataclasses.asdict(config),
        "optimizer_state_dict": {},
        "optimizer_type": None,
        "heartbeat_state": heartbeat_state,
        "training_state": {
            "step": 0,
            "cycle": 0,
            "train_tokens_seen": 0,
            "warmup_steps": 0,
            "warmup_remaining": 0,
            "accum_steps": 1,
        },
        "transplant": {
            "schema": "darwin-smol-full-brain-v1",
            "plan_id": metadata.plan_id,
            "source_identity": dict(metadata.source_identity),
            "projection_identity": metadata.projection_identity,
            "organ_projection_identity": metadata.organ_projection_identity,
            "coverage_head": metadata.coverage_head,
            "coverage_complete": True,
            "calibration_digest": metadata.calibration_digest,
            "status": "engineering_transplant_only",
        },
    }


class TransplantCheckpointWriter:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def write(
        self,
        model: torch.nn.Module,
        config: DarwinXConfig,
        metadata: TransplantCheckpointMetadata,
        *,
        cycle: int = 0,
    ) -> TransplantCheckpointResult:
        self.root.mkdir(parents=True, exist_ok=True)
        checkpoint = self.root / f"organism_cycle_{cycle:03d}.pt"
        incomplete = checkpoint.with_suffix(".pt.incomplete")
        manifest = self.root / f"organism_cycle_{cycle:03d}.manifest.json"
        manifest_incomplete = manifest.with_suffix(".json.incomplete")
        existing = [
            path
            for path in (checkpoint, incomplete, manifest, manifest_incomplete)
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                f"refusing to overwrite transplant artifact: {existing}"
            )

        payload = build_v9_payload(model, config, metadata)
        torch.save(payload, incomplete)
        with incomplete.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        checkpoint_sha256 = sha256_file(incomplete)
        manifest_payload = {
            "schema": "darwin-smol-checkpoint-manifest-v1",
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_bytes": incomplete.stat().st_size,
            "base_checkpoint_id": payload["base_checkpoint_id"],
            "plan_id": metadata.plan_id,
            "coverage_head": metadata.coverage_head,
            "status": "engineering_transplant_only",
        }
        with manifest_incomplete.open(
            "x",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            json.dump(
                manifest_payload,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(incomplete, checkpoint)
        os.replace(manifest_incomplete, manifest)

        source_hash = str(
            metadata.source_identity.get("organ")
            or metadata.source_identity.get("organ_checkpoint")
            or ""
        )
        lineage = build_lineage_root_identity(
            model_config=config,
            tokenizer_id=metadata.tokenizer_id,
            creation_mode="transplant",
            base_checkpoint_id=payload["base_checkpoint_id"],
            source_checkpoint_sha256=source_hash or None,
        )
        lineage["transplant"] = {
            "schema": "darwin-smol-full-brain-v1",
            "plan_id": metadata.plan_id,
            "source_identity": dict(metadata.source_identity),
            "projection_identity": metadata.projection_identity,
            "organ_projection_identity": metadata.organ_projection_identity,
            "coverage_head": metadata.coverage_head,
        }
        write_lineage_root_identity(self.root, lineage)
        return TransplantCheckpointResult(
            checkpoint=checkpoint,
            manifest=manifest,
            checkpoint_sha256=checkpoint_sha256,
            base_checkpoint_id=payload["base_checkpoint_id"],
        )


def verify_shard_manifest(
    checkpoint: str | Path,
    manifest: str | Path,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint).resolve()
    manifest_path = Path(manifest).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("checkpoint") != checkpoint_path.name:
        raise ValueError("checkpoint manifest target mismatch")
    actual_hash = sha256_file(checkpoint_path)
    if payload.get("checkpoint_sha256") != actual_hash:
        raise ValueError("checkpoint manifest hash mismatch")
    if int(payload.get("checkpoint_bytes", -1)) != checkpoint_path.stat().st_size:
        raise ValueError("checkpoint manifest size mismatch")
    return payload
