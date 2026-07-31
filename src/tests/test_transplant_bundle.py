from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from f51_darwin.transplant.bundle import (
    extract_first_slice_bundle,
    load_organ_bundle,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synthetic_checkpoint(path: Path) -> None:
    torch.manual_seed(51)
    torch.save(
        {
            "version": 9,
            "base_checkpoint_id": "core:test",
            "config": {
                "d_model": 512,
                "jepa_weight": 0.05,
                "gaba_enabled": True,
            },
            "model_state_dict": {
                "token_embedding.weight": torch.randn(11, 512),
                "jepa_predictor.predictor.0.weight": torch.randn(256, 512),
                "jepa_predictor.predictor.0.bias": torch.randn(256),
                "blocks.0.gaba.inhibitory_weight": torch.randn(512, 512),
                "blocks.0.gaba.inhibitory_bias": torch.randn(512),
                "blocks.1.gaba.inhibitory_weight": torch.randn(512, 512),
                "spider_confidence_head.weight": torch.randn(1, 512),
                "inter_hemispheric.output_proj.weight": torch.randn(512, 1024),
            },
            "heartbeat_state": {
                "version": 3,
                "beat": 3,
                "ff_stack_state_dict": {"layers.0.weight": torch.randn(512)},
                "tt_memory_state_dict": {"proj_key.weight": torch.randn(128, 512)},
                "thinker_state_dict": {"thought_start": torch.randn(1, 1, 512)},
                "tt_memory_slots": [
                    {
                        "key": torch.randn(128),
                        "value": torch.randn(512),
                        "timestamp": "2026-07-28T00:00:00+00:00",
                        "domain": "test",
                        "surprise_score": 0.5,
                        "access_count": 2,
                    }
                ],
            },
            "organ_identity_report": {"jepa": "organ:jepa:v1:test"},
        },
        path,
    )


def test_extracts_only_requested_original_organs(tmp_path: Path) -> None:
    checkpoint = tmp_path / "donor.pt"
    bundle_path = tmp_path / "organs.pt"
    _synthetic_checkpoint(checkpoint)

    manifest = extract_first_slice_bundle(
        checkpoint,
        bundle_path,
        source_checkpoint_sha256=_sha256(checkpoint),
        gaba_layers=(0,),
    )

    assert set(manifest.organs) == {"jepa", "gaba.0"}
    bundle = load_organ_bundle(bundle_path)
    assert "jepa_predictor.predictor.0.weight" in bundle.tensors
    assert "blocks.0.gaba.inhibitory_weight" in bundle.tensors
    assert "token_embedding.weight" not in bundle.tensors
    assert "blocks.1.gaba.inhibitory_weight" not in bundle.tensors
    assert bundle.runtime_state["organ_identity_report"]["jepa"].startswith(
        "organ:jepa:"
    )


def test_extracts_explicit_extra_organ_prefix_without_backbone(tmp_path: Path) -> None:
    checkpoint = tmp_path / "donor.pt"
    bundle_path = tmp_path / "organs.pt"
    _synthetic_checkpoint(checkpoint)

    extract_first_slice_bundle(
        checkpoint,
        bundle_path,
        source_checkpoint_sha256=_sha256(checkpoint),
        gaba_layers=(0,),
        extra_organs=("spider_confidence_head.",),
    )

    tensors = load_organ_bundle(bundle_path).tensors
    assert "spider_confidence_head.weight" in tensors
    assert "token_embedding.weight" not in tensors


def test_extracts_verified_ihs_and_heartbeat_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "donor.pt"
    bundle_path = tmp_path / "organs.pt"
    _synthetic_checkpoint(checkpoint)

    manifest = extract_first_slice_bundle(
        checkpoint,
        bundle_path,
        source_checkpoint_sha256=_sha256(checkpoint),
        gaba_layers=(0,),
        extra_organs=("inter_hemispheric.",),
        include_heartbeat=True,
    )
    bundle = load_organ_bundle(bundle_path)

    assert set(manifest.organs) == {
        "jepa",
        "gaba.0",
        "ihs",
        "ttm",
        "heartbeat",
    }
    assert "inter_hemispheric.output_proj.weight" in bundle.tensors
    assert "heartbeat.ff_stack.layers.0.weight" in bundle.tensors
    assert "heartbeat.tt_memory.proj_key.weight" in bundle.tensors
    assert "heartbeat.thinker.thought_start" in bundle.tensors
    assert "heartbeat.tt_memory_slots.0.key" in bundle.tensors
    assert "heartbeat.tt_memory_slots.0.value" in bundle.tensors
    assert bundle.runtime_state["heartbeat_state"]["beat"] == 3
    assert "ff_stack_state_dict" not in bundle.runtime_state["heartbeat_state"]
    assert "key" not in bundle.runtime_state["heartbeat_state"]["tt_memory_slots"][0]
    assert bundle.verify_tensor_hashes() == []


def test_bundle_round_trip_preserves_each_tensor_hash(tmp_path: Path) -> None:
    checkpoint = tmp_path / "donor.pt"
    bundle_path = tmp_path / "organs.pt"
    _synthetic_checkpoint(checkpoint)

    manifest = extract_first_slice_bundle(
        checkpoint,
        bundle_path,
        source_checkpoint_sha256=_sha256(checkpoint),
        gaba_layers=(0,),
    )
    restored = load_organ_bundle(bundle_path)

    assert restored.manifest == manifest
    assert restored.verify_tensor_hashes() == []


def test_extraction_rejects_wrong_source_hash(tmp_path: Path) -> None:
    checkpoint = tmp_path / "donor.pt"
    _synthetic_checkpoint(checkpoint)

    with pytest.raises(ValueError, match="source checkpoint hash mismatch"):
        extract_first_slice_bundle(
            checkpoint,
            tmp_path / "organs.pt",
            source_checkpoint_sha256="0" * 64,
            gaba_layers=(0,),
        )
