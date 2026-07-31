from __future__ import annotations

from pathlib import Path

import torch

from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.transplant_16b.checkpoint import (
    TransplantCheckpointMetadata,
    TransplantCheckpointWriter,
)


def _tiny_config() -> DarwinXConfig:
    return DarwinXConfig(
        model_name="F51-Darwin-X-Tiny-Transplant",
        init="f51_adapted",
        tokenizer="smol_49152_transplant_v1",
        vocab_size=64,
        context_length=32,
        inference_context_length=32,
        d_model=16,
        n_layers=4,
        n_heads=4,
        n_kv_heads=4,
        fine_experts=2,
        shared_experts=1,
        experts_per_token=1,
        fine_expert_hidden_dim=8,
        shared_expert_hidden_dim=8,
        heartbeat_enabled=False,
        spider_sense_enabled=False,
        nitro_enabled=False,
    )


def test_writer_round_trip_is_strict_and_tied(tmp_path: Path) -> None:
    config = _tiny_config()
    model = DarwinXModel(config)
    metadata = TransplantCheckpointMetadata(
        tokenizer_id="smol-tokenizer-contract-v1:" + "a" * 64,
        plan_id="b" * 64,
        source_identity={"smol_weights": "c" * 64, "organ": "d" * 64},
        projection_identity="e" * 64,
        organ_projection_identity="f" * 64,
        coverage_head="1" * 64,
        calibration_digest="2" * 64,
    )

    result = TransplantCheckpointWriter(tmp_path).write(
        model,
        config,
        metadata,
    )

    payload = torch.load(
        result.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    restored = DarwinXModel(config)
    incompatible = restored.load_state_dict(
        payload["model_state_dict"],
        strict=True,
    )
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert restored.lm_head.weight is restored.token_embedding.weight
    assert payload["version"] == 9
    assert payload["transplant"]["coverage_complete"] is True
    assert result.manifest.is_file()
    assert not list(tmp_path.glob("*.incomplete"))


def test_writer_refuses_to_overwrite_published_cycle(tmp_path: Path) -> None:
    config = _tiny_config()
    model = DarwinXModel(config)
    metadata = TransplantCheckpointMetadata.testing()
    writer = TransplantCheckpointWriter(tmp_path)
    writer.write(model, config, metadata)

    try:
        writer.write(model, config, metadata)
    except FileExistsError as exc:
        assert "overwrite" in str(exc)
    else:
        raise AssertionError("published transplant checkpoint was overwritten")
