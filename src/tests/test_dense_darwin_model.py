from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from f51_darwin.darwin_x import (
    DarwinXConfig,
    DarwinXModel,
    DenseSwiGLU,
)


def _config(**changes) -> DarwinXConfig:
    base = DarwinXConfig(
        model_name="dense-model-test",
        vocab_size=32,
        context_length=8,
        inference_context_length=8,
        d_model=16,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        feed_forward_kind="dense_swiglu",
        fine_experts=1,
        shared_experts=0,
        experts_per_token=1,
        fine_expert_hidden_dim=32,
        shared_expert_hidden_dim=32,
        mtp_depth=0,
        mtp_weight=0.0,
        jepa_weight=0.0,
        ghost_weight=0.0,
        ghost_enabled=False,
        curiosity_weight=0.0,
        spider_sense_enabled=False,
        heartbeat_enabled=False,
        nitro_enabled=False,
        dae_enabled=False,
        vertical_routing_scale=0.0,
        attention_indices_override=(0, 1),
    )
    return replace(base, **changes)


def test_dense_model_forward_and_topology_have_no_moe() -> None:
    model = DarwinXModel(_config()).eval()
    output = model(torch.tensor([[1, 2, 3]]), heartbeat=False)
    assert torch.isfinite(output.logits).all()
    assert output.moe_stats == []
    assert all(isinstance(block.ffn, DenseSwiGLU) for block in model.blocks)
    assert all(block.moe is None for block in model.blocks)
    assert not any(".moe." in key for key in model.state_dict())

    manifest = model.topology_manifest()
    assert manifest["base_config"]["feed_forward_kind"] == "dense_swiglu"
    assert manifest["neuroendocrine_state"] == []
    assert [row["feed_forward_kind"] for row in manifest["topology"]] == [
        "dense_swiglu",
        "dense_swiglu",
    ]
    assert model.apply_pending_autonomic_actions() == []
    assert model.apply_active_gradient_actions() == []
    assert model.execute_structural_actions() == []


def test_dense_topology_round_trip_is_shape_strict() -> None:
    model = DarwinXModel(_config())
    manifest = model.topology_manifest()
    restored = DarwinXModel(_config())
    assert DarwinXModel.restore_topology(manifest, restored) == 0

    wrong = dict(manifest)
    wrong["base_config"] = {
        **manifest["base_config"],
        "feed_forward_kind": "moe",
    }
    with pytest.raises(ValueError, match="feed-forward"):
        DarwinXModel.restore_topology(wrong, restored)


def test_dense_recommended_split_uses_native_ffn_parameters() -> None:
    with torch.device("meta"):
        model = DarwinXModel(
            replace(
                _config(),
                n_layers=24,
                attention_indices_override=tuple(range(24)),
            )
        )
    split = model.recommended_dual_gpu_split(
        gpu0_total_bytes=16_050 * 1024**2,
        gpu1_total_bytes=12_111 * 1024**2,
    )
    assert 10 <= split <= 16
