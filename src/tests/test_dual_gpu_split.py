from __future__ import annotations

import pytest
import torch

from f51_darwin.darwin_x import DarwinXConfig, DarwinXModel


def _twenty_layer_config() -> DarwinXConfig:
    return DarwinXConfig(
        model_name="split-test",
        vocab_size=512,
        context_length=64,
        inference_context_length=64,
        d_model=128,
        n_layers=20,
        n_heads=8,
        n_kv_heads=2,
        ssm_state=16,
        fine_experts=4,
        shared_experts=1,
        experts_per_token=2,
        fine_expert_hidden_dim=128,
        shared_expert_hidden_dim=128,
        heartbeat_enabled=True,
    )


def test_recommended_split_accounts_for_heterogeneous_vram() -> None:
    with torch.device("meta"):
        model = DarwinXModel(_twenty_layer_config())

    split = model.recommended_dual_gpu_split(
        gpu0_total_bytes=16_311 * 1024**2,
        gpu1_total_bytes=12_288 * 1024**2,
    )

    assert 9 <= split <= 13
    assert split != 7


def test_recommended_split_rejects_invalid_capacity() -> None:
    with torch.device("meta"):
        model = DarwinXModel(_twenty_layer_config())

    with pytest.raises(ValueError, match="capacities"):
        model.recommended_dual_gpu_split(
            gpu0_total_bytes=0,
            gpu1_total_bytes=12_288 * 1024**2,
        )
