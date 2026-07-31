from pathlib import Path

import torch
import yaml

from f51_darwin.darwin_x import (
    DarwinXConfig,
    DarwinXModel,
    estimate_darwin_x_parameters,
)


ROOT = Path(__file__).resolve().parents[2]


def test_darwin_x_default_contract_is_canonical_16b_nitro() -> None:
    config = DarwinXConfig()
    estimate = estimate_darwin_x_parameters(config)
    assert config.model_name == "F51-Darwin-X-1.6B-Nitro"
    assert 1_650_000_000 <= estimate["total"] <= 1_850_000_000
    assert 600_000_000 <= estimate["active_per_token"] <= 800_000_000
    assert config.attention_layer_indices == (3, 7, 11, 15)


def test_darwin_x_600m_config_contract() -> None:
    raw = yaml.safe_load((ROOT / "src" / "configs" / "darwin_x_600m.yaml").read_text(encoding="utf-8"))
    config = DarwinXConfig.from_mapping(raw)
    estimate = estimate_darwin_x_parameters(config)

    assert config.model_name == "F51-Darwin-X-600M"
    assert config.heartbeat_enabled is False  # desativado durante treino — economiza VRAM
    assert 430_000_000 <= estimate["total"] <= 520_000_000
    assert 200_000_000 <= estimate["active_per_token"] <= 350_000_000


def test_darwin_x_small_forward_with_mtp_and_jepa() -> None:
    torch.manual_seed(51)
    config = DarwinXConfig(
        vocab_size=128,
        context_length=16,
        inference_context_length=64,
        d_model=32,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        fine_experts=4,
        shared_experts=1,
        experts_per_token=2,
        fine_expert_hidden_dim=16,
        shared_expert_hidden_dim=16,
        mtp_depth=2,
        heartbeat_enabled=False,
    )
    model = DarwinXModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 12))
    out = model(input_ids, labels=input_ids)
    assert out.logits.shape == (2, 12, config.vocab_size)
    assert out.loss is not None
    assert out.mtp_loss is not None
    assert out.jepa_loss is not None
    assert torch.isfinite(out.loss)
    assert len(out.moe_stats) == config.n_layers
    assert out.heartbeat_stats is None


def test_darwin_x_heartbeat_connects_when_enabled() -> None:
    torch.manual_seed(51)
    config = DarwinXConfig(
        vocab_size=128,
        context_length=16,
        inference_context_length=64,
        d_model=32,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        fine_experts=4,
        shared_experts=1,
        experts_per_token=2,
        fine_expert_hidden_dim=16,
        shared_expert_hidden_dim=16,
        mtp_depth=2,
        heartbeat_enabled=True,
        heartbeat_memory_capacity=8,
    )
    model = DarwinXModel(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 12))

    out = model(input_ids, labels=input_ids, domain="math")
    heartbeat_state = model.heartbeat_state_dict()

    assert out.heartbeat_stats is not None
    assert out.heartbeat_stats["beat"] == 1
    assert heartbeat_state is not None
    assert heartbeat_state["beat"] == 1
    assert "math" in heartbeat_state["domains"]


def test_darwin_x_gqa_reduces_kv_heads() -> None:
    config = DarwinXConfig(
        vocab_size=64,
        context_length=8,
        inference_context_length=32,
        d_model=32,
        n_layers=4,
        n_heads=4,
        n_kv_heads=1,
        fine_experts=4,
        shared_experts=1,
        experts_per_token=2,
        fine_expert_hidden_dim=8,
        shared_expert_hidden_dim=8,
    )
    model = DarwinXModel(config)
    attn_block = model.blocks[3]
    assert attn_block.attention.n_heads == 4
    assert attn_block.attention.n_kv_heads == 1
    assert attn_block.attention.kv_repeat == 4


def test_darwin_x_from_mapping_coerces_string_values() -> None:
    """Bug #3 — checkpoints serialise everything as strings.

    ``from_mapping`` must coerce back to native int/float/bool so
    ``__post_init__`` validators work and downstream maths behave.
    """
    raw = {
        "vocab_size": "58162",
        "context_length": "1024",
        "inference_context_length": "4096",
        "d_model": "384",
        "n_layers": "8",
        "n_heads": "6",
        "n_kv_heads": "2",
        "dropout": "0.1",
        "rope_base_train": "10000.0",
        "residual_scale_multiplier": "1.4",
        "fine_experts": "8",
        "experts_per_token": "2",
        "spider_sense_enabled": "true",
        "heartbeat_enabled": "false",
        "aux_loss_adaptive": "1",
    }
    config = DarwinXConfig.from_mapping(raw)

    assert isinstance(config.vocab_size, int) and config.vocab_size == 58162
    assert isinstance(config.context_length, int) and config.context_length == 1024
    assert isinstance(config.inference_context_length, int)
    assert isinstance(config.d_model, int) and config.d_model == 384
    assert isinstance(config.n_layers, int) and config.n_layers == 8
    assert isinstance(config.n_heads, int) and config.n_heads == 6
    assert isinstance(config.n_kv_heads, int) and config.n_kv_heads == 2
    assert isinstance(config.dropout, float) and config.dropout == 0.1
    assert isinstance(config.rope_base_train, float) and config.rope_base_train == 10000.0
    assert isinstance(config.residual_scale_multiplier, float)
    assert isinstance(config.fine_experts, int) and config.fine_experts == 8
    assert isinstance(config.experts_per_token, int) and config.experts_per_token == 2
    assert isinstance(config.spider_sense_enabled, bool) and config.spider_sense_enabled is True
    assert isinstance(config.heartbeat_enabled, bool) and config.heartbeat_enabled is False
    assert isinstance(config.aux_loss_adaptive, bool) and config.aux_loss_adaptive is True
