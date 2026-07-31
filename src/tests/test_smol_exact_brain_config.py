from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel


ROOT = Path(__file__).resolve().parents[2]


def test_exact_brain_config_preserves_smol_anatomy() -> None:
    raw = yaml.safe_load(
        (ROOT / "src/configs/darwin_x_1.7b_smol_exact.yaml").read_text("utf-8")
    )
    config = DarwinXConfig.from_mapping(raw)
    assert (config.d_model, config.n_layers) == (2048, 24)
    assert (config.n_heads, config.n_kv_heads) == (32, 32)
    assert config.attention_layer_indices == tuple(range(24))
    assert (config.fine_experts, config.shared_experts) == (1, 0)
    assert config.fine_expert_hidden_dim == 8192
    assert math.isclose(config.residual_scale, 1.0)
    assert config.rms_norm_eps == 1e-5
    assert config.rms_norm_fp32 is True
    assert config.rope_style == "llama"
    assert config.inter_hemispheric_enabled is True
    assert config.inter_hemispheric_residual_enabled is True


def test_single_expert_is_an_exact_dense_swiglu_path() -> None:
    raw = yaml.safe_load(
        (ROOT / "src/configs/darwin_x_1.7b_smol_exact.yaml").read_text("utf-8")
    )
    config = DarwinXConfig.from_mapping(raw)
    with torch.device("meta"):
        model = DarwinXModel(config)
    block = model.blocks[0]
    assert block.is_attention_layer
    assert len(block.moe.fine_experts) == 1
    assert len(block.moe.shared_experts) == 0
    assert block.norm1.eps == 1e-5
    assert block.norm1.fp32 is True
    assert model.inter_hemispheric is not None
    assert model.inter_hemispheric_gate is not None


def test_inter_hemispheric_zero_gate_preserves_logits_exactly() -> None:
    raw = yaml.safe_load(
        (ROOT / "src/configs/darwin_x_1.7b_smol_exact.yaml").read_text("utf-8")
    )
    exact = DarwinXConfig.from_mapping(raw)
    config = replace(
        exact,
        d_model=16,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        vocab_size=32,
        context_length=16,
        inference_context_length=16,
        fine_expert_hidden_dim=32,
        shared_expert_hidden_dim=32,
        mtp_depth=0,
        attention_indices_override=(0,),
        residual_scale_multiplier=1.0,
        heartbeat_enabled=False,
        ttm_residual_enabled=False,
        spider_calibration_enabled=False,
        spider_sense_enabled=False,
        gaba_enabled=False,
    )
    model = DarwinXModel(config).eval()
    tokens = torch.tensor([[1, 2, 3]])
    gated = model(tokens, heartbeat=False).logits
    system = model.inter_hemispheric
    model.inter_hemispheric = None
    plain = model(tokens, heartbeat=False).logits
    model.inter_hemispheric = system
    torch.testing.assert_close(gated, plain, atol=0.0, rtol=0.0)
