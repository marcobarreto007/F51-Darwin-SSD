from __future__ import annotations

import math
from pathlib import Path

import torch
import yaml

from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel


ROOT = Path(__file__).resolve().parents[2]


def _config() -> DarwinXConfig:
    raw = yaml.safe_load(
        (ROOT / "src/configs/darwin_x_1.7b_smol_dense.yaml").read_text(
            "utf-8"
        )
    )
    return DarwinXConfig.from_mapping(raw)


def test_dense_brain_config_preserves_smol_anatomy_without_moe() -> None:
    config = _config()
    assert config.feed_forward_kind == "dense_swiglu"
    assert (config.d_model, config.n_layers) == (2048, 24)
    assert (config.n_heads, config.n_kv_heads) == (32, 32)
    assert config.attention_layer_indices == tuple(range(24))
    assert (config.fine_experts, config.shared_experts) == (1, 0)
    assert config.fine_expert_hidden_dim == 8192
    assert math.isclose(config.residual_scale, 1.0)
    assert config.rms_norm_eps == 1e-5
    assert config.rms_norm_fp32 is True
    assert config.rope_style == "llama"
    assert config.dae_enabled is False
    assert config.inter_hemispheric_residual_enabled is True

    with torch.device("meta"):
        model = DarwinXModel(config)
    assert all(block.moe is None for block in model.blocks)
    assert all(block.ffn is not None for block in model.blocks)
    assert not any(".moe." in key for key in model.state_dict())
    assert model.inter_hemispheric_gate is not None
