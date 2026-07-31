from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.estimation import estimate_darwin_x_parameters


ROOT = Path(__file__).resolve().parents[2]


def test_adapted_init_is_explicitly_allowed() -> None:
    assert DarwinXConfig(init="f51_adapted").init == "f51_adapted"


def test_unknown_init_is_rejected() -> None:
    with pytest.raises(ValueError, match="random or f51_adapted"):
        DarwinXConfig(init="smol")


def test_smol_target_anatomy_is_exact() -> None:
    raw = yaml.safe_load(
        (
            ROOT / "src" / "configs" / "darwin_x_1.6b_smol_transplant.yaml"
        ).read_text(encoding="utf-8")
    )
    cfg = DarwinXConfig.from_mapping(raw)

    assert (cfg.vocab_size, cfg.d_model, cfg.n_layers) == (49_152, 1_920, 16)
    assert (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim) == (30, 30, 64)
    assert cfg.attention_layer_indices == (3, 7, 11, 15)
    assert (
        cfg.fine_experts,
        cfg.shared_experts,
        cfg.fine_expert_hidden_dim,
    ) == (14, 2, 896)
    assert cfg.qkv_bias is False
    assert cfg.rope_base_train == cfg.rope_base_infer == 130_000.0
    assert estimate_darwin_x_parameters(cfg)["total"] == 1_674_720_224
