from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from f51_darwin.darwin_x_core.block import DarwinXBlock
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.estimation import estimate_darwin_x_parameters
from f51_darwin.darwin_x_core.layers import DenseSwiGLU


def _config(**changes) -> DarwinXConfig:
    base = DarwinXConfig(
        model_name="dense-test",
        vocab_size=32,
        context_length=8,
        inference_context_length=8,
        d_model=8,
        n_layers=1,
        n_heads=2,
        n_kv_heads=2,
        fine_experts=1,
        shared_experts=0,
        experts_per_token=1,
        fine_expert_hidden_dim=16,
        shared_expert_hidden_dim=16,
        mtp_depth=0,
        mtp_weight=0.0,
        jepa_weight=0.0,
        ghost_weight=0.0,
        ghost_enabled=False,
        curiosity_weight=0.0,
        spider_sense_enabled=False,
        heartbeat_enabled=False,
        nitro_enabled=False,
        attention_indices_override=(0,),
    )
    return replace(base, **changes)


def test_dense_config_is_opt_in_and_validated() -> None:
    assert DarwinXConfig().feed_forward_kind == "moe"
    dense = _config(feed_forward_kind="dense_swiglu")
    assert dense.feed_forward_kind == "dense_swiglu"
    with pytest.raises(ValueError, match="feed_forward_kind"):
        replace(dense, feed_forward_kind="unknown")
    with pytest.raises(ValueError, match="fine_experts=1"):
        replace(dense, fine_experts=2)
    with pytest.raises(ValueError, match="shared_experts=0"):
        replace(dense, shared_experts=1)


def test_dense_swiglu_matches_explicit_equation_and_has_gradients() -> None:
    torch.manual_seed(51)
    layer = DenseSwiGLU(8, 16, dropout=0.0).eval()
    inputs = torch.randn(2, 3, 8)
    expected = layer.down_proj(
        F.silu(layer.gate_proj(inputs)) * layer.up_proj(inputs)
    )
    actual = layer(inputs)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    actual.square().mean().backward()
    assert all(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad) > 0
        for parameter in layer.parameters()
    )


def test_block_registers_exactly_one_feed_forward_implementation() -> None:
    dense = DarwinXBlock(
        _config(feed_forward_kind="dense_swiglu"),
        is_attention_layer=True,
    )
    legacy = DarwinXBlock(_config(), is_attention_layer=True)
    assert dense.ffn is not None
    assert dense.moe is None
    assert legacy.ffn is None
    assert legacy.moe is not None
    assert not any(".moe." in key for key in dense.state_dict())


def test_dense_parameter_estimate_contains_no_router_or_moe() -> None:
    estimate = estimate_darwin_x_parameters(
        _config(feed_forward_kind="dense_swiglu")
    )
    expected = 3 * 8 * 16
    assert estimate["dense_ffn_total"] == expected
    assert estimate["moe_total"] == 0
    assert estimate["router_total"] == 0
