from __future__ import annotations

import math

from f51_darwin.darwin_x_core.config import DarwinXConfig


def estimate_darwin_x_parameters(config: DarwinXConfig) -> dict[str, int]:
    d = config.d_model
    head_dim = config.head_dim
    kv_dim = config.n_kv_heads * head_dim
    embedding = config.vocab_size * d
    ssd_per_layer = (
        (2 * d) * d
        + (2 * d) * config.ssm_state
        + (2 * d)
        + (2 * d) * 4
        + (math.ceil(d / 16) + config.ssm_state * 2) * (2 * d)
        + d * (2 * d)
    )
    gqa_per_layer = d * d + 2 * d * kv_dim + d * d
    fine_expert = 3 * d * config.fine_expert_hidden_dim
    shared_expert = 3 * d * config.shared_expert_hidden_dim
    moe_per_layer = config.fine_experts * fine_expert + config.shared_experts * shared_expert
    router_per_layer = d * config.fine_experts + config.fine_experts
    mtp = config.mtp_depth * d * d
    jepa = 2 * d * d + d
    norms = (2 * config.n_layers + 1) * d
    ssd_total = len(config.ssd_layer_indices) * ssd_per_layer
    gqa_total = len(config.attention_layer_indices) * gqa_per_layer
    if config.feed_forward_kind == "dense_swiglu":
        dense_ffn_total = config.n_layers * fine_expert
        router_total = 0
        moe_total = 0
        feed_forward_total = dense_ffn_total
        active_feed_forward = dense_ffn_total
    else:
        dense_ffn_total = 0
        router_total = config.n_layers * router_per_layer
        moe_total = config.n_layers * (
            moe_per_layer + router_per_layer
        )
        feed_forward_total = moe_total
        active_experts_per_layer = (
            config.experts_per_token * fine_expert
            + config.shared_experts * shared_expert
        )
        active_feed_forward = (
            config.n_layers * active_experts_per_layer
        )
    total = (
        embedding + ssd_total + gqa_total + feed_forward_total
        + mtp + jepa + norms
    )
    active_per_token = (
        embedding + ssd_total + gqa_total + active_feed_forward
        + mtp + jepa + norms
    )
    return {
        "embedding": embedding,
        "ssd_total": int(ssd_total),
        "gqa_total": int(gqa_total),
        "moe_total": int(moe_total),
        "router_total": int(router_total),
        "dense_ffn_total": int(dense_ffn_total),
        "mtp": int(mtp),
        "jepa": int(jepa),
        "norms": int(norms),
        "total": int(total),
        "active_per_token": int(active_per_token),
    }
