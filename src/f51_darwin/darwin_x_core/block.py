from __future__ import annotations

from typing import Any

import torch
from torch import nn

from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.layers import (
    DenseSwiGLU,
    GQACausalAttention,
    SSDMixerOnly,
)
from f51_darwin.darwin_x_core.moe import DeepSeekStyleMoE
from f51_darwin.ssd_block import RMSNorm


class DarwinXBlock(nn.Module):
    def __init__(self, config: DarwinXConfig, *, is_attention_layer: bool) -> None:
        super().__init__()
        self.is_attention_layer = is_attention_layer
        self.residual_scale = config.residual_scale
        self.norm1 = RMSNorm(
            config.d_model,
            eps=config.rms_norm_eps,
            fp32=config.rms_norm_fp32,
        )
        self.norm2 = RMSNorm(
            config.d_model,
            eps=config.rms_norm_eps,
            fp32=config.rms_norm_fp32,
        )
        if is_attention_layer:
            self.attention = GQACausalAttention(config)
        else:
            self.ssd = SSDMixerOnly(config)
        self.moe = None
        self.ffn = None
        if config.feed_forward_kind == "dense_swiglu":
            self.ffn = DenseSwiGLU(
                config.d_model,
                config.fine_expert_hidden_dim,
                config.dropout,
            )
        else:
            self.moe = DeepSeekStyleMoE(config)
        self.dropout = nn.Dropout(config.dropout)

        # NOVO: GABAergic inhibition (opcional, toggle via config)
        self.gaba_enabled = getattr(config, 'gaba_enabled', False)
        if self.gaba_enabled:
            from f51_darwin.gaba_inhibition import GABAergicLayer, GABAConfig
            gaba_cfg = GABAConfig(d_model=config.d_model)
            self.gaba = GABAergicLayer(gaba_cfg)
        else:
            self.gaba = None

        # NOVO: Hemispheric lateralization tag
        self.hemisphere_tag: str = "unified"

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.is_attention_layer:
            mixed = self.attention(self.norm1(x))
        else:
            mixed = self.ssd(self.norm1(x))
        x = x + self.residual_scale * self.dropout(mixed)
        normalized = self.norm2(x)
        if self.ffn is not None:
            feed_forward = self.ffn(normalized)
            aux: dict[str, Any] = {}
        else:
            assert self.moe is not None
            feed_forward, aux = self.moe(normalized)

        # NOVO: GABAergic inhibition após MoE, antes do residual
        if self.gaba is not None:
            gaba_delta, observation = self.gaba(
                feed_forward,
                mutate_state=False,
            )
            feed_forward = feed_forward + gaba_delta
            aux["gaba_observation"] = observation

        x = x + self.residual_scale * self.dropout(feed_forward)
        return x, aux
