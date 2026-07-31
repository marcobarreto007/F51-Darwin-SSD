from __future__ import annotations

import copy
from collections.abc import Iterable

import torch
from torch import nn

from .ssd_mixer import DiscreteSSDMixer


class SSDGPT2Attention(nn.Module):
    """GPT-2 block-compatible adapter backed by a discrete SSD recurrence."""

    def __init__(self, original: nn.Module, *, layer_idx: int) -> None:
        super().__init__()
        config = original.config
        self.layer_idx = int(layer_idx)
        self.mixer = DiscreteSSDMixer(
            int(config.hidden_size),
            int(config.num_attention_heads),
            dropout=float(config.resid_pdrop),
        )
        self.mixer.load_gpt2_attention_projections(
            qkv_weight=original.c_attn.weight.detach(),
            qkv_bias=original.c_attn.bias.detach(),
            out_weight=original.c_proj.weight.detach(),
            out_bias=original.c_proj.bias.detach(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values=None,
        cache_position=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions: bool = False,
        **kwargs,
    ):
        del (
            past_key_values,
            cache_position,
            attention_mask,
            head_mask,
            encoder_attention_mask,
            kwargs,
        )
        if encoder_hidden_states is not None:
            raise ValueError("Darwin SSD mixer does not implement cross-attention")
        output = self.mixer(hidden_states)
        weights = self.mixer.materialize_mixer(hidden_states) if output_attentions else None
        return output, weights


class DarwinTransferStudent(nn.Module):
    def __init__(self, model: nn.Module, replaced_layers: Iterable[int]) -> None:
        super().__init__()
        self.model = model
        self.replaced_layers = tuple(sorted(set(int(i) for i in replaced_layers)))

    @classmethod
    def from_teacher(
        cls,
        teacher: nn.Module,
        *,
        replaced_layers: Iterable[int],
    ) -> "DarwinTransferStudent":
        model = copy.deepcopy(teacher)
        indices = tuple(sorted(set(int(i) for i in replaced_layers)))
        layer_count = len(model.transformer.h)
        if not indices or indices[0] < 0 or indices[-1] >= layer_count:
            raise ValueError(
                f"replaced layers must be within [0, {layer_count}); actual={indices}"
            )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for index in indices:
            model.transformer.h[index].attn = SSDGPT2Attention(
                model.transformer.h[index].attn,
                layer_idx=index,
            )
        return cls(model, indices)

    @property
    def config(self):
        return self.model.config

    @property
    def attention_layer_count(self) -> int:
        return sum(
            not isinstance(block.attn, SSDGPT2Attention)
            for block in self.model.transformer.h
        )

    def ssd_attention(self, layer_idx: int) -> SSDGPT2Attention:
        attention = self.model.transformer.h[int(layer_idx)].attn
        if not isinstance(attention, SSDGPT2Attention):
            raise ValueError(f"layer {layer_idx} is not an SSD layer")
        return attention

    def set_trainable_stage(self, stage: str) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if stage in {"orientation", "alignment"}:
            for index in self.replaced_layers:
                for parameter in self.ssd_attention(index).parameters():
                    parameter.requires_grad_(True)
        elif stage == "distillation":
            for parameter in self.model.parameters():
                parameter.requires_grad_(True)
        else:
            raise ValueError(f"unsupported training stage: {stage}")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
