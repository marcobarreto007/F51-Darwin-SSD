#!/usr/bin/env python3
"""BTB State-Reuse Core — reutilização de estado COM prova de correção.

Diferente de ``run_e2e.py`` (que media apenas "menos tokens = mais rápido"
descartando o prefixo), este módulo executa a reutilização REAL de estado
usando a API de cache que o modelo Darwin-X de fato suporta:

  • Atenção (GQACausalAttention): KV cache — K/V dos tokens do prefixo são
    guardados e concatenados, exatamente como em ``inference_engine.py``.
  • SSD (SSDMixerOnly): ``conv_state`` + ``ssm_state`` recorrentes de tamanho
    O(1), via ``block.ssd(..., return_state=True)``.

A tese do BTB só é válida se o caminho "warm" (prefixo em cache + sufixo)
produzir os MESMOS logits do caminho "cold" (recomputar prefixo+sufixo). Este
módulo mede essa paridade primeiro; latência sem paridade não significa nada.

O forward espelha ``DarwinInferenceEngine._forward_with_cache`` em single-device
(sem dual-GPU, sem torch.compile) para ser auditável e determinístico.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from f51_darwin.rope import apply_rope, build_rope_cache


# ═══════════════════════════════════════════════════════════════════════════
# Slots de cache (uma cópia por camada)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AttnSlot:
    """KV cache de uma camada de atenção. Cresce O(N) com a sequência."""

    k: torch.Tensor | None = None  # [1, n_kv_heads, filled, head_dim]
    v: torch.Tensor | None = None
    filled: int = 0

    def clone(self) -> "AttnSlot":
        return AttnSlot(
            k=None if self.k is None else self.k.clone(),
            v=None if self.v is None else self.v.clone(),
            filled=self.filled,
        )

    def nbytes(self) -> int:
        if self.k is None:
            return 0
        return (self.k.numel() + self.v.numel()) * self.k.element_size()


@dataclass
class SsdSlot:
    """Estado recorrente de uma camada SSD. Tamanho O(1) na sequência."""

    conv: torch.Tensor | None = None  # [1, d_inner, conv_kernel-1]
    ssm: torch.Tensor | None = None  # [1, d_inner, d_state]

    def clone(self) -> "SsdSlot":
        return SsdSlot(
            conv=None if self.conv is None else self.conv.clone(),
            ssm=None if self.ssm is None else self.ssm.clone(),
        )

    def nbytes(self) -> int:
        total = 0
        if self.conv is not None:
            total += self.conv.numel() * self.conv.element_size()
        if self.ssm is not None:
            total += self.ssm.numel() * self.ssm.element_size()
        return total


@dataclass
class CacheState:
    """Cache completo do modelo: um slot por camada."""

    attn: dict[int, AttnSlot] = field(default_factory=dict)
    ssd: dict[int, SsdSlot] = field(default_factory=dict)

    @classmethod
    def empty(cls, model) -> "CacheState":
        state = cls()
        for idx, block in enumerate(model.blocks):
            if block.is_attention_layer:
                state.attn[idx] = AttnSlot()
            else:
                state.ssd[idx] = SsdSlot()
        return state

    def clone(self) -> "CacheState":
        return CacheState(
            attn={i: s.clone() for i, s in self.attn.items()},
            ssd={i: s.clone() for i, s in self.ssd.items()},
        )

    def attn_bytes(self) -> int:
        return sum(s.nbytes() for s in self.attn.values())

    def ssd_bytes(self) -> int:
        return sum(s.nbytes() for s in self.ssd.values())


# ═══════════════════════════════════════════════════════════════════════════
# Forward com cache (espelha inference_engine._forward_with_cache)
# ═══════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def cached_forward(
    model,
    input_ids: torch.Tensor,
    position: int,
    cache: CacheState,
) -> torch.Tensor:
    """Processa ``input_ids`` a partir da posição absoluta ``position``.

    Lê/escreve ``cache`` in-place. Retorna os logits do ÚLTIMO token,
    shape ``[1, vocab_size]``.
    """
    config = model.config
    x = model.token_embedding(input_ids)  # [1, seq, d_model]

    for layer_idx, block in enumerate(model.blocks):
        if block.is_attention_layer:
            normed = block.norm1(x)
            bsz, seq, _ = normed.shape
            q = block.attention.q_proj(normed).view(
                bsz, seq, config.n_heads, config.head_dim
            ).transpose(1, 2)
            k = block.attention.k_proj(normed).view(
                bsz, seq, config.n_kv_heads, config.head_dim
            ).transpose(1, 2)
            v = block.attention.v_proj(normed).view(
                bsz, seq, config.n_kv_heads, config.head_dim
            ).transpose(1, 2)

            rope_base = block.attention.rope_base_infer
            cos, sin = build_rope_cache(
                position + seq, config.head_dim, base=rope_base, device=normed.device
            )
            q = apply_rope(q, cos[position : position + seq], sin[position : position + seq])
            k = apply_rope(k, cos[position : position + seq], sin[position : position + seq])

            slot = cache.attn[layer_idx]
            if slot.filled == 0 or slot.k is None:
                slot.k, slot.v = k, v
            else:
                slot.k = torch.cat([slot.k, k], dim=2)
                slot.v = torch.cat([slot.v, v], dim=2)
            slot.filled += seq

            # Máscara causal por POSIÇÃO ABSOLUTA. Não dá para usar
            # is_causal=True com KV cache onde k_len > q_len: o SDPA do PyTorch
            # alinha a máscara triangular ao topo-esquerda, então as queries do
            # sufixo enxergariam só os primeiros keys, nunca o prefixo em cache.
            k_len = slot.filled
            q_pos = position + torch.arange(seq, device=q.device)
            k_pos = torch.arange(k_len, device=q.device)
            attn_mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)  # [seq, k_len]
            attn = F.scaled_dot_product_attention(
                q, slot.k, slot.v,
                attn_mask=attn_mask,
                enable_gqa=True,
            )
            attn = attn.transpose(1, 2).contiguous().view(bsz, seq, config.d_model)
            mixed = block.attention.o_proj(attn)
        else:
            normed = block.norm1(x)
            slot = cache.ssd[layer_idx]
            mixed, new_conv, new_ssm = block.ssd(
                normed,
                conv_state=slot.conv,
                ssm_state=slot.ssm,
                return_state=True,
            )
            slot.conv, slot.ssm = new_conv, new_ssm

        x = x + block.residual_scale * block.dropout(mixed)

        if block.moe is not None:
            moe_out, _aux = block.moe(block.norm2(x))
        else:
            moe_out = block.ffn(block.norm2(x))
        x = x + block.residual_scale * block.dropout(moe_out)

    hidden = model.norm(x)
    logits = model.lm_head(hidden[:, -1:, :])  # [1, 1, vocab]
    return logits[:, -1, :]  # [1, vocab]


# ═══════════════════════════════════════════════════════════════════════════
# Caminhos cold / warm
# ═══════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def cold_logits(model, prefix_ids: torch.Tensor, suffix_ids: torch.Tensor) -> torch.Tensor:
    """Recomputa tudo: prefixo+sufixo do zero. Retorna logits do último token."""
    full = torch.cat([prefix_ids, suffix_ids], dim=1)
    cache = CacheState.empty(model)
    return cached_forward(model, full, position=0, cache=cache)


@torch.no_grad()
def build_prefix_cache(model, prefix_ids: torch.Tensor) -> CacheState:
    """Faz o prefill do prefixo UMA vez e devolve o cache resultante."""
    cache = CacheState.empty(model)
    cached_forward(model, prefix_ids, position=0, cache=cache)
    return cache


@torch.no_grad()
def warm_logits(
    model,
    prefix_cache: CacheState,
    prefix_len: int,
    suffix_ids: torch.Tensor,
) -> torch.Tensor:
    """Reutiliza o cache do prefixo e processa só o sufixo.

    Restaura uma cópia fresca do cache do prefixo (para não contaminar entre
    requests) e continua na posição absoluta ``prefix_len``.
    """
    cache = prefix_cache.clone()
    return cached_forward(model, suffix_ids, position=prefix_len, cache=cache)


# ═══════════════════════════════════════════════════════════════════════════
# Paridade
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ParityReport:
    max_abs_diff: float
    kl_div: float
    top1_agree: bool
    cold_top1: int
    warm_top1: int


def parity(cold: torch.Tensor, warm: torch.Tensor) -> ParityReport:
    """Compara logits cold vs warm do mesmo (prefixo, sufixo)."""
    diff = (cold - warm).abs().max().item()
    logp_cold = F.log_softmax(cold.float(), dim=-1)
    logp_warm = F.log_softmax(warm.float(), dim=-1)
    kl = F.kl_div(logp_warm, logp_cold, log_target=True, reduction="batchmean").item()
    c1 = int(cold.argmax(dim=-1).item())
    w1 = int(warm.argmax(dim=-1).item())
    return ParityReport(
        max_abs_diff=diff,
        kl_div=abs(kl),
        top1_agree=(c1 == w1),
        cold_top1=c1,
        warm_top1=w1,
    )
