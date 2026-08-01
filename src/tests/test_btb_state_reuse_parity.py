"""Gate de correção do BTB: reutilização de estado == recompute.

O benchmark BTB anterior (research/btb/run_e2e.py) media "menos tokens =
mais rápido" descartando o prefixo — não provava reutilização nenhuma. Este
teste fecha esse buraco: prova que continuar a partir do cache do prefixo
(KV de atenção + estado recorrente SSD) produz os MESMOS logits do que
recomputar prefixo+sufixo do zero.

Se este teste passar, a tese do BTB é válida NESTA arquitetura. Se falhar, a
reutilização está corrompendo a saída e qualquer speedup é ilusório.

CPU, config minúscula, determinístico — roda em segundos.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.rope import apply_rope, build_rope_cache


def _config() -> DarwinXConfig:
    # 4 camadas: 0,2 = SSD ; 1,3 = atenção. Exercita AMBOS os caminhos de cache.
    return DarwinXConfig(
        model_name="F51-btb-parity-test",
        vocab_size=48,
        context_length=256,
        inference_context_length=256,
        d_model=32,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        attention_indices_override=(1, 3),
        feed_forward_kind="dense_swiglu",
        fine_experts=1,
        shared_experts=0,
        experts_per_token=1,
        fine_expert_hidden_dim=64,
        shared_expert_hidden_dim=64,
        mtp_depth=0,
        mtp_weight=0.0,
        jepa_weight=0.0,
        ghost_weight=0.0,
        ghost_enabled=False,
        curiosity_weight=0.0,
        spider_sense_enabled=False,
        heartbeat_enabled=False,
        nitro_enabled=False,
        dropout=0.0,
    )


@torch.no_grad()
def _cached_forward(model, input_ids, position, attn_cache, ssd_cache):
    """Espelha inference_engine._forward_with_cache (single-device)."""
    config = model.config
    x = model.token_embedding(input_ids)
    for li, block in enumerate(model.blocks):
        if block.is_attention_layer:
            normed = block.norm1(x)
            b, s, _ = normed.shape
            q = block.attention.q_proj(normed).view(b, s, config.n_heads, config.head_dim).transpose(1, 2)
            k = block.attention.k_proj(normed).view(b, s, config.n_kv_heads, config.head_dim).transpose(1, 2)
            v = block.attention.v_proj(normed).view(b, s, config.n_kv_heads, config.head_dim).transpose(1, 2)
            rope_base = block.attention.rope_base_infer
            cos, sin = build_rope_cache(position + s, config.head_dim, base=rope_base, device=normed.device)
            q = apply_rope(q, cos[position:position + s], sin[position:position + s])
            k = apply_rope(k, cos[position:position + s], sin[position:position + s])
            past = attn_cache.get(li)
            if past is not None:
                k = torch.cat([past[0], k], dim=2)
                v = torch.cat([past[1], v], dim=2)
            attn_cache[li] = (k, v)
            # Máscara causal por posição absoluta (is_causal=True alinha ao
            # topo-esquerda quando k_len > q_len e ignoraria o prefixo cacheado).
            k_len = k.shape[2]
            q_pos = position + torch.arange(s)
            k_pos = torch.arange(k_len)
            attn_mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=True)
            attn = attn.transpose(1, 2).contiguous().view(b, s, config.d_model)
            mixed = block.attention.o_proj(attn)
        else:
            normed = block.norm1(x)
            conv, ssm = ssd_cache.get(li, (None, None))
            mixed, nconv, nssm = block.ssd(normed, conv_state=conv, ssm_state=ssm, return_state=True)
            ssd_cache[li] = (nconv, nssm)
        x = x + block.residual_scale * block.dropout(mixed)
        moe_out = block.ffn(block.norm2(x))
        x = x + block.residual_scale * block.dropout(moe_out)
    hidden = model.norm(x)
    return model.lm_head(hidden[:, -1:, :])[:, -1, :]


@pytest.fixture
def model():
    torch.manual_seed(51)
    return DarwinXModel(_config()).eval()


def _make_ids(config, n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, config.vocab_size, (1, n), generator=g, dtype=torch.long)


def test_warm_reuse_matches_cold_recompute(model):
    """Prefixo em cache + sufixo deve dar os mesmos logits que recomputar tudo."""
    config = model.config
    prefix = _make_ids(config, 40, seed=1)
    suffix = _make_ids(config, 8, seed=2)

    # Cold: recomputa prefixo+sufixo do zero.
    cold = _cached_forward(model, torch.cat([prefix, suffix], dim=1), 0, {}, {})

    # Warm: prefill do prefixo, depois só o sufixo continuando do cache.
    attn_c: dict = {}
    ssd_c: dict = {}
    _cached_forward(model, prefix, 0, attn_c, ssd_c)
    warm = _cached_forward(model, suffix, prefix.shape[1], attn_c, ssd_c)

    max_diff = (cold - warm).abs().max().item()
    assert max_diff < 1e-4, f"logits divergiram: max_abs_diff={max_diff:.2e}"
    assert int(cold.argmax()) == int(warm.argmax()), "top-1 token divergiu"


def test_reuse_is_not_just_dropping_prefix(model):
    """Controle negativo: ignorar o prefixo (o que run_e2e.py fazia) NÃO bate.

    Garante que o teste positivo não passa trivialmente — processar só o
    sufixo sem o cache do prefixo produz logits diferentes.
    """
    config = model.config
    prefix = _make_ids(config, 40, seed=1)
    suffix = _make_ids(config, 8, seed=2)

    cold = _cached_forward(model, torch.cat([prefix, suffix], dim=1), 0, {}, {})
    # "Fake reuse": sufixo sozinho, sem prefixo, posição 0 (como o benchmark antigo).
    fake = _cached_forward(model, suffix, 0, {}, {})

    max_diff = (cold - fake).abs().max().item()
    assert max_diff > 1e-3, (
        "descartar o prefixo deveria mudar os logits; se não muda, o modelo "
        "está ignorando o contexto e o teste positivo é vacuamente verdadeiro"
    )


def test_ssd_state_is_constant_size_vs_attention(model):
    """O estado SSD é O(1); o KV de atenção cresce O(N) — mede de verdade."""
    config = model.config

    def cache_bytes(prefix_len):
        prefix = _make_ids(config, prefix_len, seed=7)
        attn_c: dict = {}
        ssd_c: dict = {}
        _cached_forward(model, prefix, 0, attn_c, ssd_c)
        attn_b = sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                     for k, v in attn_c.values())
        ssd_b = sum((0 if c is None else c.numel() * c.element_size())
                    + (0 if s is None else s.numel() * s.element_size())
                    for c, s in ssd_c.values())
        return attn_b, ssd_b

    attn_short, ssd_short = cache_bytes(16)
    attn_long, ssd_long = cache_bytes(64)

    # Atenção quadruplica com 4x tokens; SSD fica idêntico.
    assert attn_long > attn_short * 3, "KV de atenção deveria crescer ~linear com N"
    assert ssd_short == ssd_long, "estado SSD deveria ser constante na sequência"
