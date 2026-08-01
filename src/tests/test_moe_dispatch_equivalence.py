"""Gate de equivalencia do dispatch MoE em lote.

O dispatch antigo iterava os experts ativos em Python, com dois pontos de
sincronizacao GPU->CPU por expert por camada (``int(t.item())`` e
``torch.where(mask)``). No 100M isso davam 381 chamadas de ``nonzero`` por
forward (32 experts x 12 camadas), e o profiler mediu 274 ms de CPU contra
20.6 ms de CUDA — 93% do forward era a CPU esperando sync.

O dispatch novo agrupa os tokens uma vez com ``argsort`` estavel e fatia por
fronteiras de expert, com um unico sync por camada.

Este teste prova que a saida e IDENTICA. Latencia sem equivalencia nao vale
nada — foi exatamente o erro que produziu os benchmarks vacuosos do BTB.
"""

from __future__ import annotations

import pytest
import torch

from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.moe import DeepSeekStyleMoE


def _config(num_experts: int = 8, k: int = 2) -> DarwinXConfig:
    return DarwinXConfig(
        model_name="F51-moe-dispatch-test",
        vocab_size=64,
        context_length=128,
        inference_context_length=128,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        feed_forward_kind="moe",
        fine_experts=num_experts,
        shared_experts=1,
        experts_per_token=k,
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
def _legacy_dispatch(moe: DeepSeekStyleMoE, flat_x, flat_weights, flat_indices):
    """Reimplementa o loop antigo, o que estava em moe.py antes do lote.

    Mantido aqui (e nao em producao) para servir de referencia do gate.
    """
    output = torch.zeros_like(flat_x)
    active_experts = flat_indices.unique()
    for expert_idx_tensor in active_experts:
        expert_idx = int(expert_idx_tensor.item())
        expert = moe.fine_experts[expert_idx]
        mask = flat_indices == expert_idx
        token_rows, slots = torch.where(mask)
        expert_out = expert(flat_x[token_rows], moe.neuroendocrine, expert_idx)
        output[token_rows] += flat_weights[token_rows, slots].unsqueeze(-1) * expert_out
    return output


@pytest.fixture
def moe():
    torch.manual_seed(51)
    return DeepSeekStyleMoE(_config()).eval()


def _routing(moe, batch=3, seq=17):
    torch.manual_seed(7)
    config = moe.config
    x = torch.randn(batch, seq, config.d_model)
    weights, indices, _logits = moe.fine_router(x)
    flat_x = x.reshape(-1, config.d_model)
    return (
        flat_x,
        weights.reshape(-1, config.experts_per_token),
        indices.reshape(-1, config.experts_per_token),
    )


def test_batched_dispatch_matches_legacy_loop(moe):
    """Dispatch em lote deve reproduzir o loop antigo bit a bit."""
    flat_x, flat_weights, flat_indices = _routing(moe)

    with torch.no_grad():
        legacy = _legacy_dispatch(moe, flat_x, flat_weights, flat_indices)
        batched = moe._dispatch_fine_experts(flat_x, flat_weights, flat_indices)

    max_diff = (legacy - batched).abs().max().item()
    assert max_diff == 0.0, f"dispatch divergiu do loop antigo: max_abs_diff={max_diff:.3e}"


def test_batched_dispatch_is_not_trivially_zero(moe):
    """Controle negativo: a saida tem de ser nao-trivial.

    Sem isto, um dispatch que retornasse zeros passaria no teste acima se o
    loop antigo tambem retornasse zeros por algum bug de roteamento.
    """
    flat_x, flat_weights, flat_indices = _routing(moe)
    with torch.no_grad():
        batched = moe._dispatch_fine_experts(flat_x, flat_weights, flat_indices)
    assert batched.abs().max().item() > 1e-6, "dispatch produziu saida nula"
    assert torch.isfinite(batched).all(), "dispatch produziu NaN/Inf"


@pytest.mark.parametrize("num_experts,k", [(2, 1), (4, 2), (16, 2), (8, 4)])
def test_equivalence_across_expert_counts(num_experts, k):
    """A equivalencia tem de valer para qualquer (num_experts, top-k)."""
    torch.manual_seed(51)
    m = DeepSeekStyleMoE(_config(num_experts=num_experts, k=k)).eval()
    flat_x, flat_weights, flat_indices = _routing(m)
    with torch.no_grad():
        legacy = _legacy_dispatch(m, flat_x, flat_weights, flat_indices)
        batched = m._dispatch_fine_experts(flat_x, flat_weights, flat_indices)
    assert (legacy - batched).abs().max().item() == 0.0


def test_dispatch_handles_single_token(moe):
    """Caso degenerado: um unico token nao pode quebrar o fatiamento."""
    config = moe.config
    torch.manual_seed(3)
    x = torch.randn(1, 1, config.d_model)
    weights, indices, _ = moe.fine_router(x)
    flat_x = x.reshape(-1, config.d_model)
    fw = weights.reshape(-1, config.experts_per_token)
    fi = indices.reshape(-1, config.experts_per_token)
    with torch.no_grad():
        legacy = _legacy_dispatch(moe, flat_x, fw, fi)
        batched = moe._dispatch_fine_experts(flat_x, fw, fi)
    assert (legacy - batched).abs().max().item() == 0.0
