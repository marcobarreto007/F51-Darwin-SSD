"""
F51 Darwin KV Cache — Inferência com estado persistente.

Elimina recomputação O(n²) na geração autoregressiva.
SSD: cache do estado oculto h entre tokens.
Attention: cache de K e V passados.

Com KV cache, cada token gerado custa O(1) em vez de O(n).
Essencial para modelos em crescimento exponencial (MoE).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class KVCache:
    """Cache de estados para geração autoregressiva eficiente.

    Armazena:
        - SSM hidden states (para blocos SSD)
        - Attention K, V (para blocos de atenção)
        - MoE router states (para blocos MoE)
    """

    def __init__(self, batch_size: int, max_length: int, device: torch.device):
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self._ssd_states: dict[int, torch.Tensor] = {}  # layer_idx -> h state
        self._ssd_conv: dict[int, torch.Tensor] = {}  # layer_idx -> conv tail
        self._attn_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}  # layer_idx -> (K, V)
        self._seq_len: int = 0

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def increment(self):
        self._seq_len += 1

    def get_ssd_state(self, layer_idx: int) -> torch.Tensor | None:
        return self._ssd_states.get(layer_idx)

    def set_ssd_state(self, layer_idx: int, state: torch.Tensor):
        self._ssd_states[layer_idx] = state

    def get_ssd_conv(self, layer_idx: int) -> torch.Tensor | None:
        return self._ssd_conv.get(layer_idx)

    def set_ssd_conv(self, layer_idx: int, state: torch.Tensor):
        self._ssd_conv[layer_idx] = state

    def get_attn_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self._attn_kv.get(layer_idx)

    def set_attn_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        # _attn_forward_with_cache already concatenates with past before calling us.
        # We just store the pre-concatenated tensors.
        self._attn_kv[layer_idx] = (k, v)

    def clear(self):
        self._ssd_states.clear()
        self._ssd_conv.clear()
        self._attn_kv.clear()
        self._seq_len = 0

    # ═══════════════════════════════════════════════════════════
    # F51 SSD Cache Retarget — checkpoint/restore entre slots
    # Baseado no patch: f51_llama_cpp_swa_checkpoint_seqid_retarget
    # Permite salvar e restaurar cache entre diferentes slots de servidor.
    # ═══════════════════════════════════════════════════════════

    def checkpoint_state(self) -> dict:
        """Serializa estado do cache para save/restore entre slots.

        Retorna dicionário com todos os estados, pronto para serialização.
        Os tensores são detached e movidos para CPU para segurança.
        """
        state: dict = {
            "version": 1,
            "seq_len": self._seq_len,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "ssd_states": {},
            "ssd_conv": {},
            "attn_kv": {},
        }

        for layer_idx, h in self._ssd_states.items():
            state["ssd_states"][str(layer_idx)] = h.detach().cpu().clone()

        for layer_idx, c in self._ssd_conv.items():
            state["ssd_conv"][str(layer_idx)] = c.detach().cpu().clone()

        for layer_idx, (k, v) in self._attn_kv.items():
            state["attn_kv"][str(layer_idx)] = (
                k.detach().cpu().clone(),
                v.detach().cpu().clone(),
            )

        return state

    def restore_state(self, state: dict, target_device: torch.device | None = None) -> None:
        """Restaura cache a partir de estado serializado.

        Suporta retarget entre diferentes slots de servidor:
        - Se batch_size difere, redimensiona estados para o novo batch.
        - Se o slot alvo é diferente, os tensores são realocados corretamente.

        Args:
            state: dicionário retornado por checkpoint_state()
            target_device: dispositivo alvo. Se None, usa self.device.
        """
        if target_device is None:
            target_device = self.device

        self.clear()
        self._seq_len = state.get("seq_len", 0)

        # Restaura SSD states com retarget de batch
        for key, tensor in state.get("ssd_states", {}).items():
            layer_idx = int(key)
            t = tensor.to(target_device)
            # Retarget: redimensiona se batch_size mudou
            if t.shape[0] != self.batch_size:
                t = self._retarget_batch(t, self.batch_size)
            self._ssd_states[layer_idx] = t

        # Restaura SSD conv states
        for key, tensor in state.get("ssd_conv", {}).items():
            layer_idx = int(key)
            t = tensor.to(target_device)
            if t.shape[0] != self.batch_size:
                t = self._retarget_batch(t, self.batch_size)
            self._ssd_conv[layer_idx] = t

        # Restaura Attention KV states
        for key, (k, v) in state.get("attn_kv", {}).items():
            layer_idx = int(key)
            k_t = k.to(target_device)
            v_t = v.to(target_device)
            if k_t.shape[0] != self.batch_size:
                k_t = self._retarget_batch(k_t, self.batch_size)
                v_t = self._retarget_batch(v_t, self.batch_size)
            self._attn_kv[layer_idx] = (k_t, v_t)

    @staticmethod
    def _retarget_batch(tensor: torch.Tensor, target_batch: int) -> torch.Tensor:
        """Redimensiona a dimensão batch de um tensor de estado.

        - Se target_batch > tensor.shape[0]: replica o primeiro slot
        - Se target_batch < tensor.shape[0]: trunca
        - Se igual: retorna como está
        """
        src_batch = tensor.shape[0]
        if src_batch == target_batch:
            return tensor
        elif src_batch < target_batch:
            # Expande: replica primeiro slot
            repeats = [target_batch // src_batch + 1] + [1] * (tensor.ndim - 1)
            expanded = tensor.repeat(repeats)
            return expanded[:target_batch]
        else:
            # Trunca
            return tensor[:target_batch]


# ═══════════════════════════════════════════════════════════
# SSD com suporte a KV cache (estado oculto persistente)
# ═══════════════════════════════════════════════════════════

def selective_scan_with_cache(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor | None = None,
    h_prev: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Selective scan com cache de estado oculto.

    Se h_prev é fornecido, continua o scan a partir desse estado.
    Retorna (output, h_new) onde h_new é o estado final.

    Shapes:
        u: [batch, dim, seq]
        delta: [batch, dim, seq]
        a: [dim, state_size]
        b: [batch, state_size, seq]
        c: [batch, state_size, seq]
        d: [dim] optional
        h_prev: [batch, dim, state_size] optional
    """
    batch, dim, seq_len = u.shape
    state_size = a.shape[1]

    if h_prev is not None:
        h = h_prev
    else:
        h = u.new_zeros(batch, dim, state_size)

    outputs: list[torch.Tensor] = []
    a_expanded = a.unsqueeze(0)

    for step in range(seq_len):
        delta_t = delta[:, :, step].unsqueeze(-1)
        b_t = b[:, :, step].unsqueeze(1)
        c_t = c[:, :, step].unsqueeze(1)
        u_t = u[:, :, step].unsqueeze(-1)

        decay = torch.exp(delta_t * a_expanded)
        h = decay * h + delta_t * b_t * u_t
        y = (h * c_t).sum(dim=-1)
        if d is not None:
            y = y + d * u[:, :, step]
        outputs.append(y)

    output = torch.stack(outputs, dim=-1)
    return output, h


# ═══════════════════════════════════════════════════════════
# GERAÇÃO OTIMIZADA COM KV CACHE
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def generate_with_cache(
    model,
    prompt_ids: list[int],
    tokenizer=None,
    *,
    max_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    eos_id: int | None = None,
    stop_ids: set[int] | None = None,
) -> tuple[str, list[int], str]:
    """Geração autoregressiva com KV cache.

    Pré-processa o prompt uma vez (cacheia K,V e estados SSM).
    Depois gera token por token usando o cache — O(1) por token.
    """
    device = next(model.parameters()).device
    config = model.config

    if stop_ids is None:
        stop_ids = set()
    if eos_id is not None:
        stop_ids.add(eos_id)
    elif tokenizer is not None:
        stop_ids.add(tokenizer.eos_id)

    # ── Preencher prompt ──
    input_ids = list(prompt_ids)
    if len(input_ids) > config.context_length:
        input_ids = input_ids[-config.context_length:]

    # ── Forward inicial: processa o prompt inteiro e cacheia estados ──
    context = torch.tensor([input_ids], dtype=torch.long, device=device)
    x = model.token_embedding(context)  # [1, L, D]
    batch_size, seq_len, d_model = x.shape

    cache = KVCache(batch_size, config.context_length, device)
    cache._seq_len = seq_len

    # Processa cada bloco, guardando estados no cache
    hidden = x
    for i, block in enumerate(model.blocks):
        if model.experts_enabled:
            hidden, _ = block(hidden)
        elif hasattr(block, 'norm_mixer'):  # SSD block
            hidden = _ssd_forward_with_cache(block, hidden, cache, i)
        elif hasattr(block, 'norm_attn'):  # Attention block
            hidden = _attn_forward_with_cache(block, hidden, cache, i)
        else:
            hidden = block(hidden)

    hidden = model.norm(hidden)
    logits = model.lm_head(hidden[:, -1:, :])  # só último token
    next_token = _sample(logits[:, -1, :], temperature, top_p)

    # ── Loop de geração: um token por vez com cache ──
    generated: list[int] = []
    finish_reason = "max_length"

    for _ in range(max_tokens):
        if next_token.item() in stop_ids:
            finish_reason = "eos"
            break

        generated.append(next_token.item())

        # Só processa o novo token
        token_emb = model.token_embedding(next_token)
        hidden = token_emb
        cache.increment()

        for i, block in enumerate(model.blocks):
            if model.experts_enabled:
                hidden, _ = block(hidden)
            elif hasattr(block, 'norm_mixer'):
                hidden = _ssd_forward_with_cache(block, hidden, cache, i)
            elif hasattr(block, 'norm_attn'):
                hidden = _attn_forward_with_cache(block, hidden, cache, i)
            else:
                hidden = block(hidden)

        hidden = model.norm(hidden)
        logits = model.lm_head(hidden[:, -1:, :])
        next_token = _sample(logits[:, -1, :], temperature, top_p)

    # ── Decodificar ──
    all_ids = input_ids + generated
    if tokenizer is not None:
        text = tokenizer.decode(all_ids, skip_special=True)
    else:
        text = " ".join(str(t) for t in all_ids)

    return text, generated, finish_reason


def _ssd_forward_with_cache(block, x, cache, layer_idx):
    """SSD forward com cache de estado oculto e conv causal."""
    residual = x
    normed = block.norm_mixer(x)
    batch, seq_len, d_model = normed.shape

    xz = block.ssm.in_proj(normed)
    x_inner, z = xz.chunk(2, dim=-1)

    kernel = block.ssm.conv1d.kernel_size[0]
    pad = kernel - 1
    xi = x_inner.transpose(1, 2)
    conv_state = cache.get_ssd_conv(layer_idx)
    if conv_state is not None:
        x_conv_in = torch.cat([conv_state, xi], dim=-1)
    else:
        x_conv_in = F.pad(xi, (pad, 0))

    x_conv = block.ssm.conv1d(x_conv_in).transpose(1, 2)
    x_conv = F.silu(x_conv)
    if pad > 0:
        cache.set_ssd_conv(layer_idx, x_conv_in[:, :, -pad:].detach())

    d_inner = block.ssm.d_inner
    dt_rank = block.ssm.dt_rank
    d_state = block.ssm.d_state

    x_flat = x_conv.reshape(batch * seq_len, d_inner)
    x_params = block.ssm.x_proj(x_flat)
    dt_raw, b_raw, c_raw = torch.split(x_params, [dt_rank, d_state, d_state], dim=-1)

    delta = F.softplus(block.ssm.dt_proj(dt_raw))
    delta = delta.clamp(min=block.ssm.dt_min, max=block.ssm.dt_max)
    delta = delta.reshape(batch, seq_len, d_inner).transpose(1, 2)
    b_tensor = b_raw.reshape(batch, seq_len, d_state).transpose(1, 2)
    c_tensor = c_raw.reshape(batch, seq_len, d_state).transpose(1, 2)
    u = x_conv.transpose(1, 2)
    a = -torch.exp(block.ssm.a_log)

    h_prev = cache.get_ssd_state(layer_idx)
    y, h_new = selective_scan_with_cache(u, delta, a, b_tensor, c_tensor, block.ssm.d_skip, h_prev)
    cache.set_ssd_state(layer_idx, h_new.detach())

    y = y.transpose(1, 2)
    y = y * F.silu(z)
    y = residual + block.dropout(block.ssm.out_proj(y))
    return y + block.ff(block.norm_ff(y))


def _attn_forward_with_cache(block, x, cache, layer_idx):
    """Attention forward com cache de K e V e RoPE posicional."""
    residual = x
    batch, seq_len, d_model = x.shape

    normed = block.norm_attn(x)
    qkv = block.qkv(normed)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(batch, seq_len, block.n_heads, block.head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, block.n_heads, block.head_dim).transpose(1, 2)
    v = v.view(batch, seq_len, block.n_heads, block.head_dim).transpose(1, 2)

    from f51_darwin.rope import build_rope_cache, apply_rope

    pos_start = cache.seq_len - seq_len
    cos, sin = build_rope_cache(
        pos_start + seq_len, block.head_dim, base=block.rope_base, device=x.device
    )
    cos = cos[pos_start : pos_start + seq_len]
    sin = sin[pos_start : pos_start + seq_len]
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    past = cache.get_attn_kv(layer_idx)
    if past is not None:
        k = torch.cat([past[0], k], dim=2)
        v = torch.cat([past[1], v], dim=2)
    cache.set_attn_kv(layer_idx, k.detach(), v.detach())

    attn = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=(seq_len > 1),
        scale=1.0 / (block.head_dim ** 0.5),
    )
    attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
    x = residual + block.dropout(block.proj(attn))

    return x + block.ff(block.norm_ff(x))


def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Amostragem de token."""
    if temperature < 1e-8:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / max(temperature, 1e-8)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        cutoff = cumulative > top_p
        cutoff[:, 1:] = cutoff[:, :-1].clone()
        cutoff[:, 0] = False
        logits = logits.clone()
        logits.scatter_(-1, sorted_indices, sorted_logits.masked_fill(cutoff, float("-inf")))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
