"""F51 CUDA TURBO — Geração autoregressiva com KV cache (SSD h + conv + attn KV)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from f51_darwin.kv_cache import selective_scan_with_cache
from f51_darwin.rope import apply_rope, build_rope_cache


class KVCache:
    """Per-layer caches for incremental decode."""

    def __init__(self) -> None:
        self.kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.ssd_h: dict[int, torch.Tensor] = {}
        self.ssd_conv: dict[int, torch.Tensor] = {}
        self.seq_len: int = 0


@torch.no_grad()
def turbo_generate(
    model,
    prompt_ids,
    tokenizer,
    *,
    max_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    eos_id: int | None = None,
):
    device = next(model.parameters()).device
    config = model.config
    stop_ids = {eos_id} if eos_id else {tokenizer.eos_id} if tokenizer else set()

    ids = list(prompt_ids)[-config.context_length :]
    cache = KVCache()

    x = model.token_embedding(torch.tensor([ids], dtype=torch.long, device=device))
    cache.seq_len = x.shape[1]
    x = _forward_blocks(model, x, cache, store=True)
    logits = model.lm_head(model.norm(x[:, -1:, :]))
    next_id = _sample(logits[:, -1, :], temperature, top_p)

    generated: list[int] = []
    finish_reason = "max_length"

    for _ in range(max_tokens):
        token = int(next_id.item())
        if token in stop_ids:
            finish_reason = "eos"
            break

        generated.append(token)
        cache.seq_len += 1

        x = model.token_embedding(next_id)
        x = _forward_blocks(model, x, cache, store=True)
        logits = model.lm_head(model.norm(x[:, -1:, :]))
        next_id = _sample(logits[:, -1, :], temperature, top_p)

    all_ids = ids + generated
    text = tokenizer.decode(all_ids, skip_special=True) if tokenizer else " ".join(map(str, all_ids))
    return text, generated, finish_reason


def _forward_blocks(model, x, cache: KVCache, *, store: bool) -> torch.Tensor:
    for i, block in enumerate(model.blocks):
        if model.experts_enabled:
            x, _ = block(x)
        elif hasattr(block, "norm_mixer"):
            x = _ssd(block, x, cache, i, store)
        elif hasattr(block, "norm_attn"):
            x = _attn(block, x, cache, i, store)
        else:
            x = block(x)
    return x


def _ssd(block, x, cache: KVCache, idx: int, store: bool) -> torch.Tensor:
    residual = x
    ssm = block.ssm
    normed = block.norm_mixer(x)
    batch, seq_len, _ = normed.shape

    xz = ssm.in_proj(normed)
    x_inner, z = xz.chunk(2, dim=-1)

    kernel = ssm.conv1d.kernel_size[0]
    pad = kernel - 1
    xi = x_inner.transpose(1, 2)

    conv_state = cache.ssd_conv.get(idx)
    if conv_state is not None:
        x_conv_in = torch.cat([conv_state, xi], dim=-1)
    else:
        x_conv_in = F.pad(xi, (pad, 0))

    x_conv = ssm.conv1d(x_conv_in).transpose(1, 2)
    x_conv = F.silu(x_conv)

    if store and pad > 0:
        cache.ssd_conv[idx] = x_conv_in[:, :, -pad:].detach()

    x_flat = x_conv.reshape(batch * seq_len, ssm.d_inner)
    x_params = ssm.x_proj(x_flat)
    dt_raw, b_raw, c_raw = torch.split(
        x_params, [ssm.dt_rank, ssm.d_state, ssm.d_state], dim=-1
    )

    delta = F.softplus(ssm.dt_proj(dt_raw)).reshape(batch, seq_len, ssm.d_inner).transpose(1, 2)
    b_tensor = b_raw.reshape(batch, seq_len, ssm.d_state).transpose(1, 2)
    c_tensor = c_raw.reshape(batch, seq_len, ssm.d_state).transpose(1, 2)
    u = x_conv.transpose(1, 2)
    a = -torch.exp(ssm.a_log)

    h_prev = cache.ssd_h.get(idx)
    y, h_new = selective_scan_with_cache(
        u, delta, a, b_tensor, c_tensor, ssm.d_skip, h_prev
    )
    if store:
        cache.ssd_h[idx] = h_new.detach()

    y = y.transpose(1, 2)
    y = y * F.silu(z)
    y = residual + block.dropout(ssm.out_proj(y))
    return y + block.ff(block.norm_ff(y))


def _attn(block, x, cache: KVCache, idx: int, store: bool) -> torch.Tensor:
    residual = x
    batch, seq_len, d_model = x.shape
    n_heads = block.n_heads
    head_dim = block.head_dim

    normed = block.norm_attn(x)
    qkv = block.qkv(normed)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
    v = v.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)

    pos_start = cache.seq_len - seq_len
    cos, sin = build_rope_cache(pos_start + seq_len, head_dim, base=block.rope_base, device=x.device)
    cos = cos[pos_start : pos_start + seq_len]
    sin = sin[pos_start : pos_start + seq_len]
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    past = cache.kv.get(idx)
    if past is not None:
        k = torch.cat([past[0], k], dim=2)
        v = torch.cat([past[1], v], dim=2)
    if store:
        cache.kv[idx] = (k.detach(), v.detach())

    attn = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=(seq_len > 1),
    )
    attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
    x = residual + block.dropout(block.proj(attn))
    return x + block.ff(block.norm_ff(x))


def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
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

    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
