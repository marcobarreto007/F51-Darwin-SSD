"""
F51 Darwin-X Inference Engine — torch.compile + KV cache + CUDA graphs.

Drop-in accelerator for autoregressive generation on the dual‑GPU
Darwin-X pipeline.  Uses:
  • torch.compile  (PT 2.8, max-autotune on Blackwell sm_120)
  • KV cache       (for the GQA attention layers)
  • SSD state cache (conv1d + scan state per SSD layer -- decode is O(1)/token
    for real since 2026-07-26; before that fix, every decode step recomputed
    each SSD layer from scratch with zero initial state, i.e. amnesia)
  • pre‑allocated buffers to avoid per‑step allocations

``DarwinInferenceEngine`` wraps an already‑bootstrapped model and
provides ``generate(prompt) -> dict`` with the same contract as
``live_generate`` so the UI can use it transparently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from f51_darwin.rope import apply_rope, build_rope_cache


# ═══════════════════════════════════════════════════════════════════════════
# KV Cache  (GQA‑aware, one slot per attention layer)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _KVCacheSlot:
    """Pre‑allocated K + V for one GQA attention layer."""
    k: torch.Tensor   # [1, n_kv_heads, max_len, head_dim]
    v: torch.Tensor   # [1, n_kv_heads, max_len, head_dim]
    filled: int = 0   # how many positions have been written

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> int:
        """Write new K/V at position ``filled`` and return that position."""
        pos = self.filled
        self.k[:, :, pos : pos + 1, :] = k_new
        self.v[:, :, pos : pos + 1, :] = v_new
        self.filled += 1
        return pos

    def window(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return K/V slices for all filled positions."""
        return self.k[:, :, : self.filled, :], self.v[:, :, : self.filled, :]


class KVCache:
    """Manages one slot per attention layer index."""

    def __init__(
        self,
        attention_indices: tuple[int, ...],
        n_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.slots: dict[int, _KVCacheSlot] = {}
        for idx in attention_indices:
            self.slots[idx] = _KVCacheSlot(
                k=torch.zeros(1, n_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype),
                v=torch.zeros(1, n_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype),
            )

    def reset(self) -> None:
        for slot in self.slots.values():
            slot.filled = 0


# ═══════════════════════════════════════════════════════════════════════════
# SSD state cache  (conv1d context + scan state, one slot per SSD layer)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _SSDStateSlot:
    """Recurrent state for one SSD layer. None until the first forward."""
    conv_state: torch.Tensor | None = None   # [1, d_inner, conv_kernel-1]
    ssm_state: torch.Tensor | None = None     # [1, d_inner, d_state]


class SSDStateCache:
    """Manages one recurrent-state slot per SSD (non-attention) layer index.

    Unlike KVCache, this does not need pre-allocated max_seq_len buffers --
    SSD recurrent state is O(1) in sequence length, exactly the property
    Mamba is supposed to have (see module docstring).
    """

    def __init__(self, ssd_indices: tuple[int, ...]) -> None:
        self.slots: dict[int, _SSDStateSlot] = {idx: _SSDStateSlot() for idx in ssd_indices}

    def reset(self) -> None:
        for slot in self.slots.values():
            slot.conv_state = None
            slot.ssm_state = None


# ═══════════════════════════════════════════════════════════════════════════
# Optimised inference model  (thin wrapper)
# ═══════════════════════════════════════════════════════════════════════════

class DarwinInferenceEngine:
    """Compiled inference with KV cache for Darwin‑X.

    Usage
    -----
        engine = DarwinInferenceEngine(organism.model, organism.tokenizer)
        result = engine.generate("Quem é o Darwin?")
        print(result["text"])
    """

    def __init__(self, model, tokenizer, *, max_seq_len: int = 4096) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = model.config
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype   # bf16
        self._dual_gpu = getattr(model, "_dual_gpu", False)
        self._gpu0 = getattr(model, "_gpu0", None) or self.device
        self._gpu1 = getattr(model, "_gpu1", None) or self.device
        self._split_layer = getattr(model, "_split_layer", 0) or 0

        # Identify attention layers
        self._attn_indices: tuple[int, ...] = self.config.attention_layer_indices
        n_kv = self.config.n_kv_heads
        hd = self.config.head_dim

        # One KV cache per GPU (attention layers may live on different GPUs)
        gpu0_indices = tuple(i for i in self._attn_indices if i < self._split_layer)
        gpu1_indices = tuple(i for i in self._attn_indices if i >= self._split_layer)

        self._cache_gpu0 = KVCache(gpu0_indices, n_kv, hd, max_seq_len, self._gpu0, self.dtype)
        self._cache_gpu1 = KVCache(gpu1_indices, n_kv, hd, max_seq_len, self._gpu1, self.dtype)

        # SSD recurrent-state cache — every layer that isn't attention.
        attn_set = set(self._attn_indices)
        ssd_indices = tuple(i for i in range(len(self.model.blocks)) if i not in attn_set)
        ssd_gpu0_indices = tuple(i for i in ssd_indices if i < self._split_layer)
        ssd_gpu1_indices = tuple(i for i in ssd_indices if i >= self._split_layer)
        self._ssd_cache_gpu0 = SSDStateCache(ssd_gpu0_indices)
        self._ssd_cache_gpu1 = SSDStateCache(ssd_gpu1_indices)

        # torch.compile — cascade: inductor → cudagraphs → raw forward
        self._compiled = self._forward_with_cache   # default: raw
        compiled = False
        for backend, mode in [
            ("cudagraphs", None),     # CUDA graphs — no triton needed
            ("eager", None),          # plain trace
        ]:
            try:
                kwargs: dict[str, Any] = {"fullgraph": False, "dynamic": False}
                if backend:
                    kwargs["backend"] = backend
                if mode:
                    kwargs["mode"] = mode
                candidate = torch.compile(self._forward_with_cache, **kwargs)
                dummy = torch.randint(0, 100, (1, 4), device=self.device, dtype=torch.long)
                candidate(dummy, (self._cache_gpu0, self._cache_gpu1), 0)
                self._compiled = candidate
                print(f"  🚀 InferenceEngine: torch.compile backend={backend or 'inductor'}")
                compiled = True
                break
            except Exception:
                continue

        if not compiled:
            print("  ⚠️ torch.compile indisponivel — usando forward puro (KV cache funciona)")

    # ── compiled forward with cache ─────────────────────────────────────

    def _forward_with_cache(
        self,
        input_ids: torch.Tensor,
        caches: tuple[KVCache, KVCache],
        position: int,
    ) -> torch.Tensor:
        """Single‑step forward that reads/writes KV cache for attention layers.

        Returns hidden states of the LAST token only  [1, 1, d_model].
        """
        cache_gpu0, cache_gpu1 = caches
        x = self.model.token_embedding(input_ids)   # [1, seq, d_model]

        for layer_idx, block in enumerate(self.model.blocks):
            # ── attention / SSD ──
            if block.is_attention_layer:
                normed = block.norm1(x)
                # --- GQA with cache ---
                bsz, seq, d = normed.shape
                q = block.attention.q_proj(normed).view(bsz, seq, self.config.n_heads, self.config.head_dim).transpose(1, 2)
                k = block.attention.k_proj(normed).view(bsz, seq, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)
                v = block.attention.v_proj(normed).view(bsz, seq, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)

                rope_base = block.attention.rope_base_infer
                cos, sin = build_rope_cache(position + seq, self.config.head_dim, base=rope_base, device=normed.device)
                q = apply_rope(q, cos[position:position + seq], sin[position:position + seq])
                k = apply_rope(k, cos[position:position + seq], sin[position:position + seq])

                # Store KV in cache — all new positions at once
                cache = cache_gpu0 if layer_idx < self._split_layer else cache_gpu1
                slot = cache.slots[layer_idx]
                new_len = seq
                slot.k[:, :, slot.filled : slot.filled + new_len, :] = k
                slot.v[:, :, slot.filled : slot.filled + new_len, :] = v
                slot.filled += new_len

                # Retrieve full cached KV (enable_gqa=True avoids materializing
                # the repeat_interleave'd K/V -- numerically identical, less
                # memory bandwidth; see darwin_x_core/layers.py for the same
                # fix on the training-path attention).
                k_full, v_full = slot.window()
                q_full = q[:, :, -1:, :] if seq == 1 else q   # only new Q for decode

                attn = F.scaled_dot_product_attention(
                    q_full, k_full, v_full,
                    is_causal=False if seq == 1 else True,
                    enable_gqa=True,
                )
                attn = attn.transpose(1, 2).contiguous().view(attn.shape[0], -1, self.config.d_model)
                mixed = block.attention.o_proj(attn)
            else:
                ssd_cache = self._ssd_cache_gpu0 if layer_idx < self._split_layer else self._ssd_cache_gpu1
                ssd_slot = ssd_cache.slots.get(layer_idx)
                normed = block.norm1(x)
                if ssd_slot is not None:
                    mixed, new_conv_state, new_ssm_state = block.ssd(
                        normed,
                        conv_state=ssd_slot.conv_state,
                        ssm_state=ssd_slot.ssm_state,
                        return_state=True,
                    )
                    ssd_slot.conv_state = new_conv_state
                    ssd_slot.ssm_state = new_ssm_state
                else:
                    mixed = block.ssd(normed)

            x = x + block.residual_scale * block.dropout(mixed)

            # ── MoE ──
            moe_out, _aux = block.moe(block.norm2(x))
            x = x + block.residual_scale * block.dropout(moe_out)

            # ── dual‑GPU transfer ──
            if self._dual_gpu and layer_idx == self._split_layer - 1:
                x = x.to(self._gpu1)
            elif self._dual_gpu and layer_idx == len(self.model.blocks) - 1:
                x = x.to(self._gpu0)

        hidden = self.model.norm(x)
        return hidden[:, -1:, :]    # only last position

    # ── public API ──────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> dict[str, Any]:
        """Fast autoregressive generation with KV cache + torch.compile.

        Returns dict with keys: text, tokens, elapsed_s, tokens_per_sec.
        """
        stop_id = getattr(self.tokenizer, "eos_id", None)
        ids = self.tokenizer.encode(prompt)
        context = torch.tensor([ids[-self.config.context_length:]], dtype=torch.long, device=self.device)

        # ── Reset KV + SSD state caches ──
        self._cache_gpu0.reset()
        self._cache_gpu1.reset()
        self._ssd_cache_gpu0.reset()
        self._ssd_cache_gpu1.reset()

        # ── Prefill: run full context through compiled forward ──
        generated: list[int] = []
        finish_reason = "max_length"
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()

        prompt_len = context.shape[1]
        for step in range(max_tokens):
            if step == 0:
                # Prefill: all prompt tokens at once, RoPE positions 0..prompt_len-1
                hidden = self._compiled(context, (self._cache_gpu0, self._cache_gpu1), 0)
            else:
                # Decode: single token at absolute position prompt_len+step-1
                position = prompt_len + step - 1
                new_token = torch.tensor([[generated[-1]]], dtype=torch.long, device=self.device)
                hidden = self._compiled(new_token, (self._cache_gpu0, self._cache_gpu1), position)

            logits = self.model.lm_head(hidden[:, -1, :]).float() / max(temperature, 1e-8)

            # ── top‑p sampling ──
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                cutoff = cumulative > top_p
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = False
                logits_flat = logits.clone()
                logits_flat.scatter_(-1, sorted_indices, sorted_logits.masked_fill(cutoff, float("-inf")))
            else:
                logits_flat = logits

            probs = F.softmax(logits_flat, dim=-1)
            next_token = int(torch.multinomial(probs, num_samples=1).item())

            if stop_id is not None and next_token == stop_id:
                finish_reason = "eos"
                break

            generated.append(next_token)

        t1.record()
        torch.cuda.synchronize()
        elapsed_ms = t0.elapsed_time(t1)
        elapsed_s = elapsed_ms / 1000.0

        # ── Decode ──
        try:
            text = self.tokenizer.decode(generated, skip_special=True)
        except TypeError:
            text = self.tokenizer.decode(generated)

        total_tokens = len(generated)
        return {
            "text": text,
            "tokens": generated,
            "elapsed_s": round(elapsed_s, 3),
            "tokens_per_sec": round(total_tokens / max(elapsed_s, 0.001), 1),
            "finish_reason": finish_reason,
        }
