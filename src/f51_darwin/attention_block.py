from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from f51_darwin.rope import RoPECache, apply_rope
from f51_darwin.ssd_block import RMSNorm, SwiGLUFeedForward


class SparseCausalAttentionBlock(nn.Module):
    """Causal self-attention with RoPE.

    Sparsity is architectural via the 3:1 SSD/attention schedule. Future
    versions can add block-sparse masks without changing this interface.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        *,
        rope_base: float = 10000.0,
        qkv_bias: bool = True,  # P1.1: False economiza params e destrava FA varlen
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope_base = rope_base
        self.norm_attn = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = dropout
        self.norm_ff = RMSNorm(d_model)
        self.ff = SwiGLUFeedForward(d_model, mlp_ratio, dropout)
        # P1.2: cache das tabelas cos/sin do RoPE.
        self._rope_cache = RoPECache()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        qkv = self.qkv(self.norm_attn(x))
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        cos, sin = self._rope_cache.get(
            seq_len, self.head_dim, base=self.rope_base, device=x.device
        )
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        x = x + self.dropout(self.proj(attn))
        return x + self.ff(self.norm_ff(x))
