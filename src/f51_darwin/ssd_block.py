from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from f51_darwin.ssm_core import SelectiveSSM


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        *,
        fp32: bool = False,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.fp32 = fp32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fp32:
            dtype = x.dtype
            normalized = x.float() * torch.rsqrt(
                x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
            )
            return (self.weight.float() * normalized).to(dtype)
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        hidden = d_model * mlp_ratio
        self.w12 = nn.Linear(d_model, hidden * 2)
        self.out = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.w12(x).chunk(2, dim=-1)
        return self.dropout(self.out(F.silu(gate) * value))


class SSDBlock(nn.Module):
    """SSD block backed by a selective state-space mixer."""

    def __init__(
        self,
        d_model: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        *,
        ssm_expand: int = 2,
        ssm_state: int = 16,
    ) -> None:
        super().__init__()
        self.norm_mixer = RMSNorm(d_model)
        self.ssm = SelectiveSSM(d_model, expand=ssm_expand, d_state=ssm_state)
        self.dropout = nn.Dropout(dropout)
        self.norm_ff = RMSNorm(d_model)
        self.ff = SwiGLUFeedForward(d_model, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.ssm(self.norm_mixer(x)))
        return x + self.ff(self.norm_ff(x))
