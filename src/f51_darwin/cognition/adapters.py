from __future__ import annotations

import math

import torch
from torch import nn

from .contracts import COGNITIVE_ORGAN_WIDTH, ResidualCondition


class ZeroGatedCognitiveAdapter(nn.Module):
    def __init__(self, d_model: int, max_scale: float) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be positive")
        if not math.isfinite(max_scale) or not 0.0 <= max_scale <= 1.0:
            raise ValueError("max_scale must be finite and in [0, 1]")
        self.d_model = int(d_model)
        self.max_scale = float(max_scale)
        self.input_adapter = nn.Linear(
            self.d_model,
            COGNITIVE_ORGAN_WIDTH,
            bias=False,
        )
        self.output_adapter = nn.Linear(
            COGNITIVE_ORGAN_WIDTH,
            self.d_model,
            bias=False,
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError("hidden must have shape [batch, sequence, d_model]")
        if not bool(torch.isfinite(hidden).all()):
            raise ValueError("hidden must be finite")
        return self.input_adapter(hidden)

    def effective_scale(self) -> torch.Tensor:
        return self.max_scale * torch.tanh(self.gate)

    def apply_condition(
        self,
        hidden: torch.Tensor,
        condition: ResidualCondition,
    ) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError("hidden must have shape [batch, sequence, d_model]")
        if condition.batch_size != hidden.shape[0]:
            raise ValueError("condition batch does not match hidden batch")
        positions = condition.positions.to(device=hidden.device)
        if bool((positions < 0).any()) or bool((positions >= hidden.shape[1]).any()):
            raise ValueError("condition position is out of range")
        for row in positions.detach().cpu().tolist():
            if len(set(row)) != len(row):
                raise ValueError("condition positions must be unique per sample")

        values = condition.values.to(device=hidden.device, dtype=hidden.dtype)
        projected = self.output_adapter(values)
        gather_index = positions.unsqueeze(-1).expand(-1, -1, self.d_model)
        reference = hidden.gather(dim=1, index=gather_index)
        projected_norm = projected.float().norm(dim=-1, keepdim=True)
        reference_norm = reference.float().norm(dim=-1, keepdim=True)
        factor = (
            reference_norm / projected_norm.clamp_min(1e-6)
        ).clamp(max=1.0)
        bounded = projected * factor.to(dtype=projected.dtype)
        delta = self.effective_scale().to(
            device=hidden.device,
            dtype=hidden.dtype,
        ) * bounded
        if bool(self.gate.detach().eq(0).item()):
            return hidden
        result = hidden.clone()
        result.scatter_add_(dim=1, index=gather_index, src=delta)
        return result
