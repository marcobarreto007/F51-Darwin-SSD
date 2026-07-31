"""JEPA predictor minimal — integrado ao Darwin Transfer Runtime.

Prevê o próximo hidden state a partir do atual. Se confiante, pula o backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPAPredictor(nn.Module):
    """Predictor leve: hidden(t) → hidden(t+1).

    Bottleneck pequeno (32 dims) para ser barato computacionalmente.
    """

    def __init__(self, d_model: int = 768, bottleneck: int = 32):
        super().__init__()
        self.d_model = d_model
        self.encoder = nn.Sequential(
            nn.Linear(d_model, bottleneck),
            nn.GELU(),
            nn.LayerNorm(bottleneck),
        )
        self.predictor = nn.Sequential(
            nn.Linear(bottleneck, d_model * 2),
            nn.GELU(),
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(bottleneck, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (predicted_next_hidden, confidence)."""
        z = self.encoder(hidden)  # [B, bottleneck]
        pred = self.predictor(z)  # [B, d_model]
        conf = self.confidence_head(z)  # [B, 1]
        return pred, conf

    def train_step(
        self,
        hidden_current: torch.Tensor,
        hidden_next: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """One JEPA training step. Returns loss."""
        pred, _ = self.forward(hidden_current)
        # Cosine distance loss with stop-grad on target
        target = hidden_next.detach()
        loss = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.item())
