from __future__ import annotations

import torch
from torch import nn


class DynamicDepthRouter(nn.Module):
    """Minimal depth router interface for future mixture-of-depths decisions."""

    def __init__(self, d_model: int, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        scores = torch.sigmoid(self.score(hidden_states)).squeeze(-1)
        mask = scores >= self.threshold
        return {"scores": scores, "active_mask": mask}

