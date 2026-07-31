"""Darwin Transfer Runtime V2 — TTM + JEPA + Backbone.

Three-path cascade:
  TTM memory (cheapest) → JEPA predictor (cheap) → Backbone SSD (expensive)

Routes each prediction through the cheapest path with sufficient confidence.
"""

from __future__ import annotations

import torch

from .experience import ExperienceMemory
from .jepa_predictor import JEPAPredictor
from .runtime import Prediction


class DarwinTransferRuntimeV2:
    def __init__(
        self,
        backbone,
        embedding,
        memory: ExperienceMemory,
        jepa: JEPAPredictor | None = None,
        *,
        jepa_confidence_threshold: float = 0.8,
        device: str = "cuda:0",
    ):
        self.backbone = backbone
        self.embedding = embedding
        self.memory = memory
        self.jepa = jepa
        self.jepa_threshold = jepa_confidence_threshold
        self.device = device

        # Counters
        self.backbone_calls = 0
        self.experience_hits = 0
        self.jepa_hits = 0

        # JEPA training state
        self._last_hidden: torch.Tensor | None = None
        self._jepa_optimizer: torch.optim.Optimizer | None = None
        if jepa is not None:
            self._jepa_optimizer = torch.optim.Adam(jepa.parameters(), lr=1e-3)

    def signature(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.embedding(input_ids).mean(dim=(0, 1))

    def _backbone_forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run backbone. Returns (logits, last_hidden_state)."""
        output = self.backbone(input_ids=input_ids, output_hidden_states=True)
        logits = output.logits[0, -1].detach()
        # DistilGPT-2 returns hidden_states as tuple, last is final layer
        if output.hidden_states:
            hidden = output.hidden_states[-1][0, -1].detach()  # [d_model]
        else:
            hidden = self.embedding(input_ids).mean(dim=(0, 1))
        self.backbone_calls += 1
        return logits, hidden

    def predict_next(self, input_ids: torch.Tensor, *, learn: bool) -> Prediction:
        signature = self.signature(input_ids)

        # ── Path 1: TTM memory (cheapest) ──
        recalled = self.memory.retrieve(signature)
        if recalled is not None:
            logits, confidence = recalled
            self.experience_hits += 1
            return Prediction(logits.to(input_ids.device), "experience", confidence)

        # ── Path 2: JEPA predictor (cheap) ──
        if self.jepa is not None and self._last_hidden is not None:
            jepa_pred, jepa_conf = self.jepa(self._last_hidden.unsqueeze(0))
            jepa_conf_val = float(jepa_conf.squeeze())
            if jepa_conf_val >= self.jepa_threshold:
                self.jepa_hits += 1
                # We need logits from the predicted hidden — use backbone's lm_head
                # For simplicity: run just the lm_head on the predicted hidden
                with torch.no_grad():
                    logits = self.backbone.model.lm_head(jepa_pred)
                logits = logits[0].detach()
                self._last_hidden = jepa_pred.squeeze(0)  # update for next step
                return Prediction(logits, "jepa", jepa_conf_val)

        # ── Path 3: Backbone SSD (expensive) ──
        logits, hidden = self._backbone_forward(input_ids)

        # Train JEPA on backbone output
        if self.jepa is not None and self._last_hidden is not None and learn:
            self.jepa.train_step(
                self._last_hidden.unsqueeze(0),
                hidden.unsqueeze(0),
                self._jepa_optimizer,
            )

        # Write to TTM memory
        if learn:
            self.memory.write(signature, logits)

        # Update state
        self._last_hidden = hidden

        return Prediction(logits, "backbone", 0.0)

    def observe(
        self,
        input_ids: torch.Tensor,
        *,
        target_id: int,
        vocab_size: int,
    ) -> None:
        self.memory.write_target(
            self.signature(input_ids),
            target_id=target_id,
            vocab_size=vocab_size,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        learn: bool = True,
    ) -> tuple[torch.Tensor, list[str]]:
        generated = input_ids
        paths: list[str] = []
        for _ in range(max_new_tokens):
            prediction = self.predict_next(generated, learn=learn)
            next_id = prediction.logits.argmax().view(1, 1).to(generated.device)
            generated = torch.cat((generated, next_id), dim=1)
            paths.append(prediction.path)
        return generated, paths
