"""
F51 Expert Weight Router — Performance-based MoE routing.

Adaptado do SuperEzio (src/core/trading/expert_weights.py)
Integrado ao F51 Darwin-SSD MoE layer.

Princípio: Experts que PERFORMAM bem → peso SOBE.
          Experts que PERFORMAM mal → peso DESCE.
          Experts sem uso → DECAEM (dança das cadeiras).

Fórmula: weight = initial × (0.5 + accuracy) × exp(-elapsed/recency_window)

Integração NÃO-INVASIVA:
  - Durante treino: router normal do MoE
  - Durante inferência: pesos ajustados por performance
  - Ghost Token mede acurácia por domínio
  - SpiderSense detecta experts estagnados
"""

from __future__ import annotations

import math, time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExpertRecord:
    """Track record of a single MoE expert."""
    expert_id: int
    domain: str = "default"
    total_calls: int = 0
    correct_calls: int = 0
    total_confidence: float = 0.0
    last_called_at: float = 0.0
    weight: float = 1.0
    ghost_accuracy: float = 0.5  # Ghost Token prediction accuracy
    layer: int = 0

    @property
    def accuracy(self) -> float:
        if self.total_calls == 0:
            return 0.5  # neutral prior
        return self.correct_calls / self.total_calls

    @property
    def combined_score(self) -> float:
        """Combined score: 60% accuracy + 40% ghost token performance."""
        return 0.6 * self.accuracy + 0.4 * self.ghost_accuracy


class ExpertWeightRouter:
    """Routes MoE tokens to experts based on PERFORMANCE, not just logits.

    Key difference from standard MoE router:
    - Standard: picks experts with highest softmax logits
    - F51: multiplies logits by performance weights
    - Result: experts that PROVE themselves get more tokens
    """

    def __init__(
        self,
        num_experts: int = 8,
        min_weight: float = 0.05,
        max_weight: float = 3.0,
        decay_factor: float = 0.95,
        recency_window: float = 3600.0,  # 1 hour
        initial_weight: float = 1.0,
    ):
        self.num_experts = num_experts
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.decay_factor = decay_factor
        self.recency_window = recency_window
        self.initial_weight = initial_weight
        self._experts: dict[int, ExpertRecord] = {}

        # Initialize all experts
        for i in range(num_experts):
            self._experts[i] = ExpertRecord(expert_id=i)

    def record_call(
        self,
        expert_id: int,
        confidence: float = 0.5,
        correct: Optional[bool] = None,
        ghost_accuracy: float = 0.5,
    ) -> ExpertRecord:
        """Record an expert being called with its performance."""
        rec = self._experts.get(expert_id)
        if rec is None:
            rec = ExpertRecord(expert_id=expert_id, weight=self.initial_weight)
            self._experts[expert_id] = rec

        rec.total_calls += 1
        rec.total_confidence += confidence
        rec.last_called_at = time.time()

        if correct is not None and correct:
            rec.correct_calls += 1

        rec.ghost_accuracy = 0.7 * rec.ghost_accuracy + 0.3 * ghost_accuracy

        self._update_weight(rec)
        return rec

    def get_weight(self, expert_id: int) -> float:
        rec = self._experts.get(expert_id)
        return rec.weight if rec else self.initial_weight

    def get_all_weights(self) -> dict[int, float]:
        return {eid: rec.weight for eid, rec in self._experts.items()}

    def get_normalized_weights(self) -> dict[int, float]:
        weights = self.get_all_weights()
        total = sum(weights.values())
        if total == 0:
            return {k: 1.0 / len(weights) for k in weights}
        return {k: v / total for k, v in weights.items()}

    def apply_to_logits(
        self, router_logits: "torch.Tensor", layer_idx: int = 0
    ) -> "torch.Tensor":
        """Multiply MoE router logits by performance weights.

        Args:
            router_logits: [batch, seq, num_experts] from MoERouter
            layer_idx: which MoE layer (for per-layer tracking)

        Returns:
            adjusted_logits with same shape
        """
        import torch

        weights = torch.tensor(
            [self.get_weight(i) for i in range(self.num_experts)],
            device=router_logits.device,
            dtype=router_logits.dtype,
        )
        # Soft boost: multiply logits by weights (never reduce below 0)
        return router_logits * weights.unsqueeze(0).unsqueeze(0)

    def decay_all(self) -> None:
        """Periodic decay — experts that aren't used lose weight."""
        for rec in self._experts.values():
            rec.weight *= self.decay_factor
            rec.weight = max(self.min_weight, rec.weight)

    def promote_demote(self) -> dict:
        """DANÇA DAS CADEIRAS: promote best, demote worst.

        Returns dict with promoted/demoted expert IDs.
        """
        scored = [(eid, rec.combined_score) for eid, rec in self._experts.items()]
        scored.sort(key=lambda x: -x[1])

        top_k = max(1, self.num_experts // 4)
        bottom_k = max(1, self.num_experts // 4)

        promoted = [eid for eid, _ in scored[:top_k]]
        demoted = [eid for eid, _ in scored[-bottom_k:]]

        # Boost promoted, penalize demoted
        for eid in promoted:
            self._experts[eid].weight = min(self.max_weight, self._experts[eid].weight * 1.2)

        for eid in demoted:
            self._experts[eid].weight = max(self.min_weight, self._experts[eid].weight * 0.8)

        return {"promoted": promoted, "demoted": demoted}

    def status_report(self) -> str:
        """Human-readable expert performance report."""
        lines = ["EXPERT WEIGHT ROUTER STATUS", "=" * 50]
        for eid, rec in sorted(self._experts.items(),
                              key=lambda x: -x[1].combined_score):
            bar = "█" * int(rec.weight / self.max_weight * 10) + "░" * (10 - int(rec.weight / self.max_weight * 10))
            lines.append(
                f"  Expert {eid} [{bar}] w={rec.weight:.2f} "
                f"acc={rec.accuracy:.0%} ghost={rec.ghost_accuracy:.0%} "
                f"calls={rec.total_calls}"
            )
        return "\n".join(lines)

    def _update_weight(self, rec: ExpertRecord) -> None:
        """Update expert weight based on combined score and recency."""
        # Score multiplier: 0.5 to 1.5 range
        score_mult = 0.5 + rec.combined_score

        # Recency multiplier: decays if not used recently
        recency_mult = 1.0
        if rec.last_called_at > 0:
            elapsed = time.time() - rec.last_called_at
            recency_mult = math.exp(-elapsed / self.recency_window)

        rec.weight = max(
            self.min_weight,
            min(self.max_weight, self.initial_weight * score_mult * recency_mult),
        )


# ═══════════════════════════════════════════════════════
# TESTES
# ═══════════════════════════════════════════════════════

def test_basic():
    router = ExpertWeightRouter(num_experts=8)
    assert router.get_weight(0) == 1.0
    router.record_call(0, correct=True, ghost_accuracy=0.9)
    assert router.get_weight(0) > 1.0, "Expert with correct call should gain weight"
    print("  basic: OK")

def test_decay():
    router = ExpertWeightRouter(num_experts=8)
    router.record_call(0, correct=True)
    w_before = router.get_weight(0)
    router.decay_all()
    w_after = router.get_weight(0)
    assert w_after < w_before, "Decay should reduce weight"
    print("  decay: OK")

def test_dance():
    router = ExpertWeightRouter(num_experts=4)
    for i in range(4):
        router.record_call(i, correct=(i < 2))  # experts 0,1 correct; 2,3 wrong
    result = router.promote_demote()
    assert len(result["promoted"]) > 0
    assert len(result["demoted"]) > 0
    print(f"  dance: promoted={result['promoted']}, demoted={result['demoted']} — OK")

def test_report():
    router = ExpertWeightRouter(num_experts=4)
    for i in range(4):
        router.record_call(i, correct=(i % 2 == 0))
    report = router.status_report()
    assert "Expert 0" in report
    print("  report: OK")
