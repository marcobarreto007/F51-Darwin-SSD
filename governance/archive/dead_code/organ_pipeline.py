"""F51 Organ Pipeline — Connects all 7 organs into one evolution loop.

Ghost → Spider → JEPA → Curiosity → Consensus → Evolution Gate → Legacy Layers

Runs every N training steps as a meta-learning cycle.
"""

from __future__ import annotations
import torch
from dataclasses import dataclass, field
from pathlib import Path

from f51_darwin.evolution_gate import EvolutionGateMetrics, EvolutionGateThresholds, decide_promotion

@dataclass
class OrganPipelineConfig:
    """How often each organ runs and its weight."""
    ghost_mask_ratio: float = 0.15
    ghost_interval: int = 100
    spider_threshold: float = 0.3
    jepa_weight: float = 0.05
    curiosity_budget: float = 0.30
    consensus_candidates: int = 4
    evolution_interval: int = 500
    legacy_update_interval: int = 100
    legacy_save_path: str | Path = "workspace/runtime/organism/legacy"
    gate_thresholds: EvolutionGateThresholds = field(default_factory=EvolutionGateThresholds)

@dataclass
class OrganCycleResult:
    step: int
    ghost_explored: int = 0
    spider_blocked: int = 0
    jepa_quality: float = 0.0
    curiosity_reward: float = 0.0
    consensus_confidence: float = 0.0
    evolution_decision: str = "waiting"
    evolution_gates: dict[str, bool] = field(default_factory=dict)
    evolution_reasons: list[str] = field(default_factory=list)
    evolution_score: float = 0.0
    legacy_movements: int = 0
    lessons: list = field(default_factory=list)
    scars: list = field(default_factory=list)

class OrganPipeline:
    """The bridge between all organs. One ring to rule them all."""
    
    def __init__(self, model, config: OrganPipelineConfig = None):
        self.model = model
        self.config = config or OrganPipelineConfig()
        self.history: list[OrganCycleResult] = []
        
        # Lazy-load organs from F51 modules
        from f51_darwin.ghost_token import GhostTokenTrainer
        from f51_darwin.jepa import JEPAHead, jepa_loss
        from f51_darwin.spider_sense import SpiderSense
        from f51_darwin.curiosity import CuriosityDrive
        from f51_darwin.legacy_layers import LegacyLayers
        
        self.ghost = GhostTokenTrainer(mask_ratio=self.config.ghost_mask_ratio, mask_token_id=2, num_experts=12)
        self.jepa_head = JEPAHead(d_model=model.config.d_model if hasattr(model, 'config') else 2048)
        self.spider = SpiderSense(d_model=model.config.d_model if hasattr(model, 'config') else 2048)
        self.curiosity = CuriosityDrive(exploration_budget=self.config.curiosity_budget)
        self.legacy = LegacyLayers(self.config.legacy_save_path)
        
    def cycle(
        self,
        batch: torch.Tensor,
        step: int,
        loss: float,
        evolution_metrics: EvolutionGateMetrics | None = None,
    ) -> OrganCycleResult:
        """Run one organ cycle. Called every N training steps."""
        result = OrganCycleResult(step=step)
        
        # 1. GHOST: Explore 15% dark tokens → find knowledge gaps
        if step % self.config.ghost_interval == 0 and hasattr(self.model, 'token_embedding'):
            try:
                masked, mask = self.ghost.create_masked_input(batch, self.model.token_embedding)
                if mask.sum() > 0:
                    result.ghost_explored = mask.sum().item()
            except Exception as exc:
                result.scars.append(f"ghost_error:{type(exc).__name__}")
        
        # 2. SPIDER: Audit for danger patterns
        if hasattr(self.model, 'forward'):
            with torch.no_grad():
                hidden = self.model.token_embedding(batch) if hasattr(self.model, 'token_embedding') else torch.randn(1,1,2048)
                danger = self.spider(hidden).mean().item()
                if danger > self.config.spider_threshold:
                    result.spider_blocked = 1
        
        # 3. JEPA: Evaluate representation quality
        if hasattr(self.model, 'forward'):
            with torch.no_grad():
                try:
                    out = self.model(batch)
                    if out.hidden_states is not None:
                        result.jepa_quality = 1.0 - float(torch.nn.functional.cosine_similarity(
                            out.hidden_states[:, :-1], out.hidden_states[:, 1:], dim=-1).mean())
                except Exception as exc:
                    result.scars.append(f"jepa_error:{type(exc).__name__}")
        
        # 4. CURIOSITY: Reward exploration
        result.curiosity_reward = float(self.curiosity.evaluate_curiosity_reward(
            torch.randn(1,1,384), jepa_prediction_error=result.jepa_quality,
            domain="training", loss_improved=(loss < 10.0)
        ).get('curiosity_reward', 0.5))
        
        # 5. CONSENSUS: Score from all organs
        scores = [result.ghost_explored/1000, result.jepa_quality, result.curiosity_reward]
        result.consensus_confidence = sum(scores) / max(len(scores), 1)
        
        # 6. EVOLUTION GATE: Decide
        if step % self.config.evolution_interval == 0:
            if evolution_metrics is None:
                result.evolution_decision = "waiting_for_metrics"
                result.scars.append("missing_evolution_metrics")
            else:
                decision = decide_promotion(evolution_metrics, self.config.gate_thresholds)
                result.evolution_decision = decision.action
                result.evolution_gates = dict(decision.gates)
                result.evolution_reasons = list(decision.reasons)
                result.evolution_score = decision.score
                if decision.passed:
                    result.lessons.append(
                        f"step_{step}_verified_gate_score_{decision.score:.3f}"
                    )
                else:
                    result.scars.extend(decision.reasons)
        
        # 7. LEGACY: Temperature dance
        if step % self.config.legacy_update_interval == 0:
            temp = result.consensus_confidence * 12.0
            tier = self.legacy.compute_tier(temp, f"expert_step_{step}")
            result.legacy_movements = 1
        
        self.history.append(result)
        return result
    
    def report(self) -> str:
        if not self.history: return "No cycles yet"
        last = self.history[-1]
        return (f"Organs[{last.step}]: ghost={last.ghost_explored} spider={last.spider_blocked} "
                f"jepa={last.jepa_quality:.3f} curiosity={last.curiosity_reward:.3f} "
                f"consensus={last.consensus_confidence:.3f} evolve={last.evolution_decision}")
