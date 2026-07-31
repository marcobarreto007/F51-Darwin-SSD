"""
Autonomous Drives — O coração que respira do Darwin.

Portado do MUE-X AutonomousSignalGenerator e adaptado para o organismo neural.
Inspiração: MUE-X 7 drives → Darwin 5 drives + Emotional Modulation + Organ Audit.

Drives:
  1. Stagnation Detection  — pressão evolutiva adaptativa
  2. Emotional Modulation  — dopamina/frustração/confiança → estratégia
  3. Organ Health Audit    — diagnostica órgãos quebrados
  4. Curiosity Drive       — busca diversidade quando estagnado
  5. Ambition Drive        — propõe escala quando pronto
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


# ═══════════════════════════════════════════════════════════════
# Emotional State (PAD-inspired, simplified for neural training)
# ═══════════════════════════════════════════════════════════════

@dataclass
class EmotionalVector:
    """Pleasure-Arousal-Dominance + training-specific dimensions."""
    pleasure: float = 0.5
    arousal: float = 0.5
    dominance: float = 0.5
    confidence: float = 0.5
    frustration: float = 0.1
    curiosity: float = 0.7
    satisfaction: float = 0.3

    _dopamine_ema: float = 0.5
    _dopamine_samples: int = 0

    def feed_dopamine(self, dopamine_raw: float) -> None:
        self._dopamine_samples += 1
        alpha = 0.1
        self._dopamine_ema = (
            self._dopamine_ema * (1 - alpha) + dopamine_raw * alpha
        )

    def feed_loss_delta(self, delta: float) -> None:
        """delta < 0 = improvement, delta > 0 = degradation"""
        if delta < -0.01:
            self.pleasure = min(1.0, self.pleasure + 0.08)
            self.confidence = min(1.0, self.confidence + 0.05)
            self.frustration = max(0.0, self.frustration - 0.1)
            self.satisfaction = min(1.0, self.satisfaction + 0.1)
        elif delta < 0:
            self.pleasure = min(1.0, self.pleasure + 0.03)
            self.satisfaction = min(1.0, self.satisfaction + 0.04)
        elif delta > 0.02:
            self.pleasure = max(0.0, self.pleasure - 0.05)
            self.frustration = min(1.0, self.frustration + 0.08)
            self.confidence = max(0.0, self.confidence - 0.03)
            self.curiosity = min(1.0, self.curiosity + 0.04)
        else:
            self.frustration = min(1.0, self.frustration + 0.02)

    def tick(self, elapsed_seconds: float) -> None:
        decay = 1.0 - math.exp(-0.001 * elapsed_seconds * 10)
        homeo = {
            "pleasure": 0.5, "arousal": 0.5, "dominance": 0.5,
            "confidence": 0.5, "frustration": 0.05, "curiosity": 0.6,
            "satisfaction": 0.3,
        }
        for attr, target in homeo.items():
            current = getattr(self, attr)
            setattr(self, attr, current + (target - current) * decay)

    @property
    def risk_tolerance(self) -> float:
        base = 0.3
        base += self.confidence * 0.3
        base += self.pleasure * 0.15
        base -= self.frustration * 0.4
        return max(0.0, min(1.0, base))

    @property
    def exploration_drive(self) -> float:
        return max(0.0, min(1.0,
            self.curiosity * 0.6 + self.arousal * 0.2
        ))

    @property
    def mood_label(self) -> str:
        if self.pleasure > 0.7 and self.arousal > 0.6:
            return "excited"
        if self.pleasure > 0.7:
            return "content"
        if self.frustration > 0.6:
            return "frustrated"
        if self.pleasure < 0.3:
            return "discouraged"
        if self.confidence > 0.8:
            return "confident"
        if self.curiosity > 0.8:
            return "inquisitive"
        return "neutral"


# ═══════════════════════════════════════════════════════════════
# Training Strategy
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrainingStrategy:
    name: str
    ghost_enabled: bool = False
    curiosity_multiplier: float = 1.0
    jepa_weight_mod: float = 1.0
    mtp_weight_mod: float = 1.0
    replay_ratio_mod: float = 1.0
    dopamine_reward_mod: float = 1.0
    exploration_pressure: float = 1.0
    causal_mode_override: str | None = None
    accum_steps_override: int | None = None
    reason: str = ""


STRATEGIES: dict[str, TrainingStrategy] = {
    "conservative": TrainingStrategy(
        name="conservative",
        ghost_enabled=False,
        curiosity_multiplier=0.5,
        jepa_weight_mod=0.5,
        mtp_weight_mod=0.7,
        replay_ratio_mod=2.0,
        dopamine_reward_mod=0.5,
        exploration_pressure=0.5,
        reason="Frustração alta — reduzindo risco",
    ),
    "balanced": TrainingStrategy(
        name="balanced",
        ghost_enabled=False,
        curiosity_multiplier=1.0,
        jepa_weight_mod=1.0,
        mtp_weight_mod=1.0,
        replay_ratio_mod=1.0,
        dopamine_reward_mod=1.0,
        exploration_pressure=1.0,
        reason="Estratégia padrão balanceada",
    ),
    "exploratory": TrainingStrategy(
        name="exploratory",
        ghost_enabled=False,
        curiosity_multiplier=3.0,
        jepa_weight_mod=1.5,
        mtp_weight_mod=1.0,
        replay_ratio_mod=1.5,
        dopamine_reward_mod=1.3,
        exploration_pressure=2.0,
        reason="Curiosidade alta — explorando variações",
    ),
    "aggressive": TrainingStrategy(
        name="aggressive",
        ghost_enabled=True,
        curiosity_multiplier=1.5,
        jepa_weight_mod=2.0,
        mtp_weight_mod=1.3,
        replay_ratio_mod=0.7,
        dopamine_reward_mod=1.5,
        exploration_pressure=1.5,
        reason="Confiança alta — buscando breakthroughs",
    ),
    "repair": TrainingStrategy(
        name="repair",
        ghost_enabled=False,
        curiosity_multiplier=0.2,
        jepa_weight_mod=0.3,
        mtp_weight_mod=0.5,
        replay_ratio_mod=3.0,
        dopamine_reward_mod=0.3,
        exploration_pressure=0.2,
        accum_steps_override=4,
        reason="Loss degradando — modo reparo, gradiente conservador",
    ),
}


# ═══════════════════════════════════════════════════════════════
# Drive 1: Stagnation Detection
# ═══════════════════════════════════════════════════════════════

@dataclass
class StagnationDetector:
    cycles_without_improvement: int = 0
    best_loss: float = float("inf")
    exploration_pressure: float = 1.0
    _loss_history: list[float] = field(default_factory=list)
    _last_reset_cycle: int = 0

    def feed(self, cycle: int, loss_end: float) -> dict:
        self._loss_history.append(loss_end)
        if len(self._loss_history) > 20:
            self._loss_history = self._loss_history[-20:]

        improved = loss_end < self.best_loss * 0.97
        if improved:
            self.best_loss = min(self.best_loss, loss_end)
            self.cycles_without_improvement = 0
            self.exploration_pressure = max(1.0, self.exploration_pressure * 0.8)
            return {"action": "improving", "pressure": self.exploration_pressure}

        self.cycles_without_improvement += 1
        self.exploration_pressure = min(5.0, self.exploration_pressure * 1.3)

        if self.cycles_without_improvement >= 10:
            old_pressure = self.exploration_pressure
            self.cycles_without_improvement = 0
            self.exploration_pressure = 3.0
            self._last_reset_cycle = cycle
            return {
                "action": "critical_reset",
                "cycles_stagnant": 10,
                "pressure": 3.0,
                "previous_pressure": old_pressure,
            }

        if self.cycles_without_improvement >= 5:
            return {
                "action": "high_pressure",
                "cycles_stagnant": self.cycles_without_improvement,
                "pressure": self.exploration_pressure,
            }

        if self.cycles_without_improvement >= 3:
            return {
                "action": "warning",
                "cycles_stagnant": self.cycles_without_improvement,
                "pressure": self.exploration_pressure,
            }

        return {"action": "ok", "pressure": self.exploration_pressure}

    @property
    def stats(self) -> dict:
        return {
            "cycles_without_improvement": self.cycles_without_improvement,
            "exploration_pressure": round(self.exploration_pressure, 2),
            "best_loss": round(self.best_loss, 4),
            "recent_trend": (
                "improving"
                if len(self._loss_history) >= 3
                and self._loss_history[-1] < self._loss_history[-3]
                else "flat_or_degrading"
            ),
        }


# ═══════════════════════════════════════════════════════════════
# Drive 2: Emotional Strategy Modulator
# ═══════════════════════════════════════════════════════════════

class EmotionalStrategyModulator:
    def __init__(self):
        self.emotions = EmotionalVector()
        self._last_strategy: str = "balanced"
        self._prev_loss: float = float("inf")

    def feed_step(self, dopamine_raw: float, loss_val: float) -> None:
        self.emotions.feed_dopamine(dopamine_raw)
        delta = loss_val - self._prev_loss
        self._prev_loss = loss_val
        self.emotions.feed_loss_delta(delta)

    def tick(self, elapsed_seconds: float) -> None:
        self.emotions.tick(elapsed_seconds)

    def select_strategy(
        self,
        stagnation: StagnationDetector,
        organ_health: dict[str, str],
    ) -> TrainingStrategy:
        e = self.emotions
        pressure = stagnation.exploration_pressure

        if organ_health.get("ghost") == "critical":
            strat = STRATEGIES["conservative"]
            strat.reason = "Ghost quebrado — modo conservador forçado"
            self._last_strategy = strat.name
            return strat

        if stagnation.cycles_without_improvement >= 8:
            strat = STRATEGIES["exploratory"]
            strat.exploration_pressure = pressure
            strat.curiosity_multiplier *= pressure
            strat.reason = (
                f"Estagnação crítica ({stagnation.cycles_without_improvement} ciclos)"
                f" — pressão {pressure:.1f}x"
            )
            self._last_strategy = strat.name
            return strat

        if e.frustration > 0.6:
            strat = STRATEGIES["repair" if e.frustration > 0.75 else "conservative"]
            strat.exploration_pressure = pressure
            strat.reason = f"Frustração {e.frustration:.2f} — reduzindo risco"
            self._last_strategy = strat.name
            return strat

        if e.confidence < 0.3:
            strat = STRATEGIES["conservative"]
            strat.exploration_pressure = pressure
            strat.reason = f"Confiança baixa ({e.confidence:.2f}) — cautela"
            self._last_strategy = strat.name
            return strat

        if e.confidence > 0.7 and e.pleasure > 0.65:
            strat = STRATEGIES["aggressive"]
            strat.exploration_pressure = pressure
            strat.reason = f"Confiança {e.confidence:.2f} + prazer {e.pleasure:.2f}"
            self._last_strategy = strat.name
            return strat

        if e.curiosity > 0.75:
            strat = STRATEGIES["exploratory"]
            strat.exploration_pressure = pressure
            strat.reason = f"Curiosidade {e.curiosity:.2f} — explorando"
            self._last_strategy = strat.name
            return strat

        if stagnation.cycles_without_improvement >= 3:
            strat = STRATEGIES["exploratory"]
            strat.exploration_pressure = pressure
            strat.curiosity_multiplier *= pressure * 0.5
            strat.reason = (
                f"Estagnação {stagnation.cycles_without_improvement} ciclos"
                f" — pressão {pressure:.1f}x"
            )
            self._last_strategy = strat.name
            return strat

        strat = STRATEGIES["balanced"]
        strat.exploration_pressure = pressure
        strat.reason = "Estratégia balanceada padrão"
        self._last_strategy = strat.name
        return strat

    @property
    def mood(self) -> str:
        return self.emotions.mood_label

    @property
    def stats(self) -> dict:
        return {
            "mood": self.emotions.mood_label,
            "pleasure": round(self.emotions.pleasure, 3),
            "confidence": round(self.emotions.confidence, 3),
            "frustration": round(self.emotions.frustration, 3),
            "curiosity": round(self.emotions.curiosity, 3),
            "risk_tolerance": round(self.emotions.risk_tolerance, 3),
            "strategy": self._last_strategy,
        }


# ═══════════════════════════════════════════════════════════════
# Drive 3: Organ Health Audit
# ═══════════════════════════════════════════════════════════════

ORGAN_THRESHOLDS = {
    "heartbeat": {
        "dopamine_low": 0.0,
        "dopamine_critical": -0.3,
    },
    "jepa": {
        "collapse_threshold": 0.01,
        "stall_cycles": 3,
    },
    "brainstem": {
        "rejection_rate_high": 0.2,
    },
}


class OrganHealthAudit:
    def __init__(self):
        self._last_audit_time = 0.0
        self._audit_interval = 300.0
        self._organ_states: dict[str, str] = {}
        self._issue_history: list[dict] = []
        self._jepa_history: list[float] = []
        self._dopamine_history: list[float] = []
        self._brainstem_rejection_count: int = 0
        self._brainstem_total_count: int = 0
        self._jepa_stall_cycles: int = 0
        self._prev_jepa_avg: float = 1.0

    def feed_step(self, jepa_loss: float, dopamine: float, brainstem_rejected: bool) -> None:
        self._jepa_history.append(jepa_loss)
        if len(self._jepa_history) > 500:
            self._jepa_history = self._jepa_history[-500:]
        self._dopamine_history.append(dopamine)
        if len(self._dopamine_history) > 500:
            self._dopamine_history = self._dopamine_history[-500:]
        if brainstem_rejected:
            self._brainstem_rejection_count += 1
        self._brainstem_total_count += 1

    def audit(self, cycle: int, force: bool = False) -> dict[str, str]:
        now = time.time()
        if not force and (now - self._last_audit_time) < self._audit_interval:
            return self._organ_states
        self._last_audit_time = now
        states: dict[str, str] = {}

        # Heartbeat
        if len(self._dopamine_history) >= 50:
            avg_dope = sum(self._dopamine_history[-50:]) / 50
            t = ORGAN_THRESHOLDS["heartbeat"]
            if avg_dope < t["dopamine_critical"]:
                states["heartbeat"] = "critical"
                self._log_issue(cycle, "heartbeat", "critical",
                    f"Dopamina {avg_dope:.3f} < {t['dopamine_critical']}")
            elif avg_dope < t["dopamine_low"]:
                states["heartbeat"] = "warning"
                self._log_issue(cycle, "heartbeat", "warning",
                    f"Dopamina {avg_dope:.3f} < {t['dopamine_low']}")
            else:
                states["heartbeat"] = "ok"
        else:
            states["heartbeat"] = "ok"

        # JEPA
        if len(self._jepa_history) >= 100:
            recent_avg = sum(self._jepa_history[-100:]) / 100
            t = ORGAN_THRESHOLDS["jepa"]
            if recent_avg < t["collapse_threshold"]:
                states["jepa"] = "critical"
                self._log_issue(cycle, "jepa", "critical",
                    f"JEPA collapse: {recent_avg:.5f}")
            else:
                delta = abs(recent_avg - self._prev_jepa_avg)
                if delta < 0.001 and self._prev_jepa_avg > 0:
                    self._jepa_stall_cycles += 1
                else:
                    self._jepa_stall_cycles = 0
                if self._jepa_stall_cycles >= t["stall_cycles"]:
                    states["jepa"] = "warning"
                    self._log_issue(cycle, "jepa", "warning",
                        f"JEPA stalled {self._jepa_stall_cycles} cycles")
                else:
                    states["jepa"] = "ok"
            self._prev_jepa_avg = recent_avg
        else:
            states["jepa"] = "ok"

        # Brainstem
        if self._brainstem_total_count > 0:
            rate = self._brainstem_rejection_count / self._brainstem_total_count
            t = ORGAN_THRESHOLDS["brainstem"]
            if rate > t["rejection_rate_high"]:
                states["brainstem"] = "warning"
                self._log_issue(cycle, "brainstem", "warning",
                    f"Rejection rate {rate:.1%}")
            else:
                states["brainstem"] = "ok"
            self._brainstem_rejection_count = 0
            self._brainstem_total_count = 0
        else:
            states["brainstem"] = "ok"

        states["ghost"] = "disabled"
        states["curiosity"] = "ok"
        states["gaba"] = "ok"

        self._organ_states = states
        return states

    def _log_issue(self, cycle: int, organ: str, severity: str, detail: str) -> None:
        self._issue_history.append({
            "cycle": cycle, "organ": organ,
            "severity": severity, "detail": detail,
            "timestamp": time.time(),
        })
        if len(self._issue_history) > 100:
            self._issue_history = self._issue_history[-50:]

    def has_critical(self) -> bool:
        return any(s == "critical" for s in self._organ_states.values())

    def critical_organs(self) -> list[str]:
        return [o for o, s in self._organ_states.items() if s == "critical"]

    @property
    def stats(self) -> dict:
        return {
            "organ_states": dict(self._organ_states),
            "issues": self._issue_history[-3:],
            "total_issues": len(self._issue_history),
        }


# ═══════════════════════════════════════════════════════════════
# Drive 4 & 5: Curiosity + Ambition
# ═══════════════════════════════════════════════════════════════

@dataclass
class CuriosityAmbitionDrives:
    _curiosity_signals: list[dict] = field(default_factory=list)
    _ambition_signals: list[dict] = field(default_factory=list)

    def curiosity_tick(self, cycle: int, stagnation, expert_pool=None) -> list[str]:
        suggestions = []
        if stagnation.cycles_without_improvement >= 3 and expert_pool:
            records = getattr(expert_pool, "records", {})
            unused = [eid for eid, rec in records.items() if getattr(rec, "usage_count", 1) == 0]
            if unused:
                suggestions.append(
                    f"Curiosidade: {len(unused)} experts nunca usados — "
                    f"considerar prune ou forced activation"
                )
        if stagnation.cycles_without_improvement >= 5:
            pressure = stagnation.exploration_pressure
            suggestions.append(
                f"Curiosidade: pressão {pressure:.1f}x — aumentar replay ratio e JEPA weight"
            )
        return suggestions

    def ambition_tick(self, cycle: int, total_steps: int, loss_end: float, model_config=None) -> list[str]:
        suggestions = []
        if loss_end < 4.0 and total_steps > 2000:
            d_model = getattr(model_config, "d_model", 512)
            n_layers = getattr(model_config, "n_layers", 12)
            suggestions.append(
                f"Ambição: loss={loss_end:.2f} — modelo {d_model}d/{n_layers}L pronto para escalar?"
            )
        if total_steps > 5000 and loss_end < 3.5:
            suggestions.append(
                "Ambição: re-testar ghost com causal=disabled e JEPA weight ×2"
            )
        return suggestions

    @property
    def stats(self) -> dict:
        return {
            "curiosity_signals": len(self._curiosity_signals),
            "ambition_signals": len(self._ambition_signals),
        }


# ═══════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

class AutonomousDarwinDrives:
    """Orquestrador de todos os drives autônomos do organismo Darwin."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.stagnation = StagnationDetector()
        self.emotions = EmotionalStrategyModulator()
        self.health_audit = OrganHealthAudit()
        self.curiosity_ambition = CuriosityAmbitionDrives()
        self._current_strategy: TrainingStrategy = STRATEGIES["balanced"]
        self._last_cycle_time = time.time()
        self._total_signals = 0

    def feed_step(self, dopamine_raw: float, jepa_loss: float, loss_val: float,
                  brainstem_rejected: bool = False) -> None:
        if not self.enabled:
            return
        self.emotions.feed_step(dopamine_raw, loss_val)
        self.health_audit.feed_step(jepa_loss, dopamine_raw, brainstem_rejected)

    def cycle_boundary(self, cycle: int, loss_end: float,
                       model_config=None, expert_pool=None,
                       total_steps: int = 0) -> dict:
        if not self.enabled:
            return {"strategy": STRATEGIES["balanced"],
                    "diagnostics": {"autonomous_drives": "disabled"}}

        now = time.time()
        elapsed = now - self._last_cycle_time
        self._last_cycle_time = now
        self.emotions.tick(elapsed)

        stagnation_result = self.stagnation.feed(cycle, loss_end)
        organ_states = self.health_audit.audit(cycle)
        strategy = self.emotions.select_strategy(self.stagnation, organ_states)
        self._current_strategy = strategy

        curiosity_suggestions = self.curiosity_ambition.curiosity_tick(
            cycle, self.stagnation, expert_pool
        )
        ambition_suggestions = self.curiosity_ambition.ambition_tick(
            cycle, total_steps, loss_end, model_config
        )
        self._total_signals += len(curiosity_suggestions) + len(ambition_suggestions)

        # Log diagnostics
        mood = self.emotions.mood
        action = stagnation_result.get("action", "ok")
        if action in ("critical_reset", "high_pressure"):
            icon = "🚨" if action == "critical_reset" else "⚠️"
            print(
                f"  {icon} DRIVES: {action} | pressão={stagnation_result['pressure']:.1f}x | "
                f"estratégia={strategy.name} | mood={mood}",
                flush=True,
            )
        elif action == "warning":
            print(
                f"  💡 DRIVES: estagnação {stagnation_result['cycles_stagnant']} ciclos | "
                f"estratégia={strategy.name} | mood={mood}",
                flush=True,
            )
        else:
            print(
                f"  🧬 DRIVES: estratégia={strategy.name} | mood={mood} | "
                f"pressão={stagnation_result['pressure']:.1f}x",
                flush=True,
            )

        for s in curiosity_suggestions:
            print(f"  🔍 {s}", flush=True)
        for s in ambition_suggestions:
            print(f"  🚀 {s}", flush=True)
        if self.health_audit.has_critical():
            print(f"  🩺 HEALTH: CRITICAL — {', '.join(self.health_audit.critical_organs())}", flush=True)

        return {
            "strategy": strategy,
            "diagnostics": {
                "stagnation": stagnation_result,
                "organ_states": organ_states,
                "emotions": self.emotions.stats,
                "curiosity": curiosity_suggestions,
                "ambition": ambition_suggestions,
                "total_signals": self._total_signals,
            },
        }

    def apply_strategy_to_config(self, config, strategy: TrainingStrategy) -> None:
        cfg = config
        if hasattr(cfg, "ghost_enabled"):
            cfg.ghost_enabled = strategy.ghost_enabled
        if hasattr(cfg, "jepa_weight"):
            base_jepa = float(getattr(cfg, "jepa_weight", 0.5))
            cfg.jepa_weight = base_jepa * strategy.jepa_weight_mod
        if hasattr(cfg, "mtp_weight"):
            base_mtp = float(getattr(cfg, "mtp_weight", 0.15))
            cfg.mtp_weight = base_mtp * strategy.mtp_weight_mod
        if hasattr(cfg, "curiosity_weight"):
            base_cur = float(getattr(cfg, "curiosity_weight", 0.06))
            cfg.curiosity_weight = base_cur * strategy.curiosity_multiplier
        if hasattr(cfg, "replay_ratio"):
            base_replay = float(getattr(cfg, "replay_ratio", 0.5))
            cfg.replay_ratio = min(0.9, base_replay * strategy.replay_ratio_mod)
        if hasattr(cfg, "dopamine_reward_weight"):
            base_dope = float(getattr(cfg, "dopamine_reward_weight", 0.15))
            cfg.dopamine_reward_weight = base_dope * strategy.dopamine_reward_mod
        if hasattr(cfg, "exploration_pressure"):
            cfg.exploration_pressure = strategy.exploration_pressure
        if strategy.causal_mode_override and hasattr(cfg, "causal_mode"):
            cfg.causal_mode = strategy.causal_mode_override
        if strategy.accum_steps_override and hasattr(cfg, "accum_steps"):
            cfg.accum_steps = strategy.accum_steps_override

    @property
    def current_strategy(self) -> TrainingStrategy:
        return self._current_strategy

    @property
    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "stagnation": self.stagnation.stats,
            "emotions": self.emotions.stats,
            "health": self.health_audit.stats,
            "strategy": self._current_strategy.name,
            "total_signals": self._total_signals,
        }