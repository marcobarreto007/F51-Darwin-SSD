"""Organ RL Optimizer — Aprende quais combinações de órgãos funcionam.

MUE-X inspired: RLOptimizer learns which mutation strategies work per gene.
Darwin adaptation: learns which organ configurations work per training context.

Tracks: (organ_combo, context) → outcome (loss_delta, dopamine_delta)
Uses epsilon-greedy with scorer feedback for organ toggling decisions.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════
# Organ definitions
# ═══════════════════════════════════════════════════════════════

ORGAN_DEFINITIONS = {
    "ghost": {
        "description": "Internal world model (ghost tokens)",
        "risk_level": 0.4,
        "vram_cost_mb": 200,
        "known_bugs": ["causal_shadow_hang_step_1100"],
    },
    "jepa_high": {
        "description": "JEPA weight above 0.5 (strong representation learning)",
        "risk_level": 0.3,
        "vram_cost_mb": 50,
    },
    "curiosity_high": {
        "description": "Curiosity weight above 0.1",
        "risk_level": 0.2,
        "vram_cost_mb": 0,
    },
    "gaba_active": {
        "description": "GABA inhibitory control active",
        "risk_level": 0.15,
        "vram_cost_mb": 0,
    },
    "dopamine_high": {
        "description": "Dopamine reward weight above 0.15",
        "risk_level": 0.25,
        "vram_cost_mb": 0,
    },
    "replay_high": {
        "description": "Replay ratio above 0.5",
        "risk_level": 0.1,
        "vram_cost_mb": 100,
    },
    "accum_4": {
        "description": "Gradient accumulation over 4 steps",
        "risk_level": 0.1,
        "vram_cost_mb": 0,
        "known_benefits": ["stable_gradients", "larger_effective_batch"],
    },
}


@dataclass
class OrganTrial:
    """A single trial of an organ configuration."""
    organ: str
    context: str          # "causal_shadow", "causal_disabled", "any"
    enabled: bool
    loss_delta: float     # negative = improvement
    dopamine_delta: float
    cycle: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class OrganWeight:
    """Learned weight for an organ in a given context."""
    organ: str
    context: str
    weight: float = 1.0      # higher = more likely to enable
    successes: int = 0
    failures: int = 0
    total_loss_improvement: float = 0.0
    last_used: float = 0.0
    crash_count: int = 0      # times this combo caused a crash

    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / max(total, 1)

    @property
    def effective_weight(self) -> float:
        """Weight modulated by success rate, recency, and crash penalty."""
        recency_bonus = 1.0
        if self.last_used > 0:
            hours_since = (time.time() - self.last_used) / 3600
            recency_bonus = max(0.5, 1.0 - hours_since * 0.1)
        crash_penalty = 1.0 / (1.0 + self.crash_count)
        return self.weight * (0.3 + 0.7 * self.success_rate) * recency_bonus * crash_penalty


class OrganRLOptimizer:
    """Reinforcement learning for organ configuration selection.

    Learns which organs to enable/disable based on training context
    (causal mode, stagnation pressure, loss trajectory).
    """

    def __init__(self, state_path: Path | None = None):
        self.weights: dict[str, OrganWeight] = {}  # key = "organ:context"
        self.trials: list[OrganTrial] = []
        self._state_path = state_path
        self._epsilon = 0.2  # exploration rate
        self._total_trials = 0
        self._successful_trials = 0

        # Known-safe defaults
        self._safe_organs = {
            "curiosity_high", "dopamine_high", "replay_high", "accum_4",
        }
        self._dangerous_organs = {"ghost"}

        if state_path and state_path.exists():
            self._load_state()

    def _key(self, organ: str, context: str) -> str:
        return f"{organ}:{context}"

    def _get_or_create(self, organ: str, context: str) -> OrganWeight:
        key = self._key(organ, context)
        if key not in self.weights:
            self.weights[key] = OrganWeight(organ=organ, context=context)
        return self.weights[key]

    def select_organ_config(
        self,
        context: str = "any",
        pressure: float = 1.0,
    ) -> dict[str, bool]:
        """Select which organs to enable for the next cycle.

        Uses epsilon-greedy: 80% exploit (learned best), 20% explore (random).
        Pressure increases exploration probability.
        """
        config: dict[str, bool] = {}

        for organ in ORGAN_DEFINITIONS:
            # Always enable safe organs
            if organ in self._safe_organs:
                config[organ] = True
                continue

            # Dangerous organs: decision via RL
            key = self._key(organ, context)
            weight = self.weights.get(key)

            # If no data, default to disabled for dangerous organs
            if weight is None:
                config[organ] = False
                continue

            # If this organ crashed before in this context → NEVER enable
            if weight.crash_count > 0:
                config[organ] = False
                continue

            # Epsilon-greedy (epsilon boosted by pressure)
            effective_epsilon = min(0.5, self._epsilon * pressure)
            if random.random() < effective_epsilon:
                # Explore: random
                config[organ] = random.random() < 0.3  # 30% chance to enable
            else:
                # Exploit: enable if effective_weight > 1.0 (better than neutral)
                config[organ] = weight.effective_weight > 1.0

        return config

    def record_trial(
        self,
        organ: str,
        context: str,
        enabled: bool,
        loss_delta: float,
        dopamine_delta: float,
        cycle: int,
    ) -> None:
        """Record outcome of an organ configuration trial."""
        weight = self._get_or_create(organ, context)
        weight.last_used = time.time()
        self._total_trials += 1

        # Success = loss improved
        success = loss_delta < -0.005

        if success:
            weight.successes += 1
            weight.total_loss_improvement += abs(loss_delta)
            weight.weight *= 1.05  # reinforce +5%
            weight.weight = min(5.0, weight.weight)
            self._successful_trials += 1
        else:
            weight.failures += 1
            weight.weight /= 1.05  # penalize
            weight.weight = max(0.1, weight.weight)

        trial = OrganTrial(
            organ=organ, context=context, enabled=enabled,
            loss_delta=loss_delta, dopamine_delta=dopamine_delta,
            cycle=cycle,
        )
        self.trials.append(trial)
        if len(self.trials) > 200:
            self.trials = self.trials[-100:]

        # Save state
        if self._state_path:
            self._save_state()

    def record_crash(self, organ: str, context: str) -> None:
        """Record that an organ configuration caused a crash."""
        weight = self._get_or_create(organ, context)
        weight.crash_count += 1
        # Heavy penalty for crashes
        weight.weight = max(0.05, weight.weight * 0.1)

        if self._state_path:
            self._save_state()

    def get_recommendation(self, context: str = "any") -> dict[str, Any]:
        """Get human-readable recommendation for organ configuration."""
        rec = {"enable": [], "disable": [], "unsure": []}

        for organ in ORGAN_DEFINITIONS:
            if organ in self._safe_organs:
                rec["enable"].append(organ)
                continue

            key = self._key(organ, context)
            weight = self.weights.get(key)

            if weight is None:
                rec["unsure"].append({"organ": organ, "reason": "no_data"})
                continue

            if weight.crash_count > 0:
                rec["disable"].append({
                    "organ": organ,
                    "reason": f"crashed_{weight.crash_count}x",
                    "context": context,
                })
                continue

            if weight.effective_weight > 1.5:
                rec["enable"].append({
                    "organ": organ,
                    "success_rate": round(weight.success_rate, 2),
                    "trials": weight.successes + weight.failures,
                })
            elif weight.effective_weight < 0.5:
                rec["disable"].append({
                    "organ": organ,
                    "success_rate": round(weight.success_rate, 2),
                    "trials": weight.successes + weight.failures,
                })
            else:
                rec["unsure"].append({
                    "organ": organ,
                    "weight": round(weight.effective_weight, 3),
                    "trials": weight.successes + weight.failures,
                })

        return rec

    def _save_state(self) -> None:
        if not self._state_path:
            return
        try:
            state = {
                "weights": {
                    key: {
                        "organ": w.organ,
                        "context": w.context,
                        "weight": w.weight,
                        "successes": w.successes,
                        "failures": w.failures,
                        "total_loss_improvement": w.total_loss_improvement,
                        "crash_count": w.crash_count,
                    }
                    for key, w in self.weights.items()
                },
                "total_trials": self._total_trials,
                "successful_trials": self._successful_trials,
                "last_saved": time.time(),
            }
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(
                json.dumps(state, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            print(f"  ⚠️  organ_rl: falha ao salvar estado em {self._state_path}: {exc}")

    def _load_state(self) -> None:
        if not self._state_path or not self._state_path.exists():
            return
        try:
            state = json.loads(self._state_path.read_text(encoding="utf-8"))
            for key, wdata in state.get("weights", {}).items():
                self.weights[key] = OrganWeight(
                    organ=wdata["organ"],
                    context=wdata["context"],
                    weight=wdata["weight"],
                    successes=wdata["successes"],
                    failures=wdata["failures"],
                    total_loss_improvement=wdata.get("total_loss_improvement", 0.0),
                    crash_count=wdata.get("crash_count", 0),
                )
            self._total_trials = state.get("total_trials", 0)
            self._successful_trials = state.get("successful_trials", 0)
        except (json.JSONDecodeError, KeyError) as exc:
            print(
                f"  ⚠️  organ_rl: estado em {self._state_path} corrompido/incompleto "
                f"({exc}) — reiniciando histórico de RL do zero."
            )

    @property
    def stats(self) -> dict:
        return {
            "total_trials": self._total_trials,
            "success_rate": (
                self._successful_trials / max(self._total_trials, 1)
            ),
            "organs_tracked": len(self.weights),
            "recommendation": self.get_recommendation(),
        }