"""F51 Brainstem — Homeostasis enforcer.

Keeps the organism alive by enforcing:
- VRAM limits
- Loss explosion guard
- Checkpoint safety
- Growth budget
- Synthetic ratio cap
- Training stop conditions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BrainstemConfig:
    """Homeostatic limits for the neural organism."""

    # VRAM
    max_vram_gb: float = 15.0
    vram_warning_gb: float = 13.0

    # Loss
    max_loss: float = 10.0
    loss_explosion_ratio: float = 3.0  # current/baseline must be below this
    min_loss: float = 1e-6  # below this is suspicious collapse

    # Growth
    max_active_modules: int = 8
    max_candidate_modules: int = 4
    max_total_modules: int = 16

    # Synthetic
    max_synthetic_ratio: float = 0.10  # max 10% synthetic in training mix

    # Checkpoint
    min_steps_between_checkpoints: int = 10
    max_checkpoints_per_run: int = 1000

    # Training
    max_steps_per_cycle: int = 100000
    min_tokens_for_training: int = 1024

    # Stop conditions
    loss_plateau_steps: int = 5000
    loss_plateau_threshold: float = 0.001


@dataclass
class HomeostasisReport:
    alive: bool
    warnings: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


class Brainstem:
    """Enforces homeostatic limits on the neural organism."""

    def __init__(self, config: BrainstemConfig | None = None) -> None:
        self.config = config or BrainstemConfig()
        self._baseline_loss: float | None = None
        self._loss_history: list[float] = []
        self._plateau_counter: int = 0
        self._total_modules_born: int = 0
        self._total_modules_died: int = 0

    def check_vram(self, vram_gb: float | None) -> HomeostasisReport:
        if vram_gb is None:
            return HomeostasisReport(alive=True, warnings=["VRAM not measurable"])
        if vram_gb > self.config.max_vram_gb:
            return HomeostasisReport(
                alive=False,
                violations=[f"VRAM {vram_gb:.2f}GB exceeds max {self.config.max_vram_gb}GB"],
                metrics={"vram_gb": vram_gb},
            )
        if vram_gb > self.config.vram_warning_gb:
            return HomeostasisReport(
                alive=True,
                warnings=[f"VRAM {vram_gb:.2f}GB approaching max {self.config.max_vram_gb}GB"],
                metrics={"vram_gb": vram_gb},
            )
        return HomeostasisReport(alive=True, metrics={"vram_gb": vram_gb})

    def check_loss(self, loss: float) -> HomeostasisReport:
        report = HomeostasisReport(alive=True, metrics={"loss": loss})

        if loss > self.config.max_loss:
            report.alive = False
            report.violations.append(f"Loss {loss:.4f} exceeds max {self.config.max_loss}")

        if loss < self.config.min_loss:
            report.alive = False
            report.violations.append(f"Loss {loss:.6f} below min {self.config.min_loss} — possible collapse")

        if self._baseline_loss is not None and self._baseline_loss > 0:
            ratio = loss / self._baseline_loss
            if ratio > self.config.loss_explosion_ratio:
                report.alive = False
                report.violations.append(
                    f"Loss explosion: {loss:.4f} / {self._baseline_loss:.4f} = {ratio:.2f}x"
                )

        self._loss_history.append(loss)
        if len(self._loss_history) > self.config.loss_plateau_steps:
            self._loss_history = self._loss_history[-self.config.loss_plateau_steps :]

        return report

    def check_loss_plateau(self) -> bool:
        if len(self._loss_history) < self.config.loss_plateau_steps:
            return False
        recent = self._loss_history[-self.config.loss_plateau_steps :]
        delta = abs(recent[0] - recent[-1])
        return delta < self.config.loss_plateau_threshold

    def set_baseline_loss(self, loss: float) -> None:
        self._baseline_loss = loss

    def check_module_count(
        self, active: int, candidate: int, total: int
    ) -> HomeostasisReport:
        report = HomeostasisReport(alive=True)
        if active > self.config.max_active_modules:
            report.alive = False
            report.violations.append(f"Active modules {active} > max {self.config.max_active_modules}")
        if candidate > self.config.max_candidate_modules:
            report.warnings.append(f"Candidate modules {candidate} > max {self.config.max_candidate_modules}")
        if total > self.config.max_total_modules:
            report.alive = False
            report.violations.append(f"Total modules {total} > max {self.config.max_total_modules}")
        return report

    def check_synthetic_ratio(self, synthetic_tokens: int, total_tokens: int) -> HomeostasisReport:
        if total_tokens == 0:
            return HomeostasisReport(alive=True)
        ratio = synthetic_tokens / total_tokens
        if ratio > self.config.max_synthetic_ratio:
            return HomeostasisReport(
                alive=False,
                violations=[f"Synthetic ratio {ratio:.2%} exceeds max {self.config.max_synthetic_ratio:.2%}"],
                metrics={"synthetic_ratio": ratio},
            )
        return HomeostasisReport(alive=True, metrics={"synthetic_ratio": ratio})

    def full_check(
        self,
        *,
        vram_gb: float | None = None,
        loss: float | None = None,
        active_modules: int = 0,
        candidate_modules: int = 0,
        total_modules: int = 0,
        synthetic_tokens: int = 0,
        total_tokens: int = 0,
    ) -> HomeostasisReport:
        """Run all homeostatic checks. Returns first violation that would kill the organism."""
        violations: list[str] = []
        warnings: list[str] = []
        metrics: dict[str, Any] = {}

        checks = []
        if vram_gb is not None:
            checks.append(self.check_vram(vram_gb))
        if loss is not None:
            checks.append(self.check_loss(loss))
        if active_modules or candidate_modules or total_modules:
            checks.append(self.check_module_count(active_modules, candidate_modules, total_modules))
        if synthetic_tokens or total_tokens:
            checks.append(self.check_synthetic_ratio(synthetic_tokens, total_tokens))

        alive = True
        for check in checks:
            violations.extend(check.violations)
            warnings.extend(check.warnings)
            metrics.update(check.metrics)
            if not check.alive:
                alive = False

        return HomeostasisReport(alive=alive, warnings=warnings, violations=violations, metrics=metrics)
