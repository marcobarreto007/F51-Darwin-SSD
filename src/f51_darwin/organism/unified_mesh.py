from __future__ import annotations
import copy
from dataclasses import dataclass, field
import math
import random
from typing import Any, Callable, Mapping

import numpy as np
import torch

from .causal_bus import (
    Intervention,
    InterventionOperation,
    InterventionTarget,
    Phase,
    Signal,
    StepIdentity,
    StepOutcome,
)

@dataclass
class OrganSignals:
    spider_confidence: float = 0.0
    spider_danger: float = 0.0
    gaba_ei_ratio: float = 0.0
    heartbeat_dopamine: float = 0.0
    heartbeat_ttm_surprise: float = 0.0
    neuroendocrine_cortisol: float = 0.0
    neuroendocrine_bdnf: float = 0.0
    neuroendocrine_ach_pressure: float = 0.0
    decision_confidence: float = 0.0
    lateralization_index: float = 0.0
    expert_usage_gini: float = 0.0
    loss_trend: float = 0.0

@dataclass
class EvolutionReport:
    births: int = 0
    deaths: int = 0
    quarantines: int = 0
    expansions: int = 0
    pruning_decisions: int = 0
    details: list[str] = field(default_factory=list)

@dataclass
class SleepReport:
    executed: bool = False
    proposed: bool = False
    accepted: bool = False
    proposed_prunable_params: int = 0
    pruned_params: int = 0
    loss_before: float = 0.0
    loss_after: float = 0.0
    reverted: bool = False
    degradation_pct: float = 0.0
    metrics_before: dict[str, float] = field(default_factory=dict)
    metrics_after: dict[str, float] = field(default_factory=dict)
    regressions: tuple[str, ...] = ()
    error: str | None = None
    rollback_error: str | None = None
    rollback_partial: bool = False


@dataclass
class ProtectedSleepSnapshot:
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    parameter_grads: dict[str, torch.Tensor | None]
    rng_state: dict[str, Any]
    organ_state: Any
    heartbeat_state: Any
    module_training: dict[str, bool]


class StructuralProposalAdapter:
    """Propose one cycle-boundary authorization from observed mesh evidence."""

    adapter_id = "unified-mesh-structural"
    version = "v1"

    def observe(self, identity, phase, context):
        if phase is not Phase.CYCLE_BOUNDARY:
            return ()
        count = context.get("proposal_count", 0)
        actions = context.get("proposals", ())
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or not isinstance(actions, (list, tuple))
            or len(actions) != count
            or any(not isinstance(item, str) or not item for item in actions)
        ):
            return ()
        return (
            Signal(
                signal_id=f"{self.adapter_id}:{identity.key}",
                source=f"{self.adapter_id}:{self.version}",
                phase=phase,
                step_key=identity.key,
                name="structural.proposal_count",
                value=count,
            ),
        )

    def propose(self, identity, phase, signals, context):
        if phase is not Phase.CYCLE_BOUNDARY or not signals:
            return ()
        count = context.get("proposal_count", 0)
        actions = sorted(str(item) for item in context.get("proposals", ()))
        return (
            Intervention(
                intervention_id=f"structural-boundary:{identity.key}",
                source=f"{self.adapter_id}:{self.version}",
                phase=phase,
                target=InterventionTarget.STRUCTURAL_ACTION,
                operation=InterventionOperation.QUEUE,
                subject="model.deferred_structural_actions",
                value={
                    "proposal_count": int(count),
                    "actions": actions,
                },
                valid_from_step=identity.optimizer_step,
                valid_through_step=identity.optimizer_step,
                evidence_ids=tuple(signal.signal_id for signal in signals),
            ),
        )

    def feedback(self, outcome: StepOutcome) -> None:
        del outcome

class UnifiedControlMesh:
    """Barramento central — agrega sinais de todos os órgãos."""

    def __init__(self, model):
        self.model = model

    def collect_signals(self, output=None) -> OrganSignals:
        signals = OrganSignals()
        # Spider-Sense
        if hasattr(output, 'spider_confidence') and output.spider_confidence is not None:
            signals.spider_confidence = float(output.spider_confidence.mean().item())
            signals.spider_danger = float((1.0 - output.spider_confidence).mean().item())
        # Heartbeat
        heartbeat_stats = getattr(output, 'heartbeat_stats', None) or {}
        if heartbeat_stats:
            signals.heartbeat_dopamine = float(heartbeat_stats.get('dopamine', 0.0))
            memory_stats = heartbeat_stats.get('memory', {}) or {}
            signals.heartbeat_ttm_surprise = float(
                memory_stats.get('avg_surprise', 0.0)
            )
        elif getattr(self.model, 'heartbeat', None) is not None:
            signals.heartbeat_dopamine = float(
                getattr(self.model.heartbeat, 'dopamine', 0.0)
            )
        # Neuroendocrine (média entre blocos)
        cortisol_vals, bdnf_vals, ach_vals = [], [], []
        for block in self.model.blocks:
            if hasattr(block, 'moe') and hasattr(block.moe, 'neuroendocrine'):
                ne = block.moe.neuroendocrine
                state = ne.state()
                cortisol_vals.append(float(state.get('cortisol', 0.0)))
                bdnf_vals.append(_mean_signal(
                    state.get('bdnf_per_expert', state.get('bdnf', 0.0))
                ))
                ach_vals.append(float(
                    state.get('ach_pressure', state.get('ach', 0.0))
                ))
        if cortisol_vals:
            signals.neuroendocrine_cortisol = sum(cortisol_vals) / len(cortisol_vals)
            signals.neuroendocrine_bdnf = sum(bdnf_vals) / len(bdnf_vals)
            signals.neuroendocrine_ach_pressure = sum(ach_vals) / len(ach_vals)
        # Decision confidence
        if hasattr(output, 'decision_factors') and output.decision_factors:
            signals.decision_confidence = output.decision_factors.get('confidence', 0.0)
        # Expert usage Gini
        signals.expert_usage_gini = self._compute_usage_gini()
        return signals

    def _compute_usage_gini(self) -> float:
        usages = []
        for block in self.model.blocks:
            if hasattr(block, 'moe'):
                moe = block.moe
                usage = getattr(moe, '_expert_usage_buffer', None)
                if usage is None and hasattr(moe, 'neuroendocrine'):
                    usage = getattr(moe.neuroendocrine, 'expert_usage_ema', None)
                if usage is not None:
                    usages.extend(
                        float(item)
                        for item in torch.as_tensor(usage).detach().flatten()
                    )
        if not usages:
            return 0.0
        sorted_u = sorted(usages)
        n = len(sorted_u)
        total = sum(sorted_u)
        if total == 0:
            return 0.0
        return (2 * sum((i + 1) * u for i, u in enumerate(sorted_u)) / (n * total)) - (n + 1) / n

    def should_trigger_evolution(self, signals: OrganSignals) -> bool:
        return (signals.neuroendocrine_cortisol > 0.3 or signals.neuroendocrine_ach_pressure > 0.1)


def execute_evolution_actions(model, signals: OrganSignals) -> EvolutionReport:
    """Inspect legacy plasticity decisions without mutating model topology."""
    report = EvolutionReport()
    for block_idx, block in enumerate(model.blocks):
        if not hasattr(block, 'moe') or not hasattr(block.moe, 'neuroendocrine'):
            continue
        ne = block.moe.neuroendocrine
        decisions = ne.plasticity_decision()
        proposed = _normalise_plasticity_decisions(decisions)
        report.pruning_decisions += sum(len(items) for items in proposed.values())
        create_proposed = (
            isinstance(decisions, dict)
            and decisions.get("create") is True
        )
        if create_proposed:
            report.pruning_decisions += 1
            report.details.append(f"proposed:create:B{block_idx}")
        for action, expert_indices in proposed.items():
            for expert_idx in expert_indices:
                report.details.append(
                    f"proposed:{action}:B{block_idx}_E{expert_idx}"
                )
    return report


def run_sleep_cycle(model, signals: OrganSignals) -> SleepReport:
    """Propose REM/L1 pruning without mutating weights or hormonal state."""
    report = SleepReport()
    should_sleep = any(
        hasattr(block, 'moe') and hasattr(block.moe, 'neuroendocrine')
        and block.moe.neuroendocrine.should_sleep()
        for block in model.blocks
    )
    if not should_sleep:
        return report

    report.proposed = True
    prune_threshold = 0.01
    pruned_total = 0

    for block in model.blocks:
        if not hasattr(block, 'moe'):
            continue
        for name, param in block.moe.named_parameters():
            if 'weight' in name:
                pruned_total += int((param.detach().abs() < prune_threshold).sum().item())
    report.proposed_prunable_params = pruned_total
    return report


def _sleep_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch_cpu": torch.random.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [
            item.clone() for item in torch.cuda.get_rng_state_all()
        ]
    return state


def _clone_sleep_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {
            key: _clone_sleep_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_clone_sleep_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_sleep_value(item) for item in value)
    return copy.deepcopy(value)


def _restore_sleep_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def snapshot_protected_sleep(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    snapshot_organs: Callable[[], Any] | None = None,
) -> ProtectedSleepSnapshot:
    """Capture every mutable state required for exact sleep rollback."""

    return ProtectedSleepSnapshot(
        model_state={
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
        optimizer_state=_clone_sleep_value(optimizer.state_dict()),
        parameter_grads={
            name: (
                None
                if parameter.grad is None
                else parameter.grad.detach().cpu().clone()
            )
            for name, parameter in model.named_parameters()
        },
        rng_state=_sleep_rng_state(),
        organ_state=copy.deepcopy(
            snapshot_organs() if snapshot_organs is not None else None
        ),
        heartbeat_state=_clone_sleep_value(
            model.heartbeat_state_dict()
            if hasattr(model, "heartbeat_state_dict")
            else None
        ),
        module_training={
            name: bool(module.training)
            for name, module in model.named_modules()
        },
    )


def rollback_protected_sleep(
    snapshot: ProtectedSleepSnapshot,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    restore_organs: Callable[[Any], None] | None = None,
) -> None:
    """Restore a protected-sleep snapshot without weakening strictness."""

    model.load_state_dict(snapshot.model_state, strict=True)
    optimizer.load_state_dict(snapshot.optimizer_state)
    for name, parameter in model.named_parameters():
        gradient = snapshot.parameter_grads.get(name)
        parameter.grad = (
            None
            if gradient is None
            else gradient.to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
        )
    _restore_sleep_rng(snapshot.rng_state)
    if (
        snapshot.heartbeat_state is not None
        and hasattr(model, "load_heartbeat_state_dict")
    ):
        model.load_heartbeat_state_dict(
            _clone_sleep_value(snapshot.heartbeat_state)
        )
    if restore_organs is not None:
        restore_organs(copy.deepcopy(snapshot.organ_state))
    for name, module in model.named_modules():
        if name in snapshot.module_training:
            module.training = snapshot.module_training[name]


def _validated_sleep_metrics(
    metrics: Mapping[str, float],
) -> dict[str, float]:
    required = {"fresh", "replay", "heldout"}
    if not isinstance(metrics, Mapping) or set(metrics) != required:
        raise ValueError(
            "protected sleep requires fresh, replay, and heldout metrics"
        )
    validated: dict[str, float] = {}
    for name in sorted(required):
        value = metrics[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(
                f"protected sleep metric {name} must be finite and non-negative"
            )
        validated[name] = float(value)
    return validated


def run_protected_sleep(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    enabled: bool,
    prepare: Callable[[], Any] | None = None,
    consolidate: Callable[[], Any],
    evaluate: Callable[[], Mapping[str, float]],
    snapshot_organs: Callable[[], Any] | None = None,
    restore_organs: Callable[[Any], None] | None = None,
    max_regression: float = 0.0,
) -> SleepReport:
    """Run replay consolidation transactionally and rollback any regression."""

    report = SleepReport(proposed=bool(enabled))
    if not enabled:
        return report
    if (
        isinstance(max_regression, bool)
        or not isinstance(max_regression, (int, float))
        or not math.isfinite(float(max_regression))
        or float(max_regression) < 0.0
    ):
        raise ValueError("max_regression must be finite and non-negative")

    snapshot = snapshot_protected_sleep(
        model,
        optimizer,
        snapshot_organs=snapshot_organs,
    )
    report.executed = True

    def restore_snapshot(
        target: ProtectedSleepSnapshot,
        *,
        marks_reversion: bool,
    ) -> bool:
        try:
            rollback_protected_sleep(
                target,
                model,
                optimizer,
                restore_organs=restore_organs,
            )
        except Exception as rollback_exc:
            report.rollback_error = (
                f"{type(rollback_exc).__name__}:{rollback_exc}"
            )
            report.rollback_partial = True
            report.reverted = False
            return False
        if marks_reversion:
            report.reverted = True
        return True

    try:
        if prepare is not None:
            prepare()
        prepared_snapshot = snapshot_protected_sleep(
            model,
            optimizer,
            snapshot_organs=snapshot_organs,
        )
        before = _validated_sleep_metrics(evaluate())
        report.metrics_before = before
        if not restore_snapshot(
            prepared_snapshot,
            marks_reversion=False,
        ):
            report.error = "EvaluationIsolationError:baseline restore failed"
            restore_snapshot(snapshot, marks_reversion=True)
            return report

        consolidate()
        candidate_snapshot = snapshot_protected_sleep(
            model,
            optimizer,
            snapshot_organs=snapshot_organs,
        )
        # Evaluate both states with exactly the same global RNG state. The
        # accepted candidate keeps the post-consolidation RNG captured above.
        _restore_sleep_rng(prepared_snapshot.rng_state)
        after = _validated_sleep_metrics(evaluate())
        report.metrics_after = after
        if not restore_snapshot(
            candidate_snapshot,
            marks_reversion=False,
        ):
            report.error = "EvaluationIsolationError:candidate restore failed"
            restore_snapshot(snapshot, marks_reversion=True)
            return report
        regressions = tuple(
            name
            for name in ("fresh", "replay", "heldout")
            if after[name]
            > before[name] + max(abs(before[name]), 1e-12) * float(max_regression)
        )
        report.regressions = regressions
        report.loss_before = sum(before.values()) / len(before)
        report.loss_after = sum(after.values()) / len(after)
        report.degradation_pct = max(
            (
                100.0 * (after[name] - before[name])
                / max(abs(before[name]), 1e-12)
                for name in before
            ),
            default=0.0,
        )
        if regressions:
            restore_snapshot(snapshot, marks_reversion=True)
            return report
        report.accepted = True
        return report
    except Exception as exc:
        report.error = f"{type(exc).__name__}:{exc}"
        restore_snapshot(snapshot, marks_reversion=True)
        return report
    finally:
        for name, module in model.named_modules():
            if name in snapshot.module_training:
                module.training = snapshot.module_training[name]


def _mean_signal(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().item()) if value.numel() else 0.0
    if isinstance(value, (list, tuple)):
        return sum(float(item) for item in value) / len(value) if value else 0.0
    return float(value or 0.0)


def _normalise_plasticity_decisions(decisions: Any) -> dict[str, list[int]]:
    current_actions = {
        'ignore', 'update', 'protect', 'expand', 'prune', 'quarantine', 'create'
    }
    structural = {'prune': [], 'quarantine': [], 'expand': []}
    if not isinstance(decisions, dict):
        raise ValueError("plasticity decision schema must be a mapping")

    if all(isinstance(key, str) and key in current_actions for key in decisions):
        for action, value in decisions.items():
            if action == 'create':
                if not isinstance(value, bool):
                    raise ValueError("plasticity decision schema: create must be bool")
                continue
            if not isinstance(value, (list, tuple, set)):
                raise ValueError(
                    f"plasticity decision schema: {action} must be a sequence"
                )
            if action in structural:
                structural[action] = [int(item) for item in value]
        return structural

    legacy_actions = current_actions - {'create'}
    if all(
        isinstance(key, int)
        and isinstance(value, str)
        and value in legacy_actions
        for key, value in decisions.items()
    ):
        for expert_idx, action in decisions.items():
            if action in structural:
                structural[action].append(expert_idx)
        return structural

    raise ValueError("plasticity decision schema is neither current nor legacy")


def _run_sanity_checks():
    print("UnifiedControlMesh: sanity checks...")
    s = OrganSignals(spider_confidence=0.8, neuroendocrine_cortisol=0.5)
    assert s.spider_confidence == 0.8
    r = EvolutionReport(deaths=3, quarantines=1)
    assert r.deaths == 3
    sr = SleepReport(executed=True, pruned_params=42)
    assert sr.pruned_params == 42
    # Gini test
    mesh = UnifiedControlMesh.__new__(UnifiedControlMesh)
    assert mesh._compute_usage_gini.__func__(mesh) == 0.0  # sem modelo
    print("  OK: todos os dataclasses e sanity checks passaram")

if __name__ == "__main__":
    _run_sanity_checks()
