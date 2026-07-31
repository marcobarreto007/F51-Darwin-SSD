"""Pure training adapters and narrow executors for causal interventions.

Adapters in this module consume only the frozen JSON context supplied by the
causal bus.  They never receive a model, optimizer, tensor, checkpoint, or
structural lifecycle object.  Runtime mutation is isolated in
``CausalTrainingExecutor`` after the bus has durably recorded the intent.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import copy
from dataclasses import is_dataclass, replace
import hashlib
import math
from typing import Any

import torch

from f51_darwin.darwin_x_core.losses import (
    CAUSAL_LOSS_SEMANTICS_VERSION,
    LossPolicy,
    LossTerms,
    compose_loss,
)

from .causal_bus import (
    BusDecision,
    Intervention,
    InterventionOperation,
    InterventionTarget,
    Phase,
    Signal,
    StepIdentity,
    StepOutcome,
)


_TRAINING_PHASES = frozenset(
    {Phase.PRE_LOSS, Phase.PRE_BACKWARD, Phase.PRE_OPTIMIZER}
)
_TRAINING_CONTRACT_ID = "darwin-organism-causal-training-v1"


class CausalExecutionError(RuntimeError):
    """An accepted request could not be safely applied to this runtime."""


class ExplicitRequestAdapter:
    """Stateless adapter that translates only explicit JSON requests."""

    adapter_id = "explicit-training-requests"
    version = "v1"

    def observe(
        self,
        identity: StepIdentity,
        phase: Phase,
        context: Mapping[str, Any],
    ) -> Iterable[Signal]:
        losses = context.get("losses", {})
        if not isinstance(losses, Mapping):
            return ()
        return tuple(
            Signal(
                signal_id=f"{self.adapter_id}:{phase.value}:{name}",
                source=f"{self.adapter_id}:{self.version}",
                phase=phase,
                step_key=identity.key,
                name=f"loss.{name}",
                value=value,
            )
            for name, value in sorted(losses.items())
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        )

    def propose(
        self,
        identity: StepIdentity,
        phase: Phase,
        signals: Sequence[Signal],
        context: Mapping[str, Any],
    ) -> Iterable[Intervention]:
        del signals
        if phase not in _TRAINING_PHASES:
            return ()
        requests = context.get("requests", ())
        if not isinstance(requests, (list, tuple)):
            raise TypeError("causal requests must be a JSON array")

        proposals: list[Intervention] = []
        for index, request in enumerate(requests):
            if not isinstance(request, Mapping):
                raise TypeError("each causal request must be a JSON object")
            if request.get("phase") != phase.value:
                continue
            proposals.append(
                Intervention(
                    intervention_id=str(
                        request.get("intervention_id", f"request-{index}")
                    ),
                    source=f"{self.adapter_id}:{self.version}",
                    phase=phase,
                    target=_enum_or_raw(
                        InterventionTarget, request.get("target", "")
                    ),
                    operation=_enum_or_raw(
                        InterventionOperation, request.get("operation", "")
                    ),
                    subject=str(request.get("subject", "")),
                    value=request.get("value"),
                    valid_from_step=int(request.get("valid_from_step", 0)),
                    valid_through_step=int(
                        request.get(
                            "valid_through_step", identity.optimizer_step
                        )
                    ),
                    priority=int(request.get("priority", 100)),
                    evidence_ids=tuple(request.get("evidence_ids", ())),
                )
            )
        return tuple(proposals)

    def feedback(self, outcome: StepOutcome) -> None:
        del outcome


class GABAInterventionAdapter:
    """Observes GABA inhibitory balance and proposes compensation interventions."""

    adapter_id = "gaba-inhibitory-balance"
    version = "v1"

    def observe(
        self,
        identity: StepIdentity,
        phase: Phase,
        context: Mapping[str, Any],
    ) -> Iterable[Signal]:
        if phase != Phase.PRE_LOSS:
            return ()
        gaba = context.get("gaba_observations")
        if not isinstance(gaba, (list, tuple)) or not gaba:
            return ()
        signals: list[Signal] = []
        for idx, obs in enumerate(gaba):
            if not isinstance(obs, dict):
                continue
            for key in ("mean_inhibition", "max_inhibition", "inhibited_ratio"):
                value = obs.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    signals.append(Signal(
                        signal_id=f"{self.adapter_id}:{phase.value}:L{idx}:{key}",
                        source=f"{self.adapter_id}:{self.version}",
                        phase=phase,
                        step_key=identity.key,
                        name=f"gaba.L{idx}.{key}",
                        value=float(value),
                    ))
        return tuple(signals)

    def propose(
        self,
        identity: StepIdentity,
        phase: Phase,
        signals: Sequence[Signal],
        context: Mapping[str, Any],
    ) -> Iterable[Intervention]:
        del identity, signals, context
        return ()

    def feedback(self, outcome: StepOutcome) -> None:
        del outcome


class CuriosityPriorityAdapter:
    """Observes curiosity signals and proposes replay priority interventions."""

    adapter_id = "curiosity-priority"
    version = "v1"

    def observe(
        self,
        identity: StepIdentity,
        phase: Phase,
        context: Mapping[str, Any],
    ) -> Iterable[Signal]:
        if phase != Phase.PRE_LOSS:
            return ()
        losses = context.get("raw_losses", {})
        lm = losses.get("lm")
        if not isinstance(lm, (int, float)) or isinstance(lm, bool):
            return ()
        # High LM loss → high curiosity priority for replay
        priority = float(max(0.01, min(10.0, float(lm) / 2.0)))
        return (Signal(
            signal_id=f"{self.adapter_id}:{phase.value}:replay_priority",
            source=f"{self.adapter_id}:{self.version}",
            phase=phase,
            step_key=identity.key,
            name="curiosity.replay_priority",
            value=priority,
        ),)

    def propose(
        self,
        identity: StepIdentity,
        phase: Phase,
        signals: Sequence[Signal],
        context: Mapping[str, Any],
    ) -> Iterable[Intervention]:
        del identity, signals, context
        return ()

    def feedback(self, outcome: StepOutcome) -> None:
        del outcome


class SpiderCalibrationAdapter:
    """Observes spider danger signals from persistent SpiderRAM state.

    The adapter fires at PRE_BACKWARD phase — *before* the forward pass.
    Real loss data is not available yet, so it reads the *previous* step's
    telemetry from the SpiderRAM whose ``state()`` dict must be present in
    the causal context under the ``spider_ram`` key.
    """

    adapter_id = "spider-calibration"
    version = "v2"

    def observe(
        self,
        identity: StepIdentity,
        phase: Phase,
        context: Mapping[str, Any],
    ) -> Iterable[Signal]:
        if phase != Phase.PRE_BACKWARD:
            return ()
        spider_state = context.get("spider_ram", {})
        danger = float(spider_state.get("spider.danger", 0.0))
        avg_loss = float(spider_state.get("spider.avg_loss", 0.0))
        avg_conf = float(spider_state.get("spider.avg_confidence", 0.5))
        steps = int(spider_state.get("spider.steps_seen", 0))
        alive = bool(spider_state.get("spider.alive", False))
        return (Signal(
            signal_id=f"{self.adapter_id}:{phase.value}:spider_danger",
            source=f"{self.adapter_id}:{self.version}",
            phase=phase,
            step_key=identity.key,
            name="spider.danger",
            value=danger,
            evidence_ids=[],
            metadata={
                "avg_loss": avg_loss,
                "avg_confidence": avg_conf,
                "steps_seen": steps,
                "alive": alive,
            },
        ),)

    def propose(
        self,
        identity: StepIdentity,
        phase: Phase,
        signals: Sequence[Signal],
        context: Mapping[str, Any],
    ) -> Iterable[Intervention]:
        del identity, phase, signals, context
        return ()

    def feedback(self, outcome: StepOutcome) -> None:
        del outcome


def _enum_or_raw(enum_type, value):
    try:
        return enum_type(value)
    except (TypeError, ValueError):
        return str(value)


def training_step_identity(
    organism: object,
    batch: torch.Tensor,
    *,
    accumulation_window: int,
) -> StepIdentity:
    """Create a stable identity without advancing any stochastic stream."""
    batch_bytes = (
        batch.detach().to(device="cpu").contiguous().numpy().tobytes()
    )
    rng_bytes = torch.random.get_rng_state().cpu().numpy().tobytes()
    cfg = organism.cfg
    attempt_serial = int(getattr(organism, "_causal_attempt_serial", 0))
    setattr(organism, "_causal_attempt_serial", attempt_serial + 1)
    return StepIdentity(
        run_id=str(
            getattr(cfg, "canary_run_id", None)
            or getattr(cfg, "organism_name", "darwin-organism")
        ),
        cycle=int(getattr(organism, "cycle", 0)),
        optimizer_step=int(getattr(organism, "total_steps", 0)),
        accumulation_window=int(accumulation_window),
        batch_digest=hashlib.sha256(batch_bytes).hexdigest(),
        rng_digest=hashlib.sha256(rng_bytes).hexdigest(),
        base_checkpoint_id=str(
            getattr(organism, "base_checkpoint_id", "")
        ),
        training_contract_id=_TRAINING_CONTRACT_ID,
        ablation_plan_id=str(getattr(cfg, "causal_mode", "disabled")),
        attempt_id=f"attempt-{attempt_serial}",
    )


def training_context(
    *,
    raw_losses: Mapping[str, float],
    effective_losses: Mapping[str, float],
    requests: Sequence[Mapping[str, Any]],
    optimizer_due: bool,
    spider_ram: Mapping[str, Any] | None = None,
    gaba_observations: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the tensor-free context passed to every training adapter."""
    losses = {
        f"raw.{name}": float(value)
        for name, value in sorted(raw_losses.items())
    }
    losses.update(
        {
            f"effective.{name}": float(value)
            for name, value in sorted(effective_losses.items())
        }
    )
    result: dict[str, Any] = {
        "losses": losses,
        "optimizer_due": bool(optimizer_due),
        "requests": [dict(request) for request in requests],
    }
    if spider_ram:
        result["spider_ram"] = dict(spider_ram)
    if gaba_observations:
        result["gaba_observations"] = [dict(obs) for obs in gaba_observations]
    return result


def detached_losses(output: object, loss: torch.Tensor) -> dict[str, float]:
    result = {"total": float(loss.detach().cpu())}
    for name in ("lm_loss", "mtp_loss", "jepa_loss", "jepa_dist_loss", "aux_loss", "raw_ghost_loss", "spider_loss"):
        value = getattr(output, name, None)
        if isinstance(value, torch.Tensor):
            key = "ghost" if name == "raw_ghost_loss" else name.removesuffix("_loss")
            result[key] = float(value.detach().cpu())
    return result


class CausalTrainingExecutor:
    """Apply the small training-only allowlist after a durable intent."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self._applied_intervention_ids: list[str] = []

    @property
    def applied_intervention_ids(self) -> tuple[str, ...]:
        return tuple(self._applied_intervention_ids)

    def _mark_applied(self, intervention: Intervention) -> None:
        if intervention.intervention_id not in self._applied_intervention_ids:
            self._applied_intervention_ids.append(
                intervention.intervention_id
            )

    def adjust_loss(
        self,
        decision: BusDecision,
        output: object,
        loss: torch.Tensor,
    ) -> tuple[torch.Tensor, float | None]:
        adjusted = self.set_loss_scales(decision, output)
        adjusted_loss = getattr(adjusted, "loss", None)
        if not isinstance(adjusted_loss, torch.Tensor):
            raise CausalExecutionError(
                "loss scale composition did not produce a tensor loss"
            )
        effective_aux = getattr(adjusted, "effective_aux_loss", None)
        value = (
            float(effective_aux.detach().cpu())
            if isinstance(effective_aux, torch.Tensor)
            else None
        )
        return adjusted_loss, value

    def set_loss_scales(
        self,
        decision: BusDecision,
        output: object,
    ) -> object:
        """Recompose explicit raw terms once for exact accepted loss scales."""

        requests = tuple(decision.effective)

        need_recompose = bool(requests)
        if not requests and getattr(output, "ghost_mask_digest", None) is not None:
            # CONTROL / SHADOW arm with causal ghost: ghost must not
            # contribute to gradient.  Fall through to recomposition
            # where ghost_scale → 0.0 when mask_digest is present.
            need_recompose = True
        if not need_recompose:
            return output

        scales: dict[str, float] = {}
        for request in requests:
            if (
                request.target is not InterventionTarget.LOSS_TERM
                or request.operation is not InterventionOperation.SET_SCALE
                or request.subject not in {"aux", "ghost", "jepa", "spider"}
            ):
                raise CausalExecutionError(
                    "PRE_LOSS executor accepts exact aux/ghost/jepa/spider "
                    "SET_SCALE requests"
                )
            scales[request.subject] = _finite_nonnegative(
                request.value,
                f"{request.subject} scale",
            )

        raw_lm = getattr(output, "lm_loss", None)
        if not isinstance(raw_lm, torch.Tensor):
            raise CausalExecutionError(
                "loss scaling requires explicit raw lm_loss"
            )

        subject_fields = {
            "aux": "aux_loss",
            "ghost": "raw_ghost_loss",
            "jepa": "jepa_loss",
            "spider": "spider_loss",
        }
        for subject in scales:
            field = subject_fields[subject]
            if not isinstance(getattr(output, field, None), torch.Tensor):
                raise CausalExecutionError(
                    f"{subject} SET_SCALE requires explicit {field}"
                )

        def raw_term(name: str) -> torch.Tensor:
            value = getattr(output, name, None)
            return value if isinstance(value, torch.Tensor) else raw_lm.new_zeros(())

        raw_ghost = raw_term("raw_ghost_loss")

        config = getattr(self.model, "config", None)
        version = getattr(
            config,
            "loss_semantics_version",
            CAUSAL_LOSS_SEMANTICS_VERSION,
        )
        if version != CAUSAL_LOSS_SEMANTICS_VERSION:
            raise CausalExecutionError(
                "loss scale interventions require loss_semantics_version=2"
            )
        ghost_scale = float(getattr(config, "ghost_weight", 0.0))
        if getattr(output, "ghost_mask_digest", None) is not None:
            ghost_scale = 0.0
        policy = LossPolicy(
            mtp_scale=float(getattr(config, "mtp_weight", 0.0)),
            jepa_scale=scales.get(
                "jepa",
                float(getattr(config, "jepa_weight", 0.0)),
            ),
            aux_scale=scales.get(
                "aux",
                float(getattr(config, "aux_loss_scale", 1.0)),
            ),
            ghost_scale=scales.get("ghost", ghost_scale),
            spider_scale=scales.get(
                "spider",
                float(getattr(config, "spider_calibration_weight", 0.0)),
            ),
            aux_adaptive=bool(
                getattr(config, "aux_loss_adaptive", False)
            ),
            loss_semantics_version=version,
        )
        composed = compose_loss(
            LossTerms(
                lm=raw_lm,
                mtp=raw_term("mtp_loss"),
                jepa=raw_term("jepa_loss"),
                aux=raw_term("aux_loss"),
                ghost=raw_ghost,
                spider=raw_term("spider_loss"),
            ),
            policy,
        )
        updates = {
            "loss": composed.total,
            "effective_aux_loss": composed.effective.aux,
            "effective_jepa_loss": composed.effective.jepa,
            "ghost_loss": composed.effective.ghost,
            "effective_spider_loss": composed.effective.spider,
        }
        if is_dataclass(output):
            adjusted = replace(output, **updates)
        else:
            adjusted = copy.copy(output)
            for name, value in updates.items():
                setattr(adjusted, name, value)
        for request in requests:
            self._mark_applied(request)
        return adjusted

    def should_skip(self, decision: BusDecision) -> bool:
        skip = False
        for request in decision.effective:
            if request.target is not InterventionTarget.UPDATE:
                continue
            if (
                request.operation is not InterventionOperation.SKIP
                or request.subject not in {"optimizer", "update", "all"}
                or request.value is not True
            ):
                raise CausalExecutionError(
                    "UPDATE executor accepts only an explicit true SKIP"
                )
            skip = True
            self._mark_applied(request)
        return skip

    def scale_gradients(self, decision: BusDecision) -> None:
        named = dict(self.model.named_parameters())
        operations: list[tuple[Intervention, tuple[torch.nn.Parameter, ...], float]] = []
        for request in decision.effective:
            if request.target is not InterventionTarget.GRADIENT_GROUP:
                continue
            if request.operation is not InterventionOperation.SCALE:
                raise CausalExecutionError(
                    "gradient executor accepts only SCALE"
                )
            scale = _finite_nonnegative(request.value, "gradient scale")
            if request.subject == "all":
                parameters = tuple(named.values())
            elif request.subject in named:
                parameters = (named[request.subject],)
            else:
                raise CausalExecutionError(
                    f"unknown exact gradient parameter: {request.subject}"
                )
            operations.append((request, parameters, scale))
        for request, parameters, scale in operations:
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
            self._mark_applied(request)

    def clip_gradients(
        self, decision: BusDecision, default: float
    ) -> torch.Tensor:
        value = float(default)
        interventions: list[Intervention] = []
        for request in decision.effective:
            if request.target is not InterventionTarget.GRAD_CLIP:
                continue
            if (
                request.operation is not InterventionOperation.SET_MAX_NORM
                or request.subject not in {"global", "all"}
            ):
                raise CausalExecutionError(
                    "grad clip executor accepts only global SET_MAX_NORM"
                )
            value = _finite_nonnegative(request.value, "gradient max norm")
            interventions.append(request)
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), value
        )
        for request in interventions:
            self._mark_applied(request)
        return norm


def gradient_norm(model: torch.nn.Module) -> float | None:
    squares = [
        parameter.grad.detach().float().square().sum().cpu()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not squares:
        return None
    value = torch.stack(squares).sum().sqrt()
    return float(value)


def accepted_ids(decisions: Sequence[BusDecision]) -> tuple[str, ...]:
    return tuple(
        item.intervention_id
        for decision in decisions
        for item in decision.effective
    )


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CausalExecutionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise CausalExecutionError(
            f"{label} must be finite and non-negative"
        )
    return result


__all__ = [
    "CausalExecutionError",
    "CausalTrainingExecutor",
    "ExplicitRequestAdapter",
    "accepted_ids",
    "detached_losses",
    "gradient_norm",
    "training_context",
    "training_step_identity",
]
