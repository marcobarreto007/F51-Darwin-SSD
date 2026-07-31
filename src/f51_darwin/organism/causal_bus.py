"""Typed, deterministic causal-intervention contracts for the organism runtime.

The bus deliberately owns no model or optimizer reference.  Adapters receive
only immutable JSON-compatible observations and can return proposals; applying
an accepted proposal remains the responsibility of a narrow runtime executor.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import copy
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import pickle
import random
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
import torch

class Phase(str, Enum):
    PRE_LOSS = "PRE_LOSS"
    PRE_BACKWARD = "PRE_BACKWARD"
    PRE_OPTIMIZER = "PRE_OPTIMIZER"
    POST_STEP = "POST_STEP"
    CYCLE_BOUNDARY = "CYCLE_BOUNDARY"


class AblationArm(str, Enum):
    CONTROL = "CONTROL"
    SHADOW = "SHADOW"
    APPLY = "APPLY"


class InterventionTarget(str, Enum):
    LOSS_TERM = "LOSS_TERM"
    UPDATE = "UPDATE"
    GRAD_CLIP = "GRAD_CLIP"
    GRADIENT_GROUP = "GRADIENT_GROUP"
    STRUCTURAL_ACTION = "STRUCTURAL_ACTION"


class InterventionOperation(str, Enum):
    SET_SCALE = "SET_SCALE"
    SKIP = "SKIP"
    SET_MAX_NORM = "SET_MAX_NORM"
    SCALE = "SCALE"
    QUEUE = "QUEUE"


_ALLOWED_INTERVENTIONS: dict[
    Phase, frozenset[tuple[InterventionTarget, InterventionOperation]]
] = {
    Phase.PRE_LOSS: frozenset(
        {(InterventionTarget.LOSS_TERM, InterventionOperation.SET_SCALE)}
    ),
    Phase.PRE_BACKWARD: frozenset(
        {(InterventionTarget.UPDATE, InterventionOperation.SKIP)}
    ),
    Phase.PRE_OPTIMIZER: frozenset(
        {
            (InterventionTarget.UPDATE, InterventionOperation.SKIP),
            (InterventionTarget.GRAD_CLIP, InterventionOperation.SET_MAX_NORM),
            (InterventionTarget.GRADIENT_GROUP, InterventionOperation.SCALE),
        }
    ),
    Phase.POST_STEP: frozenset(),
    Phase.CYCLE_BOUNDARY: frozenset(
        {(InterventionTarget.STRUCTURAL_ACTION, InterventionOperation.QUEUE)}
    ),
}


JsonValue = str | int | float | bool | None | Mapping[str, Any] | tuple[Any, ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _thaw_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _freeze_json(value: Any, *, path: str = "value") -> JsonValue:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Enum):
        return _freeze_json(value.value, path=path)
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} must use string keys to be JSON-compatible")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{path} must be JSON-compatible, got {type(value).__name__}"
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"value must be JSON-compatible, got {type(value).__name__}")


def _contains_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, Mapping):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_nonfinite(item) for item in value)
    return False


@dataclass(frozen=True)
class StepIdentity:
    run_id: str
    cycle: int
    optimizer_step: int
    accumulation_window: int
    batch_digest: str
    rng_digest: str
    base_checkpoint_id: str
    training_contract_id: str
    ablation_plan_id: str = ""
    attempt_id: str = "attempt-0"

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "batch_digest",
            "rng_digest",
            "base_checkpoint_id",
            "training_contract_id",
            "attempt_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string anchor")
        for name in ("cycle", "optimizer_step", "accumulation_window"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(f"{name} must be a non-negative integer")
        if not isinstance(self.ablation_plan_id, str):
            raise TypeError("ablation_plan_id must be a string")

    @property
    def key(self) -> str:
        payload = {
            "ablation_plan_id": self.ablation_plan_id,
            "accumulation_window": self.accumulation_window,
            "base_checkpoint_id": self.base_checkpoint_id,
            "batch_digest": self.batch_digest,
            "cycle": self.cycle,
            "optimizer_step": self.optimizer_step,
            "rng_digest": self.rng_digest,
            "run_id": self.run_id,
            "training_contract_id": self.training_contract_id,
            "attempt_id": self.attempt_id,
        }
        digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        return f"causal-step-v1:{digest}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "run_id": self.run_id,
            "cycle": self.cycle,
            "optimizer_step": self.optimizer_step,
            "accumulation_window": self.accumulation_window,
            "batch_digest": self.batch_digest,
            "rng_digest": self.rng_digest,
            "base_checkpoint_id": self.base_checkpoint_id,
            "training_contract_id": self.training_contract_id,
            "ablation_plan_id": self.ablation_plan_id,
            "attempt_id": self.attempt_id,
        }

    def to_state(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "cycle": self.cycle,
            "optimizer_step": self.optimizer_step,
            "accumulation_window": self.accumulation_window,
            "batch_digest": self.batch_digest,
            "rng_digest": self.rng_digest,
            "base_checkpoint_id": self.base_checkpoint_id,
            "training_contract_id": self.training_contract_id,
            "ablation_plan_id": self.ablation_plan_id,
            "attempt_id": self.attempt_id,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "StepIdentity":
        return cls(
            run_id=state["run_id"],
            cycle=state["cycle"],
            optimizer_step=state["optimizer_step"],
            accumulation_window=state["accumulation_window"],
            batch_digest=state["batch_digest"],
            rng_digest=state["rng_digest"],
            base_checkpoint_id=state["base_checkpoint_id"],
            training_contract_id=state["training_contract_id"],
            ablation_plan_id=state["ablation_plan_id"],
            attempt_id=state["attempt_id"],
        )


@dataclass(frozen=True)
class Signal:
    signal_id: str
    source: str
    phase: Phase
    step_key: str
    name: str
    value: JsonValue
    unit: str = ""
    evidence_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _freeze_json(self.value, path="signal.value"))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        if any(
            not isinstance(value, str) or not value
            for value in (self.signal_id, self.source, self.name)
        ):
            raise ValueError("signal_id, source and name must not be empty")
        if not isinstance(self.phase, Phase):
            raise TypeError("signal.phase must be Phase")
        if not isinstance(self.step_key, str) or not self.step_key:
            raise ValueError("signal.step_key must be a non-empty anchor")
        if len(set(self.evidence_ids)) != len(self.evidence_ids) or any(
            not isinstance(item, str) or not item
            for item in self.evidence_ids
        ):
            raise ValueError("signal evidence_ids must be unique anchors")
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version < 1
        ):
            raise ValueError("signal schema_version must be positive")


@dataclass(frozen=True)
class Intervention:
    intervention_id: str
    source: str
    phase: Phase
    target: InterventionTarget | str
    operation: InterventionOperation | str
    subject: str
    value: JsonValue
    valid_from_step: int
    valid_through_step: int
    priority: int = 100
    evidence_ids: tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "value", _freeze_json(self.value, path="intervention.value")
        )
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        if any(
            not isinstance(value, str) or not value
            for value in (self.intervention_id, self.source, self.subject)
        ):
            raise ValueError(
                "intervention_id, source and subject must not be empty"
            )
        if not isinstance(self.phase, Phase):
            raise TypeError("intervention.phase must be Phase")
        for name in ("valid_from_step", "valid_through_step", "priority"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"intervention {name} must be an integer")
        if self.valid_from_step < 0 or self.valid_through_step < 0:
            raise ValueError("intervention step bounds must be non-negative")
        if self.valid_through_step < self.valid_from_step:
            raise ValueError(
                "valid_through_step must be greater than or equal to valid_from_step"
            )
        if len(set(self.evidence_ids)) != len(self.evidence_ids) or any(
            not isinstance(item, str) or not item
            for item in self.evidence_ids
        ):
            raise ValueError(
                "intervention evidence_ids must be unique anchors"
            )
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version < 1
        ):
            raise ValueError("intervention schema_version must be positive")

    @property
    def target_key(self) -> tuple[str, str]:
        target = self.target.value if isinstance(self.target, Enum) else str(self.target)
        return target, self.subject


@dataclass(frozen=True)
class Rejection:
    item_id: str
    source: str
    reason: str


@dataclass(frozen=True)
class StepOutcome:
    identity: StepIdentity
    optimizer_step_applied: bool
    accepted_intervention_ids: tuple[str, ...] = ()
    raw_losses: Mapping[str, float] = field(default_factory=dict)
    effective_losses: Mapping[str, float] = field(default_factory=dict)
    gradient_norm_before: float | None = None
    gradient_norm_after: float | None = None
    parameter_delta_norm: float | None = None
    heldout_snapshot: Mapping[str, float] = field(default_factory=dict)
    replay_snapshot: Mapping[str, float] = field(default_factory=dict)
    degraded_organs: tuple[str, ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, StepIdentity):
            raise TypeError("outcome.identity must be StepIdentity")
        if not isinstance(self.optimizer_step_applied, bool):
            raise TypeError("optimizer_step_applied must be bool")
        object.__setattr__(
            self, "accepted_intervention_ids", tuple(self.accepted_intervention_ids)
        )
        if any(
            not isinstance(item, str) or not item
            for item in self.accepted_intervention_ids
        ):
            raise TypeError(
                "accepted_intervention_ids must contain non-empty strings"
            )
        if len(set(self.accepted_intervention_ids)) != len(
            self.accepted_intervention_ids
        ):
            raise ValueError("accepted_intervention_ids must be unique")
        object.__setattr__(
            self,
            "raw_losses",
            _freeze_json(dict(self.raw_losses), path="outcome.raw_losses"),
        )
        object.__setattr__(
            self,
            "effective_losses",
            _freeze_json(
                dict(self.effective_losses), path="outcome.effective_losses"
            ),
        )
        object.__setattr__(
            self,
            "heldout_snapshot",
            _freeze_json(
                dict(self.heldout_snapshot), path="outcome.heldout_snapshot"
            ),
        )
        object.__setattr__(
            self,
            "replay_snapshot",
            _freeze_json(
                dict(self.replay_snapshot), path="outcome.replay_snapshot"
            ),
        )
        object.__setattr__(
            self, "degraded_organs", tuple(self.degraded_organs)
        )
        if any(
            not isinstance(item, str) or not item
            for item in self.degraded_organs
        ):
            raise TypeError("degraded_organs must contain non-empty strings")
        for mapping_name in (
            "raw_losses",
            "effective_losses",
            "heldout_snapshot",
            "replay_snapshot",
        ):
            mapping = getattr(self, mapping_name)
            for name, value in mapping.items():
                if (
                    not isinstance(name, str)
                    or isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise TypeError(
                        f"{mapping_name} must contain finite numeric values"
                    )
        for name in (
            "gradient_norm_before",
            "gradient_norm_after",
            "parameter_delta_norm",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise TypeError(f"{name} must be finite and non-negative")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be a string or None")


@dataclass(frozen=True)
class BusDecision:
    identity: StepIdentity
    phase: Phase
    arm: AblationArm
    signals: tuple[Signal, ...] = ()
    accepted: tuple[Intervention, ...] = ()
    rejected: tuple[Rejection, ...] = ()
    blocked: bool = False
    context: Mapping[str, Any] = field(default_factory=dict)

    @property
    def effective(self) -> tuple[Intervention, ...]:
        if self.arm is AblationArm.APPLY and not self.blocked:
            return self.accepted
        return ()


class OrganAdapter(Protocol):
    adapter_id: str
    version: str

    def observe(
        self,
        identity: StepIdentity,
        phase: Phase,
        context: Mapping[str, Any],
    ) -> Iterable[Signal]: ...

    def propose(
        self,
        identity: StepIdentity,
        phase: Phase,
        signals: Sequence[Signal],
        context: Mapping[str, Any],
    ) -> Iterable[Intervention]: ...

    def feedback(self, outcome: StepOutcome) -> None: ...


class IntentRecorder(Protocol):
    def record_intent(
        self,
        identity: StepIdentity,
        decisions: Sequence[BusDecision],
    ) -> Any: ...


def _adapter_sort_key(adapter: OrganAdapter) -> tuple[str, str]:
    return str(adapter.adapter_id), str(adapter.version)


def _intervention_sort_key(item: Intervention) -> tuple[int, str, str]:
    return int(item.priority), item.source, item.intervention_id


def _adapter_sources(adapter: OrganAdapter) -> frozenset[str]:
    adapter_id = str(adapter.adapter_id)
    version = str(adapter.version)
    return frozenset({f"{adapter_id}:{version}"})


_MISSING_SLOT = object()


def _slot_names(adapter: OrganAdapter) -> tuple[str, ...]:
    names: list[str] = []
    for owner in type(adapter).__mro__:
        declared = owner.__dict__.get("__slots__", ())
        if isinstance(declared, str):
            declared = (declared,)
        for name in declared:
            if name in {"__dict__", "__weakref__"}:
                continue
            if name.startswith("__") and not name.endswith("__"):
                name = f"_{owner.__name__.lstrip('_')}{name}"
            if name not in names:
                names.append(name)
    return tuple(names)


def _adapter_state_snapshot(adapter: OrganAdapter) -> dict[str, Any]:
    dictionary = getattr(adapter, "__dict__", None)
    slots = {
        name: (
            copy.deepcopy(getattr(adapter, name))
            if hasattr(adapter, name)
            else _MISSING_SLOT
        )
        for name in _slot_names(adapter)
    }
    return {
        "dict": copy.deepcopy(dictionary) if dictionary is not None else None,
        "slots": slots,
    }


def _adapter_fingerprint(snapshot: Mapping[str, Any]) -> bytes:
    fingerprintable = {
        "dict": snapshot["dict"],
        "slots": {
            name: (
                "<MISSING_SLOT>" if value is _MISSING_SLOT else value
            )
            for name, value in snapshot["slots"].items()
        },
    }
    try:
        payload = pickle.dumps(fingerprintable, protocol=5)
    except Exception:
        payload = repr(fingerprintable).encode(
            "utf-8", errors="backslashreplace"
        )
    return hashlib.sha256(payload).digest()


def _restore_adapter_state(
    adapter: OrganAdapter, snapshot: Mapping[str, Any]
) -> None:
    dictionary = getattr(adapter, "__dict__", None)
    if dictionary is not None:
        dictionary.clear()
        dictionary.update(copy.deepcopy(snapshot["dict"]))
    for name, value in snapshot["slots"].items():
        if value is _MISSING_SLOT:
            if hasattr(adapter, name):
                delattr(adapter, name)
        else:
            setattr(adapter, name, copy.deepcopy(value))


def _snapshot_rng() -> tuple[Any, Any, torch.Tensor, list[torch.Tensor] | None]:
    return (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def _restore_rng(
    snapshot: tuple[Any, Any, torch.Tensor, list[torch.Tensor] | None]
) -> None:
    python_state, numpy_state, cpu_state, cuda_state = snapshot
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


class _AdapterPurityError(RuntimeError):
    pass


class OrganCausalBus:
    """Collect signals and reduce proposals without applying runtime mutations."""

    def __init__(
        self,
        *,
        arm: AblationArm,
        adapters: Sequence[OrganAdapter] = (),
        intent_recorder: IntentRecorder | None = None,
    ) -> None:
        self.arm = AblationArm(arm)
        self._adapters = tuple(sorted(adapters, key=_adapter_sort_key))
        self._intent_recorder = intent_recorder

    def decide(
        self,
        identity: StepIdentity,
        phase: Phase,
        context: Mapping[str, Any] | None = None,
    ) -> BusDecision:
        phase = Phase(phase)
        frozen_context = _freeze_json(dict(context or {}), path="context")
        if not isinstance(frozen_context, Mapping):
            raise TypeError("context must be a JSON-compatible mapping")
        if _contains_nonfinite(frozen_context):
            return self._record(
                BusDecision(
                    identity=identity,
                    phase=phase,
                    arm=self.arm,
                    rejected=(
                        Rejection("context", "causal_bus", "nonfinite_context"),
                    ),
                    blocked=True,
                    context=frozen_context,
                )
            )
        if self.arm is AblationArm.CONTROL:
            return BusDecision(
                identity=identity,
                phase=phase,
                arm=self.arm,
                context=frozen_context,
            )

        signals: list[Signal] = []
        for adapter in self._adapters:
            try:
                observed = tuple(
                    self._invoke_adapter(
                        adapter,
                        "observe",
                        identity,
                        phase,
                        frozen_context,
                    )
                )
            except Exception as exc:
                reason = (
                    "adapter_state_mutation"
                    if isinstance(exc, _AdapterPurityError)
                    else f"adapter_observe_error:{type(exc).__name__}"
                )
                return self._record(
                    BusDecision(
                        identity=identity,
                        phase=phase,
                        arm=self.arm,
                        signals=tuple(signals),
                        rejected=(
                            Rejection(
                                str(adapter.adapter_id),
                                str(adapter.adapter_id),
                                reason,
                            ),
                        ),
                        blocked=True,
                        context=frozen_context,
                    )
                )
            for signal in observed:
                if not isinstance(signal, Signal):
                    return self._record(
                        BusDecision(
                            identity=identity,
                            phase=phase,
                            arm=self.arm,
                            signals=tuple(signals),
                            rejected=(
                                Rejection(
                                    str(adapter.adapter_id),
                                    str(adapter.adapter_id),
                                    "invalid_signal_type",
                                ),
                            ),
                            blocked=True,
                            context=frozen_context,
                        )
                    )
                if signal.source not in _adapter_sources(adapter):
                    return self._record(
                        BusDecision(
                            identity=identity,
                            phase=phase,
                            arm=self.arm,
                            signals=tuple(signals),
                            rejected=(
                                Rejection(
                                    signal.signal_id,
                                    signal.source,
                                    "signal_source_mismatch",
                                ),
                            ),
                            blocked=True,
                            context=frozen_context,
                        )
                    )
                signals.append(signal)

        signal_rejection = self._validate_signals(identity, phase, signals)
        if signal_rejection is not None:
            return self._record(
                BusDecision(
                    identity=identity,
                    phase=phase,
                    arm=self.arm,
                    signals=tuple(signals),
                    rejected=(signal_rejection,),
                    blocked=True,
                    context=frozen_context,
                )
            )

        proposals: list[Intervention] = []
        proposal_rejections: list[Rejection] = []
        frozen_signals = tuple(
            sorted(signals, key=lambda item: (item.source, item.signal_id))
        )
        signal_ids = {item.signal_id for item in frozen_signals}
        for adapter in self._adapters:
            try:
                proposed = tuple(
                    self._invoke_adapter(
                        adapter,
                        "propose",
                        identity,
                        phase,
                        frozen_signals,
                        frozen_context,
                    )
                )
            except Exception as exc:
                reason = (
                    "adapter_state_mutation"
                    if isinstance(exc, _AdapterPurityError)
                    else f"adapter_propose_error:{type(exc).__name__}"
                )
                return self._record(
                    BusDecision(
                        identity=identity,
                        phase=phase,
                        arm=self.arm,
                        signals=frozen_signals,
                        rejected=(
                            Rejection(
                                str(adapter.adapter_id),
                                str(adapter.adapter_id),
                                reason,
                            ),
                        ),
                        blocked=True,
                        context=frozen_context,
                    )
                )
            for intervention in proposed:
                if not isinstance(intervention, Intervention):
                    return self._record(
                        BusDecision(
                            identity=identity,
                            phase=phase,
                            arm=self.arm,
                            signals=frozen_signals,
                            rejected=(
                                Rejection(
                                    str(adapter.adapter_id),
                                    str(adapter.adapter_id),
                                    "invalid_intervention_type",
                                ),
                            ),
                            blocked=True,
                            context=frozen_context,
                        )
                    )
                if intervention.source not in _adapter_sources(adapter):
                    proposal_rejections.append(
                        Rejection(
                            intervention.intervention_id,
                            intervention.source,
                            "intervention_source_mismatch",
                        )
                    )
                    continue
                if not set(intervention.evidence_ids).issubset(signal_ids):
                    proposal_rejections.append(
                        Rejection(
                            intervention.intervention_id,
                            intervention.source,
                            "unknown_evidence_id",
                        )
                    )
                    continue
                proposals.append(intervention)

        accepted, rejected = self._reduce(identity, phase, proposals)
        rejected = tuple(
            sorted(
                (*proposal_rejections, *rejected),
                key=lambda item: (item.source, item.item_id, item.reason),
            )
        )
        return self._record(
            BusDecision(
                identity=identity,
                phase=phase,
                arm=self.arm,
                signals=frozen_signals,
                accepted=accepted,
                rejected=rejected,
                context=frozen_context,
            )
        )

    def feedback(self, outcome: StepOutcome) -> tuple[Rejection, ...]:
        if self.arm is AblationArm.CONTROL:
            return ()
        rejected: list[Rejection] = []
        for adapter in self._adapters:
            try:
                self._invoke_adapter(adapter, "feedback", outcome)
            except Exception as exc:
                rejected.append(
                    Rejection(
                        str(adapter.adapter_id),
                        str(adapter.adapter_id),
                        (
                            "adapter_state_mutation"
                            if isinstance(exc, _AdapterPurityError)
                            else f"adapter_feedback_error:{type(exc).__name__}"
                        ),
                    )
                )
        return tuple(rejected)

    def record_intent(
        self,
        identity: StepIdentity,
        decisions: Sequence[BusDecision],
    ) -> Any:
        """Persist one aggregate intent for exactly one optimizer attempt."""
        frozen_decisions = tuple(decisions)
        if not frozen_decisions:
            raise ValueError("aggregate intent requires at least one decision")
        if any(decision.identity != identity for decision in frozen_decisions):
            raise ValueError("aggregate intent decisions must share identity")
        if any(decision.arm is not self.arm for decision in frozen_decisions):
            raise ValueError("aggregate intent decisions must share bus arm")
        phases = tuple(decision.phase for decision in frozen_decisions)
        if len(set(phases)) != len(phases):
            raise ValueError("aggregate intent phases must be unique")
        if self._intent_recorder is None:
            return None
        return self._intent_recorder.record_intent(identity, frozen_decisions)

    def _record(self, decision: BusDecision) -> BusDecision:
        """Compatibility helper: decisions are recorded only when aggregated."""
        return decision

    def _invoke_adapter(self, adapter, method_name: str, *args):
        before_state = _adapter_state_snapshot(adapter)
        before_fingerprint = _adapter_fingerprint(before_state)
        rng_snapshot = _snapshot_rng()
        try:
            result = getattr(adapter, method_name)(*args)
            if method_name in {"observe", "propose"}:
                result = tuple(result)
        except BaseException:
            _restore_adapter_state(adapter, before_state)
            raise
        finally:
            _restore_rng(rng_snapshot)
        if method_name == "feedback" and self.arm is AblationArm.APPLY:
            return result
        try:
            after_state = _adapter_state_snapshot(adapter)
        except BaseException as exc:
            _restore_adapter_state(adapter, before_state)
            raise _AdapterPurityError(method_name) from exc
        if _adapter_fingerprint(after_state) != before_fingerprint:
            _restore_adapter_state(adapter, before_state)
            raise _AdapterPurityError(method_name)
        return result

    @staticmethod
    def _validate_signals(
        identity: StepIdentity,
        phase: Phase,
        signals: Sequence[Signal],
    ) -> Rejection | None:
        seen: set[str] = set()
        for signal in signals:
            if signal.signal_id in seen:
                return Rejection(
                    signal.signal_id, signal.source, "duplicate_signal"
                )
            seen.add(signal.signal_id)
            if signal.phase is not phase:
                return Rejection(
                    signal.signal_id, signal.source, "signal_phase_mismatch"
                )
            if signal.step_key != identity.key:
                return Rejection(
                    signal.signal_id, signal.source, "signal_step_mismatch"
                )
            if _contains_nonfinite(signal.value):
                return Rejection(
                    signal.signal_id, signal.source, "nonfinite_signal"
                )
        return None

    @staticmethod
    def _reduce(
        identity: StepIdentity,
        phase: Phase,
        proposals: Sequence[Intervention],
    ) -> tuple[tuple[Intervention, ...], tuple[Rejection, ...]]:
        valid: list[Intervention] = []
        rejected: list[Rejection] = []
        allowed = _ALLOWED_INTERVENTIONS[phase]
        for item in sorted(proposals, key=_intervention_sort_key):
            target = (
                item.target
                if isinstance(item.target, InterventionTarget)
                else None
            )
            operation = (
                item.operation
                if isinstance(item.operation, InterventionOperation)
                else None
            )
            if item.phase is not phase:
                rejected.append(
                    Rejection(item.intervention_id, item.source, "phase_mismatch")
                )
                continue
            if target is None or not any(pair[0] is target for pair in allowed):
                rejected.append(
                    Rejection(
                        item.intervention_id,
                        item.source,
                        "target_not_allowed",
                    )
                )
                continue
            if operation is None or (target, operation) not in allowed:
                rejected.append(
                    Rejection(
                        item.intervention_id,
                        item.source,
                        "operation_not_allowed",
                    )
                )
                continue
            if _contains_nonfinite(item.value):
                rejected.append(
                    Rejection(
                        item.intervention_id,
                        item.source,
                        "nonfinite_value",
                    )
                )
                continue
            domain_reason = OrganCausalBus._domain_rejection(item)
            if domain_reason is not None:
                rejected.append(
                    Rejection(
                        item.intervention_id,
                        item.source,
                        domain_reason,
                    )
                )
                continue
            if identity.optimizer_step < item.valid_from_step:
                rejected.append(
                    Rejection(
                        item.intervention_id,
                        item.source,
                        "not_yet_valid",
                    )
                )
                continue
            if identity.optimizer_step > item.valid_through_step:
                rejected.append(
                    Rejection(item.intervention_id, item.source, "expired")
                )
                continue
            valid.append(item)

        by_target: dict[tuple[str, str], list[Intervention]] = {}
        by_id: dict[str, list[Intervention]] = {}
        for item in valid:
            by_id.setdefault(item.intervention_id, []).append(item)
        duplicate_ids = {
            intervention_id
            for intervention_id, items in by_id.items()
            if len(items) > 1
        }
        for item in valid:
            if item.intervention_id in duplicate_ids:
                rejected.append(
                    Rejection(
                        item.intervention_id,
                        item.source,
                        "duplicate_intervention_id",
                    )
                )
                continue
            by_target.setdefault(item.target_key, []).append(item)

        accepted: list[Intervention] = []
        for items in by_target.values():
            if len(items) > 1:
                rejected.extend(
                    Rejection(item.intervention_id, item.source, "conflict")
                    for item in items
                )
            else:
                accepted.extend(items)
        accepted.sort(key=_intervention_sort_key)
        rejected.sort(key=lambda item: (item.source, item.item_id, item.reason))
        return tuple(accepted), tuple(rejected)

    @staticmethod
    def _domain_rejection(item: Intervention) -> str | None:
        target = item.target
        operation = item.operation
        value = item.value
        if target is InterventionTarget.LOSS_TERM:
            if item.subject not in {"aux", "ghost", "jepa", "spider"}:
                return "subject_not_allowed"
            if (
                operation is not InterventionOperation.SET_SCALE
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or float(value) < 0.0
            ):
                return "invalid_operation_value"
        elif target is InterventionTarget.UPDATE:
            if item.subject not in {"optimizer", "update", "all"}:
                return "subject_not_allowed"
            if operation is not InterventionOperation.SKIP or value is not True:
                return "invalid_operation_value"
        elif target is InterventionTarget.GRAD_CLIP:
            if item.subject not in {"global", "all"}:
                return "subject_not_allowed"
            if (
                operation is not InterventionOperation.SET_MAX_NORM
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or float(value) < 0.0
            ):
                return "invalid_operation_value"
        elif target is InterventionTarget.GRADIENT_GROUP:
            if item.subject != "all" and any(
                part in item.subject for part in ("..", "*", "[", "]")
            ):
                return "subject_not_exact"
            if (
                operation is not InterventionOperation.SCALE
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or float(value) < 0.0
            ):
                return "invalid_operation_value"
        elif target is InterventionTarget.STRUCTURAL_ACTION:
            if (
                operation is not InterventionOperation.QUEUE
                or not isinstance(value, Mapping)
            ):
                return "invalid_operation_value"
        return None


CausalBus = OrganCausalBus
Decision = BusDecision


__all__ = [
    "AblationArm",
    "BusDecision",
    "CausalBus",
    "Decision",
    "Intervention",
    "InterventionOperation",
    "InterventionTarget",
    "OrganAdapter",
    "OrganCausalBus",
    "Phase",
    "Rejection",
    "Signal",
    "StepIdentity",
    "StepOutcome",
]
