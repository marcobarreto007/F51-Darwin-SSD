# Three-Organ Cognition Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the structural foundation for exactly three universal cognitive organs, with immutable pulse events, zero-gated adapters, separate brain/organ identities, and a real shadow-equivalence probe against Darwin-Smol Native Dense V1.

**Architecture:** Add a focused `f51_darwin.cognition` package containing contracts, an append-only `CognitivePulse`, zero-gated adapters, and a shadow-only runtime. Integrate that runtime behind a new structural config version while preserving all legacy defaults and keeping the frozen backbone identity independent from organ packages.

**Tech Stack:** Python 3.12, PyTorch 2.13, frozen dataclasses, canonical SHA-256, pytest 9, PowerShell, existing DarwinX checkpoint and topology contracts.

## Global Constraints

- Work only in `C:\Users\marco\Desktop\F51-Darwin-SSD`.
- The first target is `workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt`, expected SHA-256 `9329b8d25fc23acde021161da7a3421e6e17ceea3e940e0298162f39b62bc01e`.
- Canonical cognitive width is exactly 512 for schema `three_organs_v1`.
- The only canonical organ IDs are `organ:memory:v1`, `organ:world_model:v1`, and `organ:executive:v1`.
- The decoder/backbone is the brain and is not counted as an organ.
- `CognitivePulse` is an event bus and may not alter hidden states, select actions, or write memory.
- Legacy configs remain exact defaults; cognition is disabled unless `cognitive_architecture_version=three_organs_v1`.
- Every residual gate starts at exactly zero and gate-zero logits must be exactly equal in the focused FP32 tests.
- Brain weights and buffers remain byte-identical; organ and adapter identities are separate.
- This plan does not implement memory, future prediction, trajectory scoring, training, online learning, or checkpoint publication.
- Do not launch `run247`, a trainer, or a canary worker.
- Do not edit or overwrite the published 1.7B checkpoint.
- Run the source-only `src\\scripts\\start_overnight_16b.ps1 -Canary` gate before the real shadow probe.
- Commit each task with explicit paths and without `--no-verify`.

---

## Scope Decomposition

The approved design contains five independently rejectable deliveries:

1. cognition foundation - this plan;
2. `UniversalMemory`;
3. `HierarchicalWorldModel`;
4. `UniversalExecutive`;
5. causal end-to-end integration and promotion.

Only the first delivery is planned here. The later plans must import the
contracts established here instead of redefining them.

## File Map

### New package

- `src/f51_darwin/cognition/__init__.py` - public exports only.
- `src/f51_darwin/cognition/contracts.py` - organ IDs, metadata, residual condition, pulse event.
- `src/f51_darwin/cognition/pulse.py` - append-only hash-chained event records.
- `src/f51_darwin/cognition/adapters.py` - canonical-width encoder and bounded zero-gated readout.
- `src/f51_darwin/cognition/runtime.py` - shadow execution and manifest.

### Existing files to modify

- `src/f51_darwin/darwin_x_core/config.py` - structural/operational cognition config and output events.
- `src/f51_darwin/darwin_x_core/model.py` - instantiate and call the shadow runtime.
- `src/f51_darwin/organism/checkpoint_root.py` - classify cognition tuning fields as operational.
- `src/f51_darwin/state_identity.py` - exclude organ config and tensors from frozen brain identity.

### New proof surface

- `src/scripts/probe_three_organ_shadow.py` - hash-bound, donor-free, no-save real checkpoint probe.

### Tests

- `src/tests/test_cognition_contracts.py`
- `src/tests/test_cognitive_pulse.py`
- `src/tests/test_cognitive_adapters.py`
- `src/tests/test_cognition_runtime.py`
- `src/tests/test_cognition_config_identity.py`
- `src/tests/test_cognition_model_integration.py`
- `src/tests/test_three_organ_shadow_probe.py`

### Evidence documentation

- `governance/docs/operacao/STATUS_ATUAL.md` - add only measured foundation results.
- `workspace/runtime/three_organs_v1/foundation-shadow.json` - ignored runtime artifact.

---

### Task 1: Canonical Organ and Event Contracts

**Files:**
- Create: `src/f51_darwin/cognition/__init__.py`
- Create: `src/f51_darwin/cognition/contracts.py`
- Create: `src/tests/test_cognition_contracts.py`

**Interfaces:**
- Produces: `OrganKind`, `CANONICAL_ORGAN_IDS`, `CognitiveForwardMetadata`, `ResidualCondition`, `CognitivePulseEvent`.
- Consumes: only `torch`, frozen dataclasses, and standard-library enums.

- [ ] **Step 1: Write the failing contract tests**

```python
from __future__ import annotations

import pytest
import torch

from f51_darwin.cognition.contracts import (
    CANONICAL_ORGAN_IDS,
    CognitiveForwardMetadata,
    CognitivePulseEvent,
    OrganKind,
    ResidualCondition,
)


def test_exactly_three_canonical_organs() -> None:
    assert CANONICAL_ORGAN_IDS == (
        "organ:memory:v1",
        "organ:world_model:v1",
        "organ:executive:v1",
    )
    assert tuple(item.value for item in OrganKind) == CANONICAL_ORGAN_IDS


def test_residual_condition_is_shape_strict() -> None:
    condition = ResidualCondition(
        condition_id="memory-001",
        organ=OrganKind.MEMORY,
        values=torch.zeros(2, 3, 512),
        positions=torch.tensor([[0, 2, 4], [1, 3, 5]]),
    )
    assert condition.batch_size == 2
    assert condition.count == 3

    with pytest.raises(ValueError, match="organ_width=512"):
        ResidualCondition(
            condition_id="bad",
            organ=OrganKind.MEMORY,
            values=torch.zeros(1, 1, 16),
            positions=torch.zeros(1, 1, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="Executive"):
        ResidualCondition(
            condition_id="bad",
            organ=OrganKind.EXECUTIVE,
            values=torch.zeros(1, 1, 512),
            positions=torch.zeros(1, 1, dtype=torch.long),
        )


def test_metadata_and_pulse_reject_invalid_identity() -> None:
    metadata = CognitiveForwardMetadata(
        step_id=7,
        checkpoint_id="sha256:" + "a" * 64,
        context_digest="sha256:" + "b" * 64,
    )
    event = CognitivePulseEvent.shadow(metadata)
    assert event.mode == "shadow"
    assert event.selected_memory_ids == ()
    assert event.candidate_ids == ()

    with pytest.raises(ValueError, match="step_id"):
        CognitiveForwardMetadata(
            step_id=-1,
            checkpoint_id="sha256:" + "a" * 64,
            context_digest="sha256:" + "b" * 64,
        )
```

- [ ] **Step 2: Run the test and confirm the missing module failure**

Run:

```powershell
python -m pytest src/tests/test_cognition_contracts.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'f51_darwin.cognition'`.

- [ ] **Step 3: Implement the complete contract module**

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch


COGNITIVE_ARCHITECTURE_VERSION = "three_organs_v1"
COGNITIVE_ORGAN_WIDTH = 512
PULSE_SCHEMA = "darwin-cognitive-pulse-v1"


class OrganKind(StrEnum):
    MEMORY = "organ:memory:v1"
    WORLD_MODEL = "organ:world_model:v1"
    EXECUTIVE = "organ:executive:v1"


CANONICAL_ORGAN_IDS = tuple(item.value for item in OrganKind)


def _require_digest(name: str, value: str) -> None:
    prefix = "sha256:"
    suffix = value.removeprefix(prefix)
    if not value.startswith(prefix) or len(suffix) != 64:
        raise ValueError(f"{name} must be sha256:<64 lowercase hex>")
    if any(char not in "0123456789abcdef" for char in suffix):
        raise ValueError(f"{name} must be sha256:<64 lowercase hex>")


@dataclass(frozen=True)
class CognitiveForwardMetadata:
    step_id: int
    checkpoint_id: str
    context_digest: str

    def __post_init__(self) -> None:
        if self.step_id < 0:
            raise ValueError("step_id must be non-negative")
        _require_digest("checkpoint_id", self.checkpoint_id)
        _require_digest("context_digest", self.context_digest)


@dataclass(frozen=True)
class ResidualCondition:
    condition_id: str
    organ: OrganKind
    values: torch.Tensor
    positions: torch.Tensor

    def __post_init__(self) -> None:
        if not self.condition_id:
            raise ValueError("condition_id must be non-empty")
        if self.organ is OrganKind.EXECUTIVE:
            raise ValueError("Executive may authorize but not inject residuals")
        if self.values.ndim != 3 or self.values.shape[-1] != COGNITIVE_ORGAN_WIDTH:
            raise ValueError("values must have shape [batch, count, organ_width=512]")
        if self.positions.ndim != 2:
            raise ValueError("positions must have shape [batch, count]")
        if self.positions.dtype != torch.long:
            raise ValueError("positions must use torch.long")
        if tuple(self.positions.shape) != tuple(self.values.shape[:2]):
            raise ValueError("positions and values must share batch/count")
        if not bool(torch.isfinite(self.values).all()):
            raise ValueError("condition values must be finite")

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

    @property
    def count(self) -> int:
        return int(self.values.shape[1])


@dataclass(frozen=True)
class CognitivePulseEvent:
    schema: str
    mode: str
    step_id: int
    checkpoint_id: str
    context_digest: str
    selected_memory_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    selected_candidate_id: str | None
    prediction_error: float | None
    uncertainty: float | None
    compute_spent: int

    def __post_init__(self) -> None:
        if self.schema != PULSE_SCHEMA:
            raise ValueError(f"unsupported pulse schema: {self.schema}")
        if self.mode not in {"shadow", "active"}:
            raise ValueError("pulse mode must be shadow or active")
        if self.step_id < 0 or self.compute_spent < 0:
            raise ValueError("pulse counters must be non-negative")
        _require_digest("checkpoint_id", self.checkpoint_id)
        _require_digest("context_digest", self.context_digest)
        for name, value in (
            ("prediction_error", self.prediction_error),
            ("uncertainty", self.uncertainty),
        ):
            if value is not None and not float("-inf") < float(value) < float("inf"):
                raise ValueError(f"{name} must be finite")

    @classmethod
    def shadow(cls, metadata: CognitiveForwardMetadata) -> "CognitivePulseEvent":
        return cls(
            schema=PULSE_SCHEMA,
            mode="shadow",
            step_id=metadata.step_id,
            checkpoint_id=metadata.checkpoint_id,
            context_digest=metadata.context_digest,
            selected_memory_ids=(),
            candidate_ids=(),
            selected_candidate_id=None,
            prediction_error=None,
            uncertainty=None,
            compute_spent=0,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mode": self.mode,
            "step_id": self.step_id,
            "checkpoint_id": self.checkpoint_id,
            "context_digest": self.context_digest,
            "selected_memory_ids": list(self.selected_memory_ids),
            "candidate_ids": list(self.candidate_ids),
            "selected_candidate_id": self.selected_candidate_id,
            "prediction_error": self.prediction_error,
            "uncertainty": self.uncertainty,
            "compute_spent": self.compute_spent,
        }
```

Create `src/f51_darwin/cognition/__init__.py`:

```python
from .contracts import (
    CANONICAL_ORGAN_IDS,
    COGNITIVE_ARCHITECTURE_VERSION,
    COGNITIVE_ORGAN_WIDTH,
    CognitiveForwardMetadata,
    CognitivePulseEvent,
    OrganKind,
    ResidualCondition,
)

__all__ = [
    "CANONICAL_ORGAN_IDS",
    "COGNITIVE_ARCHITECTURE_VERSION",
    "COGNITIVE_ORGAN_WIDTH",
    "CognitiveForwardMetadata",
    "CognitivePulseEvent",
    "OrganKind",
    "ResidualCondition",
]
```

- [ ] **Step 4: Run the focused tests**

Run:

```powershell
python -m pytest src/tests/test_cognition_contracts.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit the contract**

```powershell
git add -- `
  src/f51_darwin/cognition/__init__.py `
  src/f51_darwin/cognition/contracts.py `
  src/tests/test_cognition_contracts.py
git commit -m "feat(cognition): define universal organ contracts"
```

---

### Task 2: Immutable CognitivePulse Hash Chain

**Files:**
- Create: `src/f51_darwin/cognition/pulse.py`
- Create: `src/tests/test_cognitive_pulse.py`

**Interfaces:**
- Consumes: `CognitivePulseEvent`, `canonical_sha256`.
- Produces: `CognitivePulse`, `CognitivePulseRecord`, `append`, `state_dict`, `load_state_dict`.

- [ ] **Step 1: Write the failing pulse tests**

```python
from __future__ import annotations

import copy

import pytest

from f51_darwin.cognition.contracts import (
    CognitiveForwardMetadata,
    CognitivePulseEvent,
)
from f51_darwin.cognition.pulse import CognitivePulse


def _event(step: int) -> CognitivePulseEvent:
    return CognitivePulseEvent.shadow(
        CognitiveForwardMetadata(
            step_id=step,
            checkpoint_id="sha256:" + "a" * 64,
            context_digest="sha256:" + f"{step:064x}",
        )
    )


def test_pulse_is_append_only_and_hash_chained() -> None:
    pulse = CognitivePulse()
    first = pulse.append(_event(0))
    second = pulse.append(_event(1))
    assert first.previous_sha256 is None
    assert second.previous_sha256 == first.sha256
    assert pulse.head_sha256 == second.sha256

    restored = CognitivePulse.from_state_dict(pulse.state_dict())
    assert restored.state_dict() == pulse.state_dict()


def test_pulse_rejects_duplicate_or_reordered_steps() -> None:
    pulse = CognitivePulse()
    pulse.append(_event(2))
    with pytest.raises(ValueError, match="strictly increase"):
        pulse.append(_event(2))


def test_pulse_rejects_tampered_state() -> None:
    pulse = CognitivePulse()
    pulse.append(_event(0))
    payload = copy.deepcopy(pulse.state_dict())
    payload["records"][0]["event"]["context_digest"] = "sha256:" + "f" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        CognitivePulse.from_state_dict(payload)
```

- [ ] **Step 2: Run the test and confirm the missing module failure**

Run:

```powershell
python -m pytest src/tests/test_cognitive_pulse.py -q
```

Expected: collection fails because `f51_darwin.cognition.pulse` does not exist.

- [ ] **Step 3: Implement the pulse ledger**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from f51_darwin.hashing import canonical_sha256

from .contracts import CognitivePulseEvent


PULSE_STATE_SCHEMA = "darwin-cognitive-pulse-state-v1"


@dataclass(frozen=True)
class CognitivePulseRecord:
    event: CognitivePulseEvent
    previous_sha256: str | None
    sha256: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "event": self.event.to_payload(),
            "previous_sha256": self.previous_sha256,
            "sha256": self.sha256,
        }


def _event_from_payload(raw: Mapping[str, Any]) -> CognitivePulseEvent:
    return CognitivePulseEvent(
        schema=str(raw["schema"]),
        mode=str(raw["mode"]),
        step_id=int(raw["step_id"]),
        checkpoint_id=str(raw["checkpoint_id"]),
        context_digest=str(raw["context_digest"]),
        selected_memory_ids=tuple(raw["selected_memory_ids"]),
        candidate_ids=tuple(raw["candidate_ids"]),
        selected_candidate_id=raw.get("selected_candidate_id"),
        prediction_error=raw.get("prediction_error"),
        uncertainty=raw.get("uncertainty"),
        compute_spent=int(raw["compute_spent"]),
    )


class CognitivePulse:
    def __init__(self) -> None:
        self._records: list[CognitivePulseRecord] = []

    @property
    def records(self) -> tuple[CognitivePulseRecord, ...]:
        return tuple(self._records)

    @property
    def head_sha256(self) -> str | None:
        return self._records[-1].sha256 if self._records else None

    def append(self, event: CognitivePulseEvent) -> CognitivePulseRecord:
        if self._records and event.step_id <= self._records[-1].event.step_id:
            raise ValueError("CognitivePulse step_id must strictly increase")
        previous = self.head_sha256
        digest = canonical_sha256(
            {"event": event.to_payload(), "previous_sha256": previous}
        )
        record = CognitivePulseRecord(
            event=event,
            previous_sha256=previous,
            sha256=digest,
        )
        self._records.append(record)
        return record

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": PULSE_STATE_SCHEMA,
            "records": [record.to_payload() for record in self._records],
            "head_sha256": self.head_sha256,
        }

    @classmethod
    def from_state_dict(cls, raw: Mapping[str, Any]) -> "CognitivePulse":
        if raw.get("schema") != PULSE_STATE_SCHEMA:
            raise ValueError("unsupported CognitivePulse state schema")
        pulse = cls()
        for stored in raw.get("records", []):
            event = _event_from_payload(stored["event"])
            record = pulse.append(event)
            if record.previous_sha256 != stored.get("previous_sha256"):
                raise ValueError("CognitivePulse previous hash mismatch")
            if record.sha256 != stored.get("sha256"):
                raise ValueError("CognitivePulse record hash mismatch")
        if pulse.head_sha256 != raw.get("head_sha256"):
            raise ValueError("CognitivePulse head hash mismatch")
        return pulse
```

- [ ] **Step 4: Run the focused tests**

Run:

```powershell
python -m pytest src/tests/test_cognitive_pulse.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit the pulse**

```powershell
git add -- src/f51_darwin/cognition/pulse.py src/tests/test_cognitive_pulse.py
git commit -m "feat(cognition): add immutable cognitive pulse"
```

---

### Task 3: Canonical-Width Zero-Gated Adapters

**Files:**
- Create: `src/f51_darwin/cognition/adapters.py`
- Create: `src/tests/test_cognitive_adapters.py`

**Interfaces:**
- Consumes: `ResidualCondition`, canonical width 512.
- Produces: `ZeroGatedCognitiveAdapter.encode`, `apply_condition`, `effective_scale`.

- [ ] **Step 1: Write the failing adapter tests**

```python
from __future__ import annotations

import pytest
import torch

from f51_darwin.cognition.adapters import ZeroGatedCognitiveAdapter
from f51_darwin.cognition.contracts import OrganKind, ResidualCondition


def _condition(batch: int = 2) -> ResidualCondition:
    return ResidualCondition(
        condition_id="memory-001",
        organ=OrganKind.MEMORY,
        values=torch.randn(batch, 1, 512),
        positions=torch.tensor([[2], [1]], dtype=torch.long),
    )


def test_gate_zero_is_bit_exact() -> None:
    torch.manual_seed(51)
    adapter = ZeroGatedCognitiveAdapter(d_model=16, max_scale=0.15)
    hidden = torch.randn(2, 4, 16)
    result = adapter.apply_condition(hidden, _condition())
    assert torch.equal(result, hidden)
    assert adapter.gate.item() == 0.0


def test_nonzero_gate_changes_only_selected_positions_and_is_bounded() -> None:
    torch.manual_seed(51)
    adapter = ZeroGatedCognitiveAdapter(d_model=16, max_scale=0.15)
    hidden = torch.randn(2, 4, 16)
    condition = _condition()
    with torch.no_grad():
        adapter.gate.fill_(100.0)
    result = adapter.apply_condition(hidden, condition)
    delta = result - hidden
    assert torch.count_nonzero(delta[0, :2]).item() == 0
    assert torch.count_nonzero(delta[0, 3:]).item() == 0
    assert torch.count_nonzero(delta[1, :1]).item() == 0
    assert torch.count_nonzero(delta[1, 2:]).item() == 0
    for batch, position in ((0, 2), (1, 1)):
        assert delta[batch, position].norm().item() <= (
            0.15 * hidden[batch, position].norm().item() + 1e-6
        )


def test_adapter_rejects_duplicate_and_out_of_range_positions() -> None:
    adapter = ZeroGatedCognitiveAdapter(d_model=16, max_scale=0.15)
    hidden = torch.randn(1, 3, 16)
    with pytest.raises(ValueError, match="unique"):
        adapter.apply_condition(
            hidden,
            ResidualCondition(
                condition_id="dup",
                organ=OrganKind.MEMORY,
                values=torch.zeros(1, 2, 512),
                positions=torch.tensor([[1, 1]]),
            ),
        )
```

- [ ] **Step 2: Run the test and confirm the missing module failure**

Run:

```powershell
python -m pytest src/tests/test_cognitive_adapters.py -q
```

Expected: collection fails because `f51_darwin.cognition.adapters` does not exist.

- [ ] **Step 3: Implement the bounded adapter**

```python
from __future__ import annotations

import math

import torch
from torch import nn

from .contracts import COGNITIVE_ORGAN_WIDTH, ResidualCondition


class ZeroGatedCognitiveAdapter(nn.Module):
    def __init__(self, d_model: int, max_scale: float) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be positive")
        if not math.isfinite(max_scale) or not 0.0 <= max_scale <= 1.0:
            raise ValueError("max_scale must be finite and in [0, 1]")
        self.d_model = int(d_model)
        self.max_scale = float(max_scale)
        self.input_adapter = nn.Linear(
            self.d_model,
            COGNITIVE_ORGAN_WIDTH,
            bias=False,
        )
        self.output_adapter = nn.Linear(
            COGNITIVE_ORGAN_WIDTH,
            self.d_model,
            bias=False,
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError("hidden must have shape [batch, sequence, d_model]")
        if not bool(torch.isfinite(hidden).all()):
            raise ValueError("hidden must be finite")
        return self.input_adapter(hidden)

    def effective_scale(self) -> torch.Tensor:
        return self.max_scale * torch.tanh(self.gate)

    def apply_condition(
        self,
        hidden: torch.Tensor,
        condition: ResidualCondition,
    ) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError("hidden must have shape [batch, sequence, d_model]")
        if condition.batch_size != hidden.shape[0]:
            raise ValueError("condition batch does not match hidden batch")
        positions = condition.positions.to(device=hidden.device)
        if bool((positions < 0).any()) or bool((positions >= hidden.shape[1]).any()):
            raise ValueError("condition position is out of range")
        for row in positions.detach().cpu().tolist():
            if len(set(row)) != len(row):
                raise ValueError("condition positions must be unique per sample")

        values = condition.values.to(device=hidden.device, dtype=hidden.dtype)
        projected = self.output_adapter(values)
        gather_index = positions.unsqueeze(-1).expand(-1, -1, self.d_model)
        reference = hidden.gather(dim=1, index=gather_index)
        projected_norm = projected.float().norm(dim=-1, keepdim=True)
        reference_norm = reference.float().norm(dim=-1, keepdim=True)
        factor = (
            reference_norm / projected_norm.clamp_min(1e-6)
        ).clamp(max=1.0)
        bounded = projected * factor.to(dtype=projected.dtype)
        delta = self.effective_scale().to(
            device=hidden.device,
            dtype=hidden.dtype,
        ) * bounded
        if bool(self.gate.detach().eq(0).item()):
            return hidden
        result = hidden.clone()
        result.scatter_add_(dim=1, index=gather_index, src=delta)
        return result
```

- [ ] **Step 4: Fix the scalar bound assertion and run the tests**

Run:

```powershell
python -m pytest src/tests/test_cognitive_adapters.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit the adapters**

```powershell
git add -- src/f51_darwin/cognition/adapters.py src/tests/test_cognitive_adapters.py
git commit -m "feat(cognition): add zero-gated organ adapters"
```

---

### Task 4: Shadow-Only Three-Organ Runtime

**Files:**
- Create: `src/f51_darwin/cognition/runtime.py`
- Modify: `src/f51_darwin/cognition/__init__.py`
- Create: `src/tests/test_cognition_runtime.py`

**Interfaces:**
- Consumes: `CognitiveForwardMetadata`, two residual adapters, `CognitivePulse`.
- Produces: `CognitiveRuntime.observe_shadow`, `manifest`, `pulse_state_dict`.

- [ ] **Step 1: Write the failing runtime tests**

```python
from __future__ import annotations

import torch

from f51_darwin.cognition import (
    CANONICAL_ORGAN_IDS,
    CognitiveForwardMetadata,
    CognitiveRuntime,
)


def _metadata(step: int = 0) -> CognitiveForwardMetadata:
    return CognitiveForwardMetadata(
        step_id=step,
        checkpoint_id="sha256:" + "a" * 64,
        context_digest="sha256:" + "b" * 64,
    )


def test_shadow_runtime_executes_paths_without_changing_hidden() -> None:
    torch.manual_seed(51)
    runtime = CognitiveRuntime(d_model=16, max_scale=0.15)
    hidden = torch.randn(2, 4, 16)
    result, record = runtime.observe_shadow(hidden, _metadata())
    assert torch.equal(result, hidden)
    assert record.event.mode == "shadow"
    assert runtime.manifest()["organ_ids"] == list(CANONICAL_ORGAN_IDS)
    assert runtime.memory_adapter.gate.item() == 0.0
    assert runtime.world_model_adapter.gate.item() == 0.0


def test_shadow_runtime_pulse_steps_must_increase() -> None:
    runtime = CognitiveRuntime(d_model=16, max_scale=0.15)
    hidden = torch.randn(1, 2, 16)
    runtime.observe_shadow(hidden, _metadata(1))
    try:
        runtime.observe_shadow(hidden, _metadata(1))
    except ValueError as error:
        assert "strictly increase" in str(error)
    else:
        raise AssertionError("duplicate pulse step was accepted")
```

- [ ] **Step 2: Run the tests and confirm the missing runtime failure**

Run:

```powershell
python -m pytest src/tests/test_cognition_runtime.py -q
```

Expected: collection fails because `CognitiveRuntime` is not exported.

- [ ] **Step 3: Implement the shadow runtime**

```python
from __future__ import annotations

import torch
from torch import nn

from .adapters import ZeroGatedCognitiveAdapter
from .contracts import (
    CANONICAL_ORGAN_IDS,
    COGNITIVE_ARCHITECTURE_VERSION,
    COGNITIVE_ORGAN_WIDTH,
    CognitiveForwardMetadata,
    CognitivePulseEvent,
    OrganKind,
    ResidualCondition,
)
from .pulse import CognitivePulse, CognitivePulseRecord


class CognitiveRuntime(nn.Module):
    def __init__(self, d_model: int, max_scale: float) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.memory_adapter = ZeroGatedCognitiveAdapter(d_model, max_scale)
        self.world_model_adapter = ZeroGatedCognitiveAdapter(d_model, max_scale)
        self._pulse = CognitivePulse()

    def _zero_gate_path(
        self,
        hidden: torch.Tensor,
        adapter: ZeroGatedCognitiveAdapter,
        organ: OrganKind,
    ) -> torch.Tensor:
        last = hidden[:, -1:, :]
        values = adapter.encode(last)
        positions = torch.full(
            (hidden.shape[0], 1),
            hidden.shape[1] - 1,
            device=hidden.device,
            dtype=torch.long,
        )
        return adapter.apply_condition(
            hidden,
            ResidualCondition(
                condition_id=f"shadow:{organ.value}",
                organ=organ,
                values=values,
                positions=positions,
            ),
        )

    def observe_shadow(
        self,
        hidden: torch.Tensor,
        metadata: CognitiveForwardMetadata,
    ) -> tuple[torch.Tensor, CognitivePulseRecord]:
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError("hidden must have shape [batch, sequence, d_model]")
        result = self._zero_gate_path(
            hidden,
            self.memory_adapter,
            OrganKind.MEMORY,
        )
        result = self._zero_gate_path(
            result,
            self.world_model_adapter,
            OrganKind.WORLD_MODEL,
        )
        if not torch.equal(result, hidden):
            raise RuntimeError("shadow cognition changed hidden state")
        record = self._pulse.append(CognitivePulseEvent.shadow(metadata))
        return result, record

    def manifest(self) -> dict[str, object]:
        return {
            "architecture_version": COGNITIVE_ARCHITECTURE_VERSION,
            "organ_width": COGNITIVE_ORGAN_WIDTH,
            "organ_ids": list(CANONICAL_ORGAN_IDS),
            "influence_paths": [
                OrganKind.MEMORY.value,
                OrganKind.WORLD_MODEL.value,
            ],
            "executive_authority": "selection_only",
            "pulse_authority": "events_only",
        }

    def pulse_state_dict(self) -> dict[str, object]:
        return self._pulse.state_dict()
```

Create `src/f51_darwin/cognition/__init__.py`:

```python
from .contracts import (
    CANONICAL_ORGAN_IDS,
    COGNITIVE_ARCHITECTURE_VERSION,
    COGNITIVE_ORGAN_WIDTH,
    CognitiveForwardMetadata,
    CognitivePulseEvent,
    OrganKind,
    ResidualCondition,
)
from .runtime import CognitiveRuntime

__all__ = [
    "CANONICAL_ORGAN_IDS",
    "COGNITIVE_ARCHITECTURE_VERSION",
    "COGNITIVE_ORGAN_WIDTH",
    "CognitiveForwardMetadata",
    "CognitivePulseEvent",
    "CognitiveRuntime",
    "OrganKind",
    "ResidualCondition",
]
```

- [ ] **Step 4: Run all foundation package tests**

Run:

```powershell
python -m pytest `
  src/tests/test_cognition_contracts.py `
  src/tests/test_cognitive_pulse.py `
  src/tests/test_cognitive_adapters.py `
  src/tests/test_cognition_runtime.py -q
```

Expected: `11 passed`.

- [ ] **Step 5: Commit the runtime**

```powershell
git add -- `
  src/f51_darwin/cognition/__init__.py `
  src/f51_darwin/cognition/runtime.py `
  src/tests/test_cognition_runtime.py
git commit -m "feat(cognition): add three-organ shadow runtime"
```

---

### Task 5: Structural Config and Separate Brain Identity

**Files:**
- Modify: `src/f51_darwin/darwin_x_core/config.py`
- Modify: `src/f51_darwin/organism/checkpoint_root.py`
- Modify: `src/f51_darwin/state_identity.py`
- Create: `src/tests/test_cognition_config_identity.py`

**Interfaces:**
- Produces config fields `cognitive_architecture_version`, `cognitive_shadow_enabled`, `cognitive_pulse_enabled`, `cognitive_organ_width`, `cognitive_residual_max_scale`.
- Preserves `backbone_identity` when only organ configuration/state changes.

- [ ] **Step 1: Write config and identity tests**

```python
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.organism.checkpoint_root import (
    model_config_identity,
    structural_config_identity,
)
from f51_darwin.state_identity import backbone_identity


def _tiny() -> DarwinXConfig:
    return DarwinXConfig(
        model_name="cognition-config-test",
        vocab_size=32,
        context_length=8,
        inference_context_length=8,
        d_model=16,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        fine_experts=2,
        shared_experts=1,
        experts_per_token=1,
        fine_expert_hidden_dim=8,
        shared_expert_hidden_dim=8,
        mtp_depth=0,
        heartbeat_enabled=False,
        spider_sense_enabled=False,
        ghost_enabled=False,
    )


def test_legacy_config_keeps_cognition_disabled() -> None:
    config = _tiny()
    assert config.cognitive_architecture_version == "disabled"
    assert config.cognitive_shadow_enabled is False
    assert config.cognitive_pulse_enabled is False


def test_disabled_defaults_preserve_pre_cognition_identities() -> None:
    config = _tiny()
    historical = dict(config.__dict__)
    for key in (
        "cognitive_architecture_version",
        "cognitive_shadow_enabled",
        "cognitive_pulse_enabled",
        "cognitive_organ_width",
        "cognitive_residual_max_scale",
    ):
        historical.pop(key)
    assert model_config_identity(config) == model_config_identity(historical)
    assert structural_config_identity(config) == structural_config_identity(
        historical
    )


def test_three_organ_v1_requires_width_512() -> None:
    with pytest.raises(ValueError, match="organ_width=512"):
        replace(
            _tiny(),
            cognitive_architecture_version="three_organs_v1",
            cognitive_organ_width=256,
        )


def test_shadow_toggle_is_operational_but_version_is_structural() -> None:
    base = _tiny()
    enabled = replace(
        base,
        cognitive_architecture_version="three_organs_v1",
        cognitive_shadow_enabled=True,
        cognitive_pulse_enabled=True,
    )
    same_structure = replace(enabled, cognitive_shadow_enabled=False)
    assert structural_config_identity(enabled) == structural_config_identity(
        same_structure
    )
    assert structural_config_identity(base) != structural_config_identity(enabled)


def test_brain_identity_ignores_organ_config_and_tensors() -> None:
    state = {
        "token_embedding.weight": torch.arange(32, dtype=torch.float32).reshape(8, 4),
        "cognitive_runtime.memory_adapter.gate": torch.tensor(0.0),
    }
    base = _tiny()
    enabled = replace(
        base,
        cognitive_architecture_version="three_organs_v1",
        cognitive_shadow_enabled=True,
        cognitive_pulse_enabled=True,
    )
    assert backbone_identity(state, base) == backbone_identity(state, enabled)
```

- [ ] **Step 2: Run the tests and confirm missing config fields**

Run:

```powershell
python -m pytest src/tests/test_cognition_config_identity.py -q
```

Expected: failures report missing cognition attributes or unexpected constructor fields.

- [ ] **Step 3: Add the config fields and validation**

Insert into `DarwinXConfig` before the legacy organ toggles:

```python
    cognitive_architecture_version: str = "disabled"
    cognitive_shadow_enabled: bool = False
    cognitive_pulse_enabled: bool = False
    cognitive_organ_width: int = 512
    cognitive_residual_max_scale: float = 0.15
```

Insert into `__post_init__`:

```python
        if self.cognitive_architecture_version not in {
            "disabled",
            "three_organs_v1",
        }:
            raise ValueError("unsupported cognitive_architecture_version")
        if (
            self.cognitive_architecture_version == "three_organs_v1"
            and self.cognitive_organ_width != 512
        ):
            raise ValueError("three_organs_v1 requires organ_width=512")
        if (
            self.cognitive_shadow_enabled or self.cognitive_pulse_enabled
        ) and self.cognitive_architecture_version != "three_organs_v1":
            raise ValueError(
                "cognitive shadow/pulse require three_organs_v1"
            )
        if (
            not math.isfinite(self.cognitive_residual_max_scale)
            or not 0.0 <= self.cognitive_residual_max_scale <= 1.0
        ):
            raise ValueError(
                "cognitive_residual_max_scale must be finite and in [0, 1]"
            )
```

Add operational fields in `checkpoint_root.py`:

```python
    "cognitive_shadow_enabled",
    "cognitive_pulse_enabled",
    "cognitive_residual_max_scale",
```

Do not add `cognitive_architecture_version` or `cognitive_organ_width`; they
change module structure.

- [ ] **Step 4: Preserve historical config IDs and separate brain identity**

In `checkpoint_root.py`, add:

```python
_COGNITION_CONFIG_FIELDS = frozenset(
    {
        "cognitive_architecture_version",
        "cognitive_shadow_enabled",
        "cognitive_pulse_enabled",
        "cognitive_organ_width",
        "cognitive_residual_max_scale",
    }
)


def _identity_config(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    if result.get("cognitive_architecture_version", "disabled") == "disabled":
        for key in _COGNITION_CONFIG_FIELDS:
            result.pop(key, None)
    return result
```

Change `model_config_identity` to encode:

```python
    encoded = json.dumps(
        _identity_config(normalized_model_config(raw)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
```

Change `structural_config_identity` to start with:

```python
    normalized = _identity_config(normalized_model_config(raw))
```

This preserves the exact pre-cognition identity when the feature is disabled,
while enabled V1 configs remain structurally distinct.

In `state_identity.py`, add:

```python
_THREE_ORGAN_CONFIG_KEYS = frozenset(
    {
        "cognitive_architecture_version",
        "cognitive_shadow_enabled",
        "cognitive_pulse_enabled",
        "cognitive_organ_width",
        "cognitive_residual_max_scale",
    }
)
```

Extend `_TRAINING_ONLY_CONFIG_KEYS`:

```python
_TRAINING_ONLY_CONFIG_KEYS = frozenset(
    {
        "loss_semantics_version",
        *_CAUSAL_COGNITIVE_CONFIG_KEYS,
        *_THREE_ORGAN_CONFIG_KEYS,
    }
)
```

The tensor prefix `cognitive_runtime.` is already outside `_CORE_PREFIXES`, so
no tensor filter change is allowed.

- [ ] **Step 5: Run focused and existing identity tests**

Run:

```powershell
python -m pytest `
  src/tests/test_cognition_config_identity.py `
  src/tests/test_checkpoint_root_policy.py `
  src/tests/test_state_identity.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit config and identity**

```powershell
git add -- `
  src/f51_darwin/darwin_x_core/config.py `
  src/f51_darwin/organism/checkpoint_root.py `
  src/f51_darwin/state_identity.py `
  src/tests/test_cognition_config_identity.py
git commit -m "feat(cognition): separate organ and brain identity"
```

---

### Task 6: DarwinX Shadow Integration

**Files:**
- Modify: `src/f51_darwin/darwin_x_core/config.py`
- Modify: `src/f51_darwin/darwin_x_core/model.py`
- Create: `src/tests/test_cognition_model_integration.py`

**Interfaces:**
- `DarwinXModel.forward(..., cognitive_metadata=None)` consumes `CognitiveForwardMetadata`.
- `DarwinXOutput.cognitive_pulse_events` produces an immutable tuple of JSON-safe event records.

- [ ] **Step 1: Write the failing integration tests**

```python
from __future__ import annotations

from dataclasses import replace

import torch

from f51_darwin.cognition import CognitiveForwardMetadata
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.state_identity import backbone_identity


def _config(**changes: object) -> DarwinXConfig:
    base = DarwinXConfig(
        model_name="cognition-model-test",
        vocab_size=32,
        context_length=8,
        inference_context_length=8,
        d_model=16,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        fine_experts=2,
        shared_experts=1,
        experts_per_token=1,
        fine_expert_hidden_dim=8,
        shared_expert_hidden_dim=8,
        mtp_depth=0,
        mtp_weight=0.0,
        jepa_weight=0.0,
        ghost_weight=0.0,
        ghost_enabled=False,
        spider_sense_enabled=False,
        heartbeat_enabled=False,
        nitro_enabled=False,
    )
    return replace(base, **changes)


def _metadata() -> CognitiveForwardMetadata:
    return CognitiveForwardMetadata(
        step_id=0,
        checkpoint_id="sha256:" + "a" * 64,
        context_digest="sha256:" + "b" * 64,
    )


def test_disabled_config_has_no_cognitive_state_or_events() -> None:
    model = DarwinXModel(_config()).eval()
    output = model(torch.tensor([[1, 2, 3]]), heartbeat=False)
    assert model.cognitive_runtime is None
    assert output.cognitive_pulse_events == ()
    assert not any(
        key.startswith("cognitive_runtime.") for key in model.state_dict()
    )


def test_shadow_config_loads_same_brain_and_preserves_logits() -> None:
    torch.manual_seed(51)
    base = DarwinXModel(_config()).eval()
    state = base.state_dict()
    base_identity = backbone_identity(state, base.config)
    batch = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        expected = base(batch, heartbeat=False).logits

    cognition = DarwinXModel(
        _config(
            cognitive_architecture_version="three_organs_v1",
            cognitive_shadow_enabled=True,
            cognitive_pulse_enabled=True,
        )
    ).eval()
    missing, unexpected = cognition.load_state_dict(state, strict=False)
    assert unexpected == []
    assert missing
    assert all(key.startswith("cognitive_runtime.") for key in missing)
    with torch.inference_mode():
        output = cognition(
            batch,
            heartbeat=False,
            cognitive_metadata=_metadata(),
        )
    torch.testing.assert_close(output.logits, expected, rtol=0.0, atol=0.0)
    assert len(output.cognitive_pulse_events) == 1
    assert output.cognitive_pulse_events[0]["event"]["mode"] == "shadow"
    assert backbone_identity(cognition.state_dict(), cognition.config) == base_identity


def test_shadow_requires_explicit_metadata() -> None:
    model = DarwinXModel(
        _config(
            cognitive_architecture_version="three_organs_v1",
            cognitive_shadow_enabled=True,
            cognitive_pulse_enabled=True,
        )
    )
    try:
        model(torch.tensor([[1, 2, 3]]), heartbeat=False)
    except ValueError as error:
        assert "cognitive_metadata" in str(error)
    else:
        raise AssertionError("shadow forward accepted missing metadata")
```

- [ ] **Step 2: Run the tests and confirm integration failures**

Run:

```powershell
python -m pytest src/tests/test_cognition_model_integration.py -q
```

Expected: failures mention missing `cognitive_runtime`, output field, or forward argument.

- [ ] **Step 3: Extend `DarwinXOutput`**

Add:

```python
    cognitive_pulse_events: tuple[dict[str, Any], ...] = field(
        default_factory=tuple
    )
```

- [ ] **Step 4: Instantiate the runtime after legacy module initialization**

At the end of `DarwinXModel.__init__`, after `self.apply(self._init_weights)`:

```python
        self.cognitive_runtime = None
        if config.cognitive_architecture_version == "three_organs_v1":
            from f51_darwin.cognition import CognitiveRuntime

            self.cognitive_runtime = CognitiveRuntime(
                d_model=config.d_model,
                max_scale=config.cognitive_residual_max_scale,
            )
```

Keep this before `activate_organism`. Do not change legacy initialization
order before `self.apply`.

- [ ] **Step 5: Add the shadow forward path**

Add the type-only import:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from f51_darwin.cognition import CognitiveForwardMetadata
```

Add the keyword argument:

```python
        cognitive_metadata: CognitiveForwardMetadata | None = None,
```

Immediately after `hidden = self.norm(x)`:

```python
        cognitive_pulse_events: tuple[dict[str, Any], ...] = ()
        if (
            self.cognitive_runtime is not None
            and self.config.cognitive_shadow_enabled
        ):
            if cognitive_metadata is None:
                raise ValueError(
                    "cognitive_metadata is required in cognitive shadow mode"
                )
            hidden, pulse_record = self.cognitive_runtime.observe_shadow(
                hidden,
                cognitive_metadata,
            )
            if self.config.cognitive_pulse_enabled:
                cognitive_pulse_events = (pulse_record.to_payload(),)
```

Add to the `DarwinXOutput` return:

```python
            cognitive_pulse_events=cognitive_pulse_events,
```

Add cognition to `topology_manifest()` without changing the legacy topology
version:

```python
            "cognition": (
                self.cognitive_runtime.manifest()
                if self.cognitive_runtime is not None
                else None
            ),
```

- [ ] **Step 6: Run model and regression tests**

Run:

```powershell
python -m pytest `
  src/tests/test_cognition_model_integration.py `
  src/tests/test_dense_darwin_model.py `
  src/tests/test_causal_cognitive_organs.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit model integration**

```powershell
git add -- `
  src/f51_darwin/darwin_x_core/config.py `
  src/f51_darwin/darwin_x_core/model.py `
  src/tests/test_cognition_model_integration.py
git commit -m "feat(cognition): integrate shadow runtime"
```

---

### Task 7: Real Hash-Bound Shadow Probe

**Files:**
- Create: `src/scripts/probe_three_organ_shadow.py`
- Create: `src/tests/test_three_organ_shadow_probe.py`

**Interfaces:**
- Consumes an explicit checkpoint path, manifest path, expected SHA-256, and token IDs.
- Produces `foundation-shadow.json` and exits nonzero on any identity, missing-key, logit, gate, pulse, or checkpoint mutation failure.

- [ ] **Step 1: Write parser and report-validation tests**

```python
from __future__ import annotations

import pytest

from scripts.probe_three_organ_shadow import (
    parse_token_ids,
    validate_missing_keys,
    validate_report,
)


def test_parse_token_ids_is_strict() -> None:
    assert parse_token_ids("1,2,3") == (1, 2, 3)
    with pytest.raises(ValueError, match="at least two"):
        parse_token_ids("1")
    with pytest.raises(ValueError, match="non-negative"):
        parse_token_ids("1,-2")


def test_missing_keys_are_cognition_only() -> None:
    validate_missing_keys(
        [
            "cognitive_runtime.memory_adapter.gate",
            "cognitive_runtime.world_model_adapter.gate",
        ],
        [],
    )
    with pytest.raises(ValueError, match="non-cognition"):
        validate_missing_keys(["norm.weight"], [])


def test_report_requires_exact_shadow_equivalence() -> None:
    report = {
        "schema": "darwin-three-organ-foundation-shadow-v1",
        "checkpoint_sha256": "a" * 64,
        "checkpoint_sha256_after": "a" * 64,
        "brain_identity_before": "brain",
        "brain_identity_after": "brain",
        "max_abs_logit_error": 0.0,
        "pulse_events": 1,
        "canonical_organ_count": 3,
        "all_gates_zero": True,
    }
    validate_report(report)
    report["max_abs_logit_error"] = 1e-6
    with pytest.raises(ValueError, match="logit"):
        validate_report(report)
```

- [ ] **Step 2: Run the tests and confirm the missing script failure**

Run:

```powershell
python -m pytest src/tests/test_three_organ_shadow_probe.py -q
```

Expected: collection fails because `scripts.probe_three_organ_shadow` does not exist.

- [ ] **Step 3: Implement the complete no-save probe**

Create `src/scripts/probe_three_organ_shadow.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch

from f51_darwin.cognition import (
    CANONICAL_ORGAN_IDS,
    CognitiveForwardMetadata,
)
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.hashing import (
    atomic_json_write,
    sha256_file,
    tensor_sha256,
)
from f51_darwin.state_identity import backbone_identity
from f51_darwin.transplant_16b.checkpoint import verify_shard_manifest


REPORT_SCHEMA = "darwin-three-organ-foundation-shadow-v1"


def parse_token_ids(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(","))
    except ValueError as error:
        raise ValueError("token ids must be comma-separated integers") from error
    if len(values) < 2:
        raise ValueError("at least two token ids are required")
    if any(value < 0 for value in values):
        raise ValueError("token ids must be non-negative")
    return values


def validate_missing_keys(
    missing: list[str],
    unexpected: list[str],
) -> None:
    if unexpected:
        raise ValueError(f"unexpected checkpoint keys: {unexpected}")
    if not missing:
        raise ValueError("cognitive shadow load exposed no new state")
    foreign = [
        key
        for key in missing
        if not key.startswith("cognitive_runtime.")
    ]
    if foreign:
        raise ValueError(f"non-cognition missing keys: {foreign}")


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("wrong shadow report schema")
    if report["checkpoint_sha256"] != report["checkpoint_sha256_after"]:
        raise ValueError("checkpoint changed during shadow probe")
    if report["brain_identity_before"] != report["brain_identity_after"]:
        raise ValueError("brain identity changed during shadow probe")
    if float(report["max_abs_logit_error"]) != 0.0:
        raise ValueError("shadow logit error is not exactly zero")
    if int(report["pulse_events"]) != 1:
        raise ValueError("shadow must emit exactly one pulse event")
    if int(report["canonical_organ_count"]) != 3:
        raise ValueError("shadow manifest must contain exactly three organs")
    if report["all_gates_zero"] is not True:
        raise ValueError("a cognitive gate is not exactly zero")


def build_bfloat16_model(config: DarwinXConfig) -> DarwinXModel:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return DarwinXModel(config)
    finally:
        torch.set_default_dtype(previous)


def place_for_inference(
    model: DarwinXModel,
    mode: str,
) -> tuple[DarwinXModel, torch.device]:
    if mode == "dual":
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("dual mode requires two CUDA devices")
        model.to(dtype=torch.bfloat16)
        if not model.enable_dual_gpu(gpu0=0, gpu1=1):
            raise RuntimeError("DarwinXModel refused dual-GPU placement")
        return model, torch.device("cuda:0")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda mode requires CUDA")
        device = torch.device("cuda:0")
    elif mode == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"unsupported device mode: {mode}")
    model.to(device=device, dtype=torch.bfloat16)
    return model, device


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hash-bound three-organ foundation shadow probe."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--token-ids", required=True)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "dual"),
        default="dual",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    checkpoint = Path(args.checkpoint).resolve()
    manifest = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()
    token_ids = parse_token_ids(args.token_ids)
    expected_sha256 = str(args.expected_sha256).lower()
    if len(expected_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_sha256
    ):
        raise ValueError("expected SHA-256 must be 64 lowercase hex chars")

    checkpoint_sha_before = sha256_file(checkpoint)
    if checkpoint_sha_before != expected_sha256:
        raise ValueError("checkpoint SHA-256 mismatch")
    verify_shard_manifest(checkpoint, manifest)
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if payload.get("version") != 9:
        raise ValueError("foundation shadow probe requires checkpoint v9")
    base_config = DarwinXConfig.from_mapping(payload["config"])
    if any(token_id >= base_config.vocab_size for token_id in token_ids):
        raise ValueError("token id exceeds checkpoint vocabulary")
    input_cpu = torch.tensor([token_ids], dtype=torch.long)

    base_model = build_bfloat16_model(base_config)
    base_model.load_state_dict(payload["model_state_dict"], strict=True)
    brain_before = backbone_identity(base_model.state_dict(), base_config)
    base_model, base_device = place_for_inference(base_model, args.device)
    with torch.inference_mode():
        base_logits = base_model(
            input_cpu.to(base_device),
            heartbeat=False,
        ).logits.cpu()
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cognitive_config = replace(
        base_config,
        cognitive_architecture_version="three_organs_v1",
        cognitive_shadow_enabled=True,
        cognitive_pulse_enabled=True,
        cognitive_organ_width=512,
    )
    cognitive_model = build_bfloat16_model(cognitive_config)
    missing, unexpected = cognitive_model.load_state_dict(
        payload["model_state_dict"],
        strict=False,
    )
    validate_missing_keys(list(missing), list(unexpected))
    brain_after = backbone_identity(
        cognitive_model.state_dict(),
        cognitive_config,
    )
    cognitive_model, cognitive_device = place_for_inference(
        cognitive_model,
        args.device,
    )
    metadata = CognitiveForwardMetadata(
        step_id=0,
        checkpoint_id="sha256:" + checkpoint_sha_before,
        context_digest="sha256:" + tensor_sha256(input_cpu),
    )
    with torch.inference_mode():
        output = cognitive_model(
            input_cpu.to(cognitive_device),
            heartbeat=False,
            cognitive_metadata=metadata,
        )
    cognitive_logits = output.logits.cpu()
    max_abs_error = float(
        (cognitive_logits.float() - base_logits.float()).abs().max()
    )
    runtime = cognitive_model.cognitive_runtime
    if runtime is None:
        raise RuntimeError("cognitive runtime was not constructed")
    report = {
        "schema": REPORT_SCHEMA,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha_before,
        "checkpoint_sha256_after": sha256_file(checkpoint),
        "brain_identity_before": brain_before,
        "brain_identity_after": brain_after,
        "max_abs_logit_error": max_abs_error,
        "pulse_events": len(output.cognitive_pulse_events),
        "pulse_head_sha256": runtime.pulse_state_dict()["head_sha256"],
        "canonical_organ_count": len(
            runtime.manifest()["organ_ids"]
        ),
        "canonical_organ_ids": list(CANONICAL_ORGAN_IDS),
        "all_gates_zero": bool(
            runtime.memory_adapter.gate.item() == 0.0
            and runtime.world_model_adapter.gate.item() == 0.0
        ),
        "missing_cognitive_keys": sorted(missing),
        "device_mode": args.device,
    }
    validate_report(report)
    atomic_json_write(report, output_path)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    print("THREE_ORGAN_FOUNDATION_SHADOW_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

The script has no checkpoint writer, optimizer, backward call, or training
launcher.

- [ ] **Step 5: Run helper tests**

Run:

```powershell
python -m pytest src/tests/test_three_organ_shadow_probe.py -q
```

Expected: `3 passed`.

- [ ] **Step 6: Commit the probe**

```powershell
git add -- `
  src/scripts/probe_three_organ_shadow.py `
  src/tests/test_three_organ_shadow_probe.py
git commit -m "test(cognition): add hash-bound shadow probe"
```

---

### Task 8: Source Gate, Real Shadow Evidence, and Status

**Files:**
- Modify after measurement: `governance/docs/operacao/STATUS_ATUAL.md`
- Create runtime only: `workspace/runtime/three_organs_v1/foundation-shadow.json`

**Interfaces:**
- Consumes every earlier task.
- Produces source-test proof, real shadow report, and an honest status label.

- [ ] **Step 1: Run the full focused cognition suite**

Run:

```powershell
python -m pytest `
  src/tests/test_cognition_contracts.py `
  src/tests/test_cognitive_pulse.py `
  src/tests/test_cognitive_adapters.py `
  src/tests/test_cognition_runtime.py `
  src/tests/test_cognition_config_identity.py `
  src/tests/test_cognition_model_integration.py `
  src/tests/test_three_organ_shadow_probe.py -q
```

Expected: zero failures.

- [ ] **Step 2: Run impacted legacy regression tests**

Run:

```powershell
python -m pytest `
  src/tests/test_dense_darwin_model.py `
  src/tests/test_causal_cognitive_organs.py `
  src/tests/test_checkpoint_root_policy.py `
  src/tests/test_state_identity.py `
  src/tests/test_native_ttm_recall.py -q
```

Expected: zero failures.

- [ ] **Step 3: Run the repository source-only gate**

Run:

```powershell
powershell -ExecutionPolicy Bypass `
  -File src\\scripts\\start_overnight_16b.ps1 -Canary
```

Expected: source audit passes and no trainer/worker is launched. If the command
launches or any source gate is red, stop; do not run the real probe.

- [ ] **Step 4: Run the real checkpoint shadow probe**

Run:

```powershell
python src\\scripts\\probe_three_organ_shadow.py `
  --checkpoint workspace\03_CHECKPOINTS_1.7B_SMOL_DENSE_V1\organism_cycle_000.pt `
  --manifest workspace\03_CHECKPOINTS_1.7B_SMOL_DENSE_V1\organism_cycle_000.manifest.json `
  --expected-sha256 9329b8d25fc23acde021161da7a3421e6e17ceea3e940e0298162f39b62bc01e `
  --token-ids 1,2,3,4,5,6,7,8 `
  --device dual `
  --output workspace\runtime\three_organs_v1\foundation-shadow.json
```

Expected report fields:

```json
{
  "schema": "darwin-three-organ-foundation-shadow-v1",
  "checkpoint_sha256": "9329b8d25fc23acde021161da7a3421e6e17ceea3e940e0298162f39b62bc01e",
  "checkpoint_sha256_after": "9329b8d25fc23acde021161da7a3421e6e17ceea3e940e0298162f39b62bc01e",
  "max_abs_logit_error": 0.0,
  "pulse_events": 1,
  "canonical_organ_count": 3,
  "all_gates_zero": true
}
```

- [ ] **Step 5: Verify the saved artifact from disk**

Run:

```powershell
python -c "import json, pathlib; p=pathlib.Path(r'workspace/runtime/three_organs_v1/foundation-shadow.json'); d=json.loads(p.read_text(encoding='utf-8')); assert d['max_abs_logit_error']==0.0; assert d['checkpoint_sha256']==d['checkpoint_sha256_after']; assert d['brain_identity_before']==d['brain_identity_after']; assert d['canonical_organ_count']==3; print('FOUNDATION_SHADOW_OK')"
```

Expected: `FOUNDATION_SHADOW_OK`.

- [ ] **Step 6: Update current status with only measured claims**

Add a dated section to `governance/docs/operacao/STATUS_ATUAL.md` containing:

- commit range;
- tests executed and pass counts;
- checkpoint SHA before/after;
- brain identity before/after;
- exact max logit error;
- gate values;
- pulse record hash;
- device mode;
- label `foundation_shadow_only`.

State explicitly:

```text
This proves contracts, identity separation and a zero-impact shadow path.
It does not prove memory, prediction, planning, decision quality or learning.
```

- [ ] **Step 7: Review and commit the evidence documentation**

Run:

```powershell
git diff --check
git status --short
git diff -- governance/docs/operacao/STATUS_ATUAL.md
```

Then:

```powershell
git add -- governance/docs/operacao/STATUS_ATUAL.md
git commit -m "docs(cognition): publish foundation shadow evidence"
```

- [ ] **Step 8: Final verification**

Run:

```powershell
git status --short
git log --oneline -8
```

Expected: clean worktree and eight isolated foundation commits. The runtime
JSON remains ignored under `workspace/`.

---

## Foundation Completion Gate

The foundation passes only when all statements below are backed by fresh
output:

- exactly three canonical organ IDs;
- cognition disabled adds no parameters or output events;
- cognition schema is structural and tuning toggles are operational;
- brain identity is unchanged by organ config or organ state;
- both influence adapters execute with gates exactly zero;
- shadow hidden states and logits are exactly equal to base;
- one immutable hash-chained pulse event is emitted;
- checkpoint SHA is unchanged before/after;
- focused, impacted, and source-only gates are green;
- real artifact reload verifies from disk;
- no trainer, optimizer, backward, checkpoint writer, or cloud service ran.

Only after this gate passes should the `UniversalMemory` implementation plan
be written.
