# F51 Causal Organism v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** connect Soul, Decision Engine, Curiosity, GABA, Ghost Token, and Ghost Brain to the supported Darwin-X runtime through a lagged, checkpointed, causally ablated wake-shadow-apply-sleep loop.

**Architecture:** tensor organs operate from state frozen before a training attempt; observations become immutable frames only after the attempt; Soul proposes slow policy; Decision calibrates and vetoes; the causal bus is the only path to a next-step or cycle-boundary mutation. Checkpoint v8 retains its file format while `causal_cognitive_contract.version=2` strictly identifies all new organ state.

**Tech Stack:** Python 3.11+, PyTorch, dataclasses, canonical JSON/SHA-256, pytest, PowerShell, existing Darwin-X causal bus/ledger/checkpoint infrastructure.

**Design:** `governance/docs/superpowers/specs/2026-07-18-causal-organism-v2-design.md`

## Global Constraints

- Work only inside `C:\Users\marco\Desktop\F51-Darwin-SSD`.
- Never start `run247`, a concurrent trainer, or a launch canary.
- Do not modify `src/configs/darwin_x_100m.yaml`; its current 4K/32K/Heartbeat changes belong to Marco.
- Do not delete, select, merge, rename, or rewrite checkpoints, corpora, tokenizers, manifests, or history.
- Preserve v7 behavior exactly unless an explicit v8 cognitive-contract-v2 migration is enabled.
- An observation from step `t` may first mutate training at step `t+1`.
- Disabled, `CONTROL`, and `SHADOW` must be bitwise equal under paired RNG.
- External Ghost material remains quarantined until explicit approval.
- No organ may hold a mutable model or optimizer reference.
- Non-finite signals and interventions fail closed.
- No claim of improvement may rely on weighted total loss; fresh and heldout LM metrics decide promotion.
- Use explicit-path Conventional Commits and never `--no-verify`.

---

## File structure

New focused modules:

- `src/f51_darwin/organism/organ_frames.py`: immutable post-outcome frames, expiry, canonical digest, serialization.
- `src/f51_darwin/organism/organ_adapters.py`: pure Soul, Decision, Curiosity, GABA, and Ghost adapters.
- `src/f51_darwin/organism/executive_state.py`: checkpointable Soul policy and Decision calibration state.
- `src/tools/run_causal_organism_v2_ablation.py`: paired individual and 32-combination tiny harness.

Existing modules retain their responsibilities:

- `src/f51_darwin/gaba_inhibition.py`: bounded inhibitory tensor delta and deferred state commit.
- `src/f51_darwin/curiosity.py`: sample-local novelty memory.
- `src/f51_darwin/soul.py`: numerical slow policy state and backward-compatible narrative report.
- `src/f51_darwin/decision_engine.py`: deterministic calibrated approve/defer/veto controller.
- `src/f51_darwin/replay_buffer.py`: persistent priority used by the next replay sample.
- `src/f51_darwin/darwin_x_core/block.py`: apply frozen GABA delta without duplicate excitation.
- `src/f51_darwin/darwin_x_core/model.py`: deterministic Ghost probe and tensor-organ observations.
- `src/f51_darwin/darwin_x_core/config.py`: v2 toggles, bounds, and output observations.
- `src/f51_darwin/darwin_x_core/losses.py`: exact effective Ghost/JEPA/Spider/Aux loss terms.
- `src/f51_darwin/organism/causal_bus.py`: exact loss-subject allowlist and frame provenance.
- `src/f51_darwin/organism/causal_adapters.py`: execute exact allowlisted loss scales and gradient actions.
- `src/f51_darwin/organism/training.py`: temporal barrier, post-step observation, next-step application.
- `src/f51_darwin/organism/lifecycle.py`: next-boundary curiosity and protected-sleep proposals.
- `src/f51_darwin/organism/bootstrap.py`: construct and restore v2 organs/adapters.
- `src/f51_darwin/organism/checkpoint_mixin.py`: save complete cognitive-contract-v2 state.
- `src/f51_darwin/organism/checkpoint.py`: strict restore and tamper rejection.
- `src/f51_darwin/state_identity.py`: cognitive-contract-v2 identity.

---

### Task 1: Immutable temporal barrier

**Files:**

- Create: `src/f51_darwin/organism/organ_frames.py`
- Modify: `src/f51_darwin/organism/causal_bus.py`
- Test: `src/tests/test_temporal_organ_frames.py`

**Interfaces:**

- Consumes: `StepIdentity` and canonical JSON-compatible values from `causal_bus.py`.
- Produces:
  - `OrganObservationFrame.create(source, signals, valid_from_step, valid_through_step) -> OrganObservationFrame`
  - `OrganObservationFrame.validate_for(target: StepIdentity) -> None`
  - `OrganObservationFrame.to_state() -> dict[str, object]`
  - `OrganObservationFrame.from_state(state) -> OrganObservationFrame`

- [ ] **Step 1: Write failing creation, delay, expiry, finite, tamper, and roundtrip tests**

```python
from dataclasses import replace

import pytest

from f51_darwin.organism.causal_bus import StepIdentity
from f51_darwin.organism.organ_frames import OrganObservationFrame


def identity(step: int) -> StepIdentity:
    return StepIdentity(
        run_id="run-v2",
        cycle=3,
        optimizer_step=step,
        accumulation_window=0,
        batch_digest=f"batch-{step}",
        rng_digest=f"rng-{step}",
        base_checkpoint_id="base-v2",
        training_contract_id="contract-v2",
    )


def test_frame_is_valid_only_after_source_step() -> None:
    frame = OrganObservationFrame.create(
        source=identity(10),
        signals={"fresh_lm": 2.5, "heldout_lm": 2.7},
        valid_from_step=11,
        valid_through_step=11,
    )
    with pytest.raises(ValueError, match="temporal_barrier"):
        frame.validate_for(identity(10))
    frame.validate_for(identity(11))
    with pytest.raises(ValueError, match="expired"):
        frame.validate_for(identity(12))


def test_frame_rejects_nonfinite_and_tamper() -> None:
    with pytest.raises(ValueError, match="finite"):
        OrganObservationFrame.create(
            source=identity(1),
            signals={"fresh_lm": float("nan")},
            valid_from_step=2,
            valid_through_step=2,
        )
    frame = OrganObservationFrame.create(
        source=identity(1),
        signals={"fresh_lm": 3.0},
        valid_from_step=2,
        valid_through_step=2,
    )
    state = frame.to_state()
    state["signals"]["fresh_lm"] = 1.0
    with pytest.raises(ValueError, match="digest"):
        OrganObservationFrame.from_state(state)


def test_frame_roundtrip_is_exact_and_immutable() -> None:
    frame = OrganObservationFrame.create(
        source=identity(7),
        signals={"per_sample_novelty": (0.1, 0.9)},
        valid_from_step=8,
        valid_through_step=9,
    )
    restored = OrganObservationFrame.from_state(frame.to_state())
    assert restored == frame
    with pytest.raises(TypeError):
        restored.signals["x"] = 1
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```powershell
python -m pytest -q -p no:cacheprovider src\\tests\\test_temporal_organ_frames.py
```

Expected: collection fails because `organ_frames` does not exist.

- [ ] **Step 3: Implement the immutable frame**

`organ_frames.py` must define this public shape:

```python
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Mapping
import json
import math

from .causal_bus import StepIdentity


@dataclass(frozen=True)
class OrganObservationFrame:
    source: StepIdentity
    signals: Mapping[str, Any]
    valid_from_step: int
    valid_through_step: int
    digest: str

    @classmethod
    def create(
        cls,
        *,
        source: StepIdentity,
        signals: Mapping[str, Any],
        valid_from_step: int,
        valid_through_step: int,
    ) -> "OrganObservationFrame":
        if valid_from_step <= source.optimizer_step:
            raise ValueError("temporal_barrier requires a later target step")
        if valid_through_step < valid_from_step:
            raise ValueError("valid_through_step precedes valid_from_step")
        frozen = _freeze_finite(signals, path="signals")
        payload = _payload(source, frozen, valid_from_step, valid_through_step)
        digest = "organ-frame-v1:" + sha256(_canonical(payload)).hexdigest()
        return cls(source, frozen, valid_from_step, valid_through_step, digest)

    def validate_for(self, target: StepIdentity) -> None:
        if target.training_contract_id != self.source.training_contract_id:
            raise ValueError("training_contract_mismatch")
        if target.optimizer_step < self.valid_from_step:
            raise ValueError("temporal_barrier")
        if target.optimizer_step > self.valid_through_step:
            raise ValueError("expired")
        expected = self.from_state(self.to_state()).digest
        if expected != self.digest:
            raise ValueError("digest mismatch")

    def to_state(self) -> dict[str, object]:
        return _thaw(_payload(
            self.source,
            self.signals,
            self.valid_from_step,
            self.valid_through_step,
        )) | {"digest": self.digest}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "OrganObservationFrame":
        source = StepIdentity.from_state(state["source"])
        frame = cls.create(
            source=source,
            signals=state["signals"],
            valid_from_step=int(state["valid_from_step"]),
            valid_through_step=int(state["valid_through_step"]),
        )
        if state.get("digest") != frame.digest:
            raise ValueError("digest mismatch")
        return frame
```

Implement `_freeze_finite`, `_thaw`, `_payload`, and `_canonical` in the same
module. `_freeze_finite` must reject booleans masquerading as numeric evidence,
all non-finite floats, non-string mapping keys, tensors, arbitrary objects, and
mutable nested containers.

Add `StepIdentity.to_state()` and `StepIdentity.from_state()` to
`causal_bus.py` using every identity field already present in the dataclass.

- [ ] **Step 4: Run the focused and existing bus/ledger tests**

Run:

```powershell
python -m pytest -q -p no:cacheprovider src\\tests\\test_temporal_organ_frames.py src\\tests\\test_organ_causal_bus.py src\\tests\\test_causal_ledger.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add -- src/f51_darwin/organism/organ_frames.py src/f51_darwin/organism/causal_bus.py src/tests/test_temporal_organ_frames.py
git commit -m "feat(organism): add temporal observation barrier"
```

---

### Task 2: Correct GABA into a bounded deferred inhibitory delta

**Files:**

- Modify: `src/f51_darwin/gaba_inhibition.py`
- Modify: `src/f51_darwin/darwin_x_core/block.py`
- Modify: `src/f51_darwin/darwin_x_core/config.py`
- Modify: `src/f51_darwin/darwin_x_core/model.py`
- Test: `src/tests/test_gaba_causal.py`

**Interfaces:**

- Produces:
  - `GABAergicLayer.forward(x, *, mutate_state=False) -> (delta, observation)`
  - `GABAergicLayer.commit_observation(observation) -> None`
  - `DarwinXModel.commit_gaba_observations(output) -> int`
- `DarwinXOutput.gaba_observations` is a tuple ordered by block index.

- [ ] **Step 1: Write failing tensor identity and state-purity tests**

```python
import copy

import pytest
import torch

from f51_darwin.gaba_inhibition import GABAConfig, GABAergicLayer


def test_gaba_gate_zero_is_bitwise_identity() -> None:
    torch.manual_seed(51)
    layer = GABAergicLayer(GABAConfig(d_model=8, max_delta_ratio=0.25))
    x = torch.randn(2, 3, 8)
    before = copy.deepcopy(layer.state_dict())
    delta, observation = layer(x, mutate_state=False)
    assert torch.equal(delta, torch.zeros_like(delta))
    assert layer.state_dict().keys() == before.keys()
    for key in before:
        assert torch.equal(layer.state_dict()[key], before[key])
    assert observation["sample_excitation"].shape == (2,)


def test_gaba_delta_is_inhibitory_and_bounded() -> None:
    layer = GABAergicLayer(GABAConfig(d_model=8, max_delta_ratio=0.25))
    with torch.no_grad():
        layer.residual_gate.fill_(10.0)
    x = torch.randn(2, 3, 8)
    delta, _ = layer(x, mutate_state=False)
    x_norm = x.float().norm(dim=-1)
    delta_norm = delta.float().norm(dim=-1)
    assert torch.all(delta_norm <= x_norm * 0.25 + 1e-6)
    assert not torch.equal(x + delta, 2.0 * x + delta)


def test_gaba_commits_once_and_rejects_nonfinite() -> None:
    layer = GABAergicLayer(GABAConfig(d_model=8))
    x = torch.randn(2, 3, 8)
    _, observation = layer(x, mutate_state=False)
    assert layer.state_update_count.item() == 0
    layer.commit_observation(observation)
    assert layer.state_update_count.item() == 1
    with pytest.raises(ValueError, match="already committed"):
        layer.commit_observation(observation)
    bad = dict(observation)
    bad["sample_excitation"] = torch.tensor([float("nan")])
    with pytest.raises(ValueError, match="finite"):
        layer.commit_observation(bad)
```

Add a tiny Darwin block test proving the corrected equation is
`residual + residual_scale * (moe_out + gaba_delta)`, never
`residual + residual_scale * (2 * moe_out - inhibition)`.

- [ ] **Step 2: Run tests and verify semantic failures**

Run:

```powershell
python -m pytest -q -p no:cacheprovider src\\tests\\test_gaba_causal.py
```

Expected: failures for missing gate, deferred commit, bounds, and corrected
block composition.

- [ ] **Step 3: Implement zero-init bounded delta**

Extend `GABAConfig` with exact defaults:

```python
max_delta_ratio: float = 0.25
gate_limit: float = 1.0
```

`GABAergicLayer` must register:

```python
self.residual_gate = nn.Parameter(torch.zeros(()))
self.register_buffer("state_update_count", torch.zeros((), dtype=torch.long))
self._committed_observation_ids: set[str] = set()
```

The forward path must:

1. compute inhibition without changing buffers;
2. calculate `gate = tanh(residual_gate) * gate_limit`;
3. calculate `raw_delta = -gate * inhibition`;
4. clamp its per-token norm to `max_delta_ratio * input_norm`;
5. return exact zeros when the gate is exactly zero;
6. return detached per-sample excitation/inhibition and a deterministic
   observation digest.

`commit_observation` validates finite tensors and a new digest, averages only
after per-sample measurement, updates EMA/level once, increments
`state_update_count`, and records the digest in a bounded recent-ID deque.

In `DarwinXBlock.forward`, replace:

```python
gaba_out, stats = self.gaba(moe_out)
moe_out = moe_out + gaba_out
```

with:

```python
gaba_delta, observation = self.gaba(moe_out, mutate_state=False)
moe_out = moe_out + gaba_delta
aux["gaba_observation"] = observation
```

Collect ordered observations in `DarwinXOutput`. Commit them only from the
training loop after a successful attempt outcome; eval, Ghost probes, and
`SHADOW` never call the commit method.

- [ ] **Step 4: Run GABA, model, checkpoint, and batch-independence tests**

Run:

```powershell
python -m pytest -q -p no:cacheprovider src\\tests\\test_gaba_causal.py src\\tests\\test_darwin_x_training.py src\\tests\\test_checkpoint_resume.py src\\tests\\test_causal_cognitive_organs.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add -- src/f51_darwin/gaba_inhibition.py src/f51_darwin/darwin_x_core/block.py src/f51_darwin/darwin_x_core/config.py src/f51_darwin/darwin_x_core/model.py src/tests/test_gaba_causal.py
git commit -m "fix(organism): make GABA inhibition bounded and causal"
```

---

### Task 3: Deterministic Ghost probe and exact loss subjects

**Files:**

- Modify: `src/f51_darwin/darwin_x_core/model.py`
- Modify: `src/f51_darwin/darwin_x_core/config.py`
- Modify: `src/f51_darwin/darwin_x_core/losses.py`
- Modify: `src/f51_darwin/organism/causal_bus.py`
- Modify: `src/f51_darwin/organism/causal_adapters.py`
- Test: `src/tests/test_ghost_causal.py`
- Test: `src/tests/test_loss_semantics.py`

**Interfaces:**

- Produces:
  - `deterministic_ghost_mask(input_ids, step_digest, ratio) -> Tensor[bool]`
  - `DarwinXOutput.raw_ghost_loss`
  - `DarwinXOutput.ghost_mask_digest`
- Bus `LOSS_TERM/SET_SCALE` accepts exact subjects `aux`, `ghost`, `jepa`,
  and `spider`.

- [ ] **Step 1: Write failing deterministic mask and arm-equivalence tests**

```python
import torch

from f51_darwin.darwin_x_core.model import deterministic_ghost_mask


def test_ghost_mask_is_digest_deterministic_and_rng_pure() -> None:
    ids = torch.arange(24).reshape(2, 12)
    before = torch.random.get_rng_state().clone()
    first = deterministic_ghost_mask(ids, "step-a", 0.25)
    second = deterministic_ghost_mask(ids, "step-a", 0.25)
    after = torch.random.get_rng_state()
    assert torch.equal(first, second)
    assert torch.equal(before, after)
    assert not torch.equal(first, deterministic_ghost_mask(ids, "step-b", 0.25))


def test_ghost_mask_is_sample_local() -> None:
    a = torch.tensor([[1, 2, 3, 4], [9, 8, 7, 6]])
    b = torch.tensor([[1, 2, 3, 4], [0, 0, 0, 0]])
    assert torch.equal(
        deterministic_ghost_mask(a, "step", 0.5)[0],
        deterministic_ghost_mask(b, "step", 0.5)[0],
    )
```

Add a tiny model test that runs `CONTROL`, `SHADOW`, and `APPLY` from cloned
state with the same `step_digest`. Assert identical mask digest and raw Ghost
loss in all arms, exact zero effective Ghost in Control/Shadow, and the declared
weighted loss plus nonzero gradient delta only in Apply.

Add bus tests accepting subjects `ghost`, `jepa`, and `spider`, while rejecting
unknown names, wildcard names, negative scales, booleans, infinity, and NaN.

- [ ] **Step 2: Run the focused tests and verify failures**

Run:

```powershell
python -m pytest -q -p no:cacheprovider src\\tests\\test_ghost_causal.py src\\tests\\test_loss_semantics.py src\\tests\\test_organ_causal_bus.py
```

Expected: missing deterministic-mask API, raw Ghost output, and subject
allowlist failures.

- [ ] **Step 3: Implement RNG-free mask and explicit raw/effective loss**

Use an integer hash per token position, not a global generator:

```python
def deterministic_ghost_mask(
    input_ids: torch.Tensor,
    step_digest: str,
    ratio: float,
) -> torch.Tensor:
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ghost ratio must be within [0, 1]")
    seed = int.from_bytes(
        hashlib.sha256(step_digest.encode("utf-8")).digest()[:8],
        "little",
    )
    rows = torch.arange(input_ids.size(0), device=input_ids.device, dtype=torch.int64)[:, None]
    cols = torch.arange(input_ids.size(1), device=input_ids.device, dtype=torch.int64)[None, :]
    mixed = (
        input_ids.to(torch.int64) * 0x45D9F3B
        + rows * 0x27D4EB2D
        + cols * 0x165667B1
        + seed
    ) & 0x7FFFFFFF
    threshold = int(ratio * 0x80000000)
    return mixed < threshold
```

Do not mutate `input_ids`. Guarantee at least one eligible masked token per
nonempty batch only when `ratio > 0` by choosing the minimum hash if the mask is
empty. Exclude ignored/special positions using the existing eligibility rules.

Pass the step digest from the training attempt into model forward. Preserve
legacy mask behavior when causal v2 is disabled. In v2, expose raw loss and
mask digest separately and let the single loss composer apply the effective
scale.

Extend `_domain_rejection` and `CausalTrainingExecutor.set_loss_scales` to the
four exact subjects. The executor must never reconstruct raw terms and must
recompute `output.loss` through the composer exactly once.

- [ ] **Step 4: Run focused and regression tests**

Run:

```powershell
python -m pytest -q -p no:cacheprovider src\\tests\\test_ghost_causal.py src\\tests\\test_loss_semantics.py src\\tests\\test_organ_causal_bus.py src\\tests\\test_causal_training_phases.py src\\tests\\test_darwin_x_training.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add -- src/f51_darwin/darwin_x_core/model.py src/f51_darwin/darwin_x_core/config.py src/f51_darwin/darwin_x_core/losses.py src/f51_darwin/organism/causal_bus.py src/f51_darwin/organism/causal_adapters.py src/tests/test_ghost_causal.py src/tests/test_loss_semantics.py src/tests/test_organ_causal_bus.py
git commit -m "feat(organism): add deterministic Ghost intervention"
```

---

### Task 4: Sample-local Curiosity and persistent next-replay priority

**Files:**

- Modify: `src/f51_darwin/curiosity.py`
- Modify: `src/f51_darwin/replay_buffer.py`
- Create: `src/f51_darwin/organism/organ_adapters.py`
- Test: `src/tests/test_curiosity_causal.py`
- Test: `src/tests/test_replay_buffer.py`

**Interfaces:**

- Produces:
  - `CuriosityMemory.observe(hidden, jepa_error, sample_ids) -> CuriosityObservation`
  - `CuriosityMemory.commit(observation) -> None`
  - `CuriosityMemory.state_dict()` and `load_state_dict(state: Mapping[str, object])`
  - `ReplayBuffer.add(input_ids: Iterable[int], label: str = "ancestral", priority: float = 1.0)`
  - `CuriosityPriorityAdapter`

- [ ] **Step 1: Write failing sample-local, lag, noisy-TV, and replay tests**

```python
import copy

import torch

from f51_darwin.curiosity import CuriosityMemory


def test_curiosity_is_batch_permutation_independent() -> None:
    memory = CuriosityMemory(d_model=4, bank_capacity=8)
    hidden = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0, 0.0]],
    ])
    a = memory.observe(hidden, torch.tensor([0.2, 0.7]), ("a", "b"))
    b = memory.observe(hidden.flip(0), torch.tensor([0.7, 0.2]), ("b", "a"))
    assert a.novelty == tuple(reversed(b.novelty))
    assert memory.bank_size == 0


def test_curiosity_bank_updates_only_after_commit() -> None:
    memory = CuriosityMemory(d_model=4, bank_capacity=8)
    hidden = torch.ones(1, 2, 4)
    observation = memory.observe(hidden, torch.tensor([0.5]), ("sample",))
    assert memory.bank_size == 0
    memory.commit(observation)
    assert memory.bank_size == 1
    next_observation = memory.observe(hidden, torch.tensor([0.5]), ("sample-2",))
    assert next_observation.novelty[0] < observation.novelty[0]


def test_persistent_surprise_is_downweighted_as_noise() -> None:
    memory = CuriosityMemory(
        d_model=4,
        bank_capacity=8,
        noisy_repeat_limit=3,
    )
    hidden = torch.ones(1, 2, 4)
    priorities = []
    for index in range(5):
        observation = memory.observe(
            hidden,
            torch.tensor([9.0]),
            (f"same-{index}",),
            content_digests=("same-content",),
        )
        priorities.append(observation.priority[0])
        memory.commit(observation)
    assert priorities[-1] < priorities[1]
```

Replay tests must prove priority affects only future samples, survives
roundtrip, stays within `[0.5, 2.0]`, and paired RNG with equal priorities
matches legacy sampling.

- [ ] **Step 2: Run tests and verify failures**

Run:

```powershell
python -m pytest -q -p no:cacheprovider src\\tests\\test_curiosity_causal.py src\\tests\\test_replay_buffer.py
```

Expected: missing CuriosityMemory and priority-aware replay APIs.

- [ ] **Step 3: Implement CuriosityMemory and priority replay**

`CuriosityObservation` is frozen and contains:

```python
sample_ids: tuple[str, ...]
content_digests: tuple[str, ...]
representations: torch.Tensor
novelty: tuple[float, ...]
surprise: tuple[float, ...]
priority: tuple[float, ...]
observation_id: str
```

Rules:

- representation is `hidden.float().mean(dim=1)` followed by per-sample
  normalization;
- novelty is `1 - max cosine similarity` against the frozen bank, or `1.0`
  for an empty bank;
- surprise is finite clamped JEPA error per sample;
- base priority is
  `clamp(0.5 + 0.75 * novelty + 0.25 * tanh(surprise), 0.5, 2.0)`;
- after three occurrences of the same content digest without recorded loss
  improvement, multiply excess priority by `0.5 ** repeats_after_limit`;
- `observe` is pure;
- `commit` accepts an observation ID once and appends representations after
  validation.

Extend `ReplayExample` with `priority: float = 1.0`. Weighted sampling uses the
buffer's private RNG and Efraimidis-Spirakis keys
`key = rng.random() ** (1.0 / priority)`, selecting the largest keys. When all
priorities equal `1.0`, retain the exact existing `random.sample` path for
legacy equality.

`CuriosityPriorityAdapter` consumes only a valid lagged frame and returns
per-sample replay priorities for rows added after the successful source
outcome. It never scales the current loss.

- [ ] **Step 4: Run Curiosity, replay, training, and checkpoint tests**

Run:

```powershell
python -m pytest -q -p no:cacheprovider src\\tests\\test_curiosity_causal.py src\\tests\\test_replay_buffer.py src\\tests\\test_checkpoint_resume.py src\\tests\\test_darwin_x_training.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 4**

```powershell
git add -- src/f51_darwin/curiosity.py src/f51_darwin/replay_buffer.py src/f51_darwin/organism/organ_adapters.py src/tests/test_curiosity_causal.py src/tests/test_replay_buffer.py
git commit -m "feat(organism): make Curiosity schedule future replay"
```

---

### Task 5: Checkpointed Soul policy and calibrated Decision veto

**Files:**

- Modify: `src/f51_darwin/soul.py`
- Modify: `src/f51_darwin/decision_engine.py`
- Create: `src/f51_darwin/organism/executive_state.py`
- Modify: `src/f51_darwin/organism/organ_adapters.py`
- Modify: `src/f51_darwin/organism/control.py`
- Test: `src/tests/test_soul_decision_causal.py`
- Test: `src/tests/test_soul_desire_runtime.py`

**Interfaces:**

- Produces:
  - `SoulPolicyState.observe(outcome_metrics) -> SoulPolicyFrame`
  - `CausalDecisionController.decide(policy, calibration) -> ExecutiveDecision`
  - `CausalDecisionController.feedback(intervention_id, success) -> None`
  - `SoulDecisionAdapter`

- [ ] **Step 1: Write failing Soul state, real heldout, and decision tests**

```python
from f51_darwin.organism.executive_state import (
    DecisionAction,
    SoulPolicyState,
)
from f51_darwin.decision_engine import CausalDecisionController


def test_soul_counts_incremental_tokens_and_receives_heldout() -> None:
    soul = SoulPolicyState(window_size=8)
    first = soul.observe(
        step=50,
        incremental_tokens=204_800,
        fresh_lm=3.0,
        replay_lm=1.0,
        heldout_lm=3.2,
    )
    second = soul.observe(
        step=100,
        incremental_tokens=204_800,
        fresh_lm=3.1,
        replay_lm=0.9,
        heldout_lm=5.2,
    )
    assert soul.total_tokens == 409_600
    assert second.request_update_veto is True
    assert second.heldout_lm == 5.2


def test_decision_stays_shadow_before_calibration_minimum() -> None:
    controller = CausalDecisionController(
        minimum_samples=8,
        approval_lower_bound=0.55,
        regression_veto=0.02,
    )
    policy = SoulPolicyState(window_size=8).observe(
        step=50,
        incremental_tokens=100,
        fresh_lm=3.0,
        replay_lm=2.0,
        heldout_lm=3.1,
    )
    decision = controller.decide(policy)
    assert decision.action is DecisionAction.DEFER
    assert decision.reason == "insufficient_calibration"


def test_nonfinite_or_regressing_policy_is_vetoed() -> None:
    controller = CausalDecisionController(
        minimum_samples=8,
        approval_lower_bound=0.55,
        regression_veto=0.02,
    )
    for index in range(8):
        controller.feedback(f"past-{index}", success=True)
    policy = SoulPolicyState(window_size=8).observe(
        step=50,
        incremental_tokens=100,
        fresh_lm=3.0,
        replay_lm=2.0,
        heldout_lm=3.5,
        heldout_regression=0.1,
    )
    assert controller.decide(policy).action is DecisionAction.VETO
```

Add tests proving:

- Decision feedback before an outcome is rejected;
- Wilson lower bound drives approve/defer deterministically;
- stale Soul policy cannot create an intervention;
- `request_checkpoint` queues a protected request instead of printing;
- `emergency_protocol` produces a typed veto proposal;
- legacy narrative output remains available but cannot mutate training.

- [ ] **Step 2: Run focused tests and verify failures**

Run:

```powershell
python -m pytest -q -p no:cacheprovider src\\tests\\test_soul_decision_causal.py src\\tests\\test_soul_desire_runtime.py
```

Expected: missing checkpointed policy/controller APIs and print-only command
failures.

- [ ] **Step 3: Implement numerical Soul state and Wilson-calibrated Decision**

`SoulPolicyState` stores deques of length 8 for fresh/replay/heldout loss,
incremental token count, last observed step, pending requests, and last frame
digest. It rejects decreasing steps, cumulative-token inputs, and non-finite
metrics.

`CausalDecisionController` stores success/failure counts and pending
intervention IDs. Use the Wilson lower bound:

```python
def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    margin = z * math.sqrt(
        (p * (1.0 - p) + z * z / (4.0 * total)) / total
    )
    return max(0.0, (centre - margin) / denominator)
```

Exact decision order:

1. non-finite or invalid policy -> `VETO`;
2. heldout regression above `0.02` -> `VETO`;
3. fewer than 8 completed feedback samples -> `DEFER`;
4. Wilson lower bound below `0.55` -> `DEFER`;
5. otherwise -> `APPROVE`.

`SoulDecisionAdapter` turns only approved lagged policy into an allowlisted
request. Emergency veto maps to `UPDATE/SKIP`; checkpoint/sleep/benchmark/data
requests map to lifecycle queue records and never bypass their gates.

Replace supported-runtime print-only dispatch with queueing typed requests.
Keep prints as secondary telemetry after the queue operation succeeds.

- [ ] **Step 4: Run focused, bus, lifecycle, and legacy tests**

Run:

```powershell
python -m pytest -q -p no:cacheprovider src\\tests\\test_soul_decision_causal.py src\\tests\\test_soul_desire_runtime.py src\\tests\\test_organ_causal_bus.py src\\tests\\test_organism_causal_runtime.py src\\tests\\test_protected_sleep_evolution.py
```

Expected: all tests pass, including the corrected internal mathematics
threshold characterization moved into collected tests.

- [ ] **Step 5: Commit Task 5**

```powershell
git add -- src/f51_darwin/soul.py src/f51_darwin/decision_engine.py src/f51_darwin/organism/executive_state.py src/f51_darwin/organism/organ_adapters.py src/f51_darwin/organism/control.py src/tests/test_soul_decision_causal.py src/tests/test_soul_desire_runtime.py
git commit -m "feat(organism): connect Soul through calibrated decisions"
```

---

### Task 6: Integrate all organs into training and lifecycle

**Files:**

- Modify: `src/f51_darwin/organism/bootstrap.py`
- Modify: `src/f51_darwin/organism/training.py`
- Modify: `src/f51_darwin/organism/lifecycle.py`
- Modify: `src/f51_darwin/organism/config.py`
- Modify: `src/f51_darwin/organism/organ_adapters.py`
- Modify: `src/f51_darwin/darwin_x_core/config.py`
- Test: `src/tests/test_causal_organism_v2_integration.py`

**Interfaces:**

- Consumes outputs from Tasks 1–5.
- Produces:
  - `organism.pending_organ_frame`
  - `organism.causal_cognitive_contract_version == 2`
  - one post-outcome observation and at most one next-step application.

- [ ] **Step 1: Write failing one-step-delay and integrated-arm tests**

Construct a tiny organism through the real bootstrap helpers with deterministic
data. The tests must assert:

```python
assert disabled.step_10.model_digest == control.step_10.model_digest
assert control.step_10.model_digest == shadow.step_10.model_digest
assert shadow.step_10.model_digest == apply.step_10.model_digest

assert disabled.step_11.model_digest == control.step_11.model_digest
assert control.step_11.model_digest == shadow.step_11.model_digest
assert apply.step_11.model_digest != shadow.step_11.model_digest

assert apply.step_11.applied_source_step == 10
assert apply.step_11.gaba_state_updates == 1
assert apply.step_11.ghost_mask_digest == shadow.step_11.ghost_mask_digest
```

Add integration cases for:

- Soul proposal vetoed by Decision;
- expired Curiosity frame;
- GABA plus Ghost in one attempt;
- conflicting Ghost and Soul loss-scale request;
- failed forward preserves pending frame and records outcome error;
- skipped optimizer does not commit GABA or Curiosity state;
- successful outcome commits each state exactly once.

- [ ] **Step 2: Run integration tests and verify failures**

Run:

```powershell
python -m pytest -q -p no:cacheprovider src\\tests\\test_causal_organism_v2_integration.py
```

Expected: missing v2 bootstrap, frame, and commit ordering.

- [ ] **Step 3: Implement exact training order**

The v2 training attempt order is:

```text
1. build StepIdentity
2. validate pending frame for this identity
3. collect pure adapter proposals
4. decide conflicts and append aggregate STEP_INTENT
5. forward using frozen GABA/Curiosity/Soul/Decision state
6. apply exact effective loss scales
7. backward
8. apply veto/clip/named gradient scales
9. optimizer step when due
10. append STEP_OUTCOME
11. only after successful outcome:
    a. commit GABA observations once
    b. commit Curiosity observation once
    c. add replay rows with lagged priorities
    d. update Decision feedback only for evaluable past interventions
12. create immutable frame valid at `optimizer_step + 1`
```

On any exception, zero unsafe gradients, record the failed outcome, and do not
commit organ state. No broad `except` may silently substitute a constant
Curiosity or Decision signal.

Add explicit v2 config fields with safe defaults:

```python
causal_cognitive_contract_version: int = 1
causal_v2_enabled: bool = False
gaba_causal_enabled: bool = False
ghost_causal_enabled: bool = False
curiosity_causal_enabled: bool = False
soul_causal_enabled: bool = False
decision_causal_enabled: bool = False
```

Enabling any organ requires loss semantics v2, bus enabled, explicit v8
migration, and cognitive contract version 2.

- [ ] **Step 4: Run integration and adjacent regressions**

Run:

```powershell
python -m pytest -q -p no:cacheprovider src\\tests\\test_causal_organism_v2_integration.py src\\tests\\test_causal_training_phases.py src\\tests\\test_organism_causal_runtime.py src\\tests\\test_protected_sleep_evolution.py src\\tests\\test_darwin_x_training.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 6**

```powershell
git add -- src/f51_darwin/organism/bootstrap.py src/f51_darwin/organism/training.py src/f51_darwin/organism/lifecycle.py src/f51_darwin/organism/config.py src/f51_darwin/organism/organ_adapters.py src/f51_darwin/darwin_x_core/config.py src/tests/test_causal_organism_v2_integration.py
git commit -m "feat(organism): close the v2 causal organ loop"
```

---

### Task 7: Strict cognitive-contract-v2 checkpoint identity

**Files:**

- Modify: `src/f51_darwin/state_identity.py`
- Modify: `src/f51_darwin/organism/checkpoint_mixin.py`
- Modify: `src/f51_darwin/organism/checkpoint.py`
- Modify: `src/f51_darwin/organism/bootstrap.py`
- Test: `src/tests/test_causal_organism_v2_checkpoint.py`
- Test: `src/tests/test_state_identity.py`

**Interfaces:**

- Produces:
  - `causal_cognitive_contract = {"version": 2, "state_id": "causal-cognitive-v2:<sha256>"}`
  - `causal_cognitive_state_v2`
  - strict v7/v8-v1/v8-v2 compatibility matrix.

- [ ] **Step 1: Write failing full-state roundtrip and tamper tests**

Create a tiny v2 organism with non-default state in every organ, one pending
frame, and a nonempty Decision calibration history. Save its checkpoint
payload in memory and test:

- exact state roundtrip;
- exact next decision after restore;
- exact next deterministic Ghost mask;
- exact next replay sample;
- tampered Soul window rejected;
- tampered Decision counts rejected;
- tampered Curiosity bank rejected;
- tampered GABA update count rejected;
- tampered Ghost pending digest rejected;
- removed frame or TTL rejected;
- v8-v1 with v2 disabled loads;
- v8-v1 with any v2 organ enabled rejects;
- v7 remains legacy.

- [ ] **Step 2: Run checkpoint tests and verify failures**

Run:

```powershell
python -m pytest -q -p no:cacheprovider src\\tests\\test_causal_organism_v2_checkpoint.py src\\tests\\test_state_identity.py src\\tests\\test_causal_checkpoint_contract.py
```

Expected: missing v2 state and compatibility failures.

- [ ] **Step 3: Implement canonical v2 state identity**

The saved state mapping has exact top-level keys:

```python
{
    "version": 2,
    "soul": soul_state,
    "decision": decision_state,
    "curiosity": curiosity_state,
    "gaba": gaba_state,
    "ghost": ghost_state,
    "heartbeat": heartbeat_state,
    "spider": spider_state,
    "pending_frame": pending_frame_state,
    "ledger_head": ledger_head,
    "last_outcome_identity": last_outcome_identity,
}
```

Hash canonical mappings in sorted-key order; hash tensors by dtype, shape, and
contiguous CPU bytes. Reject tensors or scalar floats containing non-finite
values before hashing. Narrative strings such as mantra, blessing, reflection,
war cry, and reason prose are excluded; stable reason codes remain included.

Restore v2 state before the next adapter decision. Validate the saved state ID,
frame digest, ledger head, and contract version before mutating the live
organism.

- [ ] **Step 4: Run all checkpoint, identity, sleep, and causal tests**

Run:

```powershell
python -m pytest -q -p no:cacheprovider src\\tests\\test_causal_organism_v2_checkpoint.py src\\tests\\test_state_identity.py src\\tests\\test_causal_checkpoint_contract.py src\\tests\\test_causal_cognitive_organs.py src\\tests\\test_checkpoint_resume.py src\\tests\\test_protected_sleep_evolution.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 7**

```powershell
git add -- src/f51_darwin/state_identity.py src/f51_darwin/organism/checkpoint_mixin.py src/f51_darwin/organism/checkpoint.py src/f51_darwin/organism/bootstrap.py src/tests/test_causal_organism_v2_checkpoint.py src/tests/test_state_identity.py
git commit -m "feat(checkpoint): persist causal cognitive contract v2"
```

---

### Task 8: Individual and 32-combination causal ablation

**Files:**

- Create: `src/tools/run_causal_organism_v2_ablation.py`
- Create: `src/tests/test_causal_organism_v2_ablation.py`
- Modify: `src/tools/run_causal_ablation.py`
- Modify: `governance/docs/operacao/STATUS_ATUAL.md`

**Interfaces:**

- Produces atomic artifact under
  `workspace/runtime/organism/causal_ablation_v2/<run-id>/`.
- Produces independent `verify` mode with nonzero exit on tamper.

- [ ] **Step 1: Write failing harness contract and tamper tests**

The test invokes the CLI against a temporary output root and requires:

```text
manifest.json
individual/soul.json
individual/decision.json
individual/curiosity.json
individual/gaba.json
individual/ghost.json
factorial/00000.json through factorial/11111.json
verification.json
```

For each individual report assert:

- paired initial checkpoint/optimizer/RNG/data/frame IDs;
- disabled/control/shadow equality;
- no source-step divergence;
- declared target-step Apply divergence;
- sign-inverted/permuted negative control result;
- raw/effective losses;
- parameter and organ-state digests;
- fresh/replay/heldout tiny metrics.

For the factorial set, assert exactly 32 unique bit patterns and that each
report identifies the five enabled organs. Modify one report byte and assert
`verify` fails without rewriting the artifact.

- [ ] **Step 2: Run the harness tests and verify failure**

Run:

```powershell
python -m pytest -q -p no:cacheprovider src\\tests\\test_causal_organism_v2_ablation.py
```

Expected: missing v2 harness.

- [ ] **Step 3: Implement atomic paired harness**

The harness:

1. builds one tiny deterministic checkpoint and optimizer state;
2. stores RNG and data cursor;
3. clones that state for every arm and factorial combination;
4. runs exactly two optimizer attempts so lagged effects are observable;
5. computes fresh/replay/heldout tiny LM losses with paired RNG;
6. writes all reports to a staging directory;
7. hashes every file into `manifest.json`;
8. verifies the complete staging artifact;
9. atomically renames staging to the requested final directory.

`verify` recomputes every hash, validates schema, proves the equality and
delay invariants, and rejects extra or missing files.

- [ ] **Step 4: Run focused harness and source verification**

Run:

```powershell
python -m pytest -q -p no:cacheprovider src\\tests\\test_causal_organism_v2_ablation.py src\\tests\\test_causal_ablation.py
python src\\tools\\run_causal_organism_v2_ablation.py run --output-root workspace\runtime\organism\causal_ablation_v2\2026-07-18-source
python src\\tools\\run_causal_organism_v2_ablation.py verify --artifact workspace\runtime\organism\causal_ablation_v2\2026-07-18-source
```

Expected: tests pass and verifier prints `valid=true`.

- [ ] **Step 5: Update canonical status honestly**

Record:

- exact source commit;
- exact test counts;
- artifact path and manifest ID;
- which organs showed declared causal effects;
- whether all 32 tiny combinations passed;
- no long 100M/1.6B training was run;
- no perplexity improvement claim;
- dirty-worktree or canary blockers that remain.

- [ ] **Step 6: Commit Task 8**

```powershell
git add -- src/tools/run_causal_organism_v2_ablation.py src/tools/run_causal_ablation.py src/tests/test_causal_organism_v2_ablation.py governance/docs/operacao/STATUS_ATUAL.md
git commit -m "test(organism): prove causal organ v2 interactions"
```

---

### Task 9: Full verification and independent review

**Files:**

- Modify only files required by review findings.
- Do not modify `src/configs/darwin_x_100m.yaml`.

- [ ] **Step 1: Run exact test collection**

```powershell
python -m pytest --collect-only -q -p no:cacheprovider
```

Expected: collection succeeds; record the exact collected count.

- [ ] **Step 2: Run the complete source suite**

```powershell
python -m pytest -q -p no:cacheprovider
```

Expected: every collected test passes.

- [ ] **Step 3: Run compilation and whitespace verification**

```powershell
python -m compileall -q f51_darwin scripts tools tests
git diff --check
```

Expected: both commands exit zero; CRLF conversion warnings are informational.

- [ ] **Step 4: Verify no forbidden runtime or operator-file mutation**

```powershell
git diff -- src/configs/darwin_x_100m.yaml
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'darwin_organism|serve_davi|run247|ghost_stream' } |
  Select-Object ProcessId, ParentProcessId, Name, CommandLine
```

Expected: the YAML diff is exactly Marco's pre-existing 4K/32K/Heartbeat
change; no Darwin trainer or serve process was started by this work.

- [ ] **Step 5: Run independent per-task and whole-branch review**

Generate task-scoped review packages after each task and a whole-branch package
from commit `0fcb776` to the final v2 head. Review must explicitly check:

- temporal barrier;
- all five organ effects;
- no silent fallback constants;
- state identity completeness;
- Control/Shadow equality;
- Apply attribution;
- Ghost mask determinism;
- GABA no-double-update;
- Curiosity batch independence;
- Soul/Decision calibration;
- honest status claims.

Fix every P0/P1/P2 finding, rerun covering tests, and re-review until clean.

- [ ] **Step 6: Run no-launch canary only if source is clean**

```powershell
powershell -ExecutionPolicy Bypass -File src\\scripts\\start_overnight_16b.ps1 -Canary
```

Expected: source gates pass without `-Launch`. If the worktree is dirty because
Marco's YAML remains uncommitted, record `Tracked worktree must be clean` and
do not claim later canary gates passed.

- [ ] **Step 7: Publish final evidence summary**

The final report must distinguish:

- source correctness;
- tiny causal proof;
- checkpoint compatibility;
- no-launch readiness;
- unexecuted long-run benchmark;
- remaining operator decisions.

Do not mark the full objective complete unless every design requirement has
direct evidence and no P0/P1/P2 finding remains.
