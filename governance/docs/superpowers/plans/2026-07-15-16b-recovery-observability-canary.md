# Darwin-X 1.6B Recovery, Observability, and Canary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a measurable, crash-contained 250-step exact-backward canary from `organism_cycle_068.pt`, with disjoint holdout evidence, isolated checkpoints, and fail-closed GPU/quality gates before any new `run247` launch.

**Architecture:** Add one focused pure-Python observability module for token splitting, channel metrics, side-effect-free evaluation, durable JSONL, and acceptance decisions. Keep `src/scripts/darwin_organism.py` as the runtime orchestrator, extend the existing official PowerShell launcher for Windows/GPU controls, and preserve the existing external checkpoint comparator as the final independent quality gate.

**Tech Stack:** Python 3.11, PyTorch, NumPy/memmap, pytest, PowerShell 7, `nvidia-smi`, Windows Event Log, `powercfg`, Git.

## Global Constraints

- The final objective is the canonical `F51-Darwin-X-1.6B-Nitro` flow: load the exact v7 cycle-68 lineage, train one bounded cycle on both GPUs, publish only an isolated candidate, and produce evidence sufficient for Marco to decide whether a later `run247` is safe.
- The operator flow is: CPU tests -> launcher dry-run -> explicit `-Launch` canary -> two-minute post-run observation -> offline checkpoint verification -> external comparison -> explicit human promotion decision.
- The official start command remains `src/scripts/start_overnight_16b.ps1`; `-Canary` selects the bounded recovery path and never implies `run247`.
- A successful result is a final external canary report with status `stable_canary`, a verified isolated candidate checkpoint, finite fresh/replay/heldout metrics, no disallowed Windows GPU events, both GPU temperatures below 85 C, and no external benchmark regression.
- Current blockers are misleading replay-biased log cadence, no disjoint in-run holdout, no crash-durable metric ledger, no isolated checkpoint-root CLI, no async-save join in one-shot `cycle`, and no official bounded GPU/event gate.
- Do not add WGrad, install or change the NVIDIA driver, edit TDR, reboot, delete checkpoints, overwrite the canonical pointer, auto-promote, or auto-start `run247`.
- Keep the canonical corpus and all canary artifacts under `F51-Dataset-Organizado`; do not create physical token/checkpoint data in the repository.
- The canary must fail closed if weighted/multi-source sampling is active because positional exclusion cannot then be proven by the primary-stream split alone.
- All tracked code changes must be committed before the launcher accepts a canary. Untracked local artifacts do not block the tracked-worktree gate.
- Use PowerShell 7 explicitly for all PowerShell commands in this plan.

---

### Task 1: Add disjoint token split, independent metric channels, durable JSONL, and acceptance logic

**Files:**

- Create: `src/f51_darwin/training_observability.py`
- Create: `src/tests/test_training_observability.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class TailHoldout:
    train_tokens: object
    holdout_tokens: object
    source_token_count: int
    train_stop: int
    holdout_start: int
    holdout_token_count: int

def split_tail_holdout(token_ids: object, *, holdout_tokens: int, block_size: int) -> TailHoldout: ...
def fixed_batch_starts(*, token_count: int, block_size: int, batch_size: int,
                       batches: int, seed: int) -> tuple[tuple[int, ...], ...]: ...

@dataclass
class ChannelWindow:
    size: int = 20
    total_losses: deque[float] = field(default_factory=deque)
    lm_losses: deque[float] = field(default_factory=deque)
    count: int = 0
    def add(self, *, total_loss: float, lm_loss: float) -> None: ...
    def snapshot(self) -> dict[str, float | int | None]: ...

@dataclass
class MetricChannels:
    fresh: ChannelWindow
    replay: ChannelWindow
    successful_updates: int = 0
    skipped_updates: int = 0
    tokens_processed: int = 0
    def record_update(self, *, batch_kind: Literal["fresh", "replay"],
                      total_loss: float, lm_loss: float, tokens: int) -> None: ...
    def snapshot(self, *, elapsed_sec: float) -> dict[str, object]: ...

def append_jsonl_fsync(path: Path, record: Mapping[str, object]) -> None: ...
def recover_jsonl(path: Path) -> tuple[list[dict[str, object]], bool]: ...

@dataclass(frozen=True)
class CanaryGateInput:
    expected_steps: int
    successful_updates: int
    skipped_updates: int
    fresh: Mapping[str, object]
    replay: Mapping[str, object]
    replay_fraction: float
    base_heldout_loss: float | None
    candidate_heldout_loss: float | None
    peak_temperatures_c: tuple[float, ...]
    new_nvidia_events: tuple[Mapping[str, object], ...]
    live_kernel_event_141: bool
    checkpoint_verified: bool
    external_benchmark_passed: bool

def decide_canary(input: CanaryGateInput) -> dict[str, object]: ...
def canary_gate_from_mapping(raw: Mapping[str, object]) -> dict[str, object]: ...
```

- [ ] **Step 1: Write the failing split, window, JSONL, and gate tests**

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from f51_darwin.training_observability import (
    CanaryGateInput,
    MetricChannels,
    append_jsonl_fsync,
    canary_gate_from_mapping,
    decide_canary,
    fixed_batch_starts,
    recover_jsonl,
    split_tail_holdout,
)


def test_tail_holdout_is_positionally_disjoint_without_copy() -> None:
    source = np.arange(100, dtype=np.int32)
    split = split_tail_holdout(source, holdout_tokens=32, block_size=8)
    assert np.shares_memory(source, split.train_tokens)
    assert np.shares_memory(source, split.holdout_tokens)
    assert split.train_stop == split.holdout_start == 68
    assert split.train_tokens[-1] == 67
    assert split.holdout_tokens[0] == 68
    assert len(split.train_tokens) + len(split.holdout_tokens) == len(source)


@pytest.mark.parametrize("length", [8, 16, 39])
def test_tail_holdout_rejects_short_source(length: int) -> None:
    with pytest.raises(ValueError, match="too short"):
        split_tail_holdout(np.arange(length), holdout_tokens=32, block_size=8)


def test_fixed_starts_are_stable_and_within_holdout() -> None:
    first = fixed_batch_starts(token_count=128, block_size=8, batch_size=2, batches=16, seed=999)
    second = fixed_batch_starts(token_count=128, block_size=8, batch_size=2, batches=16, seed=999)
    assert first == second
    assert len(first) == 16
    assert all(len(row) == 2 for row in first)
    assert all(0 <= start <= 119 for row in first for start in row)


def test_metric_channels_never_mix_fresh_and_replay_windows() -> None:
    channels = MetricChannels.create(window_size=2)
    channels.record_update(batch_kind="fresh", total_loss=4.0, lm_loss=3.0, tokens=8)
    channels.record_update(batch_kind="replay", total_loss=8.0, lm_loss=7.0, tokens=8)
    channels.record_update(batch_kind="fresh", total_loss=2.0, lm_loss=1.0, tokens=8)
    snap = channels.snapshot(elapsed_sec=2.0)
    assert snap["fresh"] == {"count": 2, "total_loss": 3.0, "lm_loss": 2.0}
    assert snap["replay"] == {"count": 1, "total_loss": 8.0, "lm_loss": 7.0}
    assert snap["replay_fraction"] == pytest.approx(1 / 3)


def test_jsonl_recovery_ignores_one_partial_trailing_record(tmp_path: Path) -> None:
    ledger = tmp_path / "metrics.jsonl"
    append_jsonl_fsync(ledger, {"step": 1, "heldout_lm_loss": 2.0})
    with ledger.open("ab") as handle:
        handle.write(b'{"step":2')
    records, truncated = recover_jsonl(ledger)
    assert records == [{"heldout_lm_loss": 2.0, "step": 1}]
    assert truncated is True


def test_canary_gate_fails_closed_on_regression_or_missing_evidence() -> None:
    result = decide_canary(CanaryGateInput(
        expected_steps=250,
        successful_updates=250,
        skipped_updates=0,
        fresh={"count": 200, "lm_loss": 2.0, "total_loss": 2.1},
        replay={"count": 50, "lm_loss": 2.2, "total_loss": 2.3},
        replay_fraction=0.20,
        base_heldout_loss=2.0,
        candidate_heldout_loss=2.03,
        peak_temperatures_c=(74.0, 70.0),
        new_nvidia_events=(),
        live_kernel_event_141=False,
        checkpoint_verified=True,
        external_benchmark_passed=True,
    ))
    assert result["status"] == "quality_gate_failed"
    assert "heldout_regression" in result["failures"]


def test_canary_gate_labels_pass_as_stable_not_improved() -> None:
    result = decide_canary(CanaryGateInput(
        expected_steps=250,
        successful_updates=250,
        skipped_updates=0,
        fresh={"count": 200, "lm_loss": 2.0, "total_loss": 2.1},
        replay={"count": 50, "lm_loss": 2.2, "total_loss": 2.3},
        replay_fraction=0.20,
        base_heldout_loss=2.0,
        candidate_heldout_loss=2.01,
        peak_temperatures_c=(74.0, 70.0),
        new_nvidia_events=(),
        live_kernel_event_141=False,
        checkpoint_verified=True,
        external_benchmark_passed=True,
    ))
    assert result == {"status": "stable_canary", "passed": True, "failures": []}


def test_canary_gate_mapping_rejects_missing_required_field() -> None:
    with pytest.raises((KeyError, TypeError, ValueError)):
        canary_gate_from_mapping({"expected_steps": 250})
```

- [ ] **Step 2: Run the tests and confirm the module is missing**

Run:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& .\.venv_nitro\Scripts\python.exe -m pytest src\\tests\\test_training_observability.py -q -p no:cacheprovider
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

Expected: collection fails with `ModuleNotFoundError: No module named 'f51_darwin.training_observability'`.

- [ ] **Step 3: Implement the pure observability module**

```python
from __future__ import annotations

import json
import math
import os
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping


@dataclass(frozen=True)
class TailHoldout:
    train_tokens: object
    holdout_tokens: object
    source_token_count: int
    train_stop: int
    holdout_start: int
    holdout_token_count: int


def split_tail_holdout(token_ids: object, *, holdout_tokens: int, block_size: int) -> TailHoldout:
    total = len(token_ids)  # type: ignore[arg-type]
    if holdout_tokens <= block_size:
        raise ValueError("holdout_tokens must be longer than block_size")
    train_stop = total - holdout_tokens
    if train_stop <= block_size:
        raise ValueError(
            f"token source too short for disjoint split: total={total} "
            f"holdout={holdout_tokens} block_size={block_size}"
        )
    return TailHoldout(
        train_tokens=token_ids[:train_stop],  # type: ignore[index]
        holdout_tokens=token_ids[train_stop:],  # type: ignore[index]
        source_token_count=total,
        train_stop=train_stop,
        holdout_start=train_stop,
        holdout_token_count=holdout_tokens,
    )


def fixed_batch_starts(*, token_count: int, block_size: int, batch_size: int,
                       batches: int, seed: int) -> tuple[tuple[int, ...], ...]:
    max_start = token_count - block_size
    if max_start < 1 or batch_size < 1 or batches < 1:
        raise ValueError("invalid fixed holdout batch geometry")
    rng = random.Random(seed)
    return tuple(
        tuple(rng.randrange(0, max_start) for _ in range(batch_size))
        for _ in range(batches)
    )


@dataclass
class ChannelWindow:
    size: int = 20
    total_losses: deque[float] = field(default_factory=deque)
    lm_losses: deque[float] = field(default_factory=deque)
    count: int = 0

    def add(self, *, total_loss: float, lm_loss: float) -> None:
        if not math.isfinite(total_loss) or not math.isfinite(lm_loss):
            raise ValueError("metric losses must be finite")
        self.total_losses.append(float(total_loss))
        self.lm_losses.append(float(lm_loss))
        while len(self.total_losses) > self.size:
            self.total_losses.popleft()
            self.lm_losses.popleft()
        self.count += 1

    def snapshot(self) -> dict[str, float | int | None]:
        return {
            "count": self.count,
            "total_loss": None if not self.total_losses else sum(self.total_losses) / len(self.total_losses),
            "lm_loss": None if not self.lm_losses else sum(self.lm_losses) / len(self.lm_losses),
        }


@dataclass
class MetricChannels:
    fresh: ChannelWindow
    replay: ChannelWindow
    successful_updates: int = 0
    skipped_updates: int = 0
    tokens_processed: int = 0

    @classmethod
    def create(cls, *, window_size: int = 20) -> "MetricChannels":
        return cls(ChannelWindow(window_size), ChannelWindow(window_size))

    def record_update(self, *, batch_kind: Literal["fresh", "replay"],
                      total_loss: float, lm_loss: float, tokens: int) -> None:
        channel = self.fresh if batch_kind == "fresh" else self.replay
        channel.add(total_loss=total_loss, lm_loss=lm_loss)
        self.successful_updates += 1
        self.tokens_processed += int(tokens)

    def snapshot(self, *, elapsed_sec: float) -> dict[str, object]:
        replay_count = self.replay.count
        return {
            "fresh": self.fresh.snapshot(),
            "replay": self.replay.snapshot(),
            "successful_updates": self.successful_updates,
            "skipped_updates": self.skipped_updates,
            "tokens_processed": self.tokens_processed,
            "tokens_per_sec": self.tokens_processed / max(elapsed_sec, 1e-9),
            "replay_fraction": replay_count / max(self.successful_updates, 1),
        }


def append_jsonl_fsync(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(record), sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def recover_jsonl(path: Path) -> tuple[list[dict[str, object]], bool]:
    if not path.exists():
        return [], False
    raw = path.read_bytes()
    truncated = bool(raw and not raw.endswith(b"\n"))
    complete = raw.splitlines()[:-1] if truncated else raw.splitlines()
    return [json.loads(line) for line in complete if line.strip()], truncated


@dataclass(frozen=True)
class CanaryGateInput:
    expected_steps: int
    successful_updates: int
    skipped_updates: int
    fresh: Mapping[str, object]
    replay: Mapping[str, object]
    replay_fraction: float
    base_heldout_loss: float | None
    candidate_heldout_loss: float | None
    peak_temperatures_c: tuple[float, ...]
    new_nvidia_events: tuple[Mapping[str, object], ...]
    live_kernel_event_141: bool
    checkpoint_verified: bool
    external_benchmark_passed: bool


def decide_canary(value: CanaryGateInput) -> dict[str, object]:
    failures: list[str] = []
    if value.successful_updates + value.skipped_updates != value.expected_steps:
        failures.append("update_accounting")
    for name, channel in (("fresh", value.fresh), ("replay", value.replay)):
        if not channel.get("count") or not all(
            isinstance(channel.get(key), (int, float)) and math.isfinite(float(channel[key]))
            for key in ("total_loss", "lm_loss")
        ):
            failures.append(f"{name}_metrics")
    tolerance = 1.0 / max(value.successful_updates, 1)
    if abs(value.replay_fraction - 0.20) > tolerance:
        failures.append("replay_fraction")
    if value.base_heldout_loss is None or value.candidate_heldout_loss is None:
        failures.append("heldout_missing")
    elif not math.isfinite(value.base_heldout_loss) or not math.isfinite(value.candidate_heldout_loss):
        failures.append("heldout_nonfinite")
    elif value.candidate_heldout_loss > value.base_heldout_loss + 0.02:
        failures.append("heldout_regression")
    if len(value.peak_temperatures_c) < 2 or any(temp >= 85.0 for temp in value.peak_temperatures_c):
        failures.append("gpu_temperature")
    if value.new_nvidia_events or value.live_kernel_event_141:
        failures.append("gpu_runtime_events")
    if not value.checkpoint_verified:
        failures.append("checkpoint_verification")
    if not value.external_benchmark_passed:
        failures.append("external_benchmark")
    if not failures:
        return {"status": "stable_canary", "passed": True, "failures": []}
    hardware = {"gpu_temperature", "gpu_runtime_events"}
    status = "hardware_runtime_failed" if hardware.intersection(failures) else "quality_gate_failed"
    return {"status": status, "passed": False, "failures": failures}


def canary_gate_from_mapping(raw: Mapping[str, object]) -> dict[str, object]:
    value = CanaryGateInput(
        expected_steps=int(raw["expected_steps"]),
        successful_updates=int(raw["successful_updates"]),
        skipped_updates=int(raw["skipped_updates"]),
        fresh=dict(raw["fresh"]),  # type: ignore[arg-type]
        replay=dict(raw["replay"]),  # type: ignore[arg-type]
        replay_fraction=float(raw["replay_fraction"]),
        base_heldout_loss=None if raw["base_heldout_loss"] is None else float(raw["base_heldout_loss"]),
        candidate_heldout_loss=None if raw["candidate_heldout_loss"] is None else float(raw["candidate_heldout_loss"]),
        peak_temperatures_c=tuple(float(item) for item in raw["peak_temperatures_c"]),  # type: ignore[union-attr]
        new_nvidia_events=tuple(dict(item) for item in raw["new_nvidia_events"]),  # type: ignore[union-attr]
        live_kernel_event_141=bool(raw["live_kernel_event_141"]),
        checkpoint_verified=bool(raw["checkpoint_verified"]),
        external_benchmark_passed=bool(raw["external_benchmark_passed"]),
    )
    return decide_canary(value)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a Darwin recovery canary gate.")
    parser.add_argument("--gate-json", required=True)
    args = parser.parse_args()
    raw = json.loads(Path(args.gate_json).read_text(encoding="utf-8"))
    print(json.dumps(canary_gate_from_mapping(raw), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the focused tests and verify green**

Run the Step 2 command again.

Expected: all focused primitive tests pass.

- [ ] **Step 5: Commit the primitives**

```powershell
git add src/f51_darwin/training_observability.py src/tests/test_training_observability.py
git commit -m "feat: add canary observability primitives"
```

---

### Task 2: Add a fixed, side-effect-free heldout evaluator

**Files:**

- Modify: `src/f51_darwin/training_observability.py`
- Modify: `src/tests/test_training_observability.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class HeldoutResult:
    lm_loss: float
    ppl: float
    batches: int
    tokens: int
    starts: tuple[tuple[int, ...], ...]

def evaluate_fixed_holdout(model: torch.nn.Module, holdout_tokens: object, *,
                           starts: tuple[tuple[int, ...], ...], block_size: int,
                           device: torch.device, amp_dtype: torch.dtype | None) -> HeldoutResult: ...
```

- [ ] **Step 1: Add failing tests for identical starts, mode restoration, RNG restoration, and mutable router state**

```python
import copy
import random
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from f51_darwin.training_observability import evaluate_fixed_holdout


class _EvalMoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._expert_usage_buffer = torch.zeros(2)
        self._vertical_bias = torch.tensor([1.0, 2.0])
        self._ghost_loss = 7.0


class _EvalBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.moe = _EvalMoE()


class _EvalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(64, 8)
        self.head = nn.Linear(8, 64)
        self.blocks = nn.ModuleList([_EvalBlock()])

    def forward(self, input_ids, labels=None, *, heartbeat=None, domain="train"):
        assert heartbeat is False
        moe = self.blocks[0].moe
        moe._expert_usage_buffer.add_(1)
        moe._vertical_bias = torch.zeros(2)
        moe._ghost_loss = float(torch.rand(()))
        logits = self.head(self.embedding(input_ids))
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 64), labels[:, 1:].reshape(-1))
        return SimpleNamespace(lm_loss=loss)


def _assert_nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_fixed_holdout_restores_mode_rng_and_mutable_router_state() -> None:
    torch.manual_seed(51)
    random.seed(51)
    model = _EvalModel().train()
    tokens = np.arange(256, dtype=np.int64) % 64
    starts = fixed_batch_starts(token_count=256, block_size=8, batch_size=2, batches=4, seed=999)
    torch_state = torch.get_rng_state().clone()
    python_state = random.getstate()
    before_buffer = model.blocks[0].moe._expert_usage_buffer.clone()
    before_bias = model.blocks[0].moe._vertical_bias.clone()
    before_ghost = model.blocks[0].moe._ghost_loss

    result = evaluate_fixed_holdout(
        model, tokens, starts=starts, block_size=8,
        device=torch.device("cpu"), amp_dtype=None,
    )

    assert result.starts == starts
    assert result.batches == 4
    assert result.tokens == 64
    assert result.ppl == pytest.approx(np.exp(result.lm_loss))
    assert model.training is True
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    assert torch.equal(model.blocks[0].moe._expert_usage_buffer, before_buffer)
    assert torch.equal(model.blocks[0].moe._vertical_bias, before_bias)
    assert model.blocks[0].moe._ghost_loss == before_ghost


def test_baseline_and_candidate_evaluation_reuse_exact_starts() -> None:
    tokens = np.arange(256, dtype=np.int64) % 64
    starts = fixed_batch_starts(token_count=256, block_size=8, batch_size=2, batches=16, seed=999)
    baseline = evaluate_fixed_holdout(_EvalModel(), tokens, starts=starts, block_size=8,
                                      device=torch.device("cpu"), amp_dtype=None)
    candidate = evaluate_fixed_holdout(_EvalModel(), tokens, starts=starts, block_size=8,
                                       device=torch.device("cpu"), amp_dtype=None)
    assert baseline.starts == candidate.starts == starts


def test_real_darwin_eval_restores_nitro_placements_and_buffers() -> None:
    from f51_darwin.darwin_x import DarwinXConfig, DarwinXModel

    config = DarwinXConfig(
        vocab_size=64, context_length=8, inference_context_length=16,
        d_model=16, n_layers=4, n_heads=4, n_kv_heads=2,
        fine_experts=4, shared_experts=1, experts_per_token=2,
        fine_expert_hidden_dim=8, shared_expert_hidden_dim=8,
        heartbeat_enabled=True, ghost_enabled=True,
        nitro_gpu_expert_capacity=1,
    )
    model = DarwinXModel(config).train()
    for block in model.blocks:
        block.moe._init_expert_devices(torch.device("cpu"))
        for expert in block.moe.fine_experts:
            for parameter in expert.parameters():
                parameter.requires_grad = False
    before_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    before_placements = [dict(block.moe._expert_gpu) for block in model.blocks]
    before_requires_grad = [
        [[parameter.requires_grad for parameter in expert.parameters()]
         for expert in block.moe.fine_experts]
        for block in model.blocks
    ]
    heartbeat_before = copy.deepcopy(model.heartbeat_state_dict())
    tokens = np.arange(128, dtype=np.int64) % 64
    starts = fixed_batch_starts(token_count=128, block_size=8, batch_size=1, batches=2, seed=999)

    evaluate_fixed_holdout(model, tokens, starts=starts, block_size=8,
                           device=torch.device("cpu"), amp_dtype=None)

    assert model.training is True
    assert all(torch.equal(model.state_dict()[name], value) for name, value in before_buffers.items())
    assert [dict(block.moe._expert_gpu) for block in model.blocks] == before_placements
    assert [
        [[parameter.requires_grad for parameter in expert.parameters()]
         for expert in block.moe.fine_experts]
        for block in model.blocks
    ] == before_requires_grad
    _assert_nested_equal(model.heartbeat_state_dict(), heartbeat_before)
```

- [ ] **Step 2: Run focused tests and confirm the evaluator import fails**

Run:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& .\.venv_nitro\Scripts\python.exe -m pytest src\\tests\\test_training_observability.py -q -p no:cacheprovider
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

Expected: import error for `evaluate_fixed_holdout`.

- [ ] **Step 3: Implement evaluation state preservation and fixed heldout evaluation**

```python
import copy
import random

import numpy as np
import torch


@dataclass(frozen=True)
class HeldoutResult:
    lm_loss: float
    ppl: float
    batches: int
    tokens: int
    starts: tuple[tuple[int, ...], ...]


def _clone_runtime_value(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return copy.deepcopy(value)


def _moe_runtime_state(model: torch.nn.Module) -> list[tuple[object, dict[str, object]]]:
    names = (
        "_expert_usage_buffer", "_vertical_bias", "_ghost_loss", "_router_entropy",
        "_expert_usage", "_autonomic_forward_uses", "_autonomic_usage_accumulator",
        "_autonomic_entropy_sum", "_autonomic_entropy_observations", "_pending_autonomic_actions",
        "_sleep_active", "_sleep_steps_remaining", "_nitro_step_counter",
    )
    saved: list[tuple[object, dict[str, object]]] = []
    for block in getattr(model, "blocks", ()):
        moe = getattr(block, "moe", None)
        if moe is None:
            continue
        values = {name: _clone_runtime_value(getattr(moe, name))
                  for name in names if hasattr(moe, name)}
        neuroendocrine = getattr(moe, "neuroendocrine", None)
        if neuroendocrine is not None:
            values["__neuroendocrine_state__"] = {
                name: tensor.detach().clone()
                for name, tensor in neuroendocrine.state_dict().items()
            }
        saved.append((moe, values))
    return saved


def _restore_moe_runtime_state(saved: list[tuple[object, dict[str, object]]]) -> None:
    for moe, values in saved:
        for name, value in values.items():
            if name == "__neuroendocrine_state__":
                moe.neuroendocrine.load_state_dict(value, strict=True)
                continue
            current = getattr(moe, name, None)
            if isinstance(current, torch.Tensor) and isinstance(value, torch.Tensor):
                current.copy_(value)
            else:
                setattr(moe, name, value)


def _nitro_placements(model: torch.nn.Module) -> list[tuple[object, dict[int, tuple[object, tuple[bool, ...]]]]]:
    saved = []
    for block in getattr(model, "blocks", ()):
        moe = getattr(block, "moe", None)
        if moe is None or not hasattr(moe, "_expert_gpu"):
            continue
        placements = {
            index: (moe._expert_gpu.get(index), tuple(p.requires_grad for p in expert.parameters()))
            for index, expert in enumerate(moe.fine_experts)
        }
        saved.append((moe, placements))
    return saved


def _restore_nitro_placements(
    saved: list[tuple[object, dict[int, tuple[object, tuple[bool, ...]]]]],
) -> None:
    for moe, placements in saved:
        for index, (assigned, requires_grad) in placements.items():
            expert = moe.fine_experts[index]
            target = torch.device("cpu") if assigned is None else torch.device(assigned)
            expert.to(device=target)
            for parameter, required in zip(expert.parameters(), requires_grad):
                parameter.requires_grad = required
            moe._expert_gpu[index] = assigned


def evaluate_fixed_holdout(model: torch.nn.Module, holdout_tokens: object, *,
                           starts: tuple[tuple[int, ...], ...], block_size: int,
                           device: torch.device, amp_dtype: torch.dtype | None) -> HeldoutResult:
    was_training = model.training
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mutable_state = _moe_runtime_state(model)
    losses: list[float] = []
    try:
        model.eval()
        with torch.inference_mode():
            for row_starts in starts:
                batch_placements = _nitro_placements(model)
                rows = [np.asarray(holdout_tokens[start:start + block_size], dtype=np.int64)
                        for start in row_starts]
                batch = torch.from_numpy(np.stack(rows)).to(device)
                try:
                    with torch.amp.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=amp_dtype is not None,
                    ):
                        output = model(batch, labels=batch, domain="canary_holdout", heartbeat=False)
                    if output.lm_loss is None or not torch.isfinite(output.lm_loss):
                        raise RuntimeError("heldout lm_loss is missing or non-finite")
                    losses.append(float(output.lm_loss.detach().cpu()))
                finally:
                    _restore_nitro_placements(batch_placements)
    finally:
        _restore_moe_runtime_state(mutable_state)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        model.train(was_training)
    mean_loss = sum(losses) / len(losses)
    return HeldoutResult(
        lm_loss=mean_loss,
        ppl=math.exp(min(mean_loss, 20.0)),
        batches=len(losses),
        tokens=sum(len(row) for row in starts) * block_size,
        starts=starts,
    )
```

- [ ] **Step 4: Run focused tests and verify green**

Run the Step 2 command again.

Expected: all `src/tests/test_training_observability.py` tests pass.

- [ ] **Step 5: Commit the evaluator**

```powershell
git add src/f51_darwin/training_observability.py src/tests/test_training_observability.py
git commit -m "feat: add fixed side effect free holdout eval"
```

---

### Task 3: Integrate disjoint metrics and heldout evaluation into the organism runtime

**Files:**

- Modify: `src/scripts/darwin_organism.py:39-78,454-494,797-800,1067-1260,1342-1385`
- Modify: `src/tests/test_organism_causal_runtime.py:1-106`

**Interfaces:**

```python
class DarwinOrganismConfig:
    holdout_tokens: int = 0
    holdout_batches: int = 16
    holdout_seed: int = 999
    metrics_jsonl: str | None = None
    canary_run_id: str | None = None
    canary_metadata_json: str | None = None

class DarwinOrganism:
    token_ids: object              # full source identity
    train_token_ids: object        # head view, optimizer only
    holdout_token_ids: object      # tail view, evaluator only
    holdout_starts: tuple[tuple[int, ...], ...]
    metric_channels: MetricChannels
    def evaluate_holdout(self, *, phase: str) -> HeldoutResult: ...
    def _append_metric_record(self, *, phase: str, heldout: HeldoutResult | None) -> None: ...
```

- [ ] **Step 1: Extend the tiny runtime test to prove category separation and heldout exclusion**

Update `_TinyCausalModel.forward` to accept `heartbeat=None`, then replace the existing 10-step assertion block and add the weighted-source failure test:

```python
def test_train_cycle_uses_twenty_percent_replay_without_larger_batch(tmp_path) -> None:
    organism = DarwinOrganism.__new__(DarwinOrganism)
    organism.cfg = SimpleNamespace(
        block_size=4, batch_size=1, seed=51,
        replay_warmup_examples=4, replay_ratio=0.20,
        replay_add_every=20, replay_sample_size=2, eval_every=10,
        grad_clip=1.0, accum_steps=1,
        metrics_jsonl=None, canary_run_id="test-run",
        holdout_tokens=64, holdout_batches=2,
    )
    organism.cycle = 1
    organism.total_steps = 0
    organism.device = torch.device("cpu")
    organism.amp_dtype = None
    organism.token_ids = np.arange(256, dtype=np.int64) % 32
    split = split_tail_holdout(organism.token_ids, holdout_tokens=64, block_size=4)
    organism.train_token_ids = split.train_tokens
    organism.holdout_token_ids = split.holdout_tokens
    organism.holdout_starts = fixed_batch_starts(
        token_count=64, block_size=4, batch_size=1, batches=2, seed=999,
    )
    organism.token_count = len(organism.token_ids)
    organism.replay = ReplayBuffer(capacity=16, seed=51)
    for offset in range(4):
        organism.replay.add([offset, offset + 1, offset + 2, offset + 3])
    organism.model = _TinyCausalModel()
    organism.optimizer = torch.optim.SGD(organism.model.parameters(), lr=0.01)
    organism.soul = SimpleNamespace(heartbeat=lambda status: None)
    organism._apply_lr_schedule = lambda step: None
    report = {"forgetting": 0.0}

    organism._train_cycle(10, report)

    assert report["metrics"]["fresh"]["count"] == 8
    assert report["metrics"]["replay"]["count"] == 2
    assert report["metrics"]["replay_fraction"] == pytest.approx(0.20)
    assert report["heldout"]["batches"] == 2
    assert organism.model.batch_sizes == [1] * 14  # 10 train + 2 replay eval + 2 heldout


def test_canary_training_rejects_unprovable_weighted_sources() -> None:
    organism = DarwinOrganism.__new__(DarwinOrganism)
    organism.weighted_sources = [(np.arange(64), 1.0)]
    organism.cfg = SimpleNamespace(holdout_tokens=32)
    with pytest.raises(RuntimeError, match="weighted.*holdout exclusion"):
        organism._validate_holdout_training_source()
```

- [ ] **Step 2: Run the targeted tests and verify red**

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& .\.venv_nitro\Scripts\python.exe -m pytest src\\tests\\test_organism_causal_runtime.py -q -p no:cacheprovider
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

Expected: failures for missing `train_token_ids`, `metrics`, `heldout`, and validation method.

- [ ] **Step 3: Wire split and fixed starts immediately after the full corpus load**

Update the test imports with `split_tail_holdout` and `fixed_batch_starts` from
the new module. Add runtime imports, including the dataclass serializer used by
report records:

```python
from dataclasses import asdict, dataclass, field

from f51_darwin.training_observability import (
    HeldoutResult,
    MetricChannels,
    append_jsonl_fsync,
    evaluate_fixed_holdout,
    fixed_batch_starts,
    split_tail_holdout,
)
```

Add config fields:

```python
    holdout_tokens: int = 0
    holdout_batches: int = 16
    holdout_seed: int = 999
    metrics_jsonl: str | None = None
    canary_run_id: str | None = None
    canary_metadata_json: str | None = None
```

After `self.token_count = len(self.token_ids)` in `bootstrap()`:

```python
        self.train_token_ids = self.token_ids
        self.holdout_token_ids = None
        self.holdout_starts: tuple[tuple[int, ...], ...] = ()
        if self.cfg.holdout_tokens > 0:
            split = split_tail_holdout(
                self.token_ids,
                holdout_tokens=self.cfg.holdout_tokens,
                block_size=self.cfg.block_size,
            )
            self.train_token_ids = split.train_tokens
            self.holdout_token_ids = split.holdout_tokens
            self.holdout_starts = fixed_batch_starts(
                token_count=split.holdout_token_count,
                block_size=self.cfg.block_size,
                batch_size=self.cfg.batch_size,
                batches=self.cfg.holdout_batches,
                seed=self.cfg.holdout_seed,
            )
            self.holdout_definition = {
                "source_token_count": split.source_token_count,
                "train_stop": split.train_stop,
                "holdout_start": split.holdout_start,
                "holdout_token_count": split.holdout_token_count,
                "seed": self.cfg.holdout_seed,
                "batches": self.cfg.holdout_batches,
            }
            manifest_path = Path(str(self.token_path) + ".manifest.json")
            if not manifest_path.is_file():
                raise RuntimeError(f"token identity manifest missing: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if int(manifest.get("tokens", -1)) != self.token_count:
                raise RuntimeError("token manifest count disagrees with loaded source")
            self.token_source_identity = {
                **token_source_state(self.token_path, self.token_count),
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": manifest.get("sha256"),
                "dtype": manifest.get("dtype"),
                "little_endian": manifest.get("little_endian"),
            }
            if not self.cfg.canary_metadata_json:
                raise RuntimeError("canary runtime metadata JSON is required")
            self.canary_metadata = json.loads(
                Path(self.cfg.canary_metadata_json).read_text(encoding="utf-8")
            )
```

- [ ] **Step 4: Replace mixed loss accounting with channel-specific accounting and heldout records**

Add these methods:

```python
    def _validate_holdout_training_source(self) -> None:
        if self.cfg.holdout_tokens <= 0:
            return
        if getattr(self, "weighted_sources", None):
            raise RuntimeError("weighted sources make holdout exclusion unprovable")
        if self.holdout_token_ids is None or len(self.holdout_starts) != self.cfg.holdout_batches:
            raise RuntimeError("canary holdout is not fully initialized")

    def evaluate_holdout(self, *, phase: str) -> HeldoutResult:
        self._validate_holdout_training_source()
        result = evaluate_fixed_holdout(
            self.model,
            self.holdout_token_ids,
            starts=self.holdout_starts,
            block_size=self.cfg.block_size,
            device=self.device,
            amp_dtype=self.amp_dtype,
        )
        self._append_metric_record(phase=phase, heldout=result)
        return result

    def _append_metric_record(self, *, phase: str, heldout: HeldoutResult | None) -> None:
        if not self.cfg.metrics_jsonl:
            return
        metrics = self.metric_channels.snapshot(elapsed_sec=max(time.time() - self._metrics_started_at, 1e-9))
        append_jsonl_fsync(Path(self.cfg.metrics_jsonl), {
            "schema_version": 1,
            "run_id": self.cfg.canary_run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "pid": os.getpid(),
            "cycle": self.cycle,
            "step": self.total_steps,
            "checkpoint": str(getattr(self, "resume_checkpoint_path", "")),
            "base_checkpoint_id": getattr(self, "base_checkpoint_id", None),
            "git_commit": self.canary_metadata.get("git_commit"),
            "command": self.canary_metadata.get("command"),
            "checkpoint_sha256": self.canary_metadata.get("base_checkpoint_sha256"),
            "config_identity": self.canary_metadata.get("config_identity"),
            "driver": self.canary_metadata.get("driver"),
            "gpus": self.canary_metadata.get("gpus"),
            "starting_nvidia_event_record_id": self.canary_metadata.get("starting_nvidia_event_record_id"),
            "token_source": self.token_source_identity,
            "tokenizer_id": self.tokenizer_id,
            "holdout": getattr(self, "holdout_definition", None),
            "metrics": metrics,
            "heldout": None if heldout is None else asdict(heldout),
        })
```

At `_train_cycle` entry initialize and validate:

```python
        self._validate_holdout_training_source()
        loader_tokens = getattr(self, "train_token_ids", self.token_ids)
        self.metric_channels = MetricChannels.create(window_size=20)
        self._metrics_started_at = time.time()
```

Use `loader_tokens` in `CausalLMDataLoader`. Set `batch_kind = "replay" if replay_batch is not None else "fresh"`. For every non-finite or brainstem-skipped step increment `self.metric_channels.skipped_updates`. After a successful backward/optimizer accounting, record the exact forward values:

```python
            lm_loss_val = (
                float(output.lm_loss.detach().cpu())
                if getattr(output, "lm_loss", None) is not None else loss_val
            )
            self.metric_channels.record_update(
                batch_kind=batch_kind,
                total_loss=loss_val,
                lm_loss=lm_loss_val,
                tokens=batch.numel(),
            )
```

At each log cadence print both windows, never the current coincident batch:

```python
                snapshot = self.metric_channels.snapshot(elapsed_sec=time.time() - self._metrics_started_at)
                fresh = snapshot["fresh"]
                replay = snapshot["replay"]
                print(
                    f"  step={self.total_steps:>7d} kind={batch_kind} | "
                    f"fresh_total={fresh['total_loss']} fresh_lm={fresh['lm_loss']} | "
                    f"replay_total={replay['total_loss']} replay_lm={replay['lm_loss']} | "
                    f"tok/s={snapshot['tokens_per_sec']:.0f}",
                    flush=True,
                )
```

Initialize `last_heldout = None` before the loop. At each eval cadence set
`last_heldout = self.evaluate_holdout(phase="periodic")`. Reuse that exact
result at cycle end when the final step was already an evaluation step; do not
perform a duplicate model forward merely to change the record phase. Finalize:

```python
        metrics = self.metric_channels.snapshot(elapsed_sec=time.time() - self._metrics_started_at)
        final_heldout = last_heldout
        if self.holdout_starts and final_heldout is None:
            final_heldout = self.evaluate_holdout(phase="cycle_end")
        elif final_heldout is not None:
            self._append_metric_record(phase="cycle_end", heldout=final_heldout)
        report["steps"] = self.metric_channels.successful_updates + self.metric_channels.skipped_updates
        report["metrics"] = metrics
        report["fresh_train_count"] = self.metric_channels.fresh.count
        report["replay_train_count"] = self.metric_channels.replay.count
        report["replay_fraction"] = metrics["replay_fraction"]
        report["heldout"] = None if final_heldout is None else asdict(final_heldout)
```

Keep existing replay-buffer evaluation as a distinct `replay_eval_loss`; do not label it heldout. Add `"observability": report.get("metrics")` and `"heldout": report.get("heldout")` to the checkpoint organism payload through the existing embedded `report` object; no new top-level checkpoint schema is required.

- [ ] **Step 5: Run the targeted test and full CPU suite**

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& .\.venv_nitro\Scripts\python.exe -m py_compile src\\scripts\\darwin_organism.py src\\f51_darwin\\training_observability.py
& .\.venv_nitro\Scripts\python.exe -m pytest src\\tests\\test_training_observability.py src\\tests\\test_organism_causal_runtime.py -q -p no:cacheprovider
& .\.venv_nitro\Scripts\python.exe -m pytest -q -p no:cacheprovider
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

Expected: compile succeeds, focused tests pass, full suite passes with no CUDA allocation.

- [ ] **Step 6: Commit runtime observability**

```powershell
git add src/scripts/darwin_organism.py src/tests/test_organism_causal_runtime.py
git commit -m "feat: add disjoint organism training metrics"
```

---

### Task 4: Make one-shot `cycle` a strict isolated canary CLI

**Files:**

- Modify: `src/scripts/darwin_organism.py:1629-1720`
- Modify: `src/tests/test_organism_causal_runtime.py`

**Interfaces:**

```text
darwin_organism.py cycle --canary --cycles 1 --steps 250
  --checkpoint-root <external-isolated-directory>
  --metrics-jsonl <external-run-directory>/metrics.jsonl
  --canary-metadata-json <external-run-directory>/runtime_metadata.json
  --run-id <frozen-id>
  --holdout-tokens 1048576 --holdout-batches 16 --holdout-seed 999
  --require-clean-worktree --base-checkpoint-sha256 <exact-sha256>
```

```python
def tracked_worktree_state(root: Path) -> tuple[str, str]: ...
def validate_canary_paths(*, canonical_root: Path, canary_root: Path) -> None: ...
def validate_canary_shape(args: argparse.Namespace) -> None: ...
def sha256_file(path: Path) -> str: ...
```

- [ ] **Step 1: Add failing CLI guard tests**

```python
def test_canary_root_cannot_equal_or_contain_canonical_pointer(tmp_path: Path) -> None:
    canonical = tmp_path / "03_CHECKPOINTS"
    canonical.mkdir()
    with pytest.raises(ValueError, match="isolated"):
        validate_canary_paths(canonical_root=canonical, canary_root=canonical)
    with pytest.raises(ValueError, match="isolated"):
        validate_canary_paths(canonical_root=canonical, canary_root=tmp_path)
    # A child root owns a separate pointer and does not contain the canonical pointer.
    validate_canary_paths(canonical_root=canonical, canary_root=canonical / "canary_recovery")


def test_cycle_command_joins_async_save_before_return() -> None:
    organism = DarwinOrganism.__new__(DarwinOrganism)
    calls: list[str] = []
    organism._join_pending_save = lambda: calls.append("joined")
    finish_one_shot_cycle(organism)
    assert calls == ["joined"]


def test_canary_requires_exact_geometry() -> None:
    args = SimpleNamespace(command="cycle", canary=True, cycles=2, steps=250)
    with pytest.raises(ValueError, match="one cycle"):
        validate_canary_shape(args)
```

- [ ] **Step 2: Run and confirm missing guard functions**

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& .\.venv_nitro\Scripts\python.exe -m pytest src\\tests\\test_organism_causal_runtime.py -q -p no:cacheprovider
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

Expected: import or name failures for the canary guard functions.

- [ ] **Step 3: Add canary arguments and fail-closed guards**

```python
def tracked_worktree_state(root: Path) -> tuple[str, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return commit, dirty


def validate_canary_paths(*, canonical_root: Path, canary_root: Path) -> None:
    canonical = canonical_root.resolve()
    canary = canary_root.resolve()
    canonical_pointer = canonical / "organism_latest.json"
    if canary == canonical or canary in canonical_pointer.parents:
        raise ValueError("canary checkpoint root must be isolated from canonical root")


def validate_canary_shape(args: argparse.Namespace) -> None:
    if args.command != "cycle" or args.cycles != 1 or args.steps != 250:
        raise ValueError("canary requires exactly one cycle and 250 steps")


def finish_one_shot_cycle(organism: DarwinOrganism) -> None:
    organism._join_pending_save()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
```

Change the runtime imports to include `hashlib` and `subprocess`; `sha256_file`
must stream the ~11 GB checkpoint and must never call `read_bytes()` on it.

Add parser flags:

```python
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--metrics-jsonl", default=None)
    parser.add_argument("--canary-metadata-json", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--holdout-tokens", type=int, default=0)
    parser.add_argument("--holdout-batches", type=int, default=16)
    parser.add_argument("--holdout-seed", type=int, default=999)
    parser.add_argument("--require-clean-worktree", action="store_true")
    parser.add_argument("--base-checkpoint-sha256", default=None)
```

Before building config:

```python
    if args.canary:
        validate_canary_shape(args)
        if not all((args.resume, args.checkpoint_root, args.metrics_jsonl,
                    args.canary_metadata_json, args.run_id)):
            parser.error("--canary requires resume, isolated root, metrics, metadata, and run ID")
        canonical_root = resolve_organism_checkpoints_root(ROOT)
        validate_canary_paths(canonical_root=canonical_root, canary_root=Path(args.checkpoint_root))
        commit, dirty = tracked_worktree_state(ROOT)
        if args.require_clean_worktree and dirty:
            parser.error(f"tracked worktree is dirty: {dirty}")
        if not args.base_checkpoint_sha256:
            parser.error("--canary requires --base-checkpoint-sha256")
        actual_sha = sha256_file(Path(args.resume))
        if actual_sha.lower() != args.base_checkpoint_sha256.lower():
            parser.error("base checkpoint SHA-256 mismatch")
        args.frozen_git_commit = commit
```

Set config values from flags and use `Path(args.checkpoint_root)` only when explicitly supplied:

```python
        checkpoint_root=str(Path(args.checkpoint_root).resolve()) if args.checkpoint_root else str(resolve_organism_checkpoints_root(ROOT)),
        holdout_tokens=args.holdout_tokens,
        holdout_batches=args.holdout_batches,
        holdout_seed=args.holdout_seed,
        metrics_jsonl=args.metrics_jsonl,
        canary_run_id=args.run_id,
        canary_metadata_json=args.canary_metadata_json,
```

In the `cycle` branch, initialize the empty metric channels before the baseline
evaluation; otherwise the first ledger append has no accumulator. Then run one
cycle, join its asynchronous save, and evaluate the post-topology candidate:

```python
        if args.canary:
            org.metric_channels = MetricChannels.create(window_size=20)
            org._metrics_started_at = time.time()
            baseline_heldout = org.evaluate_holdout(phase="baseline")
        report = org.run_cycle(steps=args.steps)
        finish_one_shot_cycle(org)
        if args.canary:
            final_heldout = org.evaluate_holdout(phase="candidate")
            print(json.dumps({
                "status": "candidate_saved",
                "run_id": args.run_id,
                "git_commit": args.frozen_git_commit,
                "checkpoint_root": str(Path(args.checkpoint_root).resolve()),
                "baseline_heldout": asdict(baseline_heldout),
                "heldout": asdict(final_heldout),
            }, sort_keys=True), flush=True)
```

- [ ] **Step 4: Run focused tests and parser smoke**

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& .\.venv_nitro\Scripts\python.exe -m pytest src\\tests\\test_organism_causal_runtime.py src\\tests\\test_training_observability.py -q -p no:cacheprovider
& .\.venv_nitro\Scripts\python.exe src\\scripts\\darwin_organism.py cycle --help
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

Expected: tests pass; help lists all canary flags; no model is loaded.

- [ ] **Step 5: Commit isolated canary CLI**

```powershell
git add src/scripts/darwin_organism.py src/tests/test_organism_causal_runtime.py
git commit -m "feat: add isolated one shot canary command"
```

---

### Task 5: Extend the official 1.6B launcher with reversible power and GPU/event monitoring

**Files:**

- Modify: `src/scripts/start_overnight_16b.ps1`
- Create: `src/tests/test_canary_launcher_contract.py`

**Interfaces:**

```powershell
src/scripts/start_overnight_16b.ps1
  [-Canary] [-Launch]
  [-CanarySteps 250]
  [-CanaryHoldoutTokens 1048576]
  [-PostRunObservationSeconds 120]
  [-CanaryRoot <external-path>]
```

- [ ] **Step 1: Add a failing launcher contract test**

```python
from pathlib import Path


def test_official_launcher_has_bounded_isolated_canary_contract() -> None:
    text = Path("src/scripts/start_overnight_16b.ps1").read_text(encoding="utf-8")
    assert "[switch]$Canary" in text
    assert "[int]$CanarySteps = 250" in text
    assert '"cycle"' in text
    assert '"--checkpoint-root"' in text
    assert '"--metrics-jsonl"' in text
    assert '"--holdout-tokens"' in text
    assert '"--base-checkpoint-sha256"' in text
    assert "WaitForExit" in text
    assert "Start-Sleep -Seconds $PostRunObservationSeconds" in text
    assert "13, 14, 153" in text
    assert "LiveKernelEvent" in text
    assert "SUB_PCIEXPRESS" in text
    assert "ASPM" in text
```

- [ ] **Step 2: Run and confirm the contract test fails**

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& .\.venv_nitro\Scripts\python.exe -m pytest src\\tests\\test_canary_launcher_contract.py -q -p no:cacheprovider
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

Expected: assertion failure for missing `$Canary`.

- [ ] **Step 3: Add launcher parameters and preflight manifest fields**

Extend `param(...)`:

```powershell
    [switch]$Canary,
    [int]$CanarySteps = 250,
    [int64]$CanaryHoldoutTokens = 1048576,
    [int]$PostRunObservationSeconds = 120,
    [string]$CanaryRoot = "",
```

After resolving paths, define the isolated paths and exact base hash:

```powershell
$baseCheckpointSha256 = (Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256).Hash
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
if ($Canary) {
    if ($CanarySteps -ne 250) { throw "Recovery canary is fixed at 250 steps." }
    if (-not $CanaryRoot) {
        $CanaryRoot = Join-Path $checkpointRoot "canary_recovery_$stamp"
    }
    $CanaryRoot = [IO.Path]::GetFullPath($CanaryRoot)
    if ($CanaryRoot -eq $checkpointRoot -or
        $latestPointerPath.StartsWith($CanaryRoot + [IO.Path]::DirectorySeparatorChar,
                                      [StringComparison]::OrdinalIgnoreCase)) {
        throw "Canary root must be isolated from the canonical checkpoint root."
    }
}
```

Add a second process guard matching both `run247` and `cycle.*--canary`. Add manifest fields for `mode`, frozen commit, full command, base hash, canonical pointer path, isolated root, holdout definition, driver/GPU bus IDs, and the event-record baseline.

- [ ] **Step 4: Add reversible PCIe state capture and event/GPU helpers**

```powershell
function Get-PcieAspmState {
    $active = (& powercfg /GETACTIVESCHEME | Out-String)
    $query = (& powercfg /QUERY SCHEME_CURRENT SUB_PCIEXPRESS ASPM | Out-String)
    $acMatches = [regex]::Matches($query, '0x[0-9a-fA-F]{8}')
    if ($acMatches.Count -lt 2) { throw "Unable to parse the current PCIe ASPM AC value." }
    $acIndex = [Convert]::ToInt32($acMatches[$acMatches.Count - 2].Value.Substring(2), 16)
    return [ordered]@{
        active_scheme = $active.Trim()
        query = $query.Trim()
        ac_index = $acIndex
        rollback_command = "powercfg /SETACVALUEINDEX SCHEME_CURRENT SUB_PCIEXPRESS ASPM $acIndex; powercfg /SETACTIVE SCHEME_CURRENT"
    }
}

function Disable-PcieAspmOnAc {
    & powercfg /SETACVALUEINDEX SCHEME_CURRENT SUB_PCIEXPRESS ASPM 0 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Failed to disable PCIe ASPM on AC." }
    & powercfg /SETACTIVE SCHEME_CURRENT | Out-Null
}

function Get-NvidiaEventsSince {
    param([long]$AfterRecordId)
    return @(Get-WinEvent -FilterHashtable @{ LogName='System'; ProviderName='nvlddmkm'; Id=13,14,153 } `
        -ErrorAction SilentlyContinue | Where-Object { $_.RecordId -gt $AfterRecordId } | ForEach-Object {
            [ordered]@{ id=$_.Id; record_id=$_.RecordId; time=$_.TimeCreated.ToUniversalTime().ToString('o'); message=$_.Message }
        })
}

function Get-LiveKernel141Since {
    param([datetime]$Since)
    return @(Get-WinEvent -FilterHashtable @{
        LogName='Application'; ProviderName='Windows Error Reporting'; Id=1001; StartTime=$Since
    } -ErrorAction SilentlyContinue | Where-Object {
        $_.Message -match 'LiveKernelEvent' -and $_.Message -match '(?m)^P1:\s*141\s*$'
    })
}

function Get-GpuSample {
    $rows = & "C:\Windows\System32\nvidia-smi.exe" `
        --query-gpu=timestamp,index,name,pci.bus_id,utilization.gpu,memory.used,temperature.gpu,power.draw `
        --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0) { throw "nvidia-smi sample failed" }
    return @($rows)
}
```

The dry-run records `Get-PcieAspmState` and the exact planned `powercfg` command but does not change power state. The change occurs only inside `if ($Canary -and $Launch)` immediately before process launch.

- [ ] **Step 5: Build the bounded command, monitor without intervention, and close the report**

For canary mode, replace the `run247` argument vector with:

```powershell
$metricsJsonl = Join-Path $runDir "metrics.jsonl"
$runtimeMetadata = Join-Path $runDir "runtime_metadata.json"
$arguments = @(
    "-u", "src/scripts/darwin_organism.py", "cycle", "--canary",
    "--config", $configPath,
    "--token-bin", $tokenBin,
    "--resume", $checkpointPath,
    "--cycles", "1",
    "--steps", [string]$CanarySteps,
    "--block-size", [string]$BlockSize,
    "--batch-size", [string]$BatchSize,
    "--lr", [string]$LearningRate,
    "--device", "cuda",
    "--eval-every", [string]$EvalEvery,
    "--checkpoint-root", $CanaryRoot,
    "--metrics-jsonl", $metricsJsonl,
    "--canary-metadata-json", $runtimeMetadata,
    "--run-id", $stamp,
    "--holdout-tokens", [string]$CanaryHoldoutTokens,
    "--holdout-batches", "16",
    "--holdout-seed", "999",
    "--require-clean-worktree",
    "--base-checkpoint-sha256", $baseCheckpointSha256
)
```

Before `Start-Process`, write `$runtimeMetadata` atomically with the frozen Git
commit, the exact joined argument vector, base checkpoint SHA-256, config
SHA-256, tokenizer identity, corpus manifest identity, NVIDIA driver, both GPU
names/bus IDs, and `$startingNvidiaRecordId`. The runtime refuses to train if
this file is missing.

After `Start-Process`, monitor read-only until exit:

```powershell
$samples = [Collections.Generic.List[object]]::new()
while (-not $process.HasExited) {
    $samples.Add([ordered]@{ utc=(Get-Date).ToUniversalTime().ToString('o'); rows=(Get-GpuSample) })
    Start-Sleep -Seconds 5
    $process.Refresh()
}
$process.WaitForExit()
$exitCode = $process.ExitCode
Start-Sleep -Seconds $PostRunObservationSeconds
$newNvidiaEvents = Get-NvidiaEventsSince -AfterRecordId $startingNvidiaRecordId
$liveKernel141 = @(Get-LiveKernel141Since -Since $launchTime)
```

Then require exit code 0, find the isolated `organism_latest.json`, run
`inspect_organism_checkpoint.py --verify-identity`, parse the last complete
baseline/candidate ledger records, and compute peak temperatures from samples.
Write those normalized fields to `canary_gate_input.json`; add a module entrypoint
to `training_observability.py` that accepts `--gate-json`, constructs
`CanaryGateInput`, calls `decide_canary`, and prints JSON. Invoke it exactly as:

```powershell
$decisionText = & $python -m f51_darwin.training_observability `
    --gate-json $gateInputPath 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) { throw "Canary acceptance evaluation failed: $decisionText" }
$decision = $decisionText | ConvertFrom-Json
```

Persist `canary_final_report.json` with all raw inputs and the decision, then
write `(Get-FileHash ... -Algorithm SHA256).Hash` to
`canary_final_report.json.sha256`. Do not copy or point anything into the
canonical checkpoint directory.

- [ ] **Step 6: Parse the PowerShell AST and run contract tests**

```powershell
$tokens=$null; $errors=$null
[System.Management.Automation.Language.Parser]::ParseFile(
  (Resolve-Path 'src\\scripts\\start_overnight_16b.ps1'), [ref]$tokens, [ref]$errors
) | Out-Null
if ($errors.Count) { $errors | Format-List | Out-String | Write-Error }
$env:CUDA_VISIBLE_DEVICES='-1'
& .\.venv_nitro\Scripts\python.exe -m pytest src\\tests\\test_canary_launcher_contract.py src\\tests\\test_training_observability.py -q -p no:cacheprovider
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

Expected: no parser errors and all tests pass.

- [ ] **Step 7: Commit launcher orchestration**

```powershell
git add src/scripts/start_overnight_16b.ps1 src/tests/test_canary_launcher_contract.py
git commit -m "feat: gate 1.6b recovery canary"
```

---

### Task 6: Make the external comparison memory-bounded, persist it, and document the operator flow

**Files:**

- Modify: `src/f51_darwin/checkpoint_eval.py`
- Modify: `src/scripts/compare_checkpoints.py`
- Modify: `src/tests/test_checkpoint_eval.py`
- Modify: `src/scripts/README.md`
- Modify: `governance/docs/STATUS_ATUAL.md`

**Interfaces:**

```text
compare_checkpoints.py ... --output <external-run-dir>/external_comparison.json
```

```python
def evaluate_checkpoint_losses(checkpoint: str | Path, token_ids: object,
                               config: CheckpointEvalConfig) -> tuple[LossSummary, LossSummary]: ...
```

- [ ] **Step 1: Add a failing output-persistence test**

Add a memory-lifetime test and a helper extraction test. The fake loader refuses
to load the candidate while the baseline model remains live:

```python
import gc
import weakref

import f51_darwin.checkpoint_eval as checkpoint_eval
from scripts.compare_checkpoints import write_result


def test_compare_releases_baseline_before_loading_candidate(monkeypatch, tmp_path: Path) -> None:
    active: list[weakref.ReferenceType[torch.nn.Module]] = []

    def fake_load(path, map_location):
        gc.collect()
        assert not [reference for reference in active if reference() is not None]
        model = F51DarwinModel(DarwinConfig(
            vocab_size=32, context_length=8, d_model=16,
            n_layers=4, n_heads=4,
        ))
        active.append(weakref.ref(model))
        return model, model.config, {}

    monkeypatch.setattr(checkpoint_eval, "load_model_from_checkpoint", fake_load)
    baseline = tmp_path / "baseline.pt"
    candidate = tmp_path / "candidate.pt"
    baseline.touch()
    candidate.touch()
    tokens = tmp_path / "tokens.bin"
    _write_tokens(tokens, [index % 32 for index in range(128)])
    checkpoint_eval.compare_checkpoints(
        baseline, candidate, project_root=tmp_path, token_bin=tokens,
        config=CheckpointEvalConfig(block_size=8, batch_size=1, max_batches=1),
    )


def test_compare_checkpoint_result_can_be_persisted(tmp_path: Path) -> None:
    output = tmp_path / "comparison.json"
    write_result(output, '{"promotion":{"passed":true}}')
    assert output.read_text(encoding="utf-8") == '{"promotion":{"passed":true}}\n'
```

- [ ] **Step 2: Run and confirm the helper is missing**

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& .\.venv_nitro\Scripts\python.exe -m pytest src\\tests\\test_checkpoint_eval.py -q -p no:cacheprovider
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

Expected: import failure for `write_result` or the fake loader's live-baseline
assertion fails; no real checkpoint is loaded.

- [ ] **Step 3: Evaluate one checkpoint at a time and release it before the next load**

Implement a single-checkpoint lifetime boundary:

```python
def evaluate_checkpoint_losses(
    checkpoint: str | Path,
    token_ids: object,
    config: CheckpointEvalConfig,
) -> tuple[LossSummary, LossSummary]:
    device = torch.device(config.device)
    model, _, _ = load_model_from_checkpoint(checkpoint, map_location=device)
    try:
        model = model.to(device)
        heldout = evaluate_model_loss(model, token_ids, config, seed=config.eval_seed)
        replay = evaluate_model_loss(model, token_ids, config, seed=config.replay_seed)
        return heldout, replay
    finally:
        del model
        gc.collect()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
```

Keep the public `compare_checkpoints` signature stable, resolve/load the token
file as today, then evaluate sequentially:

```python
    token_ids = load_token_ids(token_path, max_tokens=config.max_tokens)
    heldout_baseline, replay_baseline = evaluate_checkpoint_losses(
        baseline_checkpoint, token_ids, config,
    )
    heldout_candidate, replay_candidate = evaluate_checkpoint_losses(
        candidate_checkpoint, token_ids, config,
    )
```

Import `gc`. Remove the old simultaneous `baseline_model`/`candidate_model`
loads completely.

- [ ] **Step 4: Add atomic output writing and CLI flag**

```python
def write_result(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)
```

Add `parser.add_argument("--output", default=None)` and replace the final print with:

```python
    payload = result.to_json()
    if args.output:
        write_result(Path(args.output), payload)
    print(payload)
```

- [ ] **Step 5: Document exact dry-run, launch, rollback, and non-promotion semantics**

Add to `src/scripts/README.md`:

```markdown
## Recovery canary 1.6B

Run the official gate first without `-Launch`. `-Canary` validates the exact
base, clean tracked commit, isolated output root, corpus, GPUs, power setting,
and command without changing the system. Add `-Launch` only after reading the
manifest. The canary performs one 250-step `cycle`, never `run247`, and waits
for the checkpoint plus the two-minute GPU-event window. Passing means
`stable_canary`; it does not promote the model or start continuous training.
The launch manifest records the original PCIe ASPM AC value and rollback
command. No reboot is required or performed.
```

Update `governance/docs/STATUS_ATUAL.md` only with the recovery protocol and current base checkpoint. Do not claim the canary passed before Task 7 generates the real artifacts.

- [ ] **Step 6: Run focused and full CPU validation**

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& .\.venv_nitro\Scripts\python.exe -m py_compile src\\scripts\\darwin_organism.py src\\scripts\\compare_checkpoints.py src\\f51_darwin\\training_observability.py
& .\.venv_nitro\Scripts\python.exe -m pytest src\\tests\\test_checkpoint_eval.py src\\tests\\test_training_observability.py src\\tests\\test_organism_causal_runtime.py src\\tests\\test_canary_launcher_contract.py -q -p no:cacheprovider
& .\.venv_nitro\Scripts\python.exe -m pytest -q -p no:cacheprovider
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

Expected: compile succeeds; focused and full CPU suites pass.

- [ ] **Step 7: Commit docs and external evidence persistence**

```powershell
git add src/f51_darwin/checkpoint_eval.py src/scripts/compare_checkpoints.py src/tests/test_checkpoint_eval.py src/scripts/README.md governance/docs/STATUS_ATUAL.md
git commit -m "docs: define 1.6b canary operator flow"
```

---

### Task 7: Run the real preflight and 250-step canary, then save evidence

**Files:**

- Create: `.agent_bus/CODEX_16B_CANARY_<YYYY-MM-DD_HHMM>.md`
- External only: `F51-Dataset-Organizado/04_MANIFESTOS/...`
- External only: `F51-Dataset-Organizado/03_CHECKPOINTS/canary_recovery_.../...`

- [ ] **Step 1: Verify the repository is frozen and no competing process exists**

```powershell
git status --short
git log --oneline -5
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'darwin_organism\.py\s+(run247|cycle)' } |
  Select-Object ProcessId, CommandLine
& 'C:\Windows\System32\nvidia-smi.exe'
```

Expected: clean tracked worktree, no training/canary process, both GPUs visible and idle. If not, stop; do not kill anything automatically.

- [ ] **Step 2: Run the complete CPU gate before allocating CUDA**

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& .\.venv_nitro\Scripts\python.exe -m py_compile src\\scripts\\darwin_organism.py src\\f51_darwin\\darwin_x.py src\\f51_darwin\\training_observability.py
& .\.venv_nitro\Scripts\python.exe -m pytest -q -p no:cacheprovider
& .\.venv_nitro\Scripts\python.exe src\\scripts\\darwin_inventory.py
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

Expected: compile and tests pass; inventory identifies the 1.6B lineage and external dataset roots.

- [ ] **Step 3: Run the official launcher without `-Launch`**

```powershell
& 'C:\Program Files\PowerShell\7\pwsh.exe' -NoProfile -ExecutionPolicy Bypass `
  -File src\\scripts\\start_overnight_16b.ps1 -Canary
```

Expected: `ready=true`, exact SHA-256 `4416A9AF6E1A3105FCF207616E7A134D2484CB3E0B9EE36281921446230CF6CC`, cycle 68/step 36501, isolated candidate root, 1,048,576-token tail holdout, both GPU bus IDs, current PCIe ASPM value, and the planned non-reboot power command. No system setting changes yet.

- [ ] **Step 4: Launch and let the monitor finish without intervention**

```powershell
& 'C:\Program Files\PowerShell\7\pwsh.exe' -NoProfile -ExecutionPolicy Bypass `
  -File src\\scripts\\start_overnight_16b.ps1 -Canary -Launch
```

Expected: one process, one 250-step cycle, real dual-GPU forward/backward, isolated checkpoint write joined, clean exit, and two-minute observation. Do not reboot, restart, kill, or relaunch on failure.

- [ ] **Step 5: Verify the candidate offline and run the independent external comparison**

Use paths from `canary_final_report.json`:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& .\.venv_nitro\Scripts\python.exe src\\scripts\\inspect_organism_checkpoint.py `
  $candidate --config src\\configs\\darwin_x_1.6b_nitro.yaml --verify-identity
& .\.venv_nitro\Scripts\python.exe src\\scripts\\compare_checkpoints.py `
  --baseline $base --candidate $candidate `
  --tokens 'C:\Users\marco\Desktop\F51-Dataset-Organizado\01_TOKENIZADOS\00_CORPUS_PRINCIPAL_tokens_feast.bin' `
  --device cpu --block-size 64 --batch-size 1 --max-batches 16 `
  --output $externalComparison
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

Expected: strict config/lineage/topology/model/AdamW verification passes and
external comparison is persisted. Derive `external_benchmark_passed` from the
independent non-regression contract, not from `promotion.passed`:
`heldout_candidate.loss <= heldout_baseline.loss + 0.02` and
`replay_candidate.loss <= replay_baseline.loss + 0.02`. Preserve the existing
promotion decision unchanged in the report; its verified-generation and
improvement requirements answer a stronger question than canary stability.

- [ ] **Step 6: Recompute the final acceptance decision with the external result**

Update `canary_final_report.json` with the external comparison path/hash and rerun `decide_canary`. Require:

```text
process exit = 0
successful + skipped = 250
fresh count = 200 (unless a documented skipped update changes successful denominator)
replay count = 50 (within one successful-update rounding unit)
finite fresh/replay/heldout metrics
candidate heldout <= baseline heldout + 0.02
peak GPU temperatures < 85 C
no new nvlddmkm 13/14/153
no LiveKernelEvent 141 through T+120 seconds
candidate checkpoint verified
external benchmark passed
```

Expected final label: `stable_canary`, never `model_improved`.

- [ ] **Step 7: Write the dated evidence note and commit only repository evidence**

The note must contain command lines, commit, base/candidate paths and hashes, cycle/step, metric summary, replay accounting, peak GPU samples, event-log record IDs, power setting before/after plus rollback command, checkpoint verification result, external benchmark result, and final status.

```powershell
git add .agent_bus/CODEX_16B_CANARY_*.md governance/docs/STATUS_ATUAL.md
git commit -m "evidence: record 1.6b recovery canary"
git status --short
```

Expected: evidence is saved locally and tracked state is clean. Do not promote the isolated pointer or start `run247`; those require a new explicit decision from Marco.

---

## Final Verification Checklist

- [ ] Search for unfinished placeholders:

```powershell
rg -n 'TODO|FIXME|TBD|placeholder|similar to|implement later' `
  src/f51_darwin/training_observability.py src/scripts/darwin_organism.py `
  src/scripts/start_overnight_16b.ps1 src/tests/test_training_observability.py `
  src/tests/test_canary_launcher_contract.py governance/docs/superpowers/plans/2026-07-15-16b-recovery-observability-canary.md
```

Expected: no implementation placeholders; matches inside this plan's verification wording are acceptable only in the plan itself.

- [ ] Confirm signatures and CLI names match every call site:

```powershell
rg -n 'split_tail_holdout|fixed_batch_starts|evaluate_fixed_holdout|MetricChannels|decide_canary|checkpoint-root|metrics-jsonl|base-checkpoint-sha256' f51_darwin scripts tests
```

- [ ] Confirm canonical-pointer isolation from saved manifests and filesystem paths.
- [ ] Confirm the launcher performs no driver install, TDR edit, reboot, process kill, checkpoint delete, promotion, or `run247` start in `-Canary` mode.
- [ ] Confirm the full CPU suite ran before the first CUDA allocation.
- [ ] Confirm all claims in the final evidence note point to a command output, file/hash, checkpoint load, JSONL record, GPU sample, or Windows event record.
