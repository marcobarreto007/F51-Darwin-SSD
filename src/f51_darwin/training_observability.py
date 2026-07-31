from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import subprocess
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════
# Tail holdout — positionally disjoint token split
# ═══════════════════════════════════════════════════════════

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
    # NOTE: sampling is tile-aligned and without replacement (distinct,
    # non-overlapping blocks). This is a behavior change from the earlier
    # with-replacement random-offset version — the concrete blocks returned
    # for a given seed are NOT the same as before this was fixed. `starts`
    # is recomputed fresh at every process bootstrap (not persisted in the
    # checkpoint's training_data_contract, which only stores
    # batches/seed/counts), so a process that restarts after this fix is
    # deployed will silently evaluate holdout on a different token window
    # than earlier entries in the same run's metrics.jsonl. Any regression/
    # canary comparison that spans a restart boundary should account for
    # this discontinuity explicitly rather than assume like-for-like.
    if batch_size < 1 or batches < 1:
        raise ValueError("invalid fixed holdout batch geometry")
    num_tiles = token_count // block_size
    requested = batches * batch_size
    if num_tiles < 1 or requested > num_tiles:
        raise ValueError(
            f"requested {requested} distinct blocks but only {num_tiles} "
            f"non-overlapping blocks fit in token_count={token_count} "
            f"with block_size={block_size}"
        )
    rng = random.Random(seed)
    tile_indices = rng.sample(range(num_tiles), requested)
    starts = [index * block_size for index in tile_indices]
    return tuple(
        tuple(starts[row * batch_size:(row + 1) * batch_size])
        for row in range(batches)
    )


# ═══════════════════════════════════════════════════════════
# Metric channels — fresh/replay windowed accounting
# ═══════════════════════════════════════════════════════════

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
    optimizer_steps: int = 0
    skipped_updates: int = 0
    tokens_processed: int = 0

    @classmethod
    def create(cls, *, window_size: int = 20) -> "MetricChannels":
        return cls(ChannelWindow(window_size), ChannelWindow(window_size))

    def record_update(self, *, batch_kind: Literal["fresh", "replay"],
                      total_loss: float, lm_loss: float, tokens: int,
                      optimizer_step: bool = True) -> None:
        channel = self.fresh if batch_kind == "fresh" else self.replay
        channel.add(total_loss=total_loss, lm_loss=lm_loss)
        self.successful_updates += 1
        self.optimizer_steps += int(optimizer_step)
        self.tokens_processed += int(tokens)

    def snapshot(self, *, elapsed_sec: float) -> dict[str, object]:
        replay_count = self.replay.count
        return {
            "fresh": self.fresh.snapshot(),
            "replay": self.replay.snapshot(),
            "successful_updates": self.successful_updates,
            "optimizer_steps": self.optimizer_steps,
            "skipped_updates": self.skipped_updates,
            "tokens_processed": self.tokens_processed,
            "tokens_per_sec": self.tokens_processed / max(elapsed_sec, 1e-9),
            "replay_fraction": replay_count / max(self.successful_updates, 1),
        }


# ═══════════════════════════════════════════════════════════
# Durable JSONL ledger
# ═══════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════
# Fixed side-effect-free heldout evaluator
# ═══════════════════════════════════════════════════════════

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


def _model_buffer_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot mutable buffers so held-out evaluation is observational."""

    return {
        name: value.detach().clone()
        for name, value in model.named_buffers()
    }


@torch.inference_mode()
def _restore_model_buffer_state(
    model: torch.nn.Module,
    saved: Mapping[str, torch.Tensor],
) -> None:
    current = dict(model.named_buffers())
    if current.keys() != saved.keys():
        raise RuntimeError("model buffer topology changed during held-out evaluation")
    for name, value in saved.items():
        target = current[name]
        target.copy_(value.to(device=target.device, dtype=target.dtype))


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
                setattr(moe, name, value.detach().clone())
            else:
                setattr(moe, name, value)


def _nitro_placements(
    model: torch.nn.Module,
) -> list[
    tuple[
        object,
        dict[int, tuple[object, torch.device, torch.dtype, tuple[bool, ...]]],
    ]
]:
    saved = []
    for block in getattr(model, "blocks", ()):
        moe = getattr(block, "moe", None)
        if moe is None or not hasattr(moe, "_expert_gpu"):
            continue
        placements = {}
        for index, expert in enumerate(moe.fine_experts):
            parameters = tuple(expert.parameters())
            if not parameters:
                continue
            placements[index] = (
                moe._expert_gpu.get(index),
                parameters[0].device,
                parameters[0].dtype,
                tuple(parameter.requires_grad for parameter in parameters),
            )
        saved.append((moe, placements))
    return saved


def _restore_nitro_placements(
    saved: list[
        tuple[
            object,
            dict[int, tuple[object, torch.device, torch.dtype, tuple[bool, ...]]],
        ]
    ],
) -> None:
    for moe, placements in saved:
        for index, (assigned, original_device, original_dtype, requires_grad) in placements.items():
            expert = moe.fine_experts[index]
            expert.to(device=original_device, dtype=original_dtype)
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
    buffer_state = _model_buffer_state(model)
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
        _restore_model_buffer_state(model, buffer_state)
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


# ═══════════════════════════════════════════════════════════
# Canary acceptance gate
# ═══════════════════════════════════════════════════════════

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


def external_comparison_passed(
    comparison: Mapping[str, object], *, tolerance: float = 0.02
) -> bool:
    """Evaluate stability only; the stronger promotion decision is unrelated."""
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("external comparison tolerance must be finite and non-negative")

    def loss(section: str) -> float:
        value = comparison.get(section)
        if not isinstance(value, Mapping) or "loss" not in value:
            raise ValueError(f"external comparison is missing {section}.loss")
        result = float(value["loss"])
        if not math.isfinite(result):
            raise ValueError(f"external comparison {section}.loss must be finite")
        return result

    heldout_baseline = loss("heldout_baseline")
    heldout_candidate = loss("heldout_candidate")
    replay_baseline = loss("replay_baseline")
    replay_candidate = loss("replay_candidate")
    return (
        heldout_candidate <= heldout_baseline + tolerance
        and replay_candidate <= replay_baseline + tolerance
    )


def finalize_canary_report(
    runtime_report: Mapping[str, object],
    external_comparison: Mapping[str, object],
    *,
    comparison_path: Path,
    comparison_sha256: str,
) -> dict[str, object]:
    """Recompute the final stable-canary gate without publishing a checkpoint."""
    try:
        int(comparison_sha256, 16)
    except ValueError as exc:
        raise ValueError("external comparison SHA-256 must be hexadecimal") from exc
    if len(comparison_sha256) != 64:
        raise ValueError("external comparison SHA-256 must contain 64 characters")

    report = copy.deepcopy(dict(runtime_report))
    comparison = copy.deepcopy(dict(external_comparison))
    for name, runtime_key in (
        ("baseline", "base_checkpoint"),
        ("candidate", "candidate_checkpoint"),
    ):
        expected = report.get(runtime_key)
        actual = comparison.get(name)
        if not expected or not actual:
            raise ValueError(f"missing {name} checkpoint identity")
        expected_path = os.path.normcase(str(Path(str(expected)).resolve()))
        actual_path = os.path.normcase(str(Path(str(actual)).resolve()))
        if actual_path != expected_path:
            raise ValueError(f"external comparison {name} checkpoint does not match runtime report")

    passed = external_comparison_passed(comparison)
    raw_gate = report.get("gate_input")
    if not isinstance(raw_gate, Mapping):
        raise ValueError("runtime report is missing gate_input")
    gate_input = copy.deepcopy(dict(raw_gate))
    gate_input["external_benchmark_passed"] = passed
    decision = canary_gate_from_mapping(gate_input)

    runtime_failures: list[str] = []
    if int(report.get("process_exit_code", -1)) != 0:
        runtime_failures.append("process_exit")
    if report.get("monitor_errors"):
        runtime_failures.append("monitoring_error")
    if report.get("metrics_truncated_tail"):
        runtime_failures.append("metrics_truncated_tail")
    if not report.get("candidate_checkpoint_sha256"):
        runtime_failures.append("candidate_checkpoint_sha256")
    if runtime_failures:
        failures = list(dict.fromkeys([*decision["failures"], *runtime_failures]))
        decision = {
            "status": "hardware_runtime_failed",
            "passed": False,
            "failures": failures,
        }

    comparison["path"] = str(comparison_path.resolve())
    comparison["sha256"] = comparison_sha256.lower()
    comparison["non_regression_passed"] = passed
    report["gate_input"] = gate_input
    report["external_comparison"] = comparison
    report["decision"] = decision
    report["awaiting_external_benchmark"] = False
    report["automatic_promotion"] = False
    report["finalized_at_utc"] = datetime.now(timezone.utc).isoformat()
    return report


def write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


# ═══════════════════════════════════════════════════════════
# Worktree and canary path guards (Task 4)
# ═══════════════════════════════════════════════════════════

def _git_exe() -> str:
    import shutil as _shutil
    resolved = _shutil.which("git")
    if resolved:
        return resolved
    # Common Windows user install path
    candidate = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Git\cmd\git.exe")
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError("git executable not found — install Git or set PATH")


def tracked_worktree_state(
    root: Path,
    *,
    source_commit: str | None = None,
) -> tuple[str, str]:
    """Resolve source identity locally or from a verified exported archive.

    Cloud deploys intentionally transfer ``git archive`` output without the
    repository metadata.  In that case the deployer must supply the exact
    commit used to build the archive.  A real worktree remains authoritative:
    a supplied identity that disagrees with ``HEAD`` is rejected.
    """
    supplied = (source_commit or "").strip().lower()
    if supplied and (
        len(supplied) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in supplied)
    ):
        raise ValueError("source commit must be a 40- or 64-character hex digest")

    try:
        git = _git_exe()
        commit = subprocess.run(
            [git, "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip().lower()
        dirty = subprocess.run(
            [git, "status", "--porcelain", "--untracked-files=no"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        if not supplied:
            raise RuntimeError(
                "source identity unavailable: no Git worktree and no explicit source commit"
            )
        return supplied, ""

    if supplied and supplied != commit:
        raise RuntimeError(
            f"source commit mismatch: worktree={commit} supplied={supplied}"
        )
    return commit, dirty


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


# ═══════════════════════════════════════════════════════════
# CLI entrypoint for gate evaluation (Task 5)
# ═══════════════════════════════════════════════════════════

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate or finalize a Darwin recovery canary gate.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--gate-json")
    mode.add_argument("--finalize-report")
    parser.add_argument("--external-comparison")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.gate_json:
        raw = json.loads(Path(args.gate_json).read_text(encoding="utf-8-sig"))
        print(json.dumps(canary_gate_from_mapping(raw), sort_keys=True))
        return 0
    if not args.external_comparison or not args.output:
        parser.error("--finalize-report requires --external-comparison and --output")
    report_path = Path(args.finalize_report)
    comparison_path = Path(args.external_comparison)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    final = finalize_canary_report(
        report,
        comparison,
        comparison_path=comparison_path,
        comparison_sha256=sha256_file(comparison_path),
    )
    write_json_atomic(Path(args.output), final)
    print(json.dumps(final, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
