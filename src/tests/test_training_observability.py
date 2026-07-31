from __future__ import annotations

import copy
import json
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from f51_darwin.training_observability import (
    CanaryGateInput,
    HeldoutResult,
    MetricChannels,
    append_jsonl_fsync,
    canary_gate_from_mapping,
    decide_canary,
    evaluate_fixed_holdout,
    external_comparison_passed,
    finalize_canary_report,
    fixed_batch_starts,
    recover_jsonl,
    split_tail_holdout,
)
from f51_darwin.training_observability import _nitro_placements, _restore_nitro_placements
from f51_darwin.training_observability import tracked_worktree_state


# ═══════════════ Task 1 tests: primitives ═══════════════

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
    # token_count=128, block_size=8 -> exactly 16 non-overlapping tiles.
    # batches=8 * batch_size=2 == 16 requests every tile exactly once.
    first = fixed_batch_starts(token_count=128, block_size=8, batch_size=2, batches=8, seed=999)
    second = fixed_batch_starts(token_count=128, block_size=8, batch_size=2, batches=8, seed=999)
    assert first == second
    assert len(first) == 8
    assert all(len(row) == 2 for row in first)
    assert all(0 <= start <= 120 for row in first for start in row)
    flat_starts = [start for row in first for start in row]
    assert len(flat_starts) == len(set(flat_starts)), "starts must be distinct (no overlap/repetition)"


def test_fixed_starts_cover_all_tiles_exactly_once_without_overlap() -> None:
    starts = fixed_batch_starts(token_count=65536, block_size=4096, batch_size=1, batches=16, seed=999)
    flat_starts = [start for row in starts for start in row]
    assert len(flat_starts) == 16
    covered_tiles = sorted(start // 4096 for start in flat_starts)
    assert covered_tiles == list(range(16))
    assert len(set(flat_starts)) == 16  # zero overlap/duplication


def test_fixed_starts_reject_more_distinct_blocks_than_fit() -> None:
    with pytest.raises(ValueError, match="distinct blocks"):
        fixed_batch_starts(token_count=65536, block_size=4096, batch_size=1, batches=17, seed=999)


def test_fixed_starts_are_deterministic_across_calls() -> None:
    first = fixed_batch_starts(token_count=65536, block_size=4096, batch_size=1, batches=16, seed=999)
    second = fixed_batch_starts(token_count=65536, block_size=4096, batch_size=1, batches=16, seed=999)
    assert first == second


def test_fixed_starts_truncate_non_exact_multiple_token_count() -> None:
    # token_count=65540 leaves a 4-token remainder after 16 full blocks of
    # 4096 -- the remainder must be silently dropped (whole-block sampling
    # only), never rounded up into a 17th partial/out-of-range tile.
    starts = fixed_batch_starts(token_count=65540, block_size=4096, batch_size=1, batches=16, seed=999)
    flat_starts = [start for row in starts for start in row]
    assert len(flat_starts) == 16
    assert all(start + 4096 <= 65540 for start in flat_starts)
    covered_tiles = sorted(start // 4096 for start in flat_starts)
    assert covered_tiles == list(range(16))
    with pytest.raises(ValueError, match="distinct blocks"):
        # A 17th block would need token_count >= 17*4096=69632; 65540 only
        # fits 16 whole blocks, so requesting 17 must still be rejected.
        fixed_batch_starts(token_count=65540, block_size=4096, batch_size=1, batches=17, seed=999)


def test_metric_channels_never_mix_fresh_and_replay_windows() -> None:
    channels = MetricChannels.create(window_size=2)
    channels.record_update(
        batch_kind="fresh", total_loss=4.0, lm_loss=3.0, tokens=8,
        optimizer_step=False,
    )
    channels.record_update(batch_kind="replay", total_loss=8.0, lm_loss=7.0, tokens=8)
    channels.record_update(batch_kind="fresh", total_loss=2.0, lm_loss=1.0, tokens=8)
    snap = channels.snapshot(elapsed_sec=2.0)
    assert snap["fresh"] == {"count": 2, "total_loss": 3.0, "lm_loss": 2.0}
    assert snap["replay"] == {"count": 1, "total_loss": 8.0, "lm_loss": 7.0}
    assert snap["replay_fraction"] == pytest.approx(1 / 3)
    assert snap["successful_updates"] == 3
    assert snap["optimizer_steps"] == 2


def test_jsonl_recovery_ignores_one_partial_trailing_record(tmp_path: Path) -> None:
    ledger = tmp_path / "metrics.jsonl"
    append_jsonl_fsync(ledger, {"step": 1, "heldout_lm_loss": 2.0})
    with ledger.open("ab") as handle:
        handle.write(b'{"step":2')
    records, truncated = recover_jsonl(ledger)
    assert records == [{"heldout_lm_loss": 2.0, "step": 1}]
    assert truncated is True


def test_exported_archive_requires_explicit_commit_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "f51_darwin.training_observability._git_exe",
        lambda: "git",
    )
    monkeypatch.setattr(
        "f51_darwin.training_observability.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(128, "git")
        ),
    )

    with pytest.raises(RuntimeError, match="no Git worktree"):
        tracked_worktree_state(tmp_path)

    commit = "a" * 40
    assert tracked_worktree_state(tmp_path, source_commit=commit) == (commit, "")


def test_exported_archive_rejects_malformed_commit_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="40- or 64-character"):
        tracked_worktree_state(tmp_path, source_commit="not-a-commit")


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


def _runtime_report_for_finalization(tmp_path: Path) -> dict[str, object]:
    baseline = tmp_path / "baseline.pt"
    candidate = tmp_path / "candidate.pt"
    baseline.touch()
    candidate.touch()
    return {
        "process_exit_code": 0,
        "base_checkpoint": str(baseline),
        "candidate_checkpoint": str(candidate),
        "candidate_checkpoint_sha256": "a" * 64,
        "checkpoint_verified": True,
        "monitor_errors": [],
        "metrics_truncated_tail": False,
        "awaiting_external_benchmark": True,
        "gate_input": {
            "expected_steps": 250,
            "successful_updates": 250,
            "skipped_updates": 0,
            "fresh": {"count": 200, "lm_loss": 2.0, "total_loss": 2.1},
            "replay": {"count": 50, "lm_loss": 2.2, "total_loss": 2.3},
            "replay_fraction": 0.20,
            "base_heldout_loss": 2.0,
            "candidate_heldout_loss": 2.01,
            "peak_temperatures_c": [74.0, 70.0],
            "new_nvidia_events": [],
            "live_kernel_event_141": False,
            "checkpoint_verified": True,
            "external_benchmark_passed": False,
        },
    }


def _external_comparison(runtime_report: dict[str, object]) -> dict[str, object]:
    return {
        "baseline": runtime_report["base_checkpoint"],
        "candidate": runtime_report["candidate_checkpoint"],
        "heldout_baseline": {"loss": 2.0},
        "heldout_candidate": {"loss": 2.02},
        "replay_baseline": {"loss": 2.2},
        "replay_candidate": {"loss": 2.22},
        "promotion": {"passed": False, "action": "quarantine"},
    }


def test_external_stability_does_not_depend_on_promotion_passed(tmp_path: Path) -> None:
    runtime = _runtime_report_for_finalization(tmp_path)
    comparison = _external_comparison(runtime)

    assert external_comparison_passed(comparison) is True
    final = finalize_canary_report(
        runtime,
        comparison,
        comparison_path=tmp_path / "external.json",
        comparison_sha256="b" * 64,
    )

    assert final["decision"] == {
        "status": "stable_canary",
        "passed": True,
        "failures": [],
    }
    assert final["external_comparison"]["promotion"]["passed"] is False
    assert final["awaiting_external_benchmark"] is False
    assert final["automatic_promotion"] is False


def test_external_finalizer_rejects_wrong_candidate(tmp_path: Path) -> None:
    runtime = _runtime_report_for_finalization(tmp_path)
    comparison = _external_comparison(runtime)
    comparison["candidate"] = str(tmp_path / "other.pt")

    with pytest.raises(ValueError, match="candidate"):
        finalize_canary_report(
            runtime,
            comparison,
            comparison_path=tmp_path / "external.json",
            comparison_sha256="b" * 64,
        )


def test_external_regression_blocks_final_stable_label(tmp_path: Path) -> None:
    runtime = _runtime_report_for_finalization(tmp_path)
    comparison = _external_comparison(runtime)
    comparison["replay_candidate"] = {"loss": 2.221}

    final = finalize_canary_report(
        runtime,
        comparison,
        comparison_path=tmp_path / "external.json",
        comparison_sha256="b" * 64,
    )

    assert final["decision"]["passed"] is False
    assert final["decision"]["status"] == "quality_gate_failed"
    assert "external_benchmark" in final["decision"]["failures"]


def test_finalize_cli_persists_final_report_atomically(tmp_path: Path) -> None:
    runtime = _runtime_report_for_finalization(tmp_path)
    comparison = _external_comparison(runtime)
    runtime_path = tmp_path / "runtime.json"
    comparison_path = tmp_path / "external.json"
    output_path = tmp_path / "final.json"
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "f51_darwin.training_observability",
            "--finalize-report",
            str(runtime_path),
            "--external-comparison",
            str(comparison_path),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["decision"]["status"] == "stable_canary"
    assert persisted["external_comparison"]["path"] == str(comparison_path.resolve())
    assert len(persisted["external_comparison"]["sha256"]) == 64
    assert not output_path.with_suffix(".json.tmp").exists()


# ═══════════════ Task 2 tests: evaluator ═══════════════

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


def test_nitro_restore_preserves_original_cold_expert_dtype() -> None:
    from f51_darwin.darwin_x import DarwinXConfig, DarwinXModel

    config = DarwinXConfig(
        vocab_size=64, context_length=8, inference_context_length=16,
        d_model=16, n_layers=4, n_heads=4, n_kv_heads=2,
        fine_experts=4, shared_experts=1, experts_per_token=2,
        fine_expert_hidden_dim=8, shared_expert_hidden_dim=8,
        heartbeat_enabled=False, ghost_enabled=False,
        nitro_gpu_expert_capacity=1,
    )
    model = DarwinXModel(config)
    moe = model.blocks[0].moe
    moe._init_expert_devices(torch.device("cpu"))
    moe.fine_experts[0].to(dtype=torch.bfloat16)
    moe._expert_gpu[0] = torch.device("cpu")
    moe.fine_experts[1].to(dtype=torch.float32)
    moe._expert_gpu[1] = None
    for parameter in moe.fine_experts[1].parameters():
        parameter.requires_grad = False
    saved = _nitro_placements(model)

    moe._restore_expert(1, torch.device("cpu"))
    assert next(moe.fine_experts[1].parameters()).dtype == torch.bfloat16
    _restore_nitro_placements(saved)

    assert next(moe.fine_experts[1].parameters()).dtype == torch.float32
    assert moe._expert_gpu[1] is None
    assert all(not parameter.requires_grad for parameter in moe.fine_experts[1].parameters())
