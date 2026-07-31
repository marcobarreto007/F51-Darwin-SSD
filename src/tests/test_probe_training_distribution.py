from __future__ import annotations

import hashlib
import sys

import numpy as np
import pytest
import torch

from research.probe_training_distribution import (
    ProbeRequest,
    build_parser,
    contextual_gate_exit_code,
    contextual_signal_metrics,
    contextual_signal_passes,
    infer_tokens_per_step,
    loader_seed,
    make_batch,
    main,
    parse_non_negative_float,
    parse_probe_request,
    tensor_digest,
)


def test_parse_probe_request_with_expected_digest():
    digest = "a" * 64

    request = parse_probe_request(f"23000:72:{digest}")

    assert request == ProbeRequest(step=23000, cycle=72, expected_digest=digest)


@pytest.mark.parametrize("raw", ["23000", "x:72", "23000:-1", "1:2:bad"])
def test_parse_probe_request_rejects_invalid_values(raw):
    with pytest.raises(Exception):
        parse_probe_request(raw)


def test_infer_tokens_per_step_requires_exact_checkpoint_arithmetic():
    assert (
        infer_tokens_per_step(
            {"step": 23_000, "train_tokens_seen": 94_208_000}
        )
        == 4_096
    )

    with pytest.raises(ValueError, match="cannot infer"):
        infer_tokens_per_step({"step": 3, "train_tokens_seen": 10})


def test_loader_seed_includes_cycle():
    assert loader_seed(51, 72) == 123


def test_make_batch_matches_causal_loader_formula(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(100, dtype=np.int32).tofile(path)
    tokens = np.memmap(path, dtype=np.int32, mode="r")

    batch, effective_seed = make_batch(
        tokens,
        step=2,
        cycle=3,
        block_size=8,
        batch_size=2,
        base_seed=51,
    )

    max_start = len(tokens) - 8 - 1
    starts = [
        (2 * 8 + row * 23 + (51 + 3)) % max_start
        for row in range(2)
    ]
    expected = torch.stack(
        [torch.arange(start, start + 8) for start in starts]
    )
    assert effective_seed == 54
    assert torch.equal(batch, expected)


def test_make_batch_uses_fixed_seed_for_permuted_sampler(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(256, dtype=np.int32).tofile(path)
    tokens = np.memmap(path, dtype=np.int32, mode="r")

    first, first_seed = make_batch(
        tokens,
        step=4,
        cycle=1,
        block_size=8,
        batch_size=1,
        base_seed=51,
        sampler_mode="permuted_blocks",
        sampler_version=1,
    )
    resumed, resumed_seed = make_batch(
        tokens,
        step=4,
        cycle=99,
        block_size=8,
        batch_size=1,
        base_seed=51,
        sampler_mode="permuted_blocks",
        sampler_version=1,
    )

    assert first_seed == resumed_seed == 51
    assert torch.equal(first, resumed)


def test_tensor_digest_is_sha256_of_numpy_batch_bytes():
    batch = torch.tensor([[1, 2, 3]], dtype=torch.int64)

    assert tensor_digest(batch) == hashlib.sha256(
        batch.numpy().tobytes()
    ).hexdigest()


def test_rank_step_option_can_be_repeated():
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--probe",
            "23000:72",
            "--rank-step",
            "23000",
            "--rank-step",
            "23100",
        ]
    )

    assert args.rank_step == [23000, 23100]


def test_contextual_metrics_make_both_penalties_explicit():
    metrics = contextual_signal_metrics(
        {
            "ce_full_sample": 4.407455,
            "ce_rank0": 4.405422,
            "ce_reversed": 4.408020,
        }
    )

    assert metrics["rank0_penalty"] == pytest.approx(-0.002033)
    assert metrics["alignment_break_penalty"] == pytest.approx(0.000565)
    assert metrics["contextual_gain"] == pytest.approx(-0.002033)


def test_contextual_gate_passes_only_when_both_controls_clear_minimum():
    passing = {
        "ce_full_sample": 4.0,
        "ce_rank0": 4.3,
        "ce_reversed": 4.2,
    }
    failing = {
        "ce_full_sample": 5.819942,
        "ce_rank0": 5.819510,
        "ce_reversed": 5.822101,
    }

    assert contextual_signal_passes(passing, minimum_gain=0.1)
    assert not contextual_signal_passes(failing, minimum_gain=0.0)


def test_contextual_gate_maps_pass_and_fail_to_process_exit_codes():
    assert contextual_gate_exit_code([]) == 0
    assert contextual_gate_exit_code([23_100]) == 1


def test_contextual_gate_cli_has_no_hidden_threshold():
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--probe",
            "23000:72",
        ]
    )

    assert args.min_contextual_gain is None
    assert args.json_output is None


def test_contextual_gate_cli_parses_explicit_threshold_and_json_path():
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--probe",
            "23000:72",
            "--rank-step",
            "23000",
            "--min-contextual-gain",
            "0.001",
            "--json-output",
            "probe.json",
        ]
    )

    assert args.min_contextual_gain == pytest.approx(0.001)
    assert args.json_output.name == "probe.json"


def test_contextual_gate_cli_requires_a_rank_step(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe_training_distribution.py",
            "--checkpoint",
            "checkpoint.pt",
            "--probe",
            "23000:72",
            "--min-contextual-gain",
            "0.0",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2


@pytest.mark.parametrize("raw", ["-0.1", "nan", "inf"])
def test_contextual_gate_cli_rejects_invalid_threshold(raw):
    with pytest.raises(Exception):
        parse_non_negative_float(raw)
