from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research"))

from full_holdout_eval import _resolve_holdout_geometry  # noqa: E402


def _args(**overrides):
    base = dict(
        token_bin=None,
        holdout_tokens=None,
        block_size=None,
        batch_size=None,
        holdout_seed=999,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _payload_with_contract(*, holdout_token_count=65536, block_size=4096, batch_size=1, seed=999):
    return {
        "training_data_contract": {
            "schema": "darwin-training-data-v1",
            "block_size": block_size,
            "batch_size": batch_size,
            "source": {"path": "/tokens/feast_v2.bin"},
            "holdout": {
                "source_token_count": 18_548_972_689,
                "train_stop": 18_548_907_153,
                "holdout_start": 18_548_907_153,
                "holdout_token_count": holdout_token_count,
                "seed": seed,
                "batches": 4,  # the trainer's PARTIAL batches — must be ignored by the tool
            },
        }
    }


def test_full_organism_v1_geometry_yields_16_full_coverage_batches():
    # Exact numbers recorded for the live FULL_ORGANISM_V1 launch:
    # 65536 holdout tokens, block_size=4096, batch_size=1 -> 16 non-overlapping
    # tiles, all 16 must be scored for 100% coverage (see full-organism-v1
    # metadata.json holdout block and CLAUDE.md linked commits b33615c/2a4cb88).
    geometry = _resolve_holdout_geometry(_payload_with_contract(), _args())

    assert geometry["num_tiles"] == 16
    assert geometry["full_batches"] == 16
    assert geometry["block_size"] == 4096
    assert geometry["batch_size"] == 1
    assert geometry["holdout_token_count"] == 65536
    assert geometry["seed"] == 999
    assert geometry["source_path"] == "/tokens/feast_v2.bin"


def test_geometry_ignores_trainer_partial_batches_field():
    # The checkpoint's contract carries batches=4 (the PARTIAL coverage the
    # live trainer scores). This tool must derive full coverage from the
    # token/block geometry, not from that field.
    geometry = _resolve_holdout_geometry(_payload_with_contract(), _args())
    assert geometry["full_batches"] != 4
    assert geometry["full_batches"] == 16


def test_token_bin_override_wins_over_contract_source():
    geometry = _resolve_holdout_geometry(
        _payload_with_contract(), _args(token_bin=Path("/override/tokens.bin"))
    )
    assert geometry["source_path"] == "/override/tokens.bin" or geometry["source_path"] == str(
        Path("/override/tokens.bin")
    )


def test_non_divisible_holdout_tokens_rejected():
    payload = _payload_with_contract(holdout_token_count=65530)  # not a multiple of 4096
    with pytest.raises(ValueError, match="not an exact multiple of block_size"):
        _resolve_holdout_geometry(payload, _args())


def test_non_divisible_batch_size_rejected():
    payload = _payload_with_contract(batch_size=3)  # 16 tiles not divisible by 3
    with pytest.raises(ValueError, match="not an exact multiple of batch_size"):
        _resolve_holdout_geometry(payload, _args())


def test_missing_contract_requires_explicit_cli_fallback():
    with pytest.raises(ValueError, match="training_data_contract"):
        _resolve_holdout_geometry({}, _args())


def test_missing_contract_accepts_explicit_cli_fallback():
    geometry = _resolve_holdout_geometry(
        {},
        _args(holdout_tokens=65536, block_size=4096, batch_size=1, token_bin=Path("/manual/tokens.bin")),
    )
    assert geometry["full_batches"] == 16
    assert geometry["source_path"] == str(Path("/manual/tokens.bin"))
