from __future__ import annotations

import pytest

from f51_darwin.organism.support import (
    build_training_data_contract,
    validate_training_data_contract,
)


def _contract(*, mode: str = "permuted_blocks", holdout: bool = True):
    return build_training_data_contract(
        sampler_mode=mode,
        sampler_version=1,
        base_seed=51,
        block_size=4096,
        batch_size=1,
        source_identity={
            "path": "tokens.bin",
            "token_count": 100_000,
            "size_bytes": 400_000,
            "mtime_ns": 123,
        },
        train_token_count=90_000,
        holdout_definition=(
            {
                "source_token_count": 100_000,
                "train_stop": 90_000,
                "holdout_start": 90_000,
                "holdout_token_count": 10_000,
                "seed": 999,
                "batches": 4,
            }
            if holdout
            else None
        ),
    )


def test_training_data_contract_is_stable_and_excludes_mtime():
    contract = _contract()

    assert contract["schema"] == "darwin-training-data-v1"
    assert contract["sampler"]["seed_policy"] == "fixed"
    assert "mtime_ns" not in contract["source"]


def test_training_data_contract_resume_requires_exact_match():
    contract = _contract()

    validate_training_data_contract(contract, contract)
    changed = _contract()
    changed["block_size"] = 2048
    with pytest.raises(ValueError, match="block_size"):
        validate_training_data_contract(contract, changed)


def test_fresh_start_accepts_new_shuffle_and_holdout_contract():
    validate_training_data_contract(None, _contract(), is_resume=False)

    with pytest.raises(ValueError, match="fresh-start"):
        validate_training_data_contract(
            _contract(),
            _contract(),
            is_resume=False,
        )


def test_legacy_checkpoint_cannot_adopt_shuffle_or_holdout():
    with pytest.raises(ValueError, match="fresh checkpoint root"):
        validate_training_data_contract(None, _contract())

    validate_training_data_contract(
        None,
        _contract(mode="sequential", holdout=False),
    )
