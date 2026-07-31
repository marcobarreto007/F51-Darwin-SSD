from __future__ import annotations

import pytest

from research.probe_three_organ_shadow import (
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
