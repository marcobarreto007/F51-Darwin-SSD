from __future__ import annotations

from pathlib import Path

from research.diagnose_native_ttm_recall import (
    channel_verdict,
    ceiling_user_text,
    corrected_recall_payload,
    overall_verdict,
    scientific_classification,
)


def test_ceiling_prompt_declares_answer_before_same_question() -> None:
    fact = {
        "entity": "artefato-00-123456",
        "answer": "654321",
        "question": "Qual é o código secreto do artefato-00-123456?",
        "options": ("111111", "654321", "222222", "333333"),
    }

    text = ceiling_user_text(fact)

    assert "artefato-00-123456 é 654321" in text
    assert fact["question"] in text
    assert all(option in text for option in fact["options"])


def test_channel_verdict_requires_real_native_retrieval() -> None:
    rows = [
        {
            "native_memory_retrieved": False,
            "last_logits_max_abs_diff": 0.0,
        }
    ]

    assert channel_verdict(rows) == "native_retrieval_absent"


def test_channel_verdict_detects_injection_that_misses_decoder() -> None:
    rows = [
        {
            "native_memory_retrieved": True,
            "last_logits_max_abs_diff": 0.0,
        }
    ]

    assert channel_verdict(rows) == "native_injection_misses_decoder"


def test_channel_verdict_detects_last_logit_change() -> None:
    rows = [
        {
            "native_memory_retrieved": True,
            "last_logits_max_abs_diff": 0.25,
        }
    ]

    assert channel_verdict(rows) == "native_channel_reaches_decoder"


def test_decoder_floor_has_priority_in_overall_verdict() -> None:
    assert (
        overall_verdict(
            ceiling_correct=0,
            ceiling_total=5,
            channel="native_injection_misses_decoder",
        )
        == "inconclusive_decoder_floor"
    )
    assert (
        overall_verdict(
            ceiling_correct=4,
            ceiling_total=5,
            channel="native_injection_misses_decoder",
        )
        == "native_injection_misses_decoder"
    )


def test_corrected_payload_preserves_initial_classification() -> None:
    recall = {"classification": "retrieval_only"}
    diagnostic = {
        "schema": "darwin-native-ttm-diagnostic-v1",
        "decoder_floor": {"correct": 5, "total": 5},
        "channel_verdict": "native_retrieval_absent",
        "overall_verdict": "native_retrieval_absent",
        "brain_unchanged": True,
    }

    updated = corrected_recall_payload(
        recall,
        diagnostic,
        diagnostic_path=Path("diagnostic.json"),
        diagnostic_sha256="abc123",
    )

    assert updated["classification_before_native_diagnostic"] == "retrieval_only"
    assert updated["classification"] == "no_memory_effect"
    assert updated["failure_mode"] == "native_retrieval_absent"
    assert updated["strict_gate_passed"] is False
    assert updated["native_diagnostic"]["sha256"] == "abc123"


def test_corrected_payload_separates_working_channel_from_failed_recall() -> None:
    recall = {"classification": "retrieval_only"}
    diagnostic = {
        "schema": "darwin-native-ttm-diagnostic-v1",
        "decoder_floor": {"correct": 5, "total": 5},
        "channel_verdict": "native_channel_reaches_decoder",
        "overall_verdict": "native_channel_reaches_decoder",
        "brain_unchanged": True,
    }

    updated = corrected_recall_payload(
        recall,
        diagnostic,
        diagnostic_path=Path("diagnostic.json"),
        diagnostic_sha256="abc123",
    )

    assert updated["classification"] == "no_memory_effect"
    assert updated["failure_mode"] == "value_not_behaviorally_decodable"
    assert updated["strict_gate_passed"] is False


def test_corrected_payload_keeps_real_causal_pass() -> None:
    recall = {"classification": "causal_memory_recall_pass"}
    diagnostic = {
        "schema": "darwin-native-ttm-diagnostic-v1",
        "decoder_floor": {"correct": 5, "total": 5},
        "channel_verdict": "native_channel_reaches_decoder",
        "overall_verdict": "native_channel_reaches_decoder",
        "brain_unchanged": True,
    }

    updated = corrected_recall_payload(
        recall,
        diagnostic,
        diagnostic_path=Path("diagnostic.json"),
        diagnostic_sha256="abc123",
    )

    assert updated["classification"] == "causal_memory_recall_pass"
    assert updated["failure_mode"] == "native_channel_reaches_decoder"
    assert updated["strict_gate_passed"] is True


def test_scientific_classification_uses_frozen_taxonomy() -> None:
    assert scientific_classification("native_retrieval_absent") == (
        "no_memory_effect"
    )
    assert scientific_classification("native_injection_misses_decoder") == (
        "no_memory_effect"
    )
    assert scientific_classification("inconclusive_decoder_floor") == (
        "invalid_protocol"
    )
