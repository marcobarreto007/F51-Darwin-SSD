from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from f51_darwin.heartbeat import SurpriseMemorySlot
from research.benchmark_native_ttm_recall import (
    DEFAULT_DOSES,
    association_sources,
    _set_effective_dose,
    _set_runtime_ttm_max_scale,
    classify_recall,
    clone_slots,
    memory_payload,
    named_tensors_sha256,
    parse_args,
    restore_memory_payload,
    select_baseline_misses,
    shuffle_slot_values,
    synthetic_fact_pool,
)


@dataclass(frozen=True)
class _FrozenConfig:
    ttm_residual_max_scale: float = 0.15


class _ModelWithFrozenConfig:
    config = _FrozenConfig()


class _BFloat16GateModel:
    def __init__(self) -> None:
        self.config = _FrozenConfig()
        self.ttm_residual_gate = torch.nn.Parameter(
            torch.zeros((), dtype=torch.bfloat16)
        )

    def ttm_residual_scale(self) -> torch.Tensor:
        return (
            self.config.ttm_residual_max_scale
            * torch.tanh(self.ttm_residual_gate)
        )


def _slot(index: int) -> SurpriseMemorySlot:
    return SurpriseMemorySlot(
        key=torch.tensor([float(index), 1.0]),
        value=torch.tensor([float(index), 2.0, 3.0]),
        timestamp=f"t{index}",
        domain=f"d{index}",
        surprise_score=0.5 + index,
        access_count=index,
    )


def test_association_sources_selects_prompt_tail_and_answer_span() -> None:
    prompt_hidden = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    teaching_hidden = torch.arange(48, dtype=torch.float32).reshape(1, 6, 8)

    key, value = association_sources(
        prompt_hidden,
        teaching_hidden,
        prompt_length=3,
        answer_length=2,
    )

    assert key.shape == (1, 8)
    assert value.shape == (1, 8)
    assert torch.equal(key, prompt_hidden[:, 2, :])
    assert torch.equal(value, teaching_hidden[:, 3:5, :].mean(dim=1))


@pytest.mark.parametrize(
    ("prompt_hidden", "teaching_hidden", "prompt_length", "answer_length"),
    [
        (torch.zeros(1, 3, 8), torch.zeros(1, 6, 8), 3, 0),
        (torch.zeros(1, 3, 8), torch.zeros(1, 6, 8), 0, 2),
        (torch.zeros(1, 3, 8), torch.zeros(1, 4, 8), 3, 2),
        (torch.zeros(2, 3, 8), torch.zeros(2, 6, 8), 3, 2),
        (torch.zeros(1, 3, 8), torch.zeros(1, 6, 7), 3, 2),
    ],
)
def test_association_sources_rejects_invalid_contract(
    prompt_hidden: torch.Tensor,
    teaching_hidden: torch.Tensor,
    prompt_length: int,
    answer_length: int,
) -> None:
    with pytest.raises(ValueError):
        association_sources(
            prompt_hidden,
            teaching_hidden,
            prompt_length=prompt_length,
            answer_length=answer_length,
        )


def test_association_sources_rejects_nonfinite_hidden() -> None:
    prompt_hidden = torch.zeros(1, 3, 8)
    teaching_hidden = torch.zeros(1, 6, 8)
    teaching_hidden[0, 3, 0] = torch.nan

    with pytest.raises(ValueError, match="finite"):
        association_sources(
            prompt_hidden,
            teaching_hidden,
            prompt_length=3,
            answer_length=2,
        )


def test_synthetic_fact_pool_is_deterministic_and_unique() -> None:
    left = synthetic_fact_pool(seed=51, count=20)
    right = synthetic_fact_pool(seed=51, count=20)

    assert left == right
    assert len({item["id"] for item in left}) == 20
    assert len({item["entity"] for item in left}) == 20
    assert len({item["answer"] for item in left}) == 20
    assert all(len(item["paraphrases"]) == 2 for item in left)
    assert all(len(item["answer"]) == 6 for item in left)


def test_select_baseline_misses_requires_five_real_misses() -> None:
    facts = synthetic_fact_pool(seed=51, count=8)
    rows = [
        {
            "id": item["id"],
            "response": item["answer"] if index < 3 else "ERRADO",
        }
        for index, item in enumerate(facts)
    ]

    selected = select_baseline_misses(facts, rows, count=5)

    assert tuple(item["id"] for item in selected) == tuple(
        item["id"] for item in facts[3:8]
    )


def test_select_baseline_misses_rejects_insufficient_pool() -> None:
    facts = synthetic_fact_pool(seed=51, count=6)
    rows = [
        {
            "id": item["id"],
            "response": "ERRADO" if index < 4 else item["answer"],
        }
        for index, item in enumerate(facts)
    ]

    with pytest.raises(ValueError, match="fewer than five baseline misses"):
        select_baseline_misses(facts, rows, count=5)


def test_shuffle_preserves_keys_and_rotates_values() -> None:
    source = [_slot(index) for index in range(5)]

    shuffled = shuffle_slot_values(source, order=(1, 2, 3, 4, 0))

    assert all(
        torch.equal(left.key, right.key)
        for left, right in zip(source, shuffled, strict=True)
    )
    assert all(
        torch.equal(shuffled[index].value, source[(index + 1) % 5].value)
        for index in range(5)
    )
    assert [slot.domain for slot in shuffled] == [slot.domain for slot in source]


def test_clone_slots_breaks_tensor_aliases() -> None:
    source = [_slot(0)]
    cloned = clone_slots(source)

    cloned[0].key.add_(10.0)
    cloned[0].value.mul_(0.0)

    assert not torch.equal(cloned[0].key, source[0].key)
    assert not torch.equal(cloned[0].value, source[0].value)


def test_memory_payload_round_trip_clones_slots() -> None:
    slots = [_slot(index) for index in range(2)]

    payload = memory_payload(slots)
    restored = restore_memory_payload(payload)

    assert len(restored) == 2
    assert torch.equal(restored[0].key, slots[0].key)
    assert torch.equal(restored[1].value, slots[1].value)
    assert restored[1].access_count == 1
    restored[0].key.add_(10.0)
    assert not torch.equal(restored[0].key, slots[0].key)


def test_memory_payload_rejects_shape_mismatch() -> None:
    payload = memory_payload([_slot(0)])
    payload["slots"][0]["key"] = torch.zeros(3)

    with pytest.raises(ValueError, match="inconsistent key shapes"):
        restore_memory_payload(payload)


def test_named_tensor_hash_detects_mutation() -> None:
    tensor = torch.arange(8, dtype=torch.float32)
    before = named_tensors_sha256((("brain", tensor),))

    tensor.add_(1.0)

    assert named_tensors_sha256((("brain", tensor),)) != before


def test_named_tensor_hash_is_name_sensitive() -> None:
    tensor = torch.arange(8, dtype=torch.float32)

    assert named_tensors_sha256((("a", tensor),)) != named_tensors_sha256(
        (("b", tensor),)
    )


def test_classifier_requires_behavioral_gain_and_causal_drop() -> None:
    report = {
        "baseline_misses": 5,
        "brain_hashes_identical": True,
        "protocol_valid": True,
        "doses": [
            {
                "dose": 0.1,
                "correct_memory": {
                    "literal": 5,
                    "paraphrase": 8,
                    "distractor": 4,
                    "retrieved_questions": 5,
                    "correct_retrieved_questions": 5,
                },
                "memory_disabled": {"literal": 1},
                "shuffled_memory": {
                    "literal": 2,
                    "keys_preserved": True,
                },
                "restored_memory": {"literal": 5},
            }
        ],
    }

    assert classify_recall(report) == "causal_memory_recall_pass"


@pytest.mark.parametrize(
    ("mutator", "expected"),
    [
        (
            lambda report: report.update(brain_hashes_identical=False),
            "invalid_protocol",
        ),
        (
            lambda report: report["doses"][0]["correct_memory"].update(
                literal=5, paraphrase=0, distractor=0
            ),
            "literal_only",
        ),
        (
            lambda report: report["doses"][0]["shuffled_memory"].update(literal=5),
            "noncausal_gain",
        ),
        (
            lambda report: report["doses"][0]["correct_memory"].update(
                literal=0, paraphrase=0, distractor=0
            ),
            "retrieval_only",
        ),
        (
            lambda report: report["doses"][0]["correct_memory"].update(
                literal=0,
                paraphrase=0,
                distractor=0,
                retrieved_questions=0,
                correct_retrieved_questions=0,
            ),
            "no_memory_effect",
        ),
    ],
)
def test_classifier_failure_labels(mutator, expected: str) -> None:
    report = {
        "baseline_misses": 5,
        "brain_hashes_identical": True,
        "protocol_valid": True,
        "doses": [
            {
                "dose": 0.1,
                "correct_memory": {
                    "literal": 5,
                    "paraphrase": 8,
                    "distractor": 4,
                    "retrieved_questions": 5,
                    "correct_retrieved_questions": 5,
                },
                "memory_disabled": {"literal": 1},
                "shuffled_memory": {
                    "literal": 2,
                    "keys_preserved": True,
                },
                "restored_memory": {"literal": 5},
            }
        ],
    }

    mutator(report)

    assert classify_recall(report) == expected


def test_native_diagnostic_supersedes_manual_retrieval_trace() -> None:
    report = {
        "baseline_misses": 5,
        "brain_hashes_identical": True,
        "protocol_valid": True,
        "native_diagnostic": {
            "overall_verdict": "native_retrieval_absent",
        },
        "doses": [
            {
                "correct_memory": {
                    "literal": 0,
                    "paraphrase": 2,
                    "distractor": 0,
                    "retrieved_questions": 5,
                    "correct_retrieved_questions": 5,
                },
                "memory_disabled": {"literal": 0},
                "shuffled_memory": {
                    "literal": 0,
                    "keys_preserved": True,
                },
                "restored_memory": {"literal": 0},
            }
        ],
    }

    assert classify_recall(report) == "no_memory_effect"


def test_cli_defaults_are_protocol_constants() -> None:
    args = parse_args([])

    assert args.seed == 51
    assert args.fact_count == 5
    assert tuple(args.doses) == DEFAULT_DOSES
    assert isinstance(args.output_dir, Path)
    assert args.output_dir.name == "memory-recall"
    assert args.verify_report is False


def test_cli_rejects_non_five_fact_count() -> None:
    with pytest.raises(ValueError, match="fact-count must be exactly 5"):
        parse_args(["--fact-count", "4"])


def test_cli_rejects_duplicate_or_negative_doses() -> None:
    with pytest.raises(ValueError, match="doses must be unique"):
        parse_args(["--doses", "0", "0"])
    with pytest.raises(ValueError, match="doses must be non-negative"):
        parse_args(["--doses", "-0.1", "0.1"])


def test_runtime_dose_override_supports_frozen_config() -> None:
    model = _ModelWithFrozenConfig()

    _set_runtime_ttm_max_scale(model, 0.3)

    assert model.config.ttm_residual_max_scale == pytest.approx(0.3)


def test_effective_dose_accepts_bfloat16_quantization() -> None:
    model = _BFloat16GateModel()

    effective = _set_effective_dose(model, 0.01)

    assert effective == pytest.approx(0.01, rel=0.01)
