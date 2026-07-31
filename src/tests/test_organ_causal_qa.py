from __future__ import annotations

from collections import Counter

import pytest
import torch

from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.transplant_16b.organ_causal_qa import (
    ORGAN_ARMS,
    build_answer_only_example,
    classify_arm,
    language_brain_named_parameters,
    select_organ_parameters,
    shuffled_orders,
)
from research.benchmark_smol_dense_qa import MCQ, OPEN_QA


def _model() -> DarwinXModel:
    config = DarwinXConfig(
        model_name="organ-causal-test",
        vocab_size=32,
        context_length=16,
        inference_context_length=16,
        d_model=16,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        feed_forward_kind="dense_swiglu",
        fine_experts=1,
        shared_experts=0,
        experts_per_token=1,
        fine_expert_hidden_dim=32,
        shared_expert_hidden_dim=32,
        mtp_depth=2,
        mtp_weight=0.0,
        jepa_weight=0.0,
        ghost_weight=0.0,
        ghost_enabled=False,
        curiosity_weight=0.0,
        spider_sense_enabled=True,
        heartbeat_enabled=True,
        ttm_residual_enabled=True,
        ttm_residual_max_scale=0.0,
        ttm_associative_weight=0.0,
        spider_calibration_enabled=True,
        spider_calibration_weight=0.0,
        loss_semantics_version=2,
        gaba_enabled=True,
        inter_hemispheric_enabled=True,
        inter_hemispheric_residual_enabled=True,
        nitro_enabled=False,
        dae_enabled=False,
        vertical_routing_scale=0.0,
        attention_indices_override=(0, 1),
    )
    return DarwinXModel(config)


def test_each_arm_selects_only_its_own_parameter_family() -> None:
    model = _model()
    assert ORGAN_ARMS == (
        "control",
        "gaba",
        "ihs",
        "heartbeat",
        "ttm",
        "spider",
        "jepa",
        "mtp",
    )
    expected_markers = {
        "gaba": ("gaba",),
        "ihs": ("inter_hemispheric",),
        "spider": ("_spider_sense_module",),
        "jepa": ("jepa_predictor",),
        "mtp": ("mtp_heads",),
    }
    assert select_organ_parameters(model, "control").named == ()
    for arm, markers in expected_markers.items():
        selection = select_organ_parameters(model, arm)
        assert selection.named
        assert all(any(marker in name for marker in markers) for name, _ in selection.named)
    heartbeat = select_organ_parameters(model, "heartbeat")
    assert heartbeat.named
    assert all(name.startswith("heartbeat.") for name, _ in heartbeat.named)
    ttm = select_organ_parameters(model, "ttm")
    assert any(name == "ttm_residual_gate" for name, _ in ttm.named)
    assert any(name.startswith("heartbeat.tt_memory.") for name, _ in ttm.named)
    with pytest.raises(ValueError, match="unknown organ arm"):
        select_organ_parameters(model, "brain")


def test_organ_selections_never_include_language_brain_parameters() -> None:
    model = _model()
    brain_markers = (
        "token_embedding",
        ".attention.",
        ".ffn.",
        ".norm1.",
        ".norm2.",
        "lm_head",
    )
    for arm in ORGAN_ARMS:
        selection = select_organ_parameters(model, arm)
        assert len({id(parameter) for _, parameter in selection.named}) == len(
            selection.named
        )
        assert not any(
            any(marker in name for marker in brain_markers)
            for name, _ in selection.named
        )
    brain = language_brain_named_parameters(model)
    assert brain
    assert any(name == "token_embedding.weight" for name, _ in brain)
    assert any(".attention." in name for name, _ in brain)
    assert any(".ffn." in name for name, _ in brain)
    organ_ids = {
        id(parameter)
        for arm in ORGAN_ARMS
        for _, parameter in select_organ_parameters(model, arm).named
    }
    assert all(id(parameter) not in organ_ids for _, parameter in brain)


def test_answer_only_example_masks_prompt_and_keeps_answer_and_eos() -> None:
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    inputs, labels = build_answer_only_example(
        prompt,
        answer_ids=(7, 8),
        eos_token_id=9,
    )
    assert inputs.tolist() == [[1, 2, 3, 7, 8, 9]]
    assert labels.tolist() == [[-100, -100, -100, 7, 8, 9]]
    with pytest.raises(ValueError, match="one non-empty sequence"):
        build_answer_only_example(
            torch.empty((1, 0), dtype=torch.long),
            answer_ids=(7,),
            eos_token_id=9,
        )


def test_shuffled_orders_are_complete_deterministic_permutations() -> None:
    item_ids = tuple(row["id"] for row in (*MCQ, *OPEN_QA))
    first = shuffled_orders(item_ids, seeds=range(51, 61))
    second = shuffled_orders(item_ids, seeds=range(51, 61))
    assert first == second
    assert len(first) == 10
    assert all(Counter(order) == Counter(item_ids) for order in first)
    assert len(set(first)) == 10


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        ({"frozen_brain_drift": 1}, "invalid_brain_drift"),
        (
            {
                "frozen_brain_drift": 0,
                "nonzero_gradient_parameters": 0,
                "state_changed": False,
            },
            "no_language_gradient",
        ),
        (
            {
                "frozen_brain_drift": 0,
                "nonzero_gradient_parameters": 1,
                "state_changed": False,
                "corrected": 2,
                "forgotten": 0,
            },
            "improved",
        ),
        (
            {
                "frozen_brain_drift": 0,
                "nonzero_gradient_parameters": 1,
                "state_changed": False,
                "corrected": 1,
                "forgotten": 1,
            },
            "changed_but_regressed",
        ),
        (
            {
                "frozen_brain_drift": 0,
                "nonzero_gradient_parameters": 0,
                "state_changed": True,
                "corrected": 0,
                "forgotten": 0,
            },
            "state_only",
        ),
        (
            {
                "frozen_brain_drift": 0,
                "nonzero_gradient_parameters": 1,
                "state_changed": False,
                "corrected": 0,
                "forgotten": 0,
            },
            "no_effect",
        ),
    ],
)
def test_classification_is_fail_closed(report: dict, expected: str) -> None:
    assert classify_arm(report) == expected
