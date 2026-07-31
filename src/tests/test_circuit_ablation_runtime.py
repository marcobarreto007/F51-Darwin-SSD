from __future__ import annotations

import hashlib
import math
from dataclasses import replace

import pytest
import torch
from torch import nn

from f51_darwin.circuits import TapContract
from f51_darwin.circuits.ablation import (
    AblationArm,
    AblationResult,
    ArmMetrics,
    CausalGateConfig,
    CircuitSelector,
    build_causal_verdict,
    discover_residual_channels,
    run_paired_ablation,
)


class KnownCircuitClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 4, bias=False)])
        self.head = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.blocks[0].weight.copy_(torch.eye(4))
            self.head.weight.copy_(
                torch.tensor([[3.0, -3.0, 0.2, -0.2], [-3.0, 3.0, -0.2, 0.2]])
            )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks[0](values))


def _confirmation_batch() -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    labels = []
    for index in range(20):
        target = index % 2
        core = [3.0, 0.0] if target == 0 else [0.0, 3.0]
        nuisance = [1.0, -1.0]
        rows.append(core + nuisance)
        labels.append(target)
    return torch.tensor(rows), torch.tensor(labels)


def _discover_known_selector(*, rank: int = 2) -> CircuitSelector:
    values, labels = _confirmation_batch()
    values = values.clone()
    values[:, :2] *= 2.0
    if rank == 3:
        values = values[:6].reshape(3, 2, 4)
        labels = labels[:6].reshape(3, 2)
    return discover_residual_channels(
        KnownCircuitClassifier(),
        [(values, labels)],
        TapContract(
            provider="module_path_v1",
            module_path="blocks.0",
            position="post",
            output_selector="tensor",
            rank=rank,
            width=4,
            minimum_sequence_length=1,
        ),
        count=2,
    )


def test_known_circuit_paired_ablation_has_independent_identical_anchors() -> None:
    selector = _discover_known_selector()
    result = run_paired_ablation(
        KnownCircuitClassifier,
        [_confirmation_batch()],
        selector,
        seed=51,
    )

    assert result.clean.accuracy == 1.0
    assert result.ablate.accuracy <= 0.5
    assert result.restore.accuracy == result.clean.accuracy
    assert result.restore.max_abs_logit_error <= 1e-6
    assert result.random_matched.accuracy > result.ablate.accuracy
    assert result.anchors_identical is True
    assert len(set(result.model_instance_ids.values())) == len(AblationArm)
    permutation = result.shuffle_permutations[0]
    assert all(index != target for index, target in enumerate(permutation))
    repeated = run_paired_ablation(
        KnownCircuitClassifier,
        [_confirmation_batch()],
        selector,
        seed=51,
    )
    assert repeated.shuffle_permutations == result.shuffle_permutations
    assert result.evidence_sha256 == result.to_dict()["evidence_sha256"]


def test_paired_ablation_supports_rank_three_residual_activations() -> None:
    values, labels = _confirmation_batch()
    result = run_paired_ablation(
        KnownCircuitClassifier,
        [(values[:6].reshape(3, 2, 4), labels[:6].reshape(3, 2))],
        _discover_known_selector(rank=3),
        seed=7,
    )

    assert len(result.clean.per_item_nll) == 3
    assert result.anchors_identical is True


def test_paired_ablation_rejects_selector_without_discovery_items() -> None:
    with pytest.raises(ValueError, match="discovery"):
        run_paired_ablation(
            KnownCircuitClassifier,
            [_confirmation_batch()],
            CircuitSelector("blocks.0", (0, 1)),
            seed=51,
        )


class RandomNonPersistentClassifier(KnownCircuitClassifier):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("random_offset", torch.rand(4), persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.blocks[0](values) + self.random_offset
        return self.head(hidden)


def test_paired_arms_restore_rng_initialized_nonpersistent_buffer() -> None:
    values, labels = _confirmation_batch()
    selector = discover_residual_channels(
        RandomNonPersistentClassifier(),
        [(values + 11.0, labels)],
        TapContract(
            provider="module_path_v1",
            module_path="blocks.0",
            position="post",
            output_selector="tensor",
            rank=2,
            width=4,
            minimum_sequence_length=1,
        ),
        count=2,
    )
    created_models: list[RandomNonPersistentClassifier] = []

    def factory() -> RandomNonPersistentClassifier:
        model = RandomNonPersistentClassifier()
        created_models.append(model)
        return model

    result = run_paired_ablation(
        factory,
        [(values, labels)],
        selector,
        seed=83,
    )

    assert len(created_models) == len(AblationArm) + 1
    assert all(
        torch.equal(model.random_offset, created_models[0].random_offset)
        for model in created_models[1:]
    )
    assert len({anchor["model_digest"] for anchor in result.anchors.values()}) == 1
    assert result.restore.max_abs_logit_error <= 1e-6
    assert result.anchors_identical is True


def _metrics(nll: list[float], *, accuracy: float = 1.0) -> ArmMetrics:
    return ArmMetrics(
        accuracy=accuracy,
        mean_nll=sum(nll) / len(nll),
        per_item_nll=tuple(nll),
        logits_sha256="a" * 64,
    )


def _synthetic_result(seed: int, *, matched: bool = False) -> AblationResult:
    clean = [0.25 + 0.001 * math.sin(index + seed) for index in range(200)]
    target_effect = [0.25 + 0.01 * math.sin(index * 0.3 + seed) for index in range(200)]
    control_effect = (
        target_effect
        if matched
        else [0.05 + 0.003 * math.cos(index + seed) for index in range(200)]
    )
    restore_effect = [0.005 for _ in clean]
    shuffle_effect = [0.20 for _ in clean]
    anchor = {
        "input_digest": "1" * 64,
        "model_digest": "2" * 64,
        "rng_digest": "3" * 64,
        "batch_order_digest": "4" * 64,
    }
    item_digests = tuple(
        hashlib.sha256(f"confirmation-{index}".encode()).hexdigest()
        for index in range(200)
    )
    return AblationResult(
        seed=seed,
        selector=CircuitSelector(
            "blocks.0",
            (0, 1),
            discovery_item_sha256=("b" * 64,),
        ),
        tap_width=4,
        clean=_metrics(clean),
        ablate=_metrics([a + b for a, b in zip(clean, target_effect)], accuracy=0.4),
        restore=_metrics([a + b for a, b in zip(clean, restore_effect)]),
        shuffle=_metrics([a + b for a, b in zip(clean, shuffle_effect)], accuracy=0.5),
        random_matched=_metrics([a + b for a, b in zip(clean, control_effect)], accuracy=0.9),
        domains=("known",) * 200,
        confirmation_item_sha256=item_digests,
        confirmation_batch_sizes=(200,),
        anchors={arm.value: dict(anchor) for arm in AblationArm},
        anchors_identical=True,
        shuffle_permutations=(tuple(range(1, 200)) + (0,),),
        random_matched_channels=(2, 3),
        model_instance_ids={arm.value: seed * 10 + index for index, arm in enumerate(AblationArm)},
        checks={
            "anchors_complete_and_identical": True,
            "independent_model_instances": True,
            "shuffle_derangements_valid": True,
            "matched_control_valid": True,
            "discovery_confirmation_disjoint": True,
            "selector_within_tap_width": True,
            "paired_item_count": 200,
        },
    )


def _gate_confirmation_batch() -> tuple[torch.Tensor, torch.Tensor, str]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(911)
    labels = torch.cat(
        (torch.zeros(100, dtype=torch.long), torch.ones(100, dtype=torch.long))
    )
    labels = labels[torch.randperm(200, generator=generator)]
    rows = []
    for target in labels.tolist():
        core = [3.0, 0.0] if target == 0 else [0.0, 3.0]
        nuisance = [1.0, -1.0] if target == 0 else [-1.0, 1.0]
        rows.append(core + nuisance)
    return torch.tensor(rows), labels, "known"


def test_statistical_gate_accepts_known_circuit() -> None:
    selector = _discover_known_selector()
    runs = [
        run_paired_ablation(
            KnownCircuitClassifier,
            [_gate_confirmation_batch()],
            selector,
            seed=seed,
        )
        for seed in range(5)
    ]
    verdict = build_causal_verdict(
        runs,
        CausalGateConfig(),
    )

    assert verdict.causal is True
    assert verdict.reason == "causal"
    assert verdict.target_effect_ci_low > 0.0
    assert verdict.target_effect_mean >= 0.15
    assert verdict.restore_fraction >= 0.95
    assert verdict.target_to_control_ratio >= 2.0
    assert verdict.holm_rejected is True


def test_statistical_gate_rejects_correlated_only_matched_control() -> None:
    verdict = build_causal_verdict(
        [_synthetic_result(seed, matched=True) for seed in range(5)],
        CausalGateConfig(),
    )

    assert verdict.causal is False
    assert verdict.reason == "matched_control_not_beaten"


def test_statistical_gate_fails_closed_on_insufficient_evidence() -> None:
    with pytest.raises(ValueError, match="minimum_seeds"):
        build_causal_verdict([_synthetic_result(seed) for seed in range(4)], CausalGateConfig())


def _resign(result: AblationResult, **changes: object) -> AblationResult:
    return replace(result, evidence_sha256="", **changes)


@pytest.mark.parametrize(
    ("tamper", "match"),
    [
        (
            lambda run: _resign(
                run,
                anchors={
                    **dict(run.anchors),
                    AblationArm.CLEAN.value: {"input_digest": "1" * 64},
                },
            ),
            "anchor",
        ),
        (
            lambda run: _resign(
                run,
                model_instance_ids={arm.value: 1 for arm in AblationArm},
            ),
            "model instance",
        ),
        (
            lambda run: _resign(
                run,
                shuffle_permutations=(tuple(range(200)),),
            ),
            "shuffle",
        ),
        (
            lambda run: _resign(run, random_matched_channels=(0, 3)),
            "matched control",
        ),
        (
            lambda run: _resign(run, checks={"paired_item_count": 200}),
            "checks",
        ),
        (
            lambda run: _resign(
                run,
                confirmation_item_sha256=("not-a-digest",)
                + run.confirmation_item_sha256[1:],
            ),
            "confirmation item",
        ),
    ],
)
def test_statistical_gate_recomputes_provenance_before_statistics(tamper, match: str) -> None:
    runs = [_synthetic_result(seed) for seed in range(5)]
    runs[0] = tamper(runs[0])

    with pytest.raises(ValueError, match=match):
        build_causal_verdict(runs, CausalGateConfig())


def test_statistical_gate_rejects_selector_and_evidence_hash_tampering() -> None:
    runs = [_synthetic_result(seed) for seed in range(5)]
    runs[0] = _resign(
        runs[0],
        selector=replace(runs[0].selector, channels=(0, 2)),
        random_matched_channels=(1, 3),
    )
    with pytest.raises(ValueError, match="selector"):
        build_causal_verdict(runs, CausalGateConfig())

    runs = [_synthetic_result(seed) for seed in range(5)]
    runs[0] = replace(runs[0], anchors={})
    with pytest.raises(ValueError, match="evidence_sha256"):
        build_causal_verdict(runs, CausalGateConfig())


def test_statistical_gate_rejects_resigned_discovery_confirmation_overlap() -> None:
    runs = [_synthetic_result(seed) for seed in range(5)]
    overlapping_selector = replace(
        runs[0].selector,
        discovery_item_sha256=(runs[0].confirmation_item_sha256[0],),
    )
    runs = [_resign(run, selector=overlapping_selector) for run in runs]

    with pytest.raises(ValueError, match="discovery and confirmation"):
        build_causal_verdict(runs, CausalGateConfig())


def test_statistical_gate_rejects_empty_discovery_and_out_of_range_channels() -> None:
    runs = [_synthetic_result(seed) for seed in range(5)]
    empty_selector = replace(runs[0].selector, discovery_item_sha256=())
    runs = [_resign(run, selector=empty_selector) for run in runs]
    with pytest.raises(ValueError, match="discovery"):
        build_causal_verdict(runs, CausalGateConfig())

    runs = [_synthetic_result(seed) for seed in range(5)]
    invalid_selector = replace(runs[0].selector, channels=(0, 4))
    runs = [
        _resign(
            run,
            selector=invalid_selector,
            random_matched_channels=(1, 2),
        )
        for run in runs
    ]
    with pytest.raises(ValueError, match="tap width"):
        build_causal_verdict(runs, CausalGateConfig())


@pytest.mark.parametrize(
    "changes",
    [
        {"minimum_seeds": True},
        {"minimum_items_per_domain": 5.0},
        {"minimum_items_per_domain": 1_000_001},
        {"bootstrap_draws": 1.0},
        {"signflip_draws": 0},
        {"seed": -1},
        {"minimum_effect_nats": float("nan")},
        {"minimum_control_ratio": float("inf")},
        {"minimum_restore_fraction": 1.1},
        {"minimum_shuffle_fraction": -0.1},
        {"alpha": 0.0},
    ],
)
def test_causal_gate_config_rejects_unsafe_exact_types_and_ranges(changes: dict) -> None:
    with pytest.raises(ValueError):
        CausalGateConfig(**changes)


def test_verdict_revalidates_a_config_instance_before_statistics() -> None:
    config = CausalGateConfig()
    object.__setattr__(config, "alpha", float("nan"))

    with pytest.raises(ValueError, match="alpha"):
        build_causal_verdict(
            [_synthetic_result(seed) for seed in range(5)],
            config,
        )
