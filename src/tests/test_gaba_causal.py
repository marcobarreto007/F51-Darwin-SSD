from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch
from torch import nn

from f51_darwin.darwin_x import DarwinXConfig, DarwinXModel
from f51_darwin.darwin_x_core.block import DarwinXBlock
from f51_darwin.gaba_inhibition import GABAConfig, GABAergicLayer


def _small_config(**overrides: object) -> DarwinXConfig:
    values: dict[str, object] = {
        "vocab_size": 32,
        "context_length": 8,
        "inference_context_length": 16,
        "d_model": 8,
        "n_layers": 2,
        "n_heads": 2,
        "n_kv_heads": 1,
        "fine_experts": 2,
        "shared_experts": 1,
        "experts_per_token": 1,
        "fine_expert_hidden_dim": 8,
        "shared_expert_hidden_dim": 8,
        "mtp_depth": 1,
        "heartbeat_enabled": False,
        "spider_sense_enabled": False,
        "ghost_enabled": False,
        "gaba_enabled": True,
    }
    values.update(overrides)
    return DarwinXConfig(**values)


def test_gaba_cuda_predicate_dispatch_avoids_python_bool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BoolTrap:
        device = torch.device("cuda")

        def __bool__(self) -> bool:
            pytest.fail("CUDA predicate invoked Python bool")

    calls: list[tuple[object, str]] = []
    monkeypatch.setattr(
        torch,
        "_assert_async",
        lambda predicate, message: calls.append((predicate, message)),
    )
    predicate = BoolTrap()

    GABAergicLayer._assert_finite_predicate(
        predicate,
        name="CUDA probe",
    )

    assert calls == [(predicate, "CUDA probe must be finite")]


def test_gaba_cpu_predicate_rejects_synchronously() -> None:
    with pytest.raises(ValueError, match="CPU probe must be finite"):
        GABAergicLayer._assert_finite_predicate(
            torch.tensor(False),
            name="CPU probe",
        )


def test_gaba_gate_zero_is_bitwise_identity() -> None:
    torch.manual_seed(51)
    layer = GABAergicLayer(GABAConfig(d_model=8, max_delta_ratio=0.25))
    x = torch.randn(2, 3, 8)
    before = copy.deepcopy(layer.state_dict())

    delta, observation = layer(x, mutate_state=False)

    assert torch.equal(delta, torch.zeros_like(delta))
    assert layer.state_dict().keys() == before.keys()
    for key in before:
        assert torch.equal(layer.state_dict()[key], before[key])
    assert observation["sample_excitation"].shape == (2,)
    assert observation["sample_inhibition"].shape == (2,)
    assert not observation["sample_excitation"].requires_grad
    assert not observation["sample_inhibition"].requires_grad
    _, repeated = layer(x, mutate_state=False)
    finalized = layer.validate_observations_batch(
        ((layer, observation), (layer, repeated))
    )
    assert finalized[0].observation_id == finalized[1].observation_id


def test_gaba_forward_defers_cpu_digest_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = GABAergicLayer(GABAConfig(d_model=8))

    def fail_hot_path(*_args: object, **_kwargs: object) -> str:
        pytest.fail("forward materialized a CPU digest")

    monkeypatch.setattr(layer, "_observation_digest", fail_hot_path)

    delta, observation = layer(torch.randn(2, 3, 8), mutate_state=False)

    assert torch.equal(delta, torch.zeros_like(delta))
    assert observation["observation_id"] is not None


def test_gaba_rejects_mutated_deferred_digest_snapshot() -> None:
    layer = GABAergicLayer(GABAConfig(d_model=8))
    _, observation = layer(torch.randn(2, 3, 8), mutate_state=False)
    deferred_id = observation["observation_id"]
    with torch.no_grad():
        observation["sample_excitation"].add_(1.0)
        deferred_id.sample_excitation.add_(1.0)

    with pytest.raises(ValueError, match="digest"):
        layer.commit_observation(observation)

    assert layer.state_update_count.item() == 0


@pytest.mark.parametrize("shape", [(0, 3, 8), (2, 0, 8)])
def test_gaba_rejects_empty_batch_or_sequence(
    shape: tuple[int, int, int],
) -> None:
    layer = GABAergicLayer(GABAConfig(d_model=8))

    with pytest.raises(ValueError, match="non-empty"):
        layer(torch.empty(shape), mutate_state=False)


def test_gaba_delta_is_inhibitory_and_bounded() -> None:
    layer = GABAergicLayer(GABAConfig(d_model=8, max_delta_ratio=0.25))
    with torch.no_grad():
        layer.inhibitory_weight.copy_(torch.eye(8))
        layer.inhibitory_bias.zero_()
        layer.residual_gate.fill_(10.0)
    x = torch.randn(2, 3, 8)

    delta, _ = layer(x, mutate_state=False)

    x_norm = x.float().norm(dim=-1)
    delta_norm = delta.float().norm(dim=-1)
    assert torch.all(delta_norm <= x_norm * 0.25 + 1e-6)
    assert torch.all((x * delta).sum(dim=-1) <= 1e-6)
    assert not torch.equal(x + delta, 2.0 * x + delta)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_gaba_rejects_nonfinite_input_before_returning_delta(
    bad_value: float,
) -> None:
    layer = GABAergicLayer(GABAConfig(d_model=8))
    x = torch.randn(2, 3, 8)
    x[0, 0, 0] = bad_value

    with pytest.raises(ValueError, match="finite"):
        layer(x, mutate_state=False)


@pytest.mark.parametrize(
    "state_name",
    [
        "inhibitory_weight",
        "inhibitory_bias",
        "residual_gate",
        "gaba_level",
    ],
)
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_gaba_rejects_nonfinite_relevant_state(
    state_name: str,
    bad_value: float,
) -> None:
    layer = GABAergicLayer(GABAConfig(d_model=8))
    with torch.no_grad():
        getattr(layer, state_name).fill_(bad_value)

    with pytest.raises(ValueError, match="finite"):
        layer(torch.randn(2, 3, 8), mutate_state=False)


def test_gaba_commits_once_and_rejects_nonfinite() -> None:
    layer = GABAergicLayer(GABAConfig(d_model=8))
    x = torch.randn(2, 3, 8)
    _, observation = layer(x, mutate_state=False)
    assert layer.state_update_count.item() == 0

    layer.commit_observation(observation)

    assert layer.state_update_count.item() == 1
    with pytest.raises(ValueError, match="already committed"):
        layer.commit_observation(observation)
    bad = dict(observation)
    bad["sample_excitation"] = torch.tensor([float("nan")])
    with pytest.raises(ValueError, match="finite"):
        layer.commit_observation(bad)


def test_gaba_rejects_extreme_float64_evidence_without_state_mutation() -> None:
    layer = GABAergicLayer(GABAConfig(d_model=8))
    sample_excitation = torch.tensor([1e300], dtype=torch.float64)
    sample_inhibition = torch.tensor([1.0], dtype=torch.float64)
    observation = {
        "sample_excitation": sample_excitation,
        "sample_inhibition": sample_inhibition,
        "observation_id": layer._observation_digest(
            sample_excitation,
            sample_inhibition,
        ),
    }
    before = copy.deepcopy(layer.state_dict())

    with pytest.raises(ValueError, match="dtype|float32|finite"):
        layer.commit_observation(observation)

    assert layer.state_dict().keys() == before.keys()
    for key in before:
        assert torch.equal(layer.state_dict()[key], before[key])


def test_block_composes_one_moe_signal_plus_bounded_gaba_delta() -> None:
    class ZeroMixer(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

    class FixedMoE(nn.Module):
        def __init__(self, value: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("value", value)

        def forward(
            self, x: torch.Tensor
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            return self.value.to(x), {"aux_loss": x.new_zeros(())}

    class FixedGABA(nn.Module):
        def __init__(self, delta: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("delta", delta)

        def forward(
            self, x: torch.Tensor, *, mutate_state: bool = False
        ) -> tuple[torch.Tensor, dict[str, object]]:
            assert mutate_state is False
            return self.delta.to(x), {"observation_id": "block-0"}

    config = _small_config(n_layers=1)
    block = DarwinXBlock(config, is_attention_layer=False)
    residual = torch.randn(1, 2, config.d_model)
    moe_out = torch.full_like(residual, 2.0)
    gaba_delta = torch.full_like(residual, -0.25)
    block.norm1 = nn.Identity()
    block.norm2 = nn.Identity()
    block.ssd = ZeroMixer()
    block.moe = FixedMoE(moe_out)
    block.dropout = nn.Identity()
    block.gaba = FixedGABA(gaba_delta)

    actual, aux = block(residual)

    expected = residual + config.residual_scale * (moe_out + gaba_delta)
    duplicated = residual + config.residual_scale * (2.0 * moe_out + gaba_delta)
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, duplicated)
    assert aux["gaba_observation"]["observation_id"] == "block-0"


def test_model_collects_ordered_observations_and_commits_explicitly() -> None:
    torch.manual_seed(51)
    model = DarwinXModel(_small_config())
    output = model(torch.randint(0, 32, (1, 4)), heartbeat=False)

    assert len(output.gaba_observations) == len(model.blocks)
    assert tuple(
        observation["observation_id"] for observation in output.gaba_observations
    ) == tuple(
        stats["gaba_observation"]["observation_id"] for stats in output.moe_stats
    )
    assert all(block.gaba.state_update_count.item() == 0 for block in model.blocks)

    assert model.commit_gaba_observations(output) == len(model.blocks)
    assert all(block.gaba.state_update_count.item() == 1 for block in model.blocks)


def test_model_gaba_commit_preflights_all_layers_atomically() -> None:
    torch.manual_seed(51)
    model = DarwinXModel(_small_config())
    output = model(torch.randint(0, 32, (1, 4)), heartbeat=False)
    tampered_second = dict(output.gaba_observations[1])
    tampered_second["sample_excitation"] = (
        tampered_second["sample_excitation"] + 1.0
    )
    tampered = replace(
        output,
        gaba_observations=(
            output.gaba_observations[0],
            tampered_second,
        ),
    )

    with pytest.raises(ValueError, match="digest"):
        model.commit_gaba_observations(tampered)

    assert [block.gaba.state_update_count.item() for block in model.blocks] == [
        0,
        0,
    ]


def test_model_gaba_commit_rolls_back_every_layer_on_commit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(51)
    model = DarwinXModel(_small_config())
    output = model(torch.randint(0, 32, (1, 4)), heartbeat=False)
    before = [copy.deepcopy(block.gaba.state_dict()) for block in model.blocks]

    def fail_commit(_observation: object) -> None:
        raise RuntimeError("injected commit failure")

    monkeypatch.setattr(
        model.blocks[1].gaba,
        "_commit_validated_observation",
        fail_commit,
    )

    with pytest.raises(RuntimeError, match="injected commit failure"):
        model.commit_gaba_observations(output)

    for block, expected in zip(model.blocks, before, strict=True):
        assert block.gaba.state_dict().keys() == expected.keys()
        for key in expected:
            assert torch.equal(block.gaba.state_dict()[key], expected[key])
        assert block.gaba._committed_observation_ids == set()
        assert tuple(block.gaba._committed_observation_order) == ()
