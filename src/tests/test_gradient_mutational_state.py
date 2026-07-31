from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from f51_darwin.darwin_x import (
    DarwinXConfig,
    DarwinXModel,
    DeepSeekStyleMoE,
    migrate_mutational_state_for_load,
)


def tiny_config() -> DarwinXConfig:
    return DarwinXConfig(
        model_name="F51-gradient-mutational-test",
        vocab_size=64,
        context_length=8,
        inference_context_length=16,
        d_model=16,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        fine_experts=4,
        shared_experts=1,
        experts_per_token=2,
        fine_expert_hidden_dim=8,
        shared_expert_hidden_dim=8,
        mtp_depth=1,
        mtp_weight=0.0,
        jepa_weight=0.0,
        ghost_weight=0.0,
        ghost_mask_ratio=0.0,
        curiosity_weight=0.0,
        heartbeat_enabled=False,
        nitro_gpu_expert_capacity=4,
    )


@pytest.fixture
def moe() -> DeepSeekStyleMoE:
    torch.manual_seed(51)
    return DeepSeekStyleMoE(tiny_config())


def observe(
    moe: DeepSeekStyleMoE,
    norms: torch.Tensor,
    sketches: torch.Tensor,
    routed: torch.Tensor,
    *,
    ghost_loss: float | None = None,
) -> None:
    usage = routed.float()
    if usage.sum() > 0:
        usage = usage / usage.sum()
    moe.neuroendocrine.update_from_local_signals(
        expert_grad_norms=norms,
        expert_usage=usage,
        router_entropy=0.0,
        upstream_grad_norm=0.0,
        ghost_loss=ghost_loss,
        expert_grad_sketches=sketches,
        expert_routed=routed,
        expert_trainable=torch.ones_like(routed),
    )


def test_signed_sketch_detects_reversal_with_identical_norm(moe: DeepSeekStyleMoE) -> None:
    expert = moe.fine_experts[0]
    for index, parameter in enumerate(expert.parameters()):
        values = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape_as(parameter)
        parameter.grad = values.add(index + 1)

    norms_before, sketches_before = moe._compute_expert_gradient_signatures()
    for parameter in expert.parameters():
        assert parameter.grad is not None
        parameter.grad.neg_()
    norms_after, sketches_after = moe._compute_expert_gradient_signatures()

    assert norms_after[0].item() == pytest.approx(norms_before[0].item())
    assert F.cosine_similarity(
        sketches_before[0].unsqueeze(0), sketches_after[0].unsqueeze(0)
    ).item() == pytest.approx(-1.0, abs=1e-6)


def test_direction_stability_and_reversal_are_independent_per_expert(
    moe: DeepSeekStyleMoE,
) -> None:
    system = moe.neuroendocrine
    sketches = torch.zeros(4, 64)
    sketches[0, 0] = 1.0
    sketches[1, 1] = 1.0
    norms = torch.tensor([1.0, 1.0, 0.0, 0.0])
    routed = torch.tensor([True, True, False, False])
    observe(moe, norms, sketches, routed)

    second = sketches.clone()
    second[1].neg_()
    observe(moe, norms, second, routed)

    assert system.grad_direction_stability[0] > 0
    assert system.grad_direction_stability[1] < 0
    assert system.grad_sign_flips[0].item() == pytest.approx(0.0)
    assert system.grad_sign_flips[1].item() == pytest.approx(0.1)


def test_protected_basis_records_retention_and_detects_conflict(
    moe: DeepSeekStyleMoE,
) -> None:
    system = moe.neuroendocrine
    system.step_count.fill_(9)
    sketches = torch.zeros(4, 64)
    sketches[0, 3] = 1.0
    routed = torch.tensor([True, False, False, False])
    norms = torch.tensor([1.0, 0.0, 0.0, 0.0])

    observe(moe, norms, sketches, routed, ghost_loss=1.0)

    assert system.protected_rank[0].item() == 1
    assert torch.count_nonzero(system.protected_subspace[0]).item() > 0

    observe(moe, norms, -sketches, routed)
    assert system.grad_conflict[0].item() == pytest.approx(1.0, abs=1e-6)
    assert system.grad_orthogonal_residual[0].item() == pytest.approx(0.0, abs=1e-6)


def test_mutational_geometry_stays_fp32_inside_bfloat16_model() -> None:
    bf16_moe = DeepSeekStyleMoE(tiny_config()).to(dtype=torch.bfloat16)
    system = bf16_moe.neuroendocrine

    assert bf16_moe.fine_experts[0].gate_proj.weight.dtype == torch.bfloat16
    for name in system._FP32_MUTATIONAL_BUFFERS:
        assert getattr(system, name).dtype == torch.float32

    sketches = torch.zeros(4, 64, dtype=torch.float32)
    sketches[0, 3] = 1.0
    routed = torch.tensor([True, False, False, False])
    norms = torch.tensor([1.0, 0.0, 0.0, 0.0])

    system.step_count.fill_(9)
    observe(bf16_moe, norms, sketches, routed, ghost_loss=1.0)
    assert system.protected_rank[0].item() == 1

    # Exercise the rank>0 matrix products in both geometry measurement and
    # protected-memory consolidation under a BF16 model.
    system.step_count.fill_(19)
    observe(bf16_moe, norms, sketches, routed, ghost_loss=1.0)
    assert system.protected_strength[0, 0].item() > 1.0

    observe(bf16_moe, norms, -sketches, routed)
    assert system.grad_conflict[0].item() == pytest.approx(1.0, abs=1e-6)


def test_unrouted_expert_does_not_accumulate_stagnation(moe: DeepSeekStyleMoE) -> None:
    zero_norms = torch.zeros(4)
    zero_sketches = torch.zeros(4, 64)
    routed = torch.tensor([True, False, False, False])

    for _ in range(5):
        observe(moe, zero_norms, zero_sketches, routed)

    assert moe.neuroendocrine.grad_stagnation_steps.tolist() == [5, 0, 0, 0]


def test_invalid_sketch_shape_is_rejected(moe: DeepSeekStyleMoE) -> None:
    with pytest.raises(ValueError, match="must have shape"):
        observe(
            moe,
            torch.ones(4),
            torch.ones(4, 8),
            torch.ones(4, dtype=torch.bool),
        )


def test_mutational_state_never_suppresses_token_gate(moe: DeepSeekStyleMoE) -> None:
    system = moe.neuroendocrine
    system.grad_sign_flips[0] = 1.0
    system.grad_stagnation_steps[0] = 10_000
    system.grad_magnitude[0] = 0.0

    assert system.expert_gate(0).item() == pytest.approx(1.0)


def test_controller_maps_actions_but_shadow_mode_blocks_execution(
    moe: DeepSeekStyleMoE,
) -> None:
    system = moe.neuroendocrine
    system.grad_observation_count.fill_(20)
    system.grad_magnitude.fill_(2.0)
    system.grad_direction_stability.fill_(0.9)
    system.grad_orthogonal_residual.fill_(0.8)
    system.bdnf.fill_(1.0)
    system.norepinephrine.fill_(0.2)
    before = len(moe.fine_experts)

    moe._pending_autonomic_actions = moe._propose_autonomic_actions()
    assert moe._pending_autonomic_actions["neurogenesis"] is True
    assert moe._pending_autonomic_actions["expand_experts"] == [0, 1, 2, 3]

    proposal = moe.apply_pending_autonomic_actions()
    executed = moe.execute_structural_actions()

    assert proposal["shadow_mode"] is True
    assert moe._structural_event_log[-1]["status"] == "shadow_observation"
    assert executed == {"neurogenesis": 0, "pruned": 0, "expanded": 0}
    assert len(moe.fine_experts) == before


def _active_moe() -> DeepSeekStyleMoE:
    config = DarwinXConfig.from_mapping({
        **tiny_config().__dict__,
        "dae_enabled": True,
        "dae_shadow_mode": False,
    })
    return DeepSeekStyleMoE(config)


def test_active_ignore_zeros_expert_gradient() -> None:
    moe = _active_moe()
    for parameter in moe.fine_experts[0].parameters():
        parameter.grad = torch.ones_like(parameter)
    moe._pending_autonomic_actions = {
        "plasticity": {
            "ignore": [0], "protect": [], "update": [],
            "expand": [], "prune": [], "create": False,
        },
        "shadow_mode": False,
    }

    report = moe.apply_active_gradient_actions()

    assert report["ignored"] == [0]
    assert all(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0
        for parameter in moe.fine_experts[0].parameters()
    )


def test_active_shadow_never_changes_gradient() -> None:
    moe = _active_moe()
    parameter = next(moe.fine_experts[0].parameters())
    parameter.grad = torch.ones_like(parameter)
    before = parameter.grad.clone()
    moe._pending_autonomic_actions = {
        "plasticity": {"ignore": [0], "protect": [], "update": []},
        "shadow_mode": True,
    }

    report = moe.apply_active_gradient_actions()

    assert report["shadow_mode"] is True
    assert torch.equal(parameter.grad, before)


def test_active_update_gain_is_bounded() -> None:
    moe = _active_moe()
    moe.neuroendocrine.dopamine[0] = 100.0
    parameter = next(moe.fine_experts[0].parameters())
    parameter.grad = torch.ones_like(parameter)
    moe._pending_autonomic_actions = {
        "plasticity": {"ignore": [], "protect": [], "update": [0]},
        "shadow_mode": False,
    }

    report = moe.apply_active_gradient_actions()

    assert report["updated"] == [{"expert": 0, "gain": pytest.approx(1.25)}]
    torch.testing.assert_close(parameter.grad, torch.full_like(parameter.grad, 1.25))


def test_active_protect_removes_conflicting_block_component() -> None:
    moe = _active_moe()
    parameter = next(moe.fine_experts[0].parameters())
    parameter.grad = torch.arange(
        1, parameter.numel() + 1, dtype=parameter.dtype
    ).reshape_as(parameter)
    before = parameter.grad.clone()
    sketch = moe._signed_block_sketch(
        parameter.grad, moe.neuroendocrine.gradient_sketch_dim
    )
    sketch = F.normalize(sketch.float(), dim=0)
    moe.neuroendocrine.protected_subspace[0, 0].copy_(-sketch)
    moe.neuroendocrine.protected_rank[0] = 1
    moe._pending_autonomic_actions = {
        "plasticity": {"ignore": [], "protect": [0], "update": []},
        "shadow_mode": False,
    }

    report = moe.apply_active_gradient_actions()

    assert report["protected"] == [0]
    assert not torch.equal(parameter.grad, before)
    assert parameter.grad.norm() < before.norm()


def test_nitro_enabled_is_a_consumed_config_field() -> None:
    config = DarwinXConfig.from_mapping({
        **tiny_config().__dict__,
        "nitro_enabled": False,
    })
    moe = DeepSeekStyleMoE(config)

    assert moe.nitro_enabled is False


def test_ghost_and_main_forwards_produce_one_mutational_observation() -> None:
    config = tiny_config()
    config = DarwinXConfig.from_mapping({
        **config.__dict__,
        "ghost_weight": 0.07,
        "ghost_mask_ratio": 1.0,
    })
    torch.manual_seed(51)
    model = DarwinXModel(config)
    batch = torch.randint(4, config.vocab_size, (1, 6))

    output = model(batch, labels=batch)
    assert output.loss is not None
    output.loss.backward()

    system = model.blocks[0].moe.neuroendocrine
    assert system.step_count.item() == 1
    assert system.grad_observation_count.max().item() == 1
    assert model.blocks[0].moe._autonomic_forward_uses == 0


def test_dae_accumulation_commits_one_observation_per_optimizer_boundary() -> None:
    config = DarwinXConfig.from_mapping({
        **tiny_config().__dict__,
        "dae_enabled": True,
        "dae_shadow_mode": False,
    })
    model = DarwinXModel(config)
    batch = torch.randint(4, config.vocab_size, (1, 6))
    system = model.blocks[0].moe.neuroendocrine

    for _ in range(2):
        output = model(batch, labels=batch)
        assert output.loss is not None
        (output.loss / 2).backward()

    assert system.step_count.item() == 0
    model.apply_active_gradient_actions()
    assert system.step_count.item() == 1
    model.apply_pending_autonomic_actions()


def test_structural_changes_resize_all_mutational_buffers(moe: DeepSeekStyleMoE) -> None:
    created = moe._create_expert()
    assert created == 4
    for name in (
        "grad_magnitude", "grad_direction_ema", "previous_grad_sketch",
        "grad_direction_stability", "grad_sign_flips", "grad_stagnation_steps",
        "grad_observation_count", "protected_subspace", "protected_strength",
        "protected_rank", "grad_conflict", "grad_orthogonal_residual",
    ):
        assert getattr(moe.neuroendocrine, name).shape[0] == 5

    assert moe._apoptosis(1) is True
    for name in (
        "grad_magnitude", "grad_direction_ema", "previous_grad_sketch",
        "grad_direction_stability", "grad_sign_flips", "grad_stagnation_steps",
        "grad_observation_count", "protected_subspace", "protected_strength",
        "protected_rank", "grad_conflict", "grad_orthogonal_residual",
    ):
        assert getattr(moe.neuroendocrine, name).shape[0] == 4


def test_pre_v2_v7_checkpoint_initializes_geometry_then_loads_strictly() -> None:
    source = DarwinXModel(tiny_config())
    legacy_state = {
        key: value.detach().clone() for key, value in source.state_dict().items()
    }
    for key in list(legacy_state):
        if key.endswith("mutational_state_version"):
            legacy_state.pop(key)
        elif key.endswith("grad_magnitude"):
            legacy_state[key].fill_(7.0)
        elif key.endswith("protected_subspace"):
            experts = legacy_state[key].shape[0]
            legacy_state[key] = torch.zeros(experts, 64)

    restored = DarwinXModel(tiny_config())
    initialized = migrate_mutational_state_for_load(legacy_state, restored)
    restored.load_state_dict(legacy_state, strict=True)

    assert initialized
    assert restored.blocks[0].moe.neuroendocrine.mutational_state_version.item() == 2
    assert torch.count_nonzero(
        restored.blocks[0].moe.neuroendocrine.grad_magnitude
    ).item() == 0
    assert restored.blocks[0].moe.neuroendocrine.protected_subspace.shape == (4, 4, 64)


def test_current_mutational_checkpoint_preserves_accumulated_state() -> None:
    model = DarwinXModel(tiny_config())
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    magnitude_key = next(key for key in state if key.endswith("grad_magnitude"))
    state[magnitude_key].fill_(3.0)

    initialized = migrate_mutational_state_for_load(state, model)

    assert initialized == []
    assert torch.all(state[magnitude_key] == 3.0)
