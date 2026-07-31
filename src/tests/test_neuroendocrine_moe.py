import math

import pytest
import torch

from f51_darwin.darwin_x import DarwinXConfig, DeepSeekStyleMoE, NeuroendocrineSystem


def tiny_config() -> DarwinXConfig:
    return DarwinXConfig(
        vocab_size=64,
        context_length=8,
        inference_context_length=16,
        d_model=16,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        fine_experts=4,
        shared_experts=1,
        experts_per_token=2,
        fine_expert_hidden_dim=8,
        shared_expert_hidden_dim=8,
        heartbeat_enabled=False,
    )


def local_update(system: NeuroendocrineSystem, entropy: float) -> None:
    system.update_from_local_signals(
        expert_grad_norms=torch.full((system.num_experts,), 0.1),
        expert_usage=torch.full((system.num_experts,), 1.0 / system.num_experts),
        router_entropy=entropy,
        upstream_grad_norm=0.0,
    )


def global_update(system: NeuroendocrineSystem, loss: float) -> None:
    system.update(
        expert_grad_norms=torch.full((system.num_experts,), 0.1),
        expert_usage=torch.full((system.num_experts,), 1.0 / system.num_experts),
        router_entropy=0.0,
        loss=loss,
        global_grad_norm=0.0,
    )


def test_hormonal_gate_is_neutral_then_bidirectional() -> None:
    system = NeuroendocrineSystem(2)
    assert system.expert_gate(0).item() == pytest.approx(1.0)

    system.dopamine[0] = 0.3
    assert system.expert_gate(0).item() > 1.0

    system.cortisol.fill_(1.0)
    assert system.expert_gate(0).item() < 1.0
    assert 0.5 <= system.expert_gate(0).item() <= 1.5


def test_norepinephrine_tracks_entropy_surprise_not_high_entropy() -> None:
    system = NeuroendocrineSystem(4)
    high_entropy = math.log(4)
    local_update(system, high_entropy)
    local_update(system, high_entropy)
    assert system.norepinephrine.item() == pytest.approx(0.0)

    system = NeuroendocrineSystem(4)
    local_update(system, 0.0)
    local_update(system, high_entropy)
    assert system.norepinephrine.item() > 0.0


def test_homeostatic_sleep_pressure_is_reachable_and_bounded() -> None:
    system = NeuroendocrineSystem(2, ach_tau=50.0)
    for _ in range(100):
        local_update(system, 0.0)
    assert 0.8 < system.ach_pressure.item() < 1.0
    assert system.should_sleep()
    system.trigger_sleep_phase()
    assert system.ach_pressure.item() == 0.0


def test_improving_loss_does_not_create_cortisol() -> None:
    improving = NeuroendocrineSystem(2)
    for loss in torch.linspace(2.0, 0.5, 150).tolist():
        global_update(improving, loss)
    assert improving.cortisol.item() == pytest.approx(0.0)

    worsening = NeuroendocrineSystem(2)
    for loss in torch.linspace(0.5, 2.0, 150).tolist():
        global_update(worsening, loss)
    assert worsening.cortisol.item() > 0.0


def test_backward_observes_plasticity_in_shadow_without_mutating_topology() -> None:
    torch.manual_seed(51)
    moe = DeepSeekStyleMoE(tiny_config())
    before_ids = {id(parameter) for parameter in moe.parameters()}
    before_experts = len(moe.fine_experts)

    x = torch.randn(2, 3, 16, requires_grad=True)
    output, _ = moe(x)
    output.square().mean().backward()

    assert moe._pending_autonomic_actions is not None
    assert moe._pending_autonomic_actions["shadow_mode"] is True
    assert "plasticity" in moe._pending_autonomic_actions
    assert len(moe.fine_experts) == before_experts
    assert {id(parameter) for parameter in moe.parameters()} == before_ids

    result = moe.apply_pending_autonomic_actions()
    assert result["shadow_mode"] is True
    assert len(moe.fine_experts) == before_experts


def test_vertical_routing_bias_is_transient() -> None:
    torch.manual_seed(51)
    moe = DeepSeekStyleMoE(tiny_config()).eval()
    learned_bias = moe.fine_router.expert_bias.detach().clone()
    moe._vertical_bias = torch.tensor([1.0, 0.0, 0.0, 0.0])

    moe(torch.randn(1, 3, 16))

    assert moe._vertical_bias is None
    torch.testing.assert_close(moe.fine_router.expert_bias, learned_bias)


def test_synaptic_pruning_removes_requested_weakest_fraction() -> None:
    moe = DeepSeekStyleMoE(tiny_config())
    expert = moe.fine_experts[0]
    for parameter in expert.parameters():
        parameter.data.uniform_(0.1, 1.0)

    moe._prune_synapses(0, pruning_rate=0.25)

    for parameter in expert.parameters():
        expected = int(parameter.numel() * 0.25)
        assert int((parameter == 0).sum().item()) == expected


def test_sleep_window_does_not_rewrite_expert_weights() -> None:
    moe = DeepSeekStyleMoE(tiny_config())
    before = [parameter.detach().clone() for parameter in moe.fine_experts.parameters()]

    moe._enter_sleep_phase()

    assert moe._sleep_active
    assert moe._sleep_steps_remaining == moe._sleep_phase_duration
    for old, current in zip(before, moe.fine_experts.parameters()):
        torch.testing.assert_close(current, old)
