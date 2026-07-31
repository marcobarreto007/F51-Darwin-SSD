from __future__ import annotations

import torch
from torch.nn import functional as F

from f51_darwin.transplant_16b.assembly import _fast_coupled_expert
from f51_darwin.transplant_16b.moe import (
    CoupledMLP,
    _cosine_kmeans,
    build_expert_state,
    cluster_coupled_neurons,
    fit_router,
)


def _swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Mesmo forward de darwin_x_core.layers.ExpertFFN."""
    return (F.silu(x @ gate.T) * (x @ up.T)) @ down.T


def test_triplets_never_split_and_every_neuron_is_covered() -> None:
    generator = torch.Generator().manual_seed(11)
    mlp = CoupledMLP(
        gate=torch.randn(32, 8, generator=generator),
        up=torch.randn(32, 8, generator=generator),
        down=torch.randn(8, 32, generator=generator),
    )
    signatures = torch.randn(64, 32, generator=generator)

    assignment = cluster_coupled_neurons(
        mlp,
        signatures,
        shared_experts=2,
        fine_experts=4,
        expert_width=6,
        seed=11,
    )

    flattened = [
        index
        for group in assignment.source_neurons
        for index in group
    ]
    assert sorted(flattened) == list(range(32))
    assert len(set(flattened)) == 32
    assert all(count > 0 for count in assignment.target_widths)


def test_kmeans_repairs_multiple_empty_clusters_without_stealing() -> None:
    features = torch.zeros(8, 4)
    assignments = _cosine_kmeans(
        features,
        clusters=4,
        seed=31,
        iterations=2,
    )
    counts = torch.bincount(assignments, minlength=4)
    assert counts.tolist() == [5, 1, 1, 1]


def test_expert_reduction_applies_one_basis_to_gate_up_and_down() -> None:
    generator = torch.Generator().manual_seed(17)
    mlp = CoupledMLP(
        gate=torch.randn(12, 8, generator=generator),
        up=torch.randn(12, 8, generator=generator),
        down=torch.randn(8, 12, generator=generator),
    )

    expert = build_expert_state(
        mlp,
        tuple(range(12)),
        target_width=6,
    )

    assert expert.gate.shape == (6, 8)
    assert expert.up.shape == (6, 8)
    assert expert.down.shape == (8, 6)
    assert expert.basis.shape == (6, 12)
    torch.testing.assert_close(
        expert.basis @ expert.basis.T,
        torch.eye(6),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.isfinite(expert.reconstruction_mse)


def test_expansion_preserves_the_swiglu_function_exactly() -> None:
    """Expansao (o caso de producao, 512 -> 896) nao pode alterar a funcao.

    Reconstruir o PESO nao basta: qualquer basis de colunas ortonormais
    satisfaz B.T @ B == I. Como silu e coordenada-a-coordenada e o produto e
    elementwise, so um scatter comuta com a nao-linearidade. O frame denso
    anterior reconstruia o peso a 3e-17 e ainda assim errava a funcao em 29%.
    """
    generator = torch.Generator().manual_seed(23)
    mlp = CoupledMLP(
        gate=torch.randn(12, 8, generator=generator),
        up=torch.randn(12, 8, generator=generator),
        down=torch.randn(8, 12, generator=generator),
    )
    x = torch.randn(32, 8, generator=generator)
    reference = _swiglu(mlp.gate, mlp.up, mlp.down, x)

    expert = build_expert_state(mlp, tuple(range(12)), target_width=20)

    assert expert.gate.shape == (20, 8)
    assert expert.down.shape == (8, 20)
    torch.testing.assert_close(
        _swiglu(expert.gate, expert.up, expert.down, x),
        reference,
        atol=1e-6,
        rtol=1e-6,
    )


def test_production_expert_builder_preserves_the_swiglu_function() -> None:
    """assembly._fast_coupled_expert e o caminho que o build roda de fato."""
    generator = torch.Generator().manual_seed(29)
    mlp = CoupledMLP(
        gate=torch.randn(12, 8, generator=generator),
        up=torch.randn(12, 8, generator=generator),
        down=torch.randn(8, 12, generator=generator),
    )
    signatures = torch.randn(64, 12, generator=generator)
    x = torch.randn(32, 8, generator=generator)
    reference = _swiglu(mlp.gate, mlp.up, mlp.down, x)

    gate, up, down, error = _fast_coupled_expert(
        mlp,
        signatures,
        tuple(range(12)),
        target_width=20,
        seed=29,
    )

    assert error == 0.0
    assert torch.count_nonzero(gate[12:]) > 0
    assert torch.count_nonzero(up[12:]) > 0
    assert torch.count_nonzero(down[:, 12:]) == 0
    torch.testing.assert_close(
        _swiglu(gate, up, down, x),
        reference,
        atol=1e-6,
        rtol=1e-6,
    )
    train_gate = gate.clone().requires_grad_(True)
    train_up = up.clone().requires_grad_(True)
    train_down = down.clone().requires_grad_(True)
    _swiglu(train_gate, train_up, train_down, x).square().mean().backward()
    assert torch.count_nonzero(train_down.grad[:, 12:]) > 0


def test_real_prototypes_can_cover_an_unobserved_router_expert() -> None:
    inputs = torch.eye(4)
    labels = torch.tensor([0, 0, 1, 1])
    scores = torch.tensor(
        [
            [4.0, 0.0, 0.1],
            [3.0, 0.0, 0.2],
            [0.0, 4.0, 0.3],
            [0.0, 3.0, 0.9],
        ]
    )
    prototypes = scores.argmax(dim=0)
    covered_inputs = torch.cat((inputs, inputs[prototypes]))
    covered_labels = torch.cat((labels, torch.arange(3)))
    result = fit_router(
        covered_inputs,
        covered_labels,
        num_experts=3,
        steps=2,
    )
    assert result.weight.shape == (3, 4)
