from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from f51_darwin.dae_optimizer import (
    DAEHybridOptimizer,
    build_optimizer,
    zeropower_via_newton_schulz,
)


def test_zeropower_preserves_shape_and_is_finite() -> None:
    torch.manual_seed(51)
    matrix = torch.randn(8, 4)

    result = zeropower_via_newton_schulz(matrix, steps=5)

    assert result.shape == matrix.shape
    assert torch.isfinite(result).all()


def test_dae_hybrid_updates_matrix_and_vector() -> None:
    matrix = nn.Parameter(torch.eye(4))
    vector = nn.Parameter(torch.ones(4))
    optimizer = DAEHybridOptimizer([matrix, vector], lr=1e-2)
    matrix_before = matrix.detach().clone()
    vector_before = vector.detach().clone()

    (matrix.sum() + vector.sum()).backward()
    optimizer.step()

    assert not torch.equal(matrix, matrix_before)
    assert not torch.equal(vector, vector_before)
    assert optimizer.state[matrix]["momentum"].shape == matrix.shape
    assert optimizer.state[vector]["rms"].shape == ()


def test_dae_hybrid_state_roundtrip_is_deterministic() -> None:
    left = nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4))
    first = DAEHybridOptimizer([left], lr=1e-2)
    left.sum().backward()
    first.step()
    first.zero_grad(set_to_none=True)
    saved_parameter = left.detach().clone()
    saved_state = copy.deepcopy(first.state_dict())

    right = nn.Parameter(saved_parameter.clone())
    second = DAEHybridOptimizer([right], lr=1e-2)
    second.load_state_dict(saved_state)
    left.sum().backward()
    right.sum().backward()
    first.step()
    second.step()

    torch.testing.assert_close(left, right)


def test_dae_hybrid_rejects_nonfinite_gradient_without_mutation() -> None:
    parameter = nn.Parameter(torch.ones(4))
    optimizer = DAEHybridOptimizer([parameter], lr=1e-2)
    before = parameter.detach().clone()
    parameter.grad = torch.full_like(parameter, float("nan"))

    with pytest.raises(FloatingPointError, match="non-finite gradient"):
        optimizer.step()

    torch.testing.assert_close(parameter, before)


def test_build_optimizer_uses_adaptive_path_for_embeddings() -> None:
    model = nn.ModuleDict({
        "token_embedding": nn.Embedding(32, 8),
        "hidden": nn.Linear(8, 8, bias=False),
    })

    optimizer = build_optimizer(
        "dae_hybrid", model.named_parameters(), lr=1e-3, weight_decay=0.01
    )

    assert isinstance(optimizer, DAEHybridOptimizer)
    spectral = {
        id(parameter)
        for group in optimizer.param_groups
        if group["spectral"]
        for parameter in group["params"]
    }
    assert id(model["hidden"].weight) in spectral
    assert id(model["token_embedding"].weight) not in spectral
