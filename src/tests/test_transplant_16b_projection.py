from __future__ import annotations

import torch

from f51_darwin.transplant_16b.projection import (
    derive_hidden_projection,
    project_linear,
)


def test_projection_is_semi_orthogonal_and_coherent() -> None:
    generator = torch.Generator().manual_seed(7)
    embedding = torch.randn(128, 8, generator=generator)
    activations = torch.randn(256, 8, generator=generator)

    projection = derive_hidden_projection(
        embedding,
        activations,
        target_dim=6,
        seed=7,
    )

    torch.testing.assert_close(
        projection.matrix @ projection.matrix.T,
        torch.eye(6),
        atol=1e-5,
        rtol=1e-5,
    )
    weight = torch.randn(8, 8, generator=generator)
    expected = projection.matrix @ weight @ projection.matrix.T
    torch.testing.assert_close(
        project_linear(weight, projection, projection),
        expected,
    )


def test_projection_identity_is_deterministic() -> None:
    generator = torch.Generator().manual_seed(13)
    embedding = torch.randn(64, 10, generator=generator)
    activations = torch.randn(32, 10, generator=generator)

    first = derive_hidden_projection(
        embedding,
        activations,
        target_dim=7,
        seed=13,
    )
    second = derive_hidden_projection(
        embedding,
        activations,
        target_dim=7,
        seed=13,
    )

    assert first.identity() == second.identity()
    torch.testing.assert_close(first.matrix, second.matrix)
