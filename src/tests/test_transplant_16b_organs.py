from __future__ import annotations

import torch

from f51_darwin.transplant_16b.organs import (
    derive_organ_projection,
    expand_square,
    verify_organ_subspace,
)


def test_square_growth_reproduces_source_subspace() -> None:
    generator = torch.Generator().manual_seed(23)
    source = torch.randn(8, 8, generator=generator)
    projection = derive_organ_projection(
        source_dim=8,
        target_dim=16,
        seed=23,
    )

    grown = expand_square(source, projection, complement="zero")

    assert verify_organ_subspace(source, grown, projection) >= 0.99


def test_identity_complement_is_identity_off_source_subspace() -> None:
    source = torch.eye(8)
    projection = derive_organ_projection(
        source_dim=8,
        target_dim=16,
        seed=29,
    )

    grown = expand_square(source, projection, complement="identity")
    residual = (
        torch.eye(16)
        - projection.matrix @ projection.matrix.T
    )

    torch.testing.assert_close(
        grown @ residual,
        residual,
        atol=1e-5,
        rtol=1e-5,
    )


def test_projection_is_semi_orthogonal_and_deterministic() -> None:
    first = derive_organ_projection(
        source_dim=8,
        target_dim=16,
        seed=31,
    )
    second = derive_organ_projection(
        source_dim=8,
        target_dim=16,
        seed=31,
    )

    torch.testing.assert_close(first.matrix, second.matrix)
    torch.testing.assert_close(
        first.matrix.T @ first.matrix,
        torch.eye(8),
        atol=1e-5,
        rtol=1e-5,
    )
    assert first.identity() == second.identity()
