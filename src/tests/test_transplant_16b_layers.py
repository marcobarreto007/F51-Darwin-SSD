from __future__ import annotations

import torch

from f51_darwin.transplant_16b.layers import (
    monotonic_layer_groups,
    select_and_fold_heads,
)


def test_layer_groups_cover_24_monotonically() -> None:
    groups = monotonic_layer_groups(24, 16)

    assert len(groups) == 16
    assert tuple(index for group in groups for index in group) == tuple(
        range(24)
    )
    assert all(len(group) in {1, 2} for group in groups)


def test_head_fold_selects_30_and_accounts_for_two() -> None:
    generator = torch.Generator().manual_seed(17)
    outputs = torch.randn(20, 32, 4, generator=generator)
    outputs[:, 30] = outputs[:, 0] * 0.5
    outputs[:, 31] = outputs[:, 1] * 0.25

    mapping = select_and_fold_heads(outputs, target_heads=30)

    assert len(mapping.selected_heads) == 30
    assert len(mapping.folded_heads) == 2
    assert set(mapping.selected_heads) | set(mapping.folded_heads) == set(
        range(32)
    )
    assert set(mapping.fold_targets).issubset(set(mapping.selected_heads))
    assert torch.isfinite(mapping.fold_coefficients).all()
