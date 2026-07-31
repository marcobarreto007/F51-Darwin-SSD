from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


def monotonic_layer_groups(
    donor_layers: int,
    target_layers: int,
) -> tuple[tuple[int, ...], ...]:
    if donor_layers < 1 or target_layers < 1:
        raise ValueError("layer counts must be positive")
    if target_layers > donor_layers:
        raise ValueError("target_layers cannot exceed donor_layers")
    groups: list[tuple[int, ...]] = []
    for target_index in range(target_layers):
        start = target_index * donor_layers // target_layers
        stop = (target_index + 1) * donor_layers // target_layers
        group = tuple(range(start, stop))
        if not group:
            raise ValueError("layer grouping produced an empty target group")
        groups.append(group)
    flattened = tuple(index for group in groups for index in group)
    if flattened != tuple(range(donor_layers)):
        raise AssertionError("internal layer coverage failure")
    return tuple(groups)


@dataclass(frozen=True)
class HeadFoldMapping:
    selected_heads: tuple[int, ...]
    folded_heads: tuple[int, ...]
    fold_targets: tuple[int, ...]
    fold_coefficients: torch.Tensor
    contribution_energy: torch.Tensor


def select_and_fold_heads(
    head_outputs: torch.Tensor,
    *,
    target_heads: int,
) -> HeadFoldMapping:
    if head_outputs.ndim < 3:
        raise ValueError(
            "head_outputs must have [..., heads, head_dim] shape"
        )
    donor_heads = int(head_outputs.shape[-2])
    if not 0 < target_heads <= donor_heads:
        raise ValueError("target_heads must be in (0, donor_heads]")

    by_head = head_outputs.float().movedim(-2, 0).reshape(donor_heads, -1)
    energy = by_head.square().mean(dim=1)
    ranked = sorted(
        range(donor_heads),
        key=lambda index: (-float(energy[index]), index),
    )
    selected = tuple(sorted(ranked[:target_heads]))
    folded = tuple(sorted(ranked[target_heads:]))
    if not folded:
        return HeadFoldMapping(
            selected_heads=selected,
            folded_heads=(),
            fold_targets=(),
            fold_coefficients=torch.empty(0, dtype=torch.float32),
            contribution_energy=energy,
        )

    selected_vectors = F.normalize(by_head[list(selected)], dim=1, eps=1e-12)
    rejected_vectors = F.normalize(by_head[list(folded)], dim=1, eps=1e-12)
    similarities = rejected_vectors @ selected_vectors.T
    nearest_positions = similarities.abs().argmax(dim=1)
    targets = tuple(selected[int(position)] for position in nearest_positions)
    coefficients: list[torch.Tensor] = []
    for rejected, selected_head in zip(folded, targets, strict=True):
        source = by_head[selected_head]
        residual = by_head[rejected]
        coefficient = torch.dot(source, residual) / torch.dot(
            source,
            source,
        ).clamp_min(1e-12)
        coefficients.append(coefficient)
    return HeadFoldMapping(
        selected_heads=selected,
        folded_heads=folded,
        fold_targets=targets,
        fold_coefficients=torch.stack(coefficients),
        contribution_energy=energy,
    )


def fold_output_projection(
    output_weight: torch.Tensor,
    mapping: HeadFoldMapping,
    *,
    head_dim: int,
) -> torch.Tensor:
    donor_heads = len(mapping.selected_heads) + len(mapping.folded_heads)
    if tuple(output_weight.shape)[1] != donor_heads * head_dim:
        raise ValueError("output projection input does not match donor heads")
    selected_columns = [
        output_weight[:, head * head_dim : (head + 1) * head_dim].float().clone()
        for head in mapping.selected_heads
    ]
    selected_positions = {
        head: position for position, head in enumerate(mapping.selected_heads)
    }
    for rejected, target, coefficient in zip(
        mapping.folded_heads,
        mapping.fold_targets,
        mapping.fold_coefficients,
        strict=True,
    ):
        rejected_slice = output_weight[
            :,
            rejected * head_dim : (rejected + 1) * head_dim,
        ].float()
        selected_columns[selected_positions[target]].add_(
            coefficient * rejected_slice
        )
    return torch.cat(selected_columns, dim=1)
