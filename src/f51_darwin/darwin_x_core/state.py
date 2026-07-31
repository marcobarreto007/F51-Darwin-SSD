from __future__ import annotations

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


_MUTATIONAL_STATE_SUFFIXES = (
    "mutational_state_version", "grad_magnitude", "grad_direction_ema",
    "previous_grad_sketch", "grad_direction_stability", "grad_sign_flips",
    "grad_stagnation_steps", "grad_observation_count", "protected_subspace",
    "protected_strength", "protected_rank", "grad_conflict",
    "grad_orthogonal_residual",
)


def migrate_mutational_state_for_load(
    state: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> list[str]:
    """Initialize pre-v2 gradient geometry without weakening strict loading."""
    expected = model.state_dict()
    version_suffix = ".mutational_state_version"
    prefixes = [
        key[:-len("mutational_state_version")]
        for key in expected
        if key.endswith(version_suffix)
    ]
    initialized: list[str] = []
    for prefix in prefixes:
        version_key = prefix + "mutational_state_version"
        saved_version = state.get(version_key)
        is_current = (
            isinstance(saved_version, torch.Tensor)
            and saved_version.numel() == 1
            and int(saved_version.item()) == int(expected[version_key].item())
        )
        for suffix in _MUTATIONAL_STATE_SUFFIXES:
            key = prefix + suffix
            if key not in expected:
                continue
            saved = state.get(key)
            shape_matches = (
                isinstance(saved, torch.Tensor)
                and tuple(saved.shape) == tuple(expected[key].shape)
            )
            if not is_current or not shape_matches:
                state[key] = expected[key].detach().clone()
                initialized.append(key)
    return sorted(initialized)


def _cross_entropy_sum(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    safe_logits = torch.nan_to_num(
        logits.float(),
        nan=0.0,
        posinf=30.0,
        neginf=-30.0,
    ).clamp(min=-30.0, max=30.0)
    return F.cross_entropy(
        safe_logits,
        labels,
        ignore_index=-100,
        reduction="sum",
    )


def _stable_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    token_chunk_size: int | None = None,
) -> torch.Tensor:
    """Cross entropy in fp32 with bounded logits so AMP cannot infect training."""

    flat_logits = logits.contiguous().view(-1, logits.size(-1))
    flat_labels = labels.contiguous().view(-1)
    token_count = flat_logits.size(0)
    if token_chunk_size is None:
        token_chunk_size = (
            256
            if flat_logits.numel() >= 16_000_000
            else token_count
        )
    if token_chunk_size < 1:
        raise ValueError("token_chunk_size must be positive")
    if token_chunk_size >= token_count:
        safe_logits = torch.nan_to_num(
            flat_logits.float(),
            nan=0.0,
            posinf=30.0,
            neginf=-30.0,
        ).clamp(min=-30.0, max=30.0)
        return F.cross_entropy(
            safe_logits,
            flat_labels,
            ignore_index=-100,
        )

    total = flat_logits.new_zeros((), dtype=torch.float32)
    for start in range(0, token_count, token_chunk_size):
        stop = min(start + token_chunk_size, token_count)
        chunk_logits = flat_logits[start:stop]
        chunk_labels = flat_labels[start:stop]
        if torch.is_grad_enabled() and chunk_logits.requires_grad:
            chunk_loss = checkpoint(
                _cross_entropy_sum,
                chunk_logits,
                chunk_labels,
                use_reentrant=False,
            )
        else:
            chunk_loss = _cross_entropy_sum(chunk_logits, chunk_labels)
        total = total + chunk_loss
    valid_tokens = flat_labels.ne(-100).sum().clamp_min(1)
    return total / valid_tokens.to(dtype=total.dtype)
