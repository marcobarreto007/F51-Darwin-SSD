from __future__ import annotations

import math
from dataclasses import dataclass

import torch


LEGACY_LOSS_SEMANTICS_VERSION = 1
CAUSAL_LOSS_SEMANTICS_VERSION = 2
LOSS_SEMANTICS_VERSION = CAUSAL_LOSS_SEMANTICS_VERSION
SUPPORTED_LOSS_SEMANTICS_VERSIONS = frozenset(
    {LEGACY_LOSS_SEMANTICS_VERSION, CAUSAL_LOSS_SEMANTICS_VERSION}
)


@dataclass(frozen=True)
class LossPolicy:
    """Versioned weights and adaptation policy for the training objective."""

    mtp_scale: float
    jepa_scale: float
    aux_scale: float
    ghost_scale: float
    spider_scale: float = 0.0
    aux_adaptive: bool = False
    loss_semantics_version: int = LEGACY_LOSS_SEMANTICS_VERSION

    def __post_init__(self) -> None:
        for name in (
            "mtp_scale",
            "jepa_scale",
            "aux_scale",
            "ghost_scale",
            "spider_scale",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0.")
        if self.loss_semantics_version not in SUPPORTED_LOSS_SEMANTICS_VERSIONS:
            raise ValueError(
                "Unsupported loss_semantics_version "
                f"{self.loss_semantics_version}; expected one of "
                f"{sorted(SUPPORTED_LOSS_SEMANTICS_VERSIONS)}."
            )


@dataclass(frozen=True)
class LossTerms:
    """Immutable raw or effective scalar objective terms."""

    lm: torch.Tensor
    mtp: torch.Tensor
    jepa: torch.Tensor
    aux: torch.Tensor
    ghost: torch.Tensor
    spider: torch.Tensor | None = None


@dataclass(frozen=True)
class ComposedLoss:
    """Auditable result of applying one loss policy to raw terms."""

    raw: LossTerms
    effective: LossTerms
    total: torch.Tensor
    loss_semantics_version: int


def _scaled(term: torch.Tensor, scale: float) -> torch.Tensor:
    if scale == 0.0:
        # This branch is deliberate: ``term * 0`` would preserve a gradient
        # edge and turn a non-finite raw term into NaN.
        return term.new_zeros(())
    if scale == 1.0:
        return term
    return term * scale


def _adapt_aux(
    lm: torch.Tensor,
    configured_aux: torch.Tensor,
    *,
    max_factor: float,
) -> torch.Tensor:
    """Apply the historical adaptive ratio with a versioned amplification cap."""

    lm_value = lm.detach().float().clamp(min=1e-4, max=50.0)
    aux_value = configured_aux.detach().float().clamp(min=1e-6, max=50.0)
    aux_ratio = aux_value / (lm_value + aux_value + 1e-8)
    if not bool((aux_ratio > 0.50).item()):
        return configured_aux

    target_ratio = 0.30
    factor = target_ratio * lm_value / (
        aux_value * (1.0 - target_ratio) + 1e-8
    )
    factor = factor.clamp(0.05, max_factor)
    return configured_aux * factor.to(
        device=configured_aux.device,
        dtype=configured_aux.dtype,
    )


def compose_loss(raw: LossTerms, policy: LossPolicy) -> ComposedLoss:
    """Compose every training objective exactly once under a versioned policy."""

    if policy.loss_semantics_version == LEGACY_LOSS_SEMANTICS_VERSION:
        # V1 is the exact v7 objective: aux_loss_scale was persisted but ignored.
        effective_aux = raw.aux
        if policy.aux_adaptive:
            effective_aux = _adapt_aux(
                raw.lm,
                effective_aux,
                max_factor=5.0,
            )
    else:
        effective_aux = _scaled(raw.aux, policy.aux_scale)
        if policy.aux_adaptive and policy.aux_scale > 0.0:
            effective_aux = _adapt_aux(
                raw.lm,
                effective_aux,
                max_factor=1.0,
            )

    raw_spider = (
        raw.spider
        if raw.spider is not None
        else raw.lm.new_zeros(())
    )
    effective_spider = (
        _scaled(raw_spider, policy.spider_scale)
        if policy.loss_semantics_version == CAUSAL_LOSS_SEMANTICS_VERSION
        else raw_spider.new_zeros(())
    )
    effective = LossTerms(
        lm=raw.lm,
        mtp=_scaled(raw.mtp, policy.mtp_scale),
        jepa=_scaled(raw.jepa, policy.jepa_scale),
        aux=effective_aux,
        ghost=_scaled(raw.ghost, policy.ghost_scale),
        spider=effective_spider,
    )
    total = (
        effective.lm
        + effective.mtp
        + effective.jepa
        + effective.aux
        + effective.ghost
        + effective.spider
    )
    return ComposedLoss(
        raw=raw,
        effective=effective,
        total=total,
        loss_semantics_version=policy.loss_semantics_version,
    )
