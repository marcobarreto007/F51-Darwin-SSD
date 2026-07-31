from __future__ import annotations

import math

import pytest
import torch

from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.losses import (
    CAUSAL_LOSS_SEMANTICS_VERSION,
    LEGACY_LOSS_SEMANTICS_VERSION,
    LOSS_SEMANTICS_VERSION,
    LossPolicy,
    LossTerms,
    compose_loss,
)
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.darwin_x_core.state import _stable_cross_entropy


def _terms(*, aux: torch.Tensor) -> LossTerms:
    return LossTerms(
        lm=torch.tensor(2.0, requires_grad=True),
        mtp=torch.tensor(3.0, requires_grad=True),
        jepa=torch.tensor(5.0, requires_grad=True),
        aux=aux,
        ghost=torch.tensor(7.0, requires_grad=True),
    )


def test_chunked_stable_cross_entropy_matches_unchunked_gradient() -> None:
    torch.manual_seed(51)
    baseline_logits = torch.randn(2, 5, 17, requires_grad=True)
    chunked_logits = baseline_logits.detach().clone().requires_grad_(True)
    labels = torch.randint(0, 17, (2, 5))
    labels[0, 1] = -100

    baseline = _stable_cross_entropy(
        baseline_logits,
        labels,
        token_chunk_size=10,
    )
    chunked = _stable_cross_entropy(
        chunked_logits,
        labels,
        token_chunk_size=2,
    )
    baseline.backward()
    chunked.backward()

    torch.testing.assert_close(chunked, baseline, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        chunked_logits.grad,
        baseline_logits.grad,
        rtol=1e-5,
        atol=1e-6,
    )


def _policy(
    *,
    aux_scale: float,
    adaptive: bool = False,
    version: int = CAUSAL_LOSS_SEMANTICS_VERSION,
) -> LossPolicy:
    return LossPolicy(
        mtp_scale=0.15,
        jepa_scale=0.05,
        aux_scale=aux_scale,
        ghost_scale=0.07,
        aux_adaptive=adaptive,
        loss_semantics_version=version,
    )


def _small_config(**overrides: object) -> DarwinXConfig:
    values = {
        "vocab_size": 32,
        "context_length": 8,
        "inference_context_length": 16,
        "d_model": 16,
        "n_layers": 4,
        "n_heads": 4,
        "n_kv_heads": 2,
        "fine_experts": 2,
        "shared_experts": 1,
        "experts_per_token": 1,
        "fine_expert_hidden_dim": 8,
        "shared_expert_hidden_dim": 8,
        "mtp_depth": 1,
        "aux_loss_adaptive": False,
        "ghost_enabled": False,
        "spider_sense_enabled": False,
        "heartbeat_enabled": False,
    }
    values.update(overrides)
    return DarwinXConfig(**values)


def test_default_config_uses_legacy_loss_semantics() -> None:
    assert DarwinXConfig().loss_semantics_version == LEGACY_LOSS_SEMANTICS_VERSION


@pytest.mark.parametrize("configured_scale", [0.0, 0.25, 1.0])
def test_legacy_v1_ignores_aux_scale_and_preserves_historical_objective(
    configured_scale: float,
) -> None:
    raw_aux = torch.tensor(0.8, requires_grad=True)
    terms = _terms(aux=raw_aux)
    composed = compose_loss(
        terms,
        _policy(
            aux_scale=configured_scale,
            adaptive=False,
            version=LEGACY_LOSS_SEMANTICS_VERSION,
        ),
    )
    historical = (
        terms.lm
        + 0.15 * terms.mtp
        + 0.05 * terms.jepa
        + terms.aux
        + 0.07 * terms.ghost
    )

    torch.testing.assert_close(composed.total, historical, rtol=0.0, atol=0.0)
    assert composed.effective.aux is raw_aux
    assert composed.loss_semantics_version == LEGACY_LOSS_SEMANTICS_VERSION


def test_legacy_v1_adaptive_aux_matches_historical_formula() -> None:
    raw_aux = torch.tensor(8.0, requires_grad=True)
    terms = _terms(aux=raw_aux)
    composed = compose_loss(
        terms,
        _policy(
            aux_scale=0.0,
            adaptive=True,
            version=LEGACY_LOSS_SEMANTICS_VERSION,
        ),
    )
    expected_factor = torch.tensor(
        0.30 * 2.0 / (8.0 * (1.0 - 0.30) + 1e-8)
    ).clamp(0.05, 5.0)

    torch.testing.assert_close(
        composed.effective.aux,
        raw_aux * expected_factor,
    )


def test_causal_v2_zero_aux_scale_is_exact_and_disconnects_aux_gradient() -> None:
    raw_aux = torch.tensor(11.0, requires_grad=True)
    composed = compose_loss(_terms(aux=raw_aux), _policy(aux_scale=0.0))

    assert composed.effective.aux.item() == 0.0
    assert composed.raw.aux is raw_aux
    assert composed.total.item() == pytest.approx(2.0 + 0.15 * 3.0 + 0.05 * 5.0 + 0.07 * 7.0)

    composed.total.backward()
    assert raw_aux.grad is None


def test_causal_v2_zero_aux_scale_never_multiplies_zero_by_non_finite_raw_loss() -> None:
    for raw_value in (float("nan"), float("inf"), -float("inf")):
        raw_aux = torch.tensor(raw_value, requires_grad=True)
        composed = compose_loss(_terms(aux=raw_aux), _policy(aux_scale=0.0))

        assert composed.effective.aux.item() == 0.0
        assert torch.isfinite(composed.total)


def test_causal_v2_fractional_aux_scale_preserves_raw_and_exposes_effective_term() -> None:
    raw_aux = torch.tensor(8.0, requires_grad=True)
    composed = compose_loss(_terms(aux=raw_aux), _policy(aux_scale=0.25))

    assert composed.raw.aux is raw_aux
    assert composed.effective.aux.item() == pytest.approx(2.0)
    assert composed.loss_semantics_version == LOSS_SEMANTICS_VERSION


def test_causal_v2_scale_one_without_adaptation_matches_legacy_objective() -> None:
    raw_aux = torch.tensor(0.8, requires_grad=True)
    terms = _terms(aux=raw_aux)
    composed = compose_loss(terms, _policy(aux_scale=1.0, adaptive=False))
    legacy = (
        terms.lm
        + 0.15 * terms.mtp
        + 0.05 * terms.jepa
        + terms.aux
        + 0.07 * terms.ghost
    )

    torch.testing.assert_close(composed.total, legacy, rtol=0.0, atol=0.0)
    assert composed.effective.aux is raw_aux


def test_causal_v2_adaptation_can_only_reduce_configured_aux_contribution() -> None:
    raw_aux = torch.tensor(8.0, requires_grad=True)
    configured = compose_loss(_terms(aux=raw_aux), _policy(aux_scale=0.5, adaptive=False))
    adaptive = compose_loss(_terms(aux=raw_aux), _policy(aux_scale=0.5, adaptive=True))

    assert 0.0 < adaptive.effective.aux.item() <= configured.effective.aux.item()
    assert adaptive.effective.aux.item() < configured.effective.aux.item()


@pytest.mark.parametrize(
    ("field", "bad_scale"),
    [
        (field, value)
        for field in ("mtp_weight", "jepa_weight", "ghost_weight", "aux_loss_scale")
        for value in (-1.0, float("nan"), float("inf"), -float("inf"))
    ],
)
def test_config_rejects_non_finite_or_negative_loss_scales(
    field: str,
    bad_scale: float,
) -> None:
    with pytest.raises(ValueError, match=field):
        DarwinXConfig(**{field: bad_scale})


def test_config_and_policy_reject_unsupported_loss_semantics_version() -> None:
    with pytest.raises(ValueError, match="loss_semantics_version"):
        DarwinXConfig(loss_semantics_version=999)

    with pytest.raises(ValueError, match="loss_semantics_version"):
        _policy(aux_scale=1.0, version=999)


def test_loss_policy_rejects_invalid_scales() -> None:
    for field in ("mtp_scale", "jepa_scale", "aux_scale", "ghost_scale"):
        kwargs = {
            "mtp_scale": 0.15,
            "jepa_scale": 0.05,
            "aux_scale": 1.0,
            "ghost_scale": 0.07,
        }
        kwargs[field] = math.nan
        with pytest.raises(ValueError, match=field):
            LossPolicy(**kwargs)


def test_causal_v2_model_output_keeps_raw_aux_and_reports_effective_aux() -> None:
    torch.manual_seed(51)
    config = _small_config(
        aux_loss_scale=0.25,
        loss_semantics_version=CAUSAL_LOSS_SEMANTICS_VERSION,
    )
    model = DarwinXModel(config)
    input_ids = torch.randint(4, config.vocab_size, (1, 6))

    output = model(input_ids, labels=input_ids)

    assert output.aux_loss is not None
    assert output.effective_aux_loss is not None
    torch.testing.assert_close(
        output.effective_aux_loss,
        output.aux_loss * config.aux_loss_scale,
    )


def test_legacy_v1_model_keeps_aux_gradient_when_persisted_scale_is_zero() -> None:
    torch.manual_seed(51)
    config = _small_config(
        aux_loss_scale=0.0,
        loss_semantics_version=LEGACY_LOSS_SEMANTICS_VERSION,
    )
    model = DarwinXModel(config)
    input_ids = torch.randint(4, config.vocab_size, (1, 6))

    output = model(input_ids, labels=input_ids)
    assert output.loss is not None
    assert output.aux_loss is not None
    assert output.effective_aux_loss is output.aux_loss
    output.aux_loss.retain_grad()
    output.loss.backward()

    assert output.aux_loss.grad is not None
    torch.testing.assert_close(
        output.aux_loss.grad,
        torch.ones_like(output.aux_loss),
    )


def test_model_total_is_exact_and_does_not_double_count_aux() -> None:
    torch.manual_seed(51)
    config = _small_config(
        aux_loss_scale=0.25,
        loss_semantics_version=CAUSAL_LOSS_SEMANTICS_VERSION,
    )
    model = DarwinXModel(config)
    input_ids = torch.randint(4, config.vocab_size, (1, 6))

    output = model(input_ids, labels=input_ids)

    assert output.loss is not None
    assert output.lm_loss is not None
    assert output.mtp_loss is not None
    assert output.jepa_loss is not None
    assert output.effective_aux_loss is not None
    assert output.ghost_loss is not None
    expected = (
        output.lm_loss
        + config.mtp_weight * output.mtp_loss
        + config.jepa_weight * output.jepa_loss
        + output.effective_aux_loss
        + output.ghost_loss
    )
    torch.testing.assert_close(output.loss, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("aux_scale", "expected_grad"),
    [(0.0, None), (0.25, 0.25)],
)
def test_causal_v2_model_aux_gradient_matches_configured_scale(
    aux_scale: float,
    expected_grad: float | None,
) -> None:
    torch.manual_seed(51)
    config = _small_config(
        aux_loss_scale=aux_scale,
        loss_semantics_version=CAUSAL_LOSS_SEMANTICS_VERSION,
    )
    model = DarwinXModel(config)
    input_ids = torch.randint(4, config.vocab_size, (1, 6))

    output = model(input_ids, labels=input_ids)
    assert output.loss is not None
    assert output.aux_loss is not None
    output.aux_loss.retain_grad()
    output.loss.backward()

    if expected_grad is None:
        assert output.aux_loss.grad is None
    else:
        assert output.aux_loss.grad is not None
        torch.testing.assert_close(
            output.aux_loss.grad,
            torch.full_like(output.aux_loss, expected_grad),
        )


def test_causal_v2_enabled_ghost_is_counted_once_and_has_scaled_gradient() -> None:
    torch.manual_seed(51)
    config = _small_config(
        ghost_enabled=True,
        ghost_weight=0.2,
        ghost_mask_ratio=1.0,
        loss_semantics_version=CAUSAL_LOSS_SEMANTICS_VERSION,
    )
    model = DarwinXModel(config)
    raw_ghost = torch.tensor(4.0, requires_grad=True)
    model._ghost_loss = lambda *_args: raw_ghost
    input_ids = torch.randint(4, config.vocab_size, (1, 6))

    output = model(input_ids, labels=input_ids)

    assert output.loss is not None
    assert output.lm_loss is not None
    assert output.mtp_loss is not None
    assert output.jepa_loss is not None
    assert output.effective_aux_loss is not None
    assert output.ghost_loss is not None
    torch.testing.assert_close(
        output.ghost_loss, raw_ghost * config.ghost_weight
    )
    expected = (
        output.lm_loss
        + config.mtp_weight * output.mtp_loss
        + config.jepa_weight * output.jepa_loss
        + output.effective_aux_loss
        + raw_ghost * config.ghost_weight
    )
    torch.testing.assert_close(output.loss, expected, rtol=0.0, atol=0.0)

    output.loss.backward()
    torch.testing.assert_close(
        raw_ghost.grad, torch.tensor(config.ghost_weight)
    )


def test_compose_loss_gradients_match_every_configured_scale() -> None:
    raw_aux = torch.tensor(11.0, requires_grad=True)
    terms = _terms(aux=raw_aux)
    policy = _policy(aux_scale=0.25, adaptive=False)

    compose_loss(terms, policy).total.backward()

    torch.testing.assert_close(terms.lm.grad, torch.tensor(1.0))
    torch.testing.assert_close(terms.mtp.grad, torch.tensor(policy.mtp_scale))
    torch.testing.assert_close(terms.jepa.grad, torch.tensor(policy.jepa_scale))
    torch.testing.assert_close(terms.aux.grad, torch.tensor(policy.aux_scale))
    torch.testing.assert_close(terms.ghost.grad, torch.tensor(policy.ghost_scale))


def test_causal_v2_zero_ghost_scale_is_exact_and_disconnects_gradient() -> None:
    raw_ghost = torch.tensor(float("inf"), requires_grad=True)
    terms = _terms(aux=torch.tensor(0.0, requires_grad=True))
    terms = LossTerms(
        lm=terms.lm,
        mtp=terms.mtp,
        jepa=terms.jepa,
        aux=terms.aux,
        ghost=raw_ghost,
    )
    policy = LossPolicy(
        mtp_scale=0.15,
        jepa_scale=0.05,
        aux_scale=0.0,
        ghost_scale=0.0,
        loss_semantics_version=CAUSAL_LOSS_SEMANTICS_VERSION,
    )

    composed = compose_loss(terms, policy)

    assert composed.raw.ghost is raw_ghost
    assert composed.effective.ghost.item() == 0.0
    assert torch.isfinite(composed.total)
    composed.total.backward()
    assert raw_ghost.grad is None
