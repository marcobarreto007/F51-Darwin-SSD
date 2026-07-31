from __future__ import annotations

import math

import pytest
import torch

from f51_darwin.transfer.ngram import NGramBaseline, fit_ngram_baseline

DEVICE = torch.device("cpu")


def _chunks(rows: list[list[int]]) -> list[torch.Tensor]:
    return [torch.tensor(row, dtype=torch.long) for row in rows]


def test_baseline_learns_a_deterministic_template() -> None:
    """Num texto onde o proximo token e determinado, a NLL vai a quase zero.

    E exatamente a assinatura do holdout degenerado que o portao precisa pegar.
    """
    cycle = [3, 4, 5, 6] * 8
    train = _chunks([cycle for _ in range(64)])
    baseline = fit_ngram_baseline(
        train, vocab_size=16, device=DEVICE, max_chunks=64
    )
    held = _chunks([cycle for _ in range(4)])
    assert baseline.nll(held, device=DEVICE) < 0.05


def test_baseline_stays_finite_on_unseen_tokens() -> None:
    train = _chunks([[1, 2, 3, 4] * 4 for _ in range(8)])
    baseline = fit_ngram_baseline(
        train, vocab_size=32, device=DEVICE, max_chunks=8
    )
    unseen = _chunks([[29, 30, 31, 29, 30, 31, 29, 30]])
    value = baseline.nll(unseen, device=DEVICE)
    assert math.isfinite(value)
    assert value > 1.0


def test_baseline_counts_only_the_chunks_it_is_given() -> None:
    """O baseline nao pode enxergar o holdout -- so o treino."""
    train = _chunks([[1, 2, 1, 2, 1, 2]] * 4)
    baseline = fit_ngram_baseline(
        train, vocab_size=8, device=DEVICE, max_chunks=4
    )
    assert baseline.train_tokens == 24
    # 1->2 e 2->1 sao os unicos pares.
    assert baseline.distinct_pairs == 2


def test_baseline_rejects_tokens_outside_the_vocabulary() -> None:
    train = _chunks([[1, 2, 99, 2]])
    with pytest.raises(ValueError, match="vocabulary"):
        fit_ngram_baseline(train, vocab_size=8, device=DEVICE, max_chunks=4)


def test_baseline_rejects_degenerate_configuration() -> None:
    train = _chunks([[1, 2, 3, 4]])
    with pytest.raises(ValueError):
        fit_ngram_baseline(train, vocab_size=0, device=DEVICE)
    with pytest.raises(ValueError):
        fit_ngram_baseline(train, vocab_size=8, device=DEVICE, alpha=0.0)
    with pytest.raises(ValueError):
        fit_ngram_baseline(train, vocab_size=8, device=DEVICE, max_chunks=0)


def test_probabilities_normalise_over_the_vocabulary() -> None:
    train = _chunks([[1, 2, 3, 1, 2, 3]] * 4)
    baseline = fit_ngram_baseline(
        train, vocab_size=8, device=DEVICE, max_chunks=4
    )
    context = torch.full((8,), 1, dtype=torch.long)
    targets = torch.arange(8, dtype=torch.long)
    total = baseline.log_prob(context, targets).exp().sum()
    assert torch.isclose(total, torch.tensor(1.0), atol=1e-5)


def test_unseen_context_backs_off_to_the_unigram() -> None:
    train = _chunks([[1, 2, 1, 2, 1, 2]] * 4)
    baseline = fit_ngram_baseline(
        train, vocab_size=8, device=DEVICE, max_chunks=4
    )
    unseen_context = torch.tensor([7], dtype=torch.long)
    target = torch.tensor([2], dtype=torch.long)
    observed = baseline.log_prob(unseen_context, target).exp()
    assert torch.isclose(observed, baseline.unigram[2], atol=1e-6)


def test_evaluation_rejects_empty_input() -> None:
    train = _chunks([[1, 2, 1, 2]] * 2)
    baseline = fit_ngram_baseline(
        train, vocab_size=8, device=DEVICE, max_chunks=2
    )
    with pytest.raises(ValueError):
        baseline.nll([], device=DEVICE)
    assert isinstance(baseline, NGramBaseline)
