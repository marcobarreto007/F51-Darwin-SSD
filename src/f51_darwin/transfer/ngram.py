"""Baseline de contagem para controlar holdouts degenerados.

Um holdout cujo proximo token e quase sempre determinado pelo anterior nao
mede capacidade: uma tabela de bigramas contada no train chega perto do
minimo sem nenhuma rede. Se o aluno nao bate essa tabela, o numero de NLL
nao prova nada sobre o modelo -- prova que o holdout e degenerado.

Este modulo conta o bigrama SOMENTE no split de treino e avalia no mesmo
holdout de promocao usado pelo aluno e pelo professor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _stack(chunks) -> torch.Tensor:
    if isinstance(chunks, torch.Tensor):
        return chunks if chunks.dim() == 2 else chunks.unsqueeze(0)
    if not chunks:
        raise ValueError("ngram baseline needs at least one chunk")
    return torch.stack(list(chunks))


@dataclass(frozen=True)
class NGramBaseline:
    """Bigrama com backoff para unigrama, contado apenas no treino.

    ``P(t | c) = (count(c, t) + alpha * P_uni(t)) / (count(c) + alpha)``

    Contexto nunca visto cai naturalmente em ``P_uni(t)``, e o unigrama usa
    Laplace, entao a NLL e finita para qualquer token do vocabulario.
    """

    vocab_size: int
    alpha: float
    train_tokens: int
    distinct_pairs: int
    unigram: torch.Tensor  # [V] probabilidades
    context_total: torch.Tensor  # [V] contagem de cada token como contexto
    pair_keys: torch.Tensor  # [P] chaves ordenadas: prev * V + next
    pair_counts: torch.Tensor  # [P] contagem de cada par

    @torch.no_grad()
    def log_prob(
        self, context: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        keys = context.to(torch.long) * self.vocab_size + target.to(torch.long)
        index = torch.searchsorted(self.pair_keys, keys)
        clamped = index.clamp(max=self.pair_keys.numel() - 1)
        found = (index < self.pair_keys.numel()) & (
            self.pair_keys[clamped] == keys
        )
        counts = torch.where(
            found, self.pair_counts[clamped], torch.zeros_like(self.pair_counts[clamped])
        )
        numerator = counts + self.alpha * self.unigram[target]
        denominator = self.context_total[context] + self.alpha
        return (numerator / denominator).log()

    @torch.no_grad()
    def nll(self, chunks, *, device: torch.device, microbatch: int = 256) -> float:
        rows = _stack(chunks)
        total = 0.0
        counted = 0
        for start in range(0, rows.shape[0], microbatch):
            window = rows[start : start + microbatch].to(
                device=device, dtype=torch.long
            )
            context = window[:, :-1].reshape(-1)
            target = window[:, 1:].reshape(-1)
            losses = -self.log_prob(context, target)
            total += float(losses.sum())
            counted += losses.numel()
        if counted == 0:
            raise ValueError("ngram evaluation consumed no tokens")
        return total / counted


@torch.no_grad()
def fit_ngram_baseline(
    chunks,
    *,
    vocab_size: int,
    device: torch.device,
    alpha: float = 0.5,
    max_chunks: int = 200_000,
) -> NGramBaseline:
    """Conta o bigrama nos ultimos ``max_chunks`` do treino.

    Usar a cauda do treino e deliberado: e a fatia mais proxima do holdout,
    portanto a que produz o baseline mais forte. Um baseline forte torna o
    portao mais rigoroso, que e a direcao segura.
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if max_chunks <= 0:
        raise ValueError("max_chunks must be positive")

    rows = _stack(chunks)
    rows = rows[-max_chunks:].to(device=device, dtype=torch.long)
    flat = rows.reshape(-1)
    if flat.numel() < 2:
        raise ValueError("ngram baseline needs at least two tokens")
    if int(flat.max()) >= vocab_size or int(flat.min()) < 0:
        raise ValueError("training tokens fall outside the vocabulary")

    counts = torch.bincount(flat, minlength=vocab_size).to(torch.float64)
    unigram = (counts + 1.0) / (counts.sum() + vocab_size)

    # Pares somente dentro de cada chunk -- nao cruzar a fronteira entre chunks.
    previous = rows[:, :-1].reshape(-1)
    following = rows[:, 1:].reshape(-1)
    keys = previous * vocab_size + following
    pair_keys, pair_counts = torch.unique(keys, return_counts=True)

    context_total = torch.bincount(previous, minlength=vocab_size).to(torch.float64)

    return NGramBaseline(
        vocab_size=vocab_size,
        alpha=alpha,
        train_tokens=int(flat.numel()),
        distinct_pairs=int(pair_keys.numel()),
        unigram=unigram.to(torch.float32),
        context_total=context_total.to(torch.float32),
        pair_keys=pair_keys,
        pair_counts=pair_counts.to(torch.float32),
    )
