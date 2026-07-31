from __future__ import annotations

import math


def default_progressive_schedule(layer_count: int) -> tuple[tuple[int, ...], ...]:
    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    return tuple(tuple(range(start, layer_count)) for start in range(layer_count - 1, -1, -1))


def parse_schedule(value: str) -> tuple[tuple[int, ...], ...]:
    try:
        schedule = tuple(
            tuple(sorted({int(index) for index in group.split(",")}))
            for group in value.split(";")
            if group.strip()
        )
    except ValueError as exc:
        raise ValueError(f"invalid progressive schedule: {value}") from exc
    if not schedule or any(not group for group in schedule):
        raise ValueError("progressive schedule cannot be empty")
    return schedule


def validate_schedule(
    schedule: tuple[tuple[int, ...], ...],
    *,
    layer_count: int,
) -> tuple[tuple[int, ...], ...]:
    previous: set[int] = set()
    for generation, group in enumerate(schedule, start=1):
        current = set(group)
        if any(index < 0 or index >= layer_count for index in current):
            raise ValueError(f"generation {generation} contains an invalid layer")
        if not current.issuperset(previous):
            raise ValueError(f"generation {generation} must be a superset")
        previous = current
    if previous != set(range(layer_count)):
        raise ValueError("final generation must replace every attention layer")
    return schedule


def publication_gates(
    *,
    attention_layers_remaining: int,
    teacher_holdout_nll: float,
    student_holdout_nll: float,
    ngram_holdout_nll: float | None = None,
    tolerance: float = 1.0,
    ngram_margin: float = 0.0,
) -> dict[str, bool]:
    """Portoes de publicacao do Darwin Transfer.

    ``beats_ngram_baseline`` existe porque ``quality_restored`` compara o
    aluno com um professor que pode nunca ter visto o dominio do holdout.
    Num holdout degenerado essa comparacao passa mesmo com o aluno destruido.
    A tabela de bigramas contada no treino nao tem esse ponto cego: se o
    aluno nao a supera, o numero nao mede capacidade.

    ``ngram_holdout_nll=None`` reprova o portao de proposito -- omitir o
    controle nunca deve publicar.
    """
    return {
        "zero_attention": attention_layers_remaining == 0,
        "finite_holdout": math.isfinite(student_holdout_nll),
        "quality_restored": student_holdout_nll <= teacher_holdout_nll + tolerance,
        "beats_ngram_baseline": (
            ngram_holdout_nll is not None
            and math.isfinite(ngram_holdout_nll)
            and student_holdout_nll <= ngram_holdout_nll - ngram_margin
        ),
    }
