from f51_darwin.transfer.schedule import (
    default_progressive_schedule,
    parse_schedule,
    publication_gates,
    validate_schedule,
)


def test_default_schedule_converts_every_layer_progressively() -> None:
    schedule = default_progressive_schedule(6)
    assert schedule[0] == (5,)
    assert schedule[-1] == (0, 1, 2, 3, 4, 5)
    assert validate_schedule(schedule, layer_count=6) == schedule


def test_schedule_parser_rejects_regression() -> None:
    schedule = parse_schedule("2;1,2;0,1,2")
    assert schedule == ((2,), (1, 2), (0, 1, 2))
    try:
        validate_schedule(((1, 2), (0, 2)), layer_count=3)
    except ValueError as exc:
        assert "superset" in str(exc)
    else:
        raise AssertionError("regressive schedule was accepted")


def test_publication_requires_structure_and_quality() -> None:
    passing = publication_gates(
        attention_layers_remaining=0,
        teacher_holdout_nll=4.0,
        student_holdout_nll=4.8,
        ngram_holdout_nll=5.5,
    )
    assert all(passing.values())
    failing = publication_gates(
        attention_layers_remaining=0,
        teacher_holdout_nll=4.0,
        student_holdout_nll=8.0,
        ngram_holdout_nll=9.0,
    )
    assert failing["zero_attention"]
    assert not failing["quality_restored"]


def test_publication_rejects_a_student_beaten_by_counting() -> None:
    """O caso real do generation-06-final: NLL baixa num holdout degenerado.

    O aluno passa em quality_restored porque o professor nunca viu o dominio,
    mas perde para uma tabela de bigramas contada no treino.
    """
    gates = publication_gates(
        attention_layers_remaining=0,
        teacher_holdout_nll=3.1041,
        student_holdout_nll=1.9885,
        ngram_holdout_nll=1.6943,
    )
    assert gates["zero_attention"]
    assert gates["finite_holdout"]
    assert gates["quality_restored"]
    assert not gates["beats_ngram_baseline"]
    assert not all(gates.values())


def test_publication_fails_closed_without_the_ngram_control() -> None:
    gates = publication_gates(
        attention_layers_remaining=0,
        teacher_holdout_nll=4.0,
        student_holdout_nll=2.0,
    )
    assert not gates["beats_ngram_baseline"]
    gates_nan = publication_gates(
        attention_layers_remaining=0,
        teacher_holdout_nll=4.0,
        student_holdout_nll=2.0,
        ngram_holdout_nll=float("nan"),
    )
    assert not gates_nan["beats_ngram_baseline"]


def test_publication_margin_requires_a_real_win() -> None:
    tie = publication_gates(
        attention_layers_remaining=0,
        teacher_holdout_nll=4.0,
        student_holdout_nll=2.00,
        ngram_holdout_nll=2.05,
        ngram_margin=0.25,
    )
    assert not tie["beats_ngram_baseline"]
    clear = publication_gates(
        attention_layers_remaining=0,
        teacher_holdout_nll=4.0,
        student_holdout_nll=1.50,
        ngram_holdout_nll=2.05,
        ngram_margin=0.25,
    )
    assert clear["beats_ngram_baseline"]
