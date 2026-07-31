from __future__ import annotations

from collections import Counter

from research.benchmark_smol_dense_qa import (
    MCQ,
    OPEN_QA,
    accepted_answer,
    extract_choice,
    normalize_answer,
)


def test_question_matrix_is_balanced_and_identified() -> None:
    assert len(MCQ) == 24
    assert len(OPEN_QA) == 6
    assert len({row["id"] for row in (*MCQ, *OPEN_QA)}) == 30
    assert Counter(row["answer"] for row in MCQ) == {
        "A": 6,
        "B": 6,
        "C": 6,
        "D": 6,
    }
    categories = {row["category"] for row in (*MCQ, *OPEN_QA)}
    assert categories == {
        "conhecimento",
        "português",
        "matemática",
        "raciocínio",
        "programação",
        "inglês",
    }


def test_choice_extraction_is_strict() -> None:
    assert extract_choice("B") == "B"
    assert extract_choice("Resposta: c.") == "C"
    assert extract_choice("Brasília") is None
    assert extract_choice("") is None


def test_open_answer_grading_normalizes_without_fuzzy_guessing() -> None:
    assert normalize_answer("  Brasília!  ") == "brasilia"
    assert accepted_answer("A resposta é George Orwell.", ("george orwell",))
    assert accepted_answer("42.", ("42",))
    assert not accepted_answer("142", ("42",))
