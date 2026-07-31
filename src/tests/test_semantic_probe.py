from __future__ import annotations

from dataclasses import dataclass

from f51_darwin.model import GenerationOutput
from f51_darwin.semantic_probe import (
    ArithmeticTask,
    SemanticProbeConfig,
    default_train_tasks,
    generate_arithmetic_curriculum,
    run_semantic_probe,
    verify_arithmetic_generation,
)
from f51_darwin.tokenizer import F51BPETokenizer, MIN_BPE_VOCAB_SIZE


@dataclass
class _FakeArithmeticModel:
    answer: str

    def eval(self) -> None:
        return None

    def generate(self, prompt, tokenizer, **kwargs):  # noqa: ANN001, ANN003
        return GenerationOutput(text=f"{prompt}{self.answer}", token_ids=[], finish_reason="max_length")


def test_arithmetic_generation_verifier_accepts_exact_answer() -> None:
    task = ArithmeticTask("7+8", "15")
    tokenizer = F51BPETokenizer.train([task.text], vocab_size=MIN_BPE_VOCAB_SIZE)

    passed = verify_arithmetic_generation(
        _FakeArithmeticModel("15"),  # type: ignore[arg-type]
        tokenizer,
        [task],
        max_answer_tokens=4,
    )
    failed = verify_arithmetic_generation(
        _FakeArithmeticModel("16"),  # type: ignore[arg-type]
        tokenizer,
        [task],
        max_answer_tokens=4,
    )

    assert passed[0].passed
    assert passed[0].extracted_answer == "15"
    assert not failed[0].passed
    assert failed[0].extracted_answer == "16"


def test_semantic_probe_heldout_does_not_promote_without_verified_answers() -> None:
    result = run_semantic_probe(
        config=SemanticProbeConfig(
            train_steps=8,
            d_model=16,
            n_layers=4,
            n_heads=4,
            context_length=24,
            max_answer_tokens=3,
        )
    )

    assert result.train_losses
    assert result.promotion["action"] in {"quarantine", "reject"}
    assert not result.promotion["passed"]


def test_semantic_probe_math_organ_promotes_system_not_raw_model() -> None:
    result = run_semantic_probe(
        config=SemanticProbeConfig(
            train_steps=8,
            d_model=16,
            n_layers=4,
            n_heads=4,
            context_length=24,
            max_answer_tokens=3,
            use_math_organ=True,
        )
    )

    assert result.verifications
    assert any(not item["passed"] for item in result.verifications)
    assert all(item["tool_passed"] for item in result.verifications)
    assert not result.promotion["passed"]
    assert result.system_promotion["passed"]
    assert result.system_promotion["action"] == "promote"


def test_arithmetic_curriculum_has_disjoint_heldout_split() -> None:
    train_tasks, eval_tasks = generate_arithmetic_curriculum(
        max_operand=4,
        operators="+*",
        heldout_stride=5,
        answer_prefix="ANS:",
    )
    train_texts = {task.text for task in train_tasks}
    eval_texts = {task.text for task in eval_tasks}

    assert train_texts
    assert eval_texts
    assert train_texts.isdisjoint(eval_texts)
    assert all("=" in text for text in train_texts | eval_texts)
    assert all("ANS:" in text for text in train_texts | eval_texts)


def test_semantic_probe_curriculum_keeps_gate_honest_on_heldout() -> None:
    train_tasks, eval_tasks = generate_arithmetic_curriculum(max_operand=4, operators="+", heldout_stride=4)
    result = run_semantic_probe(
        train_tasks=train_tasks,
        eval_tasks=eval_tasks,
        config=SemanticProbeConfig(
            train_steps=12,
            d_model=16,
            n_layers=4,
            n_heads=4,
            context_length=24,
            max_answer_tokens=3,
            batch_size=4,
            curriculum=True,
        ),
    )

    assert result.train_losses
    assert result.promotion["action"] in {"quarantine", "reject"}
    assert not result.promotion["passed"]


def test_semantic_probe_verifier_feedback_keeps_eval_disjoint() -> None:
    train_tasks, eval_tasks = generate_arithmetic_curriculum(max_operand=4, operators="+", heldout_stride=4)
    result = run_semantic_probe(
        train_tasks=train_tasks,
        eval_tasks=eval_tasks,
        config=SemanticProbeConfig(
            train_steps=8,
            d_model=16,
            n_layers=4,
            n_heads=4,
            context_length=24,
            max_answer_tokens=3,
            batch_size=4,
            curriculum=True,
            verifier_rounds=1,
            verifier_train_steps=2,
            verifier_max_operand=5,
            verifier_pool_limit=8,
            verifier_operators="+",
        ),
    )

    eval_texts = set(result.eval_tasks)
    repair_texts = set(result.feedback_rounds[0]["repair_tasks"])
    assert result.feedback_rounds
    assert repair_texts
    assert repair_texts.isdisjoint(eval_texts)
    assert not result.promotion["passed"]


def test_semantic_probe_can_promote_memorized_exact_answers() -> None:
    train_tasks = default_train_tasks()[:2]
    result = run_semantic_probe(
        train_tasks=train_tasks,
        eval_tasks=train_tasks,
        config=SemanticProbeConfig(
            train_steps=180,
            lr=5e-3,
            d_model=32,
            n_layers=4,
            n_heads=4,
            context_length=24,
            max_answer_tokens=3,
        ),
    )

    assert result.verifications
    assert all(item["passed"] for item in result.verifications)
    assert result.promotion["passed"]
    assert result.promotion["action"] == "promote"
