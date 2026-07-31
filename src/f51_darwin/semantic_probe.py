from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import torch

from f51_darwin.config import DarwinConfig
from f51_darwin.evolution_gate import EvolutionGateMetrics, EvolutionGateThresholds, decide_promotion
from f51_darwin.math_organ import LocalArithmeticOrgan
from f51_darwin.model import F51DarwinModel
from f51_darwin.tokenizer import F51BPETokenizer, MIN_BPE_VOCAB_SIZE


@dataclass(frozen=True)
class ArithmeticTask:
    expression: str
    expected: str
    answer_prefix: str = ""

    @property
    def prompt(self) -> str:
        return f"{self.expression}={self.answer_prefix}"

    @property
    def text(self) -> str:
        return f"{self.expression}={self.answer_prefix}{self.expected}"

    @property
    def difficulty(self) -> int:
        nums = [int(item) for item in re.findall(r"\d+", self.expression)]
        return max(nums) if nums else 0


@dataclass(frozen=True)
class SemanticProbeConfig:
    train_steps: int = 120
    lr: float = 3e-3
    seed: int = 51
    context_length: int = 24
    d_model: int = 32
    n_layers: int = 4
    n_heads: int = 4
    max_answer_tokens: int = 4
    min_verification_rate: float = 1.0
    min_verified_generated: int = 1
    device: str = "cpu"
    batch_size: int = 16
    curriculum: bool = False
    answer_prefix: str = ""
    verifier_rounds: int = 0
    verifier_train_steps: int = 0
    verifier_max_operand: int = 0
    verifier_pool_limit: int = 64
    verifier_operators: str = "+-*"
    use_math_organ: bool = False


@dataclass(frozen=True)
class SemanticVerification:
    expression: str
    expected: str
    generated_text: str
    extracted_answer: str | None
    passed: bool
    tool_answer: str | None = None
    tool_passed: bool = False


@dataclass(frozen=True)
class SemanticProbeResult:
    train_tasks: list[str]
    eval_tasks: list[str]
    loss_before: float
    loss_after: float
    replay_loss_before: float
    replay_loss_after: float
    train_losses: list[float] = field(default_factory=list)
    feedback_rounds: list[dict[str, Any]] = field(default_factory=list)
    verifications: list[dict[str, Any]] = field(default_factory=list)
    promotion: dict[str, Any] = field(default_factory=dict)
    system_promotion: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def default_train_tasks(answer_prefix: str = "") -> list[ArithmeticTask]:
    return [
        ArithmeticTask("2+2", "4", answer_prefix=answer_prefix),
        ArithmeticTask("3+5", "8", answer_prefix=answer_prefix),
        ArithmeticTask("4+6", "10", answer_prefix=answer_prefix),
        ArithmeticTask("5*7", "35", answer_prefix=answer_prefix),
        ArithmeticTask("6*6", "36", answer_prefix=answer_prefix),
        ArithmeticTask("12-5", "7", answer_prefix=answer_prefix),
    ]


def default_eval_tasks(answer_prefix: str = "") -> list[ArithmeticTask]:
    return [
        ArithmeticTask("7+8", "15", answer_prefix=answer_prefix),
        ArithmeticTask("9+4", "13", answer_prefix=answer_prefix),
        ArithmeticTask("8*3", "24", answer_prefix=answer_prefix),
        ArithmeticTask("14-6", "8", answer_prefix=answer_prefix),
    ]


def generate_arithmetic_tasks(
    *,
    max_operand: int = 12,
    operators: str = "+-*",
    answer_prefix: str = "",
) -> list[ArithmeticTask]:
    tasks: list[ArithmeticTask] = []
    for op in operators:
        for left in range(max_operand + 1):
            for right in range(max_operand + 1):
                if op == "+":
                    expected = left + right
                elif op == "-":
                    expected = left - right
                elif op == "*":
                    expected = left * right
                else:
                    raise ValueError(f"unsupported operator: {op}")
                if expected < 0:
                    continue
                tasks.append(ArithmeticTask(f"{left}{op}{right}", str(expected), answer_prefix=answer_prefix))
    return sorted(tasks, key=lambda item: (item.difficulty, item.expression))


def generate_arithmetic_curriculum(
    *,
    max_operand: int = 12,
    operators: str = "+-*",
    heldout_stride: int = 7,
    answer_prefix: str = "",
) -> tuple[list[ArithmeticTask], list[ArithmeticTask]]:
    tasks = generate_arithmetic_tasks(
        max_operand=max_operand,
        operators=operators,
        answer_prefix=answer_prefix,
    )
    train: list[ArithmeticTask] = []
    heldout: list[ArithmeticTask] = []
    stride = max(2, heldout_stride)
    for index, task in enumerate(tasks):
        if index % stride == 0:
            heldout.append(task)
        else:
            train.append(task)
    return train, heldout


def _make_batch(tokenizer: F51BPETokenizer, tasks: Iterable[ArithmeticTask], context_length: int) -> torch.Tensor:
    rows: list[list[int]] = []
    for task in tasks:
        ids = tokenizer.encode(task.text, add_bos=True, add_eos=True)[:context_length]
        ids = ids + [tokenizer.pad_id] * (context_length - len(ids))
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long)


def _make_task_batches(
    tokenizer: F51BPETokenizer,
    tasks: list[ArithmeticTask],
    *,
    context_length: int,
    batch_size: int,
) -> list[torch.Tensor]:
    if not tasks:
        raise ValueError("tasks must not be empty")
    size = max(1, batch_size)
    return [
        _make_batch(tokenizer, tasks[index : index + size], context_length)
        for index in range(0, len(tasks), size)
    ]


def _eval_loss(model: F51DarwinModel, batch: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        output = model(batch, labels=batch)
        if output.loss is None:
            raise RuntimeError("model did not return a loss")
        return float(output.loss.detach().cpu())


def _train_batches(
    model: F51DarwinModel,
    optimizer: torch.optim.Optimizer,
    batches: list[torch.Tensor],
    *,
    steps: int,
    curriculum: bool,
    device: torch.device,
) -> list[float]:
    losses: list[float] = []
    if steps <= 0:
        return losses
    model.train()
    for step in range(steps):
        if curriculum:
            unlocked = min(len(batches), max(1, 1 + step * len(batches) // max(steps, 1)))
            batch = batches[step % unlocked].to(device)
        else:
            batch = batches[step % len(batches)].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch, labels=batch)
        if output.loss is None:
            raise RuntimeError("model did not return a loss")
        loss = output.loss
        losses.append(float(loss.detach().cpu()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return losses


def _extract_numeric_answer(text: str, prompt: str) -> str | None:
    continuation = text[len(prompt) :] if text.startswith(prompt) else text
    match = re.search(r"-?\d+", continuation)
    return match.group(0) if match else None


def verify_arithmetic_generation(
    model: F51DarwinModel,
    tokenizer: F51BPETokenizer,
    tasks: Iterable[ArithmeticTask],
    *,
    max_answer_tokens: int,
    use_math_organ: bool = False,
) -> list[SemanticVerification]:
    model.eval()
    results: list[SemanticVerification] = []
    math_organ = LocalArithmeticOrgan() if use_math_organ else None
    for task in tasks:
        output = model.generate(
            task.prompt,
            tokenizer,
            max_tokens=max_answer_tokens,
            temperature=0.0,
            top_p=1.0,
            use_cache=False,
        )
        answer = _extract_numeric_answer(output.text, task.prompt)
        tool_result = math_organ.solve(task.expression) if math_organ is not None else None
        tool_answer = tool_result.answer if tool_result is not None and tool_result.success else None
        results.append(
            SemanticVerification(
                expression=task.expression,
                expected=task.expected,
                generated_text=output.text,
                extracted_answer=answer,
                passed=answer == task.expected,
                tool_answer=tool_answer,
                tool_passed=tool_answer == task.expected if tool_answer is not None else False,
            )
        )
    return results


def _feedback_pool(
    train_tasks: list[ArithmeticTask],
    eval_tasks: list[ArithmeticTask],
    config: SemanticProbeConfig,
) -> list[ArithmeticTask]:
    if config.verifier_rounds <= 0:
        return []
    max_operand = config.verifier_max_operand
    if max_operand <= 0:
        max_operand = max((task.difficulty for task in train_tasks + eval_tasks), default=12)
    seen = {task.text for task in train_tasks + eval_tasks}
    pool = [
        task
        for task in generate_arithmetic_tasks(
            max_operand=max_operand,
            operators=config.verifier_operators,
            answer_prefix=config.answer_prefix,
        )
        if task.text not in seen
    ]
    return pool[: max(0, config.verifier_pool_limit)]


def run_semantic_probe(
    *,
    train_tasks: list[ArithmeticTask] | None = None,
    eval_tasks: list[ArithmeticTask] | None = None,
    config: SemanticProbeConfig | None = None,
) -> SemanticProbeResult:
    config = config or SemanticProbeConfig()
    train_tasks = train_tasks or default_train_tasks(config.answer_prefix)
    eval_tasks = eval_tasks or default_eval_tasks(config.answer_prefix)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    if config.curriculum:
        train_tasks = sorted(train_tasks, key=lambda item: (item.difficulty, item.expression))

    tokenizer = F51BPETokenizer.train(
        [task.text for task in train_tasks + eval_tasks],
        vocab_size=MIN_BPE_VOCAB_SIZE,
        name="F51-Semantic-Probe-BPE",
    )
    model_config = DarwinConfig(
        model_name="F51-Darwin-Semantic-Probe",
        tokenizer="f51_semantic_probe_bpe",
        vocab_size=tokenizer.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
    )
    baseline = F51DarwinModel(model_config).to(device)
    candidate = F51DarwinModel(model_config).to(device)
    candidate.load_state_dict(baseline.state_dict())

    train_batch = _make_batch(tokenizer, train_tasks, config.context_length).to(device)
    eval_batch = _make_batch(tokenizer, eval_tasks, config.context_length).to(device)
    loss_before = _eval_loss(baseline, eval_batch)
    replay_loss_before = _eval_loss(baseline, train_batch)

    optimizer = torch.optim.AdamW(candidate.parameters(), lr=config.lr)
    train_batches = _make_task_batches(
        tokenizer,
        train_tasks,
        context_length=config.context_length,
        batch_size=config.batch_size,
    )
    train_losses = _train_batches(
        candidate,
        optimizer,
        train_batches,
        steps=config.train_steps,
        curriculum=config.curriculum,
        device=device,
    )

    feedback_rounds: list[dict[str, Any]] = []
    pool = _feedback_pool(train_tasks, eval_tasks, config)
    for round_index in range(config.verifier_rounds):
        verifications = verify_arithmetic_generation(
            candidate,
            tokenizer,
            pool,
            max_answer_tokens=config.max_answer_tokens,
            use_math_organ=config.use_math_organ,
        )
        accepted = [item for item in verifications if item.passed]
        rejected = [item for item in verifications if not item.passed]
        repair_tasks = [
            task
            for task, result in zip(pool, verifications)
            if not result.passed
        ]
        if repair_tasks and config.verifier_train_steps > 0:
            repair_batches = _make_task_batches(
                tokenizer,
                repair_tasks,
                context_length=config.context_length,
                batch_size=config.batch_size,
            )
            train_losses.extend(
                _train_batches(
                    candidate,
                    optimizer,
                    repair_batches,
                    steps=config.verifier_train_steps,
                    curriculum=True,
                    device=device,
                )
            )
        feedback_rounds.append(
            {
                "round": round_index + 1,
                "pool_size": len(pool),
                "accepted": len(accepted),
                "rejected": len(rejected),
                "repair_tasks": [task.text for task in repair_tasks],
                "scars": [asdict(item) for item in rejected],
            }
        )

    loss_after = _eval_loss(candidate, eval_batch)
    replay_loss_after = _eval_loss(candidate, train_batch)
    verifications = verify_arithmetic_generation(
        candidate,
        tokenizer,
        eval_tasks,
        max_answer_tokens=config.max_answer_tokens,
        use_math_organ=config.use_math_organ,
    )
    verified = sum(1 for item in verifications if item.passed)
    gate = decide_promotion(
        EvolutionGateMetrics(
            baseline_loss=loss_before,
            candidate_loss=loss_after,
            baseline_replay_loss=replay_loss_before,
            candidate_replay_loss=replay_loss_after,
            generated_total=len(verifications),
            verified_generated=verified,
            train_steps=config.train_steps,
            tokens_seen=int(train_batch.numel() * config.train_steps),
        ),
        EvolutionGateThresholds(
            min_verified_generated=config.min_verified_generated,
            min_verification_rate=config.min_verification_rate,
        ),
    )
    system_verified = sum(1 for item in verifications if item.passed or item.tool_passed)
    system_gate = decide_promotion(
        EvolutionGateMetrics(
            baseline_loss=loss_before,
            candidate_loss=loss_after,
            baseline_replay_loss=replay_loss_before,
            candidate_replay_loss=replay_loss_after,
            generated_total=len(verifications),
            verified_generated=system_verified,
            train_steps=config.train_steps,
            tokens_seen=int(train_batch.numel() * config.train_steps),
        ),
        EvolutionGateThresholds(
            min_verified_generated=config.min_verified_generated,
            min_verification_rate=config.min_verification_rate,
        ),
    )
    return SemanticProbeResult(
        train_tasks=[task.text for task in train_tasks],
        eval_tasks=[task.text for task in eval_tasks],
        loss_before=loss_before,
        loss_after=loss_after,
        replay_loss_before=replay_loss_before,
        replay_loss_after=replay_loss_after,
        train_losses=train_losses,
        feedback_rounds=feedback_rounds,
        verifications=[asdict(item) for item in verifications],
        promotion=asdict(gate),
        system_promotion=asdict(system_gate),
    )
