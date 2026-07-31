#!/usr/bin/env python3
"""F51 Darwin evolution harness.

This script is deliberately conservative: verified fixed tasks prove the
verifiers, not evolution. Evolution is only reported when model-measured
signals improve and model-generated candidates survive verification.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]

from f51_darwin.config import DarwinConfig
from f51_darwin.evolution_gate import EvolutionGateMetrics, decide_promotion
from f51_darwin.model import F51DarwinModel
from f51_darwin.tokenizer import F51BPETokenizer, MIN_BPE_VOCAB_SIZE


@dataclass(frozen=True)
class Candidate:
    task_id: str
    source: str
    mode: str
    expression: str | None = None
    expected: str | float | None = None
    code: str | None = None
    invocation: str | None = None


@dataclass
class Verification:
    task_id: str
    source: str
    verifier: str
    passed: bool
    expected: str | float | None = None
    got: str | float | None = None
    error: str | None = None


@dataclass
class ModelProbe:
    enabled: bool
    train_steps: int = 0
    loss_before: float | None = None
    loss_after: float | None = None
    loss_delta: float | None = None
    replay_loss_before: float | None = None
    replay_loss_after: float | None = None
    learning_smoke_passed: bool = False
    generated: list[dict[str, Any]] = field(default_factory=list)
    verified_generated: int = 0
    promotion_gate: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvoRound:
    rid: int
    ts: str = ""
    generated: int = 0
    verified: int = 0
    accepted: int = 0
    rejected: int = 0
    lessons: list[str] = field(default_factory=list)
    scars: list[str] = field(default_factory=list)
    verifications: list[Verification] = field(default_factory=list)
    model_probe: ModelProbe = field(default_factory=lambda: ModelProbe(enabled=False))

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / max(self.verified, 1)

    @property
    def model_generated_knowledge_smoke(self) -> bool:
        return (
            self.model_probe.enabled
            and self.model_probe.learning_smoke_passed
            and self.model_probe.verified_generated > 0
        )


class MathVerifier:
    def verify(self, candidate: Candidate) -> Verification:
        if candidate.expression is None:
            return Verification(candidate.task_id, candidate.source, "math", False, error="missing expression")
        try:
            result = eval(  # noqa: S307 - restricted namespace for arithmetic smoke tasks.
                candidate.expression,
                {"__builtins__": {}},
                {"abs": abs, "pow": pow, "sqrt": math.sqrt, "pi": math.pi},
            )
            got = float(result)
            expected = float(candidate.expected) if candidate.expected is not None else None
            passed = expected is not None and abs(got - expected) < 1e-6
            return Verification(candidate.task_id, candidate.source, "math", passed, expected, got)
        except Exception as exc:  # pragma: no cover - defensive report path
            return Verification(candidate.task_id, candidate.source, "math", False, candidate.expected, error=str(exc))


class CodeVerifier:
    def verify(self, candidate: Candidate) -> Verification:
        if candidate.code is None or candidate.invocation is None:
            return Verification(candidate.task_id, candidate.source, "code", False, error="missing code")
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as handle:
                handle.write(candidate.code)
                handle.write(f"\nprint({candidate.invocation})\n")
                tmp_path = Path(handle.name)
            result = subprocess.run(
                [sys.executable, str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            got = result.stdout.strip()
            expected = "" if candidate.expected is None else str(candidate.expected).strip()
            error = result.stderr.strip() or None
            return Verification(candidate.task_id, candidate.source, "code", got == expected, expected, got, error)
        except Exception as exc:  # pragma: no cover - defensive report path
            return Verification(candidate.task_id, candidate.source, "code", False, candidate.expected, error=str(exc))
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)


def fixed_verifier_candidates() -> list[Candidate]:
    return [
        Candidate("math_2_plus_2", "fixed_verifier_smoke", "math", expression="2+2", expected=4.0),
        Candidate("math_5_times_7", "fixed_verifier_smoke", "math", expression="5*7", expected=35.0),
        Candidate("math_sqrt_144", "fixed_verifier_smoke", "math", expression="sqrt(144)", expected=12.0),
        Candidate(
            "code_return_42",
            "fixed_verifier_smoke",
            "code",
            code="def f():\n    return 42",
            invocation="f()",
            expected="42",
        ),
    ]


def _toy_texts() -> list[str]:
    return ["2+2=4", "5*7=35", "sqrt(144)=12", "def f(): return 42"]


def _make_tiny_model(seed: int) -> tuple[F51DarwinModel, F51BPETokenizer]:
    torch.manual_seed(seed)
    tokenizer = F51BPETokenizer.train(_toy_texts(), vocab_size=MIN_BPE_VOCAB_SIZE, name="F51-Evo-Smoke-BPE")
    config = DarwinConfig(
        model_name="F51-Darwin-Evo-Smoke",
        tokenizer="f51_evo_smoke_bpe",
        vocab_size=tokenizer.vocab_size,
        context_length=24,
        d_model=32,
        n_layers=4,
        n_heads=4,
        dropout=0.0,
    )
    return F51DarwinModel(config), tokenizer


def _batch_from_texts(tokenizer: F51BPETokenizer, texts: list[str], context_length: int) -> torch.Tensor:
    rows: list[list[int]] = []
    for text in texts:
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)[:context_length]
        if len(ids) < 2:
            ids.append(tokenizer.eos_id)
        ids = ids + [tokenizer.pad_id] * (context_length - len(ids))
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long)


def _eval_loss(model: F51DarwinModel, batch: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        output = model(batch, labels=batch)
        if output.loss is None:
            raise RuntimeError("model did not return a loss")
        return float(output.loss.detach().cpu())


def _train_toy(model: F51DarwinModel, batch: torch.Tensor, steps: int) -> None:
    if steps <= 0:
        return
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch, labels=batch)
        if output.loss is None:
            raise RuntimeError("model did not return a loss")
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def run_model_probe(*, seed: int, train_steps: int, max_tokens: int) -> ModelProbe:
    model, tokenizer = _make_tiny_model(seed)
    batch = _batch_from_texts(tokenizer, _toy_texts(), model.config.context_length)
    replay_batch = _batch_from_texts(tokenizer, _toy_texts()[:2], model.config.context_length)
    loss_before = _eval_loss(model, batch)
    replay_loss_before = _eval_loss(model, replay_batch)
    _train_toy(model, batch, train_steps)
    loss_after = _eval_loss(model, batch)
    replay_loss_after = _eval_loss(model, replay_batch)

    generated: list[dict[str, Any]] = []
    verified_generated = 0
    prompts = [("2+2=", "4"), ("5*7=", "35")]
    for prompt, expected in prompts:
        out = model.generate(
            prompt,
            tokenizer,
            max_tokens=max(max_tokens, len(expected)),
            temperature=0.0,
            top_p=1.0,
            use_cache=False,
        )
        continuation = out.text[len(prompt) :].strip() if out.text.startswith(prompt) else out.text.strip()
        passed = continuation == expected
        verified_generated += int(passed)
        generated.append(
            {
                "prompt": prompt,
                "expected_prefix": expected,
                "text": out.text,
                "continuation": continuation,
                "passed": passed,
            }
        )

    loss_delta = loss_before - loss_after
    gate = decide_promotion(
        EvolutionGateMetrics(
            baseline_loss=loss_before,
            candidate_loss=loss_after,
            baseline_replay_loss=replay_loss_before,
            candidate_replay_loss=replay_loss_after,
            generated_total=len(prompts),
            verified_generated=verified_generated,
            train_steps=train_steps,
            tokens_seen=int(batch.numel() * max(train_steps, 0)),
        )
    )
    return ModelProbe(
        enabled=True,
        train_steps=train_steps,
        loss_before=loss_before,
        loss_after=loss_after,
        loss_delta=loss_delta,
        replay_loss_before=replay_loss_before,
        replay_loss_after=replay_loss_after,
        learning_smoke_passed=loss_delta > 0.0,
        generated=generated,
        verified_generated=verified_generated,
        promotion_gate=asdict(gate),
    )


class EvoEngine:
    def __init__(self, *, model_probe: bool = False, train_steps: int = 0, seed: int = 51) -> None:
        self.math = MathVerifier()
        self.code = CodeVerifier()
        self.model_probe = model_probe
        self.train_steps = train_steps
        self.seed = seed
        self.history: list[EvoRound] = []

    def run(self, rid: int) -> EvoRound:
        round_report = EvoRound(rid=rid, ts=datetime.now(timezone.utc).isoformat())
        candidates = fixed_verifier_candidates()
        round_report.generated = len(candidates)
        for candidate in candidates:
            round_report.verified += 1
            result = self.math.verify(candidate) if candidate.mode == "math" else self.code.verify(candidate)
            round_report.verifications.append(result)
            if result.passed:
                round_report.accepted += 1
                round_report.lessons.append(candidate.task_id)
            else:
                round_report.rejected += 1
                round_report.scars.append(candidate.task_id)

        if self.model_probe:
            round_report.model_probe = run_model_probe(
                seed=self.seed + rid - 1,
                train_steps=self.train_steps,
                max_tokens=4,
            )
        self.history.append(round_report)
        return round_report

    def proof_status(self) -> str:
        if any(round_report.model_probe.promotion_gate.get("passed") for round_report in self.history):
            return "PROMOTION_GATE_PASSED"
        if any(round_report.model_generated_knowledge_smoke for round_report in self.history):
            return "MODEL_GENERATION_VERIFIED_SMOKE"
        if any(round_report.model_probe.learning_smoke_passed for round_report in self.history):
            return "MODEL_LEARNING_SMOKE_ONLY"
        return "VERIFIER_SMOKE_ONLY"

    def report(self) -> str:
        lines = ["", "=" * 64, "  F51 DARWIN - EVOLUTION HARNESS", "=" * 64]
        for round_report in self.history:
            lines.append(
                "  R{rid}: {ok}/{ver} verifier ok | model_probe={probe} | status={status}".format(
                    rid=round_report.rid,
                    ok=round_report.accepted,
                    ver=round_report.verified,
                    probe=round_report.model_probe.enabled,
                    status=(
                        "self-knowledge"
                        if round_report.model_generated_knowledge_smoke
                        else "learning-smoke"
                        if round_report.model_probe.learning_smoke_passed
                        else "verifier-smoke"
                    ),
                )
            )
            if round_report.model_probe.enabled:
                lines.append(
                    "      loss_before={:.4f} loss_after={:.4f} replay_after={:.4f} generated_verified={} gate={}".format(
                        round_report.model_probe.loss_before or 0.0,
                        round_report.model_probe.loss_after or 0.0,
                        round_report.model_probe.replay_loss_after or 0.0,
                        round_report.model_probe.verified_generated,
                        round_report.model_probe.promotion_gate.get("action", "n/a"),
                    )
                )
        lines.append(f"  PROOF_STATUS: {self.proof_status()}")
        lines.append("=" * 64)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proof_status": self.proof_status(),
            "rounds": [
                {
                    **asdict(round_report),
                    "acceptance_rate": round_report.acceptance_rate,
                    "model_generated_knowledge_smoke": round_report.model_generated_knowledge_smoke,
                }
                for round_report in self.history
            ],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the F51 Darwin evolution harness.")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    parser.add_argument("--model-probe", action="store_true", help="Run an in-memory F51 model learning/generation probe.")
    parser.add_argument("--train-steps", type=int, default=0, help="Tiny in-memory train steps for --model-probe.")
    parser.add_argument("--seed", type=int, default=51)
    args = parser.parse_args()

    engine = EvoEngine(model_probe=args.model_probe, train_steps=args.train_steps, seed=args.seed)
    for index in range(args.rounds):
        engine.run(index + 1)
    print(engine.report())
    if args.output:
        Path(args.output).write_text(json.dumps(engine.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
