#!/usr/bin/env python3
"""Run a small semantic arithmetic generation probe for F51 Darwin."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from f51_darwin.semantic_probe import (
    SemanticProbeConfig,
    default_train_tasks,
    generate_arithmetic_curriculum,
    run_semantic_probe,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a semantic arithmetic generation probe.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--context-length", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--max-answer-tokens", type=int, default=4)
    parser.add_argument("--min-verification-rate", type=float, default=1.0)
    parser.add_argument("--min-verified-generated", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--max-operand", type=int, default=12)
    parser.add_argument("--operators", default="+-*")
    parser.add_argument("--heldout-stride", type=int, default=7)
    parser.add_argument("--answer-prefix", default="")
    parser.add_argument("--verifier-rounds", type=int, default=0)
    parser.add_argument("--verifier-train-steps", type=int, default=0)
    parser.add_argument("--verifier-max-operand", type=int, default=0)
    parser.add_argument("--verifier-pool-limit", type=int, default=64)
    parser.add_argument("--verifier-operators", default="+-*")
    parser.add_argument("--use-math-organ", action="store_true")
    parser.add_argument(
        "--eval-on-train",
        action="store_true",
        help="Memorization smoke only: evaluate on training tasks instead of held-out tasks.",
    )
    args = parser.parse_args()

    if args.curriculum:
        train_tasks, eval_tasks = generate_arithmetic_curriculum(
            max_operand=args.max_operand,
            operators=args.operators,
            heldout_stride=args.heldout_stride,
            answer_prefix=args.answer_prefix,
        )
    else:
        train_tasks = default_train_tasks(args.answer_prefix)
        eval_tasks = None
    result = run_semantic_probe(
        train_tasks=train_tasks,
        eval_tasks=train_tasks if args.eval_on_train else eval_tasks,
        config=SemanticProbeConfig(
            train_steps=args.train_steps,
            lr=args.lr,
            seed=args.seed,
            context_length=args.context_length,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            max_answer_tokens=args.max_answer_tokens,
            min_verification_rate=args.min_verification_rate,
            min_verified_generated=args.min_verified_generated,
            device=args.device,
            batch_size=args.batch_size,
            curriculum=args.curriculum,
            answer_prefix=args.answer_prefix,
            verifier_rounds=args.verifier_rounds,
            verifier_train_steps=args.verifier_train_steps,
            verifier_max_operand=args.verifier_max_operand,
            verifier_pool_limit=args.verifier_pool_limit,
            verifier_operators=args.verifier_operators,
            use_math_organ=args.use_math_organ,
        ),
    )
    print(result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
