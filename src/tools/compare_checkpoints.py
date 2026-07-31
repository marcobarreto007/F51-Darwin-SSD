#!/usr/bin/env python3
"""Compare two F51 checkpoints with held-out/replay loss and promotion gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from f51_darwin.checkpoint_eval import CheckpointEvalConfig, compare_checkpoints


def write_result(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare baseline/candidate F51 checkpoints.")
    parser.add_argument("--baseline", required=True, help="Baseline checkpoint path.")
    parser.add_argument("--candidate", required=True, help="Candidate checkpoint path.")
    parser.add_argument("--tokens", default=None, help="Optional int32 token bin. Defaults to artifact resolver.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=200_000)
    parser.add_argument("--verified-generated", type=int, default=0)
    parser.add_argument("--generated-total", type=int, default=0)
    parser.add_argument("--output", default=None, help="Optional atomic JSON output path.")
    args = parser.parse_args()

    result = compare_checkpoints(
        args.baseline,
        args.candidate,
        project_root=ROOT,
        token_bin=args.tokens,
        config=CheckpointEvalConfig(
            block_size=args.block_size,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            max_tokens=args.max_tokens,
            device=args.device,
        ),
        verified_generated=args.verified_generated,
        generated_total=args.generated_total,
    )
    payload = result.to_json()
    if args.output:
        write_result(Path(args.output), payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
