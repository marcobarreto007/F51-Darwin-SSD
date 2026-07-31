#!/usr/bin/env python3
"""A100 Darwin-X launcher.

The former organism/A100 launcher depended on removed legacy MoE YAML files.
Law 0 routes A100 work through Darwin-X and verified token indexes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch Darwin-X on an A100-class GPU")
    parser.add_argument("--config", default=str(ROOT / "configs" / "darwin_x_4b.yaml"))
    parser.add_argument("--token-index", default=None)
    parser.add_argument("--token-bin", default=None)
    parser.add_argument("--tokenizer", default=str(ROOT / "tokenizer" / "f51_bpe"))
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.token_index and not args.token_bin:
        print(
            json.dumps(
                {
                    "status": "missing_token_source",
                    "message": "Pass --token-index for verified corpus-factory training or --token-bin for a single bin.",
                    "example": [
                        "python",
                        "scripts/run_organism_a100.py",
                        "--token-index",
                        "/workspace/f51_corpus_factory/tokens/80k/index.json",
                        "--dry-run",
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_darwin_x.py"),
        "--config",
        args.config,
        "--tokenizer",
        args.tokenizer,
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--block-size",
        str(args.block_size),
        "--device",
        args.device,
    ]
    if args.token_index:
        command.extend(["--token-index", args.token_index])
    if args.token_bin:
        command.extend(["--token-bin", args.token_bin])
    if args.dry_run:
        command.append("--dry-run")
    return subprocess.run(command, cwd=str(ROOT), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
