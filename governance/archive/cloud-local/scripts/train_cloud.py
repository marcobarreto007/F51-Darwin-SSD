#!/usr/bin/env python3
"""Compatibility wrapper for Darwin-X cloud training.

Law 0 makes Darwin-X the training path. This wrapper preserves the old
`scripts/train_cloud.py` entrypoint while delegating to `train_darwin_x.py`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from f51_darwin.artifacts import resolve_token_bin


def load_resolved_token_ids(project_root: Path, *, ram_limit_bytes: int = 2_000_000_000):
    # Cloud bundles are isolated staging workspaces, not the canonical local repo.
    # Their packaged token bin is an explicit compatibility boundary.
    token_bin = resolve_token_bin(
        project_root, min_bytes=4, allow_repo_fallback=True
    )
    if token_bin is None:
        return None, None

    size = token_bin.stat().st_size
    if size < ram_limit_bytes:
        return np.fromfile(token_bin, dtype=np.int32), token_bin
    return np.memmap(token_bin, dtype=np.int32, mode="r"), token_bin


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="F51 Darwin-X cloud training wrapper")
    parser.add_argument("--config", default=str(ROOT / "configs" / "darwin_x_600m.yaml"))
    parser.add_argument("--token-bin", default=None)
    parser.add_argument("--token-index", default=None)
    parser.add_argument("--tokenizer", default=str(ROOT / "tokenizer" / "f51_bpe"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints" / "darwin_x_cloud"))
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-small", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_darwin_x.py"),
        "--config",
        args.config,
        "--tokenizer",
        args.tokenizer,
        "--device",
        args.device,
        "--precision",
        args.precision,
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--block-size",
        str(args.block_size),
        "--lr",
        str(args.lr),
        "--save-every",
        str(args.save_every),
        "--checkpoint-dir",
        args.checkpoint_dir,
    ]
    if args.token_bin:
        command.extend(["--token-bin", args.token_bin])
    if args.token_index:
        command.extend(["--token-index", args.token_index])
    if args.resume:
        command.extend(["--resume", args.resume])
    if args.dry_run:
        command.append("--dry-run")
    if args.smoke_small:
        command.append("--smoke-small")
    return subprocess.run(command, cwd=str(ROOT), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
