#!/usr/bin/env python3
"""Compile every active tracked Python source and emit one stable JSON summary."""

from __future__ import annotations

import argparse
import json
import py_compile
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def evaluate(root: Path) -> dict[str, int]:
    raw = subprocess.check_output(["git", "ls-files", "-z", "--", "*.py"], cwd=root)
    files = [item.decode("utf-8", "surrogateescape") for item in raw.split(b"\0") if item]
    active = [path for path in files if not path.replace("\\", "/").startswith("governance/archive/")]
    failed = 0
    for relative in active:
        try:
            py_compile.compile(str(root / relative), doraise=True)
        except (OSError, py_compile.PyCompileError):
            failed += 1
    return {
        "tracked": len(files),
        "attempted": len(active),
        "compiled": len(active) - failed,
        "skipped_archive": len(files) - len(active),
        "failed": failed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    summary = evaluate(args.root.resolve())
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
