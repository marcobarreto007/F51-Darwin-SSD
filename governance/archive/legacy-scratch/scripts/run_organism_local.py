#!/usr/bin/env python3
"""Local Darwin-X launcher.

The former Ghost organism launcher depended on removed legacy YAML files.
Law 0 routes local training through Darwin-X.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    command = [
        "python",
        "scripts/train_darwin_x.py",
        "--config",
        "configs/darwin_x_600m.yaml",
        "--token-bin",
        "data/tokens_local.bin",
        "--tokenizer",
        "tokenizer/f51_bpe",
        "--device",
        "cuda",
    ]
    print(
        json.dumps(
            {
                "status": "retired_legacy_launcher",
                "replacement": "Darwin-X local training",
                "cwd": str(ROOT),
                "command": command,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
