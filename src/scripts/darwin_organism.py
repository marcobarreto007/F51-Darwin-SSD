#!/usr/bin/env python3
"""Supported CLI adapter for the Darwin-X organism runtime."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from f51_darwin.organism.runtime import *  # noqa: F403 - compatibility facade
from f51_darwin.organism.runtime import main as _runtime_main
from scripts.darwin_inventory import main as _inventory_main


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_CONFIG = ROOT / "src" / "configs" / "darwin_x_100m.yaml"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the organism while keeping the canonical config at the boundary."""
    forwarded = list(argv) if argv is not None else None
    defaults = argparse.ArgumentParser(add_help=False)
    defaults.add_argument(
        "--config", default="src/configs/darwin_x_100m.yaml"
    )
    selected, _ = defaults.parse_known_args(forwarded)
    return _runtime_main(
        forwarded,
        default_config=selected.config,
        inventory_main=_inventory_main,
    )


if __name__ == "__main__":
    raise SystemExit(main())
