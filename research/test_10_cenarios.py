#!/usr/bin/env python3
"""Run the ten real Topology Manifest v7 regression scenarios."""
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(
            [
                str(ROOT / "src" / "tests" / "test_topology_manifest_v7.py"),
                "-vv",
                "-p",
                "no:cacheprovider",
                *sys.argv[1:],
            ]
        )
    )
