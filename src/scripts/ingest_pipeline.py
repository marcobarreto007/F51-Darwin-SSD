#!/usr/bin/env python3
"""Supported CLI adapter for quarantine-first local ingestion."""

from __future__ import annotations

from typing import Sequence

from f51_darwin.ingestion.runtime import *  # noqa: F403 - compatibility facade
from f51_darwin.ingestion.runtime import main as _runtime_main


def main(argv: Sequence[str] | None = None) -> int:
    return _runtime_main(list(argv) if argv is not None else None)


if __name__ == "__main__":
    raise SystemExit(main())
