#!/usr/bin/env python3
"""Supported CLI adapter for the local Davi service."""

from __future__ import annotations

import argparse
from typing import Sequence

from f51_darwin.serving.runtime import *  # noqa: F403 - compatibility facade
from f51_darwin.serving.runtime import main as _runtime_main
from f51_darwin.organism.runtime import DarwinOrganism, DarwinOrganismConfig


CANONICAL_CONFIG = "src/configs/darwin_x_100m.yaml"


def main(argv: Sequence[str] | None = None) -> None:
    """Run Davi with the supported 100M configuration by default."""
    forwarded = list(argv) if argv is not None else None
    defaults = argparse.ArgumentParser(add_help=False)
    defaults.add_argument(
        "--config", default="src/configs/darwin_x_100m.yaml"
    )
    selected, _ = defaults.parse_known_args(forwarded)
    _runtime_main(
        forwarded,
        default_config=selected.config,
        organism_types=(DarwinOrganism, DarwinOrganismConfig),
    )


if __name__ == "__main__":
    main()
