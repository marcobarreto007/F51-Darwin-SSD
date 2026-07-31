"""Audit-safe command line surface for the installed distribution."""

from __future__ import annotations

import argparse
import json
import platform
from importlib import metadata
from pathlib import Path
from typing import Sequence

from f51_darwin._version import __version__


def _installed_version() -> str:
    try:
        return metadata.version("f51-darwin-ssd")
    except metadata.PackageNotFoundError:
        return __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="f51-darwin")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("version", help="print the distribution version")
    commands.add_parser("doctor", help="print audit-safe installation metadata")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "version":
        print(_installed_version())
        return 0
    if args.command == "doctor":
        print(
            json.dumps(
                {
                    "distribution": "f51-darwin-ssd",
                    "version": _installed_version(),
                    "python": platform.python_version(),
                    "package_root": Path(__file__).resolve().parent.as_posix(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")
