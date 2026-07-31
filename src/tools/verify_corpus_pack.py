from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from f51_darwin.corpus_pack_verifier import verify_corpus_pack


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify an F51 approved corpus pack.")
    parser.add_argument("pack", help="Path to approved_corpus_pack.tar.gz")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = verify_corpus_pack(Path(args.pack))
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
