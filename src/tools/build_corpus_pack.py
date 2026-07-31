from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from f51_darwin.corpus_factory_pipeline import build_corpus_pack


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an approved corpus pack from an external folder using F51 policy gates.",
    )
    parser.add_argument("--input", required=True, help="External folder containing raw .txt/.md/.tex files.")
    parser.add_argument("--output", required=True, help="External output folder for manifest and pack.")
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional JSONL metadata with path, source_url, license, and domain_hint per file.",
    )
    parser.add_argument("--default-license", default="unknown")
    parser.add_argument("--source-url-base", default="file://")
    parser.add_argument("--domain-hint", default=None)
    parser.add_argument("--no-export-approved", action="store_true")
    parser.add_argument("--no-pack", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_corpus_pack(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        metadata_path=Path(args.metadata) if args.metadata else None,
        default_license=args.default_license,
        source_url_base=args.source_url_base,
        domain_hint=args.domain_hint,
        export_approved=not args.no_export_approved,
        create_pack=not args.no_pack,
    )
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
    return 0 if summary.scanned else 1


if __name__ == "__main__":
    raise SystemExit(main())
