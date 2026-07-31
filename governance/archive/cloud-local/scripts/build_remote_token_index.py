from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from f51_darwin.corpus_cloud_token_index import build_remote_token_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a remote F51 token-bin training index.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "corpus_factory.yaml"))
    parser.add_argument("--target", choices=["master", "worker"], default="master")
    parser.add_argument("--vocab-label", default="80k")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    target = raw["cloud"][args.target]
    result = build_remote_token_index(
        ssh_host=str(target["ssh"]),
        remote_workspace=str(target["workspace"]),
        vocab_label=args.vocab_label,
        force=args.force,
        dry_run=args.dry_run,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
