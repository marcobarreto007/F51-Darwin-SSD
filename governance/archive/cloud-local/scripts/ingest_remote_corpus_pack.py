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

from f51_darwin.corpus_cloud_ingest import ingest_remote_pack


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest an uploaded F51 corpus pack on the cloud corpus factory.",
    )
    parser.add_argument("--remote-pack", required=True, help="Remote path to approved_corpus_pack.tar.gz")
    parser.add_argument("--batch-name", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "corpus_factory.yaml"))
    parser.add_argument("--target", choices=["master", "worker"], default="master")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    target = raw["cloud"][args.target]
    result = ingest_remote_pack(
        ssh_host=str(target["ssh"]),
        remote_pack_path=args.remote_pack,
        remote_workspace=str(target["workspace"]),
        batch_name=args.batch_name,
        dry_run=args.dry_run,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
