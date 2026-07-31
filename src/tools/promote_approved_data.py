from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from f51_darwin.data_factory import DataFactory, DataFactoryPaths, approve_quarantine_item, load_factory_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explicitly promote approved dataset items into clean training areas.",
    )
    parser.add_argument("--config", default=str(ROOT / "src" / "configs" / "data_factory.yaml"))
    parser.add_argument(
        "--approve-id",
        action="append",
        default=[],
        help="Promote one quarantine item to approved by id. Repeatable.",
    )
    parser.add_argument(
        "--reason",
        default="manual explicit approval",
        help="Approval reason stored in ledger.",
    )
    parser.add_argument(
        "--build-corpus",
        action="store_true",
        help="Copy approved items into data/corpus/ for training.",
    )
    parser.add_argument(
        "--export-approved",
        action="store_true",
        help="Export approved items as .txt under data/approved/.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    raw = load_factory_config(Path(args.config))
    paths = DataFactoryPaths.from_project(ROOT, raw)
    factory = DataFactory(paths)

    promoted: list[str] = []
    for record_id in args.approve_id:
        record = approve_quarantine_item(factory, record_id, reason=args.reason)
        promoted.append(record.id)

    exported: list[str] = []
    if args.export_approved or args.build_corpus:
        paths_list = factory.promote_approved_to_corpus(build_corpus=args.build_corpus)
        exported = [str(path) for path in paths_list]

    payload = {
        "approved_ids": promoted,
        "exported_paths": exported,
        "build_corpus": args.build_corpus,
        "rule": "Only this script may copy approved data into data/corpus/.",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
