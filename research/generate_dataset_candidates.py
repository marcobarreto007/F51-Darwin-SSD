from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from f51_darwin.data_factory import DataFactory, DataFactoryPaths, load_factory_config
from f51_darwin.data_firewall import DataFirewall, FirewallConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or import dataset candidates. Writes only to data/generated/candidates/.",
    )
    parser.add_argument("--config", default=str(ROOT / "src" / "configs" / "data_factory.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)

    synthetic = sub.add_parser("synthetic", help="Register synthetic candidate text.")
    synthetic.add_argument("--text", required=True)
    synthetic.add_argument("--generator-model", required=True)
    synthetic.add_argument("--generator-checkpoint", required=True)
    synthetic.add_argument("--prompt", default=None)

    import_cmd = sub.add_parser("import-real", help="Import a real text file as candidate.")
    import_cmd.add_argument("--file", required=True)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    raw = load_factory_config(Path(args.config))
    paths = DataFactoryPaths.from_project(ROOT, raw)
    firewall_cfg = FirewallConfig(**raw.get("firewall", {}))
    factory = DataFactory(paths, firewall=DataFirewall(firewall_cfg))

    if args.command == "synthetic":
        record = factory.register_synthetic(
            text=args.text,
            generator_model=args.generator_model,
            generator_checkpoint=args.generator_checkpoint,
            prompt=args.prompt,
            dataset_version=str(raw.get("dataset_version", "v0")),
        )
    else:
        file_path = Path(args.file)
        if not file_path.is_absolute():
            file_path = ROOT / file_path
        record = factory.import_real_file(
            file_path,
            dataset_version=str(raw.get("dataset_version", "v0")),
        )
        if record is None:
            print(json.dumps({"skipped": True, "file": str(file_path)}, indent=2))
            return 0

    payload = {
        "id": record.id,
        "status": record.status.value,
        "source_type": record.source_type.value,
        "content_hash": record.content_hash,
        "path": str(paths.candidates / f"{record.id}.json"),
        "rule": "Synthetic data is never training data by default.",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
