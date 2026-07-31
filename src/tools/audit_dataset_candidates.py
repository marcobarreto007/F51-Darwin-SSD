from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from f51_darwin.data_factory import DataFactory, DataFactoryPaths, load_factory_config
from f51_darwin.data_firewall import DataFirewall, FirewallConfig
from f51_darwin.dataset_states import DatasetStatus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit dataset candidates through the contamination firewall.",
    )
    parser.add_argument("--config", default=str(ROOT / "src" / "configs" / "data_factory.yaml"))
    parser.add_argument("--id", default=None, help="Audit one candidate id. Defaults to all candidates.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    raw = load_factory_config(Path(args.config))
    paths = DataFactoryPaths.from_project(ROOT, raw)
    firewall = DataFirewall(FirewallConfig(**raw.get("firewall", {})))
    factory = DataFactory(paths, firewall=firewall)

    candidates = factory.list_candidates()
    if args.id:
        candidates = [item for item in candidates if item[0].id == args.id]
    if not candidates:
        print(json.dumps({"audited": 0, "message": "no candidates found"}, indent=2))
        return 0

    results = []
    for record, text in candidates:
        decision = factory.audit_candidate(record, text)
        results.append(
            {
                "id": record.id,
                "final_status": decision.status.value,
                "reason": decision.reason,
                "scores": decision.scores,
            }
        )

    summary = {
        "audited": len(results),
        "approved": sum(1 for item in results if item["final_status"] == DatasetStatus.APPROVED.value),
        "quarantine": sum(1 for item in results if item["final_status"] == DatasetStatus.QUARANTINE.value),
        "rejected": sum(1 for item in results if item["final_status"] == DatasetStatus.REJECTED.value),
        "results": results,
        "note": "Approved items stay in data/approved/. Corpus promotion requires promote_approved_data.py.",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
