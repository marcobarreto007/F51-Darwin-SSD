"""Purge failed Olavo firewall records and re-import from staging + audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from f51_darwin.data_factory import (  # noqa: E402
    DataFactory,
    DataFactoryPaths,
    load_factory_config,
    read_item,
)
from f51_darwin.data_firewall import DataFirewall, FirewallConfig  # noqa: E402

STAGING = ROOT / "data" / "generated" / "candidates" / "olavo"


def purge_olavo_rejected(paths: DataFactoryPaths) -> int:
    removed = 0
    for folder in (paths.rejected, paths.quarantine, paths.candidates):
        for path in list(folder.glob("*.json")):
            try:
                record, _ = read_item(path)
            except (KeyError, json.JSONDecodeError):
                path.unlink(missing_ok=True)
                removed += 1
                continue
            if "olavo" in record.source_path.lower():
                path.unlink(missing_ok=True)
                removed += 1
    return removed


def rebuild_hash_index(factory: DataFactory) -> None:
    index: dict[str, str] = {}
    folders = (
        factory.paths.candidates,
        factory.paths.quarantine,
        factory.paths.rejected,
        factory.paths.approved,
    )
    for folder in folders:
        for path in folder.glob("*.json"):
            record, _ = read_item(path)
            index[record.content_hash] = record.id
    factory.ledger._save_hash_index(index)


def main() -> int:
    raw = load_factory_config(ROOT / "src" / "configs" / "data_factory.yaml")
    paths = DataFactoryPaths.from_project(ROOT, raw)
    firewall = DataFirewall(FirewallConfig(**raw.get("firewall", {})))
    factory = DataFactory(paths, firewall=firewall)

    purged = purge_olavo_rejected(paths)
    rebuild_hash_index(factory)

    imported = 0
    skipped = 0
    for pattern in ("livros/*.txt", "artigos/*.txt"):
        for txt in sorted(STAGING.glob(pattern)):
            before = len(list(paths.candidates.glob("*.json")))
            record = factory.import_real_file(txt, skip_duplicates=False)
            if record is None:
                skipped += 1
                continue
            after = len(list(paths.candidates.glob("*.json")))
            if after > before:
                imported += 1
            else:
                skipped += 1

    audited = 0
    results = {"approved": 0, "quarantine": 0, "rejected": 0}
    for record, text in factory.list_candidates():
        if "olavo" not in record.source_path.lower():
            continue
        decision = factory.audit_candidate(record, text)
        audited += 1
        results[decision.status.value] = results.get(decision.status.value, 0) + 1

    summary = {
        "purged": purged,
        "imported": imported,
        "skipped": skipped,
        "audited": audited,
        **results,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
