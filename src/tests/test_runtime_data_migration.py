from __future__ import annotations

import json
from pathlib import Path

from tools.migrate_repo_runtime_data import merge_runtime_data


def _record(record_id: str, digest: str) -> str:
    return json.dumps({"id": record_id, "content_hash": digest}, sort_keys=True)


def test_runtime_data_merge_is_non_destructive_and_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "repo-data"
    target = tmp_path / "external-corpus"
    (source / "generated" / "rejected").mkdir(parents=True)
    (source / "ledger").mkdir()
    (target / "ledger").mkdir(parents=True)
    (source / "generated" / "rejected" / "ds_source.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (source / "ledger" / "index.jsonl").write_text(
        _record("ds_source", "source-hash") + "\n", encoding="utf-8"
    )
    (target / "ledger" / "index.jsonl").write_text(
        _record("ds_target", "target-hash") + "\n", encoding="utf-8"
    )

    dry_run = merge_runtime_data(source, target, apply=False)
    assert dry_run["copied"] == 1
    assert dry_run["jsonl_lines"] == 1
    assert not (target / "generated" / "rejected" / "ds_source.json").exists()

    applied = merge_runtime_data(source, target, apply=True)
    assert applied["copied"] == 1
    assert applied["jsonl_lines"] == 1
    hashes = json.loads((target / "ledger" / "hashes.json").read_text(encoding="utf-8"))
    assert hashes == {"source-hash": "ds_source", "target-hash": "ds_target"}

    repeated = merge_runtime_data(source, target, apply=True)
    assert repeated["copied"] == 0
    assert repeated["identical"] == 1
    assert repeated["jsonl_lines"] == 0
