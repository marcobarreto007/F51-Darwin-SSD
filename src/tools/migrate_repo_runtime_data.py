#!/usr/bin/env python3
"""Merge legacy repo-local runtime data into the external corpus workspace.

This helper is intentionally non-destructive: it copies unique artifacts, merges
JSONL ledgers/manifests without duplicating exact lines, and rebuilds the content
hash index. Directory cutover/junction creation remains an explicit operator step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_new_lines(source: Path, target: Path) -> list[str]:
    existing = set(target.read_text(encoding="utf-8").splitlines()) if target.exists() else set()
    return [
        line
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip() and line not in existing
    ]


def rebuild_hash_index(ledger_dir: Path) -> int:
    index_path = ledger_dir / "index.jsonl"
    hashes: dict[str, str] = {}
    if index_path.exists():
        for line_number, line in enumerate(
            index_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                hashes.setdefault(str(record["content_hash"]), str(record["id"]))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"Invalid ledger record at {index_path}:{line_number}"
                ) from exc
    ledger_dir.mkdir(parents=True, exist_ok=True)
    temporary = ledger_dir / ".hashes.json.migration.tmp"
    temporary.write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(ledger_dir / "hashes.json")
    return len(hashes)


def merge_runtime_data(source: Path, target: Path, *, apply: bool) -> dict[str, int]:
    source = source.resolve()
    target = target.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source runtime data not found: {source}")
    if not target.is_dir():
        raise FileNotFoundError(f"External corpus workspace not found: {target}")
    if source == target:
        raise ValueError("Source already resolves to the external corpus workspace")

    report = {"copied": 0, "identical": 0, "jsonl_lines": 0, "hashes": 0}
    operations: list[tuple[str, Path, Path, list[str] | None]] = []

    for source_file in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = source_file.relative_to(source)
        target_file = target / relative
        if relative.as_posix() == "ledger/hashes.json":
            continue
        if not target_file.exists():
            operations.append(("copy", source_file, target_file, None))
            report["copied"] += 1
            continue
        if source_file.suffix == ".jsonl":
            lines = jsonl_new_lines(source_file, target_file)
            operations.append(("jsonl", source_file, target_file, lines))
            report["jsonl_lines"] += len(lines)
            continue
        if file_digest(source_file) == file_digest(target_file):
            report["identical"] += 1
            continue
        raise FileExistsError(
            f"Refusing to overwrite different external artifact: {target_file}"
        )

    if not apply:
        return report

    for operation, source_file, target_file, lines in operations:
        target_file.parent.mkdir(parents=True, exist_ok=True)
        if operation == "copy":
            shutil.copy2(source_file, target_file)
        elif lines:
            with target_file.open("a", encoding="utf-8") as handle:
                for line in lines:
                    handle.write(line + "\n")

    report["hashes"] = rebuild_hash_index(target / "ledger")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    report = merge_runtime_data(args.source, args.target, apply=args.apply)
    print(json.dumps({"mode": "apply" if args.apply else "dry-run", **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
