#!/usr/bin/env python3
"""Search local roots for the exact manifest-recorded 1.6B gold checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from f51_darwin.artifact_manifest import write_json_atomic
from f51_darwin.organism.checkpoint_root import checkpoint_metadata


@dataclass(frozen=True)
class CandidateResult:
    path: str
    status: str
    reasons: tuple[str, ...]
    size_bytes: int
    sha256: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qualify_candidate(
    path: str | Path, *, expected_size: int, expected_sha256: str
) -> CandidateResult:
    candidate = Path(path).resolve()
    size = candidate.stat().st_size
    if size != expected_size:
        return CandidateResult(
            str(candidate), "rejected", ("size_mismatch",), size, None
        )
    digest = sha256_file(candidate)
    if digest.lower() != expected_sha256.lower():
        return CandidateResult(
            str(candidate), "rejected", ("sha256_mismatch",), size, digest
        )
    return CandidateResult(str(candidate), "hash_verified", (), size, digest)


def expected_gold(manifest: dict[str, Any]) -> dict[str, Any]:
    entry = manifest["checkpoints"]["71"]
    return {
        "model_name": "F51-Darwin-X-1.6B-Nitro",
        "cycle": 71,
        "step": 40751,
        "size_bytes": int(entry["checkpoint_bytes"]),
        "sha256": str(entry["file_sha256"]).lower(),
        "base_checkpoint_id": str(entry["base_checkpoint_id"]),
    }


def search_gold(
    roots: Iterable[str | Path], expected: dict[str, Any]
) -> dict[str, Any]:
    examined: list[str] = []
    results: list[CandidateResult] = []
    for raw_root in roots:
        root = Path(raw_root)
        if not root.exists():
            continue
        examined.append(str(root.resolve()))
        for candidate in root.rglob("*.pt"):
            try:
                if candidate.stat().st_size != expected["size_bytes"]:
                    continue
                result = qualify_candidate(
                    candidate,
                    expected_size=expected["size_bytes"],
                    expected_sha256=expected["sha256"],
                )
                results.append(result)
            except (OSError, PermissionError):
                continue
    verified = [result for result in results if result.status == "hash_verified"]
    return {
        "status": "candidate_found" if verified else "missing",
        "recovered_path": verified[0].path if verified else None,
        "search_roots": examined,
        "candidates": [asdict(result) for result in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, action="append", default=[])
    parser.add_argument("--current-conflict", type=Path)
    args = parser.parse_args()
    import json

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected = expected_gold(manifest)
    roots = args.search_root or [
        Path.home(),
        Path("C:/F51_LAB"),
        Path("C:/llama.cpp"),
        Path("C:/temp"),
    ]
    report = search_gold(roots, expected)
    current_conflict = None
    if args.current_conflict and args.current_conflict.exists():
        current_conflict = checkpoint_metadata(args.current_conflict)
        current_conflict["sha256"] = sha256_file(args.current_conflict)
    report.update(
        {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "expected_gold": expected,
            "current_conflicting_artifact": current_conflict,
            "vss_search": {
                "status": "requires_administrator",
                "command": "powershell -ExecutionPolicy Bypass -File src\\tools\\\find_gold_in_vss.ps1",
            },
        }
    )
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "candidate_found" else 3


if __name__ == "__main__":
    raise SystemExit(main())
