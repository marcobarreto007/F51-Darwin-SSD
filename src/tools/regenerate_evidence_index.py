#!/usr/bin/env python3
"""Recompute the supply-chain evidence index from the working tree.

The index pins a blob OID and a SHA-256 for each tracked artifact. Entries
drift whenever a tracked file changes without the index being refreshed, and
the only recourse was hand-editing hashes into the JSON -- which produces an
attestation nobody can reproduce.

Honours ``content_mode`` per entry. ``git_blob_text_lf`` normalises CRLF to
LF before hashing, so the index stays identical across checkouts on Windows
and Linux; hashing the raw bytes would make every entry platform-dependent.

Regenerating asserts that the current content is the intended content, so
``--apply`` is required and the delta is printed either way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "governance/audit/evidence/supply-chain-index.json"


def canonical_bytes(path: Path, content_mode: str) -> bytes:
    raw = path.read_bytes()
    if content_mode == "git_blob_text_lf":
        return raw.replace(b"\r\n", b"\n")
    if content_mode in ("binary", "raw", ""):
        return raw
    raise ValueError(f"unsupported content_mode: {content_mode!r}")


def blob_oid(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate the evidence index")
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    document = json.loads(args.index.read_text(encoding="utf-8"))
    artifacts = document.get("artifacts", [])
    drift: list[tuple[str, str, str]] = []
    missing: list[str] = []

    for entry in artifacts:
        target = args.root / entry["path"]
        if not target.is_file():
            missing.append(entry["path"])
            continue
        payload = canonical_bytes(target, entry.get("content_mode", ""))
        oid = blob_oid(payload)
        digest = hashlib.sha256(payload).hexdigest()
        if oid != entry.get("blob_oid") or digest != entry.get("sha256"):
            drift.append((entry["path"], entry.get("sha256", "")[:12], digest[:12]))
        entry["blob_oid"] = oid
        entry["sha256"] = digest

    print(f"artifacts: {len(artifacts)}")
    print(f"missing  : {missing if missing else 'none'}")
    print(f"drifted  ({len(drift)}):")
    for path, before, after in drift:
        print(f"  {path}\n    {before}... -> {after}...")

    if missing:
        raise SystemExit(
            "index references files that do not exist; refusing to regenerate "
            "a partial index"
        )
    if not drift:
        print("\nIndex already matches the working tree.")
        return 0
    if not args.apply:
        print("\nDry run. Regenerating asserts the current content is intended; "
              "pass --apply to write.")
        return 1

    args.index.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nwritten: {args.index}")
    print("EVIDENCE_INDEX_REGENERATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
