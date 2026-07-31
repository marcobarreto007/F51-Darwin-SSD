#!/usr/bin/env python3
"""Reject duplicate tracked content and duplicate canonical authority claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = Path("governance/audit/policy/duplicates.json")


def _source_files(root: Path) -> list[Path]:
    """Return the tracked surface plus non-ignored additions awaiting commit."""
    try:
        raw = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            stderr=subprocess.DEVNULL,
        )
        relatives = [Path(item.decode("utf-8", "surrogateescape")) for item in raw.split(b"\0") if item]
    except (OSError, subprocess.CalledProcessError):
        relatives = [path.relative_to(root) for path in root.rglob("*") if path.is_file()]
    return sorted(
        {
            relative
            for relative in relatives
            if ".git" not in relative.parts and (root / relative).is_file()
        },
        key=lambda path: path.as_posix(),
    )


def _finding(kind: str, **details: Any) -> dict[str, Any]:
    return {"type": kind, **details}


def _content_hash(payload: bytes) -> str:
    """Hash UTF-8 text with canonical LF; hash binary bytes unchanged."""
    try:
        canonical = payload.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    except UnicodeDecodeError:
        canonical = payload
    return hashlib.sha256(canonical).hexdigest()


def check(root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    policy_file = root / POLICY_PATH
    if not policy_file.is_file():
        return [_finding("missing_duplicate_policy", path=POLICY_PATH.as_posix())]
    try:
        policy = json.loads(policy_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [_finding("invalid_duplicate_policy", error=str(exc))]

    findings: list[dict[str, Any]] = []
    if policy.get("schema_version") != 1 or policy.get("scope") != "tracked_source_tree":
        findings.append(_finding("invalid_duplicate_policy", error="unsupported schema or scope"))

    files = _source_files(root)
    digest_paths: dict[str, list[str]] = defaultdict(list)
    texts: dict[str, str] = {}
    for relative in files:
        path = root / relative
        payload = path.read_bytes()
        digest_paths[_content_hash(payload)].append(relative.as_posix())
        try:
            texts[relative.as_posix()] = payload.decode("utf-8")
        except UnicodeDecodeError:
            pass

    approved: dict[frozenset[str], str] = {}
    for exception in policy.get("exact_duplicate_exceptions", []):
        paths = exception.get("paths")
        digest = exception.get("sha256")
        category = exception.get("category")
        content_mode = exception.get("content_mode")
        justification = exception.get("justification")
        valid = (
            isinstance(paths, list)
            and len(paths) >= 2
            and len(paths) == len(set(paths))
            and all(isinstance(path, str) and path in texts for path in paths)
            and isinstance(digest, str)
            and len(digest) == 64
            and category == "required_authority_mirror"
            and content_mode == "utf8_text_lf"
            and isinstance(justification, str)
            and len(justification) >= 40
            and all(_content_hash((root / path).read_bytes()) == digest for path in paths)
        )
        if not valid:
            findings.append(_finding("invalid_duplicate_exception", exception=exception))
            continue
        approved[frozenset(paths)] = digest

    observed_approved: set[frozenset[str]] = set()
    for digest, paths in sorted(digest_paths.items()):
        if len(paths) < 2:
            continue
        group = frozenset(paths)
        if approved.get(group) == digest:
            observed_approved.add(group)
        else:
            findings.append(_finding("unapproved_exact_duplicate", sha256=digest, paths=paths))
    for group in approved:
        if group not in observed_approved:
            findings.append(_finding("stale_duplicate_exception", paths=sorted(group)))

    canonical = policy.get("canonical_documents", [])
    for path in canonical:
        if not isinstance(path, str) or path not in texts:
            findings.append(_finding("missing_canonical_document", path=path))

    for claim in policy.get("authority_claims", []):
        marker = claim.get("marker")
        expected_path = claim.get("path")
        if not isinstance(marker, str) or not marker or not isinstance(expected_path, str):
            findings.append(_finding("invalid_authority_claim", claim=claim))
            continue
        locations = sorted(path for path in canonical if marker in texts.get(path, ""))
        if len(locations) > 1:
            findings.append(_finding("duplicate_authority_claim", marker=marker, paths=locations))
        elif locations != [expected_path]:
            findings.append(
                _finding("missing_or_misplaced_authority_claim", marker=marker, expected=expected_path, observed=locations)
            )

    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    findings = check(args.root)
    print(json.dumps({"status": "pass" if not findings else "fail", "findings": findings}, indent=2))
    return 0 if not findings else 2


if __name__ == "__main__":
    sys.exit(main())
