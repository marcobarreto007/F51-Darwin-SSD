#!/usr/bin/env python3
"""Fail-closed physical-root hygiene gate with fixture-safe evaluation."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = "governance/audit/policy/physical-root.json"


def _finding(kind: str, path: str, **details: object) -> dict[str, object]:
    return {"type": kind, "path": path, **details}


def _is_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    attributes = getattr(path.stat(), "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _ignore_patterns(root: Path) -> set[str]:
    path = root / ".gitignore"
    if not path.is_file():
        return set()
    return {
        line.strip().replace("\\", "/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#") and not line.startswith("!")
    }


def _cache_findings(root: Path, policy: dict[str, object]) -> list[dict[str, object]]:
    names = set(str(item) for item in policy.get("cache_names", []))
    exclusions = tuple(str(item).strip("/") for item in policy.get("cache_excluded_roots", []))
    findings: list[dict[str, object]] = []
    for current, directories, _files in os.walk(root, followlinks=False):
        current_path = Path(current)
        relative = current_path.relative_to(root).as_posix()
        first = "" if relative == "." else relative.split("/", 1)[0]
        if first in exclusions:
            directories[:] = []
            continue
        for name in sorted(set(directories) & names):
            path = (current_path / name).relative_to(root).as_posix()
            findings.append(_finding("cache_residue", path))
        directories[:] = [name for name in directories if name not in names]
    return findings


def evaluate(root: Path = ROOT) -> dict[str, object]:
    root = root.resolve()
    policy_file = root / POLICY_PATH
    if not policy_file.is_file():
        return {
            "status": "fail",
            "findings": [_finding("physical_policy_missing", POLICY_PATH)],
        }
    policy = json.loads(policy_file.read_text(encoding="utf-8"))
    findings: list[dict[str, object]] = []
    if policy.get("schema_version") != 1 or not policy.get("policy_id"):
        findings.append(_finding("physical_policy_invalid", POLICY_PATH))

    actual_root = {path.name for path in root.iterdir()}
    allowed = set(str(item) for item in policy.get("allowed_root_entries", []))
    required = set(str(item) for item in policy.get("required_root_entries", []))
    forbidden = set(str(item) for item in policy.get("forbidden_root_entries", []))
    optional_tool_owned = set(
        str(item) for item in policy.get("optional_tool_owned_entries", [])
    )
    if not optional_tool_owned <= allowed or optional_tool_owned & required:
        findings.append(_finding("physical_policy_tool_ownership_invalid", POLICY_PATH))
    for name in sorted(actual_root - allowed):
        findings.append(_finding("unclassified_root_entry", name))
    for name in sorted(required - actual_root):
        findings.append(_finding("required_root_entry_missing", name))
    for name in sorted(actual_root & forbidden):
        findings.append(_finding("forbidden_root_entry", name))

    patterns = _ignore_patterns(root)
    for item in policy.get("required_ignored_paths", []):
        path = str(item).replace("\\", "/")
        equivalents = {path, path.rstrip("/"), path.rstrip("/") + "/"}
        if not patterns.intersection(equivalents):
            findings.append(_finding("required_ignore_missing", path))

    for item in policy.get("required_physical_directories", []):
        relative = str(item).strip("/")
        path = root / relative
        if not path.is_dir():
            findings.append(_finding("required_physical_directory_missing", relative))
        elif _is_reparse(path):
            findings.append(_finding("physical_directory_is_reparse", relative))

    findings.extend(_cache_findings(root, policy))
    audit = policy.get("audit_tools", {})
    if isinstance(audit, dict):
        audit_root = root / str(audit.get("root", ".audit-tools"))
        allowed_files = set(str(item).replace("\\", "/") for item in audit.get("allowed_files", []))
        total = 0
        residue: list[str] = []
        if audit_root.exists():
            for path in sorted(audit_root.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(root).as_posix()
                total += path.stat().st_size
                if relative not in allowed_files:
                    residue.append(relative)
        if residue:
            findings.append(
                _finding(
                    "audit_tool_residue",
                    str(audit.get("root", ".audit-tools")),
                    count=len(residue),
                    sample=residue[:20],
                )
            )
        maximum = int(audit.get("max_total_bytes", 0))
        if maximum and total > maximum:
            findings.append(
                _finding("audit_tools_size_budget", str(audit.get("root", ".audit-tools")), bytes=total, limit=maximum)
            )

    findings.sort(key=lambda item: (str(item.get("type")), str(item.get("path"))))
    return {"status": "pass" if not findings else "fail", "findings": findings}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = evaluate(args.root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
