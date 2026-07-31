#!/usr/bin/env python3
"""Validate the executable and root-artifact classification policy."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXECUTABLE_SUFFIXES = {".py", ".ps1", ".bat", ".sh"}
EXPECTED_SUPPORTED = {
    "src/scripts/bob.ps1",
    "src/scripts/start_100m_auto.ps1",
    "src/scripts/start_overnight_16b.ps1",
    "src/scripts/start_100m_65b.ps1",
    "src/scripts/darwin_organism.py",
    "src/scripts/serve_davi.py",
    "src/scripts/inspect_organism_checkpoint.py",
    "src/scripts/darwin_inventory.py",
    "src/scripts/ingest_pipeline.py",
    "src/scripts/start_smol_darwin_transplant.ps1",
}


def _executables(root: Path, subtree: str) -> set[str]:
    base = root / subtree
    if not base.exists():
        return set()
    return {
        path.relative_to(root).as_posix()
        for path in base.rglob("*")
        if path.is_file()
        and path.name != "__init__.py"
        and path.suffix.lower() in EXECUTABLE_SUFFIXES
    }


def check_surface(root: Path) -> list[dict[str, object]]:
    root = root.resolve()
    policy_path = root / "governance/audit/policy/operational-surface.json"
    if not policy_path.is_file():
        return [{"type": "missing_policy", "path": "governance/audit/policy/operational-surface.json"}]
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    findings: list[dict[str, object]] = []

    actual = set().union(*(_executables(root, subtree) for subtree in ("src/scripts", "src/tools", "research")))
    classified = {row["path"] for row in policy.get("classifications", [])}
    for path in sorted(actual - classified):
        findings.append({"type": "unclassified_executable", "path": path})
    for path in sorted(classified - actual):
        findings.append({"type": "missing_classified_executable", "path": path})
    expected_roots = {
        "supported": "src/scripts/",
        "maintenance": ("src/tools/", "src/scripts/"),
        "research": "research/",
    }
    for row in policy.get("classifications", []):
        path = str(row.get("path", ""))
        classification = str(row.get("classification", ""))
        prefix = expected_roots.get(classification)
        if prefix is None or not path.startswith(prefix):
            findings.append(
                {
                    "type": "classification_directory_mismatch",
                    "path": path,
                    "classification": classification,
                }
            )

    supported = set(policy.get("supported_entrypoints", []))
    if supported != EXPECTED_SUPPORTED:
        findings.append(
            {"type": "supported_contract_mismatch", "expected": sorted(EXPECTED_SUPPORTED), "actual": sorted(supported)}
        )
    supported_rows = {
        row["path"]
        for row in policy.get("classifications", [])
        if row.get("classification") == "supported"
    }
    if supported_rows != supported:
        findings.append({"type": "supported_classification_mismatch"})

    archived = _executables(root, "governance/archive")
    declared_archived = {row["path"] for row in policy.get("archived_executables", [])}
    for path in sorted(archived - declared_archived):
        findings.append({"type": "unclassified_archived_executable", "path": path})
    for row in policy.get("archived_executables", []):
        if row.get("classification") != "historical_non_operational":
            findings.append({"type": "archive_marked_operational", "path": row.get("path")})

    if (root / ".git").exists():
        tracked = subprocess.run(
            ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True
        ).stdout.decode("utf-8").split("\0")
        root_files = {path for path in tracked if path and "/" not in path and "\\" not in path}
    else:
        root_files = {path.name for path in root.iterdir() if path.is_file()}
    declared_roots = {row["path"] for row in policy.get("root_artifacts", [])}
    for path in sorted(root_files - declared_roots):
        findings.append({"type": "unclassified_root_artifact", "path": path})
    for path in sorted(declared_roots - root_files):
        findings.append({"type": "missing_root_artifact", "path": path})
    root_exec = sorted(path for path in root_files if Path(path).suffix.lower() in EXECUTABLE_SUFFIXES)
    if root_exec:
        findings.append({"type": "root_executable", "paths": root_exec})
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    findings = check_surface(args.root)
    print(json.dumps({"status": "pass" if not findings else "fail", "findings": findings}, indent=2))
    return 0 if not findings else 2


if __name__ == "__main__":
    sys.exit(main())
