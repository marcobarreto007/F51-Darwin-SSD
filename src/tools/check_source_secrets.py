#!/usr/bin/env python3
"""Run the pinned Gitleaks scanner over tracked source and all reachable history.

Exit codes: 0 clean, 2 findings, 1 tool/config/execution failure. Secret values
are never copied to stdout/stderr by this wrapper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_VERSION = "8.30.1"
EXPECTED_EXE_SHA256 = "17157e2ee8b76fc8b1d8bee607a250e34b8a8023c8bc81822d4b5ee4d78fcb7c"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(args, cwd=cwd, check=False, capture_output=True)


def _resolve_tool(root: Path, explicit: Path | None) -> Path:
    candidates = [
        explicit,
        Path(os.environ["GITLEAKS_EXE"]) if os.environ.get("GITLEAKS_EXE") else None,
        root / ".audit-tools/gitleaks-8.30.1/gitleaks.exe",
    ]
    tool = next((p.resolve() for p in candidates if p and p.is_file()), None)
    if tool is None:
        raise RuntimeError("pinned_gitleaks_not_found")
    if _sha256(tool) != EXPECTED_EXE_SHA256:
        raise RuntimeError("gitleaks_executable_hash_mismatch")
    version = _run([str(tool), "version"], root)
    if version.returncode or version.stdout.decode("utf-8", "replace").strip() != EXPECTED_VERSION:
        raise RuntimeError("gitleaks_version_mismatch")
    return tool


def _tracked_snapshot(root: Path, destination: Path) -> int:
    listed = _run(["git", "ls-files", "-z"], root)
    if listed.returncode:
        raise RuntimeError("git_ls_files_failed")
    count = 0
    for raw in listed.stdout.split(b"\0"):
        if not raw:
            continue
        relative = raw.decode("utf-8", "surrogateescape")
        source = root / relative
        if not source.is_file():
            raise RuntimeError(f"tracked_file_missing:{relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        count += 1
    return count


def _refset(root: Path) -> tuple[list[str], str]:
    result = _run(
        ["git", "for-each-ref", "--format=%(refname) %(objectname)", "refs/heads", "refs/remotes", "refs/tags"],
        root,
    )
    if result.returncode:
        raise RuntimeError("git_refset_failed")
    refs = sorted(line for line in result.stdout.decode("utf-8", "replace").splitlines() if line)
    return refs, hashlib.sha256(("\n".join(refs) + "\n").encode()).hexdigest()


def _scan(tool: Path, root: Path, config: Path, scope: str, temp: Path) -> dict[str, object]:
    report = temp / f"gitleaks-{scope}.json"
    common = [
        "--config", str(config), "--report-format", "json", "--report-path", str(report),
        "--redact=100", "--no-banner", "--no-color", "--exit-code", "2",
        "--max-decode-depth", "2", "--max-archive-depth", "2",
    ]
    tracked_files = None
    refset_hash = None
    refs: list[str] | None = None
    if scope == "current":
        snapshot = temp / "tracked-snapshot"
        snapshot.mkdir()
        tracked_files = _tracked_snapshot(root, snapshot)
        command = [str(tool), "dir", *common, str(snapshot)]
    else:
        refs, refset_hash = _refset(root)
        command = [str(tool), "git", *common, "--log-opts=--all --full-history", str(root)]
    result = _run(command, root)
    if result.returncode not in (0, 2):
        raise RuntimeError(f"gitleaks_{scope}_execution_failed")
    try:
        findings = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else []
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"gitleaks_{scope}_invalid_report") from exc
    if not isinstance(findings, list):
        raise RuntimeError(f"gitleaks_{scope}_invalid_report")
    if bool(findings) != (result.returncode == 2):
        raise RuntimeError(f"gitleaks_{scope}_exit_report_mismatch")
    safe_findings = [
        {
            "rule_id": item.get("RuleID"),
            "path": item.get("File"),
            "line": item.get("StartLine"),
            "commit": item.get("Commit") or None,
            "fingerprint": item.get("Fingerprint") or None,
        }
        for item in findings
        if isinstance(item, dict)
    ]
    return {
        "scope": scope,
        "status": "fail" if findings else "pass",
        "finding_count": len(findings),
        "findings": safe_findings,
        "report_sha256": _sha256(report) if report.is_file() else None,
        "tracked_files": tracked_files,
        "refset_sha256": refset_hash,
        "refs": refs,
        "command": [Path(command[0]).name, *command[1:]],
    }


def evaluate(root: Path, scope: str, tool_path: Path | None = None) -> tuple[int, dict[str, object]]:
    try:
        root = root.resolve()
        config = root / ".gitleaks.toml"
        if not config.is_file():
            raise RuntimeError("gitleaks_config_missing")
        tool = _resolve_tool(root, tool_path)
        scans: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory(prefix="f51-gitleaks-") as raw_temp:
            temp = Path(raw_temp)
            if scope in {"current", "all"}:
                scans.append(_scan(tool, root, config, "current", temp))
            if scope in {"history", "all"}:
                scans.append(_scan(tool, root, config, "history", temp))
        has_findings = any(scan["finding_count"] for scan in scans)
        return (2 if has_findings else 0), {
            "status": "fail" if has_findings else "pass",
            "tool": "gitleaks",
            "tool_version": EXPECTED_VERSION,
            "tool_sha256": EXPECTED_EXE_SHA256,
            "config_sha256": _sha256(config),
            "scans": scans,
        }
    except Exception as exc:
        return 1, {"status": "error", "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("current", "history", "all"), default="all")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--gitleaks-exe", type=Path)
    args = parser.parse_args()
    code, report = evaluate(args.root, args.scope, args.gitleaks_exe)
    print(json.dumps(report, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
