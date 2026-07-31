from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.check_physical_hygiene import evaluate


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _root(tmp_path: Path) -> Path:
    for directory in ("governance/audit/policy", "src/f51_darwin", "workspace", ".audit-tools/gitleaks"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    (tmp_path / ".gitignore").write_text(
        "workspace/\n.venv_nitro/\n.audit-tools/\n", encoding="utf-8"
    )
    (tmp_path / ".audit-tools/gitleaks/gitleaks.exe").write_bytes(b"scanner")
    policy = {
        "schema_version": 1,
        "policy_id": "fixture-physical-root",
        "allowed_root_entries": [
            ".audit-tools",
            ".gitignore",
            "governance",
            "src",
            "workspace",
        ],
        "required_root_entries": ["governance", "src", "workspace"],
        "forbidden_root_entries": [".agent_bus", ".f51", ".organism", ".venv_vast"],
        "cache_names": ["__pycache__", ".pytest_cache"],
        "cache_excluded_roots": ["workspace"],
        "required_ignored_paths": ["workspace/", ".venv_nitro/", ".audit-tools/"],
        "required_physical_directories": ["workspace"],
        "audit_tools": {
            "root": ".audit-tools",
            "allowed_files": [".audit-tools/gitleaks/gitleaks.exe"],
            "max_total_bytes": 1024,
        },
    }
    (tmp_path / "governance/audit/policy/physical-root.json").write_text(
        json.dumps(policy), encoding="utf-8"
    )
    return tmp_path


def _types(root: Path) -> set[str]:
    return {finding["type"] for finding in evaluate(root)["findings"]}


def test_clean_physical_fixture_passes(tmp_path: Path) -> None:
    assert evaluate(_root(tmp_path)) == {"status": "pass", "findings": []}


def test_forbidden_legacy_state_and_unclassified_root_fail(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / ".organism").mkdir()
    (root / "mystery").mkdir()
    assert {"forbidden_root_entry", "unclassified_root_entry"} <= _types(root)


def test_active_cache_and_unpinned_audit_tool_fail(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "src/f51_darwin/__pycache__").mkdir()
    (root / ".audit-tools/cache.whl").write_bytes(b"wheel")
    assert {"cache_residue", "audit_tool_residue"} <= _types(root)


def test_required_ignore_and_physical_directory_are_fail_closed(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / ".gitignore").write_text(".audit-tools/\n", encoding="utf-8")
    (root / "workspace").rmdir()
    assert {"required_ignore_missing", "required_root_entry_missing"} <= _types(root)


def test_physical_cli_exit_code_matches_json_status(tmp_path: Path) -> None:
    root = _root(tmp_path)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "src/tools/check_physical_hygiene.py"),
        "--root",
        str(root),
    ]
    passed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert passed.returncode == 0
    assert json.loads(passed.stdout)["status"] == "pass"
    (root / ".organism").mkdir()
    failed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert failed.returncode == 1
    assert json.loads(failed.stdout)["status"] == "fail"
