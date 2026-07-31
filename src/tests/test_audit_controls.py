from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_legal_and_security_controls_exist() -> None:
    for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "SECURITY.md"):
        assert (ROOT / name).is_file(), name
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "F51 Labs" in license_text
    assert "All rights reserved" in license_text
    assert "Permission is not granted" in license_text


def test_retired_installer_provenance_records_exact_commands() -> None:
    provenance = json.loads(
        (ROOT / "governance/audit/provenance/retired-installers.json").read_text(encoding="utf-8")
    )
    assert provenance["verification"]["tools"] == {
        "powershell": "7.6.3",
        "git": "2.53.0.windows.3",
        "os": "Microsoft Windows NT 10.0.26200.0",
    }
    for installer in provenance["installers"]:
        commands = installer["verification_commands"]
        assert len(commands) == 4
        assert all(installer["path"] in command for command in commands)
        assert all("<installer>" not in command for command in commands)


def test_python_312_and_exact_direct_dependencies() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.12,<3.13"' in project
    for name in ("requirements-cpu-audit.in", "requirements-cpu-audit-dev.in"):
        lines = [
            line.strip()
            for line in (ROOT / name).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        assert lines
        assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[^\s;]+", line) for line in lines)


def test_locks_are_hash_checked_and_cover_direct_inputs() -> None:
    for input_name, lock_name in (
        ("requirements-cpu-audit.in", "requirements-cpu-audit.lock"),
        ("requirements-cpu-audit-dev.in", "requirements-cpu-audit-dev.lock"),
    ):
        direct = [
            line.split("==", 1)[0].lower()
            for line in (ROOT / input_name).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        lock = (ROOT / lock_name).read_text(encoding="utf-8")
        assert "--hash=sha256:" in lock
        assert all(re.search(rf"(?mi)^{re.escape(package)}==", lock) for package in direct)


def test_dependency_policy_and_cyclonedx_sbom_are_machine_readable() -> None:
    policy = json.loads((ROOT / "governance/audit/policy/dependencies.json").read_text(encoding="utf-8"))
    assert policy["supported_python"] == "3.12"
    assert policy["schema_version"] == 3
    assert policy["scopes"]["cpu_audit"]["authority"] == "source_ci_only_not_nitro_runtime"
    assert policy["exceptions"] == []

    sbom = json.loads((ROOT / "governance/audit/sbom/cpu-audit.cyclonedx.json").read_text(encoding="utf-8"))
    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["specVersion"] in {"1.5", "1.6", "1.7"}
    assert sbom["metadata"]["component"]["name"] == "f51-darwin-ssd"
    assert sbom["components"]


def test_ci_is_windows_pinned_and_runs_all_source_gates() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    harness = (ROOT / "src/tools/run_gold_source_audit.ps1").read_text(encoding="utf-8")
    assert "windows-latest" in ci
    assert "actions/checkout@08eba0b27e820071cde6df949e0beb9ba4906955" in ci
    assert "actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405" in ci
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in ci
    assert not re.search(r"uses:\s*[^\s@]+@v\d", ci)
    for token in (
        "requirements-cpu-audit-dev.lock",
        "d29144deff3a68aa93ced33dddf84b7fdc26070add4aa0f4513094c8332afc4e",
        "fetch --force --tags",
        "run_gold_source_audit.ps1",
    ):
        assert token in ci
    for token in (
        "CUDA_VISIBLE_DEVICES",
        "pytest",
        "py_compile",
        "Parser]::ParseFile",
        "check_docs_links.py",
        "check_canonical_docs.py",
        "check_duplicates.py",
        "check_dependency_policy.py",
        "check_operational_surface.py",
        "check_physical_hygiene.py",
        "check_python_compilation.py",
        "check_architecture_boundaries.py",
        "check_distribution.py",
        "secret",
        "diff-tree",
        "--check",
        "git status --porcelain",
    ):
        assert token in harness


def test_dependency_checker_and_generated_boundary_exist() -> None:
    assert (ROOT / "src/tools/check_dependency_policy.py").is_file()
    assert (ROOT / "governance/audit/generated/.gitkeep").is_file()
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "governance/audit/generated/" in ignore


def _run_checker(name: str, root: Path) -> subprocess.CompletedProcess[str]:
    script = ROOT / "src" / "tools" / name
    assert script.is_file(), name
    return subprocess.run(
        [sys.executable, str(script), "--root", str(root)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_docs_link_checker_fails_on_broken_relative_link(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("[missing](governance/docs/missing.md)\n", encoding="utf-8")
    result = _run_checker("check_docs_links.py", tmp_path)
    assert result.returncode != 0
    assert "broken_link" in result.stdout


def test_docs_link_checker_fails_on_broken_inline_repository_path(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "Run `src/scripts/removed_tool.py --root .` now.\n", encoding="utf-8"
    )
    result = _run_checker("check_docs_links.py", tmp_path)
    assert result.returncode != 0
    assert "broken_inline_path" in result.stdout


def test_canonical_checker_fails_on_stale_local_claim(tmp_path: Path) -> None:
    (tmp_path / "governance/docs/operacao").mkdir(parents=True)
    (tmp_path / "README.md").write_text(
        "active runtime uses C:/Users/marco/Desktop/F51-Dataset-Organizado\n",
        encoding="utf-8",
    )
    result = _run_checker("check_canonical_docs.py", tmp_path)
    assert result.returncode != 0
    assert "stale_claim" in result.stdout


def test_surface_checker_fails_on_unclassified_executable(tmp_path: Path) -> None:
    (tmp_path / "src/scripts").mkdir(parents=True)
    (tmp_path / "src/scripts/new_tool.py").write_text("print('tool')\n", encoding="utf-8")
    policy = tmp_path / "governance/audit/policy"
    policy.mkdir(parents=True)
    (policy / "operational-surface.json").write_text(
        json.dumps(
            {
                "supported_entrypoints": [],
                "classifications": [],
                "archived_executables": [],
                "root_artifacts": [],
            }
        ),
        encoding="utf-8",
    )
    result = _run_checker("check_operational_surface.py", tmp_path)
    assert result.returncode != 0
    assert "unclassified_executable" in result.stdout


def test_gold_report_schema_requires_commit_bound_evidence() -> None:
    schema = json.loads(
        (ROOT / "governance/audit/schemas/gold-source-report.schema.json").read_text(encoding="utf-8")
    )
    required = set(schema["required"])
    assert {
        "source_commit",
        "report_source_commit",
        "observed_end_commit",
        "root",
        "commands",
        "python_compilation",
        "status",
        "failures",
    } <= required
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["commit"]["pattern"] == "^[0-9a-f]{40}$"
    assert schema["properties"]["counts"]["properties"]["total"]["const"] == 15
