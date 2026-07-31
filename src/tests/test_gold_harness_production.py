from __future__ import annotations

import json
import copy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SHELL = shutil.which("pwsh") or shutil.which("powershell")


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _repository(tmp_path: Path, scenario: str) -> tuple[Path, Path]:
    repo = tmp_path / "source"
    (repo / "src/tools").mkdir(parents=True)
    (repo / "governance/audit/schemas").mkdir(parents=True)
    shutil.copy2(ROOT / "src/tools/run_gold_source_audit.ps1", repo / "src/tools")
    shutil.copy2(ROOT / "src/tools/check_python_compilation.py", repo / "src/tools")
    shutil.copy2(
        ROOT / "governance/audit/schemas/gold-source-report.schema.json",
        repo / "governance/audit/schemas",
    )
    validators = ROOT / "governance/audit/validators"
    if validators.is_dir():
        shutil.copytree(validators, repo / "governance/audit/validators")
    for candidate in ROOT.glob("src/tools/*gold*report*.py"):
        shutil.copy2(candidate, repo / "src/tools")

    passing = "def test_ok():\n    assert True\n"
    test_text = passing
    if scenario == "pytest":
        test_text = "def test_failure():\n    assert False\n"
    elif scenario == "commit":
        test_text = (
            "import subprocess\n"
            "def test_move_head():\n"
            "    subprocess.run(['git','commit','--allow-empty','-m','move head'],check=True,capture_output=True)\n"
        )
    _write(repo / "src/tests/test_sample.py", test_text)
    _write(repo / "README.md", "temporary harness repository\n")
    if scenario == "powershell":
        _write(repo / "broken.ps1", "if ( {\n")
    else:
        _write(repo / "valid.ps1", "$value = 1\n")
    if scenario == "diff":
        _write(repo / "trailing.txt", "committed trailing whitespace   \n")
    if scenario == "compile":
        _write(repo / "governance/archive/invalid.py", "this is archived invalid python !!!\n")
        _write(repo / "active_invalid.py", "this is active invalid python !!!\n")

    checker = "import sys\nprint('{}')\nsys.exit(0)\n"
    for name in (
        "check_docs_links.py",
        "check_canonical_docs.py",
        "check_duplicates.py",
        "check_dependency_policy.py",
        "check_operational_surface.py",
        "check_physical_hygiene.py",
        "check_architecture_boundaries.py",
        "check_distribution.py",
        "check_source_secrets.py",
    ):
        _write(repo / "src/tools" / name, checker)

    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "audit@example.invalid")
    _git(repo, "config", "user.name", "Audit Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", f"{scenario} fixture")
    if scenario == "dirty":
        _write(repo / "README.md", "tracked modification after commit\n")
    return repo, tmp_path / f"{scenario}-report.json"


def _run(repo: Path, report: Path) -> subprocess.CompletedProcess[str]:
    assert SHELL is not None
    return subprocess.run(
        [
            SHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(repo / "src/tools/run_gold_source_audit.ps1"),
            "-Root",
            str(repo),
            "-OutputPath",
            str(report),
            "-PythonExe",
            sys.executable,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize(
    "scenario, expected_gate",
    [
        ("pytest", "pytest"),
        ("powershell", "powershell_parse"),
        ("dirty", "tracked_worktree"),
        ("commit", "source_report_commit"),
        ("diff", "git_diff_check"),
        ("compile", "py_compile"),
    ],
)
def test_gold_harness_propagates_real_production_failure(
    tmp_path: Path, scenario: str, expected_gate: str
) -> None:
    repo, report_path = _repository(tmp_path, scenario)
    result = _run(repo, report_path)
    assert result.returncode == 2, result.stdout + result.stderr
    payload = json.loads(report_path.read_text(encoding="utf-8-sig"))
    assert len(payload["commands"]) == 15
    assert payload["counts"]["total"] == 15
    assert expected_gate in payload["failures"]
    assert "self_test_scenario" not in payload
    assert payload["source_commit"] == payload["report_source_commit"]
    if scenario == "commit":
        assert payload["observed_end_commit"] != payload["source_commit"]
    else:
        assert payload["observed_end_commit"] == payload["source_commit"]


def test_gold_harness_compile_accounting_is_honest(tmp_path: Path) -> None:
    repo, report_path = _repository(tmp_path, "compile")
    _run(repo, report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8-sig"))
    counts = payload["python_compilation"]
    assert counts["tracked"] == counts["attempted"] + counts["skipped_archive"]
    assert counts["attempted"] == counts["compiled"] + counts["failed"]
    assert counts["skipped_archive"] == 1
    assert counts["failed"] == 1


def test_gold_report_validator_rejects_schema_and_semantic_mutations(tmp_path: Path) -> None:
    repo, report_path = _repository(tmp_path, "pytest")
    result = _run(repo, report_path)
    assert result.returncode == 2
    baseline = json.loads(report_path.read_text(encoding="utf-8-sig"))
    validator = repo / "governance/audit/validators/assert_gold_source_report.ps1"
    schema = repo / "governance/audit/schemas/gold-source-report.schema.json"

    mutations = {}
    payload = copy.deepcopy(baseline)
    payload["unknown"] = True
    mutations["unknown_property"] = payload
    payload = copy.deepcopy(baseline)
    payload.pop("root")
    mutations["missing_root"] = payload
    payload = copy.deepcopy(baseline)
    payload["root"] = str(tmp_path / "wrong-root")
    mutations["wrong_root"] = payload
    payload = copy.deepcopy(baseline)
    payload["commands"][1] = copy.deepcopy(payload["commands"][0])
    mutations["duplicate_gate"] = payload
    payload = copy.deepcopy(baseline)
    payload["counts"]["passed"] += 1
    mutations["wrong_counts"] = payload
    payload = copy.deepcopy(baseline)
    payload["status"] = "pass"
    mutations["false_pass"] = payload
    payload = copy.deepcopy(baseline)
    payload["failures"] = []
    mutations["wrong_failures"] = payload
    payload = copy.deepcopy(baseline)
    payload["report_source_commit"] = "b" * 40
    mutations["report_commit_mismatch"] = payload
    payload = copy.deepcopy(baseline)
    payload["observed_end_commit"] = "b" * 40
    mutations["end_commit_gate_mismatch"] = payload
    payload = copy.deepcopy(baseline)
    payload["python_compilation"]["tracked"] += 1
    mutations["compile_identity"] = payload

    for name, mutated in mutations.items():
        path = tmp_path / f"mutated-{name}.json"
        path.write_text(json.dumps(mutated), encoding="utf-8")
        checked = subprocess.run(
            [
                SHELL,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(validator),
                "-ReportPath",
                str(path),
                "-SchemaPath",
                str(schema),
                "-ExpectedRoot",
                str(repo),
                "-ExpectedSourceCommit",
                baseline["source_commit"],
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert checked.returncode != 0, name
