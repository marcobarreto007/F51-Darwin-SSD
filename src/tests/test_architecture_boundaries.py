from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.check_architecture_boundaries import evaluate


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _root(tmp_path: Path) -> Path:
    for directory in (
        "governance/audit/policy",
        "src/configs",
        "src/f51_darwin",
        "src/scripts",
        "src/tools",
        "research",
    ):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    (tmp_path / "src/configs/canonical.yaml").write_text(
        "model_name: F51-Darwin-X-1.6B-Nitro\n", encoding="utf-8"
    )
    (tmp_path / "src/f51_darwin/__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src/f51_darwin/core.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "src/f51_darwin/checkpoint.py").write_text(
        "def load(model, state):\n    return model.load_state_dict(state, strict=True)\n",
        encoding="utf-8",
    )
    (tmp_path / "src/scripts/run.py").write_text(
        "from f51_darwin.core import VALUE\nprint(VALUE)\n", encoding="utf-8"
    )
    (tmp_path / "src/scripts/README.md").write_text("# Operators\n", encoding="utf-8")
    policy = {
        "schema_version": 1,
        "supported_entrypoints": ["src/scripts/run.py"],
        "gold_boundaries": {
            "active_python_roots": ["src/f51_darwin", "src/scripts", "src/tools", "research"],
            "script_allowed_files": ["src/scripts/README.md", "src/scripts/run.py"],
            "package_root": "src/f51_darwin",
            "forbidden_package_import_roots": ["scripts", "tools", "research"],
            "canonical_config": {
                "path": "src/configs/canonical.yaml",
                "model_name": "F51-Darwin-X-1.6B-Nitro",
                "forbidden_default_labels": ["F51-Darwin-X-4B", "F51-Darwin-X-600M"],
            },
            "checkpoint_authority": {
                "path": "src/f51_darwin/checkpoint.py",
                "strict_false_forbidden_roots": ["src/f51_darwin"],
            },
            "size_budgets": {
                "supported_file_lines": 1200,
                "entrypoint_adapter_lines": 250,
                "class_lines": 500,
                "function_lines": 180,
            },
            "exceptions": [],
        },
    }
    (tmp_path / "governance/audit/policy/operational-surface.json").write_text(
        json.dumps(policy), encoding="utf-8"
    )
    return tmp_path


def _types(root: Path) -> set[str]:
    return {finding["type"] for finding in evaluate(root)["findings"]}


def test_clean_architecture_fixture_passes(tmp_path: Path) -> None:
    assert evaluate(_root(tmp_path)) == {"status": "pass", "findings": []}


def test_sys_path_mutation_and_reverse_dependency_fail(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "src/scripts/run.py").write_text(
        "import sys\nsys.path[:] = ['.']\nsys.path.insert(0, '.')\n", encoding="utf-8"
    )
    (root / "src/f51_darwin/core.py").write_text(
        "from scripts.run import main\n", encoding="utf-8"
    )
    assert {"sys_path_mutation", "forbidden_dependency"} <= _types(root)


def test_literal_dynamic_import_cannot_bypass_reverse_dependency_gate(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "src/f51_darwin/core.py").write_text(
        "import importlib\nplugin = importlib.import_module('research.plugin')\n",
        encoding="utf-8",
    )
    assert "forbidden_dependency" in _types(root)


def test_package_cycle_is_reported(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "src/f51_darwin/a.py").write_text("from . import b\n", encoding="utf-8")
    (root / "src/f51_darwin/b.py").write_text("from . import a\n", encoding="utf-8")
    assert "package_cycle" in _types(root)


def test_exact_script_surface_and_size_budgets_fail(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "src/scripts/extra.py").write_text("print('extra')\n", encoding="utf-8")
    policy_path = root / "governance/audit/policy/operational-surface.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["gold_boundaries"]["size_budgets"]["entrypoint_adapter_lines"] = 1
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    assert {"script_surface_mismatch", "file_size_budget"} <= _types(root)


def test_file_size_budget_scans_package_tools_and_research_recursively(tmp_path: Path) -> None:
    root = _root(tmp_path)
    policy_path = root / "governance/audit/policy/operational-surface.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["gold_boundaries"]["size_budgets"]["supported_file_lines"] = 2
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    for relative in (
        "src/f51_darwin/nested/runtime.py",
        "src/tools/nested/checker.py",
        "research/nested/experiment.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ONE = 1\nTWO = 2\nTHREE = 3\n", encoding="utf-8")
    oversized = {
        finding["path"]
        for finding in evaluate(root)["findings"]
        if finding["type"] == "file_size_budget"
    }
    assert {
        "src/f51_darwin/nested/runtime.py",
        "src/tools/nested/checker.py",
        "research/nested/experiment.py",
    } <= oversized


def test_canonical_config_and_checkpoint_authority_fail_closed(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "src/configs/canonical.yaml").write_text(
        "model_name: F51-Darwin-X-4B\n", encoding="utf-8"
    )
    (root / "src/f51_darwin/checkpoint.py").write_text(
        "def load(model, state):\n    return model.load_state_dict(state, strict=False)\n",
        encoding="utf-8",
    )
    assert {"canonical_config_mismatch", "permissive_checkpoint_load"} <= _types(root)


def test_only_declared_strict_checkpoint_loaders_may_delegate(tmp_path: Path) -> None:
    root = _root(tmp_path)
    policy_path = root / "governance/audit/policy/operational-surface.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["gold_boundaries"]["checkpoint_authority"]["delegated_strict_loaders"] = [
        "src/f51_darwin/delegated_checkpoint.py"
    ]
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    delegated = root / "src/f51_darwin/delegated_checkpoint.py"
    delegated.write_text(
        "def load(model, state):\n    return model.load_state_dict(state, strict=True)\n",
        encoding="utf-8",
    )
    undeclared = root / "src/f51_darwin/rogue_checkpoint.py"
    undeclared.write_text(
        "def load(model, state):\n    return model.load_state_dict(state, strict=True)\n",
        encoding="utf-8",
    )
    findings = evaluate(root)["findings"]
    duplicates = {
        finding["path"]
        for finding in findings
        if finding["type"] == "checkpoint_authority_duplicate"
    }
    assert "src/f51_darwin/delegated_checkpoint.py" not in duplicates
    assert "src/f51_darwin/rogue_checkpoint.py" in duplicates


def test_entrypoint_config_default_is_canonical(tmp_path: Path) -> None:
    root = _root(tmp_path)
    policy_path = root / "governance/audit/policy/operational-surface.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["gold_boundaries"]["canonical_config"]["entrypoint_config_defaults"] = {
        "src/scripts/run.py": "src/configs/canonical.yaml"
    }
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    (root / "src/scripts/run.py").write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--config', default='src/configs/legacy.yaml')\n",
        encoding="utf-8",
    )
    assert "entrypoint_config_default_mismatch" in _types(root)


def test_architecture_cli_exit_code_matches_json_status(tmp_path: Path) -> None:
    root = _root(tmp_path)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "src/tools/check_architecture_boundaries.py"),
        "--root",
        str(root),
    ]
    passed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert passed.returncode == 0
    assert json.loads(passed.stdout)["status"] == "pass"
    (root / "src/scripts/extra.py").write_text("", encoding="utf-8")
    failed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert failed.returncode == 1
    assert json.loads(failed.stdout)["status"] == "fail"
