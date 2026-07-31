from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "module",
    [
        "scripts.darwin_organism",
        "scripts.serve_davi",
        "scripts.ingest_pipeline",
        "scripts.inspect_organism_checkpoint",
    ],
)
def test_supported_python_module_help_runs_with_src_pythonpath(module: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    result = subprocess.run(
        [sys.executable, "-B", "-m", module, "--help"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_inventory_module_runs_with_src_pythonpath() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    result = subprocess.run(
        [sys.executable, "-B", "-m", "scripts.darwin_inventory"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
