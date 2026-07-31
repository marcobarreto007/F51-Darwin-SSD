from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_git_archive_collects_without_workspace(tmp_path: Path) -> None:
    """A committed source archive must collect without any ignored artifacts."""
    if not (ROOT / ".git").exists():
        assert not (ROOT / "workspace").exists()
        return
    if os.environ.get("F51_CLEAN_CHECKOUT_NESTED") == "1":
        pytest.skip("outer clean-checkout contract owns recursive collection")

    archive = tmp_path / "source.zip"
    with archive.open("wb") as output:
        result = subprocess.run(
            ["git", "archive", "--format=zip", "HEAD"],
            cwd=ROOT,
            check=False,
            stdout=output,
            stderr=subprocess.PIPE,
        )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    checkout = tmp_path / "checkout"
    with zipfile.ZipFile(archive) as source:
        source.extractall(checkout)
    assert not (checkout / "workspace").exists()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["F51_CLEAN_CHECKOUT_NESTED"] = "1"
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=checkout,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert collected.returncode == 0, collected.stdout + collected.stderr
