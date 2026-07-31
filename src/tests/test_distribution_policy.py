from __future__ import annotations

import io
import base64
import csv
import hashlib
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

from tools.check_distribution import evaluate


PROJECT_ROOT = Path(__file__).resolve().parents[2]


PYPROJECT = """\
[build-system]
requires = ["setuptools==83.0.0"]
build-backend = "setuptools.build_meta"

[project]
name = "f51-darwin-ssd"
version = "0.1.0"
requires-python = ">=3.12,<3.13"

[project.scripts]
f51-darwin = "f51_darwin.cli:main"

[tool.setuptools.packages.find]
include = ["f51_darwin*"]
exclude = ["archive*", "audit*", "research*", "tests*", "tools*", "workspace*"]
"""


def _root(tmp_path: Path) -> Path:
    (tmp_path / "f51_darwin").mkdir()
    (tmp_path / "src/f51_darwin/serving/static").mkdir(parents=True)
    (tmp_path / "governance/audit/policy").mkdir(parents=True)
    (tmp_path / "governance/audit/policy/distribution.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (tmp_path / "README.md").write_text("# F51\n", encoding="utf-8")
    (tmp_path / "LICENSE").write_text("proprietary\n", encoding="utf-8")
    (tmp_path / "src/f51_darwin/__init__.py").write_text(
        "from ._version import __version__\n", encoding="utf-8"
    )
    (tmp_path / "src/f51_darwin/_version.py").write_text(
        '__version__ = "0.1.0"\n', encoding="utf-8"
    )
    (tmp_path / "src/f51_darwin/cli.py").write_text(
        "def main():\n    return 0\n", encoding="utf-8"
    )
    (tmp_path / "src/f51_darwin/__main__.py").write_text(
        "from .cli import main\n", encoding="utf-8"
    )
    (tmp_path / "src/f51_darwin/serving/__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src/f51_darwin/serving/static/davi.html").write_text(
        "<script>const TOKEN = __TOKEN__;</script>\n", encoding="utf-8"
    )
    (tmp_path / "MANIFEST.in").write_text(
        "recursive-include f51_darwin *.py *.html\n", encoding="utf-8"
    )
    return tmp_path


def _types(root: Path, dist: Path | None = None) -> set[str]:
    return {finding["type"] for finding in evaluate(root, dist)["findings"]}


def _artifacts(root: Path, *, forbidden: bool = False, local_path: bool = False) -> Path:
    dist = root / "dist"
    dist.mkdir()
    wheel = dist / "f51_darwin_ssd-0.1.0-py3-none-any.whl"
    metadata = "Name: f51-darwin-ssd\nVersion: 0.1.0\nRequires-Python: >=3.12,<3.13\n"
    if local_path:
        metadata += "Home-page: C:\\Users\\operator\\repo\n"
    payloads = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in (root / "src" / "f51_darwin").rglob("*")
        if path.is_file()
    }
    dist_info = "f51_darwin_ssd-0.1.0.dist-info"
    payloads.update(
        {
            f"{dist_info}/METADATA": metadata.encode(),
            f"{dist_info}/WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-any\n",
            f"{dist_info}/entry_points.txt": b"[console_scripts]\nf51-darwin = f51_darwin.cli:main\n",
            f"{dist_info}/top_level.txt": b"src\\f51_darwin\\n",
        }
    )
    if forbidden:
        payloads["src/tests/test_bad.py"] = b""
    record_path = f"{dist_info}/RECORD"
    rows = []
    for path, content in payloads.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
        rows.append((path, f"sha256={digest}", str(len(content))))
    rows.append((record_path, "", ""))
    output = io.StringIO()
    csv.writer(output, lineterminator="\n").writerows(rows)
    payloads[record_path] = output.getvalue().encode()
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, content in payloads.items():
            archive.writestr(path, content)
    sdist = dist / "f51_darwin_ssd-0.1.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        source_payloads = {
            "pyproject.toml": PYPROJECT.encode(),
            "README.md": b"# F51\n",
            "LICENSE": b"proprietary\n",
        }
        source_payloads.update(
            {
                path.relative_to(root).as_posix(): path.read_bytes()
                for path in (root / "src" / "f51_darwin").rglob("*")
                if path.is_file()
            }
        )
        for name, data in source_payloads.items():
            payload = data
            info = tarfile.TarInfo(f"f51_darwin_ssd-0.1.0/{name}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return dist


def _rewrite_wheel(wheel: Path, transform) -> None:
    with zipfile.ZipFile(wheel) as archive:
        payloads = {name: archive.read(name) for name in archive.namelist()}
    payloads = transform(payloads)
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in payloads.items():
            archive.writestr(name, content)


def test_complete_distribution_source_policy_passes(tmp_path: Path) -> None:
    assert evaluate(_root(tmp_path)) == {"status": "pass", "findings": []}


def test_missing_build_metadata_fails(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    assert "build_metadata_missing" in _types(root)


def test_base_package_import_must_be_lazy(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "src/f51_darwin/__init__.py").write_text(
        "from f51_darwin.darwin_x import DarwinXModel\n", encoding="utf-8"
    )
    assert "package_init_eager_runtime_import" in _types(root)


def test_clean_wheel_and_sdist_contents_pass(tmp_path: Path) -> None:
    root = _root(tmp_path)
    assert evaluate(root, _artifacts(root)) == {"status": "pass", "findings": []}


def test_distribution_rejects_forbidden_paths_and_local_absolute_paths(tmp_path: Path) -> None:
    root = _root(tmp_path)
    types = _types(root, _artifacts(root, forbidden=True, local_path=True))
    assert {"distribution_forbidden_path", "distribution_local_path"} <= types


def test_distribution_rejects_missing_package_and_metadata_files(tmp_path: Path) -> None:
    root = _root(tmp_path)
    dist = _artifacts(root)
    wheel = next(dist.glob("*.whl"))
    _rewrite_wheel(
        wheel,
        lambda payloads: {
            name: content
            for name, content in payloads.items()
            if name not in {"src/f51_darwin/cli.py", "f51_darwin_ssd-0.1.0.dist-info/WHEEL"}
        },
    )
    assert {"wheel_package_file_missing", "wheel_metadata_file_missing"} <= _types(root, dist)


def test_distribution_rejects_record_and_cross_artifact_payload_tampering(tmp_path: Path) -> None:
    root = _root(tmp_path)
    dist = _artifacts(root)
    wheel = next(dist.glob("*.whl"))
    _rewrite_wheel(
        wheel,
        lambda payloads: {**payloads, "src/f51_darwin/cli.py": b"def main(): return 51\n"},
    )
    assert {"wheel_record_hash_mismatch", "distribution_payload_mismatch"} <= _types(root, dist)


def test_distribution_rejects_extra_dist_artifact(tmp_path: Path) -> None:
    root = _root(tmp_path)
    dist = _artifacts(root)
    (dist / ".gitignore").write_text("*\n", encoding="utf-8")
    assert "distribution_artifact_extra" in _types(root, dist)


def test_distribution_cli_exit_code_matches_json_status(tmp_path: Path) -> None:
    root = _root(tmp_path)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "src/tools/check_distribution.py"),
        "--root",
        str(root),
    ]
    passed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert passed.returncode == 0
    assert json.loads(passed.stdout)["status"] == "pass"
    (root / "src/f51_darwin/cli.py").unlink()
    failed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert failed.returncode == 1
    assert json.loads(failed.stdout)["status"] == "fail"


def test_source_import_cli_and_packaged_ui_are_dependency_lazy() -> None:
    command = (
        "import importlib.resources,json,sys,f51_darwin;"
        "html=importlib.resources.files('f51_darwin.serving').joinpath('static/davi.html').read_text();"
        "print(json.dumps({'version':f51_darwin.__version__,'torch':('torch' in sys.modules),"
        "'token':('__TOKEN__' in html),'inner_html':('innerHTML' in html)}))"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", command],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "version": "0.1.0",
        "torch": False,
        "token": True,
        "inner_html": False,
    }
