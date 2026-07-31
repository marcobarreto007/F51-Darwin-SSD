#!/usr/bin/env python3
"""Build and smoke-test an auditable wheel from an immutable Git commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive_path: Path, target: Path) -> None:
    with tarfile.open(archive_path, "r:") as archive:
        target_resolved = target.resolve()
        for member in archive.getmembers():
            destination = (target / member.name).resolve()
            if target_resolved not in destination.parents and destination != target_resolved:
                raise RuntimeError(f"archive member escapes source root: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"archive contains link: {member.name}")
        archive.extractall(target, filter="data")


def _python_in(venv: Path) -> Path:
    candidate = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not candidate.is_file():
        raise RuntimeError(f"venv Python missing: {candidate}")
    return candidate


def build(commit: str, output_root: Path) -> dict[str, object]:
    status = _run(["git", "status", "--porcelain"], cwd=ROOT)
    if status:
        raise RuntimeError("gold build requires a clean worktree")
    resolved = _run(["git", "rev-parse", "--verify", f"{commit}^{{commit}}"], cwd=ROOT)
    short = resolved[:12]
    build_root = output_root.resolve() / short
    if build_root.exists():
        raise RuntimeError(f"build destination already exists: {build_root}")
    source_dir = build_root / "source"
    dist_dir = build_root / "dist"
    smoke_dir = build_root / "smoke"
    empty_dir = build_root / "empty-cwd"
    build_root.mkdir(parents=True)
    source_archive = build_root / f"source-{short}.tar"
    _run(["git", "archive", "--format=tar", "-o", str(source_archive), resolved], cwd=ROOT)
    source_dir.mkdir()
    _safe_extract(source_archive, source_dir)
    dist_dir.mkdir()

    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv executable not found")
    build_flags = ["--offline", "--no-sources", "--force-pep517"]
    _run([uv, "build", "--sdist", "--out-dir", str(dist_dir), *build_flags], cwd=source_dir)
    sdists = list(dist_dir.glob("*.tar.gz"))
    if len(sdists) != 1:
        raise RuntimeError(f"expected one sdist, found {len(sdists)}")
    _run(
        [uv, "build", "--wheel", "--out-dir", str(dist_dir), *build_flags, str(sdists[0])],
        cwd=source_dir,
    )
    generated_ignore = dist_dir / ".gitignore"
    if generated_ignore.is_file():
        generated_ignore.unlink()
    wheels = list(dist_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one wheel, found {len(wheels)}")

    _run(
        [sys.executable, "-B", str(ROOT / "src/tools/check_distribution.py"), "--dist", str(dist_dir)],
        cwd=ROOT,
    )
    _run([uv, "venv", "--no-python-downloads", "--python", sys.executable, str(smoke_dir)], cwd=ROOT)
    smoke_python = _python_in(smoke_dir)
    _run(
        [uv, "pip", "install", "--python", str(smoke_python), "--offline", "--no-index", "--no-deps", str(wheels[0])],
        cwd=ROOT,
    )
    empty_dir.mkdir()
    smoke_code = (
        "import importlib.metadata,importlib.resources,json,sys,f51_darwin;"
        "html=importlib.resources.files('f51_darwin.serving').joinpath('static/davi.html').read_text();"
        "print(json.dumps({'version':f51_darwin.__version__,"
        "'metadata':importlib.metadata.version('f51-darwin-ssd'),"
        "'torch_loaded':'torch' in sys.modules,'token':'__TOKEN__' in html,"
        "'inner_html':'innerHTML' in html},sort_keys=True))"
    )
    smoke_env = dict(os.environ)
    smoke_env["CUDA_VISIBLE_DEVICES"] = "-1"
    import_proof = _run([str(smoke_python), "-B", "-c", smoke_code], cwd=empty_dir, env=smoke_env)
    module_proof = _run([str(smoke_python), "-B", "-m", "f51_darwin", "version"], cwd=empty_dir, env=smoke_env)
    console = smoke_dir / ("Scripts/f51-darwin.exe" if os.name == "nt" else "bin/f51-darwin")
    console_proof = _run([str(console), "version"], cwd=empty_dir, env=smoke_env)
    expected_import = {
        "inner_html": False,
        "metadata": "0.1.0",
        "token": True,
        "torch_loaded": False,
        "version": "0.1.0",
    }
    if json.loads(import_proof) != expected_import or module_proof != "0.1.0" or console_proof != "0.1.0":
        raise RuntimeError("installed wheel smoke proof did not match the audited contract")

    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(dist_dir.iterdir())
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "commit": resolved,
        "source_archive": {"bytes": source_archive.stat().st_size, "sha256": _sha256(source_archive)},
        "artifacts": artifacts,
        "tools": {
            "python": sys.version.split()[0],
            "uv": _run([uv, "--version"], cwd=ROOT),
            "git": _run(["git", "--version"], cwd=ROOT),
        },
        "smoke": {
            "cuda_visible_devices": "-1",
            "import": json.loads(import_proof),
            "module_cli": module_proof,
            "console_cli": console_proof,
        },
    }
    manifest_path = build_root / "distribution-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"build_root": str(build_root), "manifest": manifest}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--output-root", type=Path, default=ROOT / "workspace/runtime/builds")
    args = parser.parse_args()
    try:
        report = {"status": "pass", **build(args.commit, args.output_root)}
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        report = {"status": "fail", "error": str(exc)}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
