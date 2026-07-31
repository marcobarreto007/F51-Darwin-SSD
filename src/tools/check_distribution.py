#!/usr/bin/env python3
"""Validate source build policy and built wheel/sdist content without imports."""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import email.parser
import hashlib
import io
import json
import re
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[2]
BUILD_REQUIRES = {"setuptools==83.0.0"}
EXCLUDES = {"archive*", "audit*", "research*", "tests*", "tools*", "workspace*"}
FORBIDDEN_PARTS = {
    ".agent_bus",
    ".f51",
    ".organism",
    "archive",
    "audit",
    "checkpoints",
    "data",
    "research",
    "tests",
    "tools",
    "workspace",
}
LOCAL_PATH = re.compile(rb"(?:[A-Za-z]:[\\/]Users[\\/]|/home/|/Users/)", re.IGNORECASE)
SDIST_ROOT_FILES = {
    "LICENSE",
    "MANIFEST.in",
    "NOTICE",
    "PKG-INFO",
    "README.md",
    "pyproject.toml",
    "setup.cfg",
}
EGG_INFO_FILES = {
    "PKG-INFO",
    "SOURCES.txt",
    "dependency_links.txt",
    "entry_points.txt",
    "requires.txt",
    "top_level.txt",
}


def _finding(kind: str, path: str, **details: object) -> dict[str, object]:
    return {"type": kind, "path": path, **details}


def _package_top_level_allowed(parts: tuple[str, ...] | list[str]) -> bool:
    """Accept both the flattened wheel layout (``f51_darwin/...``) and the
    src-layout sdist layout (``src/f51_darwin/...``). Both are the same
    package under setuptools' ``where = ["src"]`` discovery; wheels flatten
    the ``src/`` prefix on install while sdists preserve the source tree."""
    if not parts:
        return False
    if parts[0] == "f51_darwin":
        return True
    return len(parts) >= 2 and parts[0] == "src" and parts[1] == "f51_darwin"


def _source_findings(root: Path) -> tuple[list[dict[str, object]], str, str]:
    findings: list[dict[str, object]] = []
    project_file = root / "pyproject.toml"
    if not project_file.is_file():
        return [_finding("build_metadata_missing", "pyproject.toml")], "", ""
    try:
        data = tomllib.loads(project_file.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        return [_finding("build_metadata_invalid", "pyproject.toml", detail=str(exc))], "", ""
    build = data.get("build-system", {})
    project = data.get("project", {})
    name = str(project.get("name", ""))
    version = str(project.get("version", ""))
    if build.get("build-backend") != "setuptools.build_meta" or set(build.get("requires", [])) != BUILD_REQUIRES:
        findings.append(_finding("build_metadata_missing", "pyproject.toml"))
    scripts = project.get("scripts", {})
    if scripts != {"f51-darwin": "f51_darwin.cli:main"}:
        findings.append(_finding("distribution_cli_mismatch", "pyproject.toml"))
    discovery = data.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find", {})
    if discovery.get("include") != ["f51_darwin*"] or not EXCLUDES <= set(discovery.get("exclude", [])):
        findings.append(_finding("package_discovery_mismatch", "pyproject.toml"))
    for required in (
        "MANIFEST.in",
        "governance/audit/policy/distribution.json",
        "src/f51_darwin/__init__.py",
        "src/f51_darwin/__main__.py",
        "src/f51_darwin/_version.py",
        "src/f51_darwin/cli.py",
        "src/f51_darwin/serving/__init__.py",
        "src/f51_darwin/serving/static/davi.html",
    ):
        if not (root / required).is_file():
            findings.append(_finding("package_source_missing", required))
    init_file = root / "src/f51_darwin/__init__.py"
    if init_file.is_file():
        try:
            init_tree = ast.parse(init_file.read_text(encoding="utf-8-sig"))
            eager_modules: set[str] = set()
            for node in init_tree.body:
                if isinstance(node, ast.Import):
                    eager_modules.update(
                        alias.name for alias in node.names if alias.name.startswith("f51_darwin.")
                    )
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if node.level == 1:
                        module = f"f51_darwin.{module}".rstrip(".")
                    if module.startswith("f51_darwin.") and module != "f51_darwin._version":
                        eager_modules.add(module)
            if eager_modules:
                findings.append(
                    _finding(
                        "package_init_eager_runtime_import",
                        "src/f51_darwin/__init__.py",
                        modules=sorted(eager_modules),
                    )
                )
        except SyntaxError as exc:
            findings.append(_finding("package_init_invalid", "src/f51_darwin/__init__.py", detail=str(exc)))
    version_file = root / "src/f51_darwin/_version.py"
    if version_file.is_file() and not re.search(
        rf"(?m)^__version__\s*=\s*['\"]{re.escape(version)}['\"]\s*$",
        version_file.read_text(encoding="utf-8"),
    ):
        findings.append(_finding("package_version_mismatch", "src/f51_darwin/_version.py"))
    return findings, name, version


def _inspect_member(
    artifact: str, member: str, content: bytes, *, sdist_prefix: str | None
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    path = PurePosixPath(member)
    if path.is_absolute() or ".." in path.parts or "\\" in member:
        return [_finding("distribution_path_escape", f"{artifact}:{member}")]
    parts = list(path.parts)
    if sdist_prefix:
        if not parts or parts[0] != sdist_prefix:
            return [_finding("distribution_root_mismatch", f"{artifact}:{member}")]
        parts = parts[1:]
    lowered = {part.lower() for part in parts}
    if lowered & FORBIDDEN_PARTS:
        findings.append(_finding("distribution_forbidden_path", f"{artifact}:{member}"))
    if LOCAL_PATH.search(content):
        findings.append(_finding("distribution_local_path", f"{artifact}:{member}"))
    return findings


def _package_files(root: Path) -> set[str]:
    package = root / "src" / "f51_darwin"
    if not package.is_dir():
        return set()
    return {
        path.relative_to(root).as_posix()
        for path in package.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix.lower() in {".py", ".html", ".ps1"}
    }


def _metadata_findings(content: bytes, artifact: str, name: str, version: str) -> list[dict[str, object]]:
    message = email.parser.BytesParser().parsebytes(content)
    findings: list[dict[str, object]] = []
    if message.get("Name") != name or message.get("Version") != version:
        findings.append(_finding("distribution_metadata_mismatch", artifact))
    python_range = {part.strip() for part in (message.get("Requires-Python") or "").split(",")}
    if python_range != {">=3.12", "<3.13"}:
        findings.append(_finding("distribution_python_mismatch", artifact))
    return findings


def _record_findings(
    names: set[str], payloads: dict[str, bytes], record_path: str, artifact: str
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    try:
        rows = list(csv.reader(io.StringIO(payloads[record_path].decode("utf-8"))))
    except (KeyError, UnicodeDecodeError, csv.Error) as exc:
        return [_finding("wheel_record_invalid", artifact, detail=str(exc))]
    recorded = {row[0] for row in rows if len(row) == 3}
    if recorded != names or len(rows) != len(recorded):
        findings.append(_finding("wheel_record_members_mismatch", artifact))
    for row in rows:
        if len(row) != 3:
            findings.append(_finding("wheel_record_invalid", artifact))
            continue
        path, digest, size = row
        if path == record_path:
            if digest or size:
                findings.append(_finding("wheel_record_self_hashed", artifact))
            continue
        content = payloads.get(path)
        expected = (
            "sha256="
            + base64.urlsafe_b64encode(hashlib.sha256(content or b"").digest())
            .rstrip(b"=")
            .decode("ascii")
        )
        if content is None or digest != expected or size != str(len(content)):
            findings.append(_finding("wheel_record_hash_mismatch", f"{artifact}:{path}"))
    return findings


def _artifact_findings(root: Path, dist: Path, name: str, version: str) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    wheels = sorted(dist.glob("*.whl")) if dist.is_dir() else []
    sdists = sorted(dist.glob("*.tar.gz")) if dist.is_dir() else []
    allowed_artifacts = {path.name for path in wheels + sdists}
    if dist.is_dir():
        for path in sorted(item for item in dist.iterdir() if item.is_file()):
            if path.name not in allowed_artifacts:
                findings.append(_finding("distribution_artifact_extra", path.name))
    if len(wheels) != 1:
        findings.append(_finding("wheel_count_mismatch", dist.as_posix(), count=len(wheels)))
    if len(sdists) != 1:
        findings.append(_finding("sdist_count_mismatch", dist.as_posix(), count=len(sdists)))
    normalized = name.replace("-", "_")
    expected_package = _package_files(root)
    wheel_package: dict[str, bytes] = {}
    sdist_package: dict[str, bytes] = {}
    for wheel in wheels:
        try:
            with zipfile.ZipFile(wheel) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)):
                    findings.append(_finding("distribution_duplicate_member", wheel.name))
                payloads = {
                    member: archive.read(member)
                    for member in names
                    if not member.endswith("/")
                }
                for member in names:
                    content = payloads.get(member, b"")
                    findings.extend(_inspect_member(wheel.name, member, content, sdist_prefix=None))
                    member_parts = PurePosixPath(member).parts
                    first = member_parts[0] if member_parts else ""
                    if not _package_top_level_allowed(member_parts) and not first.startswith(
                        f"{normalized}-{version}.dist-info"
                    ):
                        findings.append(_finding("wheel_content_not_allowed", f"{wheel.name}:{member}"))
                    if member.startswith("src/f51_darwin/") and not member.endswith("/"):
                        wheel_package[member] = content
                actual_package = set(wheel_package)
                for missing in sorted(expected_package - actual_package):
                    findings.append(_finding("wheel_package_file_missing", f"{wheel.name}:{missing}"))
                for extra in sorted(actual_package - expected_package):
                    findings.append(_finding("wheel_package_file_extra", f"{wheel.name}:{extra}"))
                dist_info = f"{normalized}-{version}.dist-info"
                required = {
                    f"{dist_info}/METADATA",
                    f"{dist_info}/WHEEL",
                    f"{dist_info}/entry_points.txt",
                    f"{dist_info}/top_level.txt",
                    f"{dist_info}/RECORD",
                }
                for missing in sorted(required - set(payloads)):
                    findings.append(_finding("wheel_metadata_file_missing", f"{wheel.name}:{missing}"))
                metadata_path = f"{dist_info}/METADATA"
                if metadata_path in payloads:
                    findings.extend(_metadata_findings(payloads[metadata_path], wheel.name, name, version))
                wheel_path = f"{dist_info}/WHEEL"
                if wheel_path in payloads and b"Tag: py3-none-any" not in payloads[wheel_path]:
                    findings.append(_finding("wheel_tag_mismatch", wheel.name))
                entry_path = f"{dist_info}/entry_points.txt"
                if entry_path in payloads and b"f51-darwin = f51_darwin.cli:main" not in payloads[entry_path]:
                    findings.append(_finding("wheel_entrypoint_mismatch", wheel.name))
                record_path = f"{dist_info}/RECORD"
                if record_path in payloads:
                    findings.extend(_record_findings(set(payloads), payloads, record_path, wheel.name))
        except (OSError, zipfile.BadZipFile) as exc:
            findings.append(_finding("wheel_invalid", wheel.name, detail=str(exc)))
    expected_prefix = f"{normalized}-{version}"
    for sdist in sdists:
        try:
            with tarfile.open(sdist, "r:gz") as archive:
                members = archive.getmembers()
                if len([item.name for item in members]) != len({item.name for item in members}):
                    findings.append(_finding("distribution_duplicate_member", sdist.name))
                for info in members:
                    if info.issym() or info.islnk():
                        findings.append(_finding("distribution_link_member", f"{sdist.name}:{info.name}"))
                    if not info.isfile():
                        continue
                    extracted = archive.extractfile(info)
                    content = extracted.read() if extracted is not None else b""
                    findings.extend(
                        _inspect_member(sdist.name, info.name, content, sdist_prefix=expected_prefix)
                    )
                    parts = PurePosixPath(info.name).parts[1:]
                    first = parts[0] if parts else ""
                    allowed_egg = (
                        first == f"{normalized}.egg-info"
                        and len(parts) == 2
                        and parts[1] in EGG_INFO_FILES
                    )
                    if (
                        first not in SDIST_ROOT_FILES
                        and not _package_top_level_allowed(parts)
                        and not allowed_egg
                    ):
                        findings.append(_finding("sdist_content_not_allowed", f"{sdist.name}:{info.name}"))
                    relative = PurePosixPath(*parts).as_posix()
                    if relative.startswith("src/f51_darwin/"):
                        sdist_package[relative] = content
                actual_package = set(sdist_package)
                for missing in sorted(expected_package - actual_package):
                    findings.append(_finding("sdist_package_file_missing", f"{sdist.name}:{missing}"))
                for extra in sorted(actual_package - expected_package):
                    findings.append(_finding("sdist_package_file_extra", f"{sdist.name}:{extra}"))
        except (OSError, tarfile.TarError) as exc:
            findings.append(_finding("sdist_invalid", sdist.name, detail=str(exc)))
    for path in sorted(set(wheel_package) & set(sdist_package)):
        if wheel_package[path] != sdist_package[path]:
            findings.append(_finding("distribution_payload_mismatch", path))
    return findings


def evaluate(root: Path = ROOT, dist: Path | None = None) -> dict[str, object]:
    root = root.resolve()
    findings, name, version = _source_findings(root)
    if dist is not None:
        findings.extend(_artifact_findings(root, dist.resolve(), name, version))
    findings.sort(key=lambda item: (str(item.get("type")), str(item.get("path"))))
    return {"status": "pass" if not findings else "fail", "findings": findings}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dist", type=Path)
    args = parser.parse_args()
    report = evaluate(args.root, args.dist)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
