#!/usr/bin/env python3
"""Verify the supported Nitro interpreter against pinned local provenance."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def evaluate(
    root: Path = ROOT,
    manifest_path: Path | None = None,
    *,
    require_hardware: bool = False,
) -> dict[str, object]:
    failures: list[str] = []
    root = root.resolve()
    manifest_path = manifest_path or root / "governance/audit/provenance/nitro-runtime.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    python = manifest["python"]
    expected_executable = root / manifest["canary_interpreter_relative"]
    if Path(sys.executable).resolve() != expected_executable.resolve():
        failures.append("python:wrong_interpreter")
    if platform.python_implementation() != python["implementation"]:
        failures.append("python:implementation_mismatch")
    if platform.python_version() != python["version"]:
        failures.append("python:version_mismatch")
    if platform.platform() != python["platform"] or platform.machine() != python["machine"]:
        failures.append("python:platform_mismatch")
    if expected_executable.is_file() and _hash(expected_executable) != python["executable_sha256"]:
        failures.append("python:executable_hash_mismatch")

    try:
        import torch
    except Exception:
        failures.append("torch:import_failed")
        torch = None
    # Um diretorio torch/ sem __init__.py importa com sucesso como namespace
    # package vazio, entao o guarda do import acima nao pega. O crash vinha
    # depois, no primeiro atributo -- e um checker que levanta AttributeError
    # em vez de reportar a falha nao entrega o veredito que existe para dar.
    # Observado com uma instalacao parcial de torch no site-packages global do
    # usuario, fora de qualquer venv do projeto.
    if torch is not None and not hasattr(torch, "__version__"):
        failures.append("torch:namespace_package_without_runtime")
        torch = None

    if torch is not None:
        expected = manifest["torch"]
        if torch.__version__ != expected["version"]:
            failures.append("torch:version_mismatch")
        if torch.version.cuda != expected["cuda_runtime"]:
            failures.append("torch:cuda_build_mismatch")
        if getattr(torch.version, "git_version", None) != expected["git_revision"]:
            failures.append("torch:git_revision_mismatch")
        if int(torch.backends.cudnn.version() or 0) != expected["cudnn_version"]:
            failures.append("torch:cudnn_mismatch")
        for item in expected["installed_file_identities"]:
            path = root / item["path"]
            if not path.is_file() or path.stat().st_size != item["size_bytes"] or _hash(path) != item["sha256"]:
                failures.append(f"torch:installed_file_mismatch:{item['path']}")
        if require_hardware:
            binding = manifest["gpu_binding"]
            if expected["cuda_required"] and not torch.cuda.is_available():
                failures.append("cuda:not_available")
            if torch.cuda.device_count() != binding["device_count"]:
                failures.append("cuda:device_count_mismatch")
            else:
                for expected_device in binding["devices"]:
                    index = expected_device["index"]
                    properties = torch.cuda.get_device_properties(index)
                    if properties.name != expected_device["name"]:
                        failures.append(f"cuda:name_mismatch:{index}")
                    if [properties.major, properties.minor] != expected_device["compute_capability"]:
                        failures.append(f"cuda:capability_mismatch:{index}")
                    if properties.total_memory != expected_device["total_memory_bytes"]:
                        failures.append(f"cuda:memory_mismatch:{index}")
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=False, capture_output=True, text=True,
            )
            versions = {line.strip() for line in driver.stdout.splitlines() if line.strip()}
            if driver.returncode or versions != {binding["driver_observed"]}:
                failures.append("cuda:driver_mismatch")

    acquisition = manifest["torch"]["acquisition"]
    if manifest.get("reproducibility", {}).get("status") == "pass":
        if not re.fullmatch(r"[0-9a-f]{64}", str(acquisition.get("wheel_sha256", ""))):
            failures.append("provenance:wheel_hash_missing")
        if not str(acquisition.get("wheel_url", "")).startswith("https://download-r2.pytorch.org/"):
            failures.append("provenance:wheel_url_invalid")
        if not acquisition.get("wheel_filename") or not acquisition.get("wheel_size_bytes"):
            failures.append("provenance:wheel_identity_incomplete")
    sbom_path = root / manifest["sbom"]["path"]
    if not sbom_path.is_file() or _hash(sbom_path) != manifest["sbom"]["sha256"]:
        failures.append("sbom:hash_mismatch")
    else:
        sbom = json.loads(sbom_path.read_text(encoding="utf-8"))
        sbom_set = {_canonical(item["name"]): str(item["version"]) for item in sbom.get("components", [])}
        installed_set = {
            _canonical(dist.metadata["Name"]): dist.version
            for dist in importlib.metadata.distributions()
            if dist.metadata.get("Name")
        }
        if sbom_set != installed_set:
            failures.append("sbom:installed_component_set_mismatch")
    return {"status": "pass" if not failures else "fail", "failures": sorted(set(failures))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--require-hardware", action="store_true")
    args = parser.parse_args()
    report = evaluate(args.root, args.manifest, require_hardware=args.require_hardware)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
