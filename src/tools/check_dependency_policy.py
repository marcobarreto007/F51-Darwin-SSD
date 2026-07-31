#!/usr/bin/env python3
"""Fail-closed CPU audit lock, SBOM, report and evidence reconciliation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[2]
PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)$")
HASH = re.compile(r"--hash=sha256:([0-9a-f]{64})(?:\s|\\|$)")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_BLOB_OID = re.compile(r"^[0-9a-f]{40}$")
INDEX_PATH = "governance/audit/evidence/supply-chain-index.json"
INDEX_SCHEMA = "governance/audit/schemas/supply-chain-index.schema.json"
INDEX_TARGET = "cpu-governance/audit/windows-amd64/cpython-3.12"
INDEX_HASH_PROFILE = "git-blob-text-lf-v1"
INDEX_STATEMENT = (
    "Verdict is computed from artifact content and exact hashes; "
    "no handwritten pass field is authoritative."
)
CONFIGURATION_ARTIFACTS = {
    ".github/workflows/ci.yml",
    "governance/audit/policy/dependencies.json",
    INDEX_SCHEMA,
    "pyproject.toml",
    "src/tools/check_dependency_policy.py",
}
TOOLING = {
    "pip_audit": {
        "tool": "pip-audit",
        "version": "2.10.1",
        "command": [
            "pip-audit",
            "-r",
            "requirements-cpu-audit-dev.lock",
            "--require-hashes",
            "--format",
            "json",
        ],
    },
    "pip_licenses": {
        "tool": "pip-licenses",
        "version": "5.5.5",
        "command": ["pip-licenses", "--format=json", "--with-urls"],
    },
    "cyclonedx": {
        "tool": "cyclonedx-py",
        "version": "7.3.0",
        "command": [
            "cyclonedx-py",
            "environment",
            "--output-reproducible",
            "--spec-version",
            "1.6",
        ],
    },
}


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def canonical_artifact_bytes(
    path: Path, content_mode: object, label: str, failures: list[str]
) -> bytes | None:
    """Read exact source bytes using the checkout-portable declared profile."""
    if content_mode != "git_blob_text_lf":
        failures.append(f"evidence_index:content_mode_invalid:{label}")
        return None
    content = path.read_bytes().replace(b"\r\n", b"\n")
    if b"\x00" in content:
        failures.append(f"evidence_index:text_contains_nul:{label}")
        return None
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        failures.append(f"evidence_index:text_not_utf8:{label}")
        return None
    return content


def git_blob_oid(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content, usedforsecurity=False).hexdigest()


def head_blob_oid(root: Path, path: str, failures: list[str]) -> str | None:
    """Resolve the frozen HEAD blob when Git metadata is available."""
    if not (root / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"HEAD:{path}"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    oid = result.stdout.strip().lower()
    if result.returncode != 0 or not GIT_BLOB_OID.fullmatch(oid):
        failures.append(f"evidence_index:head_blob_unavailable:{path}")
        return None
    return oid


def pins(items: list[str], label: str, failures: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        match = PIN.fullmatch(item.strip())
        if not match:
            failures.append(f"{label}:not_exact:{item}")
            continue
        name, version = canonical(match.group(1)), match.group(2)
        if name in parsed:
            failures.append(f"{label}:duplicate:{name}")
        parsed[name] = version
    return parsed


def input_pins(path: Path, failures: list[str]) -> dict[str, str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    return pins(lines, path.name, failures)


def lock_pins(path: Path, failures: list[str]) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    result: dict[str, str] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line or line.lstrip().startswith("#") or line[0].isspace():
            index += 1
            continue
        block = [line]
        index += 1
        while index < len(lines) and (not lines[index] or lines[index][0].isspace()):
            block.append(lines[index])
            index += 1
        header = block[0].rstrip(" \\")
        match = PIN.fullmatch(header)
        if not match:
            failures.append(f"{path.name}:invalid_block:{block[0]}")
            continue
        name, version = canonical(match.group(1)), match.group(2)
        if name in result:
            failures.append(f"{path.name}:duplicate:{name}")
        hashes = HASH.findall("\n".join(block))
        hash_tokens = re.findall(r"--hash=sha256:([^\s\\]+)", "\n".join(block))
        if not hashes:
            failures.append(f"{path.name}:hash_missing:{name}")
        if len(hashes) != len(hash_tokens):
            failures.append(f"{path.name}:hash_malformed:{name}")
        if len(hashes) != len(set(hashes)):
            failures.append(f"{path.name}:duplicate_hash:{name}")
        result[name] = version
    return result


def component_set(items: list[dict[str, object]], name_key: str, version_key: str,
                  label: str, failures: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        name = canonical(str(item.get(name_key, "")))
        version = str(item.get(version_key, ""))
        if not name or not version:
            failures.append(f"{label}:invalid_component")
            continue
        if name in result:
            failures.append(f"{label}:duplicate:{name}")
        result[name] = version
    return result


def named_sets(scope: dict[str, object], root: Path, failures: list[str]) -> dict[str, dict[str, str]]:
    dev_lock = lock_pins(root / str(scope["dev_lock"]), failures)
    sbom = json.loads((root / str(scope["sbom"])).read_text(encoding="utf-8"))
    if sbom.get("bomFormat") != "CycloneDX" or sbom.get("specVersion") != "1.6":
        failures.append("sbom:invalid_format")
    sbom_set = component_set(sbom.get("components", []), "name", "version", "sbom", failures)
    vuln = json.loads((root / str(scope["vulnerability_report"])).read_text(encoding="utf-8"))
    vuln_items = vuln.get("dependencies", [])
    if not isinstance(vuln_items, list):
        failures.append("vulnerability:invalid_dependencies")
        vuln_items = []
    for item in vuln_items:
        if item.get("vulns"):
            failures.append(f"vulnerability:unexcepted:{canonical(str(item.get('name', 'unknown')))}")
    vuln_set = component_set(vuln_items, "name", "version", "vulnerability", failures)
    licenses = json.loads((root / str(scope["license_report"])).read_text(encoding="utf-8"))
    if not isinstance(licenses, list):
        failures.append("license:invalid_report")
        licenses = []
    license_set = component_set(licenses, "Name", "Version", "license", failures)
    return {"dev_lock": dev_lock, "sbom": sbom_set, "vulnerability": vuln_set, "license": license_set}


def validate_reconciliation(scope: dict[str, object], sets: dict[str, dict[str, str]],
                            failures: list[str]) -> None:
    universe = {(name, version) for values in sets.values() for name, version in values.items()}
    declared: set[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = set()
    for item in scope.get("reconciliations", []):
        parsed = pins([str(item.get("component", ""))], "reconciliation", failures)
        if len(parsed) != 1 or not item.get("classification") or not item.get("reason"):
            failures.append("reconciliation:incomplete")
            continue
        name, version = next(iter(parsed.items()))
        present = tuple(sorted(str(v) for v in item.get("present_in", [])))
        absent = tuple(sorted(str(v) for v in item.get("absent_from", [])))
        actual_present = tuple(sorted(label for label, values in sets.items() if values.get(name) == version))
        actual_absent = tuple(sorted(set(sets) - set(actual_present)))
        if present != actual_present or absent != actual_absent:
            failures.append(f"reconciliation:mismatch:{name}")
        declared.add((name, version, present, absent))
    for name, version in sorted(universe):
        present = tuple(sorted(label for label, values in sets.items() if values.get(name) == version))
        absent = tuple(sorted(set(sets) - set(present)))
        version_conflict = any(name in values and values[name] != version for values in sets.values())
        if version_conflict:
            failures.append(f"component:version_conflict:{name}")
        if absent and (name, version, present, absent) not in declared:
            failures.append(f"component:unreconciled:{name}=={version}")


def confined_path(root: Path, raw: object, failures: list[str]) -> Path | None:
    """Return a canonical, root-confined artifact path or record a failure."""
    value = str(raw)
    posix = PurePosixPath(value)
    invalid = (
        not isinstance(raw, str)
        or not value
        or "\\" in value
        or posix.is_absolute()
        or str(posix) != value
        or any(part in {"", ".", ".."} for part in posix.parts)
        or bool(re.match(r"^[A-Za-z]:", value))
    )
    if invalid:
        failures.append(f"evidence_index:path_escape:{value}")
        return None
    candidate = (root / Path(*posix.parts)).resolve()
    try:
        candidate.relative_to(root)
    except (ValueError, OSError):
        failures.append(f"evidence_index:path_escape:{value}")
        return None
    return candidate


def validate_evidence_index(
    scope: dict[str, object], root: Path, failures: list[str]
) -> None:
    """Bind the complete CPU audit evidence set to unique confined path hashes."""
    declaration = scope.get("evidence_index")
    if not isinstance(declaration, dict):
        failures.append("policy:evidence_index_missing")
        return
    expected_declaration_fields = {
        "path",
        "schema",
        "hash_profile",
        "target",
        "configuration_artifacts",
        "tooling",
        "statement",
    }
    if set(declaration) != expected_declaration_fields:
        failures.append("policy:evidence_index_fields_mismatch")
    if declaration.get("path") != INDEX_PATH:
        failures.append("policy:evidence_index_path_mismatch")
    if declaration.get("schema") != INDEX_SCHEMA:
        failures.append("policy:evidence_index_schema_mismatch")
    if declaration.get("hash_profile") != INDEX_HASH_PROFILE:
        failures.append("policy:evidence_index_hash_profile_mismatch")
    if declaration.get("target") != INDEX_TARGET:
        failures.append("policy:evidence_index_target_mismatch")
    if declaration.get("statement") != INDEX_STATEMENT:
        failures.append("policy:evidence_index_statement_mismatch")
    declared_configs = declaration.get("configuration_artifacts")
    if (
        not isinstance(declared_configs, list)
        or len(declared_configs) != len(set(str(item) for item in declared_configs))
        or set(str(item) for item in declared_configs) != CONFIGURATION_ARTIFACTS
    ):
        failures.append("policy:evidence_index_configuration_mismatch")
    if declaration.get("tooling") != TOOLING:
        failures.append("policy:evidence_index_tooling_mismatch")

    content_fields = (
        "input",
        "dev_input",
        "lock",
        "dev_lock",
        "sbom",
        "vulnerability_report",
        "license_report",
    )
    expected_paths = CONFIGURATION_ARTIFACTS | {
        str(scope.get(field, "")) for field in content_fields
    }
    if "" in expected_paths:
        failures.append("policy:evidence_artifact_declaration_missing")
        expected_paths.discard("")

    index_path = root / INDEX_PATH
    if not index_path.is_file():
        failures.append("evidence_index:missing")
        return
    index = json.loads(index_path.read_text(encoding="utf-8"))
    required_index_fields = {
        "schema_version",
        "schema",
        "hash_profile",
        "generated_utc",
        "target",
        "artifacts",
        "tooling",
        "statement",
    }
    if not isinstance(index, dict) or set(index) != required_index_fields:
        failures.append("evidence_index:fields_mismatch")
    if index.get("schema_version") != 2:
        failures.append("evidence_index:schema_version_mismatch")
    if index.get("schema") != INDEX_SCHEMA:
        failures.append("evidence_index:schema_mismatch")
    if index.get("hash_profile") != INDEX_HASH_PROFILE:
        failures.append("evidence_index:hash_profile_mismatch")
    if index.get("target") != INDEX_TARGET:
        failures.append("evidence_index:target_mismatch")
    if index.get("statement") != INDEX_STATEMENT:
        failures.append("evidence_index:statement_mismatch")
    try:
        generated = datetime.strptime(
            str(index.get("generated_utc", "")), "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)
        if generated > datetime.now(timezone.utc):
            failures.append("evidence_index:generated_in_future")
    except ValueError:
        failures.append("evidence_index:generated_utc_invalid")
    if index.get("tooling") != TOOLING:
        failures.append("evidence_index:tooling_mismatch")

    artifacts = index.get("artifacts")
    if not isinstance(artifacts, list):
        failures.append("evidence_index:artifacts_invalid")
        artifacts = []
    observed_paths: list[str] = []
    for position, item in enumerate(artifacts):
        if not isinstance(item, dict) or set(item) != {
            "path",
            "content_mode",
            "blob_oid",
            "sha256",
        }:
            failures.append(f"evidence_index:artifact_fields_mismatch:{position}")
            continue
        raw_path = item.get("path")
        value = str(raw_path)
        observed_paths.append(value)
        digest = item.get("sha256")
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            failures.append(f"evidence_index:hash_invalid:{value}")
        expected_oid = item.get("blob_oid")
        if not isinstance(expected_oid, str) or not GIT_BLOB_OID.fullmatch(expected_oid):
            failures.append(f"evidence_index:blob_oid_invalid:{value}")
        path = confined_path(root, raw_path, failures)
        if path is None:
            continue
        if not path.is_file():
            failures.append(f"evidence_index:artifact_unreadable:{value}")
            continue
        content = canonical_artifact_bytes(
            path, item.get("content_mode"), value, failures
        )
        if content is None:
            continue
        actual_oid = git_blob_oid(content)
        if isinstance(expected_oid, str) and GIT_BLOB_OID.fullmatch(expected_oid):
            if actual_oid != expected_oid:
                failures.append(f"evidence_index:blob_oid_mismatch:{value}")
            frozen_oid = head_blob_oid(root, value, failures)
            if frozen_oid is not None and frozen_oid != expected_oid:
                failures.append(f"evidence_index:head_blob_mismatch:{value}")
        if isinstance(digest, str) and SHA256.fullmatch(digest):
            if hashlib.sha256(content).hexdigest() != digest:
                failures.append(f"evidence_index:hash_mismatch:{value}")

    counts = {path: observed_paths.count(path) for path in set(observed_paths)}
    for path, count in sorted(counts.items()):
        if count > 1:
            failures.append(f"evidence_index:artifact_duplicate:{path}")
    observed_set = set(observed_paths)
    for path in sorted(expected_paths - observed_set):
        failures.append(f"evidence_index:artifact_missing:{path}")
    for path in sorted(observed_set - expected_paths):
        failures.append(f"evidence_index:artifact_extra:{path}")


def evaluate(root: Path = ROOT) -> dict[str, object]:
    failures: list[str] = []
    root = root.resolve()
    policy = json.loads((root / "governance/audit/policy/dependencies.json").read_text(encoding="utf-8"))
    if policy.get("schema_version") != 3 or policy.get("supported_python") != "3.12":
        failures.append("policy:invalid_header")
    scope = policy.get("scopes", {}).get("cpu_audit", {})
    direct = input_pins(root / str(scope["input"]), failures)
    dev_direct = input_pins(root / str(scope["dev_input"]), failures)
    runtime_lock = lock_pins(root / str(scope["lock"]), failures)
    dev_lock = lock_pins(root / str(scope["dev_lock"]), failures)
    policy_direct = pins(scope.get("direct_runtime_requirements", []), "policy_runtime", failures)
    policy_dev = pins(scope.get("direct_dev_requirements", []), "policy_dev", failures)
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project.get("project", {}).get("optional-dependencies", {})
    project_direct = pins(extras.get(str(scope["pyproject_extra"]), []), "pyproject_runtime", failures)
    project_dev = pins(extras.get(str(scope["pyproject_dev_extra"]), []), "pyproject_dev", failures)
    if not (direct == policy_direct == project_direct):
        failures.append("direct_runtime:set_mismatch")
    if not (dev_direct == policy_dev == project_dev):
        failures.append("direct_dev:set_mismatch")
    if not set(direct) <= set(runtime_lock) or any(runtime_lock.get(k) != v for k, v in direct.items()):
        failures.append("runtime_lock:direct_mismatch")
    if not set(dev_direct) <= set(dev_lock) or any(dev_lock.get(k) != v for k, v in dev_direct.items()):
        failures.append("dev_lock:direct_mismatch")
    if not set(runtime_lock.items()) <= set(dev_lock.items()):
        failures.append("dev_lock:not_runtime_superset")
    sets = named_sets(scope, root, failures)
    validate_reconciliation(scope, sets, failures)

    validate_evidence_index(scope, root, failures)

    now = datetime.now(timezone.utc)
    for exception in policy.get("exceptions", []):
        required = {"advisory", "owner", "reason", "mitigation", "expires_utc"}
        if not required <= set(exception):
            failures.append("policy:incomplete_exception")
            continue
        expiry = datetime.fromisoformat(str(exception["expires_utc"]).replace("Z", "+00:00"))
        if expiry <= now:
            failures.append(f"policy:expired_exception:{exception['advisory']}")
    return {"status": "pass" if not failures else "fail", "failures": sorted(set(failures))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = evaluate(args.root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
