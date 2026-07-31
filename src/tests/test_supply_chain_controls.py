from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DEPENDENCY = _load("dependency_policy", ROOT / "src/tools/check_dependency_policy.py")
SECRETS = _load("source_secrets", ROOT / "src/tools/check_source_secrets.py")


def _copy_dependency_tree(tmp_path: Path) -> Path:
    paths = {
        "pyproject.toml",
        "requirements-cpu-audit.in",
        "requirements-cpu-audit-dev.in",
        "requirements-cpu-audit.lock",
        "requirements-cpu-audit-dev.lock",
        "governance/audit/policy/dependencies.json",
        "governance/audit/evidence/supply-chain-index.json",
        "governance/audit/sbom/cpu-audit.cyclonedx.json",
        "governance/audit/sbom/cpu-audit-vulnerabilities.json",
        "governance/audit/sbom/cpu-audit-licenses.json",
    }
    index = json.loads(
        (ROOT / "governance/audit/evidence/supply-chain-index.json").read_text(encoding="utf-8")
    )
    paths.update(str(item["path"]) for item in index["artifacts"])
    for relative in sorted(paths):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    return tmp_path


def _assert_failed(root: Path, token: str) -> None:
    report = DEPENDENCY.evaluate(root)
    assert report["status"] == "fail"
    assert any(token in failure for failure in report["failures"]), report


def test_cpu_supply_chain_index_baseline_passes(tmp_path: Path) -> None:
    assert DEPENDENCY.evaluate(_copy_dependency_tree(tmp_path)) == {
        "status": "pass",
        "failures": [],
    }


def test_cpu_supply_chain_hashes_are_portable_across_crlf_checkout(
    tmp_path: Path,
) -> None:
    root = _copy_dependency_tree(tmp_path)
    path = root / "requirements-cpu-audit-dev.lock"
    canonical = path.read_bytes().replace(b"\r\n", b"\n")
    path.write_bytes(canonical)
    assert DEPENDENCY.evaluate(root) == {"status": "pass", "failures": []}
    path.write_bytes(canonical.replace(b"\n", b"\r\n"))

    assert DEPENDENCY.evaluate(root) == {"status": "pass", "failures": []}


def test_cpu_supply_chain_index_passes_fresh_head_git_archive(tmp_path: Path) -> None:
    if not (ROOT / ".git").exists():
        assert DEPENDENCY.evaluate(ROOT) == {"status": "pass", "failures": []}
        return
    archive = tmp_path / "head.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", "HEAD", "-o", str(archive)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    extracted = tmp_path / "archive"
    extracted.mkdir()
    shutil.unpack_archive(archive, extracted, filter="data")

    assert DEPENDENCY.evaluate(extracted) == {"status": "pass", "failures": []}


def test_dependency_index_rejects_removed_binding_and_mutated_content(
    tmp_path: Path,
) -> None:
    root = _copy_dependency_tree(tmp_path)
    index_path = root / "governance/audit/evidence/supply-chain-index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["artifacts"] = [
        item
        for item in index["artifacts"]
        if item["path"] != "requirements-cpu-audit-dev.lock"
    ]
    index_path.write_text(json.dumps(index), encoding="utf-8")
    lock = root / "requirements-cpu-audit-dev.lock"
    lock.write_text(lock.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    _assert_failed(root, "evidence_index:artifact_missing:requirements-cpu-audit-dev.lock")


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ("removed", "evidence_index:artifact_missing:requirements-cpu-audit-dev.lock"),
        ("duplicate", "evidence_index:artifact_duplicate:requirements-cpu-audit-dev.lock"),
        ("extra", "evidence_index:artifact_extra:unexpected.txt"),
        ("path_swap", "evidence_index:hash_mismatch:requirements-cpu-audit.lock"),
        ("unbound_content", "evidence_index:hash_mismatch:requirements-cpu-audit-dev.lock"),
        ("blob_oid", "evidence_index:blob_oid_mismatch:requirements-cpu-audit-dev.lock"),
        ("path_escape", "evidence_index:path_escape:../outside.txt"),
    ],
)
def test_cpu_supply_chain_index_is_exact_unique_and_hash_bound(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    root = _copy_dependency_tree(tmp_path)
    index_path = root / "governance/audit/evidence/supply-chain-index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    artifacts = index["artifacts"]
    dev_lock = next(
        item for item in artifacts if item["path"] == "requirements-cpu-audit-dev.lock"
    )
    if mutation == "removed":
        artifacts.remove(dev_lock)
    elif mutation == "duplicate":
        artifacts.append(dict(dev_lock))
    elif mutation == "extra":
        content = b"unexpected\n"
        (root / "unexpected.txt").write_bytes(content)
        artifacts.append(
            {
                "path": "unexpected.txt",
                "content_mode": "git_blob_text_lf",
                "blob_oid": DEPENDENCY.git_blob_oid(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    elif mutation == "path_swap":
        runtime_lock = next(
            item for item in artifacts if item["path"] == "requirements-cpu-audit.lock"
        )
        runtime_lock["path"], dev_lock["path"] = dev_lock["path"], runtime_lock["path"]
    elif mutation == "unbound_content":
        lock = root / "requirements-cpu-audit-dev.lock"
        lock.write_text(lock.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    elif mutation == "blob_oid":
        dev_lock["blob_oid"] = "0" * 40
    else:
        dev_lock["path"] = "../outside.txt"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    _assert_failed(root, expected)


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ("transitive_version", "component:version_conflict"),
        ("missing_hash", "hash_missing"),
        ("malformed_hash", "hash_malformed"),
        ("stale_sbom", "evidence_index:hash_mismatch"),
        ("stale_report", "evidence_index:hash_mismatch"),
        ("direct_mismatch", "direct_runtime:set_mismatch"),
        ("missing_reconciliation", "component:unreconciled"),
    ],
)
def test_dependency_checker_rejects_mutations(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    root = _copy_dependency_tree(tmp_path)
    if mutation == "transitive_version":
        path = root / "requirements-cpu-audit-dev.lock"
        path.write_text(path.read_text(encoding="utf-8").replace("colorama==0.4.6", "colorama==0.4.5", 1), encoding="utf-8")
    elif mutation == "missing_hash":
        path = root / "requirements-cpu-audit-dev.lock"
        text = path.read_text(encoding="utf-8")
        start = text.index("colorama==")
        end = text.index("filelock==", start)
        path.write_text(text[:start] + "colorama==0.4.6\n" + text[end:], encoding="utf-8")
    elif mutation == "malformed_hash":
        path = root / "requirements-cpu-audit-dev.lock"
        path.write_text(path.read_text(encoding="utf-8").replace("--hash=sha256:08695", "--hash=sha256:zz695", 1), encoding="utf-8")
    elif mutation == "stale_sbom":
        path = root / "governance/audit/sbom/cpu-audit.cyclonedx.json"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    elif mutation == "stale_report":
        path = root / "governance/audit/sbom/cpu-audit-vulnerabilities.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["dependencies"][0]["version"] = "0.0.0"
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "direct_mismatch":
        path = root / "requirements-cpu-audit.in"
        path.write_text(path.read_text(encoding="utf-8").replace("numpy==2.5.1", "numpy==2.5.0"), encoding="utf-8")
    else:
        path = root / "governance/audit/policy/dependencies.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["scopes"]["cpu_audit"]["reconciliations"] = []
        path.write_text(json.dumps(payload), encoding="utf-8")
    _assert_failed(root, expected)


def _gitleaks_exe() -> Path:
    explicit = os.environ.get("GITLEAKS_EXE")
    path = Path(explicit) if explicit else ROOT / ".audit-tools/gitleaks-8.30.1/gitleaks.exe"
    if not path.is_file():
        pytest.skip("pinned Gitleaks binary not bootstrapped")
    return path


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _secret_repo(tmp_path: Path) -> Path:
    shutil.copy2(ROOT / ".gitleaks.toml", tmp_path / ".gitleaks.toml")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "audit@example.invalid")
    _git(tmp_path, "config", "user.name", "Audit Test")
    return tmp_path


def _seed_values() -> list[str]:
    mixed = "aB3dE5gH7jK9mN2pQ4sT6vW8xY"
    upper = "A1B2C3D4E5F6G7H8J9K0"
    return [
        "hf" + "_" + "".join(mixed[index % len(mixed)] for index in range(34)),
        "eyJ" + mixed[:12] + "." + mixed[4:16] + "." + mixed[8:20],
        "AKIA" + upper[:16],
        "generic" + "-credential-" + mixed[:20],
    ]


def test_gitleaks_detects_seeded_current_values_without_disclosure(tmp_path: Path) -> None:
    root = _secret_repo(tmp_path)
    values = _seed_values()
    (root / "seed.txt").write_text(
        f"hf={values[0]}\njwt={values[1]}\naws={values[2]}\npassword={values[3]}\n",
        encoding="utf-8",
    )
    _git(root, "add", ".")
    code, report = SECRETS.evaluate(root, "current", _gitleaks_exe())
    serialized = json.dumps(report)
    assert code == 2
    assert report["scans"][0]["finding_count"] >= 4, [
        finding["rule_id"] for finding in report["scans"][0]["findings"]
    ]
    assert all(value not in serialized for value in values)


def test_gitleaks_scans_removed_secret_in_all_history_without_disclosure(tmp_path: Path) -> None:
    root = _secret_repo(tmp_path)
    value = _seed_values()[0]
    (root / "historical.txt").write_text(value, encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "seed")
    (root / "historical.txt").write_text("sanitized\n", encoding="utf-8")
    _git(root, "commit", "-qam", "sanitize")
    code, report = SECRETS.evaluate(root, "history", _gitleaks_exe())
    serialized = json.dumps(report)
    assert code == 2
    assert report["scans"][0]["finding_count"] >= 1
    assert value not in serialized
    assert report["scans"][0]["refset_sha256"]


def test_nitro_checker_passes_actual_runtime_and_rejects_mutated_manifest(tmp_path: Path) -> None:
    nitro = ROOT / ".venv_nitro/Scripts/python.exe"
    if not nitro.is_file():
        pytest.skip("Nitro runtime not present")
    checker = ROOT / "src/tools/check_nitro_runtime.py"
    actual = subprocess.run([str(nitro), "-I", "-B", str(checker), "--root", str(ROOT)], check=False, capture_output=True, text=True)
    assert actual.returncode == 0, actual.stdout + actual.stderr
    manifest = json.loads((ROOT / "governance/audit/provenance/nitro-runtime.json").read_text(encoding="utf-8"))
    manifest["torch"]["version"] = "0.0.0"
    mutated = tmp_path / "nitro-mutated.json"
    mutated.write_text(json.dumps(manifest), encoding="utf-8")
    failed = subprocess.run([str(nitro), "-I", "-B", str(checker), "--root", str(ROOT), "--manifest", str(mutated)], check=False, capture_output=True, text=True)
    assert failed.returncode != 0
    assert "torch:version_mismatch" in failed.stdout


def test_nitro_hardware_gate_is_explicit_and_visibility_sensitive() -> None:
    nitro = ROOT / ".venv_nitro/Scripts/python.exe"
    if not nitro.is_file():
        pytest.skip("Nitro runtime not present")
    checker = ROOT / "src/tools/check_nitro_runtime.py"
    result = subprocess.run(
        [
            str(nitro),
            "-I",
            "-B",
            str(checker),
            "--root",
            str(ROOT),
            "--require-hardware",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        assert result.returncode != 0
        assert "cuda:not_available" in result.stdout
    else:
        assert result.returncode == 0, result.stdout + result.stderr


def test_nitro_checker_rejects_cpu_audit_interpreter() -> None:
    checker = ROOT / "src/tools/check_nitro_runtime.py"
    # The suite itself may deliberately run inside the Nitro venv. Its base
    # interpreter is outside that venv and therefore exercises the boundary
    # without making this test depend on the pytest launcher's environment.
    cpu_interpreter = Path(sys._base_executable)
    assert cpu_interpreter.resolve() != (ROOT / ".venv_nitro/Scripts/python.exe").resolve()
    result = subprocess.run(
        [str(cpu_interpreter), str(checker), "--root", str(ROOT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "python:wrong_interpreter" in result.stdout
