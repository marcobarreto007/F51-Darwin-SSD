from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "governance" / "audit" / "policy" / "operational-surface.json"
EXECUTABLE_SUFFIXES = {".py", ".ps1", ".bat", ".sh"}
EXPECTED_SUPPORTED = {
    "src/scripts/start_overnight_16b.ps1",
    "src/scripts/start_100m_65b.ps1",
    "src/scripts/start_100m_auto.ps1",
    "src/scripts/darwin_organism.py",
    "src/scripts/serve_davi.py",
    "src/scripts/inspect_organism_checkpoint.py",
    "src/scripts/darwin_inventory.py",
    "src/scripts/ingest_pipeline.py",
    "src/scripts/bob.ps1",
    "src/scripts/start_smol_darwin_transplant.ps1",
}


def _policy() -> dict:
    return json.loads(POLICY.read_text(encoding="utf-8"))


def _ignored() -> set[str]:
    """Paths git is told to ignore.

    The policy manifest is a tracked artifact describing the tracked surface,
    so it must not be forced to declare files that only exist in one working
    copy. Local-only trees such as governance/archive/cloud-local/ are ignored on disk and
    absent from any clean checkout; scanning the filesystem blind makes this
    suite pass or fail depending on whose machine it runs on.
    """
    if not (ROOT / ".git").exists():
        return set()
    result = subprocess.run(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "-z"],
        cwd=ROOT,
        capture_output=True,
    )
    if result.returncode != 0:
        return set()
    return {path for path in result.stdout.decode("utf-8").split("\0") if path}


def _executables(*subtrees: str, skip_init: bool) -> set[str]:
    ignored = _ignored()
    found = set()
    for subtree in subtrees:
        for path in (ROOT / subtree).rglob("*"):
            if not path.is_file() or path.suffix.lower() not in EXECUTABLE_SUFFIXES:
                continue
            if skip_init and path.name == "__init__.py":
                continue
            relative = path.relative_to(ROOT).as_posix()
            if relative not in ignored:
                found.add(relative)
    return found


def test_exact_supported_entrypoint_contract() -> None:
    policy = _policy()
    supported = set(policy["supported_entrypoints"])
    assert supported == EXPECTED_SUPPORTED
    assert all((ROOT / path).is_file() for path in supported)
    forbidden = ("cloud", "rental", "sync", "pull", "2.5b", "_5b")
    assert not [path for path in supported if any(word in path.lower() for word in forbidden)]


def test_internal_gold_target_owns_exact_script_surface() -> None:
    policy = _policy()
    boundary = policy["gold_boundaries"]
    assert set(boundary["script_allowed_files"]) == EXPECTED_SUPPORTED | {
        "src/scripts/README.md",
        "src/scripts/__init__.py",
    }
    assert boundary["active_python_roots"] == [
        "src/f51_darwin",
        "src/scripts",
        "src/tools",
        "research",
    ]
    # Nomes de import, nao caminhos. check_architecture_boundaries.py compara
    # esta lista com `top = name.split(".", 1)[0]`, que nunca contem "/". A
    # migracao prefixou "src/" aqui e na politica, e como o teste espelhava o
    # valor em vez da intencao, o portao ficou morto sem falhar: "src/scripts"
    # nunca casa com nenhum import, entao a deteccao de forbidden_dependency
    # estava desligada para scripts e tools.
    assert boundary["forbidden_package_import_roots"] == [
        "research",
        "scripts",
        "tools",
    ]
    assert boundary["canonical_config"] == {
        "path": "src/configs/darwin_x_100m.yaml",
        "model_name": "F51-Darwin-X-100M",
        "forbidden_default_labels": ["F51-Darwin-X-4B", "F51-Darwin-X-1.2B"],
        "entrypoint_config_defaults": {
            "src/scripts/darwin_organism.py": "src/configs/darwin_x_100m.yaml",
            "src/scripts/serve_davi.py": "src/configs/darwin_x_100m.yaml",
        },
    }
    assert boundary["model_family"] == {
        "100m": "src/configs/darwin_x_100m.yaml",
        "600m": "src/configs/darwin_x_600m.yaml",
        "1.6b": "src/configs/darwin_x_1.6b_nitro.yaml",
        "1.6b-smol-transplant": "src/configs/darwin_x_1.6b_smol_transplant.yaml",
        "1.7b-smol-exact": "src/configs/darwin_x_1.7b_smol_exact.yaml",
        "1.7b-smol-dense": "src/configs/darwin_x_1.7b_smol_dense.yaml",
    }


def test_every_operational_executable_is_classified() -> None:
    policy = _policy()
    rows = policy["classifications"]
    classified = {row["path"] for row in rows}
    executables = _executables("src/scripts", "src/tools", "research", skip_init=True)
    assert classified == executables
    assert all(row["classification"] in {"supported", "maintenance", "research"} for row in rows)
    expected_roots = {
        "supported": "src/scripts/",
        "maintenance": ("src/tools/", "src/scripts/"),
        "research": "research/",
    }
    assert all(row["path"].startswith(expected_roots[row["classification"]]) for row in rows)
    supported_rows = {row["path"] for row in rows if row["classification"] == "supported"}
    assert supported_rows == EXPECTED_SUPPORTED


def test_no_executable_scratch_remains_at_repository_root() -> None:
    leftovers = sorted(
        path.name for path in ROOT.iterdir() if path.is_file() and path.suffix.lower() in EXECUTABLE_SUFFIXES
    )
    assert leftovers == []


def test_every_tracked_style_root_artifact_is_classified() -> None:
    policy = _policy()
    classified = {row["path"] for row in policy["root_artifacts"]}
    if (ROOT / ".git").exists():
        tracked = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
        ).stdout.decode("utf-8").split("\0")
        root_files = {
            path for path in tracked if path and "/" not in path and "\\" not in path
        }
    else:
        root_files = {path.name for path in ROOT.iterdir() if path.is_file()}
    assert classified == root_files
    assert all(row["classification"] in {"governance", "documentation", "metadata"} for row in policy["root_artifacts"])


def test_archived_executables_are_declared_non_operational() -> None:
    policy = _policy()
    archived = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "governance/archive").rglob("*")
        if path.is_file() and path.suffix.lower() in EXECUTABLE_SUFFIXES
    }
    declared = {row["path"] for row in policy["archived_executables"]}
    assert declared == archived
    assert all(row["classification"] == "historical_non_operational" for row in policy["archived_executables"])
