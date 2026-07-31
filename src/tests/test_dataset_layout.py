from pathlib import Path

import pytest

from f51_darwin.dataset_layout import (
    DATASET_ROOT_ENV,
    CHECKPOINTS_RELATIVE,
    DARWIN_X_CHECKPOINTS_RELATIVE,
    FEAST_TOKEN_RELATIVE,
    WorkspacePaths,
    resolve_dataset_path,
    resolve_dataset_root,
    resolve_checkpoints_root,
    resolve_corpus_workspace_root,
    resolve_darwin_x_checkpoints_root,
    resolve_feast_token_bin,
)
from f51_darwin.data_factory import DataFactoryPaths


def test_workspace_defaults_inside_project(tmp_path: Path) -> None:
    project = tmp_path / "F51-Darwin-SSD"
    project.mkdir()

    assert resolve_dataset_root(project, use_environment=False) == project / "workspace"


def test_environment_override_requires_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    external = tmp_path / "external-dataset"
    external.mkdir()
    monkeypatch.setenv(DATASET_ROOT_ENV, str(external))

    assert resolve_dataset_root(project, use_environment=False) == project / "workspace"
    assert resolve_dataset_root(project, use_environment=True, require=True) == external
    assert (
        resolve_dataset_path(project, "tokens.bin", use_environment=True)
        == external / "tokens.bin"
    )


def test_feast_token_resolution_requires_external_artifact(tmp_path: Path) -> None:
    project = tmp_path / "F51-Darwin-SSD"
    project.mkdir()
    workspace = project / "workspace"
    feast = workspace / FEAST_TOKEN_RELATIVE
    feast.parent.mkdir(parents=True)
    feast.write_bytes((51).to_bytes(4, "little", signed=True))

    assert resolve_feast_token_bin(project, require=True) == feast


def test_required_dataset_root_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=DATASET_ROOT_ENV):
        resolve_dataset_root(tmp_path / "project", require=True)


def test_checkpoint_paths_are_external(tmp_path: Path) -> None:
    project = tmp_path / "F51-Darwin-SSD"
    project.mkdir()
    workspace = project / "workspace"

    assert resolve_checkpoints_root(project) == workspace / CHECKPOINTS_RELATIVE
    assert resolve_darwin_x_checkpoints_root(project) == workspace / DARWIN_X_CHECKPOINTS_RELATIVE
    assert resolve_corpus_workspace_root(project) == workspace / "02_CORPUS"


def test_factory_paths_are_external_and_do_not_nest_repo_data(tmp_path: Path) -> None:
    project = tmp_path / "F51-Darwin-SSD"
    project.mkdir()
    corpus = project / "workspace" / "02_CORPUS"
    corpus.mkdir(parents=True)

    paths = DataFactoryPaths.from_project(project)

    assert paths.root == corpus
    assert paths.candidates == corpus / "generated" / "candidates"
    assert paths.corpus == corpus / "approved"
    assert paths.ledger == corpus / "ledger"


def test_workspace_paths_exposes_canonical_contract(tmp_path: Path) -> None:
    project = tmp_path / "F51-Darwin-SSD"
    project.mkdir()

    paths = WorkspacePaths.from_project(project)

    assert paths.root == project / "workspace"
    assert paths.feast_token_bin == paths.root / FEAST_TOKEN_RELATIVE
    assert paths.latest_pointer == paths.checkpoints / "organism_latest.json"
    assert paths.readiness_manifest == paths.manifests / "overnight_16b_readiness.json"
    assert paths.runs == paths.runtime / "runs"
    assert paths.logs == paths.runtime / "logs"
    assert paths.evaluations == paths.runtime / "evaluations"
