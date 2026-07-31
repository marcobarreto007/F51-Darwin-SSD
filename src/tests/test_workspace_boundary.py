from pathlib import Path

import pytest

from f51_darwin.dataset_layout import resolve_dataset_path


@pytest.mark.parametrize("unsafe", ["../escape", "C:/escape", "\\\\server\\share"])
def test_dataset_relative_paths_reject_escape(tmp_path: Path, unsafe: str) -> None:
    with pytest.raises(ValueError, match="workspace_boundary"):
        resolve_dataset_path(tmp_path, unsafe)


def test_dataset_relative_path_stays_inside_workspace(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()

    path = resolve_dataset_path(project, "02_CORPUS/approved/item.txt")

    assert path == project / "workspace" / "02_CORPUS" / "approved" / "item.txt"
