from pathlib import Path

from scripts.darwin_inventory import visible_project_files


def test_visible_project_files_excludes_generated_and_internal_trees(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run.py").write_text("pass", encoding="utf-8")
    for hidden in (".git", ".venv_clean", "__pycache__", ".playwright-mcp"):
        directory = tmp_path / hidden
        directory.mkdir()
        (directory / "noise.txt").write_text("noise", encoding="utf-8")

    assert visible_project_files(tmp_path) == [str(Path("scripts") / "run.py")]
