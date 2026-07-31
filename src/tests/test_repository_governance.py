import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_unsafe_repository_automation_is_not_operational() -> None:
    assert not (ROOT / "src/scripts/_autocommit.ps1").exists()
    operational = [ROOT / "src" / "scripts", ROOT / "src" / "f51_darwin"]
    forbidden = ("git add -A", "--no-verify", "Remove-Item -Recurse")
    hits = []
    for base in operational:
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".py", ".ps1", ".bat", ".sh"}:
                text = path.read_text(encoding="utf-8", errors="ignore")
                hits.extend(
                    (str(path.relative_to(ROOT)), token)
                    for token in forbidden
                    if token in text
                )
    assert hits == []


def test_heavy_workspace_is_ignored_but_audit_policy_is_tracked() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "workspace/" in ignore
    assert "!checkpoints/organism" not in ignore
    assert (ROOT / "governance/audit/policy/repository-governance.md").is_file()


def test_archived_evaluations_preserve_historical_git_blobs_exactly() -> None:
    expected_blobs = {
        "darwin_merged_v4.json": "d7993c900e9912c393ecbb51bb8db30f032bcf7e",
        "darwin_trained_v3.json": "133f9a5017a6a1ed7edae77f673752d9220cd6fa",
        "organism_cycle_235.json": "139109e23832fc56e6859ced0181ed7baf423152",
        "organism_cycle_236.json": "34e716c1573c657e160e72a73965f5c208efb55d",
        "organism_cycle_237.json": "84095c712c61f80e316a47b3f46de7aa2f37913c",
        "organism_cycle_238.json": "c7f503851379a93396741a330b3151ccc09eabd9",
        "organism_cycle_239.json": "bd666a6d93e03418ceaeec1412a8ac8ecf65647b",
        "tournament_summary.json": "918acca6ce3e1fcfdbe94e0776c0f9e501d39be3",
    }
    for name, expected_blob in expected_blobs.items():
        path = ROOT / "governance/archive/historical-evaluations" / name
        observed_blob = subprocess.run(
            ["git", "hash-object", f"--path={path.relative_to(ROOT)}", str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert observed_blob == expected_blob, name
