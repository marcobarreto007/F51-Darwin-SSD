from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "src/tools/check_duplicates.py"


def _write_policy(root: Path, *, exceptions: list[dict[str, object]] | None = None) -> None:
    path = root / "governance/audit/policy/duplicates.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "tracked_source_tree",
                "historical_prefixes": ["governance/archive/", "governance/docs/_historico/"],
                "canonical_documents": ["README.md", "governance/docs/operacao/STATUS_ATUAL.md"],
                "authority_claims": [
                    {
                        "id": "current_status",
                        "marker": "<!-- authority: current-status -->",
                        "path": "governance/docs/operacao/STATUS_ATUAL.md",
                    }
                ],
                "exact_duplicate_exceptions": exceptions or [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), "--root", str(root)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_checker_rejects_unapproved_exact_duplicate(tmp_path: Path) -> None:
    (tmp_path / "governance/docs/operacao").mkdir(parents=True)
    (tmp_path / "README.md").write_text("same\n", encoding="utf-8")
    (tmp_path / "governance/docs/operacao/STATUS_ATUAL.md").write_text(
        "same\n<!-- authority: current-status -->\n", encoding="utf-8"
    )
    (tmp_path / "one.txt").write_text("duplicate\n", encoding="utf-8")
    (tmp_path / "two.txt").write_text("duplicate\n", encoding="utf-8")
    _write_policy(tmp_path)
    result = _run(tmp_path)
    assert result.returncode != 0
    assert "unapproved_exact_duplicate" in result.stdout


def test_checker_rejects_duplicate_authority_claim(tmp_path: Path) -> None:
    marker = "<!-- authority: current-status -->"
    (tmp_path / "governance/docs/operacao").mkdir(parents=True)
    (tmp_path / "README.md").write_text(f"orientation\n{marker}\n", encoding="utf-8")
    (tmp_path / "governance/docs/operacao/STATUS_ATUAL.md").write_text(f"status\n{marker}\n", encoding="utf-8")
    _write_policy(tmp_path)
    result = _run(tmp_path)
    assert result.returncode != 0
    assert "duplicate_authority_claim" in result.stdout


def test_checker_requires_exception_hash_and_technical_justification(tmp_path: Path) -> None:
    (tmp_path / "governance/docs/operacao").mkdir(parents=True)
    marker = "<!-- authority: current-status -->"
    (tmp_path / "README.md").write_text("orientation\n", encoding="utf-8")
    (tmp_path / "governance/docs/operacao/STATUS_ATUAL.md").write_text(f"status\n{marker}\n", encoding="utf-8")
    payload = b"same authority bytes\n"
    (tmp_path / "AGENTS.md").write_bytes(payload)
    (tmp_path / "CLAUDE.md").write_bytes(payload)
    _write_policy(
        tmp_path,
        exceptions=[
            {
                "paths": ["AGENTS.md", "CLAUDE.md"],
                "sha256": hashlib.sha256(payload).hexdigest(),
                "category": "required_authority_mirror",
                "justification": "short",
            }
        ],
    )
    result = _run(tmp_path)
    assert result.returncode != 0
    assert "invalid_duplicate_exception" in result.stdout


def test_checker_accepts_required_mirror_across_windows_line_endings(tmp_path: Path) -> None:
    (tmp_path / "governance/docs/operacao").mkdir(parents=True)
    marker = "<!-- authority: current-status -->"
    (tmp_path / "README.md").write_text("orientation\n", encoding="utf-8")
    (tmp_path / "governance/docs/operacao/STATUS_ATUAL.md").write_text(f"status\n{marker}\n", encoding="utf-8")
    canonical = b"same authority bytes\n"
    windows = canonical.replace(b"\n", b"\r\n")
    (tmp_path / "AGENTS.md").write_bytes(windows)
    (tmp_path / "CLAUDE.md").write_bytes(windows)
    _write_policy(
        tmp_path,
        exceptions=[
            {
                "paths": ["AGENTS.md", "CLAUDE.md"],
                "content_mode": "utf8_text_lf",
                "sha256": hashlib.sha256(canonical).hexdigest(),
                "category": "required_authority_mirror",
                "justification": "Both agent discovery filenames are required while canonical bytes prevent authority drift.",
            }
        ],
    )
    result = _run(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_repository_has_no_unjustified_duplicate_or_authority_claim() -> None:
    result = _run(ROOT)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
