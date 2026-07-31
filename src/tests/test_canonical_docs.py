from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANONICAL = (
    "README.md",
    "AGENTS.md",
    "CLAUDE.md",
    "governance/docs/CANONICAL_MAP.md",
    "governance/docs/operacao/STATUS_ATUAL.md",
    "governance/docs/operacao/OPERACAO_SEGURA.md",
    "governance/docs/arquitetura/IMPLEMENTACAO_ATUAL.md",
    "governance/audit/README.md",
)
SUPPORTED = (
    "src/scripts/start_overnight_16b.ps1",
    "src/scripts/start_100m_65b.ps1",
    "src/scripts/darwin_organism.py",
    "src/scripts/serve_davi.py",
    "src/scripts/inspect_organism_checkpoint.py",
    "src/scripts/darwin_inventory.py",
    "src/scripts/ingest_pipeline.py",
)


def _combined_canonical_text() -> str:
    return "\n".join((ROOT / path).read_text(encoding="utf-8") for path in CANONICAL)


def test_canonical_documents_exist_and_publish_single_root_truth() -> None:
    for relative in CANONICAL:
        assert (ROOT / relative).is_file(), relative
    combined = _combined_canonical_text()
    for claim in (
        "workspace/",
        "feast_v2",
        "organism_cycle_071.pt",
        "cycle 71",
        "step 40751",
        "rollback",
        "hierarquia de prova",
        "local-only",
    ):
        assert claim.casefold() in combined.casefold(), claim
    for entrypoint in SUPPORTED:
        assert entrypoint in combined


def test_agent_authority_mirror_is_byte_identical() -> None:
    assert (ROOT / "AGENTS.md").read_bytes() == (ROOT / "CLAUDE.md").read_bytes()


def test_canonical_checker_and_links_pass() -> None:
    for checker in ("check_canonical_docs.py", "check_docs_links.py"):
        result = subprocess.run(
            [sys.executable, str(ROOT / "src" / "tools" / checker), "--root", str(ROOT)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{checker}:\n{result.stdout}\n{result.stderr}"


def test_history_has_one_index_and_is_explicitly_non_authoritative() -> None:
    index = (ROOT / "governance/docs/_historico/INDEX.md").read_text(encoding="utf-8")
    for path in (
        "governance/archive/historical-docs/",
        "governance/archive/cloud-local/",
        "governance/archive/agent-bus/",
        "governance/archive/legacy-state/f51/",
        "governance/docs/superpowers/",
    ):
        assert path in index
    assert "nao autorit" in index.casefold()


def test_duplicate_policy_declares_only_the_required_authority_mirror() -> None:
    policy = json.loads((ROOT / "governance/audit/policy/duplicates.json").read_text(encoding="utf-8"))
    exceptions = policy["exact_duplicate_exceptions"]
    assert len(exceptions) == 1
    assert exceptions[0]["paths"] == ["AGENTS.md", "CLAUDE.md"]
    assert exceptions[0]["category"] == "required_authority_mirror"
    assert len(exceptions[0]["justification"]) >= 40
