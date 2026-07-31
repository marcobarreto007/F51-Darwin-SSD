#!/usr/bin/env python3
"""Check the small canonical document set for stale or incomplete local truth."""

from __future__ import annotations

import argparse
import json
import re
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
REQUIRED_CLAIMS = (
    "workspace",
    "feast_v2",
    "organism_cycle_071.pt",
    "src/scripts/start_overnight_16b.ps1",
    "src/scripts/darwin_organism.py",
    "src/scripts/serve_davi.py",
    "src/scripts/inspect_organism_checkpoint.py",
    "src/scripts/darwin_inventory.py",
    "src/scripts/ingest_pipeline.py",
    "rollback",
)
STALE = {
    "sibling_dataset": re.compile(r"F51-Dataset-Organizado", re.I),
    "feast_v1": re.compile(r"00_CORPUS_PRINCIPAL_tokens_feast\.bin", re.I),
    "cycle_068": re.compile(r"(?:cycle|ciclo)[_\s-]*0?68\b", re.I),
    "active_2_5b_or_5b": re.compile(r"(?:darwin_x_2\.5b|darwin_x_5b|\b2\.5B\b|\b5B\b)", re.I),
    "remote_runtime": re.compile(r"(?:ssh\d*\.vast\.ai|active cloud runtime|remote gpu)", re.I),
}


def check_canonical(root: Path) -> list[dict[str, object]]:
    root = root.resolve()
    findings: list[dict[str, object]] = []
    combined: list[str] = []
    for relative in CANONICAL:
        path = root / relative
        if not path.is_file():
            findings.append({"type": "missing_canonical_document", "path": relative})
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        combined.append(text)
        for claim, pattern in STALE.items():
            for match in pattern.finditer(text):
                findings.append(
                    {
                        "type": "stale_claim",
                        "claim": claim,
                        "path": relative,
                        "line": text.count("\n", 0, match.start()) + 1,
                    }
                )
    joined = "\n".join(combined)
    for required in REQUIRED_CLAIMS:
        if required.lower() not in joined.lower():
            findings.append({"type": "missing_claim", "claim": required})
    agents = root / "AGENTS.md"
    claude = root / "CLAUDE.md"
    if agents.is_file() and claude.is_file() and agents.read_bytes() != claude.read_bytes():
        findings.append({"type": "authority_mismatch", "paths": ["AGENTS.md", "CLAUDE.md"]})
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    findings = check_canonical(args.root)
    print(json.dumps({"status": "pass" if not findings else "fail", "findings": findings}, indent=2))
    return 0 if not findings else 2


if __name__ == "__main__":
    sys.exit(main())
