"""F51 Grounded Extractor — Codex as project microscope, not truth source.

Codex may only extract from owned F51 artifacts. Every extraction must include
evidence: source file, function/class, commit, test reference.

Epistemic levels:
    observed   = directly present in source evidence → CAN train
    inferred   = derived from project evidence → lower weight
    hypothesis = plausible but unverified → CANNOT train
    approved   = passed firewall + verification → CAN train
    rejected   = failed audit → dead
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class EpistemicLevel(str, Enum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    HYPOTHESIS = "hypothesis"
    APPROVED = "approved"
    REJECTED = "rejected"


TRAINABLE_LEVELS = frozenset({EpistemicLevel.OBSERVED, EpistemicLevel.APPROVED})


@dataclass(frozen=True)
class ExtractionEvidence:
    source_file: str
    functions_or_classes: list[str] = field(default_factory=list)
    commit_or_path: str = ""
    test_reference: str = ""
    log_reference: str = ""
    confidence: float = 1.0


@dataclass
class GroundedExtraction:
    id: str
    content: str
    epistemic_level: EpistemicLevel
    evidence: ExtractionEvidence
    topic: str = ""
    status: str = "candidate"

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "epistemic_level": self.epistemic_level.value,
            "evidence": {
                "source_file": self.evidence.source_file,
                "functions_or_classes": self.evidence.functions_or_classes,
                "commit_or_path": self.evidence.commit_or_path,
                "test_reference": self.evidence.test_reference,
                "log_reference": self.evidence.log_reference,
                "confidence": self.evidence.confidence,
            },
            "topic": self.topic,
            "status": self.status,
        }


class GroundedExtractor:
    """Extracts project-grounded knowledge from F51 artifacts.

    This class models what Codex SHOULD do: read F51 source files
    and produce structured extractions with explicit evidence.
    It does NOT call an external API — it provides the contract
    that any extractor (Codex or Darwin) must satisfy.
    """

    EXTRACTION_PROMPTS = {
        "module_contract": "Describe the public API, inputs, outputs, and invariants.",
        "error_fix_lesson": "What was the error, root cause, fix, and lesson?",
        "architecture_note": "What architectural decision does this code embody?",
        "test_behavior": "What behavior does this test verify?",
        "project_qa": "Question about this project artifact.",
        "decision_record": "What decision was made and why?",
    }

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root)

    def extract_from_file(
        self,
        file_path: str | Path,
        *,
        extraction_type: str,
        functions_or_classes: list[str] | None = None,
        commit: str = "",
        test_ref: str = "",
    ) -> GroundedExtraction:
        """Simulate grounded extraction from a project file.

        In production, this would call Codex API with strict system prompt
        requiring evidence. For now, it reads the file and produces a
        structured extraction from actual content.
        """
        from hashlib import sha256
        from datetime import datetime, timezone

        full_path = self.project_root / file_path
        if not full_path.exists():
            return GroundedExtraction(
                id="",
                content="",
                epistemic_level=EpistemicLevel.REJECTED,
                evidence=ExtractionEvidence(source_file=str(file_path)),
                topic=extraction_type,
            )

        source_content = full_path.read_text(encoding="utf-8", errors="replace")
        extraction_id = sha256(
            f"{file_path}:{extraction_type}:{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        # Produce a grounded extraction from real file content
        content = self._build_extraction(
            file_path=full_path,
            source_content=source_content,
            extraction_type=extraction_type,
            functions_or_classes=functions_or_classes or [],
        )

        return GroundedExtraction(
            id=f"ext_{extraction_id}",
            content=content,
            epistemic_level=EpistemicLevel.OBSERVED,
            evidence=ExtractionEvidence(
                source_file=str(file_path),
                functions_or_classes=functions_or_classes or [],
                commit_or_path=commit or str(file_path),
                test_reference=test_ref,
                confidence=1.0,
            ),
            topic=extraction_type,
            status="candidate",
        )

    def _build_extraction(
        self,
        file_path: Path,
        source_content: str,
        extraction_type: str,
        functions_or_classes: list[str],
    ) -> str:
        """Build extraction text from real file evidence."""
        lines = source_content.splitlines()
        line_count = len(lines)
        has_imports = any(line.strip().startswith(("import ", "from ")) for line in lines)
        has_classes = any(line.strip().startswith("class ") for line in lines)
        has_functions = any(
            line.strip().startswith("def ") for line in lines
        )

        parts = [
            f"# {extraction_type.replace('_', ' ').title()}",
            f"# Source: {file_path}",
            f"# Lines: {line_count}",
        ]

        if extraction_type == "module_contract":
            parts.append(f"\n## Module: {file_path.name}")
            parts.append(f"This module contains {line_count} lines of Python.")
            if has_imports:
                parts.append("It imports dependencies from other modules.")
            if has_classes:
                parts.append("It defines classes.")
            if has_functions:
                parts.append("It defines functions.")
            if functions_or_classes:
                parts.append(f"\nPublic API: {', '.join(functions_or_classes)}")

        elif extraction_type == "architecture_note":
            parts.append(f"\n## Architecture Note: {file_path.name}")
            parts.append(f"This file is part of the F51 Darwin-SSD codebase.")
            if has_imports:
                imports = [l.strip() for l in lines if l.strip().startswith(("import ", "from "))]
                parts.append(f"\nDependencies: {', '.join(imports[:10])}")
            parts.append(f"\nThe module serves as a component in the neural organism architecture.")

        elif extraction_type == "test_behavior":
            parts.append(f"\n## Test Coverage: {file_path.name}")
            parts.append("This test verifies correct behavior of F51 Darwin-SSD components.")

        elif extraction_type == "decision_record":
            parts.append(f"\n## Decision: {file_path.name}")
            parts.append(f"Implementation choice captured in {line_count} lines.")

        parts.append(f"\n---")
        parts.append(f"Epistemic level: observed (extracted from real source file)")
        parts.append(f"Evidence: {file_path}")

        return "\n".join(parts)

    def can_train(self, extraction: GroundedExtraction) -> bool:
        return extraction.epistemic_level in TRAINABLE_LEVELS

    def extract_project_knowledge(
        self,
        source_dir: str | Path,
        *,
        extraction_types: list[str] | None = None,
    ) -> list[GroundedExtraction]:
        """Extract knowledge from all Python files in a project directory."""
        source_path = self.project_root / source_dir
        extractions: list[GroundedExtraction] = []
        types = extraction_types or ["module_contract"]

        for py_file in sorted(source_path.rglob("*.py")):
            if py_file.name.startswith("__"):
                continue
            for etype in types:
                extraction = self.extract_from_file(
                    py_file.relative_to(self.project_root),
                    extraction_type=etype,
                )
                if extraction.epistemic_level != EpistemicLevel.REJECTED:
                    extractions.append(extraction)

        return extractions
