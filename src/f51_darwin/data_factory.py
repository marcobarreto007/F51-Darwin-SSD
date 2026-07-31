from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from f51_darwin.data_firewall import DataFirewall, FirewallConfig, FirewallDecision
from f51_darwin.dataset_layout import WorkspacePaths
from f51_darwin.dataset_states import DatasetStatus, SourceType, can_transition
from f51_darwin.provenance import DatasetRecord, ProvenanceLedger, content_hash


BLOCKED_TRAINING_PREFIXES = (
    "data/generated/candidates",
    "data/generated/quarantine",
    "data/generated/rejected",
)

EXTERNAL_FACTORY_PATHS = {
    "candidates": "generated/candidates",
    "quarantine": "generated/quarantine",
    "rejected": "generated/rejected",
    "approved": "approved",
    "corpus": "approved",
    "ledger": "ledger",
}


@dataclass(frozen=True)
class DataFactoryPaths:
    root: Path
    candidates: Path
    quarantine: Path
    rejected: Path
    approved: Path
    corpus: Path
    ledger: Path

    @classmethod
    def from_root(cls, root: Path, raw: dict[str, Any] | None = None) -> "DataFactoryPaths":
        raw = raw or {}
        paths = raw.get("paths", raw)
        return cls(
            root=root,
            candidates=root / paths.get("candidates", "data/generated/candidates"),
            quarantine=root / paths.get("quarantine", "data/generated/quarantine"),
            rejected=root / paths.get("rejected", "data/generated/rejected"),
            approved=root / paths.get("approved", "data/approved"),
            corpus=root / paths.get("corpus", "data/corpus"),
            ledger=root / paths.get("ledger", "data/ledger"),
        )

    @classmethod
    def from_project(
        cls,
        project_root: Path,
        raw: dict[str, Any] | None = None,
        *,
        require_external: bool = True,
    ) -> "DataFactoryPaths":
        """Resolve the production factory under the external dataset workspace.

        ``from_root`` remains available for tests and explicitly isolated tools.
        Production entrypoints must use this constructor so generated data never
        becomes physical repository content.
        """
        workspace = WorkspacePaths.from_project(
            project_root, require=require_external
        ).corpus
        config = raw if raw is not None else {"paths": EXTERNAL_FACTORY_PATHS}
        return cls.from_root(workspace, config)

    def ensure_layout(self) -> None:
        for path in (
            self.candidates,
            self.quarantine,
            self.rejected,
            self.approved,
            self.corpus,
            self.ledger,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_factory_config(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def is_blocked_training_path(path: Path, project_root: Path) -> bool:
    try:
        relative = path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return False
    return any(
        relative == blocked or relative.startswith(f"{blocked}/")
        for blocked in BLOCKED_TRAINING_PREFIXES
    )


def assert_training_corpus_allowed(path: Path, project_root: Path, *, debug_candidates: bool = False) -> None:
    if debug_candidates:
        return
    if is_blocked_training_path(path, project_root):
        raise PermissionError(
            "Training cannot read generated candidate/quarantine/rejected data. "
            "Use data/approved or data/corpus after explicit promotion."
        )


def candidate_file(paths: DataFactoryPaths, record_id: str) -> Path:
    return paths.candidates / f"{record_id}.json"


def status_file(paths: DataFactoryPaths, status: DatasetStatus, record_id: str) -> Path:
    mapping = {
        DatasetStatus.CANDIDATE: paths.candidates,
        DatasetStatus.QUARANTINE: paths.quarantine,
        DatasetStatus.APPROVED: paths.approved,
        DatasetStatus.REJECTED: paths.rejected,
        DatasetStatus.RETIRED: paths.rejected,
    }
    return mapping[status] / f"{record_id}.json"


def write_item(path: Path, record: DatasetRecord, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"record": record.to_dict(), "text": text}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_item(path: Path) -> tuple[DatasetRecord, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return DatasetRecord.from_dict(payload["record"]), str(payload["text"])


def move_item(record: DatasetRecord, text: str, paths: DataFactoryPaths, target: DatasetStatus) -> Path:
    current_status = record.status
    if current_status != target and not can_transition(current_status, target):
        raise ValueError(f"Invalid transition {current_status.value} -> {target.value}")
    stored = DatasetRecord.from_dict(record.to_dict())
    stored.status = target
    destination = status_file(paths, target, stored.id)
    write_item(destination, stored, text)
    for status in DatasetStatus:
        source = status_file(paths, status, record.id)
        if source.exists() and source != destination:
            source.unlink()
    return destination


class DataFactory:
    def __init__(self, paths: DataFactoryPaths, firewall: DataFirewall | None = None) -> None:
        self.paths = paths
        self.paths.ensure_layout()
        self.firewall = firewall or DataFirewall()
        self.ledger = ProvenanceLedger(
            paths.ledger,
            minhash_threshold=self.firewall.config.minhash_threshold,
        )

    def register_candidate(
        self,
        *,
        text: str,
        source_type: SourceType,
        source_path: str,
        generator_model: str | None = None,
        generator_checkpoint: str | None = None,
        prompt: str | None = None,
        parent_ids: list[str] | None = None,
        dataset_version: str = "v0",
    ) -> DatasetRecord:
        record = ProvenanceLedger.new_record(
            text=text,
            source_type=source_type,
            source_path=source_path,
            generator_model=generator_model,
            generator_checkpoint=generator_checkpoint,
            prompt=prompt,
            parent_ids=parent_ids,
            dataset_version=dataset_version,
            status=DatasetStatus.CANDIDATE,
        )
        write_item(candidate_file(self.paths, record.id), record, text)
        self.ledger.append(record)
        return record

    def import_real_file(
        self,
        file_path: Path,
        *,
        dataset_version: str = "v0",
        skip_duplicates: bool = True,
    ) -> DatasetRecord | None:
        text = file_path.read_text(encoding="utf-8", errors="replace").strip()
        if skip_duplicates:
            existing = self.ledger.find_by_hash(content_hash(text))
            if existing is not None:
                return existing
        return self.register_candidate(
            text=text,
            source_type=SourceType.REAL,
            source_path=str(file_path),
            dataset_version=dataset_version,
        )

    def register_synthetic(
        self,
        *,
        text: str,
        generator_model: str,
        generator_checkpoint: str,
        prompt: str | None = None,
        parent_ids: list[str] | None = None,
        dataset_version: str = "v0",
    ) -> DatasetRecord:
        return self.register_candidate(
            text=text,
            source_type=SourceType.SYNTHETIC,
            source_path="synthetic://generated",
            generator_model=generator_model,
            generator_checkpoint=generator_checkpoint,
            prompt=prompt,
            parent_ids=parent_ids,
            dataset_version=dataset_version,
        )

    def audit_candidate(self, record: DatasetRecord, text: str) -> FirewallDecision:
        # Near-duplicate gate (MinHash LSH). SHA-256 still handles exact
        # duplicates; this layer catches paraphrases and lightly-edited
        # re-ingestions before the expensive n-gram comparison runs.
        near = self.ledger.find_near_duplicate(record)
        if near is not None:
            other, jaccard = near
            scores = {"minhash_jaccard": round(jaccard, 4)}
            decision = FirewallDecision(
                DatasetStatus.REJECTED,
                f"near-duplicate (jaccard {jaccard:.2f}) of {other.id}",
                scores,
            )
            updated = DatasetRecord.from_dict(record.to_dict())
            updated.scores = decision.scores
            updated.rejection_reason = decision.reason
            move_item(updated, text, self.paths, decision.status)
            updated.status = decision.status
            self.ledger.update(updated)
            return decision

        known_texts = self._collect_known_texts(exclude_id=record.id)
        decision = self.firewall.evaluate(
            record,
            text=text,
            ledger=self.ledger,
            corpus_texts=known_texts,
        )
        updated = DatasetRecord.from_dict(record.to_dict())
        updated.scores = decision.scores
        if decision.status == DatasetStatus.APPROVED:
            updated.approval_reason = decision.reason
        elif decision.status == DatasetStatus.REJECTED:
            updated.rejection_reason = decision.reason
        elif decision.status == DatasetStatus.QUARANTINE:
            updated.rejection_reason = decision.reason
        move_item(updated, text, self.paths, decision.status)
        updated.status = decision.status
        self.ledger.update(updated)
        return decision

    def promote_approved_to_corpus(self, *, build_corpus: bool = True) -> list[Path]:
        exported: list[Path] = []
        for path in sorted(self.paths.approved.glob("*.json")):
            record, text = read_item(path)
            if record.status != DatasetStatus.APPROVED:
                continue
            export_path = self.paths.approved / f"{record.id}.txt"
            export_path.write_text(text + "\n", encoding="utf-8")
            exported.append(export_path)
            if build_corpus:
                corpus_path = self.paths.corpus / f"{record.id}.txt"
                corpus_path.write_text(text + "\n", encoding="utf-8")
                exported.append(corpus_path)
        return exported

    def _collect_known_texts(self, *, exclude_id: str | None = None, max_corpus_mb: float = 500.0) -> list[str]:
        """Corpus texts for ngram contamination — approved + training corpus only.

        For corpus/: samples files up to max_corpus_mb total to avoid OOM.
        """
        texts: list[str] = []
        corpus_bytes_sampled = 0

        for directory in (self.paths.corpus, self.paths.approved):
            # JSONs: validate schema before reading (skip metadata files)
            for path in directory.glob("*.json"):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(payload, dict) or "record" not in payload:
                        continue  # Skip non-record JSONs (metadata, etc.)
                    record, text = DatasetRecord.from_dict(payload["record"]), str(payload.get("text", ""))
                    if exclude_id and record.id == exclude_id:
                        continue
                    texts.append(text)
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue  # Skip malformed JSONs

            # TXTs: sample from corpus/ to avoid OOM on 67GB dataset
            for path in directory.glob("*.txt"):
                if path.name == "README.txt":
                    continue
                if directory == self.paths.corpus and corpus_bytes_sampled >= max_corpus_mb * 1024 * 1024:
                    continue  # Sample limit reached for corpus/
                texts.append(path.read_text(encoding="utf-8", errors="replace"))
                corpus_bytes_sampled += path.stat().st_size

        return texts

    def list_candidates(self) -> list[tuple[DatasetRecord, str]]:
        items: list[tuple[DatasetRecord, str]] = []
        for path in sorted(self.paths.candidates.glob("*.json")):
            items.append(read_item(path))
        return items


def approve_quarantine_item(
    factory: DataFactory,
    record_id: str,
    *,
    reason: str,
) -> DatasetRecord:
    source = status_file(factory.paths, DatasetStatus.QUARANTINE, record_id)
    if not source.exists():
        raise FileNotFoundError(f"Quarantine item not found: {record_id}")
    record, text = read_item(source)
    if not can_transition(record.status, DatasetStatus.APPROVED):
        raise ValueError(f"Cannot approve item in status {record.status.value}")
    updated = DatasetRecord.from_dict(record.to_dict())
    updated.approval_reason = reason
    move_item(updated, text, factory.paths, DatasetStatus.APPROVED)
    updated.status = DatasetStatus.APPROVED
    factory.ledger.update(updated)
    return updated
