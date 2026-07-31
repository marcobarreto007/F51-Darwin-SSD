from __future__ import annotations

import hashlib
import json
import tarfile
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

from f51_darwin.corpus_policy import CorpusDecision, evaluate_corpus_source


TEXT_SUFFIXES = {".txt", ".md", ".tex"}


@dataclass(frozen=True)
class SourceMetadata:
    source_url: str = ""
    license_label: str = "unknown"
    domain_hint: str | None = None


@dataclass(frozen=True)
class CorpusBuildRecord:
    relative_path: str
    source_url: str
    license_label: str
    raw_sha256: str
    normalized_sha256: str
    raw_bytes: int
    normalized_chars: int
    decision: str
    reason: str
    tags: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    approved_path: str | None = None


@dataclass(frozen=True)
class CorpusBuildSummary:
    scanned: int
    approved: int
    quarantine: int
    rejected: int
    manifest_path: str
    approved_dir: str | None
    pack_path: str | None


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalize_training_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    lines = [line.rstrip() for line in text.split("\n")]

    normalized: list[str] = []
    blank_seen = False
    for line in lines:
        if line.strip():
            normalized.append(line.strip())
            blank_seen = False
        elif not blank_seen:
            normalized.append("")
            blank_seen = True
    return "\n".join(normalized).strip() + "\n"


def iter_text_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
    )


def load_metadata(metadata_path: Path | None, input_dir: Path) -> dict[str, SourceMetadata]:
    if metadata_path is None or not metadata_path.exists():
        return {}
    metadata: dict[str, SourceMetadata] = {}
    for line_number, line in enumerate(metadata_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        relative = str(payload.get("path", "")).replace("\\", "/")
        if not relative:
            raise ValueError(f"metadata line {line_number} missing path")
        candidate = (input_dir / relative).resolve()
        try:
            normalized_relative = candidate.relative_to(input_dir.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError(f"metadata line {line_number} path escapes input_dir") from exc
        metadata[normalized_relative] = SourceMetadata(
            source_url=str(payload.get("source_url", "")),
            license_label=str(payload.get("license", payload.get("license_label", "unknown"))),
            domain_hint=payload.get("domain_hint"),
        )
    return metadata


def _safe_output_name(relative_path: str, digest: str) -> str:
    path = Path(relative_path)
    stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)
    return f"{stem}_{digest[:12]}.txt"


def build_corpus_pack(
    *,
    input_dir: Path,
    output_dir: Path,
    metadata_path: Path | None = None,
    default_license: str = "unknown",
    source_url_base: str = "file://",
    domain_hint: str | None = None,
    export_approved: bool = True,
    create_pack: bool = True,
) -> CorpusBuildSummary:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    approved_dir = output_dir / "approved" if export_approved else None
    if approved_dir is not None:
        approved_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(metadata_path, input_dir)
    records: list[CorpusBuildRecord] = []
    seen_normalized_hashes: set[str] = set()

    for path in iter_text_files(input_dir):
        relative = path.relative_to(input_dir).as_posix()
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        normalized = normalize_training_text(text)
        raw_hash = sha256_bytes(raw)
        normalized_hash = sha256_bytes(normalized.encode("utf-8"))

        meta = metadata.get(
            relative,
            SourceMetadata(
                source_url=f"{source_url_base.rstrip('/')}/{relative}",
                license_label=default_license,
                domain_hint=domain_hint,
            ),
        )
        result = evaluate_corpus_source(
            text=normalized,
            source_url=meta.source_url,
            license_label=meta.license_label,
            domain_hint=meta.domain_hint,
        )
        decision = result.decision
        reason = result.reason
        if normalized_hash in seen_normalized_hashes:
            decision = CorpusDecision.REJECT
            reason = "duplicate normalized hash inside batch"

        approved_path: str | None = None
        if decision == CorpusDecision.APPROVE and approved_dir is not None:
            output_name = _safe_output_name(relative, normalized_hash)
            destination = approved_dir / output_name
            with destination.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(normalized)
            approved_path = destination.relative_to(output_dir).as_posix()
        seen_normalized_hashes.add(normalized_hash)

        records.append(
            CorpusBuildRecord(
                relative_path=relative,
                source_url=meta.source_url,
                license_label=meta.license_label,
                raw_sha256=raw_hash,
                normalized_sha256=normalized_hash,
                raw_bytes=len(raw),
                normalized_chars=len(normalized),
                decision=decision.value,
                reason=reason,
                tags=result.tags,
                scores=result.scores,
                approved_path=approved_path,
            )
        )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True, ensure_ascii=False) + "\n")

    pack_path: Path | None = None
    if create_pack:
        pack_path = output_dir / "approved_corpus_pack.tar.gz"
        with tarfile.open(pack_path, "w:gz") as archive:
            archive.add(manifest_path, arcname="manifest.jsonl")
            if approved_dir is not None:
                for path in sorted(approved_dir.glob("*.txt")):
                    archive.add(path, arcname=f"approved/{path.name}")

    approved_count = sum(1 for record in records if record.decision == CorpusDecision.APPROVE.value)
    quarantine_count = sum(1 for record in records if record.decision == CorpusDecision.QUARANTINE.value)
    rejected_count = sum(1 for record in records if record.decision == CorpusDecision.REJECT.value)
    return CorpusBuildSummary(
        scanned=len(records),
        approved=approved_count,
        quarantine=quarantine_count,
        rejected=rejected_count,
        manifest_path=str(manifest_path),
        approved_dir=str(approved_dir) if approved_dir is not None else None,
        pack_path=str(pack_path) if pack_path is not None else None,
    )
