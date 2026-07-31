from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

from f51_darwin.corpus_factory_pipeline import sha256_bytes
from f51_darwin.corpus_policy import CorpusDecision


@dataclass(frozen=True)
class PackVerificationReport:
    ok: bool
    manifest_records: int
    approved_records: int
    packed_approved_files: int
    errors: list[str] = field(default_factory=list)


def _load_manifest_from_pack(pack_path: Path) -> list[dict[str, object]]:
    with tarfile.open(pack_path, "r:gz") as archive:
        try:
            member = archive.getmember("manifest.jsonl")
        except KeyError as exc:
            raise ValueError("pack missing manifest.jsonl") from exc
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ValueError("pack manifest.jsonl could not be read")
        payload = extracted.read().decode("utf-8")
    return [json.loads(line) for line in payload.splitlines() if line.strip()]


def verify_corpus_pack(pack_path: Path) -> PackVerificationReport:
    pack_path = pack_path.resolve()
    if not pack_path.exists():
        return PackVerificationReport(
            ok=False,
            manifest_records=0,
            approved_records=0,
            packed_approved_files=0,
            errors=[f"pack not found: {pack_path}"],
        )

    errors: list[str] = []
    try:
        manifest = _load_manifest_from_pack(pack_path)
    except (tarfile.TarError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return PackVerificationReport(
            ok=False,
            manifest_records=0,
            approved_records=0,
            packed_approved_files=0,
            errors=[str(exc)],
        )

    approved_records = [
        record
        for record in manifest
        if str(record.get("decision")) == CorpusDecision.APPROVE.value
    ]
    approved_paths = {str(record.get("approved_path")) for record in approved_records}
    approved_paths.discard("None")
    approved_paths.discard("")

    if len(approved_paths) != len(approved_records):
        errors.append("approved records must have unique approved_path values")

    seen_hashes: set[str] = set()
    for record in manifest:
        digest = str(record.get("normalized_sha256", ""))
        if not digest:
            errors.append(f"{record.get('relative_path', '<unknown>')}: missing normalized_sha256")
            continue
        if digest in seen_hashes:
            errors.append(f"{record.get('relative_path', '<unknown>')}: duplicate normalized_sha256")
        seen_hashes.add(digest)

    try:
        with tarfile.open(pack_path, "r:gz") as archive:
            names = set(archive.getnames())
            packed_approved = {name for name in names if name.startswith("approved/") and name != "approved/"}
            for name in names:
                if name == "manifest.jsonl" or name.startswith("approved/"):
                    continue
                errors.append(f"unexpected tar member: {name}")
            if packed_approved != approved_paths:
                missing = sorted(approved_paths - packed_approved)
                extra = sorted(packed_approved - approved_paths)
                if missing:
                    errors.append(f"missing approved files: {missing}")
                if extra:
                    errors.append(f"extra approved files: {extra}")
            for record in approved_records:
                approved_path = str(record.get("approved_path"))
                try:
                    member = archive.getmember(approved_path)
                except KeyError:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    errors.append(f"{approved_path}: could not read approved file")
                    continue
                digest = sha256_bytes(extracted.read())
                expected = str(record.get("normalized_sha256"))
                if digest != expected:
                    errors.append(f"{approved_path}: sha256 mismatch")
    except tarfile.TarError as exc:
        errors.append(str(exc))
        packed_approved = set()

    return PackVerificationReport(
        ok=not errors,
        manifest_records=len(manifest),
        approved_records=len(approved_records),
        packed_approved_files=len(packed_approved),
        errors=errors,
    )
