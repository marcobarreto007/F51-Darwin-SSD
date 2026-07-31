from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXPECTED_CORPUS_BYTES = 74_195_890_756
EXPECTED_CORPUS_TOKENS = 18_548_972_689
EXPECTED_CORPUS_SHA256 = "9677e9f22f4d78efa7b25c77b2da3cbdb5fb2a499926ac641a75c13b65e25cff"
EXPECTED_CORPUS_DTYPE = "int32"
EXPECTED_CORPUS_TOKENIZER = "f51_bpe_80k"


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_corpus_manifest(
    corpus_path: str | Path,
    manifest_path: str | Path,
    *,
    verify_hash: bool = True,
) -> dict[str, Any]:
    corpus = Path(corpus_path)
    manifest_file = Path(manifest_path)
    if not corpus.is_file():
        raise FileNotFoundError(f"corpus_missing: {corpus}")
    if not manifest_file.is_file():
        raise FileNotFoundError(f"corpus_manifest_missing: {manifest_file}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    size = corpus.stat().st_size
    if size != EXPECTED_CORPUS_BYTES or manifest.get("bytes") != EXPECTED_CORPUS_BYTES:
        raise ValueError(
            f"bytes: expected={EXPECTED_CORPUS_BYTES}, file={size}, manifest={manifest.get('bytes')}"
        )
    if size % 4:
        raise ValueError(f"bytes: int32 divisibility failed for {size}")
    token_count = size // 4
    if token_count != EXPECTED_CORPUS_TOKENS or manifest.get("tokens") != EXPECTED_CORPUS_TOKENS:
        raise ValueError(
            f"tokens: expected={EXPECTED_CORPUS_TOKENS}, file={token_count}, manifest={manifest.get('tokens')}"
        )
    if manifest.get("dtype") != EXPECTED_CORPUS_DTYPE:
        raise ValueError(f"dtype: expected={EXPECTED_CORPUS_DTYPE}, observed={manifest.get('dtype')}")
    if manifest.get("little_endian") is not True:
        raise ValueError("endianness: corpus manifest must declare little_endian=true")
    if manifest.get("tokenizer") != EXPECTED_CORPUS_TOKENIZER:
        raise ValueError(
            f"tokenizer: expected={EXPECTED_CORPUS_TOKENIZER}, observed={manifest.get('tokenizer')}"
        )
    if manifest.get("sha256") != EXPECTED_CORPUS_SHA256:
        raise ValueError(
            f"sha256: expected={EXPECTED_CORPUS_SHA256}, manifest={manifest.get('sha256')}"
        )

    composition = manifest.get("composition")
    if not isinstance(composition, list) or not composition:
        raise ValueError("composition: non-empty component ledger required")
    if sum(int(item.get("bytes", -1)) for item in composition) != EXPECTED_CORPUS_BYTES:
        raise ValueError("composition: component bytes do not equal corpus bytes")
    if sum(int(item.get("tokens", -1)) for item in composition) != EXPECTED_CORPUS_TOKENS:
        raise ValueError("composition: component tokens do not equal corpus tokens")

    observed_hash = sha256_file(corpus) if verify_hash else manifest["sha256"]
    if observed_hash != EXPECTED_CORPUS_SHA256:
        raise ValueError(
            f"sha256: expected={EXPECTED_CORPUS_SHA256}, observed={observed_hash}"
        )
    return {
        "status": "ok",
        "path": str(corpus),
        "manifest": str(manifest_file),
        "bytes": size,
        "tokens": token_count,
        "dtype": EXPECTED_CORPUS_DTYPE,
        "little_endian": True,
        "tokenizer": EXPECTED_CORPUS_TOKENIZER,
        "sha256": observed_hash,
        "composition_entries": len(composition),
    }


def validate_checkpoint_report(
    report: dict[str, Any],
    *,
    expected_cycle: int,
    expected_step: int,
    expected_base_checkpoint_id: str | None = None,
) -> list[str]:
    issues: list[str] = []
    training = report.get("training_state") or {}
    if report.get("checkpoint_version") != 7:
        issues.append("checkpoint_version")
    if report.get("embedded_config_matches_file") is not True or report.get("config_diff"):
        issues.append("config")
    if training.get("cycle") != expected_cycle:
        issues.append("cycle")
    if training.get("step") != expected_step:
        issues.append("step")
    if report.get("resume_shape_compatible") is not True:
        issues.append("shapes")
    if report.get("topology_manifest_valid") is not True:
        issues.append("topology")
    if str(report.get("optimizer_type", "")).casefold() != "adamw":
        issues.append("adamw")
    if report.get("optimizer_resume_compatible") is not True:
        issues.append("adamw_resume")
    if report.get("identity_verified") is not True:
        issues.append("identity")
    if report.get("strict_resume_compatible") is not True:
        issues.append("strict_resume")
    if (
        expected_base_checkpoint_id is not None
        and report.get("base_checkpoint_id") != expected_base_checkpoint_id
    ):
        issues.append("base_checkpoint_id")
    return issues


def select_canonical_checkpoint(
    reports: Iterable[dict[str, Any]], *, approved_cycle: int = 71
) -> dict[str, Any]:
    eligible = [
        report
        for report in reports
        if report.get("gold_valid") is True
        and (report.get("training_state") or {}).get("cycle") == approved_cycle
    ]
    if len(eligible) != 1:
        raise ValueError(
            f"canonical_selection: expected one verified cycle {approved_cycle}, observed {len(eligible)}"
        )
    return eligible[0]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def publish_pointer_atomic(
    pointer_path: str | Path,
    *,
    checkpoint: str | Path,
    report: dict[str, Any],
    config: str | Path,
    source_commit: str,
    base_checkpoint_id: str,
    checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint)
    config_path = Path(config)
    if not checkpoint_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("pointer_publication: checkpoint and config must exist")
    training = report.get("training_state") or {}
    payload = {
        "version": 1,
        "checkpoint_version": int(report["checkpoint_version"]),
        "filename": checkpoint_path.name,
        "path": checkpoint_path.name,
        "cycle": int(training["cycle"]),
        "step": int(training["step"]),
        "size_bytes": checkpoint_path.stat().st_size,
        "sha256": checkpoint_sha256 or sha256_file(checkpoint_path),
        "base_checkpoint_id": base_checkpoint_id,
        "config_sha256": sha256_file(config_path),
        "source_commit": source_commit,
        "published_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "health": "ok",
    }
    _atomic_write_json(Path(pointer_path), payload)
    return payload


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    _atomic_write_json(Path(path), payload)
