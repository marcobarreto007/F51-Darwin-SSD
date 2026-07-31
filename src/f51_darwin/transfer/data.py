from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch


def collect_local_text(root: Path, *, max_bytes: int = 32_000_000) -> str:
    parts: list[str] = []
    consumed = 0
    excluded = {".git", ".venv", ".venv_nitro", "workspace", "archive"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".py"}:
            continue
        if any(part in excluded for part in path.parts):
            continue
        if consumed >= max_bytes:
            break
        text = path.read_text(encoding="utf-8", errors="ignore")
        remaining = max_bytes - consumed
        parts.append(text[:remaining])
        consumed += min(len(text), remaining)
    if not parts:
        raise ValueError(f"no local text corpus found below {root}")
    return "\n\n".join(parts)


def collect_corpus_text(root: Path, *, max_bytes: int) -> str:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    parts: list[str] = []
    consumed = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        if consumed >= max_bytes:
            break
        text = path.read_text(encoding="utf-8", errors="ignore")
        remaining = max_bytes - consumed
        parts.append(text[:remaining])
        consumed += min(len(text.encode("utf-8")), remaining)
    if not parts:
        raise ValueError(f"no text corpus found below {root}")
    return "\n\n".join(parts)


def corpus_snapshot(root: Path, *, max_bytes: int) -> dict:
    selected: list[dict[str, int | str]] = []
    consumed = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        if consumed >= max_bytes:
            break
        stat = path.stat()
        used = min(int(stat.st_size), max_bytes - consumed)
        selected.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "used_bytes": used,
            }
        )
        consumed += used
    payload = {"root": str(root.resolve()), "max_bytes": max_bytes, "files": selected}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "identity": hashlib.sha256(encoded).hexdigest()}


def load_or_build_token_cache(
    tokenizer,
    corpus_root: Path,
    *,
    max_bytes: int,
    sequence_length: int,
    cache_path: Path,
    tokenizer_identity: str,
) -> tuple[torch.Tensor, bool]:
    snapshot = corpus_snapshot(corpus_root, max_bytes=max_bytes)
    metadata = {
        "schema_version": 1,
        "corpus_identity": snapshot["identity"],
        "tokenizer_identity": tokenizer_identity,
        "sequence_length": sequence_length,
        "max_bytes": max_bytes,
    }
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if payload.get("metadata") == metadata:
            chunks = payload["chunks"]
            if chunks.ndim != 2 or chunks.shape[1] != sequence_length + 1:
                raise ValueError("token cache tensor has an invalid shape")
            return chunks, True

    text = collect_corpus_text(corpus_root, max_bytes=max_bytes)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    token_tensor = torch.tensor(ids, dtype=torch.int32)
    if token_tensor.numel() <= sequence_length:
        raise ValueError("corpus is too small for one sequence")
    chunks = token_tensor.unfold(0, sequence_length + 1, sequence_length).contiguous()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
    torch.save({"metadata": metadata, "chunks": chunks}, temporary)
    os.replace(temporary, cache_path)
    return chunks, False


def token_chunks(tokenizer, text: str, *, sequence_length: int) -> list[torch.Tensor]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [
        torch.tensor(ids[start : start + sequence_length + 1], dtype=torch.long)
        for start in range(0, len(ids) - sequence_length, sequence_length)
    ]


def split_locked_holdout(
    chunks: list[torch.Tensor],
    *,
    holdout_chunks: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if holdout_chunks <= 0 or len(chunks) <= holdout_chunks:
        raise ValueError("holdout must be positive and smaller than the corpus")
    return chunks[:-holdout_chunks], chunks[-holdout_chunks:]


def split_transfer_holdouts(
    chunks: list[torch.Tensor],
    *,
    sequence_length: int,
    monitor_chunks: int,
    promotion_tokens: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    if sequence_length <= 0 or monitor_chunks <= 0 or promotion_tokens <= 0:
        raise ValueError("holdout contract values must be positive")
    promotion_chunks = (promotion_tokens + sequence_length - 1) // sequence_length
    required = monitor_chunks + promotion_chunks + 1
    if len(chunks) < required:
        available = len(chunks) * sequence_length
        raise ValueError(
            f"corpus cannot satisfy promotion holdout: required_chunks={required} "
            f"actual_chunks={len(chunks)} available_target_tokens={available}"
        )
    promotion = chunks[-promotion_chunks:]
    monitor = chunks[-(promotion_chunks + monitor_chunks) : -promotion_chunks]
    train = chunks[: -(promotion_chunks + monitor_chunks)]
    return train, monitor, promotion


def evaluation_partitions(
    chunks: list[torch.Tensor],
    *,
    batches: int,
) -> list[list[torch.Tensor]]:
    if batches <= 0 or len(chunks) < batches:
        raise ValueError("evaluation batches must be positive and fit the holdout")
    base, remainder = divmod(len(chunks), batches)
    partitions: list[list[torch.Tensor]] = []
    cursor = 0
    for index in range(batches):
        width = base + int(index < remainder)
        partitions.append(chunks[cursor : cursor + width])
        cursor += width
    return partitions
