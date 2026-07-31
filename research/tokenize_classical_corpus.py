#!/usr/bin/env python3
"""Reproducible tokenization and verification for the classical corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from f51_darwin.state_identity import tokenizer_identity
from f51_darwin.tokenizer import F51BPETokenizer
from f51_darwin.dataset_layout import (
    CLASSICAL_TOKEN_MANIFEST_RELATIVE,
    CLASSICAL_TOKEN_RELATIVE,
    RAW_CLASSICAL_RELATIVE,
    resolve_dataset_path,
    resolve_dataset_root,
)
from tokenize_fast import tokenize_corpus


DATASET_ROOT = resolve_dataset_root(ROOT)
CORPUS_DIR = resolve_dataset_path(ROOT, RAW_CLASSICAL_RELATIVE)
CORPUS_MANIFEST = CORPUS_DIR / "manifest.json"
TOKENIZER_DIR = ROOT / "tokenizer" / "f51_bpe_80k"
TOKEN_BIN = resolve_dataset_path(ROOT, CLASSICAL_TOKEN_RELATIVE)
TOKEN_METADATA = resolve_dataset_path(ROOT, CLASSICAL_TOKEN_MANIFEST_RELATIVE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifacts() -> dict[str, object]:
    corpus = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    metadata = json.loads(TOKEN_METADATA.read_text(encoding="utf-8"))
    tokenizer = F51BPETokenizer.load(TOKENIZER_DIR)
    errors: list[str] = []

    if metadata.get("source_corpus_sha256") != corpus.get("corpus_sha256"):
        errors.append("token metadata does not match the current corpus identity")
    if metadata.get("tokenizer_id") != tokenizer_identity(tokenizer):
        errors.append("token metadata does not match the current tokenizer identity")
    if TOKEN_BIN.stat().st_size != metadata.get("size_bytes"):
        errors.append("token bin byte size does not match metadata")
    if TOKEN_BIN.stat().st_size % 4:
        errors.append("token bin is not aligned to int32")
    if _sha256(TOKEN_BIN) != metadata.get("sha256"):
        errors.append("token bin SHA-256 does not match metadata")

    tokens = np.memmap(TOKEN_BIN, dtype=np.int32, mode="r")
    if len(tokens) != metadata.get("tokens"):
        errors.append("token count does not match metadata")
    if len(tokens) and (int(tokens.min()) < 0 or int(tokens.max()) >= tokenizer.vocab_size):
        errors.append("token ids fall outside the tokenizer vocabulary")

    result = {
        "ok": not errors,
        "errors": errors,
        "corpus_sha256": corpus.get("corpus_sha256"),
        "tokenizer_id": tokenizer_identity(tokenizer),
        "token_sha256": metadata.get("sha256"),
        "tokens": len(tokens),
        "vocab_size": tokenizer.vocab_size,
    }
    if errors:
        raise RuntimeError("; ".join(errors))
    return result


def tokenize(*, workers: int, force: bool) -> dict[str, object]:
    if TOKEN_BIN.exists() and not force:
        raise FileExistsError(f"{TOKEN_BIN} exists; pass --force to replace it")
    corpus = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    tokenizer = F51BPETokenizer.load(TOKENIZER_DIR)
    fast_manifest = tokenize_corpus(
        CORPUS_DIR,
        TOKENIZER_DIR,
        TOKEN_BIN,
        workers=workers,
        keep_parts=False,
        read_bytes=512 * 1024 * 1024,   # 512 MB read buffer
        write_token_chunk=10_000_000,   # 10M tokens per write
        max_word_bytes=256,             # max token bytes
    )
    metadata = {
        "file": str(TOKEN_BIN.relative_to(DATASET_ROOT)).replace("\\", "/"),
        "tokens": int(fast_manifest["tokens"]),
        "size_bytes": int(fast_manifest["bytes"]),
        "sha256": fast_manifest["sha256"],
        "tokenizer": str(TOKENIZER_DIR.relative_to(ROOT)).replace("\\", "/"),
        "tokenizer_id": tokenizer_identity(tokenizer),
        "vocab_size": tokenizer.vocab_size,
        "source": str(CORPUS_DIR.relative_to(DATASET_ROOT)).replace("\\", "/"),
        "source_corpus_sha256": corpus["corpus_sha256"],
        "source_files": corpus["total_books"],
        "file_order": "lexicographic recursive *.txt path order",
    }
    TOKEN_METADATA.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_METADATA.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    result = verify_artifacts() if args.verify_only else tokenize(
        workers=args.workers, force=args.force
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
