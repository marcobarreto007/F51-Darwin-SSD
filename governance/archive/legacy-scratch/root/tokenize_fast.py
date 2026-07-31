#!/usr/bin/env python3
"""
F51 Fast Tokenizer V2.

Robust local tokenizer for large corpora. Workers never return token lists to
the parent process. Each worker writes one binary part plus a manifest; the
parent merges verified parts in deterministic file order.

Usage:
    python tokenize_fast.py --corpus data/corpus --output data/tokens_fast.bin --workers 8
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import os
import re
import shutil
import sys
import time
from array import array
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

WORD_RE = re.compile(r"\S+|\n")
READ_BYTES_DEFAULT = 8 * 1024 * 1024
WRITE_TOKEN_CHUNK_DEFAULT = 1_000_000
MAX_WORD_BYTES_DEFAULT = 8192

_TOKENIZER: dict[str, Any] | None = None
_PARTS_DIR: Path | None = None
_WRITE_TOKEN_CHUNK: int = WRITE_TOKEN_CHUNK_DEFAULT
_READ_BYTES: int = READ_BYTES_DEFAULT
_MAX_WORD_BYTES: int = MAX_WORD_BYTES_DEFAULT


def load_tokenizer(tokenizer_dir: str | Path) -> dict[str, Any]:
    tokenizer_path = Path(tokenizer_dir)
    with open(tokenizer_path / "vocab.json", encoding="utf-8") as handle:
        vocab = json.load(handle)
    with open(tokenizer_path / "merges.txt", encoding="utf-8") as handle:
        raw_merges = [line.strip() for line in handle if line.strip()]

    merge_result: dict[tuple[int, int], int] = {}
    merge_rank: dict[tuple[int, int], int] = {}
    for rank, raw in enumerate(raw_merges):
        parts = raw.split()
        if len(parts) != 2:
            continue
        left, right = parts
        left_id = vocab.get(left)
        right_id = vocab.get(right)
        merged_id = vocab.get(left + right)
        if left_id is None or right_id is None or merged_id is None:
            continue
        key = (int(left_id), int(right_id))
        merge_result[key] = int(merged_id)
        merge_rank[key] = rank

    return {
        "byte_to_id": {
            byte: int(vocab.get(f"<b{byte:02x}>", vocab.get("<unk>", 1)))
            for byte in range(256)
        },
        "merge_result": merge_result,
        "merge_rank": merge_rank,
        "space_id": int(vocab.get("<b20>", vocab.get("<unk>", 1))),
        "vocab_size": len(vocab),
    }


def _bpe_byte_ids_to_ids(syms: list[int], tk: dict[str, Any]) -> list[int]:
    if len(syms) < 2:
        return syms

    merge_result = tk["merge_result"]
    merge_rank = tk["merge_rank"]
    while True:
        best_rank = 1_000_000_000
        best_pos = -1
        best_merged = None
        for index in range(len(syms) - 1):
            rank = merge_rank.get((syms[index], syms[index + 1]))
            if rank is not None and rank < best_rank:
                best_rank = rank
                best_pos = index
                best_merged = merge_result[(syms[index], syms[index + 1])]
        if best_merged is None:
            break
        syms[best_pos] = best_merged
        del syms[best_pos + 1]
    return syms


def _bpe_word_to_ids(word: str, tk: dict[str, Any]) -> list[int]:
    byte_to_id = tk["byte_to_id"]
    raw = word.encode("utf-8")
    if len(raw) <= _MAX_WORD_BYTES:
        return _bpe_byte_ids_to_ids([byte_to_id[byte] for byte in raw], tk)

    ids: list[int] = []
    for start in range(0, len(raw), _MAX_WORD_BYTES):
        chunk = raw[start : start + _MAX_WORD_BYTES]
        ids.extend(_bpe_byte_ids_to_ids([byte_to_id[byte] for byte in chunk], tk))
    return ids


def _append_int32(handle, hasher: hashlib._Hash, ids: list[int]) -> int:
    if not ids:
        return 0
    values = array("i", ids)
    if values.itemsize != 4:
        raise RuntimeError("array('i') is not 32-bit on this platform")
    payload = values.tobytes()
    handle.write(payload)
    hasher.update(payload)
    return len(ids)


class StreamingEncoder:
    """Chunk-safe encoder with global-strip-like whitespace behavior."""

    def __init__(self, tk: dict[str, Any], out_handle, hasher: hashlib._Hash) -> None:
        self.tk = tk
        self.out_handle = out_handle
        self.hasher = hasher
        self.buffer: list[int] = []
        self.carry = ""
        self.has_emitted = False
        self.prev_was_newline = False
        self.pending_newlines = 0
        self.tokens = 0

    def _emit_ids(self, ids: list[int]) -> None:
        self.buffer.extend(ids)
        if len(self.buffer) >= _WRITE_TOKEN_CHUNK:
            self.flush()

    def _emit_newlines_if_needed(self) -> None:
        if not self.has_emitted:
            self.pending_newlines = 0
            return
        newline_ids = _bpe_word_to_ids("\n", self.tk)
        for _ in range(self.pending_newlines):
            self._emit_ids(newline_ids)
        if self.pending_newlines:
            self.prev_was_newline = True
        self.pending_newlines = 0

    def consume_token(self, token: str) -> None:
        if token == "\n":
            if self.has_emitted:
                self.pending_newlines += 1
            return

        self._emit_newlines_if_needed()
        if self.has_emitted and not self.prev_was_newline:
            self._emit_ids([self.tk["space_id"]])
        self._emit_ids(_bpe_word_to_ids(token, self.tk))
        self.has_emitted = True
        self.prev_was_newline = False

    def consume_text(self, text: str, *, final: bool = False) -> None:
        text = self.carry + text
        self.carry = ""
        if not final and text and not text[-1].isspace():
            last_ws = max(text.rfind(" "), text.rfind("\n"), text.rfind("\t"), text.rfind("\r"))
            if last_ws < 0:
                self.carry = text
                return
            self.carry = text[last_ws + 1 :]
            text = text[: last_ws + 1]

        for token in WORD_RE.findall(text):
            self.consume_token(token)

    def finish(self) -> int:
        if self.carry:
            for token in WORD_RE.findall(self.carry):
                self.consume_token(token)
            self.carry = ""
        # Do not flush pending newline tokens at EOF; legacy encode_text used strip().
        self.pending_newlines = 0
        self.flush()
        return self.tokens

    def flush(self) -> None:
        if self.buffer:
            self.tokens += _append_int32(self.out_handle, self.hasher, self.buffer)
            self.buffer = []


def _init_worker(
    tokenizer_dir: str,
    parts_dir: str,
    write_token_chunk: int,
    read_bytes: int,
    max_word_bytes: int,
) -> None:
    global _TOKENIZER, _PARTS_DIR, _WRITE_TOKEN_CHUNK, _READ_BYTES, _MAX_WORD_BYTES
    _TOKENIZER = load_tokenizer(tokenizer_dir)
    _PARTS_DIR = Path(parts_dir)
    _WRITE_TOKEN_CHUNK = write_token_chunk
    _READ_BYTES = read_bytes
    _MAX_WORD_BYTES = max_word_bytes


def _manifest_path(part_path: Path) -> Path:
    return part_path.with_suffix(part_path.suffix + ".json")


def _load_valid_part(manifest_path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        part = Path(payload["part_path"])
        if not part.exists():
            return None
        if int(payload["bytes"]) != part.stat().st_size:
            return None
        return payload
    except Exception:
        return None


def tokenize_file(task: tuple[int, str]) -> dict[str, Any]:
    if _TOKENIZER is None or _PARTS_DIR is None:
        raise RuntimeError("worker was not initialized")

    index, raw_path = task
    source = Path(raw_path)
    part_path = _PARTS_DIR / f"part_{index:08d}.bin"
    manifest_path = _manifest_path(part_path)
    existing = _load_valid_part(manifest_path)
    if existing is not None and existing.get("source_path") == str(source):
        existing["skipped"] = True
        return existing

    tmp_path = part_path.with_suffix(".bin.tmp")
    hasher = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    started = time.time()
    token_count = 0

    with open(source, "rb") as input_handle, open(tmp_path, "wb") as output_handle:
        encoder = StreamingEncoder(_TOKENIZER, output_handle, hasher)
        while True:
            chunk = input_handle.read(_READ_BYTES)
            if not chunk:
                break
            encoder.consume_text(decoder.decode(chunk, final=False))
        encoder.consume_text(decoder.decode(b"", final=True), final=True)
        token_count = encoder.finish()

    os.replace(tmp_path, part_path)
    payload = {
        "index": index,
        "source_path": str(source),
        "source_bytes": source.stat().st_size,
        "part_path": str(part_path),
        "bytes": part_path.stat().st_size,
        "tokens": token_count,
        "sha256": hasher.hexdigest(),
        "seconds": round(time.time() - started, 3),
        "skipped": False,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def discover_text_files(corpus_dir: str | Path) -> list[Path]:
    corpus = Path(corpus_dir)
    if not corpus.exists():
        raise FileNotFoundError(f"corpus directory not found: {corpus}")
    return sorted(path for path in corpus.rglob("*.txt") if path.is_file())


def merge_parts(parts: list[dict[str, Any]], output_path: Path, *, keep_parts: bool) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    hasher = hashlib.sha256()
    total_bytes = 0
    total_tokens = 0

    with open(tmp_output, "wb") as output_handle:
        for item in sorted(parts, key=lambda record: int(record["index"])):
            part_path = Path(item["part_path"])
            expected_bytes = int(item["bytes"])
            actual_bytes = part_path.stat().st_size
            if actual_bytes != expected_bytes:
                raise RuntimeError(f"part size mismatch: {part_path} expected={expected_bytes} actual={actual_bytes}")
            with open(part_path, "rb") as input_handle:
                while True:
                    chunk = input_handle.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    output_handle.write(chunk)
                    hasher.update(chunk)
                    total_bytes += len(chunk)
            total_tokens += int(item["tokens"])

    os.replace(tmp_output, output_path)
    manifest = {
        "output_path": str(output_path),
        "bytes": total_bytes,
        "tokens": total_tokens,
        "sha256": hasher.hexdigest(),
        "parts": len(parts),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "int32_aligned": total_bytes % 4 == 0,
    }
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not keep_parts:
        shutil.rmtree(output_path.parent / f"{output_path.name}.parts", ignore_errors=True)
    return manifest


def tokenize_corpus(
    corpus_dir: str | Path,
    tokenizer_dir: str | Path,
    output_path: str | Path,
    *,
    workers: int,
    keep_parts: bool,
    read_bytes: int,
    write_token_chunk: int,
    max_word_bytes: int,
) -> dict[str, Any]:
    output = Path(output_path)
    parts_dir = output.parent / f"{output.name}.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    files = discover_text_files(corpus_dir)
    if not files:
        raise FileNotFoundError(f"no .txt files found under {corpus_dir}")

    print(f"Files: {len(files)}")
    print(f"Parts: {parts_dir}")
    started = time.time()
    completed: list[dict[str, Any]] = []
    total_tokens = 0
    total_bytes = 0

    with Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(str(tokenizer_dir), str(parts_dir), write_token_chunk, read_bytes, max_word_bytes),
    ) as pool:
        tasks = [(index, str(path)) for index, path in enumerate(files)]
        for count, result in enumerate(pool.imap_unordered(tokenize_file, tasks, chunksize=1), start=1):
            completed.append(result)
            total_tokens += int(result["tokens"])
            total_bytes += int(result["bytes"])
            elapsed = max(time.time() - started, 1e-6)
            status = "skip" if result.get("skipped") else "done"
            print(
                f"\r  {count}/{len(files)} files | {total_tokens/1e6:.1f}M tokens "
                f"| {total_bytes/1e9:.2f} GB parts | {total_bytes/elapsed/1e6:.1f} MB/s | {status}",
                end="",
                flush=True,
            )

    print()
    if len(completed) != len(files):
        raise RuntimeError(f"incomplete tokenization: {len(completed)}/{len(files)} files")
    manifest = merge_parts(completed, output, keep_parts=keep_parts)
    elapsed = max(time.time() - started, 1e-6)
    print(
        f"Done: {manifest['tokens']/1e9:.2f}B tokens -> {output} "
        f"({manifest['bytes']/elapsed/1e6:.1f} MB/s total)"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="F51 Fast Tokenizer V2")
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--tokenizer", default="tokenizer/f51_bpe")
    parser.add_argument("--output", default="data/tokens_fast.bin")
    parser.add_argument("--workers", type=int, default=min(16, cpu_count()))
    parser.add_argument("--keep-parts", action="store_true", help="Keep per-file token parts after final merge.")
    parser.add_argument("--read-mb", type=int, default=8, help="Input chunk size per worker, in MiB.")
    parser.add_argument("--write-token-chunk", type=int, default=WRITE_TOKEN_CHUNK_DEFAULT)
    parser.add_argument(
        "--max-word-bytes",
        type=int,
        default=MAX_WORD_BYTES_DEFAULT,
        help="Defensive BPE chunk size for pathological whitespace-free spans.",
    )
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be positive")
    read_bytes = max(1, args.read_mb) * 1024 * 1024

    tokenizer = load_tokenizer(args.tokenizer)
    print(
        f"Tokenizer: {args.tokenizer} | vocab={tokenizer['vocab_size']} "
        f"| merges={len(tokenizer['merge_result'])}"
    )
    print(f"Tokenizing {args.corpus} -> {args.output} ({args.workers} workers)")
    tokenize_corpus(
        args.corpus,
        args.tokenizer,
        args.output,
        workers=args.workers,
        keep_parts=args.keep_parts,
        read_bytes=read_bytes,
        write_token_chunk=args.write_token_chunk,
        max_word_bytes=max(128, args.max_word_bytes),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
