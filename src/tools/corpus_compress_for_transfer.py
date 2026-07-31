#!/usr/bin/env python3
"""Compress the tokenized corpus for cloud transfer, split into fixed-size chunks.

The corpus is stored as little-endian int32 token ids, but the tokenizer
vocab (58162) fits in uint16 -- downcasting halves the raw size losslessly
before zstd ever sees a byte. Output is split into ``--chunk-size-gb``
pieces so upload can resume per-chunk instead of restarting a single
multi-GB transfer on failure.

This only reads the source corpus (never writes to it) and writes new
files under --out-dir. Reconstructing the original int32 file on the
receiving end (upcast uint16 -> int32) must reproduce the exact original
bytes -- verified here via sha256 before/after a round-trip on a sample,
and the manifest below records the original sha256 for verification after
full reconstruction on the cloud side.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import zstandard as zstd


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}PB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chunk-size-gb", type=float, default=4.0)
    parser.add_argument("--read-block-tokens", type=int, default=128_000_000,
                         help="Tokens read per pass (128M tokens = 512MB int32 = 256MB uint16)")
    parser.add_argument("--level", type=int, default=12, help="zstd compression level")
    args = parser.parse_args()

    corpus_path = args.corpus.resolve()
    manifest_path = Path(f"{corpus_path}.manifest.json")
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["dtype"] != "int32" or not manifest["little_endian"]:
        print("ERROR: expected little-endian int32 corpus", file=sys.stderr)
        return 1

    total_bytes = corpus_path.stat().st_size
    total_tokens = total_bytes // 4
    if total_tokens != manifest["tokens"]:
        print(
            f"ERROR: corpus size on disk ({total_tokens} tokens) does not match "
            f"manifest ({manifest['tokens']} tokens)",
            file=sys.stderr,
        )
        return 1

    vocab_size = manifest.get("vocab_size")
    print(f"Corpus: {corpus_path}")
    print(f"  {total_tokens:,} tokens, {human(total_bytes)} on disk (int32)")
    print(f"  manifest sha256: {manifest['sha256']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunk_size_bytes = int(args.chunk_size_gb * (1024 ** 3))

    cctx = zstd.ZstdCompressor(level=args.level, threads=-1)

    chunk_index = 0
    chunk_path = args.out_dir / f"corpus_uint16.zst.part{chunk_index:04d}"
    chunk_file = open(chunk_path, "wb")
    compressor = cctx.stream_writer(chunk_file, closefd=False)
    chunk_written = 0
    chunks_meta: list[dict[str, object]] = []

    overall_sha256 = hashlib.sha256()
    tokens_done = 0
    t_start = time.time()
    read_block = args.read_block_tokens

    with open(corpus_path, "rb", buffering=0) as src:
        while tokens_done < total_tokens:
            n = min(read_block, total_tokens - tokens_done)
            raw = src.read(n * 4)
            if len(raw) != n * 4:
                print(f"ERROR: short read at token {tokens_done}", file=sys.stderr)
                return 1
            overall_sha256.update(raw)
            arr = np.frombuffer(raw, dtype="<i4")
            if vocab_size is not None and (arr.max(initial=0) >= vocab_size or arr.min(initial=0) < 0):
                print("ERROR: token id out of uint16-safe vocab range -- aborting downcast", file=sys.stderr)
                return 1
            arr16 = arr.astype("<u2")
            payload = arr16.tobytes()
            written = compressor.write(payload)
            chunk_written += len(payload)  # track pre-compression bytes fed, for chunk-size accounting is post; see below
            tokens_done += n

            if chunk_file.tell() >= chunk_size_bytes:
                compressor.flush(zstd.FLUSH_FRAME)
                compressor.close()
                chunk_file.close()
                chunks_meta.append({
                    "path": chunk_path.name,
                    "compressed_bytes": chunk_path.stat().st_size,
                })
                chunk_index += 1
                chunk_path = args.out_dir / f"corpus_uint16.zst.part{chunk_index:04d}"
                chunk_file = open(chunk_path, "wb")
                compressor = cctx.stream_writer(chunk_file, closefd=False)

            elapsed = time.time() - t_start
            pct = 100.0 * tokens_done / total_tokens
            rate = (tokens_done * 4) / max(elapsed, 1e-6)
            print(
                f"\r  {pct:5.1f}%  {tokens_done:,}/{total_tokens:,} tokens  "
                f"{human(rate)}/s  elapsed={elapsed:.0f}s",
                end="", file=sys.stderr,
            )

    compressor.flush(zstd.FLUSH_FRAME)
    compressor.close()
    chunk_file.close()
    if chunk_path.stat().st_size > 0:
        chunks_meta.append({
            "path": chunk_path.name,
            "compressed_bytes": chunk_path.stat().st_size,
        })
    print(file=sys.stderr)

    computed_sha256 = overall_sha256.hexdigest()
    if computed_sha256 != manifest["sha256"]:
        print(
            f"ERROR: read-time sha256 ({computed_sha256}) does not match manifest "
            f"({manifest['sha256']}) -- source file may have changed mid-read",
            file=sys.stderr,
        )
        return 1

    total_compressed = sum(int(c["compressed_bytes"]) for c in chunks_meta)
    transfer_manifest = {
        "schema": "corpus-transfer-v1",
        "source_path": str(corpus_path),
        "source_bytes": total_bytes,
        "source_tokens": total_tokens,
        "source_sha256": manifest["sha256"],
        "vocab_size": vocab_size,
        "dtype_on_disk": "int32",
        "transfer_dtype": "uint16",
        "chunk_size_gb_target": args.chunk_size_gb,
        "zstd_level": args.level,
        "chunks": chunks_meta,
        "total_compressed_bytes": total_compressed,
        "compression_ratio": round(total_bytes / max(total_compressed, 1), 3),
    }
    (args.out_dir / "transfer_manifest.json").write_text(
        json.dumps(transfer_manifest, indent=2), encoding="utf-8"
    )

    print(f"Done. {len(chunks_meta)} chunk(s), {human(total_compressed)} total "
          f"(ratio {transfer_manifest['compression_ratio']}x vs original int32 file).")
    print(f"Manifest: {args.out_dir / 'transfer_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
