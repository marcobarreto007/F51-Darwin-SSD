#!/usr/bin/env python3
"""Reconstruct the original int32 corpus file from a compressed transfer
(uint16 + zstd chunks) and verify sha256 matches exactly before it is
trusted for training. Writes the reconstructed .bin next to the manifest
unless --out is given."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import zstandard as zstd


def main() -> int:
    transfer_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else transfer_dir / "corpus_reconstructed.bin"
    manifest = json.loads((transfer_dir / "transfer_manifest.json").read_text(encoding="utf-8"))
    expected_sha256 = manifest["source_sha256"]
    expected_tokens = manifest["source_tokens"]
    expected_bytes = manifest["source_bytes"]

    dctx = zstd.ZstdDecompressor()
    sha = hashlib.sha256()
    tokens_written = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as out:
        for chunk_meta in manifest["chunks"]:
            chunk_path = transfer_dir / chunk_meta["path"]
            with open(chunk_path, "rb") as fh:
                reader = dctx.stream_reader(fh)
                buf = bytearray()
                while True:
                    data = reader.read(1 << 20)
                    if not data:
                        break
                    buf.extend(data)
                    usable = len(buf) - (len(buf) % 2)
                    if usable:
                        arr16 = np.frombuffer(bytes(buf[:usable]), dtype="<u2")
                        arr32 = arr16.astype("<i4")
                        payload = arr32.tobytes()
                        out.write(payload)
                        sha.update(payload)
                        tokens_written += len(arr16)
                        del buf[:usable]
                if buf:
                    print(f"WARNING: {len(buf)} leftover byte(s)", file=sys.stderr)

    computed_sha256 = sha.hexdigest()
    actual_bytes = out_path.stat().st_size
    print(f"tokens_written={tokens_written:,} expected={expected_tokens:,}")
    print(f"bytes_written={actual_bytes:,} expected={expected_bytes:,}")
    print(f"computed_sha256={computed_sha256}")
    print(f"expected_sha256 ={expected_sha256}")
    ok = (
        tokens_written == expected_tokens
        and actual_bytes == expected_bytes
        and computed_sha256 == expected_sha256
    )
    print("RECONSTRUCTION_OK" if ok else "RECONSTRUCTION_FAILED")
    if not ok:
        return 1
    print(f"Wrote verified corpus to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
