#!/usr/bin/env python3
"""Round-trip verify a compressed corpus transfer: decompress all chunks,
upcast uint16 -> int32, and confirm the sha256 matches the original
manifest exactly. Read-only on inputs; writes nothing but a log line."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import zstandard as zstd


def main() -> int:
    transfer_dir = Path(sys.argv[1])
    manifest = json.loads((transfer_dir / "transfer_manifest.json").read_text(encoding="utf-8"))
    expected_sha256 = manifest["source_sha256"]
    expected_tokens = manifest["source_tokens"]

    dctx = zstd.ZstdDecompressor()
    sha = hashlib.sha256()
    tokens_seen = 0

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
                # flush whole uint16 elements (2 bytes each) as they complete
                usable = len(buf) - (len(buf) % 2)
                if usable:
                    arr16 = np.frombuffer(bytes(buf[:usable]), dtype="<u2")
                    arr32 = arr16.astype("<i4")
                    sha.update(arr32.tobytes())
                    tokens_seen += len(arr16)
                    del buf[:usable]
            if buf:
                print(f"WARNING: {len(buf)} leftover byte(s) not a full uint16 element", file=sys.stderr)

    computed_sha256 = sha.hexdigest()
    print(f"tokens_seen={tokens_seen:,} expected={expected_tokens:,}")
    print(f"reconstructed_sha256={computed_sha256}")
    print(f"expected_sha256      ={expected_sha256}")
    ok = tokens_seen == expected_tokens and computed_sha256 == expected_sha256
    print("ROUNDTRIP_OK" if ok else "ROUNDTRIP_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
