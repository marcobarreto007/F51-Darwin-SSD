#!/usr/bin/env python3
"""Repoint a checkpoint's training_data_contract['source'] path/manifest_path
fields to the current environment, for a deliberate cross-machine resume of
the SAME verified-byte-identical corpus (sha256 already confirmed equal by
the transfer/decompress pipeline).

Only 'path' and 'manifest_path' are rewritten -- every other field
(size_bytes, token_count, dtype, little_endian, manifest_sha256,
int32_aligned, exists) is asserted unchanged first, so this cannot silently
paper over a real data difference. Does not touch model_state_dict,
optimizer state, or any other checkpoint content."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from f51_darwin.organism.support import canary_token_source_identity  # noqa: E402


def main() -> int:
    checkpoint_path = Path(sys.argv[1])
    new_token_bin = Path(sys.argv[2])

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    contract = payload.get("training_data_contract")
    if not isinstance(contract, dict) or not isinstance(contract.get("source"), dict):
        print("ERROR: checkpoint has no training_data_contract.source", file=sys.stderr)
        return 1

    old_source = dict(contract["source"])
    token_count = int(old_source["token_count"])
    new_source = canary_token_source_identity(new_token_bin, token_count)
    new_source.pop("mtime_ns", None)

    mismatched = [
        key for key in old_source
        if key not in ("path", "manifest_path")
        and old_source.get(key) != new_source.get(key)
    ]
    if mismatched:
        print(f"ERROR: non-path fields differ, refusing to repath: {mismatched}", file=sys.stderr)
        print(f"  old: {old_source}", file=sys.stderr)
        print(f"  new: {new_source}", file=sys.stderr)
        return 1

    print(f"old path:          {old_source.get('path')}")
    print(f"new path:          {new_source.get('path')}")
    print(f"old manifest_path: {old_source.get('manifest_path')}")
    print(f"new manifest_path: {new_source.get('manifest_path')}")
    print("All non-path fields match (size_bytes, token_count, dtype, manifest_sha256, etc.) -- repathing.")

    contract["source"] = new_source
    payload["training_data_contract"] = contract
    torch.save(payload, checkpoint_path)
    print(f"Repathed and saved: {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
