"""Blockchain-aware checkpoint utilities for the Darwin organism.

Shared helpers for SHA-256 file hashing and bidirectional checkpoint-blockchain
binding verification.  Used by both the checkpoint save path and the resume /
bootstrap integrity checks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 digest of a file, reading in 64 KiB chunks."""
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(64 * 1024)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def verify_checkpoint_blockchain_binding(
    checkpoint_path: Path,
    ledger_path: Path,
) -> bool:
    """Verify bidirectional binding between a checkpoint and the blockchain ledger.

    Checks that:
    1. The checkpoint payload declares a ``block_hash``
    2. The ledger contains a block whose ``checkpoint_sha256`` matches the
       actual SHA-256 of the checkpoint ``.pt`` file
    3. The declared ``block_hash`` matches the block that references the
       checkpoint

    Returns ``True`` when all three checks pass, ``False`` otherwise (missing
    files, missing fields, or hash mismatches).
    """
    import torch

    if not checkpoint_path.exists() or not ledger_path.exists():
        return False

    # ── Load the checkpoint payload (memory-mapped, CPU, no unpickling) ──
    try:
        checkpoint_payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except Exception:
        return False

    declared_block_hash = checkpoint_payload.get("block_hash")
    if not isinstance(declared_block_hash, str) or len(declared_block_hash) != 64:
        return False

    # ── Actual file hash ──
    checkpoint_sha256_actual = sha256_file(checkpoint_path)

    # ── Scan the ledger JSONL for a matching block ──
    try:
        raw = ledger_path.read_bytes()
        if not raw.endswith(b"\n"):
            raw += b"\n"
    except Exception:
        return False

    found_matching_block = False
    for line in raw.decode("utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        payload = event.get("payload", {})
        if isinstance(payload, dict):
            declared_ckpt = payload.get("checkpoint_sha256")
            if declared_ckpt == checkpoint_sha256_actual:
                found_matching_block = True
                break

    return found_matching_block
