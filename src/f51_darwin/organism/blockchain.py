"""Darwin-X Organ Blockchain — append-only block chain for organ state provenance.

Each block is a JSONL line in ``workspace/runtime/organism/blockchain/blocks.jsonl``.
Blocks are linked via ``prev_block_hash`` → ``block_hash`` (raw SHA-256 hex).
A Merkle tree over per-phase transactions anchors the block.

Schema prefix (``"darwin-block-v1:"``) is used for external block identification
and follows the same pattern as ``StepIdentity.key`` (``"causal-step-v1:"``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════
# Exceptions
# ═══════════════════════════════════════════════════════════════════


class OrganLedgerError(RuntimeError):
    """Base exception for OrganLedger operations."""


class ChainIntegrityError(OrganLedgerError):
    """Raised when blockchain integrity validation fails."""


# ═══════════════════════════════════════════════════════════════════
# Canonical JSON serialization (consistent with causal_ledger.py)
# ═══════════════════════════════════════════════════════════════════


def _jsonable(value: Any, *, path: str = "value") -> Any:
    """Recursively convert *value* into a deterministic JSON-serializable form."""
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} must use string keys to be JSON-compatible")
            converted[key] = _jsonable(item, path=f"{path}.{key}")
        return converted
    if isinstance(value, (list, tuple)):
        return [
            _jsonable(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} must be JSON-compatible, got {type(value).__name__}"
    )


def _ensure_finite(value: Any, *, path: str = "value") -> None:
    """Raise ``ValueError`` if *value* contains a non-finite float anywhere."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite value")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _ensure_finite(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _ensure_finite(item, path=f"{path}[{index}]")


def canonical_json(obj) -> bytes:
    """Serialize *obj* to canonical UTF-8 JSON bytes.

    Uses ``sort_keys``, compact ``separators``, ``ensure_ascii=False``,
    and disallows ``NaN``/``Infinity``.  This is the same encoding used by
    ``causal_ledger._canonical_bytes``.
    """
    serializable = _jsonable(obj)
    _ensure_finite(serializable)
    return json.dumps(
        serializable,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Return the lowercase SHA-256 hex digest of *data*."""
    return hashlib.sha256(data).hexdigest()


# ═══════════════════════════════════════════════════════════════════
# BlockHeader
# ═══════════════════════════════════════════════════════════════════


@dataclass
class BlockHeader:
    """Immutable provenance metadata for a single organ block."""

    schema: str = "darwin-organ-block-v1"
    block_number: int = 0
    prev_block_hash: str = "0" * 64
    merkle_root: str = "0" * 64
    timestamp_unix_ms: int = 0
    step_key: str = ""  # StepIdentity.key from causal_bus
    cycle: int = 0
    optimizer_step: int = 0
    checkpoint_flag: bool = False
    checkpoint_sha256: str | None = None
    organ_identity_merkle_root: str = "0" * 64


# ═══════════════════════════════════════════════════════════════════
# MerkleTree
# ═══════════════════════════════════════════════════════════════════


class MerkleTree:
    """Binary Merkle tree over transaction leaf hashes.

    Leaves are ordered.  If the leaf count is odd the last leaf is duplicated.
    Internal nodes are ``SHA-256(left_bytes ‖ right_bytes)``.
    """

    def __init__(self, leaves: list[str], root: str) -> None:
        self.leaves: list[str] = list(leaves)
        self.root: str = root
        self._tree_levels: list[list[str]] | None = None

    @staticmethod
    def build(transactions: list[dict]) -> "MerkleTree":
        """Build a Merkle tree whose leaves are ``sha256(canonical_json(tx))``.

        Returns a ``MerkleTree`` with ``root == "0" * 64`` when *transactions*
        is empty.
        """
        if not transactions:
            return MerkleTree(leaves=[], root="0" * 64)

        leaf_hexes: list[str] = [
            sha256_hex(canonical_json(tx)) for tx in transactions
        ]

        tree_levels: list[list[str]] = [list(leaf_hexes)]
        while len(tree_levels[-1]) > 1:
            level = list(tree_levels[-1])
            if len(level) % 2 == 1:
                level.append(level[-1])  # duplicate last
            next_level: list[str] = []
            for i in range(0, len(level), 2):
                parent = sha256_hex(
                    bytes.fromhex(level[i]) + bytes.fromhex(level[i + 1])
                )
                next_level.append(parent)
            tree_levels.append(next_level)

        mt = MerkleTree(leaves=list(leaf_hexes), root=tree_levels[-1][0])
        mt._tree_levels = tree_levels
        return mt

    @staticmethod
    def build_from_hashes(hex_hashes: list[str]) -> "MerkleTree":
        """Build a Merkle tree directly from pre-computed leaf hex hashes.

        Each element of *hex_hashes* must be a 64-character hex SHA-256 digest.
        Returns a ``MerkleTree`` with ``root == "0" * 64`` when empty.
        """
        if not hex_hashes:
            return MerkleTree(leaves=[], root="0" * 64)

        for i, h in enumerate(hex_hashes):
            if len(h) != 64 or not all(c in "0123456789abcdef" for c in h):
                raise ValueError(f"leaf[{i}] is not a valid 64-char hex digest")

        tree_levels: list[list[str]] = [list(hex_hashes)]
        while len(tree_levels[-1]) > 1:
            level = list(tree_levels[-1])
            if len(level) % 2 == 1:
                level.append(level[-1])
            next_level: list[str] = []
            for i in range(0, len(level), 2):
                parent = sha256_hex(
                    bytes.fromhex(level[i]) + bytes.fromhex(level[i + 1])
                )
                next_level.append(parent)
            tree_levels.append(next_level)

        mt = MerkleTree(leaves=list(hex_hashes), root=tree_levels[-1][0])
        mt._tree_levels = tree_levels
        return mt

    def prove(self, tx_index: int) -> list[tuple[str, bool]]:
        """Generate a Merkle proof for the leaf at *tx_index*.

        Returns a list of ``(sibling_hash, is_left)`` tuples from leaf to root.
        ``is_left = True`` means the sibling is on the **left** side of the
        parent concatenation (the sibling_hash is concatenated first when
        reconstructing the parent node).
        """
        if not self.leaves:
            raise IndexError("MerkleTree has no leaves")
        if tx_index < 0 or tx_index >= len(self.leaves):
            raise IndexError(
                f"tx_index {tx_index} out of range [0, {len(self.leaves)})"
            )

        # (Re-)build tree levels if needed.
        tree_levels = self._tree_levels
        if tree_levels is None:
            tree_levels = [list(self.leaves)]
            while len(tree_levels[-1]) > 1:
                level = list(tree_levels[-1])
                if len(level) % 2 == 1:
                    level.append(level[-1])
                next_level: list[str] = []
                for j in range(0, len(level), 2):
                    parent = sha256_hex(
                        bytes.fromhex(level[j]) + bytes.fromhex(level[j + 1])
                    )
                    next_level.append(parent)
                tree_levels.append(next_level)
            self._tree_levels = tree_levels

        proof: list[tuple[str, bool]] = []
        current_index = tx_index
        for depth in range(len(tree_levels) - 1):
            level = list(tree_levels[depth])
            if len(level) % 2 == 1 and current_index == len(level) - 1:
                # Last leaf duplicated itself; sibling is itself.
                sibling = level[current_index]
                proof.append((sibling, False))
            elif current_index % 2 == 0:
                # We are a left child → sibling is on the right.
                sibling = (
                    level[current_index + 1]
                    if current_index + 1 < len(level)
                    else level[current_index]
                )
                proof.append((sibling, False))
            else:
                # We are a right child → sibling is on the left.
                sibling = level[current_index - 1]
                proof.append((sibling, True))
            current_index //= 2

        return proof

    @staticmethod
    def verify_proof(
        root: str,
        leaf_hash: str,
        proof: list[tuple[str, bool]],
    ) -> bool:
        """Verify that *leaf_hash* belongs to the tree with the given *root*.

        *proof* is a list of ``(sibling_hash, is_left)`` as produced by
        :meth:`prove`.  ``is_left=True`` means the sibling is on the left.

        Returns ``True`` iff the reconstructed root matches *root*.
        """
        current = bytes.fromhex(leaf_hash)
        for sibling_hex, is_left in proof:
            sibling = bytes.fromhex(sibling_hex)
            if is_left:
                current = hashlib.sha256(sibling + current).digest()
            else:
                current = hashlib.sha256(current + sibling).digest()
        return current.hex() == root


# ═══════════════════════════════════════════════════════════════════
# Block
# ═══════════════════════════════════════════════════════════════════


@dataclass
class Block:
    """A single block in the organ blockchain.

    Stores a header with provenance metadata and a list of transaction dicts
    (one per bus phase: ``PRE_LOSS``, ``PRE_BACKWARD``, ``PRE_OPTIMIZER``).
    """

    header: BlockHeader
    transactions: list[dict]
    block_hash: str = ""

    def _body_dict(self) -> dict[str, Any]:
        """Return the canonical body dict used for hashing (header + tx, no hash)."""
        return {
            "header": dataclasses.asdict(self.header),
            "transactions": list(self.transactions),
        }

    def compute_hash(self) -> str:
        """Compute the raw SHA-256 hex digest of the canonical block body.

        The body is ``{"header": ..., "transactions": ...}`` — the
        ``block_hash`` field itself is excluded.
        """
        return sha256_hex(canonical_json(self._body_dict()))

    def to_dict(self) -> dict[str, Any]:
        """Return the full block as a dict (header + transactions + block_hash)."""
        result = self._body_dict()
        result["block_hash"] = self.block_hash
        return result

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Block":
        """Reconstruct a ``Block`` from a dict (as produced by :meth:`to_dict`).

        If *d* contains a ``block_hash`` key, the hash is validated against
        the reconstructed body.  Raises ``ChainIntegrityError`` on mismatch.
        """
        header_raw = d["header"]
        header = BlockHeader(**{
            k: v for k, v in header_raw.items()
            if k in {f.name for f in dataclasses.fields(BlockHeader)}
        })
        transactions = list(d.get("transactions", []))
        block_hash = d.get("block_hash", "")
        block = Block(header=header, transactions=transactions, block_hash=block_hash)
        if block_hash:
            computed = block.compute_hash()
            if block_hash != computed:
                raise ChainIntegrityError(
                    f"Block hash mismatch: stored={block_hash[:16]}..., "
                    f"computed={computed[:16]}..."
                )
        return block


# ═══════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════


def block_hash(block: Block) -> str:
    """Return the externally-addressable block identity.

    Format: ``"darwin-block-v1:" + sha256(canonical_json(body))``.

    Follows the same prefix pattern as ``StepIdentity.key``
    (``"causal-step-v1:"``).
    """
    body = block._body_dict()
    return "darwin-block-v1:" + sha256_hex(canonical_json(body))


def merkle_root_hex(hashes: list[str]) -> str:
    """Compute the Merkle root from a list of leaf hex hashes.

    Returns ``"0" * 64`` when *hashes* is empty.
    """
    if not hashes:
        return "0" * 64
    tree = MerkleTree.build_from_hashes(hashes)
    return tree.root


def organ_identity_merkle_root(identities: dict[str, str]) -> str:
    """Compute the Merkle root over organ identity hashes.

    Organ identities are sorted alphabetically by organ name.  Each leaf is
    the raw SHA-256 hex digest extracted from the identity string
    (format ``"organ:<name>:v1:<sha256>"``).

    Returns ``"0" * 64`` when *identities* is empty.
    """
    sorted_names = sorted(identities)
    if not sorted_names:
        return "0" * 64
    leaf_hashes: list[str] = []
    for name in sorted_names:
        identity = identities[name]
        # Extract the trailing 64-char hex digest
        parts = identity.rsplit(":", 1)
        digest = parts[-1] if len(parts) == 2 and len(parts[-1]) == 64 else identity
        leaf_hashes.append(digest)
    tree = MerkleTree.build_from_hashes(leaf_hashes)
    return tree.root


# ═══════════════════════════════════════════════════════════════════
# ChainVerification
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ChainVerification:
    """Result of an organ blockchain integrity scan."""

    valid: bool
    block_count: int
    head_hash: str
    errors: list[str]


# ═══════════════════════════════════════════════════════════════════
# OrganLedger
# ═══════════════════════════════════════════════════════════════════


class OrganLedger:
    """Append-only blockchain for organ state provenance.

    Blocks are stored as JSONL (one canonical JSON line per block) in
    ``workspace/runtime/organism/blockchain/blocks.jsonl``.

    Every append validates ``prev_block_hash``, computes the canonical block
    hash, and calls ``fsync`` before returning.  An in-memory index maps
    ``block_number`` → ``Block`` for fast lookups.

    Construction raises ``ChainIntegrityError`` if the on-disk chain fails
    validation.
    """

    EMPTY_HEAD: str = "0" * 64
    """Sentinel for the genesis block's ``prev_block_hash`` and empty chains."""

    def __init__(self, path: str | Path, *, fsync: bool = True) -> None:
        self.path = Path(path)
        self.fsync = bool(fsync)
        self._lock = threading.Lock()

        verif, blocks = self._scan_chain()
        if not verif.valid:
            raise ChainIntegrityError("; ".join(verif.errors))

        self._blocks: dict[int, Block] = {}
        for blk in blocks:
            self._blocks[blk.header.block_number] = blk
        self._head_hash = verif.head_hash
        self._next_block_number = verif.block_count
        self._fingerprint = _file_fingerprint(self.path)

    # ── public properties ──────────────────────────────────────────

    @property
    def head_hash(self) -> str:
        """The ``block_hash`` of the most recent block (or ``EMPTY_HEAD``)."""
        return self._head_hash

    # ── block access ───────────────────────────────────────────────

    def get_head(self) -> Block | None:
        """Return the most recent block, or ``None`` if the chain is empty."""
        if self._next_block_number == 0:
            return None
        return self._blocks.get(self._next_block_number - 1)

    def get_head_hash(self) -> str:
        """Return the most recent block hash (or ``EMPTY_HEAD``)."""
        return self._head_hash

    def get_block(self, block_number: int) -> Block | None:
        """Return the block with the given *block_number*, or ``None``."""
        return self._blocks.get(block_number)

    def get_block_count(self) -> int:
        """Return the total number of blocks in the chain."""
        return self._next_block_number

    # ── append ─────────────────────────────────────────────────────

    def append_block(self, block: Block) -> Block:
        """Append *block* to the chain after validating ``prev_block_hash``.

        *   Sets ``block.header.block_number`` to the next sequence number.
        *   Computes ``block.block_hash`` via :meth:`Block.compute_hash`.
        *   Writes the canonical JSONL line and calls ``fsync``.

        Returns the block (mutated in-place with the computed hash).
        """
        with self._lock:
            # ── Detect external mutations and auto-recover ─────
            current_fp = _file_fingerprint(self.path)
            if current_fp != self._fingerprint:
                verif, blocks = self._scan_chain()
                if not verif.valid:
                    raise ChainIntegrityError(
                        "ledger externally corrupted; " + "; ".join(verif.errors)
                    )
                self._blocks = {blk.header.block_number: blk for blk in blocks}
                self._head_hash = verif.head_hash
                self._next_block_number = verif.block_count
                self._fingerprint = current_fp

            # ── Set block number and prev_block_hash under lock ─
            block.header.block_number = self._next_block_number
            block.header.prev_block_hash = self._head_hash

            # ── Compute and store hash ─────────────────────────
            block.block_hash = block.compute_hash()

            # ── Write to file ──────────────────────────────────
            encoded = canonical_json(block.to_dict()) + b"\n"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("ab") as handle:
                handle.write(encoded)
                handle.flush()
                if self.fsync:
                    os.fsync(handle.fileno())

            # ── Update in-memory state ─────────────────────────
            self._blocks[block.header.block_number] = block
            self._head_hash = block.block_hash
            self._next_block_number += 1
            self._fingerprint = _file_fingerprint(self.path)

            return block

    # ── chain verification ─────────────────────────────────────────

    def verify_chain(self) -> ChainVerification:
        """Re-scan the on-disk chain and return a ``ChainVerification``."""
        verif, _ = self._scan_chain()
        return verif

    def _scan_chain(self) -> tuple[ChainVerification, list[Block]]:
        """Read and validate every block in the JSONL file.

        Returns a ``(ChainVerification, [Block])`` pair.
        """
        if not self.path.exists() or self.path.stat().st_size == 0:
            return (
                ChainVerification(
                    valid=True,
                    block_count=0,
                    head_hash=self.EMPTY_HEAD,
                    errors=[],
                ),
                [],
            )

        try:
            text = self.path.read_text("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            return (
                ChainVerification(
                    valid=False,
                    block_count=0,
                    head_hash=self.EMPTY_HEAD,
                    errors=[f"invalid_utf8:{exc.start}"],
                ),
                [],
            )

        errors: list[str] = []
        blocks: list[Block] = []
        previous = self.EMPTY_HEAD
        expected_number = 0

        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line:
                errors.append(f"line_{line_number}:blank_line")
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line_{line_number}:invalid_json:{exc.msg}")
                break

            if not isinstance(data, dict):
                errors.append(f"line_{line_number}:not_a_dict")
                break
            if "header" not in data or "transactions" not in data:
                errors.append(f"line_{line_number}:block_fields_missing")
                break

            # ── Canonical encoding check ───────────────────────
            expected_line = canonical_json(data).decode("utf-8")
            if line != expected_line:
                errors.append(f"line_{line_number}:noncanonical_encoding")

            # ── Parse block ────────────────────────────────────
            try:
                block = Block.from_dict(data)
            except Exception as exc:
                errors.append(f"line_{line_number}:block_parse:{exc}")
                break

            # ── Structural integrity ───────────────────────────
            if block.header.prev_block_hash != previous:
                errors.append(
                    f"line_{line_number}:prev_hash_mismatch:"
                    f"expected={previous[:16]}... got={block.header.prev_block_hash[:16]}..."
                )

            if block.header.block_number != expected_number:
                errors.append(
                    f"line_{line_number}:block_number_mismatch:"
                    f"expected={expected_number} got={block.header.block_number}"
                )

            if block.header.block_number != 0 and block.header.merkle_root != "0" * 64:
                tx_tree = MerkleTree.build(block.transactions)
                if block.header.merkle_root != tx_tree.root:
                    errors.append(
                        f"line_{line_number}:merkle_root_mismatch:"
                        f"header={block.header.merkle_root[:16]}... "
                        f"computed={tx_tree.root[:16]}..."
                    )

            blocks.append(block)
            previous = block.block_hash
            expected_number += 1

        head_hash = previous if blocks else self.EMPTY_HEAD
        return (
            ChainVerification(
                valid=not errors,
                block_count=len(blocks),
                head_hash=head_hash,
                errors=errors,
            ),
            blocks,
        )

    # ── bulk operations ────────────────────────────────────────────

    def scan_chain(self) -> list[Block]:
        """Return every block in the chain (re-reads from disk)."""
        _, blocks = self._scan_chain()
        return blocks

    # ── truncation ─────────────────────────────────────────────────

    def truncate_to_block(self, block_hash: str) -> None:
        """Truncate the chain so that the last block has the given *block_hash*.

        Blocks after the target are permanently removed from disk.
        Raises ``ValueError`` if *block_hash* is not found in the chain.
        """
        target = str(block_hash or "")
        if not target or target == self.EMPTY_HEAD:
            raise ValueError("block_hash must be a non-empty, non-sentinel hex string")

        with self._lock:
            verif, blocks = self._scan_chain()
            if not verif.valid:
                raise ChainIntegrityError("; ".join(verif.errors))

            cut_at: int | None = None
            for idx, blk in enumerate(blocks):
                if blk.block_hash == target:
                    cut_at = idx + 1  # keep up to and including this block
                    break

            if cut_at is None:
                raise ValueError(
                    f"block_hash {target[:16]}... not found in chain"
                )

            # Read file, keep lines up to cut_at
            raw = self.path.read_bytes()
            if not raw.endswith(b"\n"):
                raw += b"\n"
            text = raw.decode("utf-8")
            lines = text.splitlines(keepends=True)
            kept = "".join(lines[:cut_at])
            if not kept.endswith("\n"):
                kept += "\n"

            kept_bytes = kept.encode("utf-8")
            tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp_path.write_bytes(kept_bytes)
            tmp_path.replace(self.path)

            actual = self.path.read_bytes()
            if actual != kept_bytes:
                self.path.write_bytes(raw)
                raise ChainIntegrityError(
                    "ledger content mismatch after truncation"
                )

            self._head_hash = target
            self._next_block_number = cut_at
            self._blocks = {
                blk.header.block_number: blk for blk in blocks[:cut_at]
            }
            self._fingerprint = _file_fingerprint(self.path)


# ═══════════════════════════════════════════════════════════════════
# File fingerprint helper
# ═══════════════════════════════════════════════════════════════════


def _file_fingerprint(path: Path) -> tuple[int, int] | None:
    """Return ``(st_size, st_mtime_ns)`` or ``None`` if the file does not exist."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_size, stat.st_mtime_ns


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════

__all__ = [
    "Block",
    "BlockHeader",
    "ChainIntegrityError",
    "ChainVerification",
    "MerkleTree",
    "OrganLedger",
    "OrganLedgerError",
    "block_hash",
    "canonical_json",
    "merkle_root_hex",
    "organ_identity_merkle_root",
    "sha256_hex",
]
