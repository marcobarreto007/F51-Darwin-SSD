from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from f51_darwin.dataset_states import DatasetStatus, SourceType


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def content_hash(text: str) -> str:
    normalized = text.strip().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def prompt_hash(prompt: str | None) -> str | None:
    if prompt is None:
        return None
    return hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# MinHash near-duplicate detection (pure Python, no datasketch dependency)
# ═══════════════════════════════════════════════════════════════════════════
#
# SHA-256 gives us EXACT dedup (cheap, catches identical text). MinHash LSH
# adds NEAR-duplicate detection: documents that are paraphrases, lightly
# edited, or partially overlapping. The Jaccard similarity between word
# k-shingle sets is estimated from 128 permutation hashes, and a band-based
# LSH index makes candidate lookup sub-linear instead of O(N²) all-pairs.

MINHASH_NUM_PERM = 128
MINHASH_SHINGLE_K = 5
MINHASH_DEFAULT_THRESHOLD = 0.7
# The LSH index is tuned for HIGH RECALL, not precision: it is only a candidate
# generator, and find_near_duplicate confirms every candidate with the exact
# signature Jaccard estimate against ``minhash_threshold``. A low band knee
# (~0.4) pushes the collision probability near 1 for any pair above the real
# 0.7 threshold, so true near-dups are almost never missed.
MINHASH_LSH_KNEE = 0.4
_MAX_HASH = (1 << 64) - 1


def _minhash_seed(index: int) -> int:
    """Deterministic 64-bit seed for permutation ``index``.

    Seeds are derived from SHA-256 so signatures stay comparable across runs
    and processes without needing to persist a seed table.
    """
    digest = hashlib.sha256(f"f51-minhash-seed-{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _word_shingles(text: str, k: int = MINHASH_SHINGLE_K) -> list[str]:
    """Word-level k-shingles of normalized text.

    Word shingles are more robust to minor character noise than byte/char
    shingles and keep the signature compact for long documents.
    """
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if not normalized:
        return []
    tokens = normalized.split()
    if len(tokens) < k:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)]


def minhash_signature(
    text: str,
    *,
    num_perm: int = MINHASH_NUM_PERM,
    k: int = MINHASH_SHINGLE_K,
) -> list[int]:
    """Compute a MinHash signature for ``text``.

    Each shingle is hashed once with BLAKE2b; the ``num_perm`` permutation
    minima are derived by XOR-ing the base hash with a deterministic seed,
    which avoids re-hashing every shingle for every permutation.
    """
    shingles = _word_shingles(text, k)
    if not shingles:
        return [_MAX_HASH] * num_perm
    base = [
        int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")
        for s in shingles
    ]
    signature: list[int] = []
    for i in range(num_perm):
        seed = _minhash_seed(i)
        minimum = _MAX_HASH
        for value in base:
            candidate = (value ^ seed) & _MAX_HASH
            if candidate < minimum:
                minimum = candidate
        signature.append(minimum)
    return signature


def estimate_jaccard(signature_a: list[int], signature_b: list[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    n = min(len(signature_a), len(signature_b))
    if n == 0:
        return 0.0
    matches = sum(1 for a, b in zip(signature_a[:n], signature_b[:n]) if a == b)
    return matches / n


class MinHashLSH:
    """Band-based Locality Sensitive Hashing index over MinHash signatures.

    ``insert`` a (key, signature) pair; ``query`` returns the set of keys that
    share at least one band with the query signature — candidate near-dups.
    Callers MUST confirm candidates with :func:`estimate_jaccard` because
    banding trades a false-positive rate for sub-linear lookup.

    The ``threshold`` here is the LSH *knee* used to pick band/row counts, not
    the final similarity decision: a low knee (default ``MINHASH_LSH_KNEE``)
    maximizes recall so the confirmatory Jaccard check (applied by the caller)
    is what actually decides near-duplicate status.
    """

    def __init__(
        self,
        *,
        threshold: float = MINHASH_LSH_KNEE,
        num_perm: int = MINHASH_NUM_PERM,
    ) -> None:
        self.threshold = threshold
        self.num_perm = num_perm
        self.bands, self.rows = self._choose_bands(num_perm, threshold)
        self._tables: list[dict[int, set[str]]] = [dict() for _ in range(self.bands)]
        self._signatures: dict[str, list[int]] = {}

    @staticmethod
    def _choose_bands(num_perm: int, threshold: float) -> tuple[int, int]:
        """Pick (bands, rows) with bands*rows == num_perm nearest the threshold.

        The LSH candidate-collision probability for similarity ``s`` with
        ``b`` bands of ``r`` rows is ``1 - (1 - s**r)**b``; the knee of that
        curve sits near ``s = (1/b)**(1/r)``, which we align to ``threshold``.
        """
        best = (num_perm, 1)
        best_diff = float("inf")
        for rows in range(1, num_perm + 1):
            if num_perm % rows != 0:
                continue
            bands = num_perm // rows
            knee = (1.0 / bands) ** (1.0 / rows)
            diff = abs(knee - threshold)
            if diff < best_diff:
                best_diff = diff
                best = (bands, rows)
        return best

    def _band_key(self, signature: list[int], band_index: int) -> int:
        chunk = tuple(
            signature[band_index * self.rows : band_index * self.rows + self.rows]
        )
        return int.from_bytes(
            hashlib.blake2b(repr(chunk).encode("utf-8"), digest_size=8).digest(),
            "big",
        )

    def insert(self, key: str, signature: list[int]) -> None:
        if len(signature) < self.num_perm:
            return
        self._signatures[key] = signature
        for band in range(self.bands):
            bucket = self._band_key(signature, band)
            self._tables[band].setdefault(bucket, set()).add(key)

    def query(self, signature: list[int]) -> set[str]:
        if len(signature) < self.num_perm:
            return set()
        candidates: set[str] = set()
        for band in range(self.bands):
            candidates |= self._tables[band].get(self._band_key(signature, band), set())
        return candidates

    def __len__(self) -> int:
        return len(self._signatures)


@dataclass
class DatasetRecord:
    id: str
    content_hash: str
    source_type: SourceType
    source_path: str
    generator_model: str | None
    generator_checkpoint: str | None
    prompt_hash: str | None
    created_at: str
    status: DatasetStatus
    scores: dict[str, float] = field(default_factory=dict)
    approval_reason: str | None = None
    rejection_reason: str | None = None
    parent_ids: list[str] = field(default_factory=list)
    dataset_version: str = "v0"
    # MinHash signature over word k-shingles; empty for legacy records created
    # before near-duplicate detection was introduced.
    minhash_signature: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_type"] = self.source_type.value
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetRecord":
        return cls(
            id=str(payload["id"]),
            content_hash=str(payload["content_hash"]),
            source_type=SourceType(str(payload["source_type"])),
            source_path=str(payload["source_path"]),
            generator_model=payload.get("generator_model"),
            generator_checkpoint=payload.get("generator_checkpoint"),
            prompt_hash=payload.get("prompt_hash"),
            created_at=str(payload["created_at"]),
            status=DatasetStatus(str(payload["status"])),
            scores={str(k): float(v) for k, v in dict(payload.get("scores", {})).items()},
            approval_reason=payload.get("approval_reason"),
            rejection_reason=payload.get("rejection_reason"),
            parent_ids=[str(item) for item in payload.get("parent_ids", [])],
            dataset_version=str(payload.get("dataset_version", "v0")),
            minhash_signature=[int(x) for x in payload.get("minhash_signature", [])],
        )


class ProvenanceLedger:
    def __init__(
        self,
        ledger_dir: Path,
        *,
        minhash_threshold: float = MINHASH_DEFAULT_THRESHOLD,
        minhash_num_perm: int = MINHASH_NUM_PERM,
    ) -> None:
        self.ledger_dir = ledger_dir
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.ledger_dir / "index.jsonl"
        self.hash_index_path = self.ledger_dir / "hashes.json"
        self.minhash_threshold = minhash_threshold
        self.minhash_num_perm = minhash_num_perm
        # Lazily-built caches; invalidated whenever the JSONL changes.
        self._lsh_cache: MinHashLSH | None = None
        self._latest_cache: dict[str, DatasetRecord] | None = None

    def _load_hash_index(self) -> dict[str, str]:
        if not self.hash_index_path.exists():
            return {}
        return json.loads(self.hash_index_path.read_text(encoding="utf-8"))

    def _save_hash_index(self, index: dict[str, str]) -> None:
        self.hash_index_path.write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def append(self, record: DatasetRecord) -> DatasetRecord:
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        hash_index = self._load_hash_index()
        if record.content_hash not in hash_index:
            hash_index[record.content_hash] = record.id
        self._save_hash_index(hash_index)
        # Keep the in-memory LSH index fresh without a full rebuild. The
        # latest_by_id cache is dropped because the append may update a record.
        if record.minhash_signature:
            self._get_lsh_index().insert(record.id, record.minhash_signature)
        self._latest_cache = None
        return record

    def update(self, record: DatasetRecord) -> DatasetRecord:
        self.append(record)
        return record

    def list_records(self) -> list[DatasetRecord]:
        if not self.index_path.exists():
            return []
        records: list[DatasetRecord] = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(DatasetRecord.from_dict(json.loads(line)))
        return records

    def latest_by_id(self) -> dict[str, DatasetRecord]:
        if self._latest_cache is None:
            latest: dict[str, DatasetRecord] = {}
            for record in self.list_records():
                latest[record.id] = record
            self._latest_cache = latest
        return self._latest_cache

    def known_hashes(self) -> set[str]:
        return set(self._load_hash_index().keys())

    def find_by_hash(self, digest: str) -> DatasetRecord | None:
        hash_index = self._load_hash_index()
        record_id = hash_index.get(digest)
        if record_id is None:
            return None
        return self.latest_by_id().get(record_id)

    def _get_lsh_index(self) -> MinHashLSH:
        if self._lsh_cache is None:
            # Build with a recall-focused band knee; the confirmatory
            # ``minhash_threshold`` Jaccard check is applied in
            # find_near_duplicate, not here.
            index = MinHashLSH(
                threshold=MINHASH_LSH_KNEE,
                num_perm=self.minhash_num_perm,
            )
            for record in self.latest_by_id().values():
                if record.minhash_signature:
                    index.insert(record.id, record.minhash_signature)
            self._lsh_cache = index
        return self._lsh_cache

    def find_near_duplicate(
        self,
        record: DatasetRecord,
        *,
        threshold: float | None = None,
        exclude_ids: list[str] | None = None,
    ) -> tuple[DatasetRecord, float] | None:
        """Return ``(other_record, jaccard)`` for the best near-duplicate, or None.

        Uses the band-based LSH index to short-list candidates, then confirms
        each with the exact signature Jaccard estimate. ``record`` itself and
        any ``exclude_ids`` are never returned.
        """
        target = record.minhash_signature
        if not target:
            return None
        effective_threshold = self.minhash_threshold if threshold is None else threshold
        index = self._get_lsh_index()
        candidates = index.query(target)
        excluded = {record.id, *(exclude_ids or [])}
        latest = self.latest_by_id()
        best: tuple[DatasetRecord, float] | None = None
        for record_id in candidates:
            if record_id in excluded:
                continue
            other = latest.get(record_id)
            if other is None or not other.minhash_signature:
                continue
            jaccard = estimate_jaccard(target, other.minhash_signature)
            if jaccard >= effective_threshold and (best is None or jaccard > best[1]):
                best = (other, jaccard)
        return best

    @classmethod
    def new_record(
        cls,
        *,
        text: str,
        source_type: SourceType,
        source_path: str,
        generator_model: str | None = None,
        generator_checkpoint: str | None = None,
        prompt: str | None = None,
        parent_ids: list[str] | None = None,
        dataset_version: str = "v0",
        status: DatasetStatus = DatasetStatus.CANDIDATE,
        compute_minhash: bool = True,
    ) -> DatasetRecord:
        return DatasetRecord(
            id=f"ds_{uuid.uuid4().hex[:12]}",
            content_hash=content_hash(text),
            source_type=source_type,
            source_path=source_path,
            generator_model=generator_model,
            generator_checkpoint=generator_checkpoint,
            prompt_hash=prompt_hash(prompt),
            created_at=utc_now_iso(),
            status=status,
            parent_ids=list(parent_ids or []),
            dataset_version=dataset_version,
            minhash_signature=minhash_signature(text) if compute_minhash else [],
        )
