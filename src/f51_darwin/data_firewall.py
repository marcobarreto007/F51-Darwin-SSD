from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from f51_darwin.dataset_states import DatasetStatus, SourceType
from f51_darwin.provenance import DatasetRecord, ProvenanceLedger


@dataclass(frozen=True)
class FirewallDecision:
    status: DatasetStatus
    reason: str
    scores: dict[str, float]


@dataclass(frozen=True)
class FirewallConfig:
    min_chars: int = 32
    ngram_size: int = 5
    max_ngram_overlap: float = 0.85
    max_ngram_compare_chars: int = 100_000
    uncertain_score_low: float = 0.45
    uncertain_score_high: float = 0.65
    quality_pass_score: float = 0.70
    # Jaccard threshold above which two documents are treated as near-duplicates
    # by the MinHash LSH layer. SHA-256 still handles exact dedup; this catches
    # paraphrases and lightly-edited re-ingestions.
    minhash_threshold: float = 0.7


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def ngrams(text: str, n: int) -> set[str]:
    normalized = normalize_text(text)
    if len(normalized) < n:
        return {normalized} if normalized else set()
    return {normalized[index : index + n] for index in range(len(normalized) - n + 1)}


def ngram_overlap_ratio(left: str, right: str, *, n: int) -> float:
    left_set = ngrams(left, n)
    right_set = ngrams(right, n)
    if not left_set or not right_set:
        return 0.0
    intersection = left_set.intersection(right_set)
    smaller = min(len(left_set), len(right_set))
    return len(intersection) / smaller if smaller else 0.0


def score_text(text: str, *, min_chars: int) -> float:
    if len(text.strip()) < min_chars:
        return 0.0
    alpha = sum(ch.isalpha() for ch in text)
    ratio = alpha / max(len(text), 1)
    length_bonus = min(len(text.strip()) / (min_chars * 4), 1.0)
    return round(min(1.0, 0.5 * ratio + 0.5 * length_bonus), 4)


class DataFirewall:
    def __init__(self, config: FirewallConfig | None = None) -> None:
        self.config = config or FirewallConfig()

    def evaluate(
        self,
        record: DatasetRecord,
        *,
        text: str,
        ledger: ProvenanceLedger,
        corpus_texts: Iterable[str] | None = None,
    ) -> FirewallDecision:
        scores = {"quality": score_text(text, min_chars=self.config.min_chars)}

        if not text or not text.strip():
            return FirewallDecision(DatasetStatus.REJECTED, "empty text", scores)
        if len(text.strip()) < self.config.min_chars:
            return FirewallDecision(
                DatasetStatus.REJECTED,
                f"text shorter than min_chars={self.config.min_chars}",
                scores,
            )
        if not record.source_path:
            return FirewallDecision(DatasetStatus.REJECTED, "missing provenance source_path", scores)
        if record.source_type == SourceType.SYNTHETIC and (
            not record.generator_model or not record.generator_checkpoint
        ):
            return FirewallDecision(
                DatasetStatus.REJECTED,
                "synthetic item missing generator_model or generator_checkpoint",
                scores,
            )

        if ledger.find_by_hash(record.content_hash) and record.status == DatasetStatus.CANDIDATE:
            existing = ledger.find_by_hash(record.content_hash)
            if existing and existing.id != record.id:
                return FirewallDecision(DatasetStatus.REJECTED, "duplicate content hash", scores)

        for known_text in corpus_texts or []:
            left = text[: self.config.max_ngram_compare_chars]
            right = known_text[: self.config.max_ngram_compare_chars]
            overlap = ngram_overlap_ratio(left, right, n=self.config.ngram_size)
            if overlap >= self.config.max_ngram_overlap:
                scores["ngram_overlap"] = round(overlap, 4)
                return FirewallDecision(
                    DatasetStatus.REJECTED,
                    f"ngram overlap {overlap:.2f} above threshold",
                    scores,
                )

        quality = scores["quality"]
        if quality < self.config.uncertain_score_low:
            return FirewallDecision(
                DatasetStatus.REJECTED,
                "quality score below rejection threshold",
                scores,
            )
        return FirewallDecision(
            DatasetStatus.QUARANTINE,
            "automated checks passed; explicit operator approval required",
            scores,
        )
