#!/usr/bin/env python3
"""F51 Tokenizer baseline metrics (FASE 0 — diagnostic).

Measures two quality signals on the CURRENT tokenizer BEFORE any retokenization:

  1. Fertility — mean tokens per whitespace-delimited word. Lower is better.
     A high fertility (e.g. > 2.0 for Portuguese) means the tokenizer is
     fragmenting words into many byte pieces, wasting context.

  2. STRR — Single-Token Root Recall. For a curated root vocabulary
     (common PT/EN word roots), the fraction that encode to a SINGLE token.
     High STRR means the model spends one token per common word instead of
     several. Reported separately for " de"-style space-prefixed roots vs
     bare roots, so we can compare v1 (no space info) against a future v2.

The script loads `tokenizer/f51_bpe_80k` by default and writes the result
to `workspace/runtime/baselines/tokenizer_metrics.json`. It never mutates the tokenizer
or the corpus.

Usage:
    python -m tools.measure_tokenizer_metrics
    python -m tools.measure_tokenizer_metrics --tokenizer tokenizer/f51_bpe_80k \\
        --corpus data/approved --samples 10000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from f51_darwin.data import discover_corpus_files, load_text_documents
from f51_darwin.tokenizer import F51BPETokenizer

# Default tokenizer artifact shipped with the repo.
DEFAULT_TOKENIZER_DIR = ROOT / "tokenizer" / "f51_bpe_80k"
DEFAULT_BASELINE_DIR = ROOT / "workspace" / "runtime" / "baselines"

# Curated PT/EN roots used for STRR. These are high-frequency word roots that a
# well-trained PT-heavy tokenizer should ideally encode as a single merge.
STRR_ROOTS_PT = [
    "que", "nao", "uma", "para", "com", "por", "mas", "seu", "sua", "mais",
    "este", "essa", "isso", "como", "tudo", "sempre", "ainda", "quando",
    "onde", "entre", "depois", "antes", "sobre", "ate", "muito", "pouco",
    "grande", "pequeno", "tempo", "vida", "mundo", "casa", "ano", "dia",
    "homem", "mulher", "pessoa", "historia", "trabalho", "governo", "pais",
    "cidade", "empresa", "grupo", "forma", "caso", "parte", "numero",
]
STRR_ROOTS_EN = [
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "her",
    "was", "one", "our", "out", "has", "have", "from", "they", "this",
    "that", "with", "have", "will", "your", "which", "their", "what",
    "about", "would", "there", "could", "other", "more", "when", "time",
    "very", "what", "people", "world", "great", "because", "also",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _iter_sample_documents(corpus_dir: Path, max_docs: int) -> list[str]:
    """Load up to ``max_docs`` documents from a corpus directory.

    Falls back to the repo's tiny test fixture when the corpus is missing so
    the script remains runnable in a fresh checkout (the fertility number will
    just be less representative).
    """
    if corpus_dir.exists():
        files = discover_corpus_files(corpus_dir)
        docs: list[str] = []
        for path in files:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                if "\n\n---\n\n" in text:
                    docs.extend(s.strip() for s in text.split("\n\n---\n\n") if s.strip())
                else:
                    docs.append(text)
            if len(docs) >= max_docs:
                break
        if docs:
            return docs[:max_docs]
    fixture = ROOT / "src" / "tests" / "fixtures" / "corpus"
    if fixture.exists():
        return load_text_documents(fixture)
    return []


def measure_fertility(tokenizer: F51BPETokenizer, documents: list[str]) -> dict[str, float]:
    """Mean tokens per word across the sample (excluding pure whitespace)."""
    total_tokens = 0
    total_words = 0
    for doc in documents:
        words = [w for w in doc.split() if w.strip()]
        if not words:
            continue
        ids = tokenizer.encode(doc, add_eos=False)
        total_tokens += len(ids)
        total_words += len(words)
    if total_words == 0:
        return {"fertility": 0.0, "tokens": 0, "words": 0}
    return {
        "fertility": round(total_tokens / total_words, 4),
        "tokens": total_tokens,
        "words": total_words,
    }


def measure_strr(tokenizer: F51BPETokenizer) -> dict[str, object]:
    """Single-Token Root Recall for a curated root set.

    Reports the fraction of roots that encode to exactly one token, both for
    bare roots ("de") and space-prefixed roots (" de"). The space-prefixed
    number is the target for the v2 space-prefix tokenizer: once words carry
    their leading space into BPE merges, " de" should collapse to one token.
    """
    space_prefix = bool(getattr(tokenizer.metadata, "space_prefix", False))

    def _single_token_fraction(roots: list[str]) -> tuple[float, int, int]:
        hits = 0
        checked = 0
        for root in roots:
            ids = tokenizer.encode(root, add_eos=False)
            checked += 1
            if len(ids) == 1:
                hits += 1
        return (hits / checked if checked else 0.0, hits, checked)

    bare_pt = _single_token_fraction(STRR_ROOTS_PT)
    bare_en = _single_token_fraction(STRR_ROOTS_EN)

    # Space-prefixed STRR is only meaningful when the tokenizer was trained
    # with space_prefix; otherwise encoding " de" still fragments.
    if space_prefix:
        sp_pt = _single_token_fraction([" " + r for r in STRR_ROOTS_PT])
        sp_en = _single_token_fraction([" " + r for r in STRR_ROOTS_EN])
    else:
        sp_pt = (0.0, 0, 0)
        sp_en = (0.0, 0, 0)

    all_roots = len(STRR_ROOTS_PT) + len(STRR_ROOTS_EN)
    bare_hits = bare_pt[1] + bare_en[1]
    sp_hits = sp_pt[1] + sp_en[1]
    return {
        "strr_bare": round(bare_hits / all_roots, 4) if all_roots else 0.0,
        "strr_space_prefixed": round(sp_hits / all_roots, 4) if all_roots else 0.0,
        "strr_bare_pt": round(bare_pt[0], 4),
        "strr_bare_en": round(bare_en[0], 4),
        "strr_space_prefixed_pt": round(sp_pt[0], 4),
        "strr_space_prefixed_en": round(sp_en[0], 4),
        "space_prefix_enabled": space_prefix,
    }


def measure_tokenizer(tokenizer: F51BPETokenizer, documents: list[str]) -> dict[str, object]:
    fertility = measure_fertility(tokenizer, documents)
    strr = measure_strr(tokenizer)
    return {
        "measured_at": utc_now_iso(),
        "tokenizer_name": tokenizer.metadata.name,
        "tokenizer_version": getattr(tokenizer.metadata, "version", "v1"),
        "space_prefix": bool(getattr(tokenizer.metadata, "space_prefix", False)),
        "vocab_size": tokenizer.vocab_size,
        "sample_documents": len(documents),
        "fertility": fertility,
        "strr": strr,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        default=str(DEFAULT_TOKENIZER_DIR),
        help="Tokenizer artifact directory (default: tokenizer/f51_bpe_80k).",
    )
    parser.add_argument(
        "--corpus",
        default="data/approved",
        help="Corpus directory used for the fertility sample.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Maximum documents to sample for fertility (default: 10000).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_BASELINE_DIR / "tokenizer_metrics.json"),
        help="Output JSON path.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    tokenizer_dir = Path(args.tokenizer)
    if not tokenizer_dir.exists():
        print(f"Tokenizer not found at {tokenizer_dir}", file=sys.stderr)
        return 1
    tokenizer = F51BPETokenizer.load(tokenizer_dir)

    corpus_dir = Path(args.corpus)
    if not corpus_dir.is_absolute():
        corpus_dir = ROOT / corpus_dir
    documents = _iter_sample_documents(corpus_dir, args.samples)
    if not documents:
        print("No documents available for fertility measurement.", file=sys.stderr)
        return 1

    report = measure_tokenizer(tokenizer, documents)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nBaseline salvo em: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
