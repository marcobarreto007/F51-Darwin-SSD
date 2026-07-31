from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import yaml

from f51_darwin.data import load_text_documents, resolve_corpus_dir
from f51_darwin.tokenizer import F51BPETokenizer


DEFAULT_TOKENIZER_CONFIG = {
    "corpus_dir": "data/corpus",
    "tokenizer_dir": "tokenizer/f51_bpe",
    "tokenizer_training": {
        "vocab_size": 80000,
        "name": "F51-BPE",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the F51-owned BPE tokenizer from a local text corpus.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional training YAML with corpus and tokenizer defaults. Inline defaults are used when omitted.",
    )
    parser.add_argument("--corpus", default=None, help="Override corpus directory.")
    parser.add_argument("--output", default=None, help="Directory to save tokenizer artifacts.")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Target vocabulary size. Byte-level BPE minimum is 260.",
    )
    parser.add_argument("--name", default=None, help="Tokenizer display name.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError(f"Tokenizer config must be a mapping: {config_path}")
    else:
        config = dict(DEFAULT_TOKENIZER_CONFIG)
    corpus_dir = resolve_corpus_dir(ROOT, args.corpus or config.get("corpus_dir"))
    output_dir = Path(args.output or config.get("tokenizer_dir", "tokenizer/f51_bpe"))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    tokenizer_cfg = config.get("tokenizer_training", {})
    vocab_size = args.vocab_size or int(tokenizer_cfg.get("vocab_size", 80000))
    name = args.name or str(tokenizer_cfg.get("name", "F51-BPE"))

    documents = load_text_documents(corpus_dir)
    tokenizer = F51BPETokenizer.train(documents, vocab_size=vocab_size, name=name)
    saved = tokenizer.save(output_dir)
    report = {
        "corpus_dir": str(corpus_dir),
        "documents": len(documents),
        "output_dir": str(saved),
        "vocab_size": tokenizer.vocab_size,
        "name": tokenizer.metadata.name,
        "lineage": tokenizer.metadata.lineage,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
