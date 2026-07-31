from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
DEFAULT_SPECIAL_IDS = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
BYTE_TOKEN_COUNT = 256
MIN_BPE_VOCAB_SIZE = len(SPECIAL_TOKENS) + BYTE_TOKEN_COUNT

# Tokenizer lineage versions:
#   v1 — legacy byte-level BPE; whitespace dropped before merging, so the
#         model never sees word boundaries inside BPE. Decode relies on the
#         ``_reconstruct_spacing`` heuristic to re-glue words.
#   v2 — space-prefix BPE (GPT-2 style): each word carries its leading space
#         (U+0020 → byte ``<b20>``) into the byte stream, so merges naturally
#         produce tokens like " de", " the". Decode no longer needs the
#         heuristic because spacing is baked into the vocabulary.
TOKENIZER_VERSION_V1 = "v1"
TOKENIZER_VERSION_V2 = "v2"


@dataclass(frozen=True)
class TokenizerMetadata:
    name: str
    vocab_size: int
    lineage: str = "F51-owned BPE trained from local corpus"
    version: str = TOKENIZER_VERSION_V1
    space_prefix: bool = False


class F51BPETokenizer:
    """Byte-level BPE tokenizer trained from scratch on F51-owned text only."""

    def __init__(
        self,
        *,
        vocab: dict[str, int] | None = None,
        merges: list[tuple[str, str]] | None = None,
        metadata: TokenizerMetadata | None = None,
        space_prefix: bool | None = None,
    ) -> None:
        self.vocab = vocab or dict(DEFAULT_SPECIAL_IDS)
        self.id_to_token = {index: token for token, index in self.vocab.items()}
        self.merges = merges or []
        self.merge_ranks = {pair: index for index, pair in enumerate(self.merges)}
        self.metadata = metadata or TokenizerMetadata(
            name="F51-BPE",
            vocab_size=len(self.vocab),
        )
        # space_prefix wins from the explicit argument, else from metadata.
        # Legacy artifacts (v1) default to False so encode/decode stay
        # bit-compatible with previously trained tokenizers.
        if space_prefix is not None:
            self.space_prefix = bool(space_prefix)
        else:
            self.space_prefix = bool(getattr(self.metadata, "space_prefix", False))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.vocab["<unk>"]

    @property
    def bos_id(self) -> int:
        return self.vocab["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.vocab["<eos>"]

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int,
        name: str = "F51-BPE",
        space_prefix: bool = True,
    ) -> "F51BPETokenizer":
        """Train a byte-level BPE tokenizer from scratch.

        ``space_prefix`` defaults to ``True`` (the v2 lineage): every word is
        prefixed with a leading space before byte encoding, so the space byte
        ``<b20>`` participates in merges and the vocabulary learns tokens such
        as ``" de"`` / ``" the"``. Pass ``space_prefix=False`` to reproduce the
        legacy v1 tokenizer (whitespace dropped before merging).
        """
        if vocab_size < MIN_BPE_VOCAB_SIZE:
            raise ValueError(
                f"vocab_size must be at least {MIN_BPE_VOCAB_SIZE} "
                "for byte-level BPE special tokens plus byte tokens."
            )

        documents = [text.strip() for text in texts if text and text.strip()]
        if not documents:
            raise ValueError("Cannot train tokenizer on empty corpus.")

        vocab = dict(DEFAULT_SPECIAL_IDS)
        next_id = len(vocab)
        for byte_value in range(BYTE_TOKEN_COUNT):
            token = f"<b{byte_value:02x}>"
            vocab[token] = next_id
            next_id += 1

        word_freq: Counter[tuple[str, ...]] = Counter()
        for document in documents:
            for word in _pretokenize_for_bpe(document, space_prefix):
                symbols = tuple(_byte_tokens(word))
                word_freq[symbols] += 1

        merges: list[tuple[str, str]] = []
        while len(vocab) < vocab_size and word_freq:
            pair_counts: Counter[tuple[str, str]] = Counter()
            for symbols, frequency in word_freq.items():
                if len(symbols) < 2:
                    continue
                for index in range(len(symbols) - 1):
                    pair_counts[(symbols[index], symbols[index + 1])] += frequency
            if not pair_counts:
                break
            best_pair, _ = pair_counts.most_common(1)[0]
            merged = best_pair[0] + best_pair[1]
            if merged in vocab:
                merges.append(best_pair)
                word_freq = _merge_pair_in_vocab(word_freq, best_pair, merged)
                continue
            vocab[merged] = next_id
            next_id += 1
            merges.append(best_pair)
            word_freq = _merge_pair_in_vocab(word_freq, best_pair, merged)

        version = TOKENIZER_VERSION_V2 if space_prefix else TOKENIZER_VERSION_V1
        metadata = TokenizerMetadata(
            name=name,
            vocab_size=len(vocab),
            version=version,
            space_prefix=space_prefix,
        )
        return cls(vocab=vocab, merges=merges, metadata=metadata)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        tokens: list[int] = []
        if add_bos:
            tokens.append(self.bos_id)

        if self.space_prefix:
            # v2: the leading space is part of each word's byte stream, so
            # merges already encode word boundaries. No manual space token.
            for unit in _pretokenize_for_bpe(text, space_prefix=True):
                for token in self._tokenize_word(unit):
                    tokens.append(self.vocab.get(token, self.unk_id))
        else:
            # v1 legacy: insert an explicit <b20> between words so the model
            # at least sees boundaries at encode time.
            words = _pretokenize(text)
            for i, word in enumerate(words):
                if i > 0 and not words[i - 1].endswith("\n") and not word.startswith("\n"):
                    space_id = self.vocab.get("<b20>", self.unk_id)
                    if space_id != self.unk_id:
                        tokens.append(space_id)
                for token in self._tokenize_word(word):
                    tokens.append(self.vocab.get(token, self.unk_id))

        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, token_ids: Iterable[int], *, skip_special: bool = True) -> str:
        reverse = {index: token for token, index in self.vocab.items()}
        pieces: list[str] = []
        for token_id in token_ids:
            token = reverse.get(int(token_id), "<unk>")
            if skip_special and token in SPECIAL_TOKENS:
                continue
            pieces.append(_token_to_text(token))
        raw = "".join(pieces).strip()
        # v2 bakes spacing into the vocabulary; the heuristic splitter is only
        # needed for the legacy v1 tokenizer that dropped whitespace.
        if self.space_prefix:
            return raw
        return _reconstruct_spacing(raw)

    def save(self, directory: str | Path) -> Path:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        (output / "vocab.json").write_text(
            json.dumps(self.vocab, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        merge_lines = [" ".join(pair) for pair in self.merges]
        (output / "merges.txt").write_text("\n".join(merge_lines) + ("\n" if merge_lines else ""), encoding="utf-8")
        (output / "config.json").write_text(
            json.dumps(
                {
                    "name": self.metadata.name,
                    "vocab_size": self.vocab_size,
                    "lineage": self.metadata.lineage,
                    "version": self.metadata.version,
                    "space_prefix": self.metadata.space_prefix,
                    "special_tokens": list(SPECIAL_TOKENS),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return output

    @classmethod
    def load(cls, directory: str | Path) -> "F51BPETokenizer":
        root = Path(directory)
        vocab = json.loads((root / "vocab.json").read_text(encoding="utf-8"))
        merges_raw = (root / "merges.txt").read_text(encoding="utf-8").splitlines()
        merges = [tuple(line.split()) for line in merges_raw if line.strip()]
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        metadata = TokenizerMetadata(
            name=config["name"],
            vocab_size=int(config["vocab_size"]),
            lineage=config.get("lineage", "F51-owned BPE trained from local corpus"),
            version=config.get("version", TOKENIZER_VERSION_V1),
            space_prefix=bool(config.get("space_prefix", False)),
        )
        return cls(vocab=vocab, merges=merges, metadata=metadata)

    def _tokenize_word(self, word: str) -> list[str]:
        symbols = _byte_tokens(word)
        if not symbols:
            return []
        while len(symbols) >= 2:
            ranks = [
                self.merge_ranks.get((symbols[index], symbols[index + 1]))
                for index in range(len(symbols) - 1)
            ]
            valid = [rank for rank in ranks if rank is not None]
            if not valid:
                break
            best_rank = min(valid)
            pair = self.merges[best_rank]
            symbols = _apply_merge(symbols, pair)
        return symbols


def _pretokenize(text: str) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    return re.findall(r"\S+|\n", normalized)


def _pretokenize_for_bpe(text: str, space_prefix: bool) -> list[str]:
    """Pre-tokenize text into BPE units.

    When ``space_prefix`` is True (v2), every whitespace-delimited word is
    returned with a leading space so the space byte (``<b20>``) flows into the
    byte stream and participates in BPE merges. Newlines are preserved as
    their own unit. When False, this is identical to :func:`_pretokenize`.
    """
    units = _pretokenize(text)
    if not space_prefix:
        return units
    spaced: list[str] = []
    for unit in units:
        if unit == "\n":
            spaced.append(unit)
        else:
            spaced.append(" " + unit)
    return spaced


def _byte_tokens(word: str) -> list[str]:
    return [f"<b{byte_value:02x}>" for byte_value in word.encode("utf-8")]


def _token_to_text(token: str) -> str:
    parts = re.findall(r"<b[0-9a-f]{2}>", token)
    if parts:
        return bytes(int(part[2:-1], 16) for part in parts).decode("utf-8", errors="replace")
    return token


def _apply_merge(symbols: list[str], pair: tuple[str, str]) -> list[str]:
    merged: list[str] = []
    index = 0
    while index < len(symbols):
        if index < len(symbols) - 1 and (symbols[index], symbols[index + 1]) == pair:
            merged.append(pair[0] + pair[1])
            index += 2
        else:
            merged.append(symbols[index])
            index += 1
    return merged


def _merge_pair_in_vocab(
    word_freq: Counter[tuple[str, ...]],
    pair: tuple[str, str],
    merged: str,
) -> Counter[tuple[str, ...]]:
    updated: Counter[tuple[str, ...]] = Counter()
    for symbols, frequency in word_freq.items():
        updated[_tuple_apply_merge(symbols, pair, merged)] += frequency
    return updated


def _tuple_apply_merge(
    symbols: tuple[str, ...],
    pair: tuple[str, str],
    merged: str,
) -> tuple[str, ...]:
    return tuple(_apply_merge(list(symbols), pair))


def _reconstruct_spacing(text: str) -> str:
    """Reconstruct word spacing in BPE-decoded text.

    Since the tokenizer was trained without whitespace information,
    decoded text comes out glued together. This uses aggressive
    heuristics to split at word boundaries.
    """
    import re

    # Step 1: Split at lowercase→uppercase (most common word boundary)
    text = re.sub(r"([a-zà-ú])([A-ZÀ-Ú])", r"\1 \2", text)

    # Step 2: Split at letter→digit and digit→letter
    text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)

    # Step 3: Space after punctuation
    text = re.sub(r"([.!?])([^\s\d\-])", r"\1 \2", text)

    # Step 4: Space after comma, colon, semicolon
    text = re.sub(r"([,:;])([^\s\d])", r"\1 \2", text)

    # Step 5: Space before opening parens/brackets
    text = re.sub(r"([^\s(\[\-])([\(\[\{])", r"\1 \2", text)

    # Step 6: Space after closing parens/brackets
    text = re.sub(r"([\)\]\}])([^\s,.;:!?\)\]\}])", r"\1 \2", text)

    # Step 7: Split at known English word patterns (common short words glued together)
    # e.g., "witha" → "with a", "andthe" → "and the", "ofthe" → "of the"
    common_splits = [
        (r'\b(with)([a-z]{2,})', r'\1 \2'),
        (r'\b(and)([a-z]{2,})', r'\1 \2'),
        (r'\b(the)([a-z]{2,})', r'\1 \2'),
        (r'\b(for)([a-z]{2,})', r'\1 \2'),
        (r'\b(that)([a-z]{2,})', r'\1 \2'),
        (r'\b(this)([a-z]{2,})', r'\1 \2'),
        (r'\b(from)([a-z]{2,})', r'\1 \2'),
        (r'\b(are)([a-z]{2,})', r'\1 \2'),
        (r'\b(has)([a-z]{2,})', r'\1 \2'),
        (r'\b(was)([a-z]{2,})', r'\1 \2'),
        (r'\b(is)([a-z]{2,})', r'\1 \2'),
        (r'\b(not)([a-z]{2,})', r'\1 \2'),
        (r'\b(can)([a-z]{2,})', r'\1 \2'),
        (r'([a-z]{2,})(ing)\b', r'\1 \2'),
        (r'([a-z]{2,})(tion)\b', r'\1 \2'),
    ]
    for pattern, repl in common_splits:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # Step 8: Clean up multiple spaces
    text = re.sub(r" {2,}", " ", text)

    return text.strip()
