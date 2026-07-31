from __future__ import annotations

import json
from dataclasses import dataclass
from math import gcd
from numbers import Integral
from pathlib import Path
from typing import Iterator, Protocol

import torch

from f51_darwin.data_factory import assert_training_corpus_allowed, is_blocked_training_path


DEFAULT_CORPUS_DIRS = ("data/approved", "data/corpus", "datasets/corpus", "datasets")
SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".text"}
SUPPORTED_JSONL_EXTENSIONS = {".jsonl", ".jsonl.gz"}
SAMPLER_NAME = "f51_causal_lm_sampler"
SAMPLER_VERSIONS = {
    "sequential": 1,
    "permuted_blocks": 1,
}
_UINT64_MASK = (1 << 64) - 1


class TokenizerProtocol(Protocol):
    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...

    @property
    def vocab_size(self) -> int: ...


@dataclass(frozen=True)
class CorpusStats:
    documents: int
    characters: int
    token_ids: int
    files: int


def resolve_corpus_dir(
    root: Path,
    preferred: str | None = None,
    *,
    debug_candidates: bool = False,
) -> Path:
    if preferred:
        candidate = Path(preferred)
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"Corpus directory not found: {candidate}")
        assert_training_corpus_allowed(candidate, root, debug_candidates=debug_candidates)
        return candidate

    for relative in DEFAULT_CORPUS_DIRS:
        candidate = root / relative
        if candidate.exists() and candidate.is_dir():
            assert_training_corpus_allowed(candidate, root, debug_candidates=debug_candidates)
            return candidate
    raise FileNotFoundError(
        "No corpus directory found. Create one of: "
        + ", ".join(str(root / item) for item in DEFAULT_CORPUS_DIRS)
    )


def discover_corpus_files(corpus_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_TEXT_EXTENSIONS or suffix in SUPPORTED_JSONL_EXTENSIONS:
            files.append(path)
    return files


def load_text_documents(corpus_dir: Path) -> list[str]:
    documents: list[str] = []
    for path in discover_corpus_files(corpus_dir):
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_TEXT_EXTENSIONS:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                if "\n\n---\n\n" in text:
                    for segment in text.split("\n\n---\n\n"):
                        segment = segment.strip()
                        if segment:
                            documents.append(segment)
                else:
                    documents.append(text)
            continue
        if suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                text = str(payload.get("text", "")).strip()
                if text:
                    documents.append(text)
    if not documents:
        raise ValueError(f"No readable documents found under {corpus_dir}")
    return documents


def document_sample_weight(filename: str) -> float:
    """Training oversampling weight by filename.

    Matemática pura recebe o maior peso — é a ciência fundamental.
    Identidade F51 e família também são priorizadas.
    """
    name = filename.lower()

    # ── Matemática pura: peso máximo ──
    # Ordem importa: checa substrings específicas antes das genéricas
    if "number_theory" in name:
        return 5.0  # rainha da matemática
    if "linear_algebra" in name:
        return 4.0  # base de tudo
    if "algebra" in name or "abstract_algebra" in name:
        return 5.0  # estrutura fundamental
    if "calculus" in name or "analysis" in name:
        return 5.0  # linguagem do universo
    if "geometry" in name or "topology" in name:
        return 5.0  # forma e espaço
    if "combinatorics" in name or "graph_theory" in name:
        return 4.0  # arte de contar

    # ── Matemática aplicada ──
    if "physics" in name or "math_physics" in name:
        return 3.5  # matemática em ação
    if "statistics" in name or "probability" in name:
        return 3.0  # inferência
    if "trigonometry" in name or "differential" in name:
        return 3.0

    # ── Matemática geral / descobertas ──
    if "math_" in name or "wolfram" in name:
        return 3.0

    # ── Identidade F51 ──
    if "identity" in name or name.startswith("f51_identity"):
        return 5.0

    # ── Família / clã ──
    if "familia" in name or "family" in name or "clan" in name or "barreto" in name:
        return 5.0

    # ── Filosofia ──
    if "olavo" in name:
        return 3.0

    return 1.0


def tokenize_documents(
    documents: list[str],
    tokenizer: TokenizerProtocol,
    *,
    add_eos: bool = True,
    weights: list[float] | None = None,
) -> list[int]:
    token_ids: list[int] = []
    for i, document in enumerate(documents):
        encoded = tokenizer.encode(document, add_eos=add_eos)
        if not encoded:
            continue
        repeat = 1
        if weights is not None and i < len(weights):
            repeat = max(1, int(round(weights[i])))
        for _ in range(repeat):
            token_ids.extend(encoded)
    if len(token_ids) < 2:
        raise ValueError("Tokenized corpus is too small for causal LM training.")
    return token_ids


def tokenize_to_file(
    corpus_dir: Path,
    tokenizer,
    output_path: Path,
    *,
    add_eos: bool = True,
    use_document_weights: bool = True,
    chunk_size: int = 1_000_000,
) -> int:
    """Tokenize corpus to disk in chunks — handles 100+ GB corpora.

    Returns total token count. Output is a binary file of int32 token IDs.
    """
    import struct
    files = discover_corpus_files(corpus_dir)
    total = 0
    buffer = []

    with open(output_path, 'wb') as f:
        for path in files:
            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_TEXT_EXTENSIONS:
                continue
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                continue

            weight = document_sample_weight(path.name) if use_document_weights else 1.0
            repeat = max(1, int(round(weight)))

            # Split by separator for multi-doc files
            if "\n\n---\n\n" in text:
                segments = [s.strip() for s in text.split("\n\n---\n\n") if s.strip()]
            else:
                segments = [text]

            for segment in segments:
                encoded = tokenizer.encode(segment, add_eos=add_eos)
                if not encoded:
                    continue
                for _ in range(repeat):
                    buffer.extend(encoded)
                    total += len(encoded)

                    if len(buffer) >= chunk_size:
                        f.write(struct.pack(f'{len(buffer)}i', *buffer))
                        buffer = []

        if buffer:
            f.write(struct.pack(f'{len(buffer)}i', *buffer))

    return total


def _yield_documents(corpus_dir: Path, use_document_weights: bool):
    """Yield (text, weight) tuples from corpus files. One doc per file."""
    for path in discover_corpus_files(corpus_dir):
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_TEXT_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        weight = document_sample_weight(path.name) if use_document_weights else 1.0
        yield text, weight


def tokenize_corpus_dir(
    corpus_dir: Path,
    tokenizer: TokenizerProtocol,
    *,
    add_eos: bool = True,
    use_document_weights: bool = False,
    mmap_path: str | Path | None = None,
):
    """Load and tokenize corpus files, optionally oversampling identity/family docs.

    Se mmap_path é fornecido, salva os tokens em disco (memory-mapped)
    em vez de carregar tudo em RAM. Essencial para corpus > 1 GB.
    """
    import numpy as np

    # Streaming tokenization — processa um doc por vez
    if mmap_path is not None:
        mmap_path = Path(mmap_path)
        total_tokens = 0

        # Primeiro passe: contar tokens
        print(f"  Contando tokens (passe 1)...")
        for i, (doc, weight) in enumerate(_yield_documents(corpus_dir, use_document_weights)):
            encoded = tokenizer.encode(doc, add_eos=add_eos)
            repeat = max(1, int(round(weight)))
            total_tokens += len(encoded) * repeat
            if (i + 1) % 10000 == 0:
                print(f"\r  ... {i+1:,} docs contados", end="", flush=True)
        print(f"\r  Total: {total_tokens:,} tokens ({total_tokens/1e6:.1f}M)")

        # Segundo passe: escrever para arquivo memory-mapped
        print(f"  Escrevendo tokens para {mmap_path}...")
        dtype = np.int32 if total_tokens < 2**31 else np.int64
        mmap = np.memmap(str(mmap_path), dtype=dtype, mode='w+', shape=(total_tokens,))

        pos = 0
        for doc, weight in _yield_documents(corpus_dir, use_document_weights):
            encoded = tokenizer.encode(doc, add_eos=add_eos)
            if not encoded:
                continue
            repeat = max(1, int(round(weight)))
            for _ in range(repeat):
                end = pos + len(encoded)
                if end > total_tokens:
                    break
                mmap[pos:end] = np.array(encoded, dtype=dtype)
                pos = end
        
        mmap.flush()
        # Retorna como ints (o mmap fica no disco)
        # CausalLMDataLoader precisa de uma lista, então convertemos
        # mas só se couber na RAM (corpus pequeno)
        # Para corpus grande, usamos o mmap diretamente
        return mmap  # type: ignore — retorna numpy memmap em vez de list

    # Modo legado: tudo em RAM
    import array
    token_ids = array.array('i')
    for doc, weight in _yield_documents(corpus_dir, use_document_weights):
        encoded = tokenizer.encode(doc, add_eos=add_eos)
        if not encoded:
            continue
        repeat = max(1, int(round(weight)))
        for _ in range(repeat):
            token_ids.extend(encoded)
    
    if len(token_ids) < 2:
        raise ValueError(f"No readable documents found under {corpus_dir}")
    return token_ids


def corpus_stats(corpus_dir: Path, token_ids: list[int]) -> CorpusStats:
    num_docs = 0
    characters = 0
    for doc, _ in _yield_documents(corpus_dir, False):
        num_docs += 1
        characters += len(doc)
    return CorpusStats(
        documents=num_docs,
        characters=characters,
        token_ids=len(token_ids),
        files=len(discover_corpus_files(corpus_dir)),
    )


class CausalLMDataLoader:
    """Deterministic causal LM batch sampler over a flat token stream.

    Suporta tanto list[int] (token_ids em RAM) quanto numpy memmap (token_ids em disco).

    ``sequential`` v1 preserves the historical start-position formula.
    ``permuted_blocks`` v1 visits a seed-derived permutation of disjoint
    physical blocks. Both modes are stateless with respect to sampling:
    ``seed`` plus ``step`` fully determine the next batch.
    """

    def __init__(
        self,
        token_ids: list[int] | "np.ndarray",  # type: ignore
        *,
        block_size: int,
        batch_size: int,
        seed: int = 51,
        device: torch.device | str = "cpu",
        sampler_mode: str = "sequential",
        sampler_version: int | None = None,
    ) -> None:
        import numpy as np

        if block_size < 2:
            raise ValueError("block_size must be at least 2.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if len(token_ids) <= block_size:
            raise ValueError("token stream must be longer than block_size.")
        if not isinstance(seed, Integral) or isinstance(seed, bool):
            raise ValueError("seed must be an integer.")
        if sampler_mode not in SAMPLER_VERSIONS:
            supported = ", ".join(sorted(SAMPLER_VERSIONS))
            raise ValueError(
                f"unsupported sampler_mode {sampler_mode!r}; expected one of: {supported}."
            )
        expected_version = SAMPLER_VERSIONS[sampler_mode]
        if sampler_version is None:
            sampler_version = expected_version
        if (
            not isinstance(sampler_version, Integral)
            or isinstance(sampler_version, bool)
            or int(sampler_version) != expected_version
        ):
            raise ValueError(
                f"unsupported sampler_version {sampler_version!r} "
                f"for sampler_mode {sampler_mode!r}; expected {expected_version}."
            )

        # Suporte a memmap/numpy/Tensor — mantém dtype nativo, converte só o batch
        if isinstance(token_ids, torch.Tensor):
            self.token_ids = token_ids.long()
        elif isinstance(token_ids, np.ndarray) or getattr(token_ids, "is_virtual_token_sequence", False):
            self.token_ids = token_ids
        else:
            self.token_ids = torch.tensor(token_ids, dtype=torch.long)

        self.block_size = block_size
        self.batch_size = batch_size
        self.seed = int(seed)
        self.device = torch.device(device)
        self.sampler_mode = sampler_mode
        self.sampler_version = int(sampler_version)
        self._step = 0
        self._token_count = (
            len(self.token_ids)
            if isinstance(self.token_ids, np.ndarray)
            or getattr(self.token_ids, "is_virtual_token_sequence", False)
            else int(self.token_ids.numel())
        )
        self._max_start = self._token_count - block_size - 1
        self._block_count = self._token_count // block_size
        if self.sampler_mode == "permuted_blocks" and self._block_count < self.batch_size:
            raise ValueError(
                "permuted_blocks requires at least batch_size full blocks "
                "to avoid in-batch collisions."
            )
        self._permutation_offset = self._mixed_seed(0x9E3779B97F4A7C15) % self._block_count
        self._permutation_stride = self._coprime_stride()

    @property
    def step(self) -> int:
        return self._step

    @property
    def sampler_identity(self) -> dict[str, int | str]:
        """Return the stable, checkpointable identity of the sampling contract."""

        return {
            "name": SAMPLER_NAME,
            "mode": self.sampler_mode,
            "version": self.sampler_version,
            "seed": self.seed,
            "block_size": self.block_size,
            "batch_size": self.batch_size,
            "token_count": self._token_count,
            "block_count": self._block_count,
        }

    def set_step(self, step: int) -> None:
        if step < 0:
            raise ValueError("step must be non-negative.")
        self._step = step

    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:
            yield self.next_batch()

    def next_batch(self) -> torch.Tensor:
        starts = self._starts_for_step(self._step)
        batch = self._batch_from_starts(starts)
        self._step += 1
        return batch.to(self.device)

    def take_eval_batch(self, *, eval_seed: int = 999) -> torch.Tensor:
        starts = [(eval_seed + row * 29) % self._max_start for row in range(self.batch_size)]
        batch = self._batch_from_starts(starts)
        return batch.to(self.device)

    def _starts_for_step(self, step: int) -> list[int]:
        if self.sampler_mode == "sequential":
            return [
                (step * self.block_size + row * 23 + self.seed) % self._max_start
                for row in range(self.batch_size)
            ]

        first_ordinal = step * self.batch_size
        return [
            self._permuted_block_start(first_ordinal + row)
            for row in range(self.batch_size)
        ]

    def _permuted_block_start(self, sample_ordinal: int) -> int:
        logical_block = sample_ordinal % self._block_count
        physical_block = (
            self._permutation_offset + logical_block * self._permutation_stride
        ) % self._block_count
        return physical_block * self.block_size

    def _mixed_seed(self, salt: int) -> int:
        """SplitMix64 finalizer used as a pure integer mixer, not as an RNG."""

        value = (self.seed & _UINT64_MASK) ^ salt
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & _UINT64_MASK
        value = (value ^ (value >> 27)) * 0x94D049BB133111EB & _UINT64_MASK
        return value ^ (value >> 31)

    def _coprime_stride(self) -> int:
        """Choose a deterministic non-trivial stride for an affine permutation."""

        if self._block_count <= 2:
            return 1

        candidate = self._mixed_seed(0xD1B54A32D192ED03) % self._block_count
        for allow_adjacent in (False, True):
            for delta in range(self._block_count):
                stride = (candidate + delta) % self._block_count
                if stride == 0 or gcd(stride, self._block_count) != 1:
                    continue
                if not allow_adjacent and stride in (1, self._block_count - 1):
                    continue
                return stride
        raise RuntimeError("could not derive a coprime sampler stride.")

    def _batch_from_starts(self, starts: list[int]) -> torch.Tensor:
        import numpy as np

        if isinstance(self.token_ids, np.ndarray) or getattr(self.token_ids, "is_virtual_token_sequence", False):
            rows = [
                np.array(self.token_ids[start : start + self.block_size], dtype=np.int64, copy=True)
                for start in starts
            ]
            return torch.from_numpy(np.stack(rows, axis=0))

        return torch.stack(
            [self.token_ids[start : start + self.block_size].long() for start in starts],
            dim=0,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Weighted multi‑corpus loader  (curriculum learning)
# ═══════════════════════════════════════════════════════════════════════════

class WeightedCorpusLoader:
    """Samples from multiple token sources with configurable weights.

    Designed for curriculum learning: identity (5×) > philosophy (3×) > literature (1×).

    Each step:
      1. Select a source with probability proportional to its weight.
      2. Sample a random block from that source (deterministic given seed + step).
    """

    def __init__(
        self,
        sources: list[tuple[object, float]],   # [(token_ids, weight), ...]
        *,
        block_size: int,
        batch_size: int,
        seed: int = 51,
        device: torch.device | str = "cpu",
    ) -> None:
        import numpy as np
        if not sources:
            raise ValueError("at least one source required")
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.block_size = block_size
        self.batch_size = batch_size
        self.seed = seed
        self.device = torch.device(device)

        # Normalise weights → cumulative distribution for sampling
        total_w = sum(w for _, w in sources)
        self._cum_weights: list[float] = []
        acc = 0.0
        for _, w in sources:
            acc += w / total_w
            self._cum_weights.append(acc)

        # Store sources + pre‑compute max start indices
        self._sources: list[dict[str, object]] = []
        for token_ids, _w in sources:
            if isinstance(token_ids, np.ndarray) or getattr(token_ids, "is_virtual_token_sequence", False):
                count = len(token_ids)
            else:
                count = int(token_ids.numel() if isinstance(token_ids, torch.Tensor) else len(token_ids))
            self._sources.append({
                "ids": token_ids,
                "count": count,
                "max_start": count - block_size - 1,
            })

        self._step = 0

        # Print curriculum
        total_tokens = sum(s["count"] for s in self._sources)
        print(f"  📚 Curriculum: {len(sources)} fonte(s), {total_tokens/1e6:.1f}M tokens")
        for i, s in enumerate(self._sources):
            w = self._cum_weights[i] - (self._cum_weights[i-1] if i > 0 else 0.0)
            print(f"     {i+1}. {s['count']/1e6:.1f}M tokens  peso={w:.1f}  ({w*100:.0f}%)")

    @property
    def step(self) -> int:
        return self._step

    def set_step(self, step: int) -> None:
        if step < 0:
            raise ValueError("step must be non-negative")
        self._step = step

    def _pick_source(self, step: int) -> int:
        """Deterministic source selection."""
        rng_val = (step * 71 + self.seed * 13) % 10007 / 10007.0
        for i, cum in enumerate(self._cum_weights):
            if rng_val < cum:
                return i
        return len(self._sources) - 1

    def next_batch(self) -> torch.Tensor:
        import numpy as np

        rows = []
        for batch_idx in range(self.batch_size):
            # Deterministic source selection per row
            si = self._pick_source(self._step * self.batch_size + batch_idx)
            src = self._sources[si]
            ids = src["ids"]

            # Deterministic position
            pos = (self._step * 17 + batch_idx * 23 + self.seed + si * 31) % src["max_start"]

            if isinstance(ids, np.ndarray) or getattr(ids, "is_virtual_token_sequence", False):
                row = np.array(ids[int(pos) : int(pos) + self.block_size], dtype=np.int64, copy=True)
            else:
                row = ids[int(pos) : int(pos) + self.block_size]

            rows.append(row)

        self._step += 1

        if isinstance(rows[0], np.ndarray):
            return torch.from_numpy(np.stack(rows, axis=0)).to(self.device)
        return torch.stack([r.long() for r in rows], dim=0).to(self.device)
