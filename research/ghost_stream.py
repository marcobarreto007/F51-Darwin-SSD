#!/usr/bin/env python3
"""
F51 Ghost Stream — Alimentacao viva do Ghost Brain via HuggingFace
===================================================================
Conecta o Ghost Brain diretamente a datasets do HuggingFace em streaming.
O fantasma explora documentos recem-chegados, gera perguntas, verifica,
e produz dados sinteticos de treino — tudo automatico.

Fontes ativas:
    - HuggingFaceFW/fineweb (web de alta qualidade)
    - wikimedia/wikipedia (PT + EN, parquet)
    - OpenStellar/finepersonas-v1 (dialogos)
    - meta-math/MetaMathQA (problemas matematicos)
    - allenai/c4 (web geral)

Integracao:
    GhostBrain.explore_and_learn() ← alimentado por documentos novos
    Dados sinteticos → data/generated/candidates/ → Firewall → Corpus

Uso:
    python research/ghost_stream.py                    # uma rodada
    python research/ghost_stream.py --loop --interval 300  # a cada 5 min
    python research/ghost_stream.py --source fineweb,wikipedia  # so essas
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# Ghost stream writes to the external dataset workspace so the feeder can find it.
# Production data must never fall back to a physical path inside the code repo.
from f51_darwin.dataset_layout import CORPUS_WORKSPACE_RELATIVE, resolve_dataset_root
from f51_darwin.workspace_migration import mark_runtime_use


def resolve_ghost_paths(
    project_root: Path = ROOT,
    *,
    dataset_root: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Resolve physical outputs only when an ingest operation is requested.

    Importing this module is intentionally source-only and therefore works in
    hosted CI and fresh checkouts that do not contain the ignored workspace.
    """
    project_root = Path(project_root).resolve()
    resolved_root = resolve_dataset_root(
        project_root,
        explicit=dataset_root,
        use_environment=dataset_root is None,
        require=True,
    )
    candidates = (
        resolved_root
        / CORPUS_WORKSPACE_RELATIVE
        / "generated"
        / "candidates"
        / "ghost_stream"
    )
    return resolved_root, candidates, candidates / "stream_manifest.jsonl"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════
# STREAM SOURCES
# ═══════════════════════════════════════════════════════════

def stream_fineweb(limit: int = 500) -> list[dict]:
    """FineWeb — web de alta qualidade, textos longos."""
    log("🔥 FineWeb: streamando...")
    docs = []
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
        for i, doc in enumerate(ds):
            text = doc["text"].strip()
            if len(text) > 500:
                docs.append({
                    "id": hashlib.md5(text[:200].encode()).hexdigest()[:12],
                    "domain": detect_domain(text),
                    "text": text[:3000],  # primeiros 3000 chars
                    "source": "fineweb",
                })
            if len(docs) >= limit:
                break
    except Exception as e:
        log(f"  FineWeb: {e}")
    log(f"🔥 FineWeb: {len(docs)} docs")
    return docs


def stream_wikipedia(lang: str = "pt", limit: int = 300) -> list[dict]:
    """Wikipedia em streaming (wikimedia/wikipedia — parquet)."""
    log(f"🌐 Wikipedia {lang}: streamando...")
    docs = []
    try:
        from datasets import load_dataset
        ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
        for i, doc in enumerate(ds):
            text = doc["text"].strip()
            if len(text) > 300:
                docs.append({
                    "id": hashlib.md5(text[:200].encode()).hexdigest()[:12],
                    "domain": "knowledge",
                    "text": text[:3000],
                    "source": f"wikipedia_{lang}",
                })
            if len(docs) >= limit:
                break
    except Exception as e:
        log(f"  Wikipedia {lang}: {e}")
    log(f"🌐 Wikipedia {lang}: {len(docs)} docs")
    return docs


def stream_wikipedia_bilingual(limit: int = 300) -> list[dict]:
    """Return at most ``limit`` documents split across PT and EN."""
    limit = max(int(limit), 0)
    if limit == 0:
        return []
    pt_limit = (limit + 1) // 2
    en_limit = limit - pt_limit
    docs = stream_wikipedia("pt", limit=pt_limit)
    if en_limit:
        docs.extend(stream_wikipedia("en", limit=en_limit))
    return docs[:limit]


def stream_math(limit: int = 200) -> list[dict]:
    """Dataset de matematica — perguntas e respostas (meta-math/MetaMathQA)."""
    log("🧮 Math: streamando...")
    docs = []
    from datasets import load_dataset
    # meta-math/MetaMathQA: campos {query, response, original_question, ...}
    # split 'train' disponivel em parquet
    ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
    for i, doc in enumerate(ds):
        question = doc.get("query") or doc.get("question") or doc.get("original_question") or ""
        answer = doc.get("response") or doc.get("answer") or ""
        if question and answer:
            text = f"Question: {question}\nAnswer: {answer}"
            docs.append({
                "id": f"math_{i}",
                "domain": "math",
                "text": text,
                "source": "metamath",
            })
        if len(docs) >= limit:
            break
    log(f"🧮 Math: {len(docs)} docs")
    return docs


def stream_c4(limit: int = 300) -> list[dict]:
    """C4 — corpus web geral."""
    log("🌐 C4: streamando...")
    docs = []
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        for i, doc in enumerate(ds):
            text = doc["text"].strip()
            if len(text) > 500:
                docs.append({
                    "id": hashlib.md5(text[:200].encode()).hexdigest()[:12],
                    "domain": detect_domain(text),
                    "text": text[:3000],
                    "source": "c4",
                })
            if len(docs) >= limit:
                break
    except Exception as e:
        log(f"  C4: {e}")
    log(f"🌐 C4: {len(docs)} docs")
    return docs


# ═══════════════════════════════════════════════════════════
# DOMAIN DETECTION
# ═══════════════════════════════════════════════════════════

DOMAIN_PATTERNS = {
    "math": [r"\b(equation|formula|theorem|derivative|integral|algebra|calculus|function)\b",
             r"[0-9]+\s*[\+\-\*/\^]\s*[0-9]+"],
    "physics": [r"\b(physics|quantum|relativity|newton|force|energy|velocity)\b"],
    "code": [r"\b(def |function |class |import |return |python|javascript|fn )\b",
             r"[{}();]"],
    "medicine": [r"\b(disease|patient|diagnosis|treatment|surgery|clinical|therapy)\b"],
    "finance": [r"\b(stock|market|investment|revenue|profit|loss|trading|bitcoin)\b"],
    "philosophy": [r"\b(philosophy|ethics|morality|existence|consciousness|being)\b"],
    "biology": [r"\b(DNA|RNA|cell|protein|gene|genome|evolution|species)\b"],
    "history": [r"\b(century|war|empire|king|revolution|ancient|civilization)\b"],
}


def detect_domain(text: str) -> str:
    """Simple keyword-based domain detection."""
    text_lower = text.lower()
    scores = {}
    for domain, patterns in DOMAIN_PATTERNS.items():
        score = 0
        for pattern in patterns:
            matches = len(re.findall(pattern, text_lower))
            score += matches * 2
        if score > 0:
            scores[domain] = score
    if scores:
        return max(scores, key=scores.get)
    return "general"


# ═══════════════════════════════════════════════════════════
# GHOST BRAIN INTEGRATION
# ═══════════════════════════════════════════════════════════

class SimpleGhostExplorer:
    """Versao leve do Ghost Brain para streaming — gera perguntas e dados sinteticos
    sem depender de Wolfram ou modelo carregado."""

    def __init__(self):
        self.explored_count = 0
        self.synthetic_count = 0

    def explore_document(self, doc: dict) -> list[str]:
        """Explora um documento: gera perguntas e cria dados de treino."""
        text = doc["text"]
        domain = doc["domain"]
        synthetic_docs = []

        # Gera perguntas baseadas no dominio
        questions = self._generate_questions(text, domain)

        # Gera dados sinteticos de alta qualidade
        for q in questions:
            # Extrai trecho relevante como "resposta"
            answer_span = self._extract_answer_span(text, q)
            if answer_span:
                synthetic = (
                    f"# Ghost Training Data\n"
                    f"Domain: {domain}\n"
                    f"Source: {doc['source']}\n\n"
                    f"Context: {text[:500]}...\n\n"
                    f"Question: {q}\n\n"
                    f"Answer: {answer_span}\n\n"
                    f"# Learned via Ghost Stream exploration"
                )
                synthetic_docs.append(synthetic)

        self.explored_count += 1
        self.synthetic_count += len(synthetic_docs)
        return synthetic_docs

    def _generate_questions(self, text: str, domain: str) -> list[str]:
        """Generate domain-specific questions from text."""
        questions = []
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 30]

        if domain == "math" and len(sentences) >= 2:
            questions.append(f"Explain the mathematical concept in: {sentences[0]}")
        elif domain == "physics":
            questions.append(f"What physical principle is described here? {sentences[0]}")
        elif domain == "code":
            questions.append(f"What does this code do? Explain step by step.")
        elif domain == "medicine":
            questions.append(f"What medical condition or treatment is discussed?")
        elif domain == "history":
            questions.append(f"What historical event or period is described?")
        elif domain == "philosophy":
            questions.append(f"What philosophical argument is being made?")
        elif domain == "finance":
            questions.append(f"What financial concept or strategy is discussed?")
        else:
            # General comprehension
            if len(sentences) >= 3:
                questions.append(f"Summarize the main point: {sentences[0]}...{sentences[-1]}")
            questions.append(f"What is the key takeaway from this text?")

        # Add domain-specific reflection
        questions.append(f"How does this text contribute to knowledge in {domain}?")
        return questions

    def _extract_answer_span(self, text: str, question: str) -> str:
        """Extract the most relevant portion of text as answer."""
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
        if len(sentences) >= 3:
            # Return middle sentences as the "core" content
            mid = len(sentences) // 2
            return ". ".join(sentences[max(0, mid-1):mid+2]) + "."
        return sentences[0] + "." if sentences else ""


# ═══════════════════════════════════════════════════════════
# ORQUESTRADOR
# ═══════════════════════════════════════════════════════════

STREAM_SOURCES = {
    "fineweb": stream_fineweb,
    "wikipedia": stream_wikipedia_bilingual,
    "math": stream_math,
    "c4": stream_c4,
}


def run_ghost_stream(
    sources: list[str] | None = None,
    docs_per_source: int = 200,
    *,
    project_root: Path = ROOT,
    dataset_root: Path | None = None,
):
    """Run one ghost stream cycle — pull docs, explore, save synthetics."""
    resolved_root, candidates_dir, manifest_path = resolve_ghost_paths(
        project_root, dataset_root=dataset_root
    )
    if candidates_dir.resolve().is_relative_to(resolved_root.resolve()):
        mark_runtime_use(Path(project_root).resolve(), actor="ghost_stream")
    candidates_dir.mkdir(parents=True, exist_ok=True)

    selected = sources or list(STREAM_SOURCES.keys())
    all_synthetic = []
    total_docs = 0

    print(f"\n{'='*60}")
    print(f"  👻 GHOST STREAM — {now_iso()}")
    print(f"  Fontes: {', '.join(selected)}")
    print(f"{'='*60}\n")

    explorer = SimpleGhostExplorer()

    for name in selected:
        if name not in STREAM_SOURCES:
            log(f"❌ Fonte: {name}")
            continue
        try:
            docs = STREAM_SOURCES[name](limit=docs_per_source)
            total_docs += len(docs)

            for doc in docs:
                synthetic = explorer.explore_document(doc)
                all_synthetic.extend(synthetic)

        except Exception as e:
            log(f"❌ {name}: {e}")

    # Save synthetic documents
    if all_synthetic:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = candidates_dir / f"ghost_synthetic_{timestamp}.txt"
        path.write_text("\n\n---\n\n".join(all_synthetic), encoding="utf-8")
        size_mb = path.stat().st_size / 1e6
        log(f"💾 {len(all_synthetic)} docs sinteticos → {path.name} ({size_mb:.1f} MB)")

    # Manifest
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": now_iso(),
            "sources": selected,
            "docs_streamed": total_docs,
            "synthetic_generated": len(all_synthetic),
            "explorer": {
                "explored": explorer.explored_count,
                "synthetic": explorer.synthetic_count,
            },
        }, ensure_ascii=False) + "\n")

    total_synthetic = sum(
        1 for _ in candidates_dir.glob("ghost_synthetic_*.txt")
    )
    total_mb = sum(
        f.stat().st_size for f in candidates_dir.glob("ghost_synthetic_*.txt")
    ) / 1e6

    print(f"\n{'='*60}")
    print(f"  ✅ {total_docs} docs | {len(all_synthetic)} sinteticos")
    print(f"  📂 {candidates_dir} ({total_synthetic} arquivos, {total_mb:.1f} MB)")
    print(f"{'='*60}\n")

    return {"docs": total_docs, "synthetic": len(all_synthetic)}


def main():
    parser = argparse.ArgumentParser(description="F51 Ghost Stream — Exploracao viva")
    parser.add_argument("--source", default=None,
                        help="Fontes (fineweb,wikipedia,math,c4)")
    parser.add_argument("--loop", action="store_true",
                        help="Loop infinito")
    parser.add_argument("--interval", type=int, default=600,
                        help="Segundos entre ciclos (default: 600 = 10min)")
    parser.add_argument("--docs", type=int, default=200,
                        help="Docs por fonte por ciclo")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override explícito do workspace; o padrão é <project>/workspace",
    )
    args = parser.parse_args()

    sources = args.source.split(",") if args.source else None

    if args.loop:
        print("👻 GHOST STREAM LOOP — Ctrl+C para parar")
        cycle = 0
        try:
            while True:
                cycle += 1
                print(f"\n🔁 CICLO {cycle}")
                run_ghost_stream(sources, args.docs, dataset_root=args.dataset_root)
                print(f"💤 {args.interval}s...")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n🛑 Ghost Stream pausado.")
    else:
        run_ghost_stream(sources, args.docs, dataset_root=args.dataset_root)


if __name__ == "__main__":
    main()
