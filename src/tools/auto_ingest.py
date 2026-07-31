#!/usr/bin/env python3
"""
F51 Darwin-SSD — Auto-Ingestor de Dados de Treino
==================================================
Baixa automaticamente das nossas fontes prediletas,
salva em data/real_ingestion/, e roda 24/7 como worker.

Fontes:
    📚 Project Gutenberg — livros clássicos (ciência, filosofia, literatura)
    🧬 PubMed — artigos científicos e médicos
    🌐 C4 — corpus web em inglês (AllenAI)
    🔥 FineWeb — web de alta qualidade (HuggingFace)
    🇧🇷 Wikipedia PT — Wikipédia em português
    🇧🇷 Olavo de Carvalho — obras completas (archive.org)

Uso:
    python -m tools.auto_ingest              # uma rodada completa
    python -m tools.auto_ingest --loop        # loop infinito (cada 6h)
    python -m tools.auto_ingest --source gutenberg,pubmed  # só essas
    python -m tools.auto_ingest --max-mb 500  # para após X MB baixados
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

INGEST_DIR = ROOT / "data" / "real_ingestion"
MANIFEST_FILE = INGEST_DIR / "ingestion_manifest.jsonl"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def save_chunks(texts: list[str], prefix: str, max_chunks: int = 20) -> int:
    """Save texts in chunks of ~10k documents each."""
    INGEST_DIR.mkdir(parents=True, exist_ok=True)
    chunk, n = [], 0
    total_bytes = 0
    for t in texts:
        t = t.strip()
        if len(t) < 200:
            continue
        chunk.append(t)
        if len(chunk) >= 10000:
            path = INGEST_DIR / f"{prefix}_{n:04d}.txt"
            content = "\n\n".join(chunk)
            path.write_text(content, encoding="utf-8")
            total_bytes += len(content)
            log(f"{prefix} chunk {n}: {len(content)/1e6:.1f} MB")
            n += 1
            chunk = []
            if n >= max_chunks:
                break
    if chunk and n < max_chunks:
        path = INGEST_DIR / f"{prefix}_{n:04d}.txt"
        content = "\n\n".join(chunk)
        path.write_text(content, encoding="utf-8")
        total_bytes += len(content)
        log(f"{prefix} chunk {n}: {len(content)/1e6:.1f} MB")
        n += 1
    # Manifest
    with MANIFEST_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "source": prefix,
            "chunks": n,
            "bytes": total_bytes,
            "timestamp": now_iso(),
        }, ensure_ascii=False) + "\n")
    return n


# ═══════════════════════════════════════════════════════════
# FONTES
# ═══════════════════════════════════════════════════════════

def ingest_gutenberg(max_books: int = 200) -> int:
    """Project Gutenberg — livros classicos de ciencia, filosofia, literatura."""
    log("📚 Gutenberg: baixando...")
    queries = [
        "physics", "chemistry", "biology", "mathematics", "medicine",
        "philosophy", "economics", "history", "astronomy", "geology",
        "botany", "zoology", "psychology", "sociology", "engineering",
        "literature", "poetry", "biography", "essays", "letters",
        "theology", "politics", "law", "education", "art",
        "music", "architecture", "war", "geography", "mythology",
    ]
    texts = []
    downloaded = 0
    for q in queries:
        try:
            url = f"https://gutendex.com/books?search={q}&languages=en,fr,pt,es&page=1"
            data = json.loads(urllib.request.urlopen(url, timeout=30).read())
            for book in data.get("results", []):
                bid = book["id"]
                title = re.sub(r"[^a-zA-Z0-9]", "_", book["title"])[:60]
                txt_url = f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt"
                try:
                    text = urllib.request.urlopen(txt_url, timeout=45).read().decode("utf-8", errors="replace")
                    if len(text) > 5000:
                        texts.append(f"TITLE: {book['title']}\n\n{text}")
                        downloaded += 1
                    time.sleep(0.3)
                except Exception:
                    pass
                if downloaded >= max_books:
                    break
            time.sleep(0.5)
        except Exception as e:
            log(f"  Gutenberg '{q}': {e}")
        if downloaded >= max_books:
            break
    chunks = save_chunks(texts, "gutenberg")
    log(f"📚 Gutenberg: {downloaded} livros -> {chunks} chunks ✅")
    return chunks


def ingest_pubmed() -> int:
    """PubMed — artigos cientificos e medicos."""
    log("🧬 PubMed: baixando abstracts...")
    queries = [
        "biology", "genetics", "neuroscience", "immunology",
        "physics", "mathematics", "chemistry", "medicine",
        "pharmacology", "cardiology", "oncology", "molecular",
        "epidemiology", "physiology", "biochemistry", "computational+biology",
        "artificial+intelligence+medicine", "machine+learning+health",
    ]
    all_text = []
    for field in queries:
        try:
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={field}&retmax=150&retmode=json"
            data = json.loads(urllib.request.urlopen(url, timeout=30).read())
            ids = data.get("esearchresult", {}).get("idlist", [])
            for i in range(0, len(ids), 100):
                batch = ids[i:i + 100]
                url2 = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={','.join(batch)}&rettype=abstract&retmode=text"
                text = urllib.request.urlopen(url2, timeout=60).read().decode("utf-8", errors="replace")
                all_text.append(text)
                time.sleep(0.35)
            time.sleep(0.3)
        except Exception as e:
            log(f"  PubMed '{field}': {e}")
            time.sleep(2)
    chunks = save_chunks(all_text, "pubmed", max_chunks=10)
    log(f"🧬 PubMed: {chunks} chunks ✅")
    return chunks


def ingest_c4(max_chunks: int = 10) -> int:
    """C4 — corpus web em ingles (AllenAI)."""
    log("🌐 C4: baixando...")
    try:
        from datasets import load_dataset
    except ImportError:
        log("🌐 C4: datasets nao instalado -> pulando")
        return 0
    texts = []
    try:
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        for doc in ds:
            t = doc["text"].strip()
            if len(t) > 200:
                texts.append(t)
            if len(texts) >= max_chunks * 10000:
                break
    except Exception as e:
        log(f"  C4: {e}")
    chunks = save_chunks(texts, "c4", max_chunks=max_chunks)
    log(f"🌐 C4: {chunks} chunks ✅")
    return chunks


def ingest_fineweb(max_chunks: int = 8) -> int:
    """FineWeb — web de alta qualidade (HuggingFace)."""
    log("🔥 FineWeb: baixando...")
    try:
        from datasets import load_dataset
    except ImportError:
        log("🔥 FineWeb: datasets nao instalado -> pulando")
        return 0
    texts = []
    try:
        ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
        for doc in ds:
            t = doc["text"].strip()
            if len(t) > 200:
                texts.append(t)
            if len(texts) >= max_chunks * 10000:
                break
    except Exception as e:
        log(f"  FineWeb: {e}")
    chunks = save_chunks(texts, "fineweb", max_chunks=max_chunks)
    log(f"🔥 FineWeb: {chunks} chunks ✅")
    return chunks


def ingest_wikipedia_pt() -> int:
    """Wikipedia em portugues."""
    log("🇧🇷 Wikipedia PT: baixando...")
    try:
        from datasets import load_dataset
    except ImportError:
        log("🇧🇷 Wikipedia PT: datasets nao instalado -> pulando")
        return 0
    texts = []
    try:
        ds = load_dataset("wikipedia", "20231101.pt", split="train", streaming=True)
        for doc in ds:
            t = doc["text"].strip()
            if len(t) > 200:
                texts.append(t)
            if len(texts) >= 80000:
                break
    except Exception:
        try:
            ds = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train", streaming=True)
            for doc in ds:
                t = doc["text"].strip()
                if len(t) > 200:
                    texts.append(t)
                if len(texts) >= 80000:
                    break
        except Exception as e:
            log(f"  Wikipedia PT: {e}")
            return 0
    chunks = save_chunks(texts, "wiki_pt", max_chunks=8)
    log(f"🇧🇷 Wikipedia PT: {chunks} chunks ✅")
    return chunks


def ingest_olavo() -> int:
    """Olavo de Carvalho — obras completas do archive.org."""
    log("🇧🇷 Olavo: baixando obras...")
    try:
        # Usa o script existente
        import subprocess
        result = subprocess.run(
            [sys.executable, str(ROOT / "research" / "download_olavo_sources.py")],
            capture_output=True, text=True, timeout=600,
        )
        olavo_dir = ROOT / "data" / "generated" / "candidates" / "olavo"
        count = len(list(olavo_dir.glob("*.txt"))) if olavo_dir.exists() else 0
        log(f"🇧🇷 Olavo: {count} arquivos ✅")
        return count
    except Exception as e:
        log(f"  Olavo: {e}")
        return 0


# ═══════════════════════════════════════════════════════════
# ORQUESTRADOR
# ═══════════════════════════════════════════════════════════

SOURCES = {
    "gutenberg": ingest_gutenberg,
    "pubmed": ingest_pubmed,
    "c4": ingest_c4,
    "fineweb": ingest_fineweb,
    "wiki_pt": ingest_wikipedia_pt,
    "olavo": ingest_olavo,
}


def run_ingestion(sources: list[str] | None = None, max_mb: int = 0) -> dict:
    """Run one full ingestion cycle."""
    INGEST_DIR.mkdir(parents=True, exist_ok=True)

    selected = sources or list(SOURCES.keys())
    results = {}
    total_chunks = 0

    print(f"\n{'='*60}")
    print(f"  🧬 F51 AUTO-INGEST — {now_iso()}")
    print(f"  Fontes: {', '.join(selected)}")
    print(f"{'='*60}\n")

    for name in selected:
        if name not in SOURCES:
            log(f"❌ Fonte desconhecida: {name}")
            continue
        try:
            chunks = SOURCES[name]()
            results[name] = {"chunks": chunks, "status": "ok"}
            total_chunks += chunks
        except Exception as e:
            log(f"❌ {name}: {e}")
            results[name] = {"chunks": 0, "status": str(e)}

        # Check size limit
        if max_mb > 0:
            total_bytes = sum(
                f.stat().st_size for f in INGEST_DIR.glob("*")
                if f.is_file() and f.name != "ingestion_manifest.jsonl"
            )
            if total_bytes / 1e6 >= max_mb:
                log(f"⏸️  Limite de {max_mb} MB atingido")
                break

    total_mb = sum(
        f.stat().st_size for f in INGEST_DIR.glob("*")
        if f.is_file() and f.name != "ingestion_manifest.jsonl"
    ) / 1e6

    print(f"\n{'='*60}")
    print(f"  ✅ {total_chunks} chunks | {total_mb:.1f} MB | {len(results)} fontes")
    print(f"  📂 {INGEST_DIR}")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="F51 Auto-Ingestor de Dados")
    parser.add_argument("--source", default=None,
                        help="Fontes separadas por virgula (gutenberg,pubmed,c4,fineweb,wiki_pt,olavo)")
    parser.add_argument("--loop", action="store_true",
                        help="Loop infinito — re-ingere a cada 6 horas")
    parser.add_argument("--interval-hours", type=float, default=6.0,
                        help="Intervalo entre ciclos no modo --loop")
    parser.add_argument("--max-mb", type=int, default=0,
                        help="Parar apos X MB baixados (0 = sem limite)")
    args = parser.parse_args()

    sources = args.source.split(",") if args.source else None

    if args.loop:
        print("🔄 MODO LOOP — Ctrl+C para parar")
        cycle = 0
        try:
            while True:
                cycle += 1
                print(f"\n🔁 CICLO {cycle} — {now_iso()}")
                run_ingestion(sources, args.max_mb)
                print(f"💤 Dormindo {args.interval_hours}h...")
                time.sleep(args.interval_hours * 3600)
        except KeyboardInterrupt:
            print("\n🛑 Loop interrompido.")
    else:
        run_ingestion(sources, args.max_mb)


if __name__ == "__main__":
    main()
