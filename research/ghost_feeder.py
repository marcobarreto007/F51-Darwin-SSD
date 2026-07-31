#!/usr/bin/env python3
"""
F51 Ghost Feeder — Conecta Ghost Stream ao treino do organismo
===============================================================
Pega documentos gerados pelo Ghost Stream, tokeniza, e injeta
no loop de treino para que o modelo realmente APRENDA com o
que o fantasma explora.

Fluxo completo:
    Ghost Stream → candidates/ghost_stream/*.txt
        ↓
    DataFirewall audita
        ↓
    Promove aprovados → F51-Dataset-Organizado/02_CORPUS/approved/
        ↓
    Tokeniza corpus completo → F51-Dataset-Organizado/01_TOKENIZADOS/CLASSICOS/tokens_live.bin
        ↓
    Organismo recarrega e treina com dados NOVOS
        ↓
    Curiosity Drive prioriza domínios recém-explorados

Uso como módulo:
    from research.ghost_feeder import feed_organism
    feed_organism(organism_instance)
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from scripts.ingest_pipeline import write_int32_atomic
from f51_darwin.dataset_layout import (
    LIVE_TOKEN_RELATIVE,
    resolve_dataset_path,
    resolve_feast_token_bin,
)


def feed_organism(org) -> dict:
    """Roda o ciclo completo de ingestao: candidatos → firewall → tokenize → reload.
    
    Chamar isso periodicamente do run247 loop.
    """
    from f51_darwin.data_factory import DataFactory, DataFactoryPaths
    from f51_darwin.dataset_states import DatasetStatus, SourceType
    from f51_darwin.tokenizer import F51BPETokenizer
    from f51_darwin.data import CausalLMDataLoader
    
    root = org.root
    paths = DataFactoryPaths.from_project(root)
    live_path = resolve_dataset_path(root, LIVE_TOKEN_RELATIVE)
    feast_path = resolve_feast_token_bin(root)
    result = {
        "candidates_found": 0,
        "approved": 0,
        "promoted": 0,
        "tokens_new": 0,
        "reloaded": False,
    }

    # ── STEP 1: Scan all candidate sources ──
    candidate_dirs = [
        paths.candidates,
        paths.candidates / "ghost_stream",
        paths.candidates / "olavo",
        paths.root / "real_ingestion",
    ]
    
    new_files = []
    for d in candidate_dirs:
        if d.exists():
            for f in d.glob("*.txt"):
                if f.stat().st_mtime > time.time() - 86400:  # ultimas 24h
                    new_files.append(f)
    
    result["candidates_found"] = len(new_files)
    if new_files:
        print(f"  🌱 Ghost Feeder: {len(new_files)} novos documentos encontrados")

    # ── STEP 2: Firewall audit + promote ──
    factory = DataFactory(paths, firewall=org.firewall)
    
    approved_count = 0
    for f in new_files[:50]:  # limite por ciclo
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            if len(text.strip()) < 100:
                continue
            record = factory.register_candidate(
                text=text,
                # Ghost/web material is external evidence. The firewall must
                # quarantine it until an operator explicitly approves it.
                source_type=SourceType.IMPORTED,
                source_path=str(f),
                dataset_version=f"ghost_feed_{datetime.now().strftime('%Y%m%d_%H%M')}",
            )
            decision = factory.audit_candidate(record, text)
            if decision.status == DatasetStatus.APPROVED:
                approved_count += 1
        except Exception:
            pass
    
    result["approved"] = approved_count

    # Explicitly approved quarantine records live under data/approved. Only
    # those records may be exported into the training corpus. Merely being a
    # raw Ghost candidate is never sufficient.
    approved_records = sorted(paths.approved.glob("*.json"))
    missing_corpus_items = [
        paths.corpus / f"{record_path.stem}.txt"
        for record_path in approved_records
        if not (paths.corpus / f"{record_path.stem}.txt").exists()
    ]
    if missing_corpus_items:
        factory.promote_approved_to_corpus(build_corpus=True)
        result["promoted"] = len(missing_corpus_items)
        print(f"  ✅ Promocao explicita: {len(missing_corpus_items)} → {paths.corpus}")

    # ── STEP 3: Tokenize corpus inteiro → tokens_live.bin ──
    corpus_files = sorted(paths.corpus.rglob("*.txt"))
    live_mtime = live_path.stat().st_mtime_ns if live_path.exists() else -1
    corpus_changed = bool(corpus_files) and any(
        path.stat().st_mtime_ns > live_mtime for path in corpus_files
    )
    if corpus_changed:
        try:
            from f51_darwin.data import load_text_documents, tokenize_documents
            
            corpus_dir = paths.corpus
            documents = load_text_documents(corpus_dir)
            
            if org.tokenizer is None:
                org.tokenizer = F51BPETokenizer.load(root / org.cfg.tokenizer_dir)
            
            new_tokens = tokenize_documents(documents, org.tokenizer)

            # Merge com tokens_feast se existir
            if feast_path.exists():
                feast_tokens = np.memmap(feast_path, dtype=np.int32, mode='r')
                # Amostra do feast
                feast_sample_size = min(len(feast_tokens), 20_000_000)
                feast_sample = feast_tokens[-feast_sample_size:]

                # ⚠️ DOUTRINA: max 10% de dados sintéticos/curiosidade
                max_synthetic = int(len(feast_sample) * org.cfg.max_synthetic_ratio)
                if len(new_tokens) > max_synthetic:
                    cutoff = f"{len(new_tokens):,} → {max_synthetic:,} (cap {org.cfg.max_synthetic_ratio:.0%})"
                    new_tokens = new_tokens[:max_synthetic]
                    print(f"  ⚠️  Curiosity cap: {cutoff}")

                merged = np.concatenate([feast_sample, new_tokens])
            else:
                merged = new_tokens
            
            result["tokens_new"] = len(new_tokens)
            result["tokens_total"] = len(merged)
            result["synthetic_ratio"] = len(new_tokens) / len(merged) if len(merged) > 0 else 0

            # ── Brainstem check ──
            h = org.brainstem.check_synthetic_ratio(len(new_tokens), len(merged))
            if not h.alive:
                print(f"  🛑 Brainstem: synthetic ratio {h.metrics.get('synthetic_ratio', 0):.1%} > {org.cfg.max_synthetic_ratio:.0%} — BLoqueado!")
                return result

            # Windows cannot replace an open memory-mapped target. At this
            # safe between-cycle boundary, close only the old live mapping;
            # a failed replace reopens the untouched previous target.
            previous_live_mapping_closed = False
            previous_path = getattr(org, "token_path", None)
            try:
                previous_is_live = (
                    previous_path is not None
                    and Path(previous_path).resolve() == live_path.resolve()
                )
            except (OSError, ValueError):
                previous_is_live = False
            if previous_is_live and isinstance(getattr(org, "token_ids", None), np.memmap):
                mmap_handle = getattr(org.token_ids, "_mmap", None)
                if mmap_handle is not None:
                    mmap_handle.close()
                    previous_live_mapping_closed = True

            try:
                write_int32_atomic(live_path, merged)
            except Exception:
                if previous_live_mapping_closed and live_path.exists():
                    org.token_ids = np.memmap(live_path, dtype=np.int32, mode="r")
                    org.token_count = len(org.token_ids)
                    org.token_path = str(live_path)
                raise

            # ── STEP 4: Recarregar no organismo ──
            org.token_ids = np.memmap(live_path, dtype=np.int32, mode='r')
            org.token_count = len(org.token_ids)
            org.token_path = str(live_path)
            
            result["reloaded"] = True
            print(f"  📊 Tokens: +{len(new_tokens):,} novos → {len(merged):,} total")
            print(f"  🔄 Organismo recarregado com dados frescos!")
            
        except Exception as e:
            print(f"  ⚠️ Tokenize/reload: {e}")

    return result


# ═══════════════════════════════════════════════════════════
# STANDALONE
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Ghost Feeder — use como modulo: from research.ghost_feeder import feed_organism")
    print("Ou rode o Ghost Stream primeiro: python research/ghost_stream.py")
