#!/usr/bin/env python
"""
F51 Corpus Consolidator — Junta 20k+ arquivos soltos em consolidados.

Cada grupo de arquivos é combinado em um único .txt com separador ---.
O data.py já sabe splitar por ---, então não perde boundaries de documento.

Uso:
    python -m tools.consolidate_corpus
    python -m tools.consolidate_corpus --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]

CORPUS_DIR = ROOT / "data" / "corpus"

SEPARATOR = "\n\n---\n\n"

# Grupos de consolidação: (prefixo_output, padrão glob)
CONSOLIDATION_GROUPS = [
    ("med_flashcards_consolidated", "med_flashcards_*.txt"),
    ("arch_finance_consolidated", "arch_finance_batch_*.txt"),
    ("biblia_consolidated", "biblia_*.txt"),
    ("books_classics_consolidated", "book_pt_pt-*.txt"),
    ("f51_identity_consolidated", "f51_identity*.txt"),
    ("f51_doc_consolidated", "doc_*.txt"),
    ("arch_conversation_consolidated", "arch_conversation*.txt"),
    ("arch_SuperEzio_consolidated", "arch_SuperEzio_*.txt"),
    ("arch_CV_Marco_consolidated", "arch_CV_Marco*.txt"),
    ("arch_F51_BigBrain_consolidated", "arch_F51_BigBrain*.txt"),
    ("f51_py_consolidated", "f51_*.py.txt"),
    # Synthetic files (já são consolidados, mas incluímos pra unificar)
    ("f51_synthetic_identity", "f51_identity_synthetic*.txt"),
    ("f51_synthetic_code", "f51_code_synthetic*.txt"),
    ("f51_synthetic_dialogues", "f51_dialogues_synthetic*.txt"),
    ("f51_synthetic_portuguese", "f51_portuguese_synthetic*.txt"),
    ("f51_synthetic_reasoning", "f51_reasoning_synthetic*.txt"),
]


def consolidate_corpus(corpus_dir: Path, dry_run: bool = False):
    """Agrupa arquivos soltos em consolidados e remove originais."""

    if not corpus_dir.exists():
        print(f"ERRO: Diretório não encontrado: {corpus_dir}")
        return

    total_before = 0
    total_after = 0
    total_size = 0

    for output_prefix, glob_pattern in CONSOLIDATION_GROUPS:
        matches = sorted(corpus_dir.glob(glob_pattern))

        if not matches:
            continue

        total_before += len(matches)
        combined = []
        total_size += sum(f.stat().st_size for f in matches)

        for f in matches:
            try:
                text = f.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    combined.append(text)
            except Exception as e:
                print(f"  Aviso: Não foi possível ler {f.name}: {e}")

        if not combined:
            continue

        output_file = corpus_dir / f"{output_prefix}.txt"

        if dry_run:
            print(f"[DRY RUN] {output_prefix}: {len(matches)} arquivos → {output_file.name} ({len(SEPARATOR.join(combined))/1e6:.1f} MB)")
        else:
            content = SEPARATOR.join(combined)
            output_file.write_text(content, encoding="utf-8")
            size_mb = output_file.stat().st_size / 1e6
            print(f"  Consolidado: {output_file.name} ({len(matches)} arquivos, {size_mb:.1f} MB)")

            # Remove originais
            for f in matches:
                if f != output_file:
                    f.unlink()
            total_after += 1

    if dry_run:
        print(f"\n[DRY RUN] Seriam consolidados: {total_before} arquivos → {total_after} consolidados")
    else:
        print(f"\n{'='*60}")
        print(f"  CONSOLIDAÇÃO CONCLUÍDA")
        print(f"  {total_before} arquivos → {total_after} consolidados")
        print(f"  Espaço total: {total_size/1e6:.1f} MB")

        # Conta arquivos restantes
        remaining = list(corpus_dir.glob("*.txt"))
        print(f"  Arquivos no corpus agora: {len(remaining)}")
        for r in sorted(remaining):
            size_mb = r.stat().st_size / 1e6
            print(f"    {r.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="F51 Corpus Consolidator")
    parser.add_argument("--corpus", default=str(CORPUS_DIR),
                       help="Diretório do corpus")
    parser.add_argument("--dry-run", action="store_true",
                       help="Apenas mostrar o que seria feito")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.is_absolute():
        corpus_path = ROOT / corpus_path

    print(f"F51 Corpus Consolidator")
    print(f"  Dir: {corpus_path}")
    print(f"  Modo: {'DRY RUN' if args.dry_run else 'EXECUTAR'}")
    print()

    consolidate_corpus(corpus_path, args.dry_run)
