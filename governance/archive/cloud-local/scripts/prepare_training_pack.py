#!/usr/bin/env python3
"""
F51 Darwin-SSD — PREPARADOR DE TRAINING PACK PARA NUVEM

Orquestra a criação completa do material de treino:
  1. Gera corpus clássico (Gutenberg PT/EN/FR, Bíblia, Constituição)
  2. Builda pack aprovado com manifesto (corpus policy)
  3. Comprime para nuvem (tar.gz)
  4. Gera metadados de upload

Uso:
  python scripts/prepare_training_pack.py                    # workflow completo
  python scripts/prepare_training_pack.py --skip-download     # só builda pack
  python scripts/prepare_training_pack.py --raw-only          # só download
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from f51_darwin.corpus_factory_pipeline import build_corpus_pack
from f51_darwin.corpus_pack_upload import file_sha256


@dataclass(frozen=True)
class TrainingPackMeta:
    pack_name: str
    version: str
    created_at: str
    total_files: int
    approved_files: int
    total_chars: int
    pack_path: str
    pack_sha256: str
    pack_size_mb: float
    sources: list[str] = field(default_factory=list)
    tokenizer_info: dict[str, Any] = field(default_factory=dict)
    ready_for_upload: bool = False


@dataclass(frozen=True)
class PackResult:
    success: bool
    meta: TrainingPackMeta | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class TrainingPackBuilder:
    """Orquestra a criação completa do training pack."""

    def __init__(
        self,
        project_root: Path,
        workspace: Path | None = None,
        batch_name: str | None = None,
    ):
        self.root = project_root.resolve()
        self.workspace = (workspace or Path.cwd()).resolve()
        self.batch_name = batch_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Diretórios
        self.raw_corpus_dir = self.workspace / "raw_corpus"
        self.pack_output_dir = self.workspace / "packs"
        self.final_staging_dir = self.workspace / "staging"

        self.pack_output_dir.mkdir(parents=True, exist_ok=True)
        self.final_staging_dir.mkdir(parents=True, exist_ok=True)

    def _print_header(self, title: str):
        width = 72
        print(f"\n{'='*width}")
        print(f"  {title}")
        print(f"{'='*width}\n")

    def _print_step(self, step: int, total: int, message: str):
        print(f"[{step}/{total}] {message}")

    def _print_ok(self, message: str):
        print(f"  ✓ {message}")

    def _print_error(self, message: str):
        print(f"  ✗ {message}")

    def _print_warn(self, message: str):
        print(f"  ⚠ {message}")

    def generate_raw_corpus(self) -> tuple[int, int]:
        """
        Gera corpus clássico usando o script build_classical_corpus.py.
        Retorna: (arquivos_gerados, total_chars)
        """
        self._print_header("ETAPA 1: GERANDO CORPUS CLÁSSICO")

        try:
            from scripts.build_classical_corpus import build_corpus

            manifest = build_corpus()

            total_books = manifest.get("total_books", 0)
            total_chars = manifest.get("total_chars", 0)

            self._print_ok(f"Corpus gerado: {total_books} livros, {total_chars:,} chars")
            return total_books, total_chars

        except Exception as e:
            self._print_error(f"Falha ao gerar corpus: {e}")
            traceback.print_exc()
            return 0, 0

    def build_approved_pack(
        self,
        input_dir: Path,
        default_license: str = "public-domain",
        source_url_base: str = "file://",
    ) -> tuple[bool, str]:
        """
        Builda pack aprovado com manifesto usando corpus_factory_pipeline.
        Retorna: (sucesso, pack_path)
        """
        self._print_header("ETAPA 2: BUILDANDO PACK APROVADO")

        try:
            summary = build_corpus_pack(
                input_dir=input_dir,
                output_dir=self.pack_output_dir,
                metadata_path=None,
                default_license=default_license,
                source_url_base=source_url_base,
                domain_hint="classical",
                export_approved=True,
                create_pack=True,
            )

            self._print_ok(f"Processados: {summary.scanned}")
            self._print_ok(f"Aprovados:   {summary.approved}")
            self._print_ok(f"Quarentena:  {summary.quarantine}")
            self._print_ok(f"Rejeitados:  {summary.rejected}")

            if summary.pack_path:
                pack_path = Path(summary.pack_path)
                if pack_path.exists():
                    size_mb = pack_path.stat().st_size / (1024 * 1024)
                    self._print_ok(f"Pack criado: {pack_path.name} ({size_mb:.2f} MB)")
                    return True, str(pack_path)
                else:
                    self._print_error("Pack path declarado mas não encontrado")
                    return False, ""
            else:
                self._print_error("Pack não foi criado")
                return False, ""

        except Exception as e:
            self._print_error(f"Falha ao buildar pack: {e}")
            traceback.print_exc()
            return False, ""

    def create_training_pack(
        self,
        corpus_pack_path: str,
        include_tokenizer: bool = True,
    ) -> tuple[bool, TrainingPackMeta | None]:
        """
        Cria o training pack final com:
        - Corpus pack (approved_corpus_pack.tar.gz)
        - Tokenizer (opcional)
        - Metadados completos

        Retorna: (sucesso, metadata)
        """
        self._print_header("ETAPA 3: CRIANDO TRAINING PACK FINAL")

        corpus_pack = Path(corpus_pack_path)
        if not corpus_pack.exists():
            self._print_error(f"Corpus pack não encontrado: {corpus_pack}")
            return False, None

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        final_pack_name = f"f51_training_pack_{timestamp}.tar.gz"
        final_pack_path = self.final_staging_dir / final_pack_name

        sources = ["corpus_pack"]
        tokenizer_info = {}

        try:
            with tarfile.open(final_pack_path, "w:gz") as tar:
                # Adiciona corpus pack
                tar.add(corpus_pack, arcname=corpus_pack.name)
                self._print_ok(f"Adicionado: {corpus_pack.name}")

                # Adiciona tokenizer (se existir e solicitado)
                if include_tokenizer:
                    tokenizer_dir = self.root / "tokenizer" / "f51_bpe_80k"
                    if tokenizer_dir.exists():
                        # Adiciona arquivos do tokenizer individualmente
                        for file in tokenizer_dir.glob("*"):
                            if file.is_file():
                                arcname = f"tokenizer/f51_bpe_80k/{file.name}"
                                tar.add(file, arcname=arcname)

                        # Tenta ler vocab size
                        vocab_file = tokenizer_dir / "vocab.json"
                        if vocab_file.exists():
                            try:
                                vocab_data = json.loads(vocab_file.read_text(encoding="utf-8"))
                                tokenizer_info = {
                                    "type": "BPE",
                                    "vocab_size": len(vocab_data),
                                    "files": ["vocab.json", "merges.txt", "config.json", "tokenizer.json"],
                                }
                            except:
                                tokenizer_info = {"type": "BPE", "files": ["vocab.json", "merges.txt"]}

                        sources.append("tokenizer")
                        self._print_ok(f"Adicionado: tokenizer/")

                # Cria e adiciona manifesto
                manifest = {
                    "pack_name": final_pack_name,
                    "version": "1.0",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "batch": self.batch_name,
                    "sources": sources,
                    "tokenizer": tokenizer_info,
                    "corpus_pack": corpus_pack.name,
                    "corpus_pack_sha256": file_sha256(corpus_pack),
                }

                manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False)
                manifest_path = self.final_staging_dir / "manifest.json"
                manifest_path.write_text(manifest_json + "\n", encoding="utf-8")
                tar.add(manifest_path, arcname="manifest.json")
                self._print_ok("Adicionado: manifest.json")

            # Calcula metadados finais
            pack_sha256 = file_sha256(final_pack_path)
            pack_size_mb = final_pack_path.stat().st_size / (1024 * 1024)

            meta = TrainingPackMeta(
                pack_name=final_pack_name,
                version="1.0",
                created_at=datetime.now(timezone.utc).isoformat(),
                total_files=2 + (1 if include_tokenizer else 0),
                approved_files=0,  # será preenchido se ler o manifest do corpus
                total_chars=0,
                pack_path=str(final_pack_path),
                pack_sha256=pack_sha256,
                pack_size_mb=pack_size_mb,
                sources=sources,
                tokenizer_info=tokenizer_info,
                ready_for_upload=True,
            )

            self._print_ok(f"Training pack criado: {final_pack_name}")
            self._print_ok(f"SHA256: {pack_sha256}")
            self._print_ok(f"Tamanho: {pack_size_mb:.2f} MB")

            return True, meta

        except Exception as e:
            self._print_error(f"Falha ao criar training pack: {e}")
            traceback.print_exc()
            return False, None

    def build_full_pipeline(
        self,
        skip_download: bool = False,
        skip_pack: bool = False,
        include_tokenizer: bool = True,
    ) -> PackResult:
        """
        Executa o pipeline completo.

        Etapas:
        1. Gera corpus clássico (opcional)
        2. Builda pack aprovado
        3. Cria training pack final

        Retorna: PackResult com metadata
        """
        self._print_header(f"F51 TRAINING PACK BUILDER — BATCH {self.batch_name}")

        errors = []
        warnings = []
        meta = None

        # Etapa 1: Gerar corpus (opcional)
        if not skip_download:
            total_books, total_chars = self.generate_raw_corpus()
            if total_books == 0:
                errors.append("Nenhum arquivo de corpus gerado")
                return PackResult(success=False, meta=None, errors=errors)

            # Usa o diretório data/corpus/raw_classical como input
            raw_input = self.root / "data" / "corpus" / "raw_classical"
            if not raw_input.exists():
                raw_input = self.root / "f51_darwin" / "dataset" / "raw_classical"

            if not raw_input.exists():
                # Tenta encontrar corpus em qualquer lugar
                corpus_candidates = list(self.root.rglob("gutenberg_*.txt"))
                if corpus_candidates:
                    raw_input = corpus_candidates[0].parent
                    warnings.append(f"Corpus encontrado em: {raw_input}")
                else:
                    errors.append("Diretório de corpus não encontrado")
                    return PackResult(success=False, meta=None, errors=errors)
        else:
            # Usa corpus existente
            raw_input = self.root / "data" / "corpus" / "raw_classical"
            if not raw_input.exists():
                raw_input = self.root / "f51_darwin" / "dataset" / "raw_classical"

            if not raw_input.exists():
                corpus_candidates = list(self.root.rglob("gutenberg_*.txt"))
                if corpus_candidates:
                    raw_input = corpus_candidates[0].parent
                else:
                    errors.append("Corpus não encontrado. Execute sem --skip-download")
                    return PackResult(success=False, meta=None, errors=errors)

        # Etapa 2: Buildar pack aprovado
        if not skip_pack:
            success, corpus_pack_path = self.build_approved_pack(raw_input)
            if not success:
                errors.append("Falha ao buildar pack aprovado")
                return PackResult(success=False, meta=None, errors=errors)
        else:
            # Usa pack existente
            existing_packs = list(self.pack_output_dir.glob("approved_corpus_pack*.tar.gz"))
            if existing_packs:
                corpus_pack_path = str(max(existing_packs, key=lambda p: p.stat().st_mtime))
                warnings.append(f"Usando pack existente: {Path(corpus_pack_path).name}")
            else:
                errors.append("Nenhum pack aprovado encontrado. Execute sem --skip-pack")
                return PackResult(success=False, meta=None, errors=errors)

        # Etapa 3: Criar training pack final
        success, meta = self.create_training_pack(
            corpus_pack_path=corpus_pack_path,
            include_tokenizer=include_tokenizer,
        )

        if not success:
            errors.append("Falha ao criar training pack final")
            return PackResult(success=False, meta=None, errors=errors)

        self._print_header("TRAINING PACK PRONTO")

        print(f"Pack:        {meta.pack_name}")
        print(f"SHA256:      {meta.pack_sha256}")
        print(f"Tamanho:     {meta.pack_size_mb:.2f} MB")
        print(f"Caminho:     {meta.pack_path}")
        print(f"\nPronto para upload com:")
        print(f"  python scripts/upload_corpus_pack.py {meta.pack_path}")

        return PackResult(
            success=True,
            meta=meta,
            errors=errors,
            warnings=warnings,
        )


def main():
    parser = argparse.ArgumentParser(
        description="F51 Training Pack Builder — Prepara material para nuvem"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Diretório de trabalho (default: cwd)"
    )
    parser.add_argument(
        "--batch-name",
        type=str,
        default=None,
        help="Nome do batch (default: timestamp)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Pula download de corpus (usa existente)"
    )
    parser.add_argument(
        "--skip-pack",
        action="store_true",
        help="Pusa build de pack (usa existente)"
    )
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Não inclui tokenizer no pack final"
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Apenas baixa corpus, não builda pack"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Saída em JSON"
    )

    args = parser.parse_args()

    builder = TrainingPackBuilder(
        project_root=ROOT,
        workspace=Path(args.workspace) if args.workspace else None,
        batch_name=args.batch_name,
    )

    if args.raw_only:
        # Apenas download
        builder._print_header("DOWNLOAD DE CORPUS APENAS")
        total_books, total_chars = builder.generate_raw_corpus()
        if args.json:
            print(json.dumps({
                "success": total_books > 0,
                "total_books": total_books,
                "total_chars": total_chars,
            }))
        return 0 if total_books > 0 else 1

    result = builder.build_full_pipeline(
        skip_download=args.skip_download,
        skip_pack=args.skip_pack,
        include_tokenizer=not args.no_tokenizer,
    )

    if args.json:
        output = {
            "success": result.success,
            "errors": result.errors,
            "warnings": result.warnings,
        }
        if result.meta:
            output["pack"] = asdict(result.meta)
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        if result.warnings:
            print("\nAvisos:")
            for w in result.warnings:
                print(f"  ⚠ {w}")

        if result.errors:
            print("\nErros:")
            for e in result.errors:
                print(f"  ✗ {e}")

    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
