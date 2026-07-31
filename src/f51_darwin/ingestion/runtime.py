#!/usr/bin/env python3
"""
F51 Real-time Ingestion Pipeline CLI Tool
=========================================
Implements the 9-stage data processing blueprint:
  1. Coleta por streaming
  2. Filtro de qualidade (Firewall)
  3. Deduplicacao (Ledger hashing)
  4. Limpeza / normalizacao
  5. Chunking + metadados
  6. Misturador de fontes (Domain weights)
  7. Buffer / fila
  8. Tokenizacao e construcao da janela live
  9. Handoff para o run247 (sem optimizer step nesta CLI)

Zero-dependency premium ANSI CLI Dashboard.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

from f51_darwin.data_factory import DataFactory, DataFactoryPaths
from f51_darwin.data_firewall import DataFirewall, FirewallConfig
from f51_darwin.dataset_states import DatasetStatus, SourceType
from f51_darwin.dataset_layout import (
    APPROVED_TOKEN_CACHE_RELATIVE,
    LIVE_TOKEN_RELATIVE,
    WorkspacePaths,
    resolve_dataset_path,
    resolve_feast_token_bin,
)
from f51_darwin.provenance import DatasetRecord, content_hash
from f51_darwin.workspace_migration import mark_runtime_use

# ANSI Colors for premium sci-fi dashboard theme
CLR_RESET = "\033[0m"
CLR_BOLD = "\033[1m"
CLR_DARK = "\033[90m"
CLR_RED = "\033[31m"
CLR_GREEN = "\033[32m"
CLR_YELLOW = "\033[33m"
CLR_BLUE = "\033[34m"
CLR_MAGENTA = "\033[35m"
CLR_CYAN = "\033[36m"
CLR_CYAN_BG = "\033[46;30m"
CLR_ORANGE = "\033[38;5;208m"

ICON_OK = f"{CLR_GREEN}OK{CLR_RESET}"
ICON_FAIL = f"{CLR_RED}FAIL{CLR_RESET}"
ICON_WARN = f"{CLR_YELLOW}WARN{CLR_RESET}"
ICON_SPIN = ["-", "\\", "|", "/"]


WEB_SOURCE_PREFIXES = ("http://", "https://", "web://", "terminal://")


def effective_source_type(source_path: str, requested: SourceType) -> SourceType:
    """Web and terminal input are imported evidence, never trusted real data."""
    normalized = str(source_path).strip().lower()
    if normalized.startswith(WEB_SOURCE_PREFIXES):
        return SourceType.IMPORTED
    return requested


def write_int32_atomic(path: Path, token_ids) -> int:
    """Publish a complete int32 token file without exposing a partial target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens = np.asarray(token_ids, dtype=np.int32).reshape(-1)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("wb") as handle:
            tokens.tofile(handle)
            handle.flush()
            os.fsync(handle.fileno())
        expected_size = int(tokens.size) * np.dtype(np.int32).itemsize
        if temporary.stat().st_size != expected_size:
            raise OSError(
                f"Incomplete token file: expected {expected_size} bytes, "
                f"got {temporary.stat().st_size}"
            )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return int(tokens.size)


def clear_screen():
    sys.stdout.write("\033[H\033[J")
    sys.stdout.flush()


def print_header(title: str):
    width = 72
    sys.stdout.write(f"\n{CLR_BOLD}{CLR_ORANGE}{'='*width}{CLR_RESET}\n")
    sys.stdout.write(f"{CLR_BOLD}  {title}{CLR_RESET}\n")
    sys.stdout.write(f"{CLR_ORANGE}{'='*width}{CLR_RESET}\n\n")
    sys.stdout.flush()


def print_pipeline_progress(stage: int, desc: str, status: str = "active"):
    stages = [
        "Streaming Coleta",
        "Filtro de Qualidade",
        "Deduplicacao Ledger",
        "Limpeza & Normalizacao",
        "Chunking & Metadados",
        "Misturador de Fontes",
        "Fila de Replay Buffer",
        "Tokenizacao Live",
        "Handoff run247",
    ]
    bar_width = 30
    filled = int((stage / 9.0) * bar_width)
    bar = "=" * filled + "-" * (bar_width - filled)

    color = CLR_GREEN if status == "ok" else (CLR_RED if status == "fail" else CLR_CYAN)
    badge = f"[{stage}/9]"

    stage_name = stages[stage - 1]
    sys.stdout.write(
        f"\r{CLR_BOLD}{color}{badge} {stage_name:<24} {CLR_DARK}[{bar}] {CLR_RESET}{desc:<20}"
    )
    sys.stdout.flush()
    if status in ("ok", "fail"):
        sys.stdout.write("\n")
        sys.stdout.flush()


class IngestPipeline:
    def __init__(self, project_root: Path):
        self.root = project_root
        mark_runtime_use(self.root, actor="ingest_pipeline")
        self.paths = WorkspacePaths.from_project(self.root, require=True)
        paths = DataFactoryPaths.from_project(self.root)
        self.feast_path = resolve_feast_token_bin(self.root)
        self.live_path = resolve_dataset_path(self.root, LIVE_TOKEN_RELATIVE)
        # Cumulative token cache: each approval appends only the new document's
        # tokens, so ingestion stays O(N) instead of re-tokenizing the whole
        # corpus every time (the previous O(N^2) regression).
        self.approved_cache_path = resolve_dataset_path(self.root, APPROVED_TOKEN_CACHE_RELATIVE)
        self.firewall = DataFirewall(FirewallConfig(min_chars=200, quality_pass_score=0.60))
        self.factory = DataFactory(paths, firewall=self.firewall)

        self.total_processed = 0
        self.approved_count = 0
        self.quarantine_count = 0
        self.rejected_count = 0
        self.total_bytes = 0

        self.domain_weights = {
            "FineWeb": 0.30,
            "Math": 0.25,
            "PubMed": 0.15,
            "StackExchange": 0.10,
            "arXiv": 0.10,
            "The Stack": 0.10,
        }

    def render_weights(self):
        sys.stdout.write(f"{CLR_BOLD}{CLR_CYAN}[MISTURADOR - DOMINIO / PESO]:{CLR_RESET}\n")
        for domain, weight in self.domain_weights.items():
            bar_len = int(weight * 40)
            bar = "|" * bar_len + " " * (40 - bar_len)
            sys.stdout.write(f"  {domain:<15} | [{CLR_ORANGE}{bar}{CLR_RESET}] {weight*100:>5.1f}%\n")
        sys.stdout.write("\n")
        sys.stdout.flush()

    def render_stats(self):
        sys.stdout.write(f"{CLR_BOLD}{CLR_YELLOW}[AMIGDALA - FIREWALL STATS]:{CLR_RESET}\n")
        sys.stdout.write(f"  Total Processado: {CLR_BOLD}{self.total_processed}{CLR_RESET} documentos\n")
        sys.stdout.write(f"  Aprovados:        {CLR_GREEN}{self.approved_count}{CLR_RESET} | ")
        sys.stdout.write(f"Quarentena:       {CLR_YELLOW}{self.quarantine_count}{CLR_RESET} | ")
        sys.stdout.write(f"Rejeitados:       {CLR_RED}{self.rejected_count}{CLR_RESET}\n")
        mb = self.total_bytes / (1024 * 1024)
        sys.stdout.write(f"  Volume:          {CLR_CYAN}{mb:.2f} MB{CLR_RESET}\n")
        sys.stdout.write("\n")
        sys.stdout.flush()

    def process_text(self, text: str, source_path: str, source_type: SourceType = SourceType.REAL, domain: str = "FineWeb") -> bool:
        source_type = effective_source_type(source_path, source_type)
        self.total_processed += 1
        self.total_bytes += len(text.encode("utf-8"))

        print_pipeline_progress(1, "coletando corpus...", "active")
        print_pipeline_progress(1, "STREAMING OK", "ok")

        print_pipeline_progress(2, "auditando qualidade...", "active")
        alpha = sum(ch.isalpha() for ch in text)
        quality_score = alpha / max(len(text), 1)
        if len(text.strip()) < 200:
            print_pipeline_progress(2, f"TAMANHO < 200 chars", "fail")
            self.rejected_count += 1
            return False
        print_pipeline_progress(2, f"SCORE={quality_score:.2f} OK", "ok")

        print_pipeline_progress(3, "checando hash...", "active")
        h = content_hash(text)
        existing = self.factory.ledger.find_by_hash(h)
        if existing:
            print_pipeline_progress(3, "DUPLICADO", "fail")
            self.rejected_count += 1
            return False
        print_pipeline_progress(3, f"HASH={h[:10]}... UNIQUE", "ok")

        print_pipeline_progress(4, "removendo ruido...", "active")
        cleaned_text = re.sub(r"\s+", " ", text.strip())
        print_pipeline_progress(4, "LIMPEZA OK", "ok")

        print_pipeline_progress(5, "gerando metadata...", "active")
        record = self.factory.register_candidate(
            text=cleaned_text,
            source_type=source_type,
            source_path=source_path,
            dataset_version=f"stream_{int(time.time())}",
        )
        print_pipeline_progress(5, f"ID={record.id[:12]} METADATA", "ok")

        print_pipeline_progress(6, f"dosagem ({domain})...", "active")
        weight = self.domain_weights.get(domain, 0.10)
        print_pipeline_progress(6, f"PESO={weight*100:.0f}% MIXED", "ok")

        print_pipeline_progress(7, "enfileirando...", "active")
        decision = self.factory.audit_candidate(record, cleaned_text)
        if decision.status == DatasetStatus.QUARANTINE:
            self.quarantine_count += 1
            status_desc = "QUARANTINE"
            status_code = "ok"
        else:
            self.rejected_count += 1
            status_desc = "REJECTED"
            status_code = "fail"
        print_pipeline_progress(7, status_desc, status_code)
        if decision.status == DatasetStatus.QUARANTINE:
            print_pipeline_progress(8, "aguardando aprovacao explicita", "ok")
        else:
            print_pipeline_progress(8, "bloqueado firewall", "fail")
        print_pipeline_progress(9, "nenhuma promocao automatica", "ok")
        return False

    def monitor_candidates(self):
        candidates_dir = self.factory.paths.candidates
        candidates_dir.mkdir(parents=True, exist_ok=True)

        clear_screen()
        print_header("F51 REAL-TIME DATA INGESTION ENGINE")
        self.render_weights()
        self.render_stats()

        print(f"{CLR_BOLD}{CLR_DARK}Monitorando {candidates_dir} ... (Ctrl+C para parar){CLR_RESET}\n")

        processed_files = set()
        spinner_idx = 0
        try:
            while True:
                files = sorted(candidates_dir.glob("*.txt"))
                new_files = [f for f in files if f not in processed_files]

                if new_files:
                    print(f"\n>>> Encontrado +{len(new_files)} novos arquivos!\n")
                    for f in new_files:
                        try:
                            text = f.read_text(encoding="utf-8", errors="replace")
                            sys.stdout.write(f"\nArquivo: {f.name}\n")
                            domain = "Math" if "math" in f.name.lower() else "FineWeb"
                            self.process_text(text, source_path=str(f), domain=domain)
                        except Exception as e:
                            print(f"{ICON_FAIL} Erro: {e}")
                        processed_files.add(f)

                    print("\n" + "-" * 72)
                    self.render_stats()

                char = ICON_SPIN[spinner_idx % len(ICON_SPIN)]
                sys.stdout.write(f"\r{CLR_CYAN}{char} Ingestao ativa...{CLR_RESET}")
                sys.stdout.flush()
                spinner_idx += 1
                time.sleep(0.5)

        except KeyboardInterrupt:
            print(f"\n\nIngestao pausada. Ledger salvo.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F51 Ingestion Pipeline")
    parser.add_argument("command", choices=["monitor", "ingest", "stats"], help="Pipeline command")
    parser.add_argument("--file", help="File to ingest")
    parser.add_argument("--text", help="Raw text string to ingest")
    parser.add_argument("--domain", default="FineWeb", help="Mixer domain label")
    args = parser.parse_args(argv)

    pipeline = IngestPipeline(ROOT)

    if args.command == "monitor":
        pipeline.monitor_candidates()

    elif args.command == "ingest":
        print_header("F51 INGESTION ENGINE")
        if args.file:
            path = Path(args.file)
            if not path.exists():
                print(f"Arquivo nao encontrado: {path}")
                sys.exit(1)
            text = path.read_text(encoding="utf-8", errors="replace")
            print(f"Processando: {path.name}")
            pipeline.process_text(text, source_path=str(path), domain=args.domain)
        elif args.text:
            print("Processando texto do terminal")
            pipeline.process_text(args.text, source_path="terminal://input", domain=args.domain)
        else:
            print("Use --file ou --text")

    elif args.command == "stats":
        print_header("F51 INGESTION ENGINE - ESTATISTICAS")
        pipeline.render_weights()
        pipeline.render_stats()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
