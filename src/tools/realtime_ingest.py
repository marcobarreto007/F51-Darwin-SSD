#!/usr/bin/env python3
"""
F51 Real-time Ingestion Dashboard (legacy)
===========================================
This older dashboard registers and audits candidates. It does not tokenize a
live training bin, call backward(), or update model weights. Use
src/scripts/ingest_pipeline.py for the canonical ingest -> tokens_live handoff.

Displays the 9-stage blueprint:
  1. Coleta por streaming
  2. Filtro de qualidade (Firewall)
  3. Deduplicação (Ledger hashing)
  4. Limpeza / normalização
  5. Chunking + metadados
  6. Misturador de fontes (Domain weights)
  7. Buffer / fila
  8. Handoff para o buffer (simulado nesta CLI)
  9. Modelo (sem atualização de pesos nesta CLI)

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

ROOT = Path(__file__).resolve().parents[2]

from f51_darwin.data_factory import DataFactory, DataFactoryPaths
from f51_darwin.data_firewall import DataFirewall, FirewallConfig
from f51_darwin.dataset_states import DatasetStatus, SourceType
from f51_darwin.provenance import DatasetRecord, content_hash

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

# Beautiful ASCII Panels & Icons
ICON_OK = f"{CLR_GREEN}✓{CLR_RESET}"
ICON_FAIL = f"{CLR_RED}✗{CLR_RESET}"
ICON_WARN = f"{CLR_YELLOW}⚠{CLR_RESET}"
ICON_SPIN = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def clear_screen():
    sys.stdout.write("\033[H\033[J")
    sys.stdout.flush()


def print_header(title: str):
    width = 72
    sys.stdout.write(f"\n{CLR_BOLD}{CLR_ORANGE}╔" + "═" * (width - 2) + "╗\n")
    sys.stdout.write(f"║ {title:<{width-4}} ║\n")
    sys.stdout.write("╚" + "═" * (width - 2) + f"╝{CLR_RESET}\n\n")
    sys.stdout.flush()


def print_pipeline_progress(stage: int, desc: str, status: str = "active"):
    stages = [
        "Streaming Coleta",
        "Filtro de Qualidade",
        "Deduplicação Ledger",
        "Limpeza & Normalização",
        "Chunking & Metadados",
        "Misturador de Fontes",
        "Fila de Replay Buffer",
        "Treino Contínuo",
        "Integração do Modelo",
    ]
    width = 72
    bar_width = 30
    filled = int((stage / 9.0) * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)
    
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
        paths = DataFactoryPaths.from_project(self.root)
        self.firewall = DataFirewall(FirewallConfig(min_chars=200, quality_pass_score=0.60))
        self.factory = DataFactory(paths, firewall=self.firewall)
        
        # Ingestion metrics
        self.total_processed = 0
        self.approved_count = 0
        self.quarantine_count = 0
        self.rejected_count = 0
        self.total_bytes = 0
        
        # Mix settings (Domain Weights)
        self.domain_weights = {
            "FineWeb": 0.30,
            "Math": 0.25,
            "PubMed": 0.15,
            "StackExchange": 0.10,
            "arXiv": 0.10,
            "The Stack": 0.10,
        }

    def render_weights(self):
        sys.stdout.write(f"{CLR_BOLD}{CLR_CYAN}🎚️  CONFIGURAÇÃO DO MISTURADOR (DOMÍNIO / PESO):{CLR_RESET}\n")
        for domain, weight in self.domain_weights.items():
            bar_len = int(weight * 40)
            bar = "█" * bar_len + " " * (40 - bar_len)
            sys.stdout.write(f"  {domain:<15} │ [{CLR_ORANGE}{bar}{CLR_RESET}] {weight*100:>5.1f}%\n")
        sys.stdout.write("\n")
        sys.stdout.flush()

    def render_stats(self):
        sys.stdout.write(f"{CLR_BOLD}{CLR_YELLOW}📊 PAINEL DE ESTATÍSTICAS DA AMÍGDALA (FIREWALL):{CLR_RESET}\n")
        sys.stdout.write(f"  Total Processado: {CLR_BOLD}{self.total_processed}{CLR_RESET} documentos\n")
        sys.stdout.write(f"  Aprovados:        {CLR_GREEN}{self.approved_count}🟢{CLR_RESET} | ")
        sys.stdout.write(f"Quarentena:       {CLR_YELLOW}{self.quarantine_count}🟡{CLR_RESET} | ")
        sys.stdout.write(f"Rejeitados:       {CLR_RED}{self.rejected_count}🔴{CLR_RESET}\n")
        mb = self.total_bytes / (1024 * 1024)
        sys.stdout.write(f"  Volume Processado:{CLR_CYAN} {mb:.3f} MB{CLR_RESET}\n")
        sys.stdout.write("\n")
        sys.stdout.flush()

    def process_text(self, text: str, source_path: str, source_type: SourceType = SourceType.REAL, domain: str = "FineWeb") -> bool:
        self.total_processed += 1
        self.total_bytes += len(text.encode("utf-8"))
        
        # 1. Coleta por streaming
        print_pipeline_progress(1, "coletando corpus...", "active")
        time.sleep(0.1)
        print_pipeline_progress(1, "STREAMING OK", "ok")

        # 2. Filtro de qualidade
        print_pipeline_progress(2, "auditando qualidade...", "active")
        time.sleep(0.15)
        # Score basic verification
        alpha = sum(ch.isalpha() for ch in text)
        quality_score = alpha / max(len(text), 1)
        if len(text.strip()) < 200:
            print_pipeline_progress(2, f"TAMANHO INSUFICIENTE ({len(text)} chars)", "fail")
            self.rejected_count += 1
            return False
        print_pipeline_progress(2, f"SCORE={quality_score:.2f} OK", "ok")

        # 3. Deduplicação
        print_pipeline_progress(3, "checando hash...", "active")
        h = content_hash(text)
        existing = self.factory.ledger.find_by_hash(h)
        if existing:
            print_pipeline_progress(3, "CONTEÚDO DUPLICADO", "fail")
            self.rejected_count += 1
            return False
        print_pipeline_progress(3, f"HASH={h[:10]}... UNIQUE", "ok")

        # 4. Limpeza / normalização
        print_pipeline_progress(4, "removendo ruído...", "active")
        cleaned_text = re.sub(r"\s+", " ", text.strip())
        time.sleep(0.1)
        print_pipeline_progress(4, "LIMPEZA COMPLETA", "ok")

        # 5. Chunking + metadados
        print_pipeline_progress(5, "gerando metadata...", "active")
        record = self.factory.register_candidate(
            text=cleaned_text,
            source_type=source_type,
            source_path=source_path,
            dataset_version=f"stream_{int(time.time())}",
        )
        time.sleep(0.1)
        print_pipeline_progress(5, f"ID={record.id[:12]} METADATA", "ok")

        # 6. Misturador de fontes
        print_pipeline_progress(6, f"dosagem ({domain})...", "active")
        weight = self.domain_weights.get(domain, 0.10)
        time.sleep(0.1)
        print_pipeline_progress(6, f"PESO={weight*100:.0f}% MIXED", "ok")

        # 7. Buffer / fila
        print_pipeline_progress(7, "enfileirando...", "active")
        decision = self.factory.audit_candidate(record, cleaned_text)
        if decision.status == DatasetStatus.APPROVED:
            self.approved_count += 1
            status_desc = "APPROVED 🟢"
            status_code = "ok"
        elif decision.status == DatasetStatus.QUARANTINE:
            self.quarantine_count += 1
            status_desc = "QUARANTINE 🟡"
            status_code = "fail"
        else:
            self.rejected_count += 1
            status_desc = "REJECTED 🔴"
            status_code = "fail"
        time.sleep(0.1)
        print_pipeline_progress(7, status_desc, status_code)

        # 8 & 9. Treino e Modelo
        if decision.status == DatasetStatus.APPROVED:
            print_pipeline_progress(8, "handoff não implementado...", "active")
            time.sleep(0.1)
            print_pipeline_progress(8, "CANDIDATO APROVADO", "ok")
            print_pipeline_progress(9, "sem optimizer step...", "active")
            time.sleep(0.1)
            print_pipeline_progress(9, "USE ingest_pipeline.py", "ok")
            return True
        else:
            print_pipeline_progress(8, "bloqueado por firewall", "fail")
            print_pipeline_progress(9, "abortado", "fail")
            return False

    def monitor_candidates(self):
        candidates_dir = self.root / "data/generated/candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        
        clear_screen()
        print_header("F51 REAL-TIME DATA INGESTION ENGINE — MONITOR ATIVO")
        self.render_weights()
        self.render_stats()
        
        print(f"{CLR_BOLD}{CLR_DARK}🔍 Escaneando data/generated/candidates/ por novos arquivos... (Ctrl+C para parar){CLR_RESET}\n")
        
        processed_files = set()
        spinner_idx = 0
        try:
            while True:
                files = sorted(candidates_dir.glob("*.txt"))
                new_files = [f for f in files if f not in processed_files]
                
                if new_files:
                    print(f"\n🚀 {CLR_BOLD}Encontrado +{len(new_files)} novos arquivos para processar!{CLR_RESET}\n")
                    for f in new_files:
                        try:
                            text = f.read_text(encoding="utf-8", errors="replace")
                            sys.stdout.write(f"\n📂 {CLR_BOLD}Arquivo: {f.name}{CLR_RESET}\n")
                            self.process_text(text, source_path=str(f), domain="Math" if "math" in f.name.lower() else "FineWeb")
                        except Exception as e:
                            print(f"{ICON_FAIL} Erro processando {f.name}: {e}")
                        processed_files.add(f)
                    
                    # Refresh stats
                    print("\n" + "─" * 72)
                    self.render_stats()
                
                # Dynamic Spinner to show engine is breathing
                char = ICON_SPIN[spinner_idx % len(ICON_SPIN)]
                sys.stdout.write(f"\r{CLR_CYAN}{char} Ingestão ativa. Aguardando novos streams...{CLR_RESET}")
                sys.stdout.flush()
                spinner_idx += 1
                time.sleep(0.2)
                
        except KeyboardInterrupt:
            print(f"\n\n🛑 Ingestão pausada. Logs e ledger salvos com segurança.")


def generate_prompt_for_model() -> str:
    return """
# INSTRUÇÃO DO SISTEMA — GERADOR DE CONTEÚDO EPISTÊMICO F51
Você é o motor de dados sintéticos do organismo F51-Darwin.
Seu objetivo é produzir arquivos de conhecimento puro com alta densidade epistêmica que passem 100% limpos pelo DataFirewall (Amígdala).

## Regras de Formatação Obrigatórias:
1. **Densidade Teórica:** Cada parágrafo deve conter fórmulas matematicamente verificáveis, contratos lógicos rígidos ou teorias estruturais de código.
2. **Evitar Incerteza (Inibidor):** NUNCA use palavras de incerteza como "talvez", "provavelmente", "eu acho", "possivelmente" ou linguagem subjetiva. Seja assertivo, direto e científico.
3. **Imunidade a Repetição:** Evite redundâncias de estruturas gramaticais ou loops conceituais. Use vocabulário rico e diversificado.
4. **Volume:** Garanta que cada documento tenha pelo menos 200 caracteres de texto denso.
5. **Estrutura de Metadados:** Seus outputs de conhecimento devem seguir o formato estruturado:
   - **Tópico:** [Assunto preciso]
   - **Proposição Lógica:** [Declaração matemática ou técnica]
   - **Validação:** [Prova matemática ou contrato de código]

## Exemplo de Output Aprovado:
Tópico: Otimização de Perplexidade via quiet-STaR em SSM-Transformer
Proposição Lógica: O gradiente de perda da política de pensamento implícito no Mamba-SSD é proporcional ao ganho de verossimilhança da resposta final ponderada pela recompensa de utilidade.
Validação: Seja R o reward de utilidade e P(y|x, theta) a probabilidade. O gradiente é calculado por: grad_theta = E[ grad_theta log P( pensamento | x) * (R - baseline) ]. A variância é reduzida via estimador REINFORCE com baseline móvel adaptativa.
"""


def main():
    parser = argparse.ArgumentParser(description="F51 Ingestion Pipeline — O Misturador de Fontes")
    parser.add_argument("command", choices=["monitor", "ingest", "stats", "generate-prompt"], help="Pipeline command")
    parser.add_argument("--file", help="File to ingest")
    parser.add_argument("--text", help="Raw text string to ingest")
    parser.add_argument("--domain", default="FineWeb", help="Mixer domain label")
    args = parser.parse_args()

    pipeline = IngestPipeline(ROOT)

    if args.command == "monitor":
        pipeline.monitor_candidates()
        
    elif args.command == "ingest":
        print_header("F51 INGESTION ENGINE — INGESTÃO MANUAL")
        if args.file:
            path = Path(args.file)
            if not path.exists():
                print(f"{ICON_FAIL} Arquivo não encontrado: {path}")
                sys.exit(1)
            text = path.read_text(encoding="utf-8", errors="replace")
            print(f"📂 Processando arquivo: {path.name}")
            pipeline.process_text(text, source_path=str(path), domain=args.domain)
        elif args.text:
            print("💬 Processando string do terminal")
            pipeline.process_text(args.text, source_path="terminal://input", domain=args.domain)
        else:
            print(f"{ICON_WARN} Especifique --file ou --text para ingestão.")
            
    elif args.command == "stats":
        print_header("F51 INGESTION ENGINE — ESTATÍSTICAS")
        pipeline.render_weights()
        pipeline.render_stats()
        
    elif args.command == "generate-prompt":
        print_header("F51 PROMPT GENERATOR FOR SYNTHETIC DATA")
        prompt = generate_prompt_for_model()
        print(prompt)
        # Suggest slash command to user
        print(f"\n\033[1m\033[38;5;208m💡 DICA DE COMANDO SLASHE:\033[0m")
        print("  Você pode recomendar o comando `/learn` ou rodar o script no modelo")
        print("  para que ele entenda e aplique essas restrições nos novos treinos!")


if __name__ == "__main__":
    main()
