#!/usr/bin/env python3
"""
FASE A — Fundação trilíngue com curriculum ponderado.

Treina o organismo do zero (ou de checkpoint existente) usando
WeightedCorpusLoader com pesos configuráveis por fase.

Uso:
  # Fase A: só literatura clássica
  python research/train_curriculum.py --phase foundation

  # Fase B: literatura + identidade (peso 5×)
  python research/train_curriculum.py --phase identity --resume checkpoints/organism/organism_cycle_050.pt

  # Fase C: literatura + identidade + filosofia (peso 3×)
  python research/train_curriculum.py --phase philosophy --resume ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from f51_darwin.data import WeightedCorpusLoader
from f51_darwin.dataset_layout import TOKENIZED_RELATIVE, resolve_dataset_path
from scripts.darwin_organism import DarwinOrganism, DarwinOrganismConfig
from scripts.serve_davi import serve_organism


# ═══════════════════════════════════════════════════════════════════════════
# Curriculum phases
# ═══════════════════════════════════════════════════════════════════════════

CURRICULUM = {
    "foundation": {
        "desc": "Fase A: Fundação — Literatura Clássica Trilíngue e Wikipédia",
        "sources": [
            ("CLASSICOS/tokens_classical.bin", 0.30),
            ("CLASSICOS/tokens_gutenberg_classics.bin", 0.30),
            ("WIKI/tokens_wiki_en.bin", 0.20),
            ("WIKI/tokens_wiki_pt.bin", 0.20),
        ],
    },
    "reasoning": {
        "desc": "Fase B: Raciocínio — Foco em Matemática e Lógica",
        "sources": [
            ("MATEMATICA/tokens_openwebmath2.bin", 0.20),
            ("MATEMATICA/tokens_openwebmath.bin", 0.10),
            ("MATEMATICA/tokens_fineweb_edu.bin", 0.10),
            ("MATEMATICA/tokens_metamath.bin", 0.10),
            ("00_CORPUS_PRINCIPAL_tokens_feast.bin", 0.30),
            ("WIKI/tokens_wiki_en.bin", 0.10),
            ("WIKI/tokens_wiki_pt.bin", 0.10),
        ],
    },
    "identity": {
        "desc": "Fase C: Identidade — Alinhamento F51 e Personalidade do Davi",
        "sources": [
            ("CORPUS_ESPECIALIZADOS/tokens_identity_200x.bin", 0.25),
            ("CORPUS_ESPECIALIZADOS/tokens_prefix.bin", 0.25),
            ("00_CORPUS_PRINCIPAL_tokens_feast.bin", 0.30),
            ("MATEMATICA/tokens_openwebmath2.bin", 0.10),
            ("WIKI/tokens_wiki_en.bin", 0.10),
        ],
    },
    "balanced": {
        "desc": "Fase D: Misto Equilibrado — Todas as áreas consolidadas (Foco em STEM e Valores)",
        "sources": [
            ("00_CORPUS_PRINCIPAL_tokens_feast.bin", 0.42),
            ("MATEMATICA/tokens_openwebmath2.bin", 0.18),
            ("MATEMATICA/tokens_openwebmath.bin", 0.05),
            ("MATEMATICA/tokens_fineweb_edu.bin", 0.08),
            ("MATEMATICA/tokens_metamath.bin", 0.08),
            ("CORPUS_ESPECIALIZADOS/tokens_identity_200x.bin", 0.08),
            ("CORPUS_ESPECIALIZADOS/tokens_prefix.bin", 0.08),
            ("CLASSICOS/tokens_gutenberg_classics.bin", 0.015),
            ("CLASSICOS/tokens_classical.bin", 0.01),
            ("CLASSICOS/tokens_live.bin", 0.005),
        ],
    },
}


def load_weighted_sources(phase: str, root: Path) -> list[tuple[np.ndarray, float]]:
    """Load token files as memmap with curriculum weights."""
    spec = CURRICULUM[phase]
    print(f"📖 {spec['desc']}")
    print()

    tokenized_root = resolve_dataset_path(root, TOKENIZED_RELATIVE, require=True)

    sources = []
    for path_str, weight in spec["sources"]:
        full = tokenized_root / path_str
        if not full.exists():
            print(f"  ⚠️  {path_str}: arquivo não encontrado, pulando")
            continue
        tokens = np.memmap(full, dtype=np.int32, mode="r")
        sources.append((tokens, weight))
        print(f"  ✅ {path_str}: {len(tokens)/1e6:.1f}M tokens × {weight:.0f}")

    if not sources:
        raise FileNotFoundError("Nenhuma fonte de tokens encontrada")

    print()
    return sources



# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="F51 Curriculum Training")
    parser.add_argument(
        "--phase",
        default="foundation",
        choices=list(CURRICULUM.keys()),
        help="Curriculum phase",
    )
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--config", default="src/configs/darwin_x_600m.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=5151)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=500, help="Steps per cycle")
    parser.add_argument("--no-train", action="store_true")
    args = parser.parse_args()

    root = ROOT.resolve()
    tokenized_root = resolve_dataset_path(root, TOKENIZED_RELATIVE, require=True)
    primary_token_bin = tokenized_root / CURRICULUM[args.phase]["sources"][0][0]

    # Load weighted sources
    sources = load_weighted_sources(args.phase, root)

    # Bootstrap organism
    config = DarwinOrganismConfig(
        project_root=str(root),
        device=args.device,
        block_size=args.block_size,
        heartbeat_enabled=True,
        ghost_enabled=True,
        jepa_enabled=True,
        curiosity_enabled=True,
        spider_enabled=True,
    )
    organism = DarwinOrganism(config, str(root / args.config))

    # Override token source for single-source backward compat
    organism.token_ids = sources[0][0]
    organism.token_path = str(primary_token_bin)
    organism.token_count = len(organism.token_ids)

    # Inject weighted sources for curriculum loader
    organism.weighted_sources = sources

    organism.bootstrap(
        token_bin=str(primary_token_bin),
        resume=args.resume,
    )
    # Re-override after bootstrap (bootstrap reloads from args)
    organism.weighted_sources = sources

    serve_organism(
        organism,
        host="127.0.0.1",
        port=args.port,
        train=not args.no_train,
        steps_per_cycle=args.steps,
    )


if __name__ == "__main__":
    main()
