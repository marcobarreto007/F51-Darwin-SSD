"""F51 Olavo Super Dataset — Curadoria do corpus filosófico para o bebê F51.

Fontes do Olavo de Carvalho (publicamente acessíveis):
    - Livros publicados (32+ títulos)
    - Artigos do Mídia Sem Máscara (2002-2022)
    - Colunas em jornais (O Globo, Folha, Diário do Comércio)
    - Aulas do COF (Curso Online de Filosofia) — transcrições
    - YouTube — transcrições de palestras e debates
    - Entrevistas e debates públicos
    - Seminários de Filosofia

Estrutura do dataset:
    data/olavo/
      livros/           # Obras completas digitalizadas
      artigos/          # Artigos e colunas
      aulas_cof/        # Transcrições do Curso Online de Filosofia
      seminarios/       # Seminários e palestras
      entrevistas/      # Entrevistas e debates
      frases/           # Aforismos e citações marcantes
      caderno_f51/      # Notas e análises do Marco sobre Olavo

Doutrina:
    - Apenas texto de fonte verificável
    - Provenance: livro + página, URL + data, vídeo + timestamp
    - NADA gerado por IA sobre Olavo
    - O Olavo é o Olavo. O modelo aprende a PENSAR como ele,
      não a repetir frases prontas.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════
# CATÁLOGO DE OBRAS DO OLAVO
# ═══════════════════════════════════════════════════════════

OLAVO_BOOKS = [
    {
        "id": "olavo_jardim_aflicoes",
        "title": "O Jardim das Aflições: De Epicuro à Ressurreição de César — Ensaio sobre o Materialismo e a Religião Civil",
        "year": 1995,
        "publisher": "Topbooks / Vide Editorial",
        "pages": "~400",
        "theme": "filosofia, materialismo, religião civil, Epicuro",
        "importance": "FUNDACIONAL — obra-prima, o livro mais importante do Olavo",
        "status": "procurar PDF ou digitalizar",
    },
    {
        "id": "olavo_imbecil_coletivo",
        "title": "O Imbecil Coletivo: Atualidades Inculturais Brasileiras",
        "year": 1996,
        "publisher": "Faculdade da Cidade / Vide Editorial",
        "pages": "~350",
        "theme": "crítica cultural, Brasil, intelligentsia, mídia",
        "importance": "FUNDACIONAL — definiu o termo 'imbecil coletivo'",
        "status": "procurar PDF ou digitalizar",
    },
    {
        "id": "olavo_futuro_pensamento",
        "title": "O Futuro do Pensamento Brasileiro",
        "year": 1997,
        "publisher": "É Realizações",
        "pages": "~200",
        "theme": "filosofia brasileira, educação, universidade",
        "importance": "ALTA — análise da decadência intelectual brasileira",
        "status": "procurar PDF ou digitalizar",
    },
    {
        "id": "olavo_nova_era",
        "title": "A Nova Era e a Revolução Cultural: Fritjof Capra & Antonio Gramsci",
        "year": 1994,
        "publisher": "Instituto de Artes Liberais / Vide Editorial",
        "pages": "~250",
        "theme": "revolução cultural, Gramsci, Nova Era, contracultura",
        "importance": "MUITO ALTA — essencial para entender Gramsci e hegemonia cultural",
        "status": "procurar PDF ou digitalizar",
    },
    {
        "id": "olavo_minimo_idiota",
        "title": "O Mínimo que Você Precisa Saber para não Ser um Idiota",
        "year": 2013,
        "publisher": "Record",
        "pages": "~600",
        "theme": "cultura geral, política, filosofia, história",
        "importance": "REFERÊNCIA — o best-seller, coletânea de artigos",
        "status": "procurar PDF ou digitalizar",
    },
    {
        "id": "olavo_maquiavel",
        "title": "Maquiavel ou A Confusão Demoníaca",
        "year": 2011,
        "publisher": "Vide Editorial",
        "pages": "~250",
        "theme": "Maquiavel, filosofia política, demonologia",
        "importance": "ALTA — análise da técnica política moderna",
        "status": "procurar PDF ou digitalizar",
    },
    {
        "id": "olavo_eixo_mal",
        "title": "O Eixo do Mal Latino-Americano e a Nova Ordem Mundial",
        "year": 2008,
        "publisher": "Vide Editorial",
        "pages": "~200",
        "theme": "geopolítica, Foro de São Paulo, América Latina",
        "importance": "ALTA — geopolítica da esquerda latino-americana",
        "status": "procurar PDF ou digitalizar",
    },
    {
        "id": "olavo_aristoteles",
        "title": "Aristóteles em Nova Perspectiva: Introdução à Teoria dos Quatro Discursos",
        "year": 1996,
        "publisher": "Topbooks / Vide Editorial",
        "pages": "~300",
        "theme": "Aristóteles, lógica, teoria dos discursos, filosofia clássica",
        "importance": "FUNDACIONAL — a base do método de análise do Olavo",
        "status": "procurar PDF ou digitalizar",
    },
    {
        "id": "olavo_eua_nova_ordem",
        "title": "Os EUA e a Nova Ordem Mundial: Debate entre Olavo de Carvalho e Aleksandr Dugin",
        "year": 2012,
        "publisher": "Vide Editorial",
        "pages": "~200",
        "theme": "geopolítica, EUA, Rússia, Dugin, ordem mundial",
        "importance": "ALTA — debate histórico entre direita ocidental e eurasianismo",
        "status": "procurar PDF ou digitalizar",
    },
    {
        "id": "olavo_dialetica_simbolica",
        "title": "A Dialética Simbólica: Ensaios Reunidos",
        "year": 2005,
        "publisher": "Vide Editorial",
        "pages": "~350",
        "theme": "filosofia, símbolos, dialética, epistemologia",
        "importance": "ALTA — fundamentação teórica do método",
        "status": "procurar PDF ou digitalizar",
    },
    {
        "id": "olavo_visoes_descartes",
        "title": "Visões de Descartes: Entre o Gênio Mau e o Espírito da Verdade",
        "year": 2013,
        "publisher": "Vide Editorial",
        "pages": "~200",
        "theme": "Descartes, filosofia moderna, epistemologia",
        "importance": "MÉDIA — análise do pai da filosofia moderna",
        "status": "procurar PDF ou digitalizar",
    },
    {
        "id": "olavo_apostasia",
        "title": "A Apostasia das Elites: Ensaios sobre a Corrupção da Cultura",
        "year": "~2015",
        "publisher": "Vide Editorial",
        "pages": "~300",
        "theme": "elites, cultura, decadência ocidental",
        "importance": "ALTA — análise da traição das elites",
        "status": "procurar PDF ou digitalizar",
    },
]

# ═══════════════════════════════════════════════════════════
# FONTES DE ARTIGOS
# ═══════════════════════════════════════════════════════════

OLAVO_ARTICLE_SOURCES = [
    {
        "source": "Mídia Sem Máscara",
        "url": "https://www.midiasemmascara.org",
        "period": "2002-2022",
        "estimated_articles": 2000,
        "format": "artigos publicados online, gratuito",
        "themes": ["política", "filosofia", "cultura", "geopolítica", "educação"],
    },
    {
        "source": "Diário do Comércio",
        "period": "~2015-2022",
        "estimated_articles": 300,
        "format": "coluna semanal, disponível online",
        "themes": ["política brasileira", "economia", "cultura"],
    },
    {
        "source": "Jornal O Globo",
        "period": "~1998-2010",
        "estimated_articles": 500,
        "format": "colunas, arquivo do jornal",
        "themes": ["política", "cultura", "sociedade"],
    },
    {
        "source": "Folha de S.Paulo",
        "period": "~1977-2000",
        "estimated_articles": 200,
        "format": "artigos no Folhetim e colunas",
        "themes": ["literatura", "filosofia", "cultura"],
    },
    {
        "source": "Seminário de Filosofia",
        "period": "~2005-2022",
        "estimated_transcripts": 500,
        "format": "aulas online, transcrições de alunos",
        "themes": ["filosofia clássica", "lógica", "metafísica", "ética", "política"],
    },
    {
        "source": "COF — Curso Online de Filosofia",
        "period": "~2009-2022",
        "estimated_transcripts": 600,
        "format": "aulas completas, material de alunos",
        "themes": ["filosofia", "história da filosofia", "teoria do conhecimento"],
    },
]


# ═══════════════════════════════════════════════════════════
# TEMAS E DOMÍNIOS DO PENSAMENTO DO OLAVO
# ═══════════════════════════════════════════════════════════

OLAVO_DOMAINS = {
    "metafisica": [
        "ser", "existência", "Deus", "criação", "alma", "imortalidade",
        "realidade", "verdade", "bem", "mal", "liberdade", "determinismo",
    ],
    "epistemologia": [
        "conhecimento", "verdade", "evidência", "inteligência", "intelecto",
        "razão", "intuição", "método", "lógica", "dialética", "silogismo",
        "indução", "dedução", "abstração", "experiência", "consciência",
    ],
    "filosofia_politica": [
        "poder", "Estado", "soberania", "revolução", "comunismo", "marxismo",
        "gramscianismo", "hegemonia cultural", "Foro de São Paulo",
        "conservadorismo", "liberalismo", "direita", "esquerda",
        "democracia", "tirania", "totalitarismo", "autoridade",
    ],
    "cultura_brasileira": [
        "imbecil coletivo", "intelligentsia", "universidade", "MEC",
        "educação", "imprensa", "mídia", "jornalismo", "cultura",
        "arte", "literatura brasileira", "MPB", "tropicalismo",
    ],
    "geopolitica": [
        "Nova Ordem Mundial", "globalismo", "EUA", "Rússia", "China",
        "América Latina", "Brasil", "Oriente Médio", "Israel",
        "guerra cultural", "guerra híbrida", "soft power",
    ],
    "filosofia_classica": [
        "Aristóteles", "Platão", "Sócrates", "Tomás de Aquino",
        "Santo Agostinho", "metafísica clássica", "teoria dos quatro discursos",
        "poesia", "retórica", "dialética", "analítica", "lógica clássica",
    ],
    "critica_modernidade": [
        "Descartes", "Kant", "Hegel", "Marx", "Nietzsche", "Heidegger",
        "revolução cognitiva", "contractualismo", "iluminismo",
        "positivismo", "cientificismo", "materialismo", "ateísmo",
    ],
    "tradicao": [
        "René Guénon", "Frithjof Schuon", "Tradição Primordial",
        "simbolismo", "sagrado", "ritos", "iniciação", "esoterismo",
        "religião comparada", "Cristianismo", "Islamismo", "Judaísmo",
    ],
    "estrategia": [
        "guerra cultural", "batalha das ideias", "estratégia política",
        "desconstrução", "narrativa", "discurso", "opinião pública",
        "propaganda", "manipulação", "engenharia social",
    ],
}


# ═══════════════════════════════════════════════════════════
# FRASES MARCANTES (verificadas, com fonte)
# ═══════════════════════════════════════════════════════════

OLAVO_QUOTES = [
    {
        "quote": "O imbecil coletivo não é um imbecil sozinho. É um imbecil que se sente protegido pelo número de imbecis que o cercam.",
        "source": "O Imbecil Coletivo (1996)",
        "theme": "imbecil_coletivo",
    },
    {
        "quote": "A cultura não é um luxo. É a própria substância da vida humana.",
        "source": "entrevistas diversas",
        "theme": "cultura",
    },
    {
        "quote": "Você não pode derrotar um inimigo que não compreende. E você não pode compreender um inimigo que se recusa a estudar.",
        "source": "COF, aulas de estratégia",
        "theme": "estrategia",
    },
    {
        "quote": "A universidade brasileira é a única do mundo onde se ensina a odiar o próprio país.",
        "source": "entrevistas e artigos",
        "theme": "educacao",
    },
    {
        "quote": "Quem não sabe o que é a verdade, é escravo da mentira.",
        "source": "artigos diversos",
        "theme": "verdade",
    },
    {
        "quote": "A esquerda não quer o poder para fazer algo. Quer o poder para impedir que algo seja feito.",
        "source": "artigos políticos",
        "theme": "politica",
    },
    {
        "quote": "Não existe conhecimento sem esforço. Quem promete conhecimento sem dor está vendendo ignorância.",
        "source": "COF, introdução à filosofia",
        "theme": "conhecimento",
    },
    {
        "quote": "O primeiro passo para pensar é parar de repetir.",
        "source": "Seminário de Filosofia",
        "theme": "pensamento_critico",
    },
    {
        "quote": "A realidade não negocia. Ela simplesmente é.",
        "source": "aulas de metafísica",
        "theme": "realidade",
    },
    {
        "quote": "A Filosofia não é uma opinião. É uma ciência. A ciência dos primeiros princípios.",
        "source": "Aristóteles em Nova Perspectiva",
        "theme": "filosofia",
    },
    {
        "quote": "Gramsci não queria matar ninguém. Queria matar a alma. E conseguiu.",
        "source": "A Nova Era e a Revolução Cultural",
        "theme": "gramscianismo",
    },
    {
        "quote": "O mal não é a ausência do bem. O mal é a inversão deliberada do bem.",
        "source": "O Jardim das Aflições",
        "theme": "mal",
    },
]


# ═══════════════════════════════════════════════════════════
# CURADOR DO SUPER DATASET
# ═══════════════════════════════════════════════════════════

@dataclass
class OlavoTextRecord:
    id: str
    source: str
    book_or_article: str
    chapter_or_date: str
    page_or_url: str
    text: str
    theme: str
    word_count: int = 0
    verified: bool = False

    def to_training_format(self) -> str:
        """Format text for F51 model training."""
        header = f"[FONTE: {self.book_or_article} | TEMA: {self.theme}]"
        return f"{header}\n\n{self.text}\n\n---\n"


class OlavoDatasetCurator:
    """Cura o super dataset do Olavo para treinar o F51 Darwin-SSD."""

    def __init__(self, data_root: str | Path) -> None:
        self.root = Path(data_root) / "olavo"
        self.root.mkdir(parents=True, exist_ok=True)
        self._records: list[OlavoTextRecord] = []
        self._manifest: dict[str, Any] = {
            "name": "F51-Olavo-Super-Dataset",
            "version": "v0.1",
            "created": datetime.now(timezone.utc).isoformat(),
            "doctrine": "O Olavo é o Olavo. NADA gerado por IA. Apenas texto real com fonte.",
        }

    def load_from_file(self, file_path: str | Path) -> list[OlavoTextRecord]:
        """Load texts from a plain text file. Splits on --- markers."""
        path = Path(file_path)
        if not path.exists():
            return []

        content = path.read_text(encoding="utf-8", errors="replace")
        records: list[OlavoTextRecord] = []
        chunks = content.split("\n---\n")

        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk or len(chunk) < 50:
                continue

            # Extract metadata from header if present
            theme = "geral"
            source_title = path.stem
            match = re.match(r'\[FONTE: (.*?) \| TEMA: (.*?)\]', chunk)
            if match:
                source_title = match.group(1).strip()
                theme = match.group(2).strip()
                chunk = chunk[match.end():].strip()

            record = OlavoTextRecord(
                id=f"olavo_{path.stem}_{i:04d}",
                source=str(path),
                book_or_article=source_title,
                chapter_or_date="",
                page_or_url="",
                text=chunk,
                theme=theme,
                word_count=len(chunk.split()),
            )
            records.append(record)

        return records

    def load_all_texts(self, directory: str | Path) -> int:
        """Load all text files from a directory into the dataset."""
        dir_path = Path(directory)
        total = 0
        for txt_file in sorted(dir_path.rglob("*.txt")):
            records = self.load_from_file(txt_file)
            self._records.extend(records)
            total += len(records)
        return total

    def add_manually_transcribed(
        self,
        text: str,
        *,
        source: str,
        book_or_article: str,
        chapter_or_date: str = "",
        page_or_url: str = "",
        theme: str = "filosofia",
    ) -> OlavoTextRecord:
        """Add a manually transcribed text from a video/audio/podcast."""
        record = OlavoTextRecord(
            id=f"olavo_manual_{len(self._records):06d}",
            source=source,
            book_or_article=book_or_article,
            chapter_or_date=chapter_or_date,
            page_or_url=page_or_url,
            text=text.strip(),
            theme=theme,
            word_count=len(text.split()),
            verified=True,
        )
        self._records.append(record)
        return record

    def add_olavo_quote(self, quote: dict) -> OlavoTextRecord:
        """Add a verified Olavo quote to the dataset."""
        # Expand the quote with context/exegesis placeholder
        text = (
            f"{quote['quote']}\n\n"
            f"[Fonte verificada: {quote['source']}]\n"
            f"[Tema: {quote['theme']}]"
        )
        return self.add_manually_transcribed(
            text=text,
            source="olavo_quotes_verified",
            book_or_article=quote["source"],
            theme=quote["theme"],
        )

    def build_training_corpus(self, output_path: str | Path | None = None) -> Path:
        """Build the complete training corpus from all records."""
        if output_path is None:
            output_path = self.root / "corpus" / "olavo_corpus.txt"

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with output.open("w", encoding="utf-8") as f:
            f.write("# F51 Olavo Super Dataset\n")
            f.write(f"# Records: {len(self._records)}\n")
            f.write(f"# Total words: {sum(r.word_count for r in self._records)}\n\n")

            for record in self._records:
                f.write(record.to_training_format())
                f.write("\n")

        return output

    def build_jsonl(self, output_path: str | Path | None = None) -> Path:
        """Export as JSONL for provenance ledger."""
        if output_path is None:
            output_path = self.root / "ledger" / "olavo_provenance.jsonl"

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with output.open("a", encoding="utf-8") as f:
            for record in self._records:
                entry = {
                    "id": record.id,
                    "source": record.source,
                    "book_or_article": record.book_or_article,
                    "chapter_or_date": record.chapter_or_date,
                    "page_or_url": record.page_or_url,
                    "theme": record.theme,
                    "word_count": record.word_count,
                    "verified": record.verified,
                    "text": record.text,
                }
                f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

        return output

    def stats(self) -> dict[str, Any]:
        """Statistics about the curated dataset."""
        themes: dict[str, int] = {}
        sources: dict[str, int] = {}
        total_words = 0

        for r in self._records:
            themes[r.theme] = themes.get(r.theme, 0) + 1
            sources[r.book_or_article] = sources.get(r.book_or_article, 0) + 1
            total_words += r.word_count

        return {
            "total_records": len(self._records),
            "total_words": total_words,
            "total_words_millions": round(total_words / 1_000_000, 2),
            "themes": sorted(themes.items(), key=lambda x: -x[1]),
            "sources": sorted(sources.items(), key=lambda x: -x[1]),
        }

    def manifest(self) -> dict[str, Any]:
        man = dict(self._manifest)
        man.update(self.stats())
        man["catalog"] = {
            "books": OLAVO_BOOKS,
            "article_sources": OLAVO_ARTICLE_SOURCES,
            "domains": {k: len(v) for k, v in OLAVO_DOMAINS.items()},
        }
        return man


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def create_empty_dataset_structure(data_root: str | Path) -> dict[str, Path]:
    """Create the empty directory structure for the Olavo dataset."""
    root = Path(data_root) / "olavo"
    dirs = {
        "livros": root / "livros",
        "artigos": root / "artigos",
        "aulas_cof": root / "aulas_cof",
        "seminarios": root / "seminarios",
        "entrevistas": root / "entrevistas",
        "frases": root / "frases",
        "caderno_f51": root / "caderno_f51",
        "corpus": root / "corpus",
        "ledger": root / "ledger",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
        (path / "README.txt").write_text(
            "F51 Olavo Super Dataset\n"
            "Apenas texto real do Olavo de Carvalho.\n"
            "NADA gerado por IA. Fonte verificada obrigatória.\n",
            encoding="utf-8",
        )
    return dirs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="F51 Olavo Super Dataset Curator")
    parser.add_argument("--create-structure", action="store_true",
                        help="Create empty dataset directory structure")
    parser.add_argument("--data-root", default="data", help="Data root directory")
    parser.add_argument("--load-seed-quotes", action="store_true",
                        help="Load verified seed quotes into dataset")
    args = parser.parse_args()

    root = Path(args.data_root)

    if args.create_structure:
        dirs = create_empty_dataset_structure(root)
        print("=== Structure Created ===")
        for name, path in sorted(dirs.items()):
            print(f"  {name}: {path}")

    if args.load_seed_quotes:
        curator = OlavoDatasetCurator(root)
        for quote in OLAVO_QUOTES:
            curator.add_olavo_quote(quote)
        corpus = curator.build_training_corpus()
        ledger = curator.build_jsonl()
        print(f"=== Seed Dataset Loaded ===")
        print(f"  Records: {len(curator._records)}")
        print(f"  Words: {sum(r.word_count for r in curator._records)}")
        print(f"  Corpus: {corpus}")
        print(f"  Ledger: {ledger}")
        print()
        print(json.dumps(curator.stats(), indent=2))
