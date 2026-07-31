from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationReport:
    documents: list[str]
    word_count: int
    target_words: int


SUBJECTS = [
    "modelo",
    "sistema",
    "processo",
    "experimento",
    "observador",
    "algoritmo",
    "memória",
    "contexto",
    "hipótese",
    "evidência",
    "cidade",
    "rio",
    "floresta",
    "laboratório",
    "equipe",
    "sensor",
    "rede",
    "documento",
    "corpus",
    "tokenizer",
]

VERBS = [
    "observa",
    "valida",
    "registra",
    "compara",
    "ajusta",
    "preserva",
    "analisa",
    "sintetiza",
    "documenta",
    "refina",
    "protege",
    "mede",
    "organiza",
    "integra",
    "explica",
]

ADJECTIVES = [
    "claro",
    "estável",
    "cuidadoso",
    "preciso",
    "gradual",
    "robusto",
    "local",
    "consistente",
    "auditável",
    "transparente",
    "útil",
    "coerente",
    "modular",
    "seguro",
    "verificável",
]

OBJECTS = [
    "padrões linguísticos",
    "sequências longas",
    "erros recorrentes",
    "sinais de qualidade",
    "limites de memória",
    "dados aprovados",
    "candidatos sintéticos",
    "resultados de avaliação",
    "métricas de retenção",
    "artefatos versionados",
    "estruturas causais",
    "regras de firewall",
    "proveniência completa",
    "checkpoints imutáveis",
    "mixturas de treino",
]

CONTEXTS = [
    "o corpus cresce de forma controlada",
    "a auditoria detecta duplicação cedo",
    "o replay protege capacidade antiga",
    "a quarentena isola risco recursivo",
    "a promoção exige aprovação explícita",
    "o treino respeita limites de VRAM",
    "a evolução só ocorre por incapacidade demonstrada",
    "o tokenizer permanece proprietário da F51",
    "cada ciclo registra scores e razões",
    "nenhum checkpoint externo entra no fluxo",
]

OPENERS = [
    "Em português claro,",
    "Na prática operacional,",
    "Durante a revisão diária,",
    "Quando o sistema amadurece,",
    "Sob restrições de segurança,",
    "Com foco em reproducibilidade,",
    "Em um cenário de laboratório,",
    "Ao consolidar conhecimento,",
]

CLOSERS = [
    "Esse comportamento precisa continuar auditável.",
    "A decisão fica registrada no ledger.",
    "O próximo passo depende de evidência local.",
    "Nada entra no treino sem aprovação.",
    "A retenção ancestral continua obrigatória.",
    "O ciclo só avança com métricas estáveis.",
    "A promoção permanece manual e explícita.",
    "O firewall mantém o corpus limpo.",
]


def _sentence(rng: random.Random) -> str:
    pattern = rng.randint(0, 3)
    if pattern == 0:
        return (
            f"{rng.choice(OPENERS)} o {rng.choice(SUBJECTS)} {rng.choice(VERBS)} "
            f"{rng.choice(OBJECTS)} de forma {rng.choice(ADJECTIVES)}."
        )
    if pattern == 1:
        return (
            f"Quando {rng.choice(CONTEXTS)}, o {rng.choice(SUBJECTS)} "
            f"{rng.choice(VERBS)} {rng.choice(OBJECTS)} com rigor {rng.choice(ADJECTIVES)}."
        )
    if pattern == 2:
        return (
            f"O {rng.choice(SUBJECTS)} {rng.choice(ADJECTIVES)} {rng.choice(VERBS)} "
            f"{rng.choice(OBJECTS)}, porque {rng.choice(CONTEXTS)}."
        )
    return f"{rng.choice(CLOSERS)}"


def _paragraph(rng: random.Random, min_sentences: int = 4, max_sentences: int = 7) -> str:
    count = rng.randint(min_sentences, max_sentences)
    sentences = [_sentence(rng) for _ in range(count)]
    return " ".join(sentences)


def _word_count(text: str) -> int:
    return len(text.split())


def generate_portuguese_documents(
    *,
    target_words: int = 20_000,
    chunk_words: int = 500,
    seed: int = 51,
) -> GenerationReport:
    if target_words <= 0:
        raise ValueError("target_words must be positive.")
    if chunk_words < 120:
        raise ValueError("chunk_words is too small.")

    rng = random.Random(seed)
    documents: list[str] = []
    total_words = 0

    topics = [
        "linguagem e modelos causais",
        "segurança de dados sintéticos",
        "evolução modular e poda",
        "memória replay e retenção",
        "tokenizer e corpus limpo",
        "treino base from scratch",
        "observabilidade e métricas",
        "governança de proveniência",
    ]

    while total_words < target_words:
        header = f"Nota sintética sobre {rng.choice(topics)}."
        body_parts = [_paragraph(rng) for _ in range(rng.randint(3, 5))]
        document = header + " " + " ".join(body_parts)
        words = _word_count(document)

        if total_words + words > target_words + chunk_words:
            remaining = target_words - total_words
            if remaining < 120:
                break
            trimmed = " ".join(document.split()[:remaining])
            documents.append(trimmed)
            total_words += _word_count(trimmed)
            break

        documents.append(document)
        total_words += words

    return GenerationReport(documents=documents, word_count=total_words, target_words=target_words)
