from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse


class CorpusDecision(str, Enum):
    APPROVE = "approve"
    QUARANTINE = "quarantine"
    REJECT = "reject"


SCIENCE_TERMS: dict[str, tuple[str, ...]] = {
    "mathematics": (
        "theorem",
        "proof",
        "lemma",
        "calculus",
        "algebra",
        "geometry",
        "topology",
        "statistics",
        "probability",
        "differential equation",
    ),
    "physics": (
        "quantum",
        "mechanics",
        "thermodynamics",
        "relativity",
        "field equation",
        "particle",
        "energy conservation",
    ),
    "chemistry": (
        "molecule",
        "reaction",
        "stoichiometry",
        "catalyst",
        "spectroscopy",
        "organic chemistry",
    ),
    "biology": (
        "cell",
        "protein",
        "genome",
        "evolution",
        "enzyme",
        "metabolism",
        "ecology",
    ),
    "medicine": (
        "clinical",
        "diagnosis",
        "epidemiology",
        "pathophysiology",
        "randomized trial",
        "biomarker",
    ),
    "computer_science": (
        "algorithm",
        "complexity",
        "compiler",
        "operating system",
        "database",
        "neural network",
    ),
    "engineering": (
        "control system",
        "circuit",
        "signal processing",
        "materials",
        "finite element",
    ),
}


RIGHT_THOUGHT_TERMS: dict[str, tuple[str, ...]] = {
    "classical_liberalism": (
        "natural rights",
        "limited government",
        "rule of law",
        "private property",
        "individual liberty",
    ),
    "conservatism": (
        "tradition",
        "ordered liberty",
        "civil society",
        "natural law",
        "prudence",
        "subsidiarity",
    ),
    "austrian_economics": (
        "mises",
        "hayek",
        "economic calculation",
        "spontaneous order",
        "malinvestment",
    ),
    "anti_totalitarian": (
        "communism",
        "totalitarian",
        "gulag",
        "central planning",
        "bureaucracy",
    ),
}


ACTIVISM_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(join|support|donate to|sign|march with|organize for)\b.{0,80}\b(movement|campaign|resistance|coalition)\b",
        r"\b(activists?|organizers?)\b.{0,80}\b(demand|call for|mobilize|pressure)\b",
        r"\b(we must|you must|everyone must)\b.{0,80}\b(abolish|dismantle|decolonize|defund)\b",
        r"\b(oppressor|oppressed|privilege|intersectional|liberation struggle)\b.{0,80}\b(action|praxis|movement)\b",
        r"\b(critical race theory|queer theory|settler colonialism|white fragility)\b",
    )
)


PIRATE_OR_LOW_PROVENANCE_HOSTS = {
    "pdfcoffee.com",
    "z-lib.io",
    "zlibrary",
    "libgen",
    "sci-hub",
    "scribd.com",
    "academia.edu",
    "researchgate.net",
}


COMMERCIAL_OK_LICENSE_HINTS = {
    "public domain",
    "pd",
    "pd-us",
    "cc0",
    "cc-by",
    "cc by",
    "cc-by-sa",
    "cc by-sa",
    "mit",
    "apache",
    "bsd",
    "us government",
    "government work",
}


NONCOMMERCIAL_OR_VERIFY_HINTS = {
    "cc-by-nc",
    "cc by-nc",
    "non-commercial",
    "noncommercial",
    "permission",
    "verify",
    "unknown",
    "copyright",
}


@dataclass(frozen=True)
class CorpusPolicyResult:
    decision: CorpusDecision
    reason: str
    tags: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


def _contains_any(text: str, terms: tuple[str, ...]) -> int:
    lowered = text.lower()
    return sum(1 for term in terms if term in lowered)


def _host(source_url: str) -> str:
    parsed = urlparse(source_url)
    return (parsed.netloc or parsed.path).lower()


def classify_domains(text: str) -> list[str]:
    matches: list[tuple[str, int]] = []
    for domain, terms in SCIENCE_TERMS.items():
        score = _contains_any(text, terms)
        if score:
            matches.append((domain, score))
    matches.sort(key=lambda item: (-item[1], item[0]))
    return [domain for domain, _ in matches]


def classify_right_thought(text: str) -> list[str]:
    matches: list[tuple[str, int]] = []
    for tag, terms in RIGHT_THOUGHT_TERMS.items():
        score = _contains_any(text, terms)
        if score:
            matches.append((tag, score))
    matches.sort(key=lambda item: (-item[1], item[0]))
    return [tag for tag, _ in matches]


def activism_score(text: str) -> float:
    sample = text[:100_000]
    hits = sum(len(pattern.findall(sample)) for pattern in ACTIVISM_PATTERNS)
    return round(min(1.0, hits / 4.0), 4)


def license_score(license_label: str) -> float:
    lowered = license_label.strip().lower()
    if not lowered:
        return 0.0
    if any(hint in lowered for hint in NONCOMMERCIAL_OR_VERIFY_HINTS):
        return 0.35
    if any(hint in lowered for hint in COMMERCIAL_OK_LICENSE_HINTS):
        return 1.0
    return 0.5


def evaluate_corpus_source(
    *,
    text: str,
    source_url: str,
    license_label: str,
    domain_hint: str | None = None,
) -> CorpusPolicyResult:
    stripped = text.strip()
    scores = {
        "length": round(min(1.0, len(stripped) / 2_000), 4),
        "license": license_score(license_label),
        "activism": activism_score(stripped),
    }

    if not stripped:
        return CorpusPolicyResult(CorpusDecision.REJECT, "empty text", scores=scores)
    if len(stripped) < 200:
        return CorpusPolicyResult(CorpusDecision.REJECT, "too short for corpus item", scores=scores)

    host = _host(source_url)
    if any(blocked in host for blocked in PIRATE_OR_LOW_PROVENANCE_HOSTS):
        return CorpusPolicyResult(
            CorpusDecision.REJECT,
            f"low-provenance or pirate-prone host: {host}",
            scores=scores,
        )

    science_tags = classify_domains(stripped)
    right_tags = classify_right_thought(stripped)
    tags = [f"science:{tag}" for tag in science_tags]
    tags.extend(f"right_thought:{tag}" for tag in right_tags)
    if domain_hint:
        tags.append(f"hint:{domain_hint}")

    if scores["license"] < 0.5:
        return CorpusPolicyResult(
            CorpusDecision.QUARANTINE,
            "license requires manual verification before training use",
            tags=tags,
            scores=scores,
        )

    if scores["activism"] >= 0.5 and not science_tags:
        return CorpusPolicyResult(
            CorpusDecision.QUARANTINE,
            "high activism language without scientific domain signal",
            tags=tags,
            scores=scores,
        )

    if not science_tags and not right_tags:
        return CorpusPolicyResult(
            CorpusDecision.QUARANTINE,
            "no science, mathematics, or approved right-thought signal",
            tags=tags,
            scores=scores,
        )

    if scores["activism"] >= 0.75:
        return CorpusPolicyResult(
            CorpusDecision.QUARANTINE,
            "activism language is high enough to require review",
            tags=tags,
            scores=scores,
        )

    return CorpusPolicyResult(
        CorpusDecision.APPROVE,
        "source passed corpus policy",
        tags=tags,
        scores=scores,
    )
