"""
F51 Math Genius — O Córtex Matemático do Organismo.

Arquitetura para descoberta de fórmulas, teoremas e curas.

Ciclo:
    1. Modelo gera HIPÓTESE matemática
    2. Wolfram verifica (computação exata)
    3. Se verdadeira → DOPAMINA + entra no corpus
    4. Se falsa → descarta, modelo aprende com erro
    5. Hipóteses boas sobrevivem, ruins morrem
    6. Evolução: novos experts nascem para domínios promissores

Inspiração:
    "A matemática é a linguagem com a qual Deus escreveu o universo." — Galileu
    "O organismo não chuta. O organismo verifica." — F51 Doctrine
"""

from __future__ import annotations

import json
import time
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from f51_darwin.wolfram_bridge import WolframBridge, WolframConfig, WolframResult


# ═══════════════════════════════════════════════════════════
# HIPÓTESE MATEMÁTICA
# ═══════════════════════════════════════════════════════════

@dataclass
class MathHypothesis:
    """Uma hipótese gerada pelo organismo e verificada pelo Wolfram."""
    id: str
    domain: str  # algebra, calculus, number_theory, physics, chemistry, medicine...
    statement: str  # a hipótese em linguagem natural
    wolfram_query: str  # query para verificar
    expected_result: str | None = None  # o que o modelo espera
    actual_result: str | None = None  # o que Wolfram retornou
    is_true: bool | None = None  # verificado?
    confidence: float = 0.0  # 0-1
    novelty_score: float = 0.0  # quão nova é essa descoberta?
    timestamp: str = ""
    generation: int = 0
    parent_hypothesis_id: str | None = None  # de qual hipótese esta derivou?

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'domain': self.domain,
            'statement': self.statement,
            'wolfram_query': self.wolfram_query,
            'expected_result': self.expected_result,
            'actual_result': self.actual_result,
            'is_true': self.is_true,
            'confidence': self.confidence,
            'novelty_score': self.novelty_score,
            'timestamp': self.timestamp,
            'generation': self.generation,
            'parent': self.parent_hypothesis_id,
        }


# ═══════════════════════════════════════════════════════════
# TEMPLATES DE HIPÓTESES — Sementes de Descoberta
# ═══════════════════════════════════════════════════════════

HYPOTHESIS_TEMPLATES = {
    'algebra': [
        ("is {a} * {b} always equal to {b} * {a} for complex numbers?", 
         "is {a}*{b} = {b}*{a} for complex numbers"),
        ("does x^{n} + y^{n} = z^{n} have integer solutions for n > 2?",
         "does x^3 + y^3 = z^3 have integer solutions"),
        ("can {n} be expressed as sum of two squares?",
         "is {n} expressible as sum of two squares"),
    ],
    'number_theory': [
        ("is there a prime number between {n} and {m}?",
         "next prime after {n}"),
        ("does the sequence of prime numbers follow any predictable pattern starting at {n}?",
         "prime numbers between {n} and {m}"),
        ("is {n} a perfect number?",
         "sigma({n}) - {n}"),
        ("are there infinitely many primes of the form {a}n + {b}?",
         "primes of the form {a}n + {b}"),
    ],
    'calculus': [
        ("does the integral of x^{n} * e^x have a closed form?",
         "integrate x^{n} * e^x for n={n}"),
        ("is the derivative of sin^{n}(x) always expressible in terms of sin and cos?",
         "derivative of sin^{n}(x) for n={n}"),
        ("does sum 1/n^{p} converge for p > 1?",
         "does sum 1/n^{p} converge for p={p}"),
    ],
    'physics': [
        ("can we predict the orbital period of a planet from its distance to the sun?",
         "orbital period of planet at distance {d} AU from sun"),
        ("what is the relationship between temperature and pressure for an ideal gas?",
         "ideal gas law P V = n R T"),
        ("is E = mc^2 always true for all forms of energy?",
         "E = m c^2 for m = 1 kg"),
    ],
    'medicine': [
        ("does the concentration of {drug} in bloodstream follow exponential decay?",
         "half life of {drug}"),
        ("what is the mathematical relationship between dose and response for {drug}?",
         "dose response curve {drug}"),
        ("can we model tumor growth using logistic differential equations?",
         "logistic growth model r={r} K={K}"),
        ("does the pH of blood buffer follow the Henderson-Hasselbalch equation?",
         "Henderson Hasselbalch equation pH = pKa + log([HCO3-]/[CO2])"),
    ],
    'chemistry': [
        ("is the reaction rate of {a} + {b} proportional to the concentration product?",
         "rate law for reaction {a} + {b}"),
        ("does the ideal gas law PV = nRT predict volume accurately at STP?",
         "ideal gas law V at STP for n=1 mol"),
        ("what is the equilibrium constant for the dissociation of water?",
         "Kw water autoionization"),
    ],
}


# ═══════════════════════════════════════════════════════════
# MATH GENIUS ENGINE
# ═══════════════════════════════════════════════════════════

@dataclass
class DiscoveryStats:
    total_hypotheses: int = 0
    verified_true: int = 0
    verified_false: int = 0
    novel_discoveries: int = 0
    domains_explored: dict[str, int] = field(default_factory=dict)
    best_discoveries: list[dict] = field(default_factory=list)
    current_generation: int = 0
    dopamine_from_math: float = 0.0

    def to_dict(self) -> dict:
        return {
            'total': self.total_hypotheses,
            'verified_true': self.verified_true,
            'verified_false': self.verified_false,
            'novel': self.novel_discoveries,
            'domains': self.domains_explored,
            'best': self.best_discoveries[-5:],
            'generation': self.current_generation,
            'dopamine': round(self.dopamine_from_math, 3),
        }


class MathGenius:
    """O córtex matemático do organismo F51.

    Gera hipóteses → Wolfram verifica → Dopamina → Evolução.

    O organismo não "chuta" respostas matemáticas.
    O organismo GERA hipóteses e o Wolfram VERIFICA.
    Só o que é verdadeiro sobrevive.
    """

    def __init__(
        self,
        wolfram_app_id: str,
        save_dir: str | Path = "workspace/runtime/organism/math_genius",
    ):
        self.bridge = WolframBridge(WolframConfig(
            app_id=wolfram_app_id,
            cache_enabled=True,
            cache_ttl_hours=720,  # math facts don't change
            log_queries=True,
        ))
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.stats = DiscoveryStats()
        self.history: list[MathHypothesis] = []
        self._rng = random.Random(51)
        self._load()

    def _load(self):
        stats_file = self.save_dir / "math_stats.json"
        if stats_file.exists():
            data = json.loads(stats_file.read_text())
            for k, v in data.items():
                if hasattr(self.stats, k):
                    setattr(self.stats, k, v)

        history_file = self.save_dir / "math_history.jsonl"
        if history_file.exists():
            for line in history_file.read_text().splitlines():
                if line.strip():
                    data = json.loads(line)
                    if 'parent' in data:
                        data['parent_hypothesis_id'] = data.pop('parent')
                    self.history.append(MathHypothesis(**data))

    def save(self):
        (self.save_dir / "math_stats.json").write_text(
            json.dumps(self.stats.to_dict(), indent=2) + "\n"
        )
        with (self.save_dir / "math_history.jsonl").open("a") as f:
            if self.history:
                f.write(json.dumps(self.history[-1].to_dict()) + "\n")

    def generate_hypothesis(
        self,
        domain: str | None = None,
        *,
        parent: MathHypothesis | None = None,
    ) -> MathHypothesis:
        """Gera uma hipótese matemática para verificação.

        Pode ser aleatória (exploração) ou derivada de uma hipótese-mãe (evolução).
        """
        if domain is None:
            domain = self._rng.choice(list(HYPOTHESIS_TEMPLATES.keys()))

        templates = HYPOTHESIS_TEMPLATES.get(domain, HYPOTHESIS_TEMPLATES['algebra'])
        template_statement, template_query = self._rng.choice(templates)

        # Fill template with random values
        n = self._rng.randint(2, 100)
        m = self._rng.randint(n + 1, n + 50)
        a = self._rng.randint(2, 20)
        b = self._rng.randint(1, 10)
        p = self._rng.randint(2, 5)
        d = round(self._rng.uniform(0.5, 30), 1)
        r = round(self._rng.uniform(0.1, 2.0), 2)
        drug = self._rng.choice(['aspirin', 'ibuprofen', 'metformin', 'amoxicillin',
                                  'lisinopril', 'atorvastatin', 'omeprazole'])

        statement = template_statement.format(n=n, m=m, a=a, b=b, p=p, d=d, r=r, K=m, drug=drug)
        query = template_query.format(n=n, m=m, a=a, b=b, p=p, d=d, r=r, K=m, drug=drug)

        self.stats.current_generation += 1

        hypothesis = MathHypothesis(
            id=f"math_{self.stats.total_hypotheses:06d}",
            domain=domain,
            statement=statement,
            wolfram_query=query,
            timestamp=datetime.now(timezone.utc).isoformat(),
            generation=self.stats.current_generation,
            parent_hypothesis_id=parent.id if parent else None,
        )

        return hypothesis

    def verify_hypothesis(self, hypothesis: MathHypothesis) -> MathHypothesis:
        """Verifica uma hipótese usando Wolfram Alpha."""
        result = self.bridge.query(hypothesis.wolfram_query)

        hypothesis.actual_result = result.result_text if result.success else "VERIFICATION_FAILED"
        hypothesis.is_true = result.success and len(result.result_text) > 0
        hypothesis.confidence = 1.0 if result.success else 0.0

        # Calculate novelty: how different is this from known results?
        hypothesis.novelty_score = self._calculate_novelty(hypothesis)

        self.stats.total_hypotheses += 1
        if hypothesis.is_true:
            self.stats.verified_true += 1
            self.stats.dopamine_from_math = min(1.0, self.stats.dopamine_from_math + 0.05)
            if hypothesis.novelty_score > 0.5:
                self.stats.novel_discoveries += 1
                self.stats.best_discoveries.append(hypothesis.to_dict())
        else:
            self.stats.verified_false += 1
            self.stats.dopamine_from_math = max(0.0, self.stats.dopamine_from_math - 0.01)

        self.stats.domains_explored[hypothesis.domain] = \
            self.stats.domains_explored.get(hypothesis.domain, 0) + 1

        self.history.append(hypothesis)
        self.save()

        return hypothesis

    def _calculate_novelty(self, hypothesis: MathHypothesis) -> float:
        """Estima quão nova é esta descoberta."""
        if not hypothesis.is_true:
            return 0.0

        # Check if this result is already in our verified corpus
        for h in self.history:
            if h.is_true and h.actual_result == hypothesis.actual_result and h.id != hypothesis.id:
                return 0.1  # already known to us

        # Check if the query produced detailed results (more novel)
        if hypothesis.actual_result and len(hypothesis.actual_result) > 100:
            return 0.7

        return 0.5

    def evolve_hypothesis(self, parent: MathHypothesis) -> MathHypothesis | None:
        """Evolui uma hipótese verdadeira — gera uma derivada mais complexa."""
        if not parent.is_true:
            return None

        # Generate a child hypothesis in the same domain
        child = self.generate_hypothesis(parent.domain, parent=parent)
        return child

    def research_cycle(self, cycles: int = 5) -> list[MathHypothesis]:
        """Executa um ciclo de pesquisa: gerar → verificar → evoluir.

        Returns list of verified true hypotheses from this cycle.
        """
        verified: list[MathHypothesis] = []

        for i in range(cycles):
            # 70%: explore new domain, 30%: evolve from existing truth
            if self.history and self._rng.random() < 0.3:
                true_ones = [h for h in self.history if h.is_true]
                if true_ones:
                    parent = self._rng.choice(true_ones)
                    hypothesis = self.evolve_hypothesis(parent)
                    if hypothesis is None:
                        hypothesis = self.generate_hypothesis()
                else:
                    hypothesis = self.generate_hypothesis()
            else:
                hypothesis = self.generate_hypothesis()

            verified_h = self.verify_hypothesis(hypothesis)
            if verified_h.is_true:
                verified.append(verified_h)

            time.sleep(0.3)  # rate limit

        return verified

    def to_training_corpus(self) -> str:
        """Converte todas as hipóteses verdadeiras em texto de treino."""
        docs: list[str] = []
        for h in self.history:
            if h.is_true:
                doc = (
                    f"[MATH DISCOVERY] [DOMAIN: {h.domain}] [GEN: {h.generation}]\n"
                    f"Hypothesis: {h.statement}\n"
                    f"Verification: {h.actual_result}\n"
                    f"Confidence: {h.confidence}\n"
                    f"Novelty: {h.novelty_score}\n"
                    f"---\n"
                )
                docs.append(doc)

        return "\n".join(docs)

    def status_report(self) -> str:
        """Relatório do gênio matemático."""
        s = self.stats
        return (
            f"🧮 MATH GENIUS STATUS\n"
            f"{'='*50}\n"
            f"  Hipóteses geradas:   {s.total_hypotheses}\n"
            f"  ✅ Verdadeiras:       {s.verified_true}\n"
            f"  ❌ Falsas:            {s.verified_false}\n"
            f"  🆕 Descobertas novas: {s.novel_discoveries}\n"
            f"  🧬 Geração atual:     {s.current_generation}\n"
            f"  🧠 Dopamina matemática: {s.dopamine_from_math:.2f}\n"
            f"  📊 Domínios: {json.dumps(s.domains_explored)}\n"
            f"{'='*50}\n"
        )


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="F51 Math Genius — Descoberta Matemática")
    parser.add_argument("--cycles", type=int, default=10, help="Research cycles")
    parser.add_argument("--app-id", default="W4E3KY-LEV3Q82QP9")
    parser.add_argument("--save-corpus", action="store_true")
    args = parser.parse_args()

    genius = MathGenius(args.app_id)

    print("=" * 50)
    print("  F51 MATH GENIUS — Research Mode")
    print("=" * 50)
    print(f"  Cycles: {args.cycles}")
    print(f"  Wolfram: {'conectado' if genius.bridge.config.app_id else 'desconectado'}")
    print()

    for cycle in range(args.cycles):
        print(f"--- Cycle {cycle + 1}/{args.cycles} ---")
        verified = genius.research_cycle(cycles=1)
        for h in verified:
            icon = "🆕" if h.novelty_score > 0.5 else "✅"
            print(f"  {icon} [{h.domain}] {h.statement[:80]}...")
            print(f"     → {h.actual_result[:100]}")
        print()

    print(genius.status_report())

    if args.save_corpus:
        corpus = genius.to_training_corpus()
        path = Path("data/generated/candidates/math_discoveries.txt")
        path.write_text(corpus, encoding="utf-8")
        print(f"Corpus salvo: {path} ({len(corpus)} chars)")
