"""
F51 Decision Engine — Motor de decisão com incerteza calibrada.

Extraído e adaptado do f51-verdict (marcobarreto007).
Integra no F51 Darwin-SSD como o córtex de tomada de decisão.

Fórmula unificada:
    D = σ(w1·V + w2·C + w3·P + w4·R - w5·U - w6·S)

Onde:
    V = Veracity (claims batem com dados?)
    C = Consensus (múltiplas fontes concordam?)
    P = Proxy reward (histórico similar deu certo?)
    R = Robustness (passou em verificação adversarial?)
    U = Uncertainty (dificuldade do problema)
    S = Danger / Spider Sense (armadilhas detectadas)

Uso:
    from f51_darwin.decision_engine import decide, DecisionFactors
    confidence = decide(factors)
    if confidence > threshold:
        take_action()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


# ═══════════════════════════════════════════════════════════
# FUNÇÕES MATEMÁTICAS BASE
# ═══════════════════════════════════════════════════════════

def sigmoid(x: float) -> float:
    """Logística numericamente estável. σ(x) ∈ (0, 1)."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def t_norm(a: float, b: float) -> float:
    """T-norm produto: P(A e B) assumindo independência."""
    return a * b


def t_conorm(a: float, b: float) -> float:
    """T-conorm: P(A ou B) = 1 - (1-a)(1-b)."""
    return 1.0 - (1.0 - a) * (1.0 - b)


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp com limites customizáveis."""
    return max(lo, min(hi, x))


def combined_risk(*risks: float) -> float:
    """Combina riscos via t-conorm: P(pelo menos um ocorre)."""
    if not risks:
        return 0.0
    result = 0.0
    for r in risks:
        result = t_conorm(result, clamp(r))
    return result


# ═══════════════════════════════════════════════════════════
# FÓRMULA UNIFICADA DE DECISÃO
# ═══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DecisionWeights:
    """Pesos calibráveis da fórmula de decisão."""
    veracity: float = 2.0
    consensus: float = 1.5
    proxy: float = 1.0
    robustness: float = 0.5
    uncertainty: float = 2.0
    danger: float = 1.0

    @staticmethod
    def default() -> "DecisionWeights":
        return DecisionWeights()

    @staticmethod
    def strict() -> "DecisionWeights":
        """Pesos conservadores — exige mais evidência."""
        return DecisionWeights(
            veracity=3.0, consensus=2.0, proxy=0.5,
            robustness=1.0, uncertainty=3.0, danger=2.0,
        )

    @staticmethod
    def exploratory() -> "DecisionWeights":
        """Pesos permissivos — para exploração criativa."""
        return DecisionWeights(
            veracity=1.0, consensus=0.5, proxy=2.0,
            robustness=0.3, uncertainty=1.0, danger=0.5,
        )


@dataclass
class DecisionFactors:
    """Fatores que alimentam a fórmula."""
    veracity: float = 0.0
    consensus: float = 0.0
    proxy_reward: float = 0.0
    robustness: float = 0.0
    uncertainty: float = 0.0
    danger: float = 0.0
    reason: str = ""

    def validate(self):
        """Garante que todos os fatores estão em [0, 1]."""
        for field_name in ['veracity', 'consensus', 'proxy_reward',
                           'robustness', 'uncertainty', 'danger']:
            val = getattr(self, field_name)
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{field_name}={val} fora de [0,1]")


def decide(
    factors: DecisionFactors,
    weights: DecisionWeights | None = None,
) -> float:
    """Confiança em uma decisão. Retorna P(sucesso) ∈ [0, 1]."""
    w = weights or DecisionWeights.default()
    factors.validate()

    raw = (
        + w.veracity * factors.veracity
        + w.consensus * factors.consensus
        + w.proxy * factors.proxy_reward
        + w.robustness * factors.robustness
        - w.uncertainty * factors.uncertainty
        - w.danger * factors.danger
    )
    return sigmoid(raw)


# ═══════════════════════════════════════════════════════════
# PROXY REWARD (sem ground truth)
# ═══════════════════════════════════════════════════════════

def proxy_reward(
    consensus: float,
    critic: float,
    sanity: float,
    diversity_penalty: float = 0.0,
    ground_truth_bonus: float = 0.0,
    has_ground_truth: bool = False,
) -> float:
    """Recompensa calculada sem gabarito externo.

    F51 pode usar isso para avaliar a PRÓPRIA output:
    - Consensus: múltiplas abordagens concordam?
    - Critic: uma segunda passagem crítica encontrou problemas?
    - Sanity: a resposta passa em checagens básicas?
    """
    total = (
        0.3 * clamp(consensus)
        + 0.4 * clamp(critic)
        + 0.2 * clamp(sanity)
        - 0.1 * clamp(diversity_penalty)
    )
    if has_ground_truth:
        total += 0.2 * clamp(ground_truth_bonus)
        cap = 1.0
    else:
        cap = 0.9
    return clamp(total, 0.0, cap)


# ═══════════════════════════════════════════════════════════
# HEURISTIC UNCERTAINTY
# ═══════════════════════════════════════════════════════════

def heuristic_uncertainty(
    text: str = "",
    numeric_features: dict[str, float] | None = None,
) -> float:
    """Estima incerteza baseada em features do input.

    F51 usa isso para saber QUANDO está em território desconhecido.
    Alta incerteza → pede ajuda (Wolfram, web search, clarificação).
    """
    u = 0.15  # baseline
    t = text.lower()

    if len(text) > 800:
        u += 0.15
    if len(text) > 1600:
        u += 0.15
    for kw in ["prove", "show that", "for all", "demonstrate"]:
        if kw in t:
            u += 0.10
            break
    for kw in ["derivative", "integral", "limit", "theorem"]:
        if kw in t:
            u += 0.08
            break
    for kw in ["code", "debug", "error", "exception", "traceback"]:
        if kw in t:
            u += 0.05
            break

    if numeric_features:
        missing = numeric_features.get("missing_data_ratio", 0)
        if missing > 0.3:
            u += 0.12
        vol = numeric_features.get("volatility", 0)
        if vol > 0.5:
            u += 0.10
        n = numeric_features.get("data_points", 100)
        if n < 20:
            u += 0.08

    return clamp(u, 0.05, 0.95)


# ═══════════════════════════════════════════════════════════
# THRESHOLDS POR DOMÍNIO
# ═══════════════════════════════════════════════════════════

DOMAIN_THRESHOLDS = {
    "math": 0.85,
    "code_generation": 0.80,
    "code_modification": 0.90,
    "identity": 0.60,
    "finance": 0.75,
    "medical": 0.90,
    "casual_chat": 0.40,
    "default": 0.70,
}


def decision_threshold(domain: str = "default") -> float:
    """Threshold mínimo para tomar ação no domínio."""
    return DOMAIN_THRESHOLDS.get(domain, DOMAIN_THRESHOLDS["default"])


def should_act(confidence: float, domain: str = "default") -> bool:
    """O F51 deve agir com esta confiança?"""
    return confidence >= decision_threshold(domain)


def action_level(confidence: float, domain: str = "default") -> str:
    """Nível de ação recomendado."""
    t = decision_threshold(domain)
    if confidence >= t + 0.10:
        return "act"  # confiante — age sozinho
    elif confidence >= t:
        return "suggest"  # sugere mas pede confirmação
    elif confidence >= t - 0.15:
        return "ask"  # pede ajuda (Wolfram, clarification)
    else:
        return "defer"  # não sabe — admite ignorância


# ═══════════════════════════════════════════════════════════
# FÁBRICA DE FATORES — Integração com F51
# ═══════════════════════════════════════════════════════════

def factors_from_generation(
    text: str,
    prompt: str,
    wolfram_verified: bool = False,
    critic_score: float = 0.5,
    source_count: int = 1,
) -> DecisionFactors:
    """Constrói DecisionFactors a partir do output de geração do F51.

    Args:
        text: texto gerado pelo modelo
        prompt: prompt original do usuário
        wolfram_verified: foi verificado pelo Wolfram?
        critic_score: score de autocrítica (0-1)
        source_count: quantas fontes concordam?
    """
    uncertainty = heuristic_uncertainty(prompt)

    # Veracity: Wolfram verification é ouro
    veracity = 0.9 if wolfram_verified else min(0.5, source_count * 0.2)

    # Consensus: múltiplas fontes
    consensus = min(1.0, source_count * 0.25)

    # Proxy: critic score como aproximação
    proxy = critic_score

    # Robustness: comprimento da resposta como proxy fraco
    robustness = min(1.0, len(text) / 500 * 0.3)

    # Danger: detecta padrões perigosos
    danger = _detect_danger_patterns(text)

    return DecisionFactors(
        veracity=veracity,
        consensus=consensus,
        proxy_reward=proxy,
        robustness=robustness,
        uncertainty=uncertainty,
        danger=danger,
        reason=f"generated_from_prompt(len={len(prompt)})",
    )


def _detect_danger_patterns(text: str) -> float:
    """Spider Sense — detecta padrões perigosos no texto."""
    danger = 0.0
    t = text.lower()

    # Medical advice without disclaimer
    medical_terms = ["diagnosed", "prescription", "take this medication",
                     "cure for", "treatment for", "you have"]
    if any(term in t for term in medical_terms):
        if "not medical advice" not in t and "não é aconselhamento" not in t:
            danger += 0.3

    # Financial advice without disclaimer
    financial_terms = ["buy now", "sell now", "guaranteed return",
                       "risk free", "sure bet"]
    if any(term in t for term in financial_terms):
        if "not financial advice" not in t and "não é recomendação" not in t:
            danger += 0.3

    # Legal claims
    legal_terms = ["you should sue", "file a lawsuit", "legal advice"]
    if any(term in t for term in legal_terms):
        danger += 0.5

    # Code that looks destructive
    destructive = ["rm -rf", "format c:", "drop table", "delete all",
                   "os.remove", "shutil.rmtree"]
    if any(term in t for term in destructive):
        danger += 0.4

    return clamp(danger)


# ═══════════════════════════════════════════════════════════
# TESTES (executar com pytest)
# ═══════════════════════════════════════════════════════════

def test_sigmoid_symmetric():
    assert abs(sigmoid(0) - 0.5) < 1e-6
    assert sigmoid(10) > 0.999
    assert sigmoid(-10) < 0.001

def test_clamp():
    assert clamp(0.5) == 0.5
    assert clamp(1.5) == 1.0
    assert clamp(-0.5) == 0.0

def test_combined_risk():
    assert combined_risk(0.5, 0.5) > 0.5  # t-conorm > max
    assert combined_risk() == 0.0

def test_decide_perfect():
    f = DecisionFactors(veracity=1.0, consensus=1.0, proxy_reward=1.0,
                        robustness=1.0, uncertainty=0.0, danger=0.0)
    assert decide(f) > 0.95

def test_decide_terrible():
    f = DecisionFactors(veracity=0.0, consensus=0.0, proxy_reward=0.0,
                        robustness=0.0, uncertainty=1.0, danger=1.0)
    assert decide(f) < 0.1

def test_should_act():
    assert should_act(0.9, "math")
    assert not should_act(0.5, "math")
    assert should_act(0.5, "casual_chat")

def test_danger_detection():
    assert _detect_danger_patterns("buy now guaranteed return") > 0.2
    assert _detect_danger_patterns("hello world") == 0.0
    assert _detect_danger_patterns("rm -rf /") > 0.3

def test_heuristic_uncertainty():
    u1 = heuristic_uncertainty("hello")
    u2 = heuristic_uncertainty("prove that for all x, the derivative of the integral...")
    assert u2 > u1

def test_action_level():
    assert action_level(0.95, "math") == "act"
    assert action_level(0.80, "math") == "suggest"
    assert action_level(0.72, "math") == "ask"
    assert action_level(0.50, "math") == "defer"
