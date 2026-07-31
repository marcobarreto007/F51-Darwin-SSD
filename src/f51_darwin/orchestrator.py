"""
F51 Orchestrator — Multi-source decision pipeline.

Adaptado do Canada-Tax orchestrator.py.
Decide qual ferramenta usar para cada query: Wolfram? Web? Código? Memória?

Padrão:
    1. Intent Policy → classifica intenção
    2. Tool Decision → escolhe ferramenta
    3. Execute → chama Wolfram, RAG, ou geração direta
    4. Quality Gate → valida output
    5. Deliver → entrega ou fallback

Uso:
    from f51_darwin.orchestrator import Orchestrator, ToolDecision
    orch = Orchestrator()
    decision = orch.decide(prompt)
    result = orch.execute(decision)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable


# ═══════════════════════════════════════════════════════════
# TOOL DECISION
# ═══════════════════════════════════════════════════════════

@dataclass
class ToolDecision:
    """Decisão de quais ferramentas usar para uma query."""
    requires_wolfram: bool = False
    requires_web_search: bool = False
    requires_rag: bool = False
    requires_code_exec: bool = False
    requires_self_reflection: bool = False
    reply_mode: str = "direct"  # direct, ask_clarification, tool_first
    domain: str = "default"
    reasons: list[str] = field(default_factory=list)
    missing_inputs: list[str] = field(default_factory=list)
    wolfram_query: str = ""
    clarification_message: str = ""


# ═══════════════════════════════════════════════════════════
# INTENT PATTERNS
# ═══════════════════════════════════════════════════════════

INTENT_PATTERNS = {
    "math": {
        "keywords": [
            "calcule", "calculate", "solve", "derivative", "integral",
            "equation", "limit", "factor", "expand", "simplify",
            "derivada", "integral", "equação", "resolva", "fator",
            "sqrt", "log", "sin", "cos", "tan", "matrix",
            "prime", "fibonacci", "gcd", "lcm", "factorial",
        ],
        "tools": ["wolfram"],
        "domain": "math",
    },
    "code": {
        "keywords": [
            "código", "code", "script", "function", "debug",
            "python", "javascript", "html", "css", "sql",
            "program", "compile", "error", "exception",
            "import", "class", "def", "async", "await",
        ],
        "tools": ["code_exec", "self_reflection"],
        "domain": "code_generation",
    },
    "finance": {
        "keywords": [
            "stock", "ação", "invest", "trade", "market",
            "price", "buy", "sell", "portfolio", "dividend",
            "small cap", "nano cap", "revenue", "profit",
        ],
        "tools": ["web_search"],
        "domain": "finance",
    },
    "medical": {
        "keywords": [
            "symptom", "disease", "treatment", "diagnosis",
            "medicine", "drug", "dosage", "prescription",
            "sintoma", "doença", "tratamento", "diagnóstico",
            "remédio", "medicamento", "dosagem",
        ],
        "tools": ["web_search"],
        "domain": "medical",
    },
    "identity": {
        "keywords": [
            "quem é você", "quem és", "who are you",
            "f51", "darwin", "organismo", "marco barreto",
            "sua missão", "seu propósito", "your purpose",
        ],
        "tools": ["self_reflection"],
        "domain": "identity",
    },
    "clarification": {
        "keywords": [],
        "tools": [],
        "domain": "default",
    },
}


# ═══════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════

class Orchestrator:
    """Pipeline de decisão multi-source para o F51.

    Decide qual ferramenta usar baseado na intenção do prompt.
    Ordem de prioridade:
        1. Wolfram (math queries — verificação exata)
        2. Self-reflection (identity queries — o modelo sabe quem é)
        3. Web search (finance/medical — precisa de dados atualizados)
        4. Direct generation (casual chat, general knowledge)
    """

    def __init__(
        self,
        wolfram_callback: Callable | None = None,
        web_search_callback: Callable | None = None,
        code_exec_callback: Callable | None = None,
        enable_wolfram: bool = True,
        enable_web_search: bool = False,
        enable_code_exec: bool = False,
    ):
        self.wolfram_callback = wolfram_callback
        self.web_search_callback = web_search_callback
        self.code_exec_callback = code_exec_callback
        self.enable_wolfram = enable_wolfram
        self.enable_web_search = enable_web_search
        self.enable_code_exec = enable_code_exec

    def decide(self, prompt: str) -> ToolDecision:
        """Analisa o prompt e decide quais ferramentas usar."""
        decision = ToolDecision()
        prompt_lower = prompt.lower().strip()

        # Detecta intenção
        for intent_name, config in INTENT_PATTERNS.items():
            if intent_name == "clarification":
                continue
            keywords = config["keywords"]
            if any(kw in prompt_lower for kw in keywords):
                decision.domain = config["domain"]
                for tool in config["tools"]:
                    if tool == "wolfram":
                        decision.requires_wolfram = True
                        decision.wolfram_query = prompt
                        decision.reasons.append(f"math intent detected: wolfram verification needed")
                    elif tool == "web_search":
                        decision.requires_web_search = True
                        decision.reasons.append(f"{intent_name} intent: web search for current data")
                    elif tool == "code_exec":
                        decision.requires_code_exec = True
                        decision.reasons.append("code intent: execution needed")
                    elif tool == "self_reflection":
                        decision.requires_self_reflection = True
                        decision.reasons.append("identity intent: self-reflection")
                break

        # Fallback: prompt muito curto → pede clarificação
        if len(prompt_lower) < 3:
            decision.reply_mode = "ask_clarification"
            decision.clarification_message = (
                "Sua mensagem está muito curta. Pode elaborar um pouco mais?"
            )
            decision.missing_inputs.append("prompt_too_short")
            return decision

        # Prompt vazio
        if not prompt_lower:
            decision.reply_mode = "ask_clarification"
            decision.clarification_message = "Como posso ajudar?"
            decision.missing_inputs.append("empty_prompt")
            return decision

        # Decide modo de resposta
        if decision.requires_wolfram or decision.requires_web_search:
            decision.reply_mode = "tool_first"
        else:
            decision.reply_mode = "direct"

        return decision

    def execute(
        self,
        decision: ToolDecision,
        prompt: str,
        direct_generator: Callable | None = None,
    ) -> dict:
        """Executa a decisão: chama ferramentas ou geração direta."""
        result = {
            "mode": decision.reply_mode,
            "domain": decision.domain,
            "tools_used": [],
            "wolfram_result": None,
            "web_result": None,
            "generated_text": None,
            "error": None,
        }

        # ── Step 1: Wolfram ──
        if decision.requires_wolfram and self.enable_wolfram and self.wolfram_callback:
            try:
                wolfram = self.wolfram_callback(decision.wolfram_query)
                result["wolfram_result"] = wolfram
                result["tools_used"].append("wolfram")
            except Exception as e:
                result["error"] = f"wolfram: {e}"

        # ── Step 2: Web Search ──
        if decision.requires_web_search and self.enable_web_search and self.web_search_callback:
            try:
                web = self.web_search_callback(prompt)
                result["web_result"] = web
                result["tools_used"].append("web_search")
            except Exception as e:
                result["error"] = (result["error"] or "") + f" web: {e}"

        # ── Step 3: Direct generation (always available) ──
        if direct_generator:
            try:
                text = direct_generator(prompt)
                result["generated_text"] = text
            except Exception as e:
                result["error"] = (result["error"] or "") + f" generation: {e}"

        return result

    def decide_and_execute(
        self,
        prompt: str,
        direct_generator: Callable | None = None,
    ) -> dict:
        """Atalho: decide + execute em um passo."""
        decision = self.decide(prompt)
        result = self.execute(decision, prompt, direct_generator)
        result["decision"] = {
            "domain": decision.domain,
            "requires_wolfram": decision.requires_wolfram,
            "requires_web_search": decision.requires_web_search,
            "reply_mode": decision.reply_mode,
            "reasons": decision.reasons,
        }
        return result


# ═══════════════════════════════════════════════════════════
# TESTES
# ═══════════════════════════════════════════════════════════

def test_math_intent():
    orch = Orchestrator()
    d = orch.decide("calcule a derivada de x^2")
    assert d.requires_wolfram
    assert d.domain == "math"
    assert d.reply_mode == "tool_first"

def test_code_intent():
    orch = Orchestrator()
    d = orch.decide("write a python function to sort a list")
    assert d.requires_code_exec or d.requires_self_reflection
    assert d.domain == "code_generation"

def test_identity_intent():
    orch = Orchestrator()
    d = orch.decide("quem é você?")
    assert d.requires_self_reflection
    assert d.domain == "identity"

def test_finance_intent():
    orch = Orchestrator()
    d = orch.decide("what is the stock price of AAPL?")
    assert d.requires_web_search or d.domain == "finance"

def test_short_prompt():
    orch = Orchestrator()
    d = orch.decide("oi")
    assert d.reply_mode == "direct" or d.reply_mode == "ask_clarification"

def test_empty_prompt():
    orch = Orchestrator()
    d = orch.decide("")
    assert d.reply_mode == "ask_clarification"

def test_casual_chat():
    orch = Orchestrator()
    d = orch.decide("what is the capital of France?")
    assert d.reply_mode == "direct"
    assert d.domain == "default"

def test_execute():
    orch = Orchestrator()
    d = orch.decide("hello")
    r = orch.execute(d, "hello", direct_generator=lambda p: "Hello, world!")
    assert r["generated_text"] == "Hello, world!"
    assert r["mode"] == "direct"

def test_decide_and_execute():
    orch = Orchestrator()
    r = orch.decide_and_execute("hello", direct_generator=lambda p: "Hi!")
    assert "decision" in r
    assert r["generated_text"] == "Hi!"
