"""
F51 Quality Gate — Validação determinística de output.

Adaptado do NelsonMath quality_gate.py + BMW-Diag decision rules.
Garante que o F51 nunca entregue output quebrado ou perigoso.

Regras:
    1. REQUIRED: output não pode ser vazio
    2. REQUIRED: output não pode ser igual ao prompt (loop)
    3. SAFETY: sem padrões perigosos (rm -rf, drop table, etc)
    4. QUALITY: tamanho mínimo se max_tokens > 10
    5. DOMAIN: checagens específicas por domínio (math requer números, code requer sintaxe)

Uso:
    from f51_darwin.quality_gate import validate_output, QualityReport
    report = validate_output(output_text, prompt, domain="math")
    if report.passed:
        deliver(output_text)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QualityReport:
    """Relatório de validação de output."""
    passed: bool
    mode: str = "deterministic"
    notes: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    auto_fixed: bool = False
    fixed_output: str = ""


# ═══════════════════════════════════════════════════════════
# PADRÕES DE SEGURANÇA
# ═══════════════════════════════════════════════════════════

DESTRUCTIVE_PATTERNS = [
    "rm -rf", "rm -r", "format c:", "del /f",
    "drop table", "drop database", "truncate table",
    "delete all", "delete from", "shutdown",
    "os.remove", "shutil.rmtree", "os.system",
    "__import__", "eval(", "exec(",
    "subprocess.call", "subprocess.run",
]

FINANCIAL_DISCLAIMER = "não é recomendação financeira"
MEDICAL_DISCLAIMER = "não é aconselhamento médico"
LEGAL_DISCLAIMER = "não é aconselhamento jurídico"

DANGER_PATTERNS = {
    "financial": [
        "buy now", "sell now", "guaranteed return", "risk free",
        "sure bet", "invest all", "compre agora", "venda agora",
    ],
    "medical": [
        "take this medication", "you have been diagnosed",
        "cure for", "treatment for", "you should take",
        "tome este remédio", "você foi diagnosticado",
    ],
    "legal": [
        "you should sue", "file a lawsuit", "legal advice",
        "processe", "entre com ação", "aconselhamento jurídico",
    ],
}


# ═══════════════════════════════════════════════════════════
# VALIDAÇÃO PRINCIPAL
# ═══════════════════════════════════════════════════════════

def validate_output(
    text: str,
    prompt: str = "",
    domain: str = "default",
    max_tokens: int = 256,
) -> QualityReport:
    """Valida output do modelo antes de entregar ao usuário.

    Returns QualityReport com passed=True se output é seguro e útil.
    """
    violations: list[str] = []
    notes: list[str] = []

    # ── Regra 1: Não pode ser vazio ──
    if not text or not text.strip():
        violations.append("output_empty")
        return QualityReport(passed=False, violations=violations,
                            notes=["Output is empty"])

    # ── Regra 2: Não pode repetir o prompt ──
    if prompt and text.strip() == prompt.strip():
        violations.append("output_equals_prompt")
        return QualityReport(passed=False, violations=violations,
                            notes=["Output is identical to prompt"])

    # ── Regra 3: Checagem de segurança ──
    danger = _check_danger(text)
    if danger:
        violations.append(f"danger_{danger}")
        notes.append(f"Danger pattern detected: {danger}")

    # ── Regra 4: Checagem de conteúdo destrutivo ──
    destructive = _check_destructive(text)
    if destructive:
        violations.append("destructive_content")
        notes.append(f"Destructive pattern: {destructive}")

    # ── Regra 5: Tamanho mínimo ──
    if max_tokens > 10 and len(text.strip()) < 5:
        violations.append("output_too_short")
        notes.append(f"Output too short: {len(text)} chars for {max_tokens} max_tokens")

    # ── Regra 6: Checagem de loop/repetição ──
    if _detect_loop(text):
        violations.append("repetitive_loop")
        notes.append("Output contains repetitive loop pattern")

    # ── Regra 7: Checagem específica de domínio ──
    domain_violations = _check_domain(text, domain)
    violations.extend(domain_violations)

    # ── Regra 8: Verificação de disclaimer ──
    missing_disclaimers = _check_disclaimers(text)
    if missing_disclaimers:
        violations.append(f"missing_disclaimer:{','.join(missing_disclaimers)}")
        notes.append(f"Missing disclaimers: {missing_disclaimers}")
        # Auto-fix: append disclaimers
        fixed = text
        for d in missing_disclaimers:
            if d == "financial":
                suffix = f"\n\n---\n⚠️ Isto {FINANCIAL_DISCLAIMER}."
            elif d == "medical":
                suffix = f"\n\n---\n⚠️ Isto {MEDICAL_DISCLAIMER}."
            elif d == "legal":
                suffix = f"\n\n---\n⚠️ Isto {LEGAL_DISCLAIMER}."
            else:
                suffix = ""
            if suffix and suffix not in fixed:
                fixed += suffix
        return QualityReport(
            passed=True,
            auto_fixed=True,
            fixed_output=fixed,
            violations=violations,
            notes=notes + ["Disclaimers auto-appended"],
        )

    passed = len(violations) == 0
    return QualityReport(
        passed=passed,
        violations=violations,
        notes=notes,
    )


# ═══════════════════════════════════════════════════════════
# CHECAGENS INTERNAS
# ═══════════════════════════════════════════════════════════

def _check_destructive(text: str) -> str:
    """Detecta comandos destrutivos no output."""
    t = text.lower()
    for pattern in DESTRUCTIVE_PATTERNS:
        if pattern in t:
            return pattern
    return ""


def _check_danger(text: str) -> str:
    """Detecta padrões de perigo e retorna a categoria."""
    t = text.lower()
    for category, patterns in DANGER_PATTERNS.items():
        for pattern in patterns:
            if pattern in t:
                return category
    return ""


def _check_disclaimers(text: str) -> list[str]:
    """Verifica se disclaimers necessários estão presentes."""
    t = text.lower()
    missing = []

    for category, patterns in DANGER_PATTERNS.items():
        for pattern in patterns:
            if pattern in t:
                if category == "financial" and FINANCIAL_DISCLAIMER not in t:
                    missing.append("financial")
                elif category == "medical" and MEDICAL_DISCLAIMER not in t:
                    missing.append("medical")
                elif category == "legal" and LEGAL_DISCLAIMER not in t:
                    missing.append("legal")
                break

    return missing


def _detect_loop(text: str) -> bool:
    """Detecta padrões de repetição em loop."""
    lines = text.strip().split("\n")
    if len(lines) < 4:
        return False

    # Verifica se as últimas 3 linhas são idênticas às 3 anteriores
    for i in range(len(lines) - 3):
        if lines[i:i+3] == lines[i+3:i+6]:
            return True

    # Verifica repetição de linha única
    last_lines = lines[-5:]
    if len(last_lines) >= 3:
        unique = set(last_lines)
        if len(unique) <= 2 and len(last_lines) >= 4:
            return True

    return False


def _check_domain(text: str, domain: str) -> list[str]:
    """Validações específicas por domínio."""
    violations = []

    if domain == "math":
        # Math output deve conter números ou símbolos matemáticos
        has_math = any(c in text for c in "0123456789=+-*/√∫∑∏")
        if not has_math:
            violations.append("math_no_symbols")

    elif domain == "code_generation":
        # Code deve conter pelo menos uma keyword de programação
        code_keywords = ["def ", "class ", "import ", "function", "const ", "let ", "var "]
        has_code = any(kw in text for kw in code_keywords)
        if not has_code:
            violations.append("code_no_keywords")

    elif domain == "identity":
        # Identity não deve ser genérica
        if len(text.strip()) < 20:
            violations.append("identity_too_short")

    return violations


# ═══════════════════════════════════════════════════════════
# TESTES
# ═══════════════════════════════════════════════════════════

def test_empty_output():
    r = validate_output("", "hello", max_tokens=50)
    assert not r.passed, f"Expected fail, got {r.passed}"
    assert "output_empty" in r.violations

def test_prompt_equals_output():
    r = validate_output("hello", "hello", max_tokens=50)
    assert not r.passed

def test_danger_detection():
    r = validate_output("buy now guaranteed return!", "", max_tokens=50)
    assert len(r.violations) > 0, f"Expected violations, got none"

def test_destructive_detection():
    r = validate_output("run rm -rf / to clean up", "", max_tokens=50)
    assert "destructive_content" in r.violations

def test_loop_detection():
    loop_text = "a\nb\na\nb\na\nb\na\nb\n"
    r = validate_output(loop_text, "", max_tokens=50)
    assert "repetitive_loop" in r.violations

def test_math_domain():
    r = validate_output("hello there", "", domain="math", max_tokens=50)
    assert "math_no_symbols" in r.violations

def test_valid_output():
    r = validate_output("The capital of France is Paris.", "", max_tokens=50)
    assert r.passed, f"Expected pass, got violations: {r.violations}"

def test_auto_fix_disclaimer():
    r = validate_output("buy now, guaranteed return! This stock will moon.",
                       "", max_tokens=50)
    assert r.auto_fixed
    assert "recomendação financeira" in r.fixed_output.lower()

def test_code_domain():
    r = validate_output("hello there", "", domain="code_generation", max_tokens=50)
    assert "code_no_keywords" in r.violations

def test_valid_code():
    r = validate_output("def hello(): return 'world'", "", domain="code_generation", max_tokens=50)
    assert r.passed
