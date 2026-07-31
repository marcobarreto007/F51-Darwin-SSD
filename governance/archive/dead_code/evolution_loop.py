"""
F51 Evolution Loop — CDF (Cerveau de Diagnostic / Diagnostic Brain).

Adaptado do BMW-Diag diagnostic_flow.py.
Aplica o ciclo CDF à evolução de módulos do organismo F51.

Ciclo CDF:
    Hypothesis → Evidence → Test → Result → Recalculate → Validate

Cada módulo do F51 é tratado como uma "hipótese de diagnóstico":
    - Nasce como CANDIDATE (hipótese)
    - Passa por testes (eval, ablation)
    - Ganha ou perde confiança baseado em EVIDÊNCIA REAL
    - Se confiante → promovido a ACTIVE
    - Se consistentemente falha → QUARANTINE → DEAD

Uso:
    from f51_darwin.evolution_loop import EvolutionLoop, ModuleHypothesis
    loop = EvolutionLoop()
    loop.evaluate_module(module_id, test_result)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from f51_darwin.decision_engine import decide, DecisionFactors, clamp


# ═══════════════════════════════════════════════════════════
# HIPÓTESE DE MÓDULO
# ═══════════════════════════════════════════════════════════

@dataclass
class ModuleHypothesis:
    """Um módulo candidato no ciclo CDF evolutivo."""
    module_id: str
    domain: str = "default"
    confidence: float = 0.50
    test_results: dict[int, dict] = field(default_factory=dict)
    tests_passed: int = 0
    tests_failed: int = 0
    tests_total: int = 0
    born_at_cycle: int = 0
    last_evaluated: str = ""
    status: str = "candidate"  # candidate → active → quarantine → dead
    evidence_log: list[str] = field(default_factory=list)

    def record_test(self, step: int, passed: bool, measurement: str = "", note: str = ""):
        """Registra resultado de um teste."""
        self.test_results[step] = {
            "passed": passed,
            "measurement": measurement,
            "note": note,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.tests_total += 1
        if passed:
            self.tests_passed += 1
        else:
            self.tests_failed += 1
        self.last_evaluated = datetime.now(timezone.utc).isoformat()

    @property
    def pass_rate(self) -> float:
        if self.tests_total == 0:
            return 0.0
        return self.tests_passed / self.tests_total


# ═══════════════════════════════════════════════════════════
# CDF EVOLUTION LOOP
# ═══════════════════════════════════════════════════════════

class EvolutionLoop:
    """Motor de evolução baseado no ciclo CDF do BMW-Diag.

    Hypothesis → Evidence → Test → Result → Recalculate → Validate
    """

    def __init__(self):
        self.hypotheses: dict[str, ModuleHypothesis] = {}

    def register_hypothesis(
        self,
        module_id: str,
        domain: str = "default",
        initial_confidence: float = 0.50,
        cycle: int = 0,
    ) -> ModuleHypothesis:
        """Registra um novo módulo como hipótese a ser testada."""
        h = ModuleHypothesis(
            module_id=module_id,
            domain=domain,
            confidence=initial_confidence,
            born_at_cycle=cycle,
        )
        self.hypotheses[module_id] = h
        return h

    def apply_test_result(
        self,
        module_id: str,
        step: int,
        passed: bool,
        measurement: str = "",
    ) -> dict:
        """Aplica resultado de teste e recalcula hipóteses.

        Este é o CORE do CDF loop:
        1. Registra pass/fail
        2. Recalcula confiança baseado em evidência
        3. Atualiza status do módulo
        4. Sugere próximo teste
        """
        h = self.hypotheses.get(module_id)
        if h is None:
            return {"error": f"Module {module_id} not found"}

        h.record_test(step, passed, measurement)
        self._recalculate_confidence(h)
        self._update_status(h)
        next_test = self._suggest_next_test(h)

        return {
            "module_id": module_id,
            "confidence": h.confidence,
            "status": h.status,
            "pass_rate": h.pass_rate,
            "tests_done": h.tests_total,
            "next_test": next_test,
            "evidence": h.evidence_log[-3:],
        }

    def _recalculate_confidence(self, h: ModuleHypothesis):
        """Recalcula confiança baseado em evidência acumulada.

        Regras:
        - Cada PASS: +0.10 confiança (mas menos que o anterior)
        - Cada FAIL: -0.15 confiança
        - Pass rate baixo: penalidade extra
        - Evidência consistente: bônus
        """
        old_confidence = h.confidence
        confidence = 0.50  # baseline

        for step, result in sorted(h.test_results.items()):
            if result["passed"]:
                # Ganho decrescente com cada passo
                gain = 0.10 / (1 + 0.1 * (step - 1))
                confidence = clamp(confidence + gain)
            else:
                confidence = clamp(confidence - 0.15)

        # Penalidade por pass rate baixo
        if h.tests_total >= 3 and h.pass_rate < 0.5:
            confidence = clamp(confidence - 0.10)

        # Bônus por evidência consistente (todos passaram)
        if h.tests_total >= 3 and h.pass_rate >= 0.9:
            confidence = clamp(confidence + 0.08)

        h.confidence = round(confidence, 3)

        if h.confidence != old_confidence:
            direction = "↑" if h.confidence > old_confidence else "↓"
            h.evidence_log.append(
                f"[{h.last_evaluated}] confidence {old_confidence:.3f}→{h.confidence:.3f} "
                f"{direction} (step={len(h.test_results)}, pass_rate={h.pass_rate:.0%})"
            )

    def _update_status(self, h: ModuleHypothesis):
        """Atualiza status baseado na confiança atual."""
        if h.confidence >= 0.75:
            h.status = "active"
        elif h.confidence >= 0.40:
            h.status = "candidate"
        elif h.confidence >= 0.20:
            h.status = "quarantine"
        else:
            h.status = "dead"

    def _suggest_next_test(self, h: ModuleHypothesis) -> dict:
        """Sugere o próximo teste baseado no estado atual."""
        if h.status == "dead":
            return {"action": "remove", "reason": "confidence too low"}

        if h.status == "quarantine":
            return {
                "action": "ablation_test",
                "reason": "module in quarantine — run ablation to confirm",
            }

        if h.tests_total == 0:
            return {"action": "eval_test", "reason": "run first evaluation"}

        if h.pass_rate >= 0.8 and h.tests_total >= 3:
            return {"action": "promote", "reason": "consistently passing — ready for activation"}

        if h.tests_failed > h.tests_passed:
            return {
                "action": "investigate",
                "reason": f"more failures ({h.tests_failed}) than passes ({h.tests_passed})",
            }

        return {"action": "continue_testing", "reason": "gather more evidence"}

    def evaluate_with_factors(
        self,
        module_id: str,
        eval_loss: float,
        replay_loss: float | None = None,
        forgetting_proxy: float = 0.0,
    ) -> dict:
        """Avalia módulo usando fatores de decisão do decision_engine."""
        h = self.hypotheses.get(module_id)
        if h is None:
            return {"error": f"Module {module_id} not found"}

        # Constrói fatores a partir das métricas de treino
        veracity = clamp(1.0 - eval_loss / 10.0)  # loss baixa → alta veracidade
        robustness = clamp(1.0 - forgetting_proxy)  # sem esquecimento → robusto
        consensus = clamp(h.pass_rate)
        proxy = clamp(0.5 + h.tests_total * 0.05)  # mais testes → mais confiança no proxy
        uncertainty = clamp(eval_loss / 5.0)  # loss alta → incerteza

        factors = DecisionFactors(
            veracity=veracity,
            consensus=consensus,
            proxy_reward=proxy,
            robustness=robustness,
            uncertainty=uncertainty,
        )
        confidence = decide(factors)

        # Registra como "teste" implícito
        passed = eval_loss < 2.0
        self.apply_test_result(module_id, h.tests_total + 1, passed,
                              f"eval_loss={eval_loss:.3f}")

        return {
            "module_id": module_id,
            "confidence": round(confidence, 3),
            "factors": {
                "veracity": veracity,
                "consensus": consensus,
                "proxy": proxy,
                "robustness": robustness,
                "uncertainty": uncertainty,
            },
            "status": h.status,
        }

    def status_report(self) -> str:
        """Relatório de todos os módulos no loop."""
        if not self.hypotheses:
            return "No modules in evolution loop."

        lines = ["EVOLUTION LOOP STATUS", "=" * 50]
        for h in sorted(self.hypotheses.values(),
                       key=lambda x: x.confidence, reverse=True):
            bar = "█" * int(h.confidence * 10) + "░" * (10 - int(h.confidence * 10))
            lines.append(
                f"  {h.module_id:<30} [{bar}] {h.confidence:.2f} "
                f"({h.status}) tests={h.tests_total} pass={h.pass_rate:.0%}"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# TESTES
# ═══════════════════════════════════════════════════════════

def test_evolution_loop_basic():
    loop = EvolutionLoop()
    h = loop.register_hypothesis("module_a", domain="math", cycle=0)
    assert h.status == "candidate"
    assert h.confidence == 0.50

    # Test 1: PASS
    result = loop.apply_test_result("module_a", 1, True, "loss=0.5")
    assert result["confidence"] > 0.50  # ganhou confiança

    # Test 2: PASS
    loop.apply_test_result("module_a", 2, True, "loss=0.4")
    assert h.confidence > 0.55

    # Test 3: PASS → deve promover
    result = loop.apply_test_result("module_a", 3, True, "loss=0.3")
    assert h.status == "active", f"Expected active, got {h.status}"

    print("  evolution_loop_basic: OK")


def test_module_dies():
    loop = EvolutionLoop()
    loop.register_hypothesis("module_b", cycle=0)

    # 5 failures em sequência → deve morrer
    for i in range(5):
        loop.apply_test_result("module_b", i + 1, False, f"fail_{i}")
    h = loop.hypotheses["module_b"]
    assert h.status in ("quarantine", "dead"), f"Expected quarantine/dead, got {h.status}"

    print("  module_dies: OK")


def test_evaluate_with_factors():
    loop = EvolutionLoop()
    loop.register_hypothesis("module_c", domain="math", cycle=0)

    result = loop.evaluate_with_factors("module_c", eval_loss=0.5,
                                        replay_loss=1.0, forgetting_proxy=0.1)
    assert "confidence" in result
    assert "factors" in result
    assert 0.0 < result["confidence"] < 1.0

    print("  evaluate_with_factors: OK")


def test_status_report():
    loop = EvolutionLoop()
    loop.register_hypothesis("module_x", cycle=0)
    loop.register_hypothesis("module_y", cycle=1)
    report = loop.status_report()
    assert "module_x" in report
    assert "module_y" in report

    print("  status_report: OK")
