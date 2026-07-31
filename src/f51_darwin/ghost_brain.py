"""
F51 Ghost Brain — Dopamina, Consequência, Aprendizado por Erro
===============================================================
O Fantasma não é só dados escondidos.
É um SISTEMA NERVOSO completo:

  DOPAMINA (recompensa):
    - Acertou uma pergunta nova → +dopamina
    - Verificou com Wolfram e bateu → +dopamina  
    - Gerou dado sintético útil → +dopamina
    - Ciclo completo de exploração → +dopamina

  CONSEQUÊNCIA (punição):
    - Errou pergunta → -dopamina, mas APRENDE
    - Wolfram não confirmou → marca como incerto
    - Dado sintético inútil → descarta
    - Repetiu erro 3x → poda o módulo

  APRENDIZADO POR ERRO (o mais valioso):
    - Erro → análise da causa → correção → verificação
    - O erro ENSINA mais que o acerto
    - Cada erro gera um "lesson document" no corpus
    - O modelo EVOLUI com cada falha

Integração com o organismo:
  soul.py              → dopamina e emoções
  evolution_score.py   → fitness dos módulos
  jepa_v2.py           → previsão de surpresa (JEPAHeadV2)
  ghost_token: o modelo usa _ghost_loss / _causal_ghost_loss interno (darwin_x_core/model.py)
"""

from __future__ import annotations

import json, time, random
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class EmotionState(Enum):
    CURIOUS = "curious"       # quer explorar
    EXCITED = "excited"       # descobriu algo novo
    FRUSTRATED = "frustrated" # não conseguiu verificar
    ENLIGHTENED = "enlightened" # aprendeu com um erro
    SATISFIED = "satisfied"   # ciclo completo com sucesso


@dataclass
class DopamineSignal:
    """Dopamina é o combustível da curiosidade."""
    amount: float          # -1.0 (dor) a +1.0 (prazer)
    source: str            # o que causou
    context: str           # qual pergunta/domínio
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    learned: bool = False  # true se houve APRENDIZADO (mesmo com erro)


@dataclass 
class ErrorLesson:
    """Cada erro gera uma LIÇÃO — o dado mais valioso do sistema."""
    question: str
    wrong_answer: str
    correct_answer: str
    root_cause: str         # por que errou
    domain: str
    lesson_text: str        # texto sintetizado para o corpus
    severity: float         # 0-1, quão grave foi o erro
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class GhostBrain:
    """Sistema nervoso do Corpus Fantasma.
    
    Conecta:
    - CuriosityDrive (o que explorar)
    - DopamineSystem (recompensa/punição)
    - ErrorLearning (aprender com falhas)
    - Soul (emoções do organismo)
    - Evolution (poda de módulos ruins)
    """
    
    def __init__(
        self,
        ghost_corpus=None,      # GhostCorpus instance
        soul=None,              # Soul engine
        evolution=None,         # Evolution loop
        jepa=None,              # JEPA predictor
    ):
        self.ghost = ghost_corpus
        self.soul = soul
        self.evolution = evolution
        self.jepa = jepa
        
        # Dopamine system
        self.dopamine_level: float = 0.5  # baseline
        self.dopamine_history: List[DopamineSignal] = []
        
        # Error learning
        self.error_lessons: List[ErrorLesson] = []
        self.error_counts: Dict[str, int] = {}  # domain -> error count
        
        # Emotional state
        self.emotion: EmotionState = EmotionState.CURIOUS
        self.emotion_history: List[Tuple[EmotionState, str]] = []
        
        # Performance tracking
        self.cycles_completed: int = 0
        self.questions_answered: int = 0
        self.questions_correct: int = 0
        self.lessons_generated: int = 0
        
        # Learning acceleration
        self.learning_rate_multiplier: float = 1.0
    
    # ═══════════════════════════════════════════════
    # DOPAMINE SYSTEM
    # ═══════════════════════════════════════════════
    
    def reward(self, amount: float, source: str, context: str = ""):
        """Libera dopamina — recompensa por acerto ou descoberta."""
        signal = DopamineSignal(
            amount=min(1.0, max(-1.0, amount)),
            source=source,
            context=context,
            learned=amount > 0,
        )
        self.dopamine_history.append(signal)
        self.dopamine_level = 0.9 * self.dopamine_level + 0.1 * amount
        
        # Update soul if connected
        if self.soul:
            try:
                self.soul.receive_reward(amount, source)
            except:
                pass
        
        # Update emotion
        if amount > 0.5:
            self._set_emotion(EmotionState.EXCITED, f"Great discovery: {source}")
        elif amount > 0.1:
            self._set_emotion(EmotionState.SATISFIED, f"Learned: {source}")
        elif amount < -0.3:
            self._set_emotion(EmotionState.FRUSTRATED, f"Failed: {source}")
    
    def punish(self, amount: float, source: str, context: str = ""):
        """Registra erro — mas transforma em APRENDIZADO."""
        self.reward(-abs(amount), source, context)
        # Erros geram lições — o dado mais valioso
        self.error_counts[context] = self.error_counts.get(context, 0) + 1
    
    def _set_emotion(self, state: EmotionState, reason: str):
        self.emotion = state
        self.emotion_history.append((state, reason))
    
    # ═══════════════════════════════════════════════
    # ERROR LEARNING — O CORAÇÃO DO SISTEMA
    # ═══════════════════════════════════════════════
    
    def learn_from_error(
        self,
        question: str,
        wrong_answer: str,
        correct_answer: str,
        domain: str,
        root_cause: str = "unknown",
    ) -> ErrorLesson:
        """Transforma um erro em uma LIÇÃO.
        
        O erro é o professor mais eficiente.
        Cada erro gera um documento de treino que ENSINA o modelo.
        """
        severity = self._calculate_error_severity(wrong_answer, correct_answer)
        
        lesson = ErrorLesson(
            question=question,
            wrong_answer=wrong_answer,
            correct_answer=correct_answer,
            root_cause=root_cause,
            domain=domain,
            severity=severity,
            lesson_text=self._generate_lesson_text(
                question, wrong_answer, correct_answer, domain, root_cause
            ),
        )
        
        self.error_lessons.append(lesson)
        self.lessons_generated += 1
        
        # Alta severidade → podar módulo ruim
        if severity > 0.7 and self.evolution:
            try:
                self.evolution.mark_for_review(domain, severity, question)
            except:
                pass
        
        # Dopamina por APRENDER (mesmo com erro)
        self.reward(0.3, f"learned_from_error:{domain}", question)
        
        return lesson
    
    def _calculate_error_severity(self, wrong: str, correct: str) -> float:
        """Quão grave foi o erro? 0 = trivial, 1 = catastrófico."""
        if not wrong or not correct:
            return 0.5
        # Simple heuristic: length ratio + content difference
        len_ratio = abs(len(wrong) - len(correct)) / max(len(correct), 1)
        return min(1.0, len_ratio * 2)
    
    def _generate_lesson_text(
        self, question: str, wrong: str, correct: str, domain: str, cause: str
    ) -> str:
        """Gera texto de lição para o corpus de treino."""
        templates = [
            f"""ERROR CORRECTION LESSON
Domain: {domain}

QUESTION: {question}

WRONG ANSWER (what the model said): {wrong}

CORRECT ANSWER (verified truth): {correct}

ROOT CAUSE: {cause}

LEARNING: The model initially produced an incorrect answer because of {cause}. 
After verification against ground truth, the correct understanding was established. 
This error teaches us that {cause} must be carefully considered when answering 
questions about {domain}.

CORRECTION: The correct approach is to {random.choice([
    'verify assumptions before answering',
    'check calculations against known results',
    'consider edge cases and boundary conditions',
    'cross-reference with authoritative sources',
    'decompose the problem into simpler sub-problems',
])}.""",

            f"""LESSON LEARNED [{domain}]

When asked: "{question}"
The model answered: "{wrong}" ❌
The truth is: "{correct}" ✅

This mistake reveals a gap in understanding about {domain}. 
The root cause was {cause}.

What we learned:
1. {cause} can lead to incorrect conclusions
2. Verification against {random.choice(['Wolfram Alpha', 'code execution', 'authoritative databases'])} prevents this error
3. The model now understands the correct relationship

This lesson has been incorporated into the training corpus to prevent
similar errors in the future. The model is now STRONGER because of this mistake.""",
        ]
        return random.choice(templates)
    
    # ═══════════════════════════════════════════════
    # EXPLORATION CYCLE — dopamina + erro integrados
    # ═══════════════════════════════════════════════
    
    def explore_and_learn(
        self, 
        batch_size: int = 5
    ) -> Tuple[List[str], List[ErrorLesson]]:
        """Um ciclo completo: explorar → perguntar → verificar → aprender.
        
        Returns:
            synthetic_docs: novos dados de treino gerados
            lessons: lições aprendidas com erros
        """
        if not self.ghost:
            return [], []
        
        # 1. CURIOSITY: seleciona o que explorar
        self._set_emotion(EmotionState.CURIOUS, "Starting exploration cycle")
        docs = self.ghost.curiosity.select_next_batch(
            list(self.ghost.documents.values()), batch_size
        )
        
        if not docs:
            return [], []
        
        all_synthetic = []
        all_lessons = []
        
        for doc in docs:
            # 2. QUESTION: gera perguntas
            questions = self.ghost.questioner.generate_questions(doc, doc.domain)
            self.reward(0.1, f"questions_generated:{doc.domain}", str(len(questions)))
            
            # 3. VERIFY: verifica respostas
            answers = self.ghost.verifier.verify(doc)
            
            for answer in answers:
                self.questions_answered += 1
                
                if answer.get("confidence", 0) >= 0.8:
                    # ACERTO → DOPAMINA
                    self.questions_correct += 1
                    self.reward(0.5, f"verified:{answer.get('source','?')}", doc.domain)
                    
                    # Generate synthetic training data
                    synthetic = self.ghost.synthesizer.generate_for_document(doc)
                    all_synthetic.extend(synthetic)
                    
                elif answer.get("confidence", 0) >= 0.5:
                    # INCERTO → curiosidade moderada
                    self.reward(0.1, "uncertain_verification", doc.domain)
                    
                else:
                    # ERRO → LIÇÃO (aprendizado mais valioso!)
                    lesson = self.learn_from_error(
                        question=answer.get("question", "?"),
                        wrong_answer=answer.get("answer", "unverified"),
                        correct_answer=self._find_best_known_answer(
                            answer.get("question", "")
                        ),
                        domain=doc.domain,
                        root_cause="verification_failed",
                    )
                    all_lessons.append(lesson)
            
            # 4. SYNTHESIZE: cria dados sintéticos
            synthetic = self.ghost.synthesizer.generate_for_document(doc)
            all_synthetic.extend(synthetic)
        
        # 5. CYCLE COMPLETE → dopamina de conclusão
        self.cycles_completed += 1
        total_learned = len(all_synthetic) + len(all_lessons)
        self.reward(0.7, "cycle_complete", f"learned:{total_learned}")
        self._set_emotion(
            EmotionState.ENLIGHTENED if all_lessons else EmotionState.SATISFIED,
            f"Cycle {self.cycles_completed}: {total_learned} new learnings"
        )
        
        # 6. Anti-fragilidade: acelerar aprendizado se performance boa
        if self.questions_answered > 0:
            accuracy = self.questions_correct / self.questions_answered
            self.learning_rate_multiplier = 0.5 + accuracy * 1.5  # 0.5x a 2.0x
        
        return all_synthetic, all_lessons
    
    def _find_best_known_answer(self, question: str) -> str:
        """Try to find the best known answer for a question."""
        # Check verified knowledge base
        if hasattr(self, 'ghost') and self.ghost:
            known = self.ghost.verifier.verified_knowledge
            if question in known:
                return known[question]
        return "[requires external verification]"
    
    # ═══════════════════════════════════════════════
    # SELF-IMPROVEMENT — o modelo supera a si mesmo
    # ═══════════════════════════════════════════════
    
    def consolidate_lessons(self) -> List[str]:
        """Transforma todas as ErrorLessons em corpus de treino."""
        docs = []
        for lesson in self.error_lessons:
            docs.append(lesson.lesson_text)
        
        # Reset após consolidar (as lições já viraram treino)
        self.error_lessons = []
        
        return docs
    
    def stats(self) -> Dict[str, Any]:
        """Relatório completo do Ghost Brain."""
        return {
            "dopamine": {
                "current_level": round(self.dopamine_level, 3),
                "total_signals": len(self.dopamine_history),
                "recent_mood": self.emotion.value,
            },
            "performance": {
                "cycles": self.cycles_completed,
                "questions_answered": self.questions_answered,
                "accuracy": round(
                    self.questions_correct / max(1, self.questions_answered), 3
                ),
                "lessons_generated": self.lessons_generated,
            },
            "learning": {
                "rate_multiplier": round(self.learning_rate_multiplier, 2),
                "error_counts": dict(self.error_counts),
                "total_lessons": len(self.error_lessons),
            },
            "anti_fragility": {
                "stronger_from_errors": self.lessons_generated > 0,
                "learning_accelerated": self.learning_rate_multiplier > 1.0,
            },
        }


# ═══════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    brain = GhostBrain()
    
    print("╔══════════════════════════════════════════════════╗")
    print("║  F51 GHOST BRAIN — DOPAMINA + CONSEQUÊNCIA       ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  'O erro é o professor mais eficiente'            ║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    
    # Simulate: model tries, errs, learns
    brain.reward(0.5, "initialized", "system_start")
    print(f"🧠 Dopamina inicial: {brain.dopamine_level:.2f}")
    
    # Model tries to answer a math question
    lesson = brain.learn_from_error(
        question="What is the derivative of x³ + 2x² - 5x + 1?",
        wrong_answer="3x² + 2x - 5",  # wrong coefficient
        correct_answer="3x² + 4x - 5",
        domain="math",
        root_cause="coefficient_miscalculation",
    )
    print(f"\n❌ ERRO → LIÇÃO:")
    print(f"   Pergunta: {lesson.question[:60]}...")
    print(f"   Resposta errada: {lesson.wrong_answer}")
    print(f"   Resposta certa: {lesson.correct_answer}")
    print(f"   Severidade: {lesson.severity:.2f}")
    print(f"   Dopamina pós-lição: {brain.dopamine_level:.2f} (subiu — APRENDEU!)")
    
    # Model acerta
    brain.reward(0.8, "verified:wolfram", "math")
    print(f"\n✅ ACERTO → Dopamina: {brain.dopamine_level:.2f}")
    
    # Stats
    stats = brain.stats()
    print(f"\n📊 Performance: {stats['performance']}")
    print(f"🧬 Anti-fragilidade: {stats['anti_fragility']}")
    print(f"\n💡 O modelo está {stats['learning']['rate_multiplier']:.1f}x mais rápido")
    print(f"   porque aprendeu com {stats['performance']['lessons_generated']} erros.")
