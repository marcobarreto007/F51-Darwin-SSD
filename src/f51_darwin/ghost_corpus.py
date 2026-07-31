"""
F51 Ghost Corpus — Auto-Evolution Engine
=========================================
O modelo não depende mais de dados externos.
Ele CRIA seu próprio conhecimento.
Ele EXPLORA o desconhecido.
Ele se torna MAIS INTELIGENTE que os dados originais.

Pipeline:
  Corpus Total → 80% Treino Inicial / 20% Fantasma
  ┌─ Treino Inicial: modelo nasce, aprende o básico
  └─ AUTO-EVOLUÇÃO:
      1. CuriosityDrive: escolhe o que explorar do fantasma
      2. QuestionGenerator: cria perguntas sobre o desconhecido
      3. Verifier (Wolfram/Web/Code): verifica respostas
      4. SyntheticDataGenerator: cria novos dados de treino
      5. SelfTrainer: auto-treina nos próprios dados gerados
      6. Evolution Loop: módulos nascem/morrem baseado em performance
"""

from __future__ import annotations

import json, random, hashlib, time
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ExplorationStatus(str, Enum):
    UNEXPLORED = "unexplored"
    QUESTIONED = "questioned"
    VERIFIED = "verified"
    SYNTHESIZED = "synthesized"
    LEARNED = "learned"
    FAILED = "failed"


@dataclass
class GhostDocument:
    """A document held back from initial training — unknown to the model."""
    id: str
    domain: str
    text: str
    size_bytes: int
    status: ExplorationStatus = ExplorationStatus.UNEXPLORED
    curiosity_score: float = 0.0
    questions_generated: List[str] = field(default_factory=list)
    verified_answers: List[Dict] = field(default_factory=list)
    synthetic_data: List[str] = field(default_factory=list)
    exploration_attempts: int = 0
    last_explored: Optional[str] = None


class CuriosityDrive:
    """Decides WHAT to explore from the ghost corpus.
    
    Not all ghost documents are equally valuable.
    Curiosity scores based on:
    - Domain alignment (math > finance > bio > quantum)
    - Novelty (how different from already-learned data)
    - Uncertainty (JEPA prediction error)
    - Information density (tokens per concept)
    """
    
    DOMAIN_WEIGHTS = {
        "math": 5.0,
        "physics": 4.5,
        "finance": 4.0,
        "quantum": 3.5,
        "code": 3.0,
        "biology": 2.5,
        "philosophy": 2.0,
        "history": 1.5,
        "general": 1.0,
    }
    
    def __init__(self, jepa_predictor=None):
        self.jepa = jepa_predictor
        self.exploration_history: Dict[str, float] = {}
    
    def score_document(self, doc: GhostDocument) -> float:
        """Calculate curiosity score for a ghost document.
        
        Higher score = more valuable to explore NOW.
        """
        score = 0.0
        
        # 1. Domain priority
        for domain, weight in self.DOMAIN_WEIGHTS.items():
            if domain in doc.domain.lower():
                score += weight * 2.0
                break
        
        # 2. Novelty bonus (unexplored = high value)
        if doc.status == ExplorationStatus.UNEXPLORED:
            score += 5.0
        elif doc.status == ExplorationStatus.FAILED:
            score -= 2.0  # reduce retries on failed
        
        # 3. JEPA prediction error (high surprise = high curiosity)
        if self.jepa and doc.text:
            try:
                surprise = self.jepa.predict_surprise(doc.text[:1000])
                score += surprise * 3.0
            except:
                pass
        
        # 4. Information density (longer docs = more to learn)
        score += min(doc.size_bytes / 10000, 5.0)
        
        # 5. Diminishing returns on re-exploration
        score *= 0.7 ** doc.exploration_attempts
        
        doc.curiosity_score = score
        return score
    
    def select_next_batch(
        self, 
        documents: List[GhostDocument], 
        batch_size: int = 10
    ) -> List[GhostDocument]:
        """Select the most curious documents to explore next."""
        # Score all unexplored/questioned docs
        candidates = [
            d for d in documents 
            if d.status in (ExplorationStatus.UNEXPLORED, ExplorationStatus.QUESTIONED)
        ]
        
        for doc in candidates:
            self.score_document(doc)
        
        # Pick top-N by curiosity score
        candidates.sort(key=lambda d: d.curiosity_score, reverse=True)
        selected = candidates[:batch_size]
        
        for doc in selected:
            doc.exploration_attempts += 1
            doc.last_explored = datetime.now(timezone.utc).isoformat()
        
        return selected


class QuestionGenerator:
    """Generates QUESTIONS from ghost documents.
    
    The model reads unknown data and asks:
    - 'What is X?' (definitions)
    - 'Why does X happen?' (causal)
    - 'How is X calculated?' (procedural)
    - 'What if X changes?' (counterfactual)
    - 'How does X relate to Y?' (relational)
    """
    
    QUESTION_TEMPLATES = [
        "What is {concept}? Explain in detail.",
        "Why does {concept} matter in {domain}?",
        "How is {concept} calculated or derived?",
        "What are the key assumptions behind {concept}?",
        "How does {concept} relate to other concepts in {domain}?",
        "What would happen if the assumptions of {concept} were violated?",
        "What are the practical applications of {concept}?",
        "What is the historical origin of {concept}?",
        "What are the limitations or criticisms of {concept}?",
        "How would you explain {concept} to a beginner?",
    ]
    
    def extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text for questioning."""
        # Simple: extract capitalized phrases and key terms
        concepts = []
        words = text.split()
        for i, word in enumerate(words):
            # Capitalized phrases (likely proper nouns/concepts)
            if word[0].isupper() and len(word) > 3 and word.isalpha():
                # Get surrounding context (2 words before/after)
                start = max(0, i-2)
                end = min(len(words), i+3)
                phrase = ' '.join(words[start:end])
                if len(phrase) > 10:
                    concepts.append(phrase.strip('.,;:()'))
        
        # Deduplicate and limit
        seen = set()
        unique = []
        for c in concepts:
            if c.lower() not in seen:
                seen.add(c.lower())
                unique.append(c)
        
        return unique[:5]
    
    def generate_questions(self, doc: GhostDocument, domain: str) -> List[str]:
        """Generate curiosity-driven questions from a ghost document."""
        concepts = self.extract_concepts(doc.text)
        if not concepts:
            # Fallback: extract first meaningful sentence
            sentences = [s.strip() for s in doc.text.split('.') if len(s.strip()) > 30]
            concepts = sentences[:3] if sentences else ["this topic"]
        
        questions = []
        for concept in concepts[:3]:
            template = random.choice(self.QUESTION_TEMPLATES)
            q = template.format(concept=concept, domain=domain)
            questions.append(q)
        
        doc.questions_generated = questions
        doc.status = ExplorationStatus.QUESTIONED
        return questions


class Verifier:
    """Verifies answers using Wolfram Alpha, web search, or code execution.
    
    Verification sources:
    - Wolfram Alpha: mathematical verification
    - Code execution: computational verification
    - Known facts database: previously verified knowledge
    """
    
    def __init__(self, wolfram_bridge=None):
        self.wolfram = wolfram_bridge
        self.verified_knowledge: Dict[str, str] = {}
    
    def verify_with_wolfram(self, question: str) -> Optional[Dict]:
        """Verify a mathematical/scientific question with Wolfram Alpha."""
        if not self.wolfram:
            return None
        
        try:
            result = self.wolfram.query(question)
            if result.success:
                return {
                    "source": "wolfram",
                    "question": question,
                    "answer": result.result_text,
                    "confidence": 0.95 if result.success else 0.3,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
        except Exception:
            pass
        return None
    
    def verify_with_code(self, question: str) -> Optional[Dict]:
        """Verify a computational question by executing code."""
        # For math/calculation questions, try to extract and compute
        math_keywords = ["calculate", "compute", "solve", "evaluate", "what is", "how many"]
        if not any(kw in question.lower() for kw in math_keywords):
            return None
        
        # Generate and execute simple Python to verify
        try:
            # Extract mathematical expression
            import re
            numbers = re.findall(r'\d+\.?\d*', question)
            if len(numbers) >= 2:
                a, b = float(numbers[0]), float(numbers[1])
                if "sum" in question.lower() or "add" in question.lower():
                    result = a + b
                elif "product" in question.lower() or "multiply" in question.lower():
                    result = a * b
                elif "divide" in question.lower():
                    result = a / b if b != 0 else float('inf')
                else:
                    return None
                
                return {
                    "source": "code",
                    "question": question,
                    "answer": str(result),
                    "confidence": 0.99,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
        except Exception:
            pass
        return None
    
    def verify(self, doc: GhostDocument) -> List[Dict]:
        """Verify all questions for a ghost document."""
        answers = []
        
        for question in doc.questions_generated:
            # Check known knowledge first
            if question in self.verified_knowledge:
                answers.append({
                    "source": "cache",
                    "question": question,
                    "answer": self.verified_knowledge[question],
                    "confidence": 1.0,
                })
                continue
            
            # Try Wolfram
            result = self.verify_with_wolfram(question)
            if result:
                answers.append(result)
                self.verified_knowledge[question] = result["answer"]
                continue
            
            # Try code execution
            result = self.verify_with_code(question)
            if result:
                answers.append(result)
                self.verified_knowledge[question] = result["answer"]
                continue
            
            # Unverified — mark for human review or retry
            answers.append({
                "source": "unverified",
                "question": question,
                "answer": None,
                "confidence": 0.0,
            })
        
        doc.verified_answers = answers
        doc.status = ExplorationStatus.VERIFIED
        return answers


class SyntheticDataGenerator:
    """Creates NEW training data from verified answers.
    
    Takes verified Q&A pairs and generates:
    - Explanatory text with the verified answer
    - Related questions for deeper exploration
    - Counterfactual scenarios
    - Application examples
    """
    
    def generate_from_qa(self, question: str, answer: Dict) -> str:
        """Generate synthetic training document from verified Q&A."""
        if not answer.get("answer"):
            return ""
        
        source = answer.get("source", "unknown")
        confidence = answer.get("confidence", 0.0)
        
        if confidence < 0.5:
            return ""  # Don't train on low-confidence answers
        
        templates = [
            f"QUESTION: {question}\n\nANSWER (verified by {source}): {answer['answer']}\n\nThis knowledge was verified through {source} with confidence {confidence:.0%}. The answer represents ground truth that the model did not know during initial training.",
            
            f"DISCOVERY: While exploring the unknown, the model encountered the concept behind '{question}'. After verification via {source}, it learned: {answer['answer']}. This new understanding connects to the model's existing knowledge of related domains.",
            
            f"LEARNING: {question}\n\n{answer['answer']}\n\nSource: {source} (confidence: {confidence:.0%}). This represents genuine learning — the model did NOT have this knowledge from initial training. It was DISCOVERED through curiosity-driven exploration.",
        ]
        
        return random.choice(templates)
    
    def generate_for_document(self, doc: GhostDocument) -> List[str]:
        """Generate synthetic training data from a ghost document's verified answers."""
        synthetic = []
        
        for answer in doc.verified_answers:
            if answer.get("answer") and answer.get("confidence", 0) >= 0.5:
                text = self.generate_from_qa(answer["question"], answer)
                if text:
                    synthetic.append(text)
        
        doc.synthetic_data = synthetic
        doc.status = ExplorationStatus.SYNTHESIZED
        return synthetic


class GhostCorpus:
    """Complete Ghost Corpus System — 80/20 split, curiosity, verification, auto-evolution."""
    
    def __init__(
        self,
        corpus_dir: Path,
        ghost_dir: Path,
        split_ratio: float = 0.20,
        wolfram_bridge=None,
        jepa_predictor=None,
    ):
        self.corpus_dir = Path(corpus_dir)
        self.ghost_dir = Path(ghost_dir)
        self.split_ratio = split_ratio
        self.ghost_dir.mkdir(parents=True, exist_ok=True)
        
        self.curiosity = CuriosityDrive(jepa_predictor)
        self.questioner = QuestionGenerator()
        self.verifier = Verifier(wolfram_bridge)
        self.synthesizer = SyntheticDataGenerator()
        
        self.documents: Dict[str, GhostDocument] = {}
        self.evolution_log: List[Dict] = []
        
        self._load_or_split()
    
    def _load_or_split(self):
        """Load existing ghost corpus or split from training corpus."""
        meta_path = self.ghost_dir / "ghost_meta.json"
        
        if meta_path.exists():
            self._load_ghost_corpus()
        else:
            self._split_corpus()
    
    def _split_corpus(self):
        """Split corpus: 80% training, 20% ghost."""
        from f51_darwin.data import discover_corpus_files
        
        all_files = discover_corpus_files(self.corpus_dir)
        random.shuffle(all_files)
        
        split_point = int(len(all_files) * self.split_ratio)
        ghost_files = all_files[:split_point]
        
        for path in ghost_files:
            text = path.read_text(encoding="utf-8", errors="replace")
            domain = self._guess_domain(path.name)
            doc_id = hashlib.md5(text[:100].encode()).hexdigest()[:12]
            
            self.documents[doc_id] = GhostDocument(
                id=doc_id,
                domain=domain,
                text=text[:50000],  # cap for memory
                size_bytes=len(text),
            )
        
        self._save_ghost_corpus()
        print(f"Ghost Corpus: {len(self.documents)} docs separados do treino inicial")
    
    def _guess_domain(self, filename: str) -> str:
        name = filename.lower()
        if any(k in name for k in ['math', 'calculus', 'algebra', 'geometry', 'number']):
            return 'math'
        if any(k in name for k in ['physics', 'quantum', 'mechanics']):
            return 'physics'
        if any(k in name for k in ['finance', 'trading', 'market', 'risk']):
            return 'finance'
        if any(k in name for k in ['bio', 'cell', 'gene', 'dna', 'evolution']):
            return 'biology'
        if any(k in name for k in ['code', 'python', 'algorithm', 'program']):
            return 'code'
        return 'general'
    
    def _save_ghost_corpus(self):
        """Save ghost corpus metadata."""
        meta = {
            "total_docs": len(self.documents),
            "split_ratio": self.split_ratio,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "domains": {},
        }
        for doc in self.documents.values():
            meta["domains"][doc.domain] = meta["domains"].get(doc.domain, 0) + 1
        
        (self.ghost_dir / "ghost_meta.json").write_text(json.dumps(meta, indent=2))
    
    def _load_ghost_corpus(self):
        """Load existing ghost corpus."""
        meta = json.loads((self.ghost_dir / "ghost_meta.json").read_text())
        print(f"Ghost Corpus carregado: {meta['total_docs']} docs")
    
    def explore_cycle(self, batch_size: int = 10) -> List[str]:
        """One complete exploration cycle.
        
        Returns: list of synthetic training documents generated.
        """
        # 1. Curiosity: select what to explore
        docs = self.curiosity.select_next_batch(
            list(self.documents.values()), batch_size
        )
        
        if not docs:
            print("No unexplored documents remaining.")
            return []
        
        all_synthetic = []
        
        for doc in docs:
            # 2. Generate questions
            questions = self.questioner.generate_questions(doc, doc.domain)
            
            # 3. Verify answers
            answers = self.verifier.verify(doc)
            
            # 4. Generate synthetic training data
            synthetic = self.synthesizer.generate_for_document(doc)
            all_synthetic.extend(synthetic)
        
        # 5. Log evolution
        self.evolution_log.append({
            "cycle": len(self.evolution_log) + 1,
            "docs_explored": len(docs),
            "questions_generated": sum(len(d.questions_generated) for d in docs),
            "answers_verified": sum(1 for d in docs for a in d.verified_answers if a.get("confidence", 0) >= 0.5),
            "synthetic_docs": len(all_synthetic),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        return all_synthetic
    
    def stats(self) -> Dict:
        """Get ghost corpus statistics."""
        unexplored = sum(1 for d in self.documents.values() if d.status == ExplorationStatus.UNEXPLORED)
        verified = sum(1 for d in self.documents.values() if d.status == ExplorationStatus.VERIFIED)
        synthesized = sum(1 for d in self.documents.values() if d.status == ExplorationStatus.SYNTHESIZED)
        learned = sum(1 for d in self.documents.values() if d.status == ExplorationStatus.LEARNED)
        
        return {
            "total_ghost_docs": len(self.documents),
            "unexplored": unexplored,
            "verified": verified,
            "synthesized": synthesized,
            "learned": learned,
            "evolution_cycles": len(self.evolution_log),
            "total_synthetic_generated": sum(c["synthetic_docs"] for c in self.evolution_log),
            "domains": {},
        }


# ═══════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║  F51 GHOST CORPUS — AUTO-EVOLUTION ENGINE        ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  80% Treino Inicial → modelo nasce               ║")
    print("║  20% Fantasma → curiosidade → perguntas          ║")
    print("║  Wolfram/Code → verificação → novos dados        ║")
    print("║  Auto-treino → modelo supera dados originais     ║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    print("Módulos:")
    print("  CuriosityDrive  — escolhe O QUE explorar")
    print("  QuestionGenerator — cria PERGUNTAS sobre o desconhecido")
    print("  Verifier — confere respostas (Wolfram/Code)")
    print("  SyntheticDataGenerator — cria NOVOS dados de treino")
    print("  GhostCorpus — orquestrador completo")
    print()
    print("Integração:")
    print("  JEPA — prevê surpresa, guia curiosidade")
    print("  Ghost Token — experts se especializam nos novos dados")
    print("  Evolution Loop — módulos nascem/morrem por performance")
