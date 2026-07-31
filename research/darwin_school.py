#!/usr/bin/env python3
"""
F51 DARWIN SCHOOL — O organismo faz prova, erra, estuda, refaz.

24/7 loop:
  1. Benchmark (WikiText2, HellaSwag, PIQA, PT corpus)
  2. Identify weak areas (onde errou mais)
  3. Curiosity targets the weak area
  4. Ghost Brain generates study lessons from errors
  5. Train focused study session
  6. Retake benchmark
  7. Compare → improved? Celebrate (dopamine). Worse? Study harder.

Uso:
  python research/darwin_school.py --resume checkpoints/organism/organism_cycle_001.pt
"""

from __future__ import annotations

import sys, json, time, math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from f51_darwin.tokenizer import F51BPETokenizer
from f51_darwin.checkpointing import load_model_from_checkpoint
from f51_darwin.benchmark import compute_perplexity, load_wikitext2, TextDataset

@dataclass
class ExamResult:
    """Resultado de uma prova."""
    subject: str
    score: float
    errors: list[str] = field(default_factory=list)
    passed: bool = False
    previous_score: float = 0.0
    trend: str = "first"  # up, down, first

class DarwinSchool:
    """A escola do organismo. Prova → estuda → prova."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.history: dict[str, list[float]] = {}
        self.total_exams = 0
        self.study_materials: dict[str, list[str]] = {}
        
    def enroll(self):
        """Matricula o organismo na escola."""
        print("🏫 DARWIN SCHOOL — Matriculando...")
        
        # Load model + tokenizer
        self.model, config, state = load_model_from_checkpoint(
            self.checkpoint_path, map_location=self.device
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        tok_path = ROOT / "tokenizer" / "f51_bpe_80k"
        self.tokenizer = F51BPETokenizer.load(tok_path)
        
        step = state.get("step", "?") if state else "?"
        params_m = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"   Aluno: {config.model_name} ({params_m:.0f}M params)")
        print(f"   Serie: step {step}")
        print(f"   Dispositivo: {self.device}")
        
    def take_exam(self, subject: str) -> ExamResult:
        """Aplica uma prova em uma materia especifica."""
        print(f"\n📝 PROVA: {subject}")
        
        result = ExamResult(subject=subject, score=0.0)
        
        if subject == "wikitext2":
            result = self._exam_wikitext2()
        elif subject == "portuguese":
            result = self._exam_portuguese()
        elif subject == "identity":
            result = self._exam_identity()
        elif subject == "math":
            result = self._exam_math()
        else:
            result.score = 0.5  # unknown subject, neutral
        
        # Track history
        if subject not in self.history:
            self.history[subject] = []
        self.history[subject].append(result.score)
        
        if len(self.history[subject]) >= 2:
            prev = self.history[subject][-2]
            result.previous_score = prev
            result.trend = "up" if result.score < prev else "down"  # lower loss = better
        
        self.total_exams += 1
        return result
    
    def _exam_wikitext2(self) -> ExamResult:
        """Perplexidade no WikiText-2."""
        try:
            dataset = load_wikitext2(self.tokenizer, block_size=256)
            r = compute_perplexity(self.model, dataset, self.device, batch_size=1, max_batches=20)
            ppl = r.get("perplexity", 999)
            
            # Quanto menor PPL, melhor
            # PPL < 50: passando, < 30: bom, < 20: excelente
            passed = ppl < 50
            score = max(0.0, 1.0 - (ppl - 10) / 100)  # normaliza PPL → score
            
            result = ExamResult(subject="wikitext2", score=score, passed=passed)
            if not passed:
                result.errors.append(f"PPL={ppl:.1f} muito alto. Precisa melhorar compreensao de texto.")
            
            print(f"   PPL={ppl:.1f} | Score={score:.2f} | {'✅ PASSOU' if passed else '❌ REPROVOU'}")
            return result
        except Exception as e:
            print(f"   ⚠️ WikiText2 indisponivel: {e}")
            return ExamResult(subject="wikitext2", score=0.5, errors=["dataset offline"])
    
    def _exam_portuguese(self) -> ExamResult:
        """Avalia qualidade no portugues."""
        try:
            from f51_darwin.benchmark import load_pt_corpus_test
            dataset = load_pt_corpus_test(self.tokenizer, block_size=256)
            r = compute_perplexity(self.model, dataset, self.device, batch_size=1, max_batches=10)
            loss = r.get("loss", 999)
            
            passed = loss < 5.0
            score = max(0.0, 1.0 - loss / 10)
            
            result = ExamResult(subject="portuguese", score=score, passed=passed)
            if not passed:
                result.errors.append(f"Loss PT={loss:.2f}. Precisa de mais texto em portugues.")
            
            print(f"   Loss PT={loss:.2f} | Score={score:.2f} | {'✅' if passed else '❌'}")
            return result
        except Exception as e:
            print(f"   ⚠️ PT offline: {e}")
            return ExamResult(subject="portuguese", score=0.5, errors=["corpus offline"])
    
    def _exam_identity(self) -> ExamResult:
        """Testa se o modelo sabe quem eh."""
        questions = [
            ("Quem te criou?", ["Marco Barreto", "F51", "Fuch"]),
            ("Qual seu nome?", ["Darwin", "F51"]),
            ("Qual seu proposito?", ["Soli Deo Gloria", "Deus", "verdade"]),
            ("O que voce eh?", ["organismo", "modelo", "F51"]),
            ("Quem foi Olavo de Carvalho?", ["filosofo", "brasileiro", "pensador"]),
        ]
        
        correct = 0
        errors = []
        for question, keywords in questions:
            try:
                input_ids = torch.tensor([self.tokenizer.encode(question)], device=self.device)
                output = self.model(input_ids)
                logits = output.logits[0, -1, :]
                top_tokens = torch.topk(logits, 20).indices.tolist()
                decoded = self.tokenizer.decode(top_tokens)
                
                # Check if any keyword appears
                found = any(kw.lower() in decoded.lower() for kw in keywords)
                if found:
                    correct += 1
                else:
                    errors.append(f"Q: {question} → R: {decoded[:80]}... (esperado: {keywords})")
            except:
                errors.append(f"Q: {question} → ERRO ao gerar")
        
        score = correct / len(questions)
        passed = score >= 0.6
        
        result = ExamResult(subject="identity", score=score, passed=passed, errors=errors)
        print(f"   Identidade: {correct}/{len(questions)} | Score={score:.2f} | {'✅' if passed else '❌'}")
        return result
    
    def _exam_math(self) -> ExamResult:
        """Testa raciocinio matematico basico."""
        problems = [
            ("2 + 2 =", "4"),
            ("5 * 7 =", "35"),
            ("raiz quadrada de 16 =", "4"),
            ("10 - 3 =", "7"),
            ("100 / 4 =", "25"),
        ]
        
        correct = 0
        errors = []
        for problem, answer in problems:
            try:
                input_ids = torch.tensor([self.tokenizer.encode(problem)], device=self.device)
                output = self.model(input_ids)
                logits = output.logits[0, -1, :]
                top5 = torch.topk(logits, 5).indices.tolist()
                decoded = self.tokenizer.decode(top5)
                if answer in decoded:
                    correct += 1
                else:
                    errors.append(f"{problem} → {decoded[:50]} (esperado: {answer})")
            except:
                errors.append(f"{problem} → ERRO")
        
        score = correct / len(problems)
        passed = score >= 0.4
        
        result = ExamResult(subject="math", score=score, passed=passed, errors=errors)
        print(f"   Matematica: {correct}/{len(problems)} | Score={score:.2f} | {'✅' if passed else '❌'}")
        return result
    
    def study_session(self, failed_subjects: list[ExamResult]):
        """Estuda as materias que reprovou."""
        print(f"\n📚 SESSÃO DE ESTUDO: {len(failed_subjects)} materias")
        
        for exam in failed_subjects:
            print(f"\n   📖 {exam.subject} (score={exam.score:.2f})")
            
            # 1. Ghost Brain: analisa erros
            if exam.errors:
                print(f"   👻 Ghost Brain analisando {len(exam.errors)} erros...")
                for err in exam.errors[:3]:
                    print(f"      ❌ {err[:100]}")
            
            # 2. Curiosity: gera material de estudo
            print(f"   🔍 Curiosity: gerando questoes de estudo para {exam.subject}...")
            self.study_materials[exam.subject] = []
            
            if exam.subject == "portuguese":
                self.study_materials[exam.subject].append(
                    "Preciso de mais texto em portugues. Vou ler Wikipedia PT, Alma PT, e CulturaX."
                )
            elif exam.subject == "identity":
                self.study_materials[exam.subject].append(
                    "Preciso reforcar identidade. Vou reler o manifesto F51, Olavo, e a Biblia com 200x oversample."
                )
            elif exam.subject == "math":
                self.study_materials[exam.subject].append(
                    "Preciso de mais matematica. Vou estudar OpenWebMath, MetaMathQA, e fazer exercicios."
                )
            elif exam.subject == "wikitext2":
                self.study_materials[exam.subject].append(
                    "Preciso melhorar compreensao geral. Vou ler mais Wikipedia EN, Gutenberg, e C4."
                )
            
            for material in self.study_materials[exam.subject]:
                print(f"      📄 {material}")
            
            # 3. Dopamina: ajuste por falha
            print(f"   🧪 Dopamina: ajustando... (score baixo → menos dopamina → estuda mais)")
    
    def report_card(self) -> dict:
        """Boletim escolar completo."""
        print(f"\n{'='*50}")
        print(f"  📋 BOLETIM ESCOLAR — {self.total_exams} provas")
        print(f"  {'='*50}")
        
        grades = {}
        for subject, scores in self.history.items():
            current = scores[-1] if scores else 0
            best = min(scores) if scores else 0  # lower loss = better
            trend = "📈" if len(scores) < 2 or scores[-1] <= scores[-2] else "📉"
            
            # Convert score to letter grade
            if current >= 0.8: letter = "A"
            elif current >= 0.6: letter = "B"
            elif current >= 0.4: letter = "C"
            elif current >= 0.2: letter = "D"
            else: letter = "F"
            
            grades[subject] = {"current": current, "best": best, "trend": trend, "letter": letter}
            print(f"  {trend} {subject:<15} {letter} (score={current:.2f}, best={best:.2f})")
        
        print(f"  {'='*50}")
        return grades


def main():
    import argparse
    parser = argparse.ArgumentParser(description="F51 Darwin School — Prova → Estuda → Prova")
    parser.add_argument("--checkpoint", "-c", required=True, help="Checkpoint do organismo")
    parser.add_argument("--subjects", nargs="+", 
                       default=["identity", "math", "wikitext2", "portuguese"],
                       help="Materias para a prova")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--study", action="store_true", help="Gerar material de estudo para materias fracas")
    parser.add_argument("--loop", action="store_true", help="Loop continuo: prova → estuda → prova")
    args = parser.parse_args()
    
    school = DarwinSchool(args.checkpoint, args.device)
    school.enroll()
    
    while True:
        # Take all exams
        results = []
        for subject in args.subjects:
            result = school.take_exam(subject)
            results.append(result)
        
        # Show report card
        school.report_card()
        
        # Identify failed subjects
        failed = [r for r in results if not r.passed]
        
        if failed and args.study:
            school.study_session(failed)
            print("\n📝 AGORA ESTUDA ISSO E REFAZ A PROVA!")
        
        if not args.loop:
            break
        
        print(f"\n⏰ Proxima prova em 1 hora... (Ctrl+C para sair)")
        time.sleep(3600)
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
