"""
F51 Spider-Sense — O modelo SABE quando NÃO SABE.

Sentido Aranha: detector de confiança que avisa o Ghost Brain
quando o modelo está em território desconhecido.

Como funciona:
  1. Hidden states da última camada → Confidence Head
  2. Score 0-1: "quão certo estou desta resposta?"
  3. Score < 0.3 → 🚨 ARANHA PINICA → "Não sei, vou verificar!"
  4. Score > 0.7 → ✅ "Sei disso, pode confiar."

Treino: aprende com os PRÓPRIOS ERROS do modelo.
  - Se o modelo errou → deveria ter tido baixa confiança
  - Se o modelo acertou → deveria ter tido alta confiança

Integração:
  Ghost Brain → Spider-Sense → Wolfram Verify → Error Lesson → Auto-train
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SpiderSense(nn.Module):
    """Detector de confiança — o modelo sabe quando não sabe.

    Uma cabeça leve (2-layer MLP) em cima dos hidden states
    que prediz se a resposta do modelo está correta ou não.
    """

    def __init__(self, d_model: int = 384, hidden: int = 128):
        super().__init__()
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        # Estatísticas
        self.register_buffer('total_checks', torch.zeros(1))
        self.register_buffer('correct_tingles', torch.zeros(1))
        self.register_buffer('false_confidences', torch.zeros(1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Retorna score de confiança [0-1] para cada token.

        Args:
            hidden_states: [batch, seq, d_model] — saída da última camada

        Returns:
            confidence: [batch, seq] — 0 = não sei, 1 = certeza absoluta
        """
        return self.confidence_head(hidden_states).squeeze(-1)

    def should_verify(
        self,
        hidden_states: torch.Tensor,
        threshold: float = 0.3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decide quais tokens precisam de verificação externa.

        Args:
            hidden_states: [batch, seq, d_model]
            threshold: abaixo disso → verificar

        Returns:
            needs_verify: [batch, seq] bool — tokens que precisam de Wolfram
            confidence: [batch, seq] float — scores de confiança
        """
        confidence = self.forward(hidden_states)
        needs_verify = confidence < threshold
        return needs_verify, confidence

    def learn_from_outcome(
        self,
        hidden_states: torch.Tensor,
        was_correct: torch.Tensor,
    ) -> torch.Tensor:
        """Aprende com o resultado: acertou ou errou?

        Loss = |confidence - was_correct|²
        Se acertou com baixa confiança → punição (devia confiar mais)
        Se errou com alta confiança → punição (devia desconfiar)

        Args:
            hidden_states: [batch, seq, d_model]
            was_correct: [batch, seq] bool — o modelo acertou esse token?

        Returns:
            loss: scalar — spider-sense calibration loss
        """
        confidence = self.forward(hidden_states)
        loss = self.brier_loss(confidence, was_correct)

        # Atualiza estatísticas
        with torch.no_grad():
            low_conf = (confidence < 0.3)
            high_conf = (confidence > 0.7)
            false_high = (high_conf & ~was_correct).float().mean()  # confiante no erro
            true_low = (low_conf & ~was_correct).float().mean()     # desconfiou certo

            self.total_checks += 1
            self.correct_tingles += true_low
            self.false_confidences += false_high

        return loss

    @staticmethod
    def token_correctness(
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Detached next-token correctness labels for calibration."""
        if logits.ndim != 3 or labels.ndim != 2:
            raise ValueError(
                "logits and labels must have shapes [batch, sequence, vocab] "
                "and [batch, sequence]"
            )
        if logits.shape[:2] != labels.shape:
            raise ValueError("logits and labels must share batch/sequence shape")
        predictions = logits[:, :-1].detach().argmax(dim=-1)
        targets = labels[:, 1:].detach()
        return predictions.eq(targets)

    @staticmethod
    def brier_loss(
        confidence: torch.Tensor,
        was_correct: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pure Brier loss; correctness is evidence, never a gradient path."""
        if confidence.shape != was_correct.shape:
            raise ValueError("confidence and correctness must have identical shape")
        target = was_correct.detach().to(
            device=confidence.device,
            dtype=confidence.dtype,
        )
        squared_error = (confidence - target).square()
        if valid_mask is None:
            return squared_error.mean()
        if valid_mask.shape != confidence.shape:
            raise ValueError("valid_mask and confidence must have identical shape")
        valid = valid_mask.detach().to(
            device=confidence.device,
            dtype=torch.bool,
        )
        if not bool(valid.any().item()):
            return confidence.sum() * 0.0
        return squared_error.masked_select(valid).mean()

    def stats(self) -> dict:
        """Métricas do Sentido Aranha."""
        total = self.total_checks.item()
        if total == 0:
            return {"status": "no_data"}
        return {
            "total_checks": int(total),
            "tingle_accuracy": f"{self.correct_tingles.item() / max(total, 1):.1%}",
            "false_confidence": f"{self.false_confidences.item() / max(total, 1):.1%}",
            "status": "🟢 ativo" if self.correct_tingles.item() > self.false_confidences.item() else "🟡 calibrando",
        }

    def report(self, confidence: torch.Tensor) -> str:
        """Relatório legível do estado atual."""
        mean_conf = confidence.mean().item()
        low_pct = (confidence < 0.3).float().mean().item()
        high_pct = (confidence > 0.7).float().mean().item()

        if mean_conf < 0.3:
            mood = "🚨 Inseguro — preciso estudar mais"
        elif mean_conf > 0.7:
            mood = "💪 Confiante — tô dominando"
        else:
            mood = "🤔 Na dúvida — melhor verificar"

        return (
            f"{mood}\n"
            f"  Confiança média: {mean_conf:.2f}\n"
            f"  Baixa confiança: {low_pct:.0%} dos tokens\n"
            f"  Alta confiança:  {high_pct:.0%} dos tokens"
        )


# ═══════════════════════════════════════════════════
# INTEGRAÇÃO COM GHOST BRAIN
# ═══════════════════════════════════════════════════

class SpiderGhostBrain:
    """Sentido Aranha + Ghost Brain = modelo que SABE quando não sabe.

    Pipeline:
      1. Modelo gera resposta
      2. Spider-Sense avalia confiança
      3. Baixa confiança → Ghost Brain explora
      4. Ghost Brain verifica (Wolfram/Code)
      5. Se errou → Error Lesson → dados sintéticos
      6. Spider-Sense aprende com o resultado
      7. Repete — cada ciclo melhora o Sentido Aranha
    """

    def __init__(
        self,
        d_model: int = 384,
        spider_threshold: float = 0.3,
        ghost_brain=None,
    ):
        self.spider = SpiderSense(d_model)
        self.threshold = spider_threshold
        self.ghost = ghost_brain
        self.exploration_log: list[dict] = []

    def analyze_response(
        self,
        hidden_states: torch.Tensor,
        model_predictions: torch.Tensor,
    ) -> dict:
        """Analisa a resposta do modelo e decide próximos passos.

        Returns:
            dict com:
                - needs_exploration: bool — iniciar Ghost Brain?
                - confidence_score: float
                - uncertain_tokens: int — quantos tokens o modelo não confia
                - action: str — 'trust', 'verify', 'explore'
        """
        needs_verify, confidence = self.spider.should_verify(
            hidden_states, self.threshold
        )

        uncertain_count = needs_verify.sum().item()
        mean_conf = confidence.mean().item()

        if uncertain_count == 0:
            action = "trust"
        elif mean_conf > 0.5:
            action = "verify"  # confiante o suficiente pra verificar sozinho
        else:
            action = "explore"  # precisa de ajuda externa

        result = {
            "needs_exploration": bool(needs_verify.any()),
            "confidence_score": mean_conf,
            "uncertain_tokens": uncertain_count,
            "action": action,
            "spider_report": self.spider.report(confidence),
        }

        self.exploration_log.append(result)
        return result

    def train_spider(
        self,
        hidden_states: torch.Tensor,
        model_predictions: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> torch.Tensor:
        """Treina o Sentido Aranha: compara predição com verdade.

        Args:
            hidden_states: estados ocultos do modelo
            model_predictions: tokens que o modelo previu
            ground_truth: tokens corretos (do corpus ou Wolfram)

        Returns:
            loss para backpropagation
        """
        was_correct = (model_predictions == ground_truth)
        return self.spider.learn_from_outcome(hidden_states, was_correct)

    def stats(self) -> dict:
        return {
            "spider": self.spider.stats(),
            "explorations": len(self.exploration_log),
            "last_action": self.exploration_log[-1]["action"] if self.exploration_log else "none",
        }


# ═══════════════════════════════════════════════════
# SPIDER RAM — persistent CPU-resident danger memory
# ═══════════════════════════════════════════════════


class SpiderRAM:
    """Persistent CPU-resident memory of Spider-Sense calibration signals.

    Survives across training steps without touching GPU VRAM.  The causal
    bus adapters fire *before* the forward pass, so ``spider.danger`` must
    come from the *previous* step.  SpiderRAM stores a short rolling window
    and exposes a smoothed danger level to the adapter context.

    Integração:
      1. After forward pass:  spider_ram.update(output)
      2. Before decide():     context includes spider_ram.state()
      3. SpiderCalibrationAdapter reads danger from context
    """

    def __init__(self, window_size: int = 8, danger_smooth: float = 0.3):
        self.window_size = max(1, int(window_size))
        self.danger_smooth = float(danger_smooth)
        self._spider_loss: list[float] = []
        self._danger: list[float] = []
        self._confidence: list[float] = []
        self._steps_seen: int = 0

    def update(
        self,
        *,
        spider_loss: float | None = None,
        spider_danger: float | None = None,
        confidence_mean: float | None = None,
    ) -> None:
        """Push one step of spider telemetry into the rolling window."""
        self._steps_seen += 1
        if spider_loss is not None:
            self._spider_loss.append(float(spider_loss))
            if len(self._spider_loss) > self.window_size:
                self._spider_loss.pop(0)
        if spider_danger is not None:
            self._danger.append(float(spider_danger))
            if len(self._danger) > self.window_size:
                self._danger.pop(0)
        if confidence_mean is not None:
            self._confidence.append(float(confidence_mean))
            if len(self._confidence) > self.window_size:
                self._confidence.pop(0)

    def update_from_output(self, output: object) -> None:
        """Convenience: extract spider telemetry from a DarwinXOutput."""
        spider_loss = None
        raw_spider = getattr(output, "spider_loss", None)
        if hasattr(raw_spider, "detach"):
            spider_loss = float(raw_spider.detach().cpu())

        spider_danger = None
        spider_conf = getattr(output, "spider_confidence", None)
        if spider_conf is not None and hasattr(spider_conf, "mean"):
            spider_danger = float(
                (1.0 - spider_conf.detach().mean()).cpu()
            )

        confidence_mean = None
        if spider_conf is not None and hasattr(spider_conf, "mean"):
            confidence_mean = float(spider_conf.detach().mean().cpu())

        self.update(
            spider_loss=spider_loss,
            spider_danger=spider_danger,
            confidence_mean=confidence_mean,
        )

    @property
    def danger(self) -> float:
        """Exponentially-smoothed danger level [0, 1] from recent window."""
        if not self._danger:
            return 0.0
        alpha = self.danger_smooth
        smoothed = self._danger[0]
        for value in self._danger[1:]:
            smoothed = alpha * value + (1.0 - alpha) * smoothed
        return smoothed

    @property
    def avg_spider_loss(self) -> float:
        if not self._spider_loss:
            return 0.0
        return sum(self._spider_loss) / len(self._spider_loss)

    @property
    def avg_confidence(self) -> float:
        if not self._confidence:
            return 0.5
        return sum(self._confidence) / len(self._confidence)

    @property
    def alive(self) -> bool:
        """Whether we have any real data (at least one update)."""
        return self._steps_seen > 0

    def state(self) -> dict:
        """Plain dict for injection into the causal bus context."""
        return {
            "spider.danger": self.danger,
            "spider.avg_loss": self.avg_spider_loss,
            "spider.avg_confidence": self.avg_confidence,
            "spider.window_size": self.window_size,
            "spider.steps_seen": self._steps_seen,
            "spider.alive": self.alive,
        }

    def reset(self) -> None:
        self._spider_loss.clear()
        self._danger.clear()
        self._confidence.clear()
        self._steps_seen = 0


# ═══════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔══════════════════════════════════════════╗")
    print("║  🕷️  F51 SPIDER-SENSE                     ║")
    print("║  O modelo SABE quando NÃO SABE            ║")
    print("╚══════════════════════════════════════════╝")
    print()

    # Simulação: modelo 33M, hidden states aleatórios
    d_model = 384
    spider = SpiderSense(d_model)

    # Cenário 1: Modelo confiante (hidden states "fortes")
    strong_hidden = torch.randn(2, 16, d_model) * 2.0
    conf1 = spider(strong_hidden)
    print("Cenário 1: Domínio conhecido (ex: matemática)")
    print(spider.report(conf1))
    print()

    # Cenário 2: Modelo inseguro (hidden states "fracos")
    weak_hidden = torch.randn(2, 16, d_model) * 0.2
    conf2 = spider(weak_hidden)
    print("Cenário 2: Domínio novo (ex: odontologia)")
    print(spider.report(conf2))
    print()

    # Treino: modelo acertou → spider deve ter alta confiança
    loss = spider.learn_from_outcome(
        strong_hidden,
        was_correct=torch.ones(2, 16, dtype=torch.bool),
    )
    print(f"Treino (acerto): loss={loss.item():.4f}")
    print(f"Stats: {spider.stats()}")
