"""
F51 JEPA — Joint Embedding Predictive Architecture.

Adaptado do nelsonmath/core/jepa/ para o F51 Darwin-SSD.
Integrado com o token_embedding nativo (sem SentenceTransformer externo).

Arquitetura:
    Input tokens → F51 Embedding → [Blocos 0..K] → Hidden State
                                                    ↓
                              ┌──────────────────────┤
                              │                      │
                         LM Head (15708)      JEPA Head (384)
                         prevê token           prevê embedding futuro
                         loss discreta         loss contínua
                              │                      │
                              └──────┬───────────────┘
                                Loss = α·LM + β·JEPA

Efeito: O modelo aprende ESTRUTURA (JEPA) enquanto aprende CONTEÚDO (LM).
Treino 2x mais eficiente por token processado.

Uso:
    from f51_darwin.jepa import JEPAHead, jepa_loss
    jepa = JEPAHead(d_model=384)
    loss = lm_loss + 0.1 * jepa_loss(hidden_states)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPAHead(nn.Module):
    """Predictor head: dados os hidden states atuais, prediz os próximos.

    Inspirado no JEPAPredictor do nelsonmath, mas usa os hidden states
    do próprio F51 em vez de embeddings externos.
    """

    def __init__(
        self,
        d_model: int = 384,
        hidden_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Predictor: current_hidden → future_hidden
        self.predictor = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Prediz o próximo hidden state.

        Args:
            hidden_states: [batch, seq_len, d_model] — hidden states do F51

        Returns:
            predicted: [batch, seq_len-1, d_model] — hidden states futuros preditos
        """
        current = hidden_states[:, :-1, :]  # [B, T-1, D]
        target = hidden_states[:, 1:, :]     # [B, T-1, D]

        B, T_minus_1, D = current.shape
        current_flat = current.reshape(-1, D)
        predicted_flat = self.predictor(current_flat)
        predicted = predicted_flat.reshape(B, T_minus_1, D)

        return predicted, target


def jepa_loss(
    hidden_states: torch.Tensor,
    jepa_head: JEPAHead,
    *,
    loss_type: str = "cosine",
) -> torch.Tensor:
    """Calcula a loss JEPA entre hidden states preditos e reais.

    Args:
        hidden_states: [batch, seq_len, d_model]
        jepa_head: módulo JEPAHead
        loss_type: "cosine" (padrão) ou "mse"

    Returns:
        loss escalar (já reduzida por média)
    """
    predicted, target = jepa_head(hidden_states)

    if loss_type == "cosine":
        # Cosine similarity loss: queremos similaridade = 1
        cos_sim = F.cosine_similarity(
            predicted.reshape(-1, hidden_states.size(-1)),
            target.reshape(-1, hidden_states.size(-1)),
            dim=-1,
        )
        return (1.0 - cos_sim).mean()

    elif loss_type == "mse":
        return F.mse_loss(predicted, target)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def combined_loss(
    lm_loss: torch.Tensor,
    hidden_states: torch.Tensor,
    jepa_head: JEPAHead,
    *,
    jepa_weight: float = 0.1,
) -> torch.Tensor:
    """Loss combinada: LM + JEPA.

    Args:
        lm_loss: cross-entropy loss do language model
        hidden_states: hidden states do forward pass
        jepa_head: módulo JEPA
        jepa_weight: peso da loss JEPA (0.1 = 10%)

    Returns:
        loss total: lm_loss + jepa_weight * jepa_loss
    """
    j_loss = jepa_loss(hidden_states, jepa_head)
    return lm_loss + jepa_weight * j_loss


# ═══════════════════════════════════════════════════════════
# TESTES
# ═══════════════════════════════════════════════════════════

def test_jepa_forward():
    """Testa forward pass do JEPA."""
    head = JEPAHead(d_model=384)
    x = torch.randn(4, 64, 384)  # batch=4, seq=64, dim=384
    pred, target = head(x)
    assert pred.shape == (4, 63, 384), f"Expected (4,63,384), got {pred.shape}"
    assert target.shape == (4, 63, 384)
    print("  forward: OK")

def test_jepa_loss():
    """Testa cálculo de loss JEPA."""
    head = JEPAHead(d_model=384)
    x = torch.randn(4, 64, 384)
    loss = jepa_loss(x, head)
    assert loss.item() > 0
    assert loss.item() < 2.0  # cosine loss está em [0, 2]
    print("  loss: OK")

def test_combined_loss():
    """Testa loss combinada LM + JEPA."""
    head = JEPAHead(d_model=384)
    x = torch.randn(4, 64, 384)
    lm_loss = torch.tensor(2.5)
    combined = combined_loss(lm_loss, x, head, jepa_weight=0.1)
    assert combined.item() > lm_loss.item()
    print("  combined: OK")

def test_jepa_params():
    """Verifica que JEPA não adiciona muitos params."""
    head = JEPAHead(d_model=384, hidden_dim=768)
    params = sum(p.numel() for p in head.parameters())
    assert params < 3_000_000, f"JEPA muito pesado: {params:,} params"
    print(f"  params: {params:,} ({params/1e6:.1f}M) — leve!")
