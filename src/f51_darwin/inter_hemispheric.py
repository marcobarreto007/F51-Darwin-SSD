"""
F51 INTER-HEMISPHERIC SYSTEM — Especialização e Integração
===========================================================

Biologia: O cérebro humano tem dois hemisférios especializados:
- Esquerdo: linguagem, lógica, detalhes
- Direito: espaço, emoção, contexto global

Corpo Caloso: integra informação entre hemisférios via comissuras.

Arquitetura:
- LeftHemisphere: especializado em linguagem/lógica
- RightHemisphere: especializado em espaço/contexto
- CorpusCallosum: cross-attention integration
- HemisphericSpecialization: treinamento especializado

Paper inspiração:
- "The divided brain" (Gazzaniga)
- "Lateralization in the brain" (Corballis)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HemisphereConfig:
    """Configuração do sistema inter-hemisférico."""
    d_model: int = 2048
    n_heads: int = 8
    dropout: float = 0.0
    left_specialization: str = "language"  # language, logic, detail
    right_specialization: str = "spatial"  # spatial, emotion, context
    integration_method: str = "cross_attention"  # cross_attention, concat, gating


class LeftHemisphere(nn.Module):
    """Hemisfério Esquerdo: especializado em linguagem e lógica.

    Processa informações sequenciais, gramática, detalhes locais.
    """

    def __init__(self, config: HemisphereConfig):
        super().__init__()
        self.config = config
        self.specialization = config.left_specialization

        # Language/logic processing
        self.processor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model)
        )

        # Local detail attention
        self.local_attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Processa com especialização esquerda."""
        batch, seq_len, d_model = x.shape

        # Processamento base
        processed = self.processor(x)

        # Local attention (foco em detalhes locais)
        attn_out, _ = self.local_attention(processed, processed, processed, attn_mask=mask)

        return attn_out


class RightHemisphere(nn.Module):
    """Hemisfério Direito: especializado em espaço e contexto.

    Processa informações espaciais, contexto global, emoções.
    """

    def __init__(self, config: HemisphereConfig):
        super().__init__()
        self.config = config
        self.specialization = config.right_specialization

        # Spatial/context processing
        self.processor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model)
        )

        # Global context attention
        self.global_attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Processa com especialização direita."""
        batch, seq_len, d_model = x.shape

        # Processamento base
        processed = self.processor(x)

        # Global attention (foco em contexto)
        attn_out, _ = self.global_attention(processed, processed, processed, attn_mask=mask)

        return attn_out


class CorpusCallosum(nn.Module):
    """Corpo Caloso: integração entre hemisférios.

    Permite que os hemisférios compartilhem informação.
    """

    def __init__(self, config: HemisphereConfig):
        super().__init__()
        self.config = config

        if config.integration_method == "cross_attention":
            # Cross-attention: L attends to R, R attends to L
            self.left_to_right = nn.MultiheadAttention(
                config.d_model,
                config.n_heads,
                dropout=config.dropout,
                batch_first=True
            )
            self.right_to_left = nn.MultiheadAttention(
                config.d_model,
                config.n_heads,
                dropout=config.dropout,
                batch_first=True
            )

            # Integration projection
            self.integration = nn.Sequential(
                nn.Linear(config.d_model * 2, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.d_model)
            )

        elif config.integration_method == "gating":
            # Gating mechanism
            self.gate = nn.Linear(config.d_model * 2, 2)  # pesos L e R

    def forward(
        self,
        left_output: torch.Tensor,
        right_output: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Integra saídas dos hemisférios.

        Args:
            left_output: [batch, seq_len, d_model]
            right_output: [batch, seq_len, d_model]
            mask: máscara causal opcional [seq_len, seq_len] (bool, True =
                bloqueado), aplicada à cross-attention entre hemisférios.

        Returns:
            integrated_left: [batch, seq_len, d_model]
            integrated_right: [batch, seq_len, d_model]
            stats: estatísticas de integração
        """
        stats = {}

        if self.config.integration_method == "cross_attention":
            # Left attends to Right
            left_enhanced, left_weights = self.left_to_right(
                left_output, right_output, right_output, attn_mask=mask
            )

            # Right attends to Left
            right_enhanced, right_weights = self.right_to_left(
                right_output, left_output, left_output, attn_mask=mask
            )

            # Integração combinada
            combined = torch.cat([left_enhanced, right_enhanced], dim=-1)
            integrated = self.integration(combined)

            stats['cross_attention_entropy'] = (
                -(left_weights * torch.log(left_weights + 1e-10)).sum(dim=-1).mean().item()
            )

            # Cada hemisfério mantém sua própria representação realçada somada
            # à integração partilhada — evita alimentar output_proj com o
            # mesmo tensor duplicado (integrated, integrated).
            return left_enhanced + integrated, right_enhanced + integrated, stats

        elif self.config.integration_method == "gating":
            # Gating: decide quanto de cada hemisfério usar
            combined = torch.cat([left_output, right_output], dim=-1)
            gates = F.softmax(self.gate(combined), dim=-1)  # [B, S, 2]

            left_gate = gates[..., 0:1]
            right_gate = gates[..., 1:2]

            integrated_left = left_output * left_gate + right_output * (1 - left_gate)
            integrated_right = right_output * right_gate + left_output * (1 - right_gate)

            stats['left_gate_mean'] = left_gate.mean().item()
            stats['right_gate_mean'] = right_gate.mean().item()

            return integrated_left, integrated_right, stats

        else:  # concat
            # Simples concatenação
            integrated = torch.cat([left_output, right_output], dim=-1)
            # Split de volta para manter shape
            integrated_left = integrated[..., :self.config.d_model]
            integrated_right = integrated[..., self.config.d_model:]

            return integrated_left, integrated_right, stats


class InterHemisphericSystem(nn.Module):
    """Sistema completo inter-hemisférico.

    Coordenada os dois hemisférios e sua integração.
    """

    def __init__(self, config: HemisphereConfig):
        super().__init__()
        self.config = config

        self.left = LeftHemisphere(config)
        self.right = RightHemisphere(config)
        self.callosum = CorpusCallosum(config)

        # Output integration
        self.output_proj = nn.Linear(config.d_model * 2, config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass completo.

        Args:
            x: [batch, seq_len, d_model]
            mask: attention mask opcional. Quando omitida, uma máscara
                causal é construída automaticamente — este módulo roda
                logo após o embedding, antes do stack causal principal, e
                tanto a self-attention de cada hemisfério quanto a
                cross-attention do corpo caloso vazam posições futuras sem
                ela.

        Returns:
            [batch, seq_len, d_model] saída integrada
        """
        if mask is None:
            seq_len = x.size(1)
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            )

        # Processamento paralelo
        left_out = self.left(x, mask)
        right_out = self.right(x, mask)

        # Integração via corpo caloso
        left_int, right_int, stats = self.callosum(left_out, right_out, mask)

        # Combina saídas
        combined = torch.cat([left_int, right_int], dim=-1)
        output = self.output_proj(combined)

        return output

    def get_lateralization_stats(
        self,
        x: torch.Tensor
    ) -> dict[str, Any]:
        """Retorna estatísticas de lateralização.

        Mede quanto cada hemisfério contribuiu.
        """
        with torch.no_grad():
            left_out = self.left(x)
            right_out = self.right(x)

            left_norm = left_out.norm(dim=-1).mean().item()
            right_norm = right_out.norm(dim=-1).mean().item()

            total = left_norm + right_norm + 1e-8

            return {
                'left_contribution': left_norm / total,
                'right_contribution': right_norm / total,
                'lateralization_index': (left_norm - right_norm) / total
            }


# ═══════════════════════ TESTE ═══════════════════════

def _test_inter_hemispheric():
    """Smoke test do sistema Inter-Hemispheric."""
    print("Testing Inter-Hemispheric System...")

    config = HemisphereConfig(d_model=256, n_heads=4)

    # Test 1: Left Hemisphere
    left = LeftHemisphere(config)
    x = torch.randn(2, 8, 256)
    left_out = left(x)
    assert left_out.shape == (2, 8, 256)
    print("✓ Test 1: Left hemisphere")

    # Test 2: Right Hemisphere
    right = RightHemisphere(config)
    right_out = right(x)
    assert right_out.shape == (2, 8, 256)
    print("✓ Test 2: Right hemisphere")

    # Test 3: Corpus Callosum integration
    callosum = CorpusCallosum(config)
    left_int, right_int, stats = callosum(left_out, right_out)
    assert left_int.shape == (2, 8, 256)
    assert right_int.shape == (2, 8, 256)
    print("✓ Test 3: Corpus callosum integration")

    # Test 4: Full system
    system = InterHemisphericSystem(config)
    output = system(x)
    assert output.shape == (2, 8, 256)
    print("✓ Test 4: Full system forward")

    # Test 5: Lateralization stats
    lat_stats = system.get_lateralization_stats(x)
    assert 'left_contribution' in lat_stats
    assert 'right_contribution' in lat_stats
    assert 'lateralization_index' in lat_stats
    print("✓ Test 5: Lateralization stats")

    # Test 6: Different integration methods
    config_gating = HemisphereConfig(d_model=256, integration_method="gating")
    callosum_gating = CorpusCallosum(config_gating)
    left_g, right_g, stats_g = callosum_gating(left_out, right_out)
    assert 'left_gate_mean' in stats_g
    assert 'right_gate_mean' in stats_g
    print("✓ Test 6: Gating integration method")

    # Test 7: Training mode
    system.train()
    output_train = system(x)
    assert output_train.requires_grad
    print("✓ Test 7: Training mode")

    # Test 8: Specialization strings
    assert system.left.specialization == config.left_specialization
    assert system.right.specialization == config.right_specialization
    print("✓ Test 8: Specialization attributes")

    print("\n✅ All Inter-Hemispheric System tests passed!")


if __name__ == "__main__":
    _test_inter_hemispheric()
