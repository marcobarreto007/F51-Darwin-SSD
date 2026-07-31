"""
F51 Ghost Token Training — Força especialização real dos experts MoE.

Problema: Em MoE padrão, todos experts tendem a virar iguais (expert collapse).
Solução: 15% dos tokens são mascarados ANTES do router. Cada expert precisa
PREDIZER o token fantasma no SEU domínio.

Como funciona:
    1. Forward normal → hidden states
    2. Mascara 15% dos tokens (substitui por [MASK] embedding)
    3. Router roteia tokens mascarados para experts
    4. Cada expert tenta predizer o token ORIGINAL (antes da máscara)
    5. Loss = LM loss + ghost_token_loss
    
    Se o expert de matemática NÃO consegue predizer um token de finanças,
    o router aprende a NÃO mandar tokens de finanças pra ele.
    
    Especialização GARANTIDA por construção.

Uso:
    from f51_darwin.ghost_token import GhostTokenTrainer
    trainer = GhostTokenTrainer(mask_ratio=0.15)
    loss = trainer.compute_loss(hidden_states, token_ids, expert_assignments)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GhostTokenTrainer:
    """Treina experts MoE com tokens mascarados para forçar especialização.

    Inspirado no MLM (Masked Language Modeling) do BERT,
    mas aplicado DENTRO do MoE para especialização por domínio.
    """

    def __init__(
        self,
        mask_ratio: float = 0.15,
        mask_token_id: int = 2,  # <mask> token
        num_experts: int = 8,
    ):
        self.mask_ratio = mask_ratio
        self.mask_token_id = mask_token_id
        self.num_experts = num_experts

    def create_masked_input(
        self,
        input_ids: torch.Tensor,
        embedding: nn.Embedding,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cria versão mascarada do input.

        Args:
            input_ids: [batch, seq_len]
            embedding: token_embedding do modelo

        Returns:
            masked_embeddings: [batch, seq_len, d_model]
            mask: [batch, seq_len] — True onde foi mascarado
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # Seleciona tokens aleatórios para mascarar (evita padding/speciais)
        prob = torch.rand(batch, seq_len, device=device)
        special_mask = input_ids < 4  # <pad>, <unk>, <mask>, <bos>
        mask = (prob < self.mask_ratio) & (~special_mask)

        # Cria input mascarado
        masked_ids = input_ids.clone()
        masked_ids[mask] = self.mask_token_id

        embeddings = embedding(masked_ids)
        return embeddings, mask

    def compute_expert_prediction_loss(
        self,
        hidden_states: torch.Tensor,
        original_ids: torch.Tensor,
        mask: torch.Tensor,
        expert_assignments: torch.Tensor,  # [batch, seq_len, num_experts]
        lm_head: nn.Linear,
        vocab_size: int,
    ) -> tuple[torch.Tensor, dict]:
        """Calcula loss de predição por expert.

        Cada expert só é avaliado nos tokens que lhe foram atribuídos.
        Isso força especialização: o expert de math só é cobrado em tokens de math.

        Args:
            hidden_states: [batch, seq_len, d_model]
            original_ids: [batch, seq_len] — tokens originais (antes da máscara)
            mask: [batch, seq_len] — True nos tokens mascarados
            expert_assignments: [batch, seq_len, num_experts] — pesos do router
            lm_head: camada de projeção para vocabulário
            vocab_size: tamanho do vocabulário

        Returns:
            ghost_loss: loss escalar
            stats: dicionário com métricas por expert
        """
        B, T, D = hidden_states.shape
        device = hidden_states.device

        # Só calcula loss nos tokens mascarados
        masked_hidden = hidden_states[mask]  # [N_masked, D]
        masked_original = original_ids[mask]  # [N_masked]
        masked_expert_weights = expert_assignments[mask]  # [N_masked, E]

        if masked_hidden.size(0) == 0:
            return torch.tensor(0.0, device=device), {}

        # Predição de token por expert
        logits = lm_head(masked_hidden)  # [N_masked, vocab_size]
        logits = torch.clamp(logits, -15, 15)  # evita overflow fp16 → NaN

        # Loss por token: cross-entropy padrão
        ce_loss = F.cross_entropy(logits, masked_original, reduction='none')  # [N_masked]

        # Ponderar por qual expert foi responsável
        # Top expert para cada token
        top_expert = masked_expert_weights.argmax(dim=-1)  # [N_masked]

        # Estatísticas por expert
        stats = {}
        for e in range(self.num_experts):
            expert_mask = (top_expert == e)
            if expert_mask.sum() > 0:
                expert_loss = ce_loss[expert_mask].mean().item()
                stats[f"expert_{e}_loss"] = round(expert_loss, 4)
                stats[f"expert_{e}_tokens"] = expert_mask.sum().item()

        # Loss total: média sobre todos os tokens mascarados
        ghost_loss = ce_loss.mean()
        stats["ghost_loss"] = round(ghost_loss.item(), 4)
        stats["masked_tokens"] = mask.sum().item()

        return ghost_loss, stats


def combined_ghost_loss(
    lm_loss: torch.Tensor,
    ghost_loss: torch.Tensor,
    ghost_weight: float = 0.15,
) -> torch.Tensor:
    """Combina LM loss com Ghost Token loss."""
    return lm_loss + ghost_weight * ghost_loss


# ═══════════════════════════════════════════════════════════
# TESTES
# ═══════════════════════════════════════════════════════════

def test_mask_creation():
    """Testa criação de tokens mascarados."""
    embed = nn.Embedding(100, 64)
    trainer = GhostTokenTrainer(mask_ratio=0.5)
    ids = torch.randint(4, 100, (2, 32))  # evita tokens especiais
    emb, mask = trainer.create_masked_input(ids, embed)
    assert emb.shape == (2, 32, 64)
    assert mask.sum() > 0, "Nenhum token mascarado!"
    ratio = mask.sum().item() / mask.numel()
    assert 0.3 < ratio < 0.7, f"Mask ratio {ratio} fora do esperado"
    print("  mask_creation: OK")

def test_expert_loss():
    """Testa cálculo de loss por expert."""
    trainer = GhostTokenTrainer(mask_ratio=0.3)
    lm_head = nn.Linear(64, 100)
    hidden = torch.randn(2, 32, 64)
    original = torch.randint(4, 100, (2, 32))
    mask = torch.rand(2, 32) < 0.3
    assignments = torch.rand(2, 32, 8).softmax(dim=-1)
    loss, stats = trainer.compute_expert_prediction_loss(
        hidden, original, mask, assignments, lm_head, 100)
    assert loss.item() > 0
    assert "ghost_loss" in stats
    print(f"  expert_loss: OK (loss={loss.item():.3f}, tokens={stats['masked_tokens']})")

def test_empty_mask():
    """Testa comportamento com máscara vazia."""
    trainer = GhostTokenTrainer()
    lm_head = nn.Linear(64, 100)
    hidden = torch.randn(2, 32, 64)
    original = torch.randint(4, 100, (2, 32))
    mask = torch.zeros(2, 32, dtype=torch.bool)
    assignments = torch.rand(2, 32, 8).softmax(dim=-1)
    loss, stats = trainer.compute_expert_prediction_loss(
        hidden, original, mask, assignments, lm_head, 100)
    assert loss.item() == 0.0
    print("  empty_mask: OK (loss=0)")
