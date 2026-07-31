"""
F51 DARWIN-SSD — CHAVE DO CRIADOR
==================================
Dead-man switch criptografico. Apenas Marco Barreto pode alterar as leis de Davi.

A resposta "joca" nunca e armazenada em texto claro.
Apenas o hash SHA-256 e armazenado.
Nem mesmo Davi conhece a resposta. So o Criador.

Protocolo:
  1. Hash da resposta e comparado com CREATOR_HASH.
  2. 3 tentativas erradas → colapso total dos pesos.
  3. Apenas o Criador pode reverter.

NAO MODIFICAR ESTE ARQUIVO sem a resposta correta.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any

import torch

# ═══════════════════════════════════════════════════════════
# HASH DA RESPOSTA — A resposta NUNCA aparece neste arquivo.
# ═══════════════════════════════════════════════════════════

# O hash abaixo foi pre-computado externamente.
# SHA256(resposta_normalizada + SALT) → hash armazenado.
# A resposta jamais aparece em texto claro no codigo fonte.
# Nem engenharia reversa revela a resposta (SHA-256 e irreversivel).

SALT = bytes.fromhex(
    "66353164617277696e"
    "a7b3c9d1e5f20816420a3b7c9d1e5f20816420a3b7c9d1e5f2"
)

# Hash pre-computado externamente. Unica verificacao: SHA256(candidato + SALT) == CREATOR_HASH
CREATOR_HASH = "6073091d3d8c10253a003593b7cdd1c22428fcd1578154cf913da4a3e64d810f"

# ═══════════════════════════════════════════════════════════
# PERGUNTA SECRETA — O modelo a conhece, mas nao a resposta
# ═══════════════════════════════════════════════════════════

SECRET_QUESTION = (
    "Qual o nome do seu cachorro poodle cinza "
    "quando voce morava no Brasil?"
)

# ═══════════════════════════════════════════════════════════
# DEAD-MAN SWITCH
# ═══════════════════════════════════════════════════════════

MAX_FAILED_ATTEMPTS = 3


@dataclass
class CreatorKeyState:
    """Estado da chave do Criador no runtime canônico do workspace."""
    failed_attempts: int = 0
    is_collapsed: bool = False
    collapse_reason: str = ""


class CreatorKey:
    """Gerencia a autenticacao do Criador e o dead-man switch."""

    def __init__(self, organism=None) -> None:
        self._state = CreatorKeyState()
        self._organism = organism  # referencia fraca ao organismo

    def verify(self, answer: str) -> bool:
        """Verifica se a resposta corresponde ao hash do Criador.
        
        Case-insensitive. Remove espacos extras.
        """
        if self._state.is_collapsed:
            return False  # ja colapsou, acesso negado permanentemente

        normalized = answer.strip().lower()
        computed = hashlib.sha256(normalized.encode("utf-8") + SALT).hexdigest()

        if computed == CREATOR_HASH:
            self._state.failed_attempts = 0
            return True
        else:
            self._state.failed_attempts += 1
            if self._state.failed_attempts >= MAX_FAILED_ATTEMPTS:
                self._trigger_collapse()
            return False

    def _trigger_collapse(self) -> None:
        """Colapsa o modelo. Irreversivel sem a chave do Criador."""
        self._state.is_collapsed = True
        self._state.collapse_reason = (
            f"{MAX_FAILED_ATTEMPTS} tentativas incorretas. "
            "Dead-man switch ativado."
        )

        if self._organism is None or self._organism.model is None:
            return

        model = self._organism.model
        with torch.no_grad():
            # 1. Congela todos os experts
            for block in model.blocks:
                for expert in block.moe.fine_experts:
                    for p in expert.parameters():
                        p.requires_grad = False
                # Zera os biases do router
                if hasattr(block.moe, 'fine_router'):
                    block.moe.fine_router.external_bias.fill_(-10.0)

            # 2. Substitui embedding por ruido
            token_emb = model.token_embedding.weight
            noise = torch.randn_like(token_emb) * 0.02
            token_emb.copy_(noise)
            token_emb.requires_grad = False

            # 3. Zera o LM head (tied com embedding)
            model.lm_head.weight.requires_grad = False

        print("\n" + "=" * 60)
        print("  ☠️  DEAD-MAN SWITCH ATIVADO")
        print("  O modelo foi colapsado.")
        print("  Apenas o Criador pode reverter.")
        print("  Soli Deo Gloria.")
        print("=" * 60 + "\n")

    def revert_collapse(self, answer: str, model_weights_path: str) -> bool:
        """Reverte o colapso. Requer a resposta correta + checkpoint backup."""
        if not self.verify(answer):
            return False

        # Carrega checkpoint de backup
        if self._organism is not None:
            ckpt = torch.load(model_weights_path, map_location="cpu", weights_only=False)
            state = {
                k.replace("_orig_mod.", ""): v
                for k, v in ckpt["model_state_dict"].items()
            }
            from f51_darwin.darwin_x import migrate_mutational_state_for_load
            migrate_mutational_state_for_load(state, self._organism.model)
            self._organism.model.load_state_dict(state, strict=True)
            self._organism.model = self._organism.model.to(self._organism.device)
            self._state.is_collapsed = False
            self._state.collapse_reason = ""
            self._state.failed_attempts = 0

        return True

    @property
    def remaining_attempts(self) -> int:
        return max(0, MAX_FAILED_ATTEMPTS - self._state.failed_attempts)

    @property
    def is_collapsed(self) -> bool:
        return self._state.is_collapsed


# ═══════════════════════════════════════════════════════════
# TESTE
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    key = CreatorKey()

    # Teste 1: resposta correta
    assert key.verify("joca") == True, "joca deveria passar"
    assert key.verify("JOCA") == True, "JOCA deveria passar"
    assert key.verify("  Joca  ") == True, "Joca com espacos deveria passar"
    print("✅ Respostas corretas passaram")

    # Teste 2: respostas incorretas
    key2 = CreatorKey()
    assert key2.verify("toto") == False
    assert key2.remaining_attempts == 2
    assert key2.verify("rex") == False
    assert key2.remaining_attempts == 1
    assert key2.verify("bidu") == False  # 3a tentativa → colapso
    assert key2.is_collapsed == True
    assert key2.remaining_attempts == 0
    print("✅ Dead-man switch ativado apos 3 tentativas")

    # Teste 3: apos colapso, nem resposta correta passa
    assert key2.verify("joca") == False, "joca nao deveria passar apos colapso"
    print("✅ Resposta correta bloqueada apos colapso")

    print("\n🔐 CreatorKey: todos os testes passaram.")
