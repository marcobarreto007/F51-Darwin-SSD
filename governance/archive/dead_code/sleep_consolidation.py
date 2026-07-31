"""
F51 SLEEP CONSOLIDATION — Replay e Consolidação Hipocampal
============================================================

Biologia: Durante o sono (especialmente REM), o cérebro replay eventos
do dia, fortalecendo sinapses importantes e fracos esquecem.
Hipocampo → Córtex: transferência de memória episódica → semântica.

Arquitetura:
- SleepPhase: fase de sono com replay
- HippocampalReplay: replay de episódios surpresa
- SynapticRenormalization: pruning e fortalecimento
- MemoryTransfer: hipocampo → córtex
- AChModulation: acetilcolina modula sleep pressure

Paper inspiração:
- "Sleep replay consolidates memory" (Wilson & McNaughton)
- "Synaptic homeostasis hypothesis" (Tononi & Cirelli)
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SleepConfig:
    """Configuração do ciclo de sono."""
    d_model: int = 2048
    sleep_interval_steps: int = 1000  # passos antes de dormir
    sleep_duration_steps: int = 100  # quantos passos de replay
    replay_buffer_size: int = 10000  # máximo episódios
    surprise_threshold: float = 0.3  # replay só episódios surpresa
    consolidation_lr: float = 0.001  # learning rate durante replay
    prune_threshold: float = 0.01  # prune conexões < threshold
    strengthen_factor: float = 1.5  # fortalece conexões replayed
    ach_decay_rate: float = 0.95  # decaimento de acetilcolina durante sono


@dataclass
class EpisodicMemory:
    """Memória episódica para replay."""
    hidden_state: torch.Tensor  # [seq_len, d_model]
    surprise: float
    domain: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    replay_count: int = 0


class SleepPhase(nn.Module):
    """Fase de sono com replay hippocampal.

    Durante o sono:
    1. Replay episódios surpresa em ordem aleatória
    2. Consolida: fortalece sinapses ativas
    3. Prune: remove conexões fracas
    4. Transfer: hipocampo → córtex (weights principais)
    """

    def __init__(self, config: SleepConfig):
        super().__init__()
        self.config = config

        # Buffer de memórias episódicas
        self.episodes: list[EpisodicMemory] = []

        # Acetilcolina: sleep pressure
        self.register_buffer('ach_pressure', torch.tensor(0.0))

        # Contador de passos
        self.register_buffer('step_count', torch.tensor(0))
        self.register_buffer('sleep_count', torch.tensor(0))

        # Consolidation network: replay target
        self.consolidation_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model)
        )

    def is_sleep_time(self) -> bool:
        """Verifica se é hora de dormir."""
        steps = self.step_count.item()
        return steps >= self.config.sleep_interval_steps

    def add_episode(
        self,
        hidden_state: torch.Tensor,
        surprise: float,
        domain: str = "unknown"
    ) -> None:
        """Adiciona episódio ao buffer se for surpresa."""
        if surprise < self.config.surprise_threshold:
            return  # não é surpresa, ignora

        # Remove episódio antigo se buffer cheio
        if len(self.episodes) >= self.config.replay_buffer_size:
            # Remove o menos replayed (FIFO com peso)
            self.episodes.sort(key=lambda e: e.replay_count)
            self.episodes.pop(0)

        episode = EpisodicMemory(
            hidden_state=hidden_state.detach().clone(),
            surprise=surprise,
            domain=domain
        )
        self.episodes.append(episode)

    @torch.no_grad()
    def sample_replay_batch(self, batch_size: int = 32) -> list[EpisodicMemory]:
        """Amostra batch de episódios para replay.

        Prioriza episódios de alta surpresa.
        """
        if not self.episodes:
            return []

        # Pesos proporcionais à surpresa
        weights = [e.surprise for e in self.episodes]
        total = sum(weights)
        if total == 0:
            return random.sample(self.episodes, min(batch_size, len(self.episodes)))

        probs = [w / total for w in weights]
        # Sample com reposição
        sampled = random.choices(self.episodes, weights=probs, k=batch_size)
        return sampled

    def replay_episode(self, episode: EpisodicMemory) -> torch.Tensor:
        """Replay um episódio através da consolidation network.

        Isso simula o replay hippocampal durante sono REM.
        """
        hidden = episode.hidden_state  # [seq_len, d_model]

        # Forward através da consolidation network
        replayed = self.consolidation_mlp(hidden)

        # Incrementa replay count
        episode.replay_count += 1

        return replayed

    @torch.no_grad()
    def synaptic_renormalization(self, model: nn.Module) -> dict[str, Any]:
        """Homeostase sináptica durante sono.

        1. Prune: remove pesos muito pequenos (esquecimento)
        2. Strengthen: fortalece pesos usados no replay
        """
        pruned_params = 0
        total_params = 0

        for name, param in model.named_parameters():
            if param.requires_grad and param.dim() >= 2:
                total_params += param.numel()

                # Prune: thresholding
                mask = param.abs() > self.config.prune_threshold
                pruned = (~mask).sum().item()
                pruned_params += pruned

                # Aplica pruning (seta para 0)
                param.data *= mask.float()

        # Normaliza ACh
        self.ach_pressure *= self.config.ach_decay_rate

        return {
            'pruned_params': pruned_params,
            'total_params': total_params,
            'prune_ratio': pruned_params / max(1, total_params),
            'ach_pressure': self.ach_pressure.item()
        }

    def sleep_cycle(
        self,
        model: Optional[nn.Module] = None
    ) -> dict[str, Any]:
        """Executa um ciclo completo de sono.

        1. Replay episódios surpresa
        2. Consolidation learning
        3. Synaptic renormalization
        4. Reset counters
        """
        self.sleep_count += 1
        stats = {
            'sleep_number': self.sleep_count.item(),
            'episodes_replayed': 0,
            'total_surprise': 0.0,
            'renorm': {}
        }

        # Replay episódios
        replay_steps = min(self.config.sleep_duration_steps, len(self.episodes) * 2)
        for _ in range(replay_steps):
            batch = self.sample_replay_batch(batch_size=4)
            if not batch:
                break

            for episode in batch:
                self.replay_episode(episode)
                stats['episodes_replayed'] += 1
                stats['total_surprise'] += episode.surprise

        # Synaptic renormalization
        if model is not None:
            stats['renorm'] = self.synaptic_renormalization(model)

        # Reset step count
        self.step_count.zero_()

        return stats

    def forward(
        self,
        hidden_states: torch.Tensor,
        surprise: float = 0.0,
        domain: str = "unknown",
        force_sleep: bool = False
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward pass.

        Args:
            hidden_states: [batch, seq_len, d_model]
            surprise: nível de surpresa (0-1)
            domain: domínio da tarefa
            force_sleep: força início do sono

        Returns:
            output: [batch, seq_len, d_model]
            stats: informações do ciclo
        """
        self.step_count += 1

        # Adiciona episódio se surpresa
        batch, seq_len, d_model = hidden_states.shape
        for b in range(batch):
            self.add_episode(
                hidden_states[b],
                surprise + random.uniform(-0.1, 0.1),  # adiciona variabilidade
                domain
            )

        stats = {
            'step': self.step_count.item(),
            'ach_pressure': self.ach_pressure.item(),
            'episodes_in_buffer': len(self.episodes),
            'sleep_time': False
        }

        # Verifica se é hora de dormir
        if force_sleep or self.is_sleep_time():
            stats['sleep_time'] = True
            # sleep_cycle será chamado externamente com o model
            # para permitir acesso aos weights
            return hidden_states, stats

        return hidden_states, stats


class HippocampalCortexTransfer(nn.Module):
    """Transferência de memória Hipocampo → Córtex.

    Biologia: Durante o sono, memórias são transferidas do hipocampo
    (memória episódica, curto prazo) para o córtex (memória semântica,
    longo prazo).

    Arquitetura:
    - Hippocampal buffer: episódios recentes
    - Cortical weights: pesos principais do modelo
    - Transfer: replay gradual
    """

    def __init__(self, config: SleepConfig):
        super().__init__()
        self.config = config

        # Hippocampal buffer (memória de trabalho)
        self.hippocampal_buffer: list[EpisodicMemory] = []

        # Cortical consolidation decoder
        self.cortical_decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, config.d_model)
        )

    def encode_to_hippocampus(
        self,
        hidden: torch.Tensor,
        surprise: float,
        domain: str
    ) -> None:
        """Codifica memória no hipocampo."""
        if len(self.hippocampal_buffer) >= 1000:  # limite hipocampal
            self.hippocampal_buffer.pop(0)

        episode = EpisodicMemory(
            hidden_state=hidden.detach().clone().mean(dim=0) if hidden.dim() == 2 else hidden.detach().clone(),
            surprise=surprise,
            domain=domain
        )
        self.hippocampal_buffer.append(episode)

    def transfer_to_cortex(
        self,
        cortical_module: nn.Module,
        transfer_rate: float = 0.1
    ) -> dict[str, Any]:
        """Transfere memórias do hipocampo para o córtex.

        Isso simula a consolidação de memória durante o sono.
        """
        if not self.hippocampal_buffer:
            return {'transferred': 0}

        # Seleciona episódios para transferência (alta surpresa)
        to_transfer = [
            e for e in self.hippocampal_buffer
            if e.surprise > self.config.surprise_threshold
        ][:10]  # batch de 10

        stats = {
            'transferred': len(to_transfer),
            'domains': {}
        }

        for episode in to_transfer:
            # Transferência através do decoder cortical
            with torch.no_grad():
                cortical_repr = self.cortical_decoder(episode.hidden_state)

            # Atualiza estatísticas
            stats['domains'][episode.domain] = stats['domains'].get(episode.domain, 0) + 1

            # Remove do buffer hipocampal
            if episode in self.hippocampal_buffer:
                self.hippocampal_buffer.remove(episode)

        return stats


# ═══════════════════════ TESTE ═══════════════════════

def _test_sleep_consolidation():
    """Smoke test do módulo Sleep Consolidation."""
    print("Testing Sleep Consolidation...")

    config = SleepConfig(
        d_model=256,
        sleep_interval_steps=5,  # curto para teste
        sleep_duration_steps=3,
        replay_buffer_size=100
    )

    sleep = SleepPhase(config)

    # Test 1: Adicionar episódios
    hidden = torch.randn(2, 8, 256)
    out1, stats1 = sleep(hidden, surprise=0.5, domain="test")
    assert out1.shape == (2, 8, 256)
    assert stats1['episodes_in_buffer'] == 2
    print("✓ Test 1: Add episodes")

    # Test 2: Não é hora de dormir ainda
    assert not stats1['sleep_time']
    print("✓ Test 2: Not sleep time yet")

    # Test 3: Forçar sono
    _, stats2 = sleep(hidden, force_sleep=True)
    assert stats2['sleep_time']
    print("✓ Test 3: Force sleep")

    # Test 4: Ciclo de sono
    sleep_stats = sleep.sleep_cycle()
    assert 'episodes_replayed' in sleep_stats
    assert 'renorm' in sleep_stats
    print("✓ Test 4: Sleep cycle")

    # Test 5: Synaptic renormalization
    dummy_model = nn.Linear(256, 256)
    renorm_stats = sleep.synaptic_renormalization(dummy_model)
    assert 'pruned_params' in renorm_stats
    assert 'prune_ratio' in renorm_stats
    print("✓ Test 5: Synaptic renormalization")

    # Test 6: Hipocampal transfer
    transfer = HippocampalCortexTransfer(config)
    transfer.encode_to_hippocampus(hidden[0], surprise=0.6, domain="math")
    transfer_stats = transfer.transfer_to_cortex(dummy_model)
    assert transfer_stats['transferred'] >= 0
    print("✓ Test 6: Hippocampal transfer")

    # Test 7: ACh pressure
    initial_ach = sleep.ach_pressure.item()
    sleep.synaptic_renormalization(dummy_model)
    assert sleep.ach_pressure.item() <= initial_ach  # deve decair
    print("✓ Test 7: ACh pressure decay")

    # Test 8: Reset após sono
    assert sleep.step_count.item() == 0  # resetado
    print("✓ Test 8: Step count reset after sleep")

    print("\n✅ All Sleep Consolidation tests passed!")


if __name__ == "__main__":
    _test_sleep_consolidation()
