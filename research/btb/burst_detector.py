#!/usr/bin/env python3
"""BTB Burst Detector — Fase 2.

Detector de rajadas de prefixos em tempo real para o F51 Darwin-X.
Monitora o fluxo de requisicoes no Gateway/Router e identifica padroes
de prefixos idênticos frequentes (data labeling, fanout de subagentes, etc.).

Algoritmo:
    - Recebe o prefixo de tamanho >= Y tokens (ou seu hash SHA-256).
    - Registra a chegada com timestamp.
    - Se o mesmo prefix_hash ocorrer >= X vezes dentro de Z segundos:
        1. Marca o prefixo como rajada ativa (active_burst).
        2. Dispara sinal de pre-aquecimento especulativo (speculative warmup)
           para M replicas/slots secundários.
        3. Roteia requisicoes posteriores entre os slots aquecidos (least-load).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class BurstConfig:
    """Configuracao de parametros do detector BTB."""
    min_prefix_tokens: int = 128    # Y: tamanho minimo do prefixo em tokens
    min_arrivals: int = 2           # X: chegadas do mesmo prefixo para disparar alarme
    window_seconds: float = 1.0     # Z: janela temporal de detecção em segundos
    warmup_replicas: int = 4        # M: numero de replicas a aquecer proativamente
    block_size: int = 16            # tamanho do bloco de KV cache


@dataclass
class PrefixBurstState:
    """Estado rastreado de um prefixo no detector."""
    prefix_hash: str
    prefix_tokens: list[int]
    arrival_timestamps: list[float] = field(default_factory=list)
    is_active_burst: bool = False
    warmed_replica_ids: set[int] = field(default_factory=set)
    first_detected_at: float = 0.0


class BurstDetector:
    """Detector preditivo de rajadas de prefixo (BTB)."""

    def __init__(self, config: BurstConfig | None = None):
        self.config = config or BurstConfig()
        self._prefix_states: dict[str, PrefixBurstState] = {}
        self._warmup_handlers: list[Callable[[str, list[int], int], None]] = []

    @staticmethod
    def compute_prefix_hash(prefix_tokens: list[int], min_tokens: int = 128) -> str:
        """Calcula o SHA-256 dos primeiros min_tokens do prefixo."""
        slice_tokens = prefix_tokens[:min_tokens]
        byte_data = bytes().join(t.to_bytes(4, byteorder="big") for t in slice_tokens)
        return hashlib.sha256(byte_data).hexdigest()

    def register_warmup_handler(self, handler: Callable[[str, list[int], int], None]) -> None:
        """Registra um callback para ser notificado quando um burst for detectado."""
        self._warmup_handlers.append(handler)

    def process_request(
        self,
        prompt_tokens: list[int],
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        """Processa a chegada de uma requisicao e avalia gatilho de burst.

        Args:

            prompt_tokens: sequencia de tokens do prompt
            timestamp: timestamp de chegada (se None, usa time.time())

        Returns:

            Dict com informacoes de roteamento e estado do burst:
            {
                "prefix_hash": str,
                "is_burst": bool,
                "action": "route_cold" | "trigger_warmup" | "route_warm",
                "target_replicas": int,
            }
        """
        now = timestamp if timestamp is not None else time.time()
        
        # Se o prompt for menor que o limite minimo Y, roteia normalmente (cold)
        if len(prompt_tokens) < self.config.min_prefix_tokens:
            return {
                "prefix_hash": "",
                "is_burst": False,
                "action": "route_cold",
                "target_replicas": 1,
            }

        prefix_hash = self.compute_prefix_hash(prompt_tokens, self.config.min_prefix_tokens)
        
        if prefix_hash not in self._prefix_states:
            self._prefix_states[prefix_hash] = PrefixBurstState(
                prefix_hash=prefix_hash,
                prefix_tokens=prompt_tokens[: self.config.min_prefix_tokens],
            )
            
        state = self._prefix_states[prefix_hash]
        
        # Limpa timestamps fora da janela Z
        cutoff = now - self.config.window_seconds
        state.arrival_timestamps = [t for t in state.arrival_timestamps if t >= cutoff]
        state.arrival_timestamps.append(now)

        # Se ja for um burst ativo
        if state.is_active_burst:
            return {
                "prefix_hash": prefix_hash,
                "is_burst": True,
                "action": "route_warm",
                "target_replicas": self.config.warmup_replicas,
            }

        # Avalia se atingiu X chegadas na janela Z
        if len(state.arrival_timestamps) >= self.config.min_arrivals:
            state.is_active_burst = True
            state.first_detected_at = now
            
            # Notifica os handlers para pre-aquecimento via RDMA / PCIe
            for handler in self._warmup_handlers:
                handler(prefix_hash, state.prefix_tokens, self.config.warmup_replicas)

            return {
                "prefix_hash": prefix_hash,
                "is_burst": True,
                "action": "trigger_warmup",
                "target_replicas": self.config.warmup_replicas,
            }

        return {
            "prefix_hash": prefix_hash,
            "is_burst": False,
            "action": "route_cold",
            "target_replicas": 1,
        }

    def purge_expired(self, max_idle_seconds: float = 60.0) -> int:
        """Remove estados de prefixo inativos por mais de max_idle_seconds."""
        now = time.time()
        expired = [
            h for h, s in self._prefix_states.items()
            if not s.arrival_timestamps or (now - s.arrival_timestamps[-1]) > max_idle_seconds
        ]
        for h in expired:
            del self._prefix_states[h]
        return len(expired)
