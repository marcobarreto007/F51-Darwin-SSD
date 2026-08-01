#!/usr/bin/env python3
"""BTB Shared Prefix Cache Store — Fase 3.

Armazenamento e replicação proativa de KV Cache de prefixos para o F51 Darwin-X.
Permite salvar o estado do cache de um prefixo (Attention K/V + SSM states)
e restaura-lo instantaneamente em slots de replicas sem recomputacao.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import torch
from f51_darwin.kv_cache import KVCache


@dataclass
class CachedPrefixEntry:
    """Entrada de um prefixo aquecido no Store."""
    prefix_hash: str
    prefix_length: int
    serialized_state: dict[str, Any]
    created_at: float
    access_count: int = 1
    last_accessed: float = 0.0


class SharedPrefixCacheStore:
    """Gerenciador central de KV Caches de prefixos compartilhados."""

    def __init__(self, max_entries: int = 64):
        self.max_entries = max_entries
        self._store: dict[str, CachedPrefixEntry] = {}

    def has_prefix(self, prefix_hash: str) -> bool:
        """Verifica se um prefixo aquecido esta presente no store."""
        return prefix_hash in self._store

    def store_prefix(
        self,
        prefix_hash: str,
        prefix_length: int,
        source_cache: KVCache,
    ) -> None:
        """Serializa e armazena o KV Cache de um prefixo recem-computado.

        Args:
            prefix_hash: Hash SHA-256 do prefixo
            prefix_length: Tamanho em tokens do prefixo
            source_cache: KVCache aquecido no prefill
        """
        now = time.perf_counter()
        # Usa o mecanismo nativo do F51 para checkpoint de estado
        serialized = source_cache.checkpoint_state()
        
        # Evict LRU se atingiu a capacidade maxima
        if len(self._store) >= self.max_entries and prefix_hash not in self._store:
            self._evict_lru()

        self._store[prefix_hash] = CachedPrefixEntry(
            prefix_hash=prefix_hash,
            prefix_length=prefix_length,
            serialized_state=serialized,
            created_at=now,
            last_accessed=now,
        )

    def restore_to_cache(
        self,
        prefix_hash: str,
        target_cache: KVCache,
        target_device: torch.device | None = None,
    ) -> bool:
        """Restaura o estado aquecido de um prefixo no target_cache.

        Args:
            prefix_hash: Hash SHA-256 do prefixo
            target_cache: Instancia de KVCache de destino
            target_device: Dispositivo de destino (CPU/CUDA)

        Returns:
            True se restaurou com sucesso, False se nao encontrou.
        """
        if prefix_hash not in self._store:
            return False

        entry = self._store[prefix_hash]
        entry.access_count += 1
        entry.last_accessed = time.perf_counter()

        # Restaura o estado serializado usando a logica nativa de retarget do F51
        target_cache.restore_state(entry.serialized_state, target_device=target_device)
        return True

    def _evict_lru(self) -> None:
        """Remove o prefixo menos recentemente acessado (LRU)."""
        if not self._store:
            return
        lru_hash = min(self._store.keys(), key=lambda k: self._store[k].last_accessed)
        del self._store[lru_hash]

    def clear(self) -> None:
        """Limpa todo o armazenamento de prefixos."""
        self._store.clear()
