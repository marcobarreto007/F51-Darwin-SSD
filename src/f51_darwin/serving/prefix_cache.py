#!/usr/bin/env python3
"""Cache de prefixo compartilhado GPU-residente.

Diferente do ``KVCache.checkpoint_state()`` que serializa para CPU,
este cache mantem os tensores de estado KV na GPU via ``.clone()``,
eliminando o overhead de ~230ms de round-trip CPU-GPU.

Uso::

    from f51_darwin.serving.prefix_cache import PrefixCache

    cache = PrefixCache(max_entries=16)
    state = cache.lookup(prefix_hash)   # -> dict | None
    cache.store(prefix_hash, state)     # armazena clone GPU-residente
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class PrefixCacheStats:
    """Metricas do cache de prefixo."""
    hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0
    total_bytes_stored: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class PrefixCache:
    """Cache LRU de estados KV de prefixo, GPU-residente.

    Armazena um dicionario de tensores clonados na GPU para cada
    prefixo. A eviccao e LRU sobre entradas (nao sobre bytes).
    """

    def __init__(self, max_entries: int = 16) -> None:
        self._max = max_entries
        self._cache: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
        self.stats = PrefixCacheStats()

    # ── API publica ──────────────────────────────────────────────

    def lookup(self, prefix_hash: str) -> dict[str, torch.Tensor] | None:
        """Retorna estado KV cacheado para o prefixo, ou None.

        O estado retornado contem tensores no device original (GPU).
        O caller deve clona-los antes de modificar.
        """
        if prefix_hash not in self._cache:
            self.stats.misses += 1
            return None

        # LRU: move para o fim (mais recente)
        self._cache.move_to_end(prefix_hash)
        self.stats.hits += 1
        return self._cache[prefix_hash]

    def store(self, prefix_hash: str, state: dict[str, Any]) -> None:
        """Armazena uma copia GPU-residente do estado KV.

        Args:
            prefix_hash: identificador do prefixo.
            state: dicionario de tensores torch (no device GPU).
                   Cada tensor e clonado via .detach().clone().
                   Valores nao-tensor sao armazenados como estao.
        """
        # Eviccao LRU se necessario
        evicted_bytes = 0
        while len(self._cache) >= self._max:
            evicted_key, evicted_state = self._cache.popitem(last=False)
            self.stats.evictions += 1
            evicted_bytes += sum(
                self._tensor_bytes(v) for v in evicted_state.values()
            )
            self._free_tensors(evicted_state)

        # Clona tensores para GPU (sem CPU round-trip)
        clone: dict[str, Any] = {}
        total_bytes = 0
        for key, tensor in state.items():
            if isinstance(tensor, torch.Tensor):
                c = tensor.detach().clone()
                clone[key] = c
                total_bytes += c.numel() * c.element_size()
            elif isinstance(tensor, tuple) and len(tensor) == 2 and isinstance(tensor[0], torch.Tensor):
                # (K, V) tuples
                c0 = tensor[0].detach().clone()
                c1 = tensor[1].detach().clone()
                clone[key] = (c0, c1)
                total_bytes += c0.numel() * c0.element_size() + c1.numel() * c1.element_size()
            else:
                # non-tensor metadata (int, float, str, bool, tuple, etc.)
                clone[key] = tensor

        self._cache[prefix_hash] = clone
        self._cache.move_to_end(prefix_hash)
        self.stats.stores += 1
        self.stats.total_bytes_stored = self.stats.total_bytes_stored + total_bytes - evicted_bytes

    def has(self, prefix_hash: str) -> bool:
        """Retorna True se o prefixo esta em cache."""
        return prefix_hash in self._cache

    def clear(self) -> None:
        """Limpa todo o cache e libera memoria GPU."""
        for state in self._cache.values():
            self._free_tensors(state)
        self._cache.clear()
        self.stats = PrefixCacheStats()

    def size(self) -> int:
        """Numero de entradas no cache."""
        return len(self._cache)

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _free_tensors(state: dict) -> None:
        """Libera tensores em um state dict."""
        for v in state.values():
            if isinstance(v, torch.Tensor):
                del v
            elif isinstance(v, tuple):
                for t in v:
                    if isinstance(t, torch.Tensor):
                        del t

    @staticmethod
    def _tensor_bytes(v: Any) -> int:
        """Estima bytes de um tensor ou tuple de tensores."""
        if isinstance(v, torch.Tensor):
            return v.numel() * v.element_size()
        if isinstance(v, tuple):
            return sum(PrefixCache._tensor_bytes(t) for t in v)
        return 0


# ═══════════════════════════════════════════════════════════════════════════
# Integracao com KVCache do Darwin
# ═══════════════════════════════════════════════════════════════════════════


def kv_cache_to_state(cache) -> dict[str, Any]:
    """Extrai estado GPU-residente de um KVCache.

    Diferente de ``checkpoint_state()``, mantem tensores na GPU
    via ``.detach().clone()`` sem passar pela CPU.

    Args:
        cache: instancia de KVCache populada por prefill.

    Returns:
        Dicionario com tensores clonados na GPU.
    """
    state: dict[str, Any] = {}
    for layer_idx, h in cache._ssd_states.items():
        state[f"ssd_{layer_idx}"] = h.detach().clone()
    for layer_idx, c in cache._ssd_conv.items():
        state[f"conv_{layer_idx}"] = c.detach().clone()
    for layer_idx, (k, v) in cache._attn_kv.items():
        state[f"attn_{layer_idx}"] = (k.detach().clone(), v.detach().clone())
    state["seq_len"] = cache._seq_len
    return state


def state_to_kv_cache(state: dict, cache, target_device: torch.device | None = None) -> None:
    """Restaura um KVCache a partir de estado GPU-residente.

    Args:
        state: dicionario retornado por ``kv_cache_to_state()``.
        cache: instancia de KVCache a ser populada (ja deve existir).
        target_device: dispositivo alvo (se None, usa o device dos tensores).
    """
    cache.clear()
    cache._seq_len = state.get("seq_len", 0)

    for key, tensor in state.items():
        if key == "seq_len":
            continue
        if key.startswith("ssd_"):
            layer_idx = int(key.split("_")[1])
            cache._ssd_states[layer_idx] = tensor.detach().clone()
        elif key.startswith("conv_"):
            layer_idx = int(key.split("_")[1])
            cache._ssd_conv[layer_idx] = tensor.detach().clone()
        elif key.startswith("attn_"):
            layer_idx = int(key.split("_")[1])
            k, v = tensor
            cache._attn_kv[layer_idx] = (k.detach().clone(), v.detach().clone())


# ═══════════════════════════════════════════════════════════════════════════

__all__ = ["PrefixCache", "PrefixCacheStats", "kv_cache_to_state", "state_to_kv_cache"]
