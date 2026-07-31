"""
F51 Expert Cache — Reuso de saída de experts por similaridade de entrada.

Baseado no F51 Nitro cache (67.9%–87.6% hit rate em GPT-OSS 120B).
Evita recomputar experts quando a entrada é similar a uma já processada.
Essencial para reduzir latência do MoE em inferência.

Arquitetura:
    input → quantize → hash → lookup LRU cache
    cache hit  → retorna saída salva (O(1))
    cache miss → computa expert → armazena no cache
"""

from __future__ import annotations

import torch
from collections import OrderedDict
from typing import Optional


class ExpertCache:
    """Cache LRU de saídas de experts com hashing por similaridade.

    Cada entrada é indexada por (expert_idx, bucket_hash) onde bucket_hash
    é derivado da quantização do vetor de entrada (8-bit por dimensão).

    Attributes:
        capacity: número máximo de entradas no cache
        bucket_bits: bits de quantização por dimensão (default 8)
        hit_count: total de cache hits desde o reset
        miss_count: total de cache misses desde o reset
    """

    def __init__(self, capacity: int = 2048, bucket_bits: int = 8) -> None:
        self.capacity = capacity
        self.bucket_bits = bucket_bits
        self._store: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.hit_count: int = 0
        self.miss_count: int = 0
        self._total_queries: int = 0
        # Pesos primos pré-computados para hash vetorizado
        self._primes = torch.tensor(
            [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
             41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
             89, 97, 101, 103, 107, 109, 113, 127, 131],
            dtype=torch.long,
        )

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

    def _batch_hashes(self, x: torch.Tensor) -> torch.Tensor:
        """Vetorizado: computa hash para todos os tokens de uma vez.

        Args:
            x: [N, D] — batch de tokens

        Returns:
            hashes: [N] — hash inteiro por token
        """
        with torch.no_grad():
            n, d = x.shape
            # Blocos de ~24 dimensões → 32 features por token (16 médias + 16 stds)
            n_blocks = 16
            block_size = max(1, d // n_blocks)

            # Pad para ser divisível
            padded_d = n_blocks * block_size
            if d < padded_d:
                x_pad = torch.cat([x, torch.zeros(n, padded_d - d, device=x.device)], dim=1)
            else:
                x_pad = x[:, :padded_d]

            # Reshape: [N, n_blocks, block_size]
            x_blocks = x_pad.reshape(n, n_blocks, block_size)

            # Features: média e std por bloco → [N, n_blocks * 2]
            means = x_blocks.mean(dim=2)     # [N, n_blocks]
            stds = x_blocks.std(dim=2)        # [N, n_blocks]
            features = torch.cat([means, stds], dim=1)  # [N, 32]

            # Normalizar cada token individualmente para [0, 1]
            f_min = features.min(dim=1, keepdim=True).values  # [N, 1]
            f_max = features.max(dim=1, keepdim=True).values  # [N, 1]
            denom = (f_max - f_min).clamp(min=1e-8)
            f_norm = (features - f_min) / denom  # [N, 32]

            # Quantizar para 15 níveis (4 bits)
            quantized = (f_norm * 15).long()  # [N, 32]

            # Hash: soma ponderada com pesos primos, módulo 2^31-1
            weights = self._primes[:quantized.shape[1]].to(x.device)  # [32]
            hashes = (quantized * weights).sum(dim=1) & 0x7FFFFFFF  # [N]
            return hashes

    def _make_keys(self, expert_idx: int, hashes: torch.Tensor) -> torch.Tensor:
        """Combina expert_idx com hashes para chaves de cache."""
        return ((expert_idx << 31) | (hashes & 0x7FFFFFFF)).cpu()

    def batch_lookup(
        self, expert_idx: int, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Vetorizado: lookup em batch — GPU-friendly.

        Args:
            expert_idx: índice do expert
            x: [N, D] — batch de tokens

        Returns:
            hit_mask: [N] bool — True se token teve cache hit
            hit_outputs: [N, D] ou None — outputs cacheados (device = x.device)
        """
        n, d = x.shape
        device = x.device
        hashes = self._batch_hashes(x)  # [N] — computado na GPU
        keys = self._make_keys(expert_idx, hashes)  # [N] — CPU

        hit_mask = torch.zeros(n, dtype=torch.bool)
        hit_outputs = torch.zeros(n, d, dtype=x.dtype, device=device)
        any_hit = False

        for i in range(n):
            self._total_queries += 1
            key = int(keys[i].item())
            if key in self._store:
                value = self._store.pop(key)
                self._store[key] = value  # LRU: move pro fim
                self.hit_count += 1
                hit_mask[i] = True
                hit_outputs[i] = value.to(device)
                any_hit = True
            else:
                self.miss_count += 1

        if any_hit:
            return hit_mask, hit_outputs
        return hit_mask, None

    def batch_store(self, expert_idx: int, x: torch.Tensor, outputs: torch.Tensor) -> None:
        """Vetorizado: armazena múltiplos tokens no cache."""
        hashes = self._batch_hashes(x)
        keys = self._make_keys(expert_idx, hashes)
        for i in range(x.shape[0]):
            key = int(keys[i].item())
            if key in self._store:
                self._store.pop(key)
            elif len(self._store) >= self.capacity:
                self._store.popitem(last=False)
            self._store[key] = outputs[i:i + 1].detach().cpu().clone()

    def clear(self) -> None:
        """Limpa o cache e reseta estatísticas."""
        self._store.clear()
        self.hit_count = 0
        self.miss_count = 0
        self._total_queries = 0

    def stats(self) -> dict:
        """Retorna estatísticas do cache."""
        return {
            "capacity": self.capacity,
            "size": len(self._store),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_rate,
            "total_queries": self._total_queries,
        }


class ExpertCacheConfig:
    """Configuração do F51 Expert Cache."""

    def __init__(
        self,
        enabled: bool = True,
        capacity: int = 2048,
        bucket_bits: int = 8,
        warmup_steps: int = 100,
    ) -> None:
        self.enabled = enabled
        self.capacity = capacity
        self.bucket_bits = bucket_bits
        self.warmup_steps = warmup_steps


def compute_expert_with_cache(
    expert_fn,
    expert_idx: int,
    x: torch.Tensor,
    cache: ExpertCache,
    *,
    training: bool = False,
) -> torch.Tensor:
    """Computa expert com F51 cache lookup — vetorizado para GPU.

    Estratégia:
      1. batch_lookup: hashes vetorizados na GPU para todos os tokens.
      2. Misses: forward em batch único na GPU (paralelo total).
      3. batch_store: armazena resultados dos misses.

    Args:
        expert_fn: callable que recebe (x) e retorna output
        expert_idx: índice do expert
        x: tensor de entrada [N, D]
        cache: instância de ExpertCache
        training: se True, bypass cache

    Returns:
        output: tensor de saída [N, D]
    """
    if training or not cache:
        return expert_fn(x)

    # Fase 1: lookup vetorizado
    hit_mask, hit_outputs = cache.batch_lookup(expert_idx, x)

    # Se tudo hit, retorna direto
    if hit_mask.all():
        return hit_outputs

    # Se nada hit, computa tudo em batch e armazena
    if not hit_mask.any():
        output = expert_fn(x)
        cache.batch_store(expert_idx, x, output)
        return output

    # Caso misto: computa só misses em batch
    miss_mask = ~hit_mask
    miss_indices = torch.where(miss_mask)[0]
    miss_input = x[miss_indices]  # [M, D]
    miss_output = expert_fn(miss_input)  # GPU batch forward
    cache.batch_store(expert_idx, miss_input, miss_output)

    # Combina hits + misses
    output = torch.empty_like(x)
    output[hit_mask] = hit_outputs[hit_mask]
    output[miss_mask] = miss_output
    return output
