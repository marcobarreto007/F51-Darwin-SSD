#!/usr/bin/env python3
"""Detector de rajadas de prefixo — algoritmo X/Y/Z/M do BTB.

Implementa a regra de 4 constantes do ``EarlyRdma`` original, adaptada
para o contexto single-node do Darwin-X (sem RDMA, sem replicas).

Uso::

    from f51_darwin.serving.burst_detector import BurstDetector

    detector = BurstDetector(threshold=2, prefix_blocks=256, window_s=1.0)
    detector.ingest(prefix_hash, timestamp)  # -> BurstState
    detector.is_active(prefix_hash)          # -> bool
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BurstState:
    """Resultado de uma chamada a ``ingest()``."""
    prefix_hash: str
    active: bool          # prefixo esta ativo (>= X chegadas em Z segundos)
    just_triggered: bool  # este evento disparou o burst (exatamente X-esima chegada)
    recent_count: int     # chegadas na janela Z
    elapsed_s: float = 0.0


class BurstDetector:
    """Detector de rajadas de prefixo — algoritmo X/Y/Z/M.

    Se o mesmo prefixo chega **X** vezes dentro de **Z** segundos,
    o detector marca como "ativo" e notifica via callback.

    Parametros
    ----------
    threshold : int
        X — numero de chegadas do mesmo prefixo para disparar (default: 2).
    prefix_blocks : int
        Y — numero de blocos do prefixo considerados na chave (default: 256).
        No Darwin, cada "bloco" equivale a N tokens (tipicamente 256).
    window_s : float
        Z — janela de deteccao em segundos (default: 1.0).
    warm_copies : int
        M — quantidade de copias a aquecer (default: 4).
        No contexto Darwin, > 0 significa "ative o warming".
    on_trigger : callable | None
        Callback chamado quando um burst e detectado.
        Assinatura: ``on_trigger(prefix_hash, state: BurstState) -> None``.
    """

    def __init__(
        self,
        threshold: int = 2,
        prefix_blocks: int = 256,
        window_s: float = 1.0,
        warm_copies: int = 4,
        *,
        on_trigger: callable | None = None,
    ) -> None:
        self.threshold = int(threshold)
        self.prefix_blocks = int(prefix_blocks)
        self.window_s = float(window_s)
        self.warm_copies = int(warm_copies)
        self._on_trigger = on_trigger

        # prefix_hash -> lista de timestamps (ordenada, monotonicamente crescente)
        self._hist: dict[str, list[float]] = defaultdict(list)

        # prefixos ativos (em burst)
        self._active: set[str] = set()

        # Estatisticas
        self.stats = BurstStats()

    # ── API publica ──────────────────────────────────────────────

    def ingest(self, prefix_hash: str, timestamp_s: float | None = None) -> BurstState:
        """Registra uma chegada de request com o prefixo dado.

        Args:
            prefix_hash: hash identificador do prefixo (ex: sha256 dos tokens).
            timestamp_s: timestamp da chegada em segundos.
                         Se None, usa ``time.perf_counter()``.

        Returns:
            BurstState com status do prefixo apos esta chegada.
        """
        import time as _time_mod
        now = timestamp_s if timestamp_s is not None else _time_mod.perf_counter()

        h = self._hist[prefix_hash]
        h.append(now)

        # Expurga entradas fora da janela Z
        cut = now - self.window_s
        while h and h[0] < cut:
            h.pop(0)

        recent = len(h)
        active = recent >= self.threshold
        just_triggered = active and prefix_hash not in self._active

        # Atualiza estado
        if active:
            if just_triggered:
                self._active.add(prefix_hash)
                self.stats.bursts_detected += 1
                if self._on_trigger is not None:
                    self._on_trigger(prefix_hash, BurstState(
                        prefix_hash=prefix_hash,
                        active=True, just_triggered=True,
                        recent_count=recent,
                        elapsed_s=0.0,
                    ))
        elif prefix_hash in self._active:
            # Prefixo deixou de ser ativo (janela expirou)
            self._active.discard(prefix_hash)

        self.stats.total_arrivals += 1

        # Limpa historico vazio para nao acumular memoria
        if not h:
            del self._hist[prefix_hash]

        return BurstState(
            prefix_hash=prefix_hash,
            active=active,
            just_triggered=just_triggered,
            recent_count=recent,
            elapsed_s=now - h[0] if h else float("inf"),
        )

    def is_active(self, prefix_hash: str) -> bool:
        """Retorna True se o prefixo esta em burst ativo."""
        return prefix_hash in self._active

    def active_prefixes(self) -> set[str]:
        """Retorna o conjunto de prefixos ativos no momento."""
        # Expurga expirados antes de retornar (limpeza lazy)
        import time as _time_mod
        now = _time_mod.perf_counter()
        expired = set()
        for key in self._active:
            h = self._hist.get(key, [])
            while h and h[0] < now - self.window_s:
                h.pop(0)
            if len(h) < self.threshold:
                expired.add(key)
                if not h:
                    del self._hist[key]
        self._active -= expired
        return set(self._active)

    def reset(self) -> None:
        """Limpa todo o estado do detector."""
        self._hist.clear()
        self._active.clear()
        self.stats = BurstStats()


@dataclass
class BurstStats:
    """Estatisticas acumuladas do detector."""
    total_arrivals: int = 0
    bursts_detected: int = 0

    @property
    def false_positive_ratio(self) -> float:
        """Bursts detectados por chegada (aproximacao grosseira)."""
        if self.total_arrivals == 0:
            return 0.0
        return self.bursts_detected / self.total_arrivals


# ═══════════════════════════════════════════════════════════════════════════
# Exportacao canonica
# ═══════════════════════════════════════════════════════════════════════════

__all__ = ["BurstDetector", "BurstState", "BurstStats"]
