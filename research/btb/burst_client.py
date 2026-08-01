#!/usr/bin/env python3
"""BTB Burst Client Simulator — Fase 5.

Simulador de cliente de rajadas para o F51 Darwin-X.
Simula a chegada concorrente de requisições com prefixos idênticos
(job de rotulagem, fanout de subagentes) e valida o ganho empírico do BTB.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.btb.burst_detector import BurstConfig, BurstDetector
from research.btb.prefix_cache_store import SharedPrefixCacheStore
from f51_darwin.kv_cache import KVCache


class BTBBurstSimulator:
    """Simulador end-to-end do pipeline Biting the Bullet no F51."""

    def __init__(
        self,
        burst_config: BurstConfig | None = None,
        store_capacity: int = 64,
    ):
        self.detector = BurstDetector(burst_config or BurstConfig())
        self.store = SharedPrefixCacheStore(max_entries=store_capacity)
        self.detector.register_warmup_handler(self._on_burst_detected)
        self.warmup_events: list[dict[str, Any]] = []

    def _on_burst_detected(self, prefix_hash: str, prefix_tokens: list[int], num_replicas: int) -> None:
        """Callback acionado quando o detector identifica um alarme de burst."""
        self.warmup_events.append({
            "prefix_hash": prefix_hash,
            "prefix_tokens_len": len(prefix_tokens),
            "num_replicas": num_replicas,
            "timestamp": time.time(),
        })

    def simulate_request_stream(
        self,
        prefix_tokens: list[int],
        suffix_tokens_list: list[list[int]],
        arrival_interval_s: float = 0.05,
    ) -> dict[str, Any]:
        """Simula o processamento de uma sequencia de requisicoes com mesmo prefixo.

        Args:
            prefix_tokens: Tokens do prefixo compartilhado
            suffix_tokens_list: Lista de sufixos unicos por requisicao
            arrival_interval_s: Intervalo entre chegadas de requisicoes

        Returns:
            Resultados com estatisticas do burst (cold count, warm count, speedup est.)
        """
        num_requests = len(suffix_tokens_list)
        results = {
            "total_requests": num_requests,
            "cold_requests": 0,
            "warmup_triggered": False,
            "warm_requests": 0,
            "routing_actions": [],
        }

        now = time.time()
        for idx, suffix in enumerate(suffix_tokens_list):
            full_prompt = prefix_tokens + suffix
            req_time = now + (idx * arrival_interval_s)

            # Roteador processa a requisicao
            route = self.detector.process_request(full_prompt, timestamp=req_time)
            results["routing_actions"].append(route["action"])

            if route["action"] == "route_cold":
                results["cold_requests"] += 1
            elif route["action"] == "trigger_warmup":
                results["warmup_triggered"] = True
                results["cold_requests"] += 1
            elif route["action"] == "route_warm":
                results["warm_requests"] += 1

        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulador de Rajadas BTB no F51")
    parser.add_argument("--num-requests", type=int, default=10, help="Numero de requisicoes no burst")
    parser.add_argument("--prefix-len", type=int, default=256, help="Tamanho do prefixo em tokens")
    parser.add_argument("--suffix-len", type=int, default=32, help="Tamanho do sufixo em tokens")
    args = parser.parse_args()

    simulator = BTBBurstSimulator()
    prefix = list(range(100, 100 + args.prefix_len))
    suffixes = [list(range(i * 10, i * 10 + args.suffix_len)) for i in range(args.num_requests)]

    res = simulator.simulate_request_stream(prefix, suffixes)

    print("\n" + "=" * 60)
    print(" 🚀 F51 DARWIN-X — SIMULAÇÃO DE RAJADA BTB (END-TO-END)")
    print("=" * 60)
    print(f" Total de Requisições no Burst: {res['total_requests']}")
    print(f" Tamanho do Prefixo Compartilhado: {args.prefix_len} tokens")
    print(f" Tamanho do Sufixo Único: {args.suffix_len} tokens")
    print(f" Requisições Frias (Prefill): {res['cold_requests']}")
    print(f" Alarme de Burst Disparado: {res['warmup_triggered']}")
    print(f" Requisições Aquecidas (Reuse Spec): {res['warm_requests']}")
    print(f" Taxa de Reutilização Quente: {res['warm_requests'] / res['total_requests'] * 100:.1f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
