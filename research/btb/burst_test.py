#!/usr/bin/env python3
"""Teste do BurstDetector — bursts sinteticos + decoys.

Simula o padrao do Bursted-ART: 8 bursts de 500 requests cada,
com prefixo compartilhado de 65K tokens, mais 120 decoys.

Uso:
    python research/btb/burst_test.py
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from f51_darwin.serving.burst_detector import BurstDetector, BurstState


def prefix_hash(job_id: int, prefix_blocks: int = 256) -> str:
    """Simula o hash de um prefixo (sha256 dos primeiros Y blocos)."""
    key = f"burst_job_{job_id}:prefix_{prefix_blocks}"
    return hashlib.sha256(key.encode()).hexdigest()


def run_test() -> dict:
    """Executa o cenario de teste Bursted-ART-like."""
    # Parametros (identicos ao BTB: X=2, Y=256, Z=1.0s, M=4)
    detector = BurstDetector(
        threshold=2,
        prefix_blocks=256,
        window_s=1.0,
        warm_copies=4,
    )

    # ═══════════════════════════════════════════════════════════
    # 8 bursts, 500 requests cada, prefixo compartilhado
    # ═══════════════════════════════════════════════════════════
    NUM_BURSTS = 8
    BURST_SIZE = 500
    BURST_WINDOW_S = 60.0  # distribuido em 60s

    burst_detections = 0
    burst_prefixes = set()

    t0 = time.perf_counter()
    for job_idx in range(NUM_BURSTS):
        burst_hash = prefix_hash(job_idx)
        start_offset = job_idx * 40.0  # 40s entre bursts

        for i in range(BURST_SIZE):
            arrival = start_offset + BURST_WINDOW_S * i / max(1, BURST_SIZE - 1)
            t_abs = t0 + arrival
            result = detector.ingest(burst_hash, t_abs)
            if result.just_triggered:
                burst_detections += 1
                burst_prefixes.add(job_idx)

    # ═══════════════════════════════════════════════════════════
    # 120 decoys — 1 request cada, prefixo unico
    # ═══════════════════════════════════════════════════════════
    NUM_DECOYS = 120
    decoy_detections = 0

    for decoy_idx in range(NUM_DECOYS):
        decoy_hash = prefix_hash(1000 + decoy_idx)  # ID fora da faixa de burst
        arrival = decoy_idx * 0.5  # espalhados no tempo
        t_abs = t0 + arrival
        result = detector.ingest(decoy_hash, t_abs)
        if result.just_triggered:
            decoy_detections += 1

    return {
        "num_bursts": NUM_BURSTS,
        "burst_size": BURST_SIZE,
        "num_decoys": NUM_DECOYS,
        "bursts_detected": burst_detections,
        "burst_prefixes_triggered": len(burst_prefixes),
        "decoy_detections": decoy_detections,
        "total_arrivals": detector.stats.total_arrivals,
        "all_bursts_detected": burst_detections == NUM_BURSTS,
        "no_decoys_detected": decoy_detections == 0,
        "passed": burst_detections == NUM_BURSTS and decoy_detections == 0,
    }


def main() -> None:
    print("BTB Burst Detector — Teste Bursted-ART-like")
    print(f"  Algoritmo: X=2, Y=256, Z=1.0s, M=4")
    print()

    results = run_test()

    print(f"  Bursts:       {results['num_bursts']} jobs x {results['burst_size']} requests")
    print(f"  Decoys:       {results['num_decoys']} requests (prefixo unico)")
    print(f"  Total arrivals: {results['total_arrivals']}")
    print()
    print(f"  Bursts detectados:  {results['bursts_detected']}/{results['num_bursts']}")
    print(f"  Decoys detectados:  {results['decoy_detections']}/{results['num_decoys']}")
    print()
    if results["passed"]:
        print("  ✅ PASSOU: 8/8 bursts detectados, 0/120 decoys disparados")
    else:
        print(f"  ❌ FALHOU: bursts={results['bursts_detected']}/{results['num_bursts']}, "
              f"decoys={results['decoy_detections']}/{results['num_decoys']}")
    print()


if __name__ == "__main__":
    main()
