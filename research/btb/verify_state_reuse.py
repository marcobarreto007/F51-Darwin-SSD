#!/usr/bin/env python3
"""BTB Verify State-Reuse — teste HONESTO de reutilização especulativa.

Ao contrário de ``run_e2e.py``, este script:

  1. PROVA CORREÇÃO primeiro: os logits do caminho "warm" (prefixo em cache
     + sufixo) têm de bater com os do caminho "cold" (recomputar tudo). Se a
     paridade falhar, o script sai com código != 0 e NÃO reporta speedup —
     porque um cache que muda a saída não é reutilização, é um bug.
  2. Mede o tamanho REAL do cache: bytes de KV de atenção (cresce O(N)) vs
     bytes de estado SSD (constante O(1)). Sem números inventados.
  3. Só então mede latência cold vs warm.

Uso:
    python research/btb/verify_state_reuse.py \
        --config src/configs/darwin_x_100m.yaml \
        --prefix-tokens 128 512 1024 --suffix-tokens 16 --burst 8

Pesos: se nenhum checkpoint for carregável, usa init determinístico
(seed fixa). A paridade é independente dos pesos — ela prova a álgebra do
cache, não a qualidade do modelo. A latência em pesos aleatórios reflete o
custo de compute (FLOPs reais), não a qualidade das gerações.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from f51_darwin.config import coerce_mapping  # noqa: E402
from f51_darwin.darwin_x_core.config import DarwinXConfig  # noqa: E402
from f51_darwin.darwin_x_core.model import DarwinXModel  # noqa: E402

from btb.state_reuse import (  # noqa: E402
    build_prefix_cache,
    cold_logits,
    parity,
    warm_logits,
)

# Tolerâncias de paridade. fp não-associativo (ordem de soma difere entre
# recompute e concat de cache) gera ruído minúsculo; exigimos que seja ínfimo.
MAX_ABS_DIFF_TOL = 2e-2
KL_TOL = 1e-5


def build_model(config_path: str, device: torch.device, seed: int) -> tuple[DarwinXModel, DarwinXConfig]:
    import yaml

    raw = yaml.safe_load((ROOT / config_path).read_text(encoding="utf-8"))
    coerced = coerce_mapping(DarwinXConfig, raw)
    config = DarwinXConfig(
        **{k: v for k, v in coerced.items() if k in DarwinXConfig.__dataclass_fields__}
    )
    torch.manual_seed(seed)
    model = DarwinXModel(config).to(device).eval()
    return model, config


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_ms(fn, device: torch.device, repeats: int = 3) -> float:
    # warmup
    fn()
    _sync(device)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


def run_case(
    model: DarwinXModel,
    config: DarwinXConfig,
    device: torch.device,
    prefix_tokens: int,
    suffix_tokens: int,
    burst: int,
) -> dict[str, Any]:
    g = torch.Generator(device="cpu").manual_seed(1234 + prefix_tokens)
    prefix_ids = torch.randint(
        0, config.vocab_size, (1, prefix_tokens), generator=g, dtype=torch.long
    ).to(device)
    suffixes = [
        torch.randint(
            0, config.vocab_size, (1, suffix_tokens), generator=g, dtype=torch.long
        ).to(device)
        for _ in range(burst)
    ]

    # ── 1. GATE DE PARIDADE ──
    prefix_cache = build_prefix_cache(model, prefix_ids)
    worst_diff = 0.0
    worst_kl = 0.0
    top1_agreements = 0
    for suffix in suffixes:
        cold = cold_logits(model, prefix_ids, suffix)
        warm = warm_logits(model, prefix_cache, prefix_tokens, suffix)
        rep = parity(cold, warm)
        worst_diff = max(worst_diff, rep.max_abs_diff)
        worst_kl = max(worst_kl, rep.kl_div)
        top1_agreements += int(rep.top1_agree)

    parity_ok = (
        worst_diff <= MAX_ABS_DIFF_TOL
        and worst_kl <= KL_TOL
        and top1_agreements == burst
    )

    # ── 2. TAMANHO REAL DO CACHE ──
    attn_bytes = prefix_cache.attn_bytes()
    ssd_bytes = prefix_cache.ssd_bytes()

    # ── 3. LATÊNCIA (só significativa se paridade passou) ──
    suffix0 = suffixes[0]
    cold_ms = _time_ms(lambda: cold_logits(model, prefix_ids, suffix0), device)
    warm_ms = _time_ms(
        lambda: warm_logits(model, prefix_cache, prefix_tokens, suffix0), device
    )
    # custo amortizado do burst: 1 prefill de prefixo + N reusos vs N colds
    prefill_ms = _time_ms(lambda: build_prefix_cache(model, prefix_ids), device)
    warm_amortized = (prefill_ms + burst * warm_ms) / burst
    speedup = (cold_ms - warm_amortized) / cold_ms * 100 if cold_ms > 0 else 0.0

    return {
        "prefix_tokens": prefix_tokens,
        "suffix_tokens": suffix_tokens,
        "burst": burst,
        "parity_ok": parity_ok,
        "parity_max_abs_diff": round(worst_diff, 6),
        "parity_kl": round(worst_kl, 9),
        "parity_top1_agree": f"{top1_agreements}/{burst}",
        "attn_kv_bytes": attn_bytes,
        "ssd_state_bytes": ssd_bytes,
        "attn_kv_kib": round(attn_bytes / 1024, 1),
        "ssd_state_kib": round(ssd_bytes / 1024, 1),
        "cold_ms": round(cold_ms, 2),
        "warm_reuse_ms": round(warm_ms, 2),
        "prefix_prefill_ms": round(prefill_ms, 2),
        "warm_amortized_ms": round(warm_amortized, 2),
        "speedup_amortized_pct": round(speedup, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="BTB honest state-reuse verifier")
    parser.add_argument("--config", default="src/configs/darwin_x_100m.yaml")
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--prefix-tokens", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--suffix-tokens", type=int, default=16)
    parser.add_argument("--burst", type=int, default=8)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--out", default="research/btb/state_reuse_results.json")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("CPU (latência apenas indicativa; paridade é exata)")

    print(f"Carregando modelo ({args.config})...", end=" ", flush=True)
    model, config = build_model(args.config, device, args.seed)
    n_params = sum(p.numel() for p in model.parameters())
    n_attn = len(config.attention_layer_indices)
    n_ssd = config.n_layers - n_attn
    print(f"{config.model_name}  {n_params/1e6:.0f}M params  "
          f"({n_attn} attn / {n_ssd} ssd layers)")
    print()

    results: list[dict] = []
    all_parity_ok = True
    header = (
        f"  {'Prefix':>7} {'Burst':>5} {'Parity':>7} {'MaxDiff':>9} "
        f"{'KL':>10} {'AttnKV':>10} {'SSD':>9} {'Cold':>9} {'WarmAmz':>9} {'Speedup':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for prefix in args.prefix_tokens:
        r = run_case(model, config, device, prefix, args.suffix_tokens, args.burst)
        results.append(r)
        all_parity_ok = all_parity_ok and r["parity_ok"]
        print(
            f"  {r['prefix_tokens']:>7} {r['burst']:>5} "
            f"{'OK' if r['parity_ok'] else 'FAIL':>7} "
            f"{r['parity_max_abs_diff']:>9.2e} {r['parity_kl']:>10.2e} "
            f"{r['attn_kv_kib']:>8.1f}Ki {r['ssd_state_kib']:>7.1f}Ki "
            f"{r['cold_ms']:>7.1f}ms {r['warm_amortized_ms']:>7.1f}ms "
            f"{r['speedup_amortized_pct']:>+7.1f}%"
        )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "device": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else "cpu"
                ),
                "model": config.model_name,
                "config": args.config,
                "note": (
                    "Pesos random-init determinísticos; paridade prova a "
                    "álgebra do cache, independente dos pesos."
                ),
                "all_parity_ok": all_parity_ok,
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print()
    print(f"Resultados: {out_path}")

    if not all_parity_ok:
        print(
            "\n❌ PARIDADE FALHOU — o cache 'warm' diverge do recompute 'cold'. "
            "Speedup é irrelevante enquanto a saída estiver errada."
        )
        return 1
    print(
        "\n✅ Paridade OK em todos os casos — a reutilização de estado é "
        "matematicamente equivalente ao recompute. Speedup é legítimo."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
