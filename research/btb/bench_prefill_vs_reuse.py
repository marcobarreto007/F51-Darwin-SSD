#!/usr/bin/env python3
"""BTB Prefill-vs-Reuse Benchmark — Fase 1.

Mede a razao real de custo entre recomputar um prefixo (prefill) e reusar
seu KV cache ja computado, no hardware real do Darwin-X.

Inspirado no ``1-prefill-vs-transfer/compute_costs.py`` do BTB original,
mas operando com o modelo Darwin-X-100M real em vez de um simulador.

Uso:
    python research/btb/bench_prefill_vs_reuse.py

Output:
    Tabela de custo por comprimento de prefixo e tamanho de burst,
    com razao prefill/reuse e speedup percentual.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from f51_darwin.config import coerce_mapping
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.kv_cache import KVCache

# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════


def build_config(config_path: str = "src/configs/darwin_x_100m.yaml") -> DarwinXConfig:
    """Constroi DarwinXConfig a partir do YAML canonico."""
    import yaml
    raw = yaml.safe_load((ROOT / config_path).read_text(encoding="utf-8"))
    coerced = coerce_mapping(DarwinXConfig, raw)
    return DarwinXConfig(**{k: v for k, v in coerced.items() if k in DarwinXConfig.__dataclass_fields__})


def build_model(config: DarwinXConfig, device: torch.device) -> DarwinXModel:
    """Instancia o modelo 100M com pesos aleatorios e move para GPU."""
    model = DarwinXModel(config)
    model = model.to(device)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Prefill / generation primitives (adaptados de kv_cache.py)
# ═══════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def prefill_with_cache(
    model: DarwinXModel,
    input_ids: torch.Tensor,
    cache: KVCache,
) -> None:
    """Prefill: processa o prompt inteiro, preenchendo o KVCache.

    Nao gera tokens — apenas popula o cache. O ultimo hidden state
    fica disponivel para o primeiro passo de decode.

    NOTA: DarwinXBlock.forward() retorna (tensor, aux_dict). O aux_dict
    e descartado aqui (benchmark de performance, nao de treino).
    """
    x = model.token_embedding(input_ids)
    seq_len = input_ids.shape[1]
    cache._seq_len = seq_len
    hidden = x

    for i, block in enumerate(model.blocks):
        # DarwinXBlock.forward() sempre retorna (tensor, dict)
        hidden, _aux = block(hidden)

    hidden = model.norm(hidden)
    # Guarda o ultimo hidden state para decode posterior
    cache._last_hidden = hidden[:, -1:, :].detach()


# ═══════════════════════════════════════════════════════════════════════════
# Timing helpers (CUDA events quando disponivel, perf_counter fallback)
# ═══════════════════════════════════════════════════════════════════════════

_HAS_CUDA = torch.cuda.is_available()
_CUDA_STARTER: Any = None
_CUDA_ENDER: Any = None


def _init_timers() -> None:
    global _CUDA_STARTER, _CUDA_ENDER
    if _HAS_CUDA:
        _CUDA_STARTER = torch.cuda.Event(enable_timing=True)
        _CUDA_ENDER = torch.cuda.Event(enable_timing=True)


def _measure_ms(fn, *args, **kwargs) -> float:
    """Executa fn(*args, **kwargs) e retorna tempo em millissegundos."""
    if _HAS_CUDA and _CUDA_STARTER is not None:
        torch.cuda.synchronize()
        _CUDA_STARTER.record()
        fn(*args, **kwargs)
        _CUDA_ENDER.record()
        torch.cuda.synchronize()
        return _CUDA_STARTER.elapsed_time(_CUDA_ENDER)
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return (time.perf_counter() - t0) * 1000.0


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark core
# ═══════════════════════════════════════════════════════════════════════════


def benchmark_burst(
    model: DarwinXModel,
    config: DarwinXConfig,
    device: torch.device,
    prefix_tokens: int,
    suffix_tokens: int,
    num_requests: int,
    num_warmup: int = 2,
    num_repeats: int = 3,
) -> dict[str, Any]:
    """Mede TTFT cold vs warm para um burst de requests com mesmo prefixo.

    Args:
        model: modelo Darwin-X em eval mode
        config: DarwinXConfig com parametros do modelo
        device: torch device (cpu ou cuda)
        prefix_tokens: tokens compartilhados entre todos os requests
        suffix_tokens: tokens unicos por request (sufixo)
        num_requests: total de requests no burst
        num_warmup: warmup iterations (descartadas da medicao)
        num_repeats: repeticoes por medicao

    Returns:
        Dict com cold_ttft_ms, warm_ttft_ms, ratio, speedup_pct, etc.
    """
    max_len = prefix_tokens + suffix_tokens + 64  # margem para decode
    batch_size = 1

    _init_timers()

    # Gera prefixo compartilhado (tokens aleatorios no vocab)
    prefix_ids = torch.randint(
        0, config.vocab_size,
        (1, prefix_tokens),
        dtype=torch.long, device=device,
    )

    # ═══════════════════════════════════════════════════════
    # Cold: prefill do prompt completo toda vez
    # ═══════════════════════════════════════════════════════
    cold_times: list[float] = []

    for r in range(num_requests + num_warmup):
        suffix_ids = torch.randint(
            0, config.vocab_size,
            (1, suffix_tokens),
            dtype=torch.long, device=device,
        )
        full_ids = torch.cat([prefix_ids, suffix_ids], dim=1)

        cache = KVCache(batch_size, max_len, device)
        t_ms = _measure_ms(prefill_with_cache, model, full_ids, cache)
        if r >= num_warmup:
            cold_times.append(t_ms)

    cold_mean = sum(cold_times) / len(cold_times)

    # ═══════════════════════════════════════════════════════
    # Warm: prefill do prefixo uma vez, checkpoint, restore
    # ═══════════════════════════════════════════════════════
    warm_setup_times: list[float] = []
    warm_reuse_times: list[float] = []

    for rep in range(num_repeats):
        # Prefill do prefixo UMA vez
        prefix_cache = KVCache(batch_size, max_len, device)
        setup_ms = _measure_ms(prefill_with_cache, model, prefix_ids, prefix_cache)
        if rep >= num_repeats - 1:
            warm_setup_times.append(setup_ms)

        # Checkpoint do estado KV
        checkpoint = prefix_cache.checkpoint_state()

        # Restaura para cada request do burst
        for r in range(num_requests + num_warmup):
            suffix_ids = torch.randint(
                0, config.vocab_size,
                (1, suffix_tokens),
                dtype=torch.long, device=device,
            )

            reused_cache = KVCache(batch_size, max_len, device)

            def _restore_and_prefill():
                reused_cache.restore_state(checkpoint, target_device=device)
                prefill_with_cache(model, suffix_ids, reused_cache)

            t_ms = _measure_ms(_restore_and_prefill)
            if r >= num_warmup:
                warm_reuse_times.append(t_ms)

    warm_setup_mean = sum(warm_setup_times) / len(warm_setup_times) if warm_setup_times else 0.0
    warm_reuse_mean = sum(warm_reuse_times) / len(warm_reuse_times)

    # Custo total amortizado: setup + N * reuse
    warm_amortized = (warm_setup_mean + num_requests * warm_reuse_mean) / num_requests
    ratio = cold_mean / warm_reuse_mean if warm_reuse_mean > 0 else float("inf")
    speedup = (cold_mean - warm_amortized) / cold_mean * 100 if cold_mean > 0 else 0.0

    # KV cache size
    kv_per_tok = (
        2 * config.n_layers * config.n_kv_heads
        * (config.d_model // config.n_heads)
        * model.token_embedding.weight.element_size()
    )

    return {
        "prefix_tokens": prefix_tokens,
        "suffix_tokens": suffix_tokens,
        "num_requests": num_requests,
        "kv_size_kib": round(kv_per_tok * prefix_tokens / 1024, 1),
        "cold_ttft_ms": round(cold_mean, 3),
        "warm_setup_ms": round(warm_setup_mean, 3),
        "warm_reuse_ms": round(warm_reuse_mean, 3),
        "warm_amortized_ms": round(warm_amortized, 3),
        "prefill_reuse_ratio": round(ratio, 1),
        "speedup_pct": round(speedup, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="BTB Prefill-vs-Reuse Benchmark")
    parser.add_argument("--config", default="src/configs/darwin_x_100m.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prefix-tokens", type=int, nargs="+",
                        default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--suffix-tokens", type=int, nargs="+",
                        default=[16, 64, 128])
    parser.add_argument("--num-requests", type=int, nargs="+",
                        default=[2, 5, 10, 20, 50])
    parser.add_argument("--out", default="research/btb/results.json")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("CUDA indisponivel — rodando em CPU (tempos nao representativos)")
    else:
        gpu_name = torch.cuda.get_device_name(device)
        gpu_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_mem:.0f} GB)")

    print(f"Carregando config: {args.config}")
    config = build_config(args.config)
    model = build_model(config, device)

    n_params = sum(p.numel() for p in model.parameters())
    kv_per_tok = 2 * config.n_layers * config.n_kv_heads * (config.d_model // config.n_heads) * model.token_embedding.weight.element_size()
    print(f"Modelo: {config.model_name}  params={n_params/1e6:.1f}M  "
          f"d_model={config.d_model}  n_layers={config.n_layers}  "
          f"KV/tok={kv_per_tok/1024:.1f} KiB")
    print()

    results: list[dict] = []

    for suffix in args.suffix_tokens:
        print(f"{'='*70}")
        print(f"  Sufixo: {suffix} tokens")
        print(f"  {'Prefix':>8} {'Burst':>6} {'Cold':>9} {'Setup':>9} {'Reuse':>9} "
              f"{'Amort':>9} {'Ratio':>7} {'Speedup':>8}")
        print(f"  {'-'*66}")

        for prefix in args.prefix_tokens:
            for n_req in args.num_requests:
                r = benchmark_burst(
                    model, config, device,
                    prefix_tokens=prefix,
                    suffix_tokens=suffix,
                    num_requests=n_req,
                )
                results.append(r)
                print(f"  {prefix:>8} {n_req:>6} {r['cold_ttft_ms']:>8.1f}ms "
                      f"{r['warm_setup_ms']:>8.1f}ms {r['warm_reuse_ms']:>8.2f}ms "
                      f"{r['warm_amortized_ms']:>8.2f}ms {r['prefill_reuse_ratio']:>6.0f}x "
                      f"{r['speedup_pct']:>+7.1f}%")
            print()

        print()

    # Sumario final (estilo BTB)
    print(f"{'='*70}")
    print("Sumario — Razao Prefill/Reuse por comprimento de prefixo:")
    print(f"  {'Prefix':>8} {'Cold':>9} {'Reuse':>9} {'Ratio':>7}")
    print(f"  {'-'*35}")
    for prefix in args.prefix_tokens:
        subset = [r for r in results if r["prefix_tokens"] == prefix and r["suffix_tokens"] == args.suffix_tokens[0]]
        if subset:
            r = subset[len(subset) // 2]  # burst medio
            print(f"  {prefix:>8} {r['cold_ttft_ms']:>8.1f}ms {r['warm_reuse_ms']:>8.2f}ms "
                  f"{r['prefill_reuse_ratio']:>6.0f}x")

    # Salva resultados
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "model": config.model_name,
        "config": args.config,
        "kv_kib_per_token": kv_per_tok / 1024,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nResultados salvos em: {out_path}")


if __name__ == "__main__":
    main()
