#!/usr/bin/env python3
"""BTB Benchmark Honesto v4 — min-of-N medido com warmup global.

Principios:
  1. Warmup GLOBAL: todos os tamanhos de tensor antes de qualquer medicao
  2. min-of-N: 10 repeticoes, pega o MINIMO (GPU benchmark standard)
  3. torch.cuda.synchronize() entre cada medicao
  4. T_full, T_prefix, T_suffix medidos com os mesmos blocos reais
  5. T_clone medido com tensores nos shapes exatos do estado KV
  6. Speedup = N*T_full / (T_prefix + T_clone + N*T_suffix)
  7. Ideal = N*T_full / (T_prefix + N*T_suffix)  [clone custo zero]
  8. Reporta TODOS os numeros, inclusive negativos

Uso:
    python research/btb/bench_honest.py --config src/configs/darwin_x_600m.yaml
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

from f51_darwin.config import coerce_mapping
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel


def build_model(config_path: str, device: torch.device) -> tuple[DarwinXModel, DarwinXConfig]:
    import yaml
    raw = yaml.safe_load((ROOT / config_path).read_text(encoding="utf-8"))
    coerced = coerce_mapping(DarwinXConfig, raw)
    fields = set(DarwinXConfig.__dataclass_fields__)
    cfg = DarwinXConfig(**{k: v for k, v in coerced.items() if k in fields})
    model = DarwinXModel(cfg).to(device).eval()
    return model, cfg


@torch.no_grad()
def real_forward(model: DarwinXModel, input_ids: torch.Tensor) -> torch.Tensor:
    x = model.token_embedding(input_ids)
    for block in model.blocks:
        hidden, _aux = block(x)
        x = hidden
    return model.norm(x)


_HAS_CUDA = torch.cuda.is_available()


def measure_min_ms(fn, *, warmup: int = 5, repeats: int = 10) -> float:
    """Minimo de N repeticoes — padrao ouro para benchmark GPU."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        if _HAS_CUDA:
            torch.cuda.synchronize()
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            fn()
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))
        else:
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000.0)
    return min(times)


def build_kv_state_tensors(model: DarwinXModel, config: DarwinXConfig,
                            prefix_tokens: int, device: torch.device) -> tuple[list[torch.Tensor], float]:
    """Constroi tensores com shapes EXATOS do estado KV + bytes total."""
    tensors = []
    d_inner = config.d_model * config.ssm_expand
    conv_kernel = 4

    for block in model.blocks:
        if block.is_attention_layer:
            k = torch.randn(1, config.n_kv_heads, prefix_tokens,
                           config.head_dim, device=device, dtype=torch.float32)
            v = torch.randn(1, config.n_kv_heads, prefix_tokens,
                           config.head_dim, device=device, dtype=torch.float32)
            tensors.append(k)
            tensors.append(v)
        else:
            h = torch.randn(1, d_inner, config.ssm_state,
                           device=device, dtype=torch.float32)
            c = torch.randn(1, d_inner, conv_kernel - 1,
                           device=device, dtype=torch.float32)
            tensors.append(h)
            tensors.append(c)

    total_bytes = sum(t.numel() * t.element_size() for t in tensors)
    return tensors, total_bytes


def run_benchmark(
    model: DarwinXModel,
    config: DarwinXConfig,
    device: torch.device,
    prefix_tokens: int,
    suffix_tokens: int,
    burst_size: int,
) -> dict[str, Any]:
    """Benchmark HONESTO: min-of-N, GPU sincronizada, todos os caminhos
    com warmup ANTES de qualquer medicao."""

    # Tokens
    prefix_ids = torch.randint(0, config.vocab_size, (1, prefix_tokens),
                               dtype=torch.long, device=device)
    suffix_ids = torch.randint(0, config.vocab_size, (1, suffix_tokens),
                               dtype=torch.long, device=device)
    full_ids = torch.cat([prefix_ids, suffix_ids], dim=1)

    # KV state tensors
    kv_state, kv_bytes = build_kv_state_tensors(model, config, prefix_tokens, device)

    # ═══════════════════════════════════════════════════════
    # Warmup GLOBAL: roda TODAS as operacoes antes de medir
    # ═══════════════════════════════════════════════════════
    for _ in range(5):
        real_forward(model, full_ids)
        real_forward(model, prefix_ids)
        real_forward(model, suffix_ids)
        [t.detach().clone() for t in kv_state]
        [t.detach().cpu().clone() for t in kv_state]
    if _HAS_CUDA:
        torch.cuda.synchronize()

    # ═══════════════════════════════════════════════════════
    # Medicoes (min-of-10)
    # ═══════════════════════════════════════════════════════
    t_full = measure_min_ms(lambda: real_forward(model, full_ids))
    t_prefix = measure_min_ms(lambda: real_forward(model, prefix_ids))
    t_suffix = measure_min_ms(lambda: real_forward(model, suffix_ids))

    # Clone GPU (piso teorico — .clone() na GPU)
    t_clone = measure_min_ms(lambda: [t.detach().clone() for t in kv_state])

    # Clone CPU (checkpoint_state real — .cpu().clone() + volta)
    t_clone_cpu = measure_min_ms(lambda: (
        [c.to(device) for c in [t.detach().cpu().clone() for t in kv_state]]
    ))

    kv_mb = kv_bytes / (1024 * 1024)

    # ═══════════════════════════════════════════════════════
    # Speedup formulas
    # ═══════════════════════════════════════════════════════
    cold = burst_size * t_full
    warm_ideal = t_prefix + burst_size * t_suffix     # clone instantaneo
    warm_gpu = t_prefix + t_clone + burst_size * t_suffix
    warm_cpu = t_prefix + t_clone_cpu + burst_size * t_suffix

    def _ratio(c, w):
        return c / w if w > 0 else 0.0

    return {
        "prefix_tokens": prefix_tokens,
        "suffix_tokens": suffix_tokens,
        "burst_size": burst_size,
        "t_full_ms": round(t_full, 2),
        "t_prefix_ms": round(t_prefix, 2),
        "t_suffix_ms": round(t_suffix, 2),
        "t_clone_gpu_ms": round(t_clone, 3),
        "t_clone_cpu_ms": round(t_clone_cpu, 2),
        "kv_state_mb": round(kv_mb, 2),
        "cold_total_ms": round(cold, 1),
        "warm_gpu_ms": round(warm_gpu, 1),
        "warm_cpu_ms": round(warm_cpu, 1),
        "warm_ideal_ms": round(warm_ideal, 1),
        "ratio_gpu": round(_ratio(cold, warm_gpu), 2),
        "ratio_cpu": round(_ratio(cold, warm_cpu), 2),
        "ratio_ideal": round(_ratio(cold, warm_ideal), 2),
        "speedup_gpu_pct": round((_ratio(cold, warm_gpu) - 1.0) * 100, 1),
        "speedup_cpu_pct": round((_ratio(cold, warm_cpu) - 1.0) * 100, 1),
        "speedup_ideal_pct": round((_ratio(cold, warm_ideal) - 1.0) * 100, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="BTB Benchmark Honesto v4")
    parser.add_argument("--config", default="src/configs/darwin_x_100m.yaml")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prefix-tokens", type=int, nargs="+",
                        default=[256, 512, 1024, 2048])
    parser.add_argument("--suffix-tokens", type=int, default=32)
    parser.add_argument("--burst-size", type=int, nargs="+",
                        default=[5, 10, 20, 50])
    parser.add_argument("--out", default="research/btb/honest_results.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"GPU: {gpu_name}")

    model, config = build_model(args.config, device)
    n_params = sum(p.numel() for p in model.parameters())
    kv_per_tok = (2 * config.n_layers * config.n_kv_heads
                  * (config.d_model // config.n_heads)
                  * model.token_embedding.weight.element_size())
    print(f"Modelo: {config.model_name} ({n_params/1e6:.0f}M params)  "
          f"KV/tok: {kv_per_tok/1024:.1f} KiB  "
          f"d_model={config.d_model}  layers={config.n_layers}")
    print()

    print(f"  {'Pref':>6} {'Brst':>5} {'T_full':>8} {'T_pref':>8} {'T_suf':>7} "
          f"{'T_clone':>8} {'KV':>6}MB {'Cold':>8} {'WGpu':>8} {'WIdeal':>8} "
          f"{'Ratio':>7} {'Ideal':>7}")
    print(f"  {'-'*90}")

    all_results = []
    for prefix in args.prefix_tokens:
        for burst in args.burst_size:
            r = run_benchmark(model, config, device,
                            prefix_tokens=prefix,
                            suffix_tokens=args.suffix_tokens,
                            burst_size=burst)
            all_results.append(r)
            note = " ⚠️" if r["speedup_gpu_pct"] < 0 else ""
            print(f"  {prefix:>6} {burst:>5} {r['t_full_ms']:>7.1f}ms "
                  f"{r['t_prefix_ms']:>7.1f}ms {r['t_suffix_ms']:>6.1f}ms "
                  f"{r['t_clone_gpu_ms']:>7.2f}ms {r['kv_state_mb']:>5.1f}MB "
                  f"{r['cold_total_ms']:>7.0f}ms {r['warm_gpu_ms']:>7.0f}ms "
                  f"{r['warm_ideal_ms']:>7.0f}ms "
                  f"{r['ratio_gpu']:>6.2f}x {r['speedup_ideal_pct']:>+6.1f}%"
                  f"{note}")
        print()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gpu": gpu_name,
        "model": config.model_name,
        "method": "min-of-10 com warmup global, GPU sincronizada, blocos REAIS",
        "results": all_results,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Resultados: {out_path}")


if __name__ == "__main__":
    main()
