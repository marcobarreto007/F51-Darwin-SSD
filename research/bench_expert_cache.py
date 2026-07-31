#!/usr/bin/env python
"""Benchmark F51 Expert Cache — MoE latency com/sem cache."""
import sys
from pathlib import Path
import torch
import time
from f51_darwin.moe_layer import MoEConfig, MoELayer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
print(f'{"="*60}')

config = MoEConfig(d_model=384, num_experts=8, experts_per_token=2)

# ── Setup ──
layer_no = MoELayer(config).to(device)
layer_no._cache_enabled = False
layer_no.eval()

layer_yes = MoELayer(config).to(device)
layer_yes._cache_enabled = True
layer_yes.eval()

x = torch.randn(1, 64, 384, device=device)

def sync():
    if device == 'cuda':
        torch.cuda.synchronize()

def bench(layer, x_input, iters=50, warmup=5):
    for _ in range(warmup):
        layer(x_input)
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        layer(x_input)
    sync()
    elapsed = (time.perf_counter() - t0) / iters * 1000
    return elapsed, layer.expert_cache.stats() if hasattr(layer, 'expert_cache') else {}

# ── Benchmark 1: Sem cache ──
t_no, _ = bench(layer_no, x)
print(f'Sem cache:              {t_no:>8.2f} ms')

# ── Benchmark 2: Com cache (cold start) ──
layer_yes.expert_cache.clear()
t_cold, s_cold = bench(layer_yes, x)
print(f'Com cache (cold):       {t_cold:>8.2f} ms  | cache_size={s_cold["size"]}')

# ── Benchmark 3: Com cache (warm — mesmo input) ──
layer_yes.expert_cache.clear()
layer_yes(x)  # preenche cache
sync()
t0 = time.perf_counter()
for _ in range(50):
    layer_yes(x)
sync()
t_warm = (time.perf_counter() - t0) / 50 * 1000
s_warm = layer_yes.expert_cache.stats()
print(f'Com cache (warm 100%):  {t_warm:>8.2f} ms  | hits={s_warm["hit_count"]:>5}  miss={s_warm["miss_count"]:>5}  rate={s_warm["hit_rate"]:.0%}')

# ── Benchmark 4: Com cache (noisy input) ──
layer_yes.expert_cache.clear()
layer_yes(x)  # preenche cache
xn = x + torch.randn_like(x) * 0.01
sync()
t0 = time.perf_counter()
for _ in range(50):
    layer_yes(xn)
sync()
t_noisy = (time.perf_counter() - t0) / 50 * 1000
s_noisy = layer_yes.expert_cache.stats()
print(f'Com cache (noisy 1%):   {t_noisy:>8.2f} ms  | hits={s_noisy["hit_count"]:>5}  miss={s_noisy["miss_count"]:>5}  rate={s_noisy["hit_rate"]:.0%}')

# ── Resumo ──
print()
print(f'{"="*60}')
print(f'  SPEEDUP (warm, 100% hit):  {t_no/t_warm:.1f}x')
print(f'  SPEEDUP (noisy 1%):       {t_no/t_noisy:.1f}x')
print(f'{"="*60}')
