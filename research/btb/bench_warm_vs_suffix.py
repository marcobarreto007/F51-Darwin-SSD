#!/usr/bin/env python3
"""T_suffix (sufixo sozinho, o que bench_honest mede) vs warm real (com cache).

O warm real vem de state_reuse.py, cuja paridade warm==cold ja esta provada
(KL 2.4e-7). Se os dois baterem, os numeros do bench_honest valem.
Se o warm real for muito mais caro, os speedups estao inflados.
"""
from __future__ import annotations
import sys, time, statistics, json, argparse
from pathlib import Path
import torch, yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from f51_darwin.config import coerce_mapping
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from btb.state_reuse import build_prefix_cache, cold_logits, warm_logits, parity

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="src/configs/darwin_x_600m.yaml")
ap.add_argument("--prefix-tokens", type=int, nargs="+", default=[256, 512, 1024, 2048])
ap.add_argument("--suffix-tokens", type=int, default=32)
ap.add_argument("--burst", type=int, default=50)
a = ap.parse_args()

dev = torch.device("cuda:0")
raw = yaml.safe_load((ROOT / a.config).read_text(encoding="utf-8"))
c = coerce_mapping(DarwinXConfig, raw)
cfg = DarwinXConfig(**{k: v for k, v in c.items() if k in DarwinXConfig.__dataclass_fields__})
torch.manual_seed(51)
model = DarwinXModel(cfg).to(dev).eval()
print(f"{cfg.model_name}  {sum(p.numel() for p in model.parameters())/1e6:.0f}M  d_model={cfg.d_model}")
print(f"burst={a.burst}  suffix={a.suffix_tokens}\n")


@torch.no_grad()
def plain_forward(ids):
    """Exatamente o real_forward do bench_honest.py: sem cache nenhum."""
    x = model.token_embedding(ids)
    for b in model.blocks:
        h, _ = b(x)
        x = h
    return model.norm(x)


def mmin(fn, warmup=5, reps=10):
    """min-of-N, mesma metodologia do bench_honest."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return min(ts)


print(f"{'prefix':>7} {'T_full':>9} {'T_suffix':>10} {'WARM_real':>10} {'infl':>6} "
      f"{'spd_bench':>10} {'spd_real':>9} {'paridade':>10}")
print("-" * 85)

rows = []
for P in a.prefix_tokens:
    g = torch.Generator().manual_seed(1234 + P)
    pre = torch.randint(0, cfg.vocab_size, (1, P), generator=g, dtype=torch.long).to(dev)
    suf = torch.randint(0, cfg.vocab_size, (1, a.suffix_tokens), generator=g, dtype=torch.long).to(dev)
    full = torch.cat([pre, suf], dim=1)

    pc = build_prefix_cache(model, pre)

    # gate de paridade: o warm real esta correto?
    rep = parity(cold_logits(model, pre, suf), warm_logits(model, pc, P, suf))
    par_ok = rep.kl_div <= 1e-5 and rep.top1_agree

    t_full = mmin(lambda: plain_forward(full))
    t_prefix = mmin(lambda: plain_forward(pre))
    t_suffix = mmin(lambda: plain_forward(suf))          # o que o bench_honest usa
    t_warm = mmin(lambda: warm_logits(model, pc, P, suf))  # o custo warm REAL

    N = a.burst
    cold_total = N * t_full
    warm_bench = t_prefix + N * t_suffix   # modelo do bench_honest
    warm_real = t_prefix + N * t_warm      # com o custo warm verdadeiro
    spd_bench = cold_total / warm_bench
    spd_real = cold_total / warm_real
    infl = t_warm / t_suffix

    print(f"{P:>7} {t_full:>8.1f}ms {t_suffix:>9.1f}ms {t_warm:>9.1f}ms "
          f"{infl:>5.2f}x {spd_bench:>9.2f}x {spd_real:>8.2f}x {'OK' if par_ok else 'FAIL':>10}")
    rows.append(dict(prefix=P, t_full=t_full, t_prefix=t_prefix, t_suffix=t_suffix,
                     t_warm=t_warm, inflation=infl, spd_bench=spd_bench,
                     spd_real=spd_real, parity_ok=par_ok, kl=rep.kl_div))

print("\nT_suffix cresce com o prefixo?  (warm real DEVE crescer; sufixo-sozinho nao)")
for r in rows:
    print(f"  prefixo {r['prefix']:>5}:  T_suffix {r['t_suffix']:7.1f} ms   "
          f"WARM_real {r['t_warm']:7.1f} ms")

out = ROOT / "research/btb/warm_vs_suffix_results.json"
out.write_text(json.dumps({"model": cfg.model_name, "burst": a.burst, "rows": rows}, indent=2))
print(f"\n-> {out}")
