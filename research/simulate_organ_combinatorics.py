#!/usr/bin/env python3
"""Organ Sequence Combinatorial Simulator — F51 Darwin-X
CPU-only, numpy puro, deterministico. 72 combinações válidas × 50 cenários."""

from __future__ import annotations

import json, os, sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any
import numpy as np

np.random.seed(42); RNG = np.random.RandomState(42)
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "workspace" / "runtime" / "organ_sequence_sim"

ORGANS: dict[str, dict[str, Any]] = {
    "jepa": {"params":659712,"type":"sidecar","tap":"post_ln_f","depends":[],"required_by":["heartbeat"],"compute":0.03,"retention":0.01,"quality":0.08,"future":0.05},
    "gaba": {"params":3151884,"type":"residual","tap":"post_mlp","depends":[],"required_by":[],"compute":0.02,"retention":0.05,"quality":0.05,"future":0.03,"self_interference":0.02},
    "spider":{"params":65793,"type":"sidecar","tap":"post_ln_f","depends":[],"required_by":["ttm"],"compute":0.01,"retention":0.0,"quality":0.02,"future":0.01},
    "ttm":   {"params":1,"type":"residual","tap":"post_ln_f","depends":["spider"],"required_by":[],"compute":0.005,"retention":0.02,"quality":0.10,"future":0.08},
    "ihs":   {"params":6567424,"type":"global","tap":"pre_blocks","depends":[],"required_by":[],"compute":0.15,"retention":0.10,"quality":0.12,"future":0.06},
    "heartbeat":{"params":857729,"type":"meta","tap":"detached","depends":["jepa"],"required_by":[],"compute":0.01,"retention":0.01,"quality":0.03,"future":0.02},
    "mtp":   {"params":524288,"type":"aux","tap":"multi","depends":[],"required_by":[],"compute":0.05,"retention":0.03,"quality":0.04,"future":0.03},
}
ORGAN_NAMES = sorted(ORGANS)

def is_valid_subset(active): return all(dep in active for o in active for dep in ORGANS[o]["depends"])

def generate_valid_subsets():
    valid = []
    for r in range(len(ORGAN_NAMES)+1):
        for combo in combinations(ORGAN_NAMES, r):
            s = set(combo)
            if is_valid_subset(s): valid.append(s)
    return valid

def compute_utility(active, scenario):
    tokens = scenario.get("tokens", 1_000_000)
    budget = min(1.0, np.log10(max(tokens,1)/10000.0)/4.0)
    quality, future, retention_loss, compute = 0.0, 0.0, 0.0, 0.0
    for o in active:
        q = ORGANS[o]["quality"] * budget
        f = ORGANS[o]["future"] * budget
        r = ORGANS[o]["retention"]
        c = ORGANS[o]["compute"] * scenario.get("adapter_cost", 0.05)
        if "position_penalty" in scenario and o in scenario["position_penalty"]:
            pos = scenario["position_penalty"][o]
            q *= 1.0 - pos * 0.5; f *= 1.0 - pos * 0.5
        deps_met = all(d in active for d in ORGANS[o]["depends"])
        if not deps_met: q *= 0.4; f *= 0.4
        quality += q; future += f; retention_loss += r; compute += c
    interference = 0.0
    active_list = sorted(active)
    for i in range(len(active_list)):
        for j in range(i+1, len(active_list)):
            a, b = active_list[i], active_list[j]
            if ORGANS[a]["tap"] == ORGANS[b]["tap"]:
                interference += scenario.get("interference", 0.1) * 0.01
    quality *= 1.0 - interference; future *= 1.0 - interference
    return quality + future - retention_loss - compute

def greedy_sequence(scenario, max_steps=None):
    active, sequence, marginals = set(), [], []
    remaining = set(ORGAN_NAMES)
    while remaining:
        best_organ, best_marginal = None, -float("inf")
        for o in sorted(remaining):
            if not all(d in active for d in ORGANS[o]["depends"]): continue
            new_u = compute_utility(active | {o}, scenario)
            cur_u = compute_utility(active, scenario) if active else 0.0
            if new_u - cur_u > best_marginal: best_marginal = new_u - cur_u; best_organ = o
        if best_organ is None: break
        active.add(best_organ); sequence.append(best_organ)
        marginals.append(round(best_marginal, 8))
        remaining.discard(best_organ)
        if max_steps and len(sequence) >= max_steps: break
    return sequence, marginals

def generate_scenarios():
    scenarios = []; sid = 0
    for tokens in [10_000,50_000,100_000,500_000,1_000_000,5_000_000,10_000_000,50_000_000,100_000_000,500_000_000]:
        sid+=1; scenarios.append({"id":f"A{sid:03d}","type":"token_budget","tokens":tokens,"interference":0.1,"adapter_cost":0.05})
    for idx, seed in enumerate([42,123,456,789,1011,1213,1415,1617,1819,2021]):
        sid+=1; rng=np.random.RandomState(seed); perm=ORGAN_NAMES.copy(); rng.shuffle(perm)
        scenarios.append({"id":f"B{sid:03d}","type":"activation_sequence","tokens":1_000_000,"interference":0.1,"adapter_cost":0.05,"permutation":perm,"position_penalty":{org:i/len(perm) for i,org in enumerate(perm)}})
    for k in [1,2,3,4,5,6,7,"greedy_3","greedy_5","all"]:
        sid+=1; scenarios.append({"id":f"C{sid:03d}","type":"topk_selection","tokens":1_000_000,"interference":0.1,"adapter_cost":0.05,"k":k})
    for rate in [0.0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.75]:
        sid+=1; scenarios.append({"id":f"D{sid:03d}","type":"interference","tokens":1_000_000,"interference":rate,"adapter_cost":0.05})
    for mult in [0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0]:
        sid+=1; scenarios.append({"id":f"E{sid:03d}","type":"adapter_cost","tokens":1_000_000,"interference":0.1,"adapter_cost":mult})
    return scenarios

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    scenarios = generate_scenarios()
    valid_subsets = generate_valid_subsets()
    print(f"Scenarios: {len(scenarios)} | Valid subsets: {len(valid_subsets)}/128 | Evals: {len(scenarios)*len(valid_subsets)}")

    results = []
    for sc in scenarios:
        for subset in valid_subsets:
            results.append({"scenario_id":sc["id"],"scenario_type":sc["type"],"active_organs":sorted(subset),"num_active":len(subset),"utility":round(compute_utility(subset,sc),6)})

    with open(OUT_DIR/"results.jsonl","w",encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r)+"\n")
    print(f"Wrote {len(results):,} results to {OUT_DIR/'results.jsonl'}")

    greedy_results = {}
    for sc in scenarios:
        max_steps = None
        if sc["type"]=="topk_selection":
            k = sc["k"]
            if isinstance(k,int): max_steps = k
            elif k=="greedy_3": max_steps = 3
            elif k=="greedy_5": max_steps = 5
        seq, utils = greedy_sequence(sc, max_steps=max_steps)
        greedy_results[sc["id"]] = {"sequence":seq,"marginal_utilities":utils,"final_utility":round(sum(utils),6)}

    # Report
    default_sc = {"tokens":1_000_000,"interference":0.1,"adapter_cost":0.05}
    all_seqs = [tuple(gr["sequence"]) for gr in greedy_results.values() if gr["sequence"]]
    consensus = Counter(all_seqs).most_common(1)

    print("\n" + "="*60)
    print("CONSENSUS SEQUENCE")
    if consensus:
        seq, count = consensus[0]
        print(f"  {' >> '.join(seq)}  ({count}/50 scenarios)")
    print("\nMarginal utilities (default 1M tokens):")
    for o in ORGAN_NAMES:
        deps = set(ORGANS[o]["depends"])
        marginal = compute_utility(deps|{o},default_sc) - compute_utility(deps,default_sc)
        standalone = compute_utility({o},default_sc)
        print(f"  {o:12s} standalone={standalone:+.4f}  marginal={marginal:+.4f}")
    full_u = compute_utility(set(ORGAN_NAMES),default_sc)
    print(f"\nFull set (7): {full_u:+.4f} | Empty: 0.0000")
    print(f"Output: {OUT_DIR}")
    return 0

if __name__=="__main__": sys.exit(main())
