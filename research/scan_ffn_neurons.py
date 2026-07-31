#!/usr/bin/env python3
"""FFN Neuron Scanner — stamps all 196,608 intermediate neurons across 24 layers.

Each DenseSwiGLU has 8192 intermediate neurons (SiLU(gate) * up). This scanner
taps the pre-hook of down_proj to read the intermediate activation, classifies
each neuron by its preference between factual and syntactic probes, and stamps
it with the SHA-256 of its own down_proj output column.

SAMPLING CONTRACT — v1 of this scanner produced labels that were barely more
reliable than a coin flip (split-half Cohen's kappa 0.19 at layers 10/16/22).
Four causes were identified and are addressed here:

  1. POSITION. v1 read only the final token (`[0, -1, :]`). Because every
     factual probe ends in a period, it was measuring activation at the
     sentence-final punctuation. v2 averages |activation| over every real
     token position of the probe.

  2. BALANCE. v1 compared 16 factual probes against 8 syntactic ones. v2 uses
     16 against 16 and records the token counts of both families, since
     activation magnitude tracks sequence length.

  3. ABSTENTION. v1 labelled all 196,608 neurons with `f > s`, so a dead
     neuron (f = s = 0) always became "syntactic". v2 emits four labels:
     `inactive` (below the layer's activity floor), `mixed` (no significant
     preference), `factual`, `syntactic`. A label is only assigned when a
     Welch t-statistic over the per-probe scores clears a threshold, so
     "confidence" reflects separation against within-family variance rather
     than the ratio max(f,s)/(f+s), which cannot fall below 0.5 and so made
     a coin flip look like 50% confidence.

  4. SELF-MEASUREMENT. v2 computes split-half reliability during the scan and
     writes Cohen's kappa per layer into the artifact, so the catalog carries
     its own reliability rather than requiring a separate audit.

STILL NOT ESTABLISHED: this is activation magnitude, not causal role. A neuron
can respond strongly to factual probes and carry none of the fact. Establishing
causal role requires ablation (zero the neuron, measure delta loss). Do not
treat `label == "factual"` as evidence that a neuron encodes a fact.
"""

from __future__ import annotations

import argparse, json, os, time
from pathlib import Path
from typing import Any

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer
from f51_darwin.transplant_16b.cli import DEFAULT_SOURCE_ROOT
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.hashing import sha256_file, tensor_sha256, atomic_json_write

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt"
OUTPUT = ROOT / "workspace/runtime/circuit-catalog"
SCHEMA = "darwin-ffn-neuron-catalog-v2"

# Fix 2: both families hold 16 probes of comparable length.
FACT_PROBES = [
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius.",
    "The Earth orbits around the Sun.",
    "DNA contains genetic information.",
    "The speed of light is approximately 300000 km per second.",
    "Photosynthesis converts sunlight into energy.",
    "The human body has 206 bones.",
    "Shakespeare wrote Romeo and Juliet.",
    "World War 2 ended in 1945.",
    "The chemical formula of water is H2O.",
    "Brazil is the largest country in South America.",
    "The Amazon rainforest produces 20% of the worlds oxygen.",
    "Mount Everest is the highest mountain on Earth.",
    "The Pacific Ocean is the largest ocean.",
    "Bees produce honey from nectar.",
    "The Great Wall of China is over 21000 kilometers long.",
]

SYNTACTIC_PROBES = [
    "The quick brown fox jumps over the lazy dog.",
    "If it rains tomorrow, I will stay home.",
    "She said that she would come, but she did not.",
    "Despite being tired, he finished the work on time.",
    "When I arrived at the station, the train had already left.",
    "The book that I bought yesterday is very interesting.",
    "Neither the teacher nor the students were satisfied.",
    "Having finished the exam, she left the room quietly.",
    "Whoever arrives first should unlock the door and wait inside.",
    "The letter was written by someone who preferred to remain unknown.",
    "Not only did he apologize, but he also offered to help.",
    "Although she tried repeatedly, the door would not open.",
    "The person whom you mentioned earlier has just called again.",
    "Had I known about it sooner, I would have acted differently.",
    "They kept walking until the path became too narrow to follow.",
    "What surprised everyone was how calmly she responded.",
]


# ── Activation collection ───────────────────────────────────────────────────

@torch.no_grad()
def collect_scores(model, tokenizer, probes: list[str], layers: list[int],
                   device: torch.device, max_len: int) -> tuple[torch.Tensor, int]:
    """Per-probe mean |activation| at every layer's down_proj input.

    Fix 1: averages over ALL real token positions instead of the last one.
    Every probe is run once with hooks on all layers at once, so the cost is
    len(probes) forwards rather than len(probes) * len(layers).

    Returns:
        scores: [n_probes, n_layers, d_ff]
        tokens: total token count across the probe family
    """
    d_ff = model.blocks[layers[0]].ffn.down_proj.weight.shape[1]
    scores = torch.zeros(len(probes), len(layers), d_ff)
    total_tokens = 0

    for pi, probe in enumerate(probes):
        ids = tokenizer(probe, add_special_tokens=False,
                        return_tensors="pt").input_ids[:, :max_len].to(device)
        total_tokens += int(ids.shape[1])

        captured: dict[int, torch.Tensor] = {}
        handles = []
        for slot, li in enumerate(layers):
            def make_hook(idx: int):
                def pre_hook(module, args):
                    # args[0] is [B, T, d_ff]; mean over all token positions.
                    captured[idx] = args[0].detach().float().abs().mean(dim=1)[0].cpu()
                return pre_hook
            handles.append(
                model.blocks[li].ffn.down_proj.register_forward_pre_hook(make_hook(slot))
            )
        try:
            model(ids, heartbeat=False)
        finally:
            for handle in handles:
                handle.remove()

        for slot in range(len(layers)):
            scores[pi, slot] = captured[slot]

    return scores, total_tokens


# ── Classification ──────────────────────────────────────────────────────────

def welch_t(fact: torch.Tensor, syn: torch.Tensor) -> torch.Tensor:
    """Welch t-statistic per neuron between the two probe families.

    fact: [n_f, d_ff]   syn: [n_s, d_ff]   ->   [d_ff]

    Positive means the neuron prefers factual probes. Using the within-family
    variance is the point: it asks whether the gap is large relative to how
    much each family's own probes disagree, which a raw f > s comparison
    cannot see.
    """
    n_f, n_s = fact.shape[0], syn.shape[0]
    mean_f, mean_s = fact.mean(dim=0), syn.mean(dim=0)
    var_f = fact.var(dim=0, unbiased=True) / n_f
    var_s = syn.var(dim=0, unbiased=True) / n_s
    denom = (var_f + var_s).clamp_min(1e-12).sqrt()
    return (mean_f - mean_s) / denom


def classify(fact: torch.Tensor, syn: torch.Tensor, t_threshold: float,
             floor_frac: float) -> tuple[list[str], torch.Tensor, torch.Tensor, float]:
    """Four-way label per neuron.

    Fix 3: a neuron is only called factual or syntactic when it is both
    active enough to matter and separated enough to be reproducible.

    The activity floor is a fraction of the layer's MEDIAN activity, not a
    quantile of it. A quantile floor is tautological -- asking for the bottom
    decile always marks exactly 10% of neurons whether or not any of them are
    actually dead. A fraction of the median lets the count vary per layer and
    reach zero when a layer has no silent neurons.
    """
    mean_f, mean_s = fact.mean(dim=0), syn.mean(dim=0)
    activity = torch.maximum(mean_f, mean_s)
    floor = activity.median() * floor_frac

    t_stat = welch_t(fact, syn)

    labels: list[str] = []
    for i in range(activity.shape[0]):
        if activity[i] < floor:
            labels.append("inactive")
        elif t_stat[i] > t_threshold:
            labels.append("factual")
        elif t_stat[i] < -t_threshold:
            labels.append("syntactic")
        else:
            labels.append("mixed")

    return labels, t_stat, activity, float(floor)


# ── Split-half reliability (fix 4) ──────────────────────────────────────────

def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Multi-category Cohen's kappa between two labellings."""
    categories = sorted(set(a) | set(b))
    index = {c: i for i, c in enumerate(categories)}
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n

    count_a = [0] * len(categories)
    count_b = [0] * len(categories)
    for x, y in zip(a, b):
        count_a[index[x]] += 1
        count_b[index[y]] += 1
    expected = sum((count_a[i] / n) * (count_b[i] / n) for i in range(len(categories)))

    if expected >= 1.0:
        return float("nan")
    return (observed - expected) / (1.0 - expected)


def split_half_reliability(fact: torch.Tensor, syn: torch.Tensor,
                           t_threshold: float, floor_frac: float) -> dict[str, float]:
    """Label the layer twice from disjoint probe halves and compare.

    This is a LOWER BOUND on the reliability of the full-data labelling:
    each half sees half the probes, so its t-statistics are noisier.
    """
    labels_a, _, _, _ = classify(fact[0::2], syn[0::2], t_threshold, floor_frac)
    labels_b, _, _, _ = classify(fact[1::2], syn[1::2], t_threshold, floor_frac)

    agreement = sum(1 for x, y in zip(labels_a, labels_b) if x == y) / len(labels_a)

    # Restricted to neurons both halves called decisively.
    decisive = [(x, y) for x, y in zip(labels_a, labels_b)
                if x in ("factual", "syntactic") and y in ("factual", "syntactic")]
    decisive_agreement = (sum(1 for x, y in decisive if x == y) / len(decisive)
                          if decisive else float("nan"))

    return {
        "agreement": round(agreement, 4),
        "kappa": round(cohen_kappa(labels_a, labels_b), 4),
        "decisive_pairs": len(decisive),
        "decisive_agreement": (round(decisive_agreement, 4)
                               if decisive else None),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="FFN Neuron Scanner v2")
    parser.add_argument("--checkpoint", default=str(CKPT))
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--t-threshold", type=float, default=2.0,
                        help="Welch |t| required to assign factual/syntactic")
    parser.add_argument("--floor-frac", type=float, default=0.01,
                        help="Inactive if activity < this fraction of the "
                             "layer's median activity")
    args = parser.parse_args()

    print("=" * 70)
    print("FFN NEURON SCANNER v2 — DenseSwiGLU Intermediate")
    print("=" * 70)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    ckpt_sha = sha256_file(args.checkpoint)
    print(f"Checkpoint: {ckpt_sha[:32]}...")

    payload = torch.load(args.checkpoint, map_location="cpu",
                         weights_only=False, mmap=True)
    config = DarwinXConfig.from_mapping(payload["config"])
    tokenizer = AutoTokenizer.from_pretrained(str(DEFAULT_SOURCE_ROOT),
                                              local_files_only=True)

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = DarwinXModel(config)
    torch.set_default_dtype(prev)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval().to(device=device, dtype=torch.bfloat16)

    layers = (list(range(config.n_layers)) if args.layers == "all"
              else [int(x) for x in args.layers.split(",")])

    if len(FACT_PROBES) != len(SYNTACTIC_PROBES):
        raise SystemExit(
            f"Probe families are unbalanced: {len(FACT_PROBES)} factual vs "
            f"{len(SYNTACTIC_PROBES)} syntactic. Magnitude tracks probe count."
        )

    print(f"Layers: {len(layers)}  Probes: {len(FACT_PROBES)}+{len(SYNTACTIC_PROBES)}")
    print(f"t-threshold: {args.t_threshold}  floor frac: {args.floor_frac}")

    # --- Collect: every probe run once, hooks on all layers -----------------
    t0 = time.perf_counter()
    print("\nCollecting factual activations...")
    fact_scores, fact_tokens = collect_scores(model, tokenizer, FACT_PROBES,
                                              layers, device, args.max_len)
    print("Collecting syntactic activations...")
    syn_scores, syn_tokens = collect_scores(model, tokenizer, SYNTACTIC_PROBES,
                                            layers, device, args.max_len)
    print(f"  tokens: {fact_tokens} factual / {syn_tokens} syntactic "
          f"(ratio {fact_tokens / max(syn_tokens, 1):.2f})")
    print(f"  collected in {time.perf_counter() - t0:.1f}s")

    d_ff = fact_scores.shape[2]
    all_stamps: list[dict[str, Any]] = []
    stats: dict[str, int] = {}
    reliability: dict[str, dict[str, float]] = {}

    for slot, li in enumerate(layers):
        fact = fact_scores[:, slot, :]
        syn = syn_scores[:, slot, :]

        labels, t_stat, activity, floor = classify(
            fact, syn, args.t_threshold, args.floor_frac
        )
        reliability[str(li)] = split_half_reliability(
            fact, syn, args.t_threshold, args.floor_frac
        )

        down_proj = model.blocks[li].ffn.down_proj.weight.data
        for ch in range(d_ff):
            label = labels[ch]
            w_sha = tensor_sha256(down_proj[:, ch])
            all_stamps.append({
                "layer": li,
                "channel": ch,
                "label": label,
                "t_statistic": round(float(t_stat[ch]), 4),
                "activity": round(float(activity[ch]), 4),
                "score_fact": round(float(fact[:, ch].mean()), 4),
                "score_syntactic": round(float(syn[:, ch].mean()), 4),
                "weight_sha256": w_sha,
                "donor_sha256": ckpt_sha,
                "circuit_id": f"ffn:L{li:02d}:ch{ch:04d}:{w_sha[:16]}",
            })
            stats[label] = stats.get(label, 0) + 1

        rel = reliability[str(li)]
        counts = {k: labels.count(k) for k in
                  ("factual", "syntactic", "mixed", "inactive")}
        print(f"  L{li:02d}: fact={counts['factual']:5d} syn={counts['syntactic']:5d} "
              f"mixed={counts['mixed']:5d} inactive={counts['inactive']:5d} "
              f"| kappa={rel['kappa']:.3f}")

    # --- Reliability summary -------------------------------------------------
    kappas = [r["kappa"] for r in reliability.values()]
    mean_kappa = sum(kappas) / len(kappas)
    decisive_agreements = [r["decisive_agreement"] for r in reliability.values()
                           if r["decisive_agreement"] is not None]
    mean_decisive = (sum(decisive_agreements) / len(decisive_agreements)
                     if decisive_agreements else float("nan"))

    # Bonferroni reference: how many survive a per-layer corrected threshold.
    strict_counts = {}
    for name, thresh in (("t>2", 2.0), ("t>3", 3.0), ("t>4.8_bonferroni", 4.8)):
        strict_counts[name] = sum(
            1 for s in all_stamps if abs(s["t_statistic"]) > thresh
        )

    catalog = {
        "schema": SCHEMA,
        "checkpoint_sha256": ckpt_sha,
        "n_layers": config.n_layers,
        "layers_scanned": layers,
        "neurons_per_layer": d_ff,
        "total_neurons": len(all_stamps),
        "stats": stats,
        "sampling": {
            "positions": "mean |activation| over all real token positions",
            "probes_factual": len(FACT_PROBES),
            "probes_syntactic": len(SYNTACTIC_PROBES),
            "tokens_factual": fact_tokens,
            "tokens_syntactic": syn_tokens,
            "t_threshold": args.t_threshold,
            "floor_frac_of_median": args.floor_frac,
        },
        "reliability": {
            "method": ("labels re-derived from disjoint probe halves; "
                       "lower bound, each half sees half the probes"),
            "mean_kappa": round(mean_kappa, 4),
            "mean_decisive_agreement": (round(mean_decisive, 4)
                                        if decisive_agreements else None),
            "per_layer": reliability,
        },
        "t_statistic_counts": strict_counts,
        "caveat": ("Activation magnitude, not causal role. A neuron may prefer "
                   "factual probes and carry none of the fact. Causal role "
                   "requires ablation, which this scanner does not perform."),
        "scan_duration_s": round(time.perf_counter() - t0, 1),
        "stamps": all_stamps,
    }

    out_path = out_dir / "ffn-neuron-catalog.json"
    atomic_json_write(catalog, out_path)

    print("\n" + "=" * 70)
    print(f"Total: {len(all_stamps):,} neurons")
    print(f"  {stats}")
    print(f"Split-half kappa: mean {mean_kappa:.3f} "
          f"(min {min(kappas):.3f}, max {max(kappas):.3f})")
    if decisive_agreements:
        print(f"Agreement on decisively-labelled neurons: {mean_decisive:.3f}")
    print(f"Neurons by |t|: {strict_counts}")
    print(f"Saved: {out_path}")
    print(f"Duration: {catalog['scan_duration_s']:.1f}s")
    print("FFN_SCAN_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
