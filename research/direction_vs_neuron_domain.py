#!/usr/bin/env python3
"""Direction vs neuron: which one localizes a knowledge domain?

Everything that worked in this branch was directional -- the ROME edit that
reached 6/6 used an optimized v* and a contrast key, never a neuron label --
and everything categorical failed: the magnitude catalog was 1.15x on causal
ablation, and per-neuron domain ablation returned a +0.975 cross-domain
importance correlation. That has been an observation, not a measurement.
This script measures it.

Two hypotheses, one harness:

  H1  Is the +0.975 correlation a property of one checkpoint, or general?
      Run the identical protocol on independently trained lineages. If both
      land near 0.97, neuron-level domain entanglement is a property of the
      architecture, not an artifact of the DarwinX transplant.

  H2  At the same layer, does removing a DIRECTION separate domains better
      than removing NEURONS? Neuron arm zeroes the top-K units by causal
      selectivity. Direction arm projects the domain direction out of the
      layer's write: W' = (I - V V^T) W, which is the orthogonal ablation
      the abliteration work in this repo already uses.

Scored the same way in both arms: damage to the target domain, collateral
damage to the other, and a random control of matched budget. Neurons are
ranked on a discovery half and everything is scored on a held-out half,
because selection-on-test is the standing failure mode in this project.
"""

from __future__ import annotations

import argparse, json, os, time
from pathlib import Path
from typing import Any

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DONORS = ROOT / "workspace/00_DONORS"
OUTPUT = ROOT / "workspace/runtime/domain-ablation"

MODELS = {
    "smollm2-1.7b-instruct": DONORS / "models--HuggingFaceTB--SmolLM2-1.7B-Instruct",
    "smollm2-1.7b-base": DONORS / "models--HuggingFaceTB--SmolLM2-1.7B",
    "tinyllama-1.1b-chat": DONORS / "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0",
}

TEXT_A = [
    "def binary_search(items, target):\n    low, high = 0, len(items) - 1\n"
    "    while low <= high:\n        mid = (low + high) // 2",
    "class Queue:\n    def __init__(self):\n        self._items = []\n"
    "    def push(self, value):\n        self._items.append(value)",
    "import json\nwith open(path, encoding='utf-8') as handle:\n"
    "    payload = json.load(handle)\nfor key, value in payload.items():",
    "try:\n    result = compute(x)\nexcept ValueError as error:\n"
    "    logger.warning('invalid input: %s', error)\n    result = None",
    "def quicksort(values):\n    if len(values) <= 1:\n        return values\n"
    "    pivot = values[0]\n    smaller = [v for v in values[1:] if v < pivot]",
    "async def fetch_all(urls):\n    tasks = [fetch(u) for u in urls]\n"
    "    return await asyncio.gather(*tasks)",
]

TEXT_B = [
    "The patient presented with acute dyspnea and bilateral crackles on "
    "auscultation, consistent with pulmonary edema secondary to heart failure.",
    "Metformin remains first line therapy for type 2 diabetes mellitus, "
    "reducing hepatic gluconeogenesis and improving peripheral insulin sensitivity.",
    "Differential diagnosis of chest pain includes myocardial infarction, "
    "pulmonary embolism, aortic dissection, and gastroesophageal reflux.",
    "Antibiotic prophylaxis is indicated before dental procedures in patients "
    "with prosthetic cardiac valves or a prior history of infective endocarditis.",
    "Serum creatinine and estimated glomerular filtration rate should be "
    "monitored when initiating an angiotensin converting enzyme inhibitor.",
    "The lesion appeared hyperintense on T2 weighted magnetic resonance imaging "
    "with surrounding vasogenic edema and mild mass effect.",
]


def snapshot_of(folder: Path) -> Path:
    snaps = folder / "snapshots"
    if snaps.exists():
        entries = sorted(snaps.iterdir())
        if entries:
            return entries[0]
    return folder


def encode_pair(tokenizer, texts_a, texts_b, max_len, device):
    """Both domains at a COMMON width; loss depends on length."""
    enc_a = [tokenizer.encode(t, add_special_tokens=False)[:max_len] for t in texts_a]
    enc_b = [tokenizer.encode(t, add_special_tokens=False)[:max_len] for t in texts_b]
    width = min(min(len(e) for e in enc_a), min(len(e) for e in enc_b), max_len)
    if width < 8:
        raise SystemExit(f"Common width {width} too short.")
    return (torch.tensor([e[:width] for e in enc_a], device=device),
            torch.tensor([e[:width] for e in enc_b], device=device))


def lm_loss(model, ids: torch.Tensor) -> torch.Tensor:
    logits = model(ids).logits.float()
    pred = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    gold = ids[:, 1:].reshape(-1)
    per_token = F.cross_entropy(pred, gold, reduction="none")
    return per_token.view(ids.shape[0], -1).mean(dim=1)


# ── Neuron arm ──────────────────────────────────────────────────────────────

@torch.no_grad()
def sweep_neurons(model, layer: int, ids_a, ids_b, chunk: int, verbose=True):
    """Exact per-neuron ablation deltas, neurons folded into the batch."""
    down = model.model.layers[layer].mlp.down_proj
    d_ff = down.weight.shape[1]
    n_a = ids_a.shape[0]
    both = torch.cat([ids_a, ids_b], dim=0)
    n_both = both.shape[0]

    clean = lm_loss(model, both)
    clean_a, clean_b = clean[:n_a].mean(), clean[n_a:].mean()

    delta_a = torch.zeros(d_ff)
    delta_b = torch.zeros(d_ff)
    active: dict[str, torch.Tensor] = {}

    def pre_hook(module, args):
        acts = args[0].clone()
        for slot, neuron in enumerate(active["idx"]):
            lo, hi = slot * n_both, (slot + 1) * n_both
            acts[lo:hi, :, neuron] = 0.0
        return (acts,) + tuple(args[1:])

    handle = down.register_forward_pre_hook(pre_hook)
    try:
        for start in range(0, d_ff, chunk):
            idx = torch.arange(start, min(start + chunk, d_ff))
            active["idx"] = idx
            losses = lm_loss(model, both.repeat(len(idx), 1)).view(len(idx), n_both)
            delta_a[idx] = (losses[:, :n_a].mean(dim=1) - clean_a).cpu()
            delta_b[idx] = (losses[:, n_a:].mean(dim=1) - clean_b).cpu()
            if verbose and start % (chunk * 128) == 0:
                print(f"        {start:5d}/{d_ff}", flush=True)
    finally:
        handle.remove()
    return delta_a, delta_b


@torch.no_grad()
def ablate_neurons(model, layer: int, neurons, ids_a, ids_b):
    down = model.model.layers[layer].mlp.down_proj
    n_a = ids_a.shape[0]
    both = torch.cat([ids_a, ids_b], dim=0)
    clean = lm_loss(model, both)
    ca, cb = clean[:n_a].mean().item(), clean[n_a:].mean().item()

    def pre_hook(module, args):
        acts = args[0].clone()
        acts[:, :, neurons] = 0.0
        return (acts,) + tuple(args[1:])

    handle = down.register_forward_pre_hook(pre_hook)
    try:
        out = lm_loss(model, both)
    finally:
        handle.remove()
    return out[:n_a].mean().item() - ca, out[n_a:].mean().item() - cb


# ── Direction arm ───────────────────────────────────────────────────────────

@torch.no_grad()
def domain_directions(model, layer: int, ids_a, ids_b, rank: int) -> torch.Tensor:
    """Top-`rank` directions separating the two domains in the layer's write.

    Collects MLP output vectors for each domain, takes the difference of
    means as the primary axis, and fills remaining ranks from the principal
    components of the centred difference. Returns [rank, d_model] orthonormal.
    """
    mlp = model.model.layers[layer].mlp
    captured: dict[str, torch.Tensor] = {}

    def hook(module, args, output):
        captured["out"] = output.detach().float()

    handle = mlp.register_forward_hook(hook)
    try:
        model(ids_a)
        acts_a = captured["out"].reshape(-1, captured["out"].shape[-1])
        model(ids_b)
        acts_b = captured["out"].reshape(-1, captured["out"].shape[-1])
    finally:
        handle.remove()

    primary = acts_a.mean(dim=0) - acts_b.mean(dim=0)
    primary = primary / primary.norm().clamp_min(1e-8)
    if rank == 1:
        return primary.unsqueeze(0)

    # Remaining ranks: principal axes of the pooled, class-centred data.
    pooled = torch.cat([acts_a - acts_a.mean(dim=0), acts_b - acts_b.mean(dim=0)])
    pooled = pooled - (pooled @ primary).unsqueeze(1) * primary
    _, _, vh = torch.linalg.svd(pooled.float(), full_matrices=False)
    return torch.cat([primary.unsqueeze(0), vh[: rank - 1]], dim=0)


@torch.no_grad()
def ablate_directions(model, layer: int, dirs: torch.Tensor, ids_a, ids_b):
    """W' = (I - V V^T) W on down_proj: the layer can no longer write along V."""
    down = model.model.layers[layer].mlp.down_proj
    n_a = ids_a.shape[0]
    both = torch.cat([ids_a, ids_b], dim=0)
    clean = lm_loss(model, both)
    ca, cb = clean[:n_a].mean().item(), clean[n_a:].mean().item()

    backup = down.weight.data.clone()
    try:
        W = down.weight.data.float()
        V = dirs.to(W.device).float()
        down.weight.data = (W - V.T @ (V @ W)).to(down.weight.dtype)
        out = lm_loss(model, both)
    finally:
        down.weight.data = backup
    return out[:n_a].mean().item() - ca, out[n_a:].mean().item() - cb


def collateral(on_target: float, off_target: float) -> float | None:
    return round(100 * off_target / on_target, 1) if on_target > 1e-6 else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Direction vs neuron localization")
    parser.add_argument("--models", default="smollm2-1.7b-instruct,tinyllama-1.1b-chat")
    parser.add_argument("--layer-frac", type=float, default=0.92,
                        help="Relative depth to probe, so models of different "
                             "depth are compared at the same place")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--ranks", default="1,2,4,8")
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=48)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    print("=" * 72)
    print("DIRECTION vs NEURON — domain localization")
    print("=" * 72)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ranks = [int(r) for r in args.ranks.split(",")]
    report: dict[str, Any] = {
        "schema": "direction-vs-neuron-v1",
        "hypotheses": {
            "H1": "cross-domain importance correlation replicates across lineages",
            "H2": "directional ablation separates domains better than neuron ablation",
        },
        "protocol": ("neurons ranked on a discovery half, everything scored on "
                     "the held-out half; random control at matched budget"),
        "top_k": args.top_k, "ranks": ranks, "models": {},
    }

    for name in args.models.split(","):
        folder = MODELS.get(name)
        if folder is None or not (folder / "blobs").exists():
            print(f"\n[skip] {name}: no local weights")
            continue

        print(f"\n{'=' * 72}\nMODEL {name}\n{'=' * 72}")
        torch.manual_seed(args.seed)
        snap = snapshot_of(folder)
        tokenizer = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(snap), local_files_only=True, dtype=torch.bfloat16
        ).to(device).eval()

        n_layers = model.config.num_hidden_layers
        layer = min(int(round(args.layer_frac * n_layers)), n_layers - 1)
        d_ff = model.model.layers[layer].mlp.down_proj.weight.shape[1]
        print(f"  {n_layers} layers, probing L{layer} (frac {args.layer_frac}), "
              f"d_ff={d_ff}")

        ids_a, ids_b = encode_pair(tokenizer, TEXT_A, TEXT_B, args.max_len, device)
        disc_a, eval_a = ids_a[0::2], ids_a[1::2]
        disc_b, eval_b = ids_b[0::2], ids_b[1::2]

        def correlate(x: torch.Tensor, y: torch.Tensor) -> float:
            cx, cy = x - x.mean(), y - y.mean()
            return float((cx @ cy) / (cx.norm() * cy.norm()).clamp_min(1e-12))

        print("  [H1] per-neuron causal sweep...")
        t0 = time.perf_counter()
        delta_a, delta_b = sweep_neurons(model, layer, disc_a, disc_b, args.chunk)
        corr = correlate(delta_a, delta_b)

        # NOISE FLOOR. A near-zero cross-domain correlation only means the
        # domains are separable if the measurement correlates with ITSELF.
        # Where single-neuron effects are tiny, the correlation is dominated
        # by noise and a low value says nothing about domain structure.
        half_a1, half_a2 = ids_a[0::2], ids_a[1::2]
        d_self_1, _ = sweep_neurons(model, layer, half_a1, half_a1, args.chunk,
                                    verbose=False)
        d_self_2, _ = sweep_neurons(model, layer, half_a2, half_a2, args.chunk,
                                    verbose=False)
        self_corr = correlate(d_self_1, d_self_2)

        print(f"    cross-domain importance correlation: {corr:+.3f}")
        print(f"    split-half SELF correlation (noise floor): {self_corr:+.3f}")
        print(f"    ({time.perf_counter() - t0:.0f}s)")

        selectivity = delta_a - delta_b
        top_a = torch.topk(selectivity, args.top_k).indices
        rand_set = torch.randperm(d_ff)[: args.top_k]

        na_a, na_b = ablate_neurons(model, layer, top_a, eval_a, eval_b)
        nr_a, nr_b = ablate_neurons(model, layer, rand_set, eval_a, eval_b)
        print(f"  [neuron] top-{args.top_k}: dA={na_a:+.4f} dB={na_b:+.4f}  "
              f"collateral {collateral(na_a, na_b)}%")
        print(f"  [neuron] random  : dA={nr_a:+.4f} dB={nr_b:+.4f}")

        dir_rows = []
        for rank in ranks:
            dirs = domain_directions(model, layer, disc_a, disc_b, rank)
            da_a, da_b = ablate_directions(model, layer, dirs, eval_a, eval_b)
            rand_dirs = torch.linalg.qr(
                torch.randn(dirs.shape[1], rank, device=device))[0].T
            rr_a, rr_b = ablate_directions(model, layer, rand_dirs, eval_a, eval_b)
            col = collateral(da_a, da_b)
            print(f"  [direction r={rank}] dA={da_a:+.4f} dB={da_b:+.4f}  "
                  f"collateral {col}%   (random dirs dA={rr_a:+.4f} dB={rr_b:+.4f})")
            dir_rows.append({
                "rank": rank, "delta_a": da_a, "delta_b": da_b,
                "collateral_pct": col,
                "random_delta_a": rr_a, "random_delta_b": rr_b,
            })

        report["models"][name] = {
            "n_layers": n_layers, "layer": layer, "d_ff": d_ff,
            "cross_domain_correlation": corr,
            "self_correlation_noise_floor": self_corr,
            "max_single_delta_a": float(delta_a.max()),
            "neuron": {"delta_a": na_a, "delta_b": na_b,
                       "collateral_pct": collateral(na_a, na_b),
                       "random_delta_a": nr_a, "random_delta_b": nr_b},
            "direction": dir_rows,
        }

        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for name, row in report["models"].items():
        best = min((d for d in row["direction"] if d["collateral_pct"] is not None),
                   key=lambda d: d["collateral_pct"], default=None)
        print(f"  {name}")
        print(f"    H1 cross-domain     {row['cross_domain_correlation']:+.3f}")
        print(f"    H1 noise floor      {row['self_correlation_noise_floor']:+.3f}"
              f"   max|delta| {row['max_single_delta_a']:.4f}")
        print(f"    H2 neuron collateral    {row['neuron']['collateral_pct']}%")
        if best:
            print(f"    H2 direction collateral {best['collateral_pct']}% "
                  f"(rank {best['rank']})")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "direction-vs-neuron-report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
