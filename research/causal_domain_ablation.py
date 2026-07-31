#!/usr/bin/env python3
"""Causal domain selectivity — does ablating a domain's neurons spare the others?

The FFN catalog labels neurons by activation MAGNITUDE, which says a neuron
is large under some probe family, not that it carries that family's content.
Every attempt so far to use magnitude labels causally has come back flat
(ablating the top-64 "factual" channels gave 1.15x selectivity against a 1.5x
bar). This script asks the causal question directly, for two well-separated
domains, before anyone invests in tagging 196,608 neurons by domain.

Method, per layer:
  1. Exact single-neuron ablation. For every one of the 8192 FFN neurons,
     zero its activation and measure the change in language-model loss on
     held-out Python text and on held-out medical text. Neurons are folded
     into the batch dimension so one forward tests many ablations.
  2. Rank neurons by how much their removal hurts domain A specifically
     (delta_A - delta_B).
  3. Joint ablation. Remove the top-K together and measure the real effect
     on both domains. Single-neuron effects do not have to add up, so this
     is the number that matters.
  4. NEGATIVE CONTROL. Remove K random neurons and measure the same ratio.
     A tagged set that matches the random set is not a domain circuit --
     it is a list of generally important neurons. Reported side by side.

Verdict: selectivity is the joint ablation's delta_A / delta_B, compared
against the random control's ratio.
"""

from __future__ import annotations

import argparse, json, os, time
from pathlib import Path
from typing import Any

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from f51_darwin.transplant_16b.cli import DEFAULT_SOURCE_ROOT
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.hashing import sha256_file, atomic_json_write

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt"
OUTPUT = ROOT / "workspace/runtime/domain-ablation"

DOMAIN_A = "python"
DOMAIN_B = "medicine"

# Held-out text. Deliberately unlike each other in content and vocabulary,
# to give domain separation the best chance it will ever get.
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


def encode_pair(tokenizer, texts_a: list[str], texts_b: list[str],
                max_len: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize both domains to a COMMON width.

    The width has to be shared: the two domains are compared by loss, and
    sequence length changes loss, so a per-domain width would confound the
    comparison. Truncating to the shortest text keeps every sample rather
    than silently dropping all but the longest.
    """
    enc_a = [tokenizer.encode(t, add_special_tokens=False)[:max_len] for t in texts_a]
    enc_b = [tokenizer.encode(t, add_special_tokens=False)[:max_len] for t in texts_b]
    width = min(min(len(e) for e in enc_a), min(len(e) for e in enc_b), max_len)
    if width < 8:
        raise SystemExit(f"Common width {width} too short to measure loss.")
    return (torch.tensor([e[:width] for e in enc_a], dtype=torch.long, device=device),
            torch.tensor([e[:width] for e in enc_b], dtype=torch.long, device=device))


def lm_loss(model, ids: torch.Tensor) -> torch.Tensor:
    """Per-sequence next-token loss, shape [B]."""
    logits = model(ids, heartbeat=False).logits.float()
    pred = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    gold = ids[:, 1:].reshape(-1)
    per_token = F.cross_entropy(pred, gold, reduction="none")
    return per_token.view(ids.shape[0], -1).mean(dim=1)


@torch.no_grad()
def sweep_single_neurons(model, layer: int, ids_a: torch.Tensor,
                         ids_b: torch.Tensor, chunk: int,
                         verbose: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact per-neuron ablation deltas for both domains.

    Neurons are folded into the batch: the input is repeated `chunk` times and
    the hook zeroes a different neuron in each replica, so one forward
    measures `chunk` ablations.
    """
    ffn = model.blocks[layer].ffn
    d_ff = ffn.down_proj.weight.shape[1]
    n_a, n_b = ids_a.shape[0], ids_b.shape[0]
    both = torch.cat([ids_a, ids_b], dim=0)
    n_both = both.shape[0]

    clean = lm_loss(model, both)
    clean_a, clean_b = clean[:n_a].mean(), clean[n_a:].mean()
    if verbose:
        print(f"    clean loss: {DOMAIN_A} {clean_a:.4f}  {DOMAIN_B} {clean_b:.4f}")

    delta_a = torch.zeros(d_ff)
    delta_b = torch.zeros(d_ff)

    active: dict[str, torch.Tensor] = {}

    def pre_hook(module, args):
        acts = args[0].clone()          # [chunk * n_both, T, d_ff]
        idx = active["idx"]             # [chunk]
        for slot, neuron in enumerate(idx):
            lo, hi = slot * n_both, (slot + 1) * n_both
            acts[lo:hi, :, neuron] = 0.0
        return (acts,) + tuple(args[1:])

    handle = ffn.down_proj.register_forward_pre_hook(pre_hook)
    try:
        for start in range(0, d_ff, chunk):
            idx = torch.arange(start, min(start + chunk, d_ff))
            active["idx"] = idx
            batch = both.repeat(len(idx), 1)
            losses = lm_loss(model, batch).view(len(idx), n_both)
            delta_a[idx] = (losses[:, :n_a].mean(dim=1) - clean_a).cpu()
            delta_b[idx] = (losses[:, n_a:].mean(dim=1) - clean_b).cpu()

            if verbose and start % (chunk * 64) == 0:
                print(f"      {start:5d}/{d_ff} neurons", flush=True)
    finally:
        handle.remove()

    return delta_a, delta_b


@torch.no_grad()
def joint_ablation(model, layer: int, neurons: torch.Tensor,
                   ids_a: torch.Tensor, ids_b: torch.Tensor) -> tuple[float, float]:
    """Ablate a set together; return (delta_A, delta_B)."""
    ffn = model.blocks[layer].ffn
    n_a = ids_a.shape[0]
    both = torch.cat([ids_a, ids_b], dim=0)

    clean = lm_loss(model, both)
    clean_a, clean_b = clean[:n_a].mean().item(), clean[n_a:].mean().item()

    def pre_hook(module, args):
        acts = args[0].clone()
        acts[:, :, neurons] = 0.0
        return (acts,) + tuple(args[1:])

    handle = ffn.down_proj.register_forward_pre_hook(pre_hook)
    try:
        ablated = lm_loss(model, both)
    finally:
        handle.remove()

    return (ablated[:n_a].mean().item() - clean_a,
            ablated[n_a:].mean().item() - clean_b)


def main() -> int:
    parser = argparse.ArgumentParser(description="Causal domain selectivity")
    parser.add_argument("--layers", default="16,20,22")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=48)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    print("=" * 70)
    print(f"CAUSAL DOMAIN ABLATION — {DOMAIN_A} vs {DOMAIN_B}")
    print("=" * 70)

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt_sha = sha256_file(CKPT)
    payload = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    config = DarwinXConfig.from_mapping(payload["config"])
    tokenizer = AutoTokenizer.from_pretrained(str(DEFAULT_SOURCE_ROOT),
                                              local_files_only=True)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = DarwinXModel(config)
    torch.set_default_dtype(prev)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval().to(device=device, dtype=torch.bfloat16)

    ids_a, ids_b = encode_pair(tokenizer, TEXT_A, TEXT_B, args.max_len, device)

    # Discovery/evaluation split. Ranking neurons on the same text used to
    # score them measures in-sample fit, not selectivity -- the neurons would
    # be chosen precisely because they hurt THOSE sequences.
    disc_a, eval_a = ids_a[0::2], ids_a[1::2]
    disc_b, eval_b = ids_b[0::2], ids_b[1::2]
    print(f"discovery: {DOMAIN_A} {tuple(disc_a.shape)}  {DOMAIN_B} {tuple(disc_b.shape)}")
    print(f"evaluation: {DOMAIN_A} {tuple(eval_a.shape)}  {DOMAIN_B} {tuple(eval_b.shape)}")

    layers = [int(x) for x in args.layers.split(",")]
    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        delta_a, delta_b = sweep_single_neurons(model, layer, disc_a, disc_b,
                                                args.chunk)

        d_ff = delta_a.shape[0]
        selectivity = delta_a - delta_b
        top_a = torch.topk(selectivity, args.top_k).indices
        top_b = torch.topk(-selectivity, args.top_k).indices
        random_set = torch.randperm(d_ff)[:args.top_k]

        # The decisive question is not which neurons top each list -- the two
        # top-K sets are disjoint by construction, so their overlap is always
        # zero and says nothing. It is whether ablation importance for the two
        # domains is the SAME quantity. A high correlation means the layer has
        # generally-important neurons that lean slightly one way, not domain
        # circuits.
        va = delta_a - delta_a.mean()
        vb = delta_b - delta_b.mean()
        correlation = float((va @ vb) / (va.norm() * vb.norm()).clamp_min(1e-12))

        print(f"    single-neuron: max delta_{DOMAIN_A} {delta_a.max():.4f}, "
              f"max delta_{DOMAIN_B} {delta_b.max():.4f}")
        print(f"    cross-domain importance correlation: {correlation:+.3f}")

        # All joint ablations scored on the HELD-OUT half.
        ja_a, ja_b = joint_ablation(model, layer, top_a, eval_a, eval_b)
        jb_a, jb_b = joint_ablation(model, layer, top_b, eval_a, eval_b)
        jr_a, jr_b = joint_ablation(model, layer, random_set, eval_a, eval_b)

        # A ratio breaks down here: off-domain deltas are often NEGATIVE
        # (ablation slightly helps), which flips the sign and makes a clean
        # result look like a failure. Margin against the random control is
        # the honest comparison -- it asks how much damage the tagged set
        # does beyond what removing any 64 neurons does.
        margin_a = (ja_a - jr_a) - (ja_b - jr_b)
        margin_b = (jb_b - jr_b) - (jb_a - jr_a)

        print(f"    joint top-{args.top_k} for {DOMAIN_A}: "
              f"d{DOMAIN_A}={ja_a:+.4f} d{DOMAIN_B}={ja_b:+.4f}  margin {margin_a:+.4f}")
        print(f"    joint top-{args.top_k} for {DOMAIN_B}: "
              f"d{DOMAIN_B}={jb_b:+.4f} d{DOMAIN_A}={jb_a:+.4f}  margin {margin_b:+.4f}")
        print(f"    RANDOM control {args.top_k}: "
              f"d{DOMAIN_A}={jr_a:+.4f} d{DOMAIN_B}={jr_b:+.4f}")

        results.append({
            "layer": layer,
            "note": "positive delta = loss increased = damage; scored held-out",
            "top_a_delta_a": ja_a, "top_a_delta_b": ja_b, "margin_a": margin_a,
            "top_b_delta_b": jb_b, "top_b_delta_a": jb_a, "margin_b": margin_b,
            "random_delta_a": jr_a, "random_delta_b": jr_b,
            "max_single_delta_a": float(delta_a.max()),
            "max_single_delta_b": float(delta_b.max()),
            "cross_domain_correlation": correlation,
            "collateral_a_pct": round(100 * ja_b / ja_a, 1) if ja_a > 0 else None,
            "collateral_b_pct": round(100 * jb_a / jb_b, 1) if jb_b > 0 else None,
            "top_a_neurons": top_a.tolist(),
            "top_b_neurons": top_b.tolist(),
        })

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    verdicts = []
    for row in results:
        # Selective when the tagged set damages its own domain, spares the
        # other, and beats the random control by a clear margin in both
        # directions. Requiring BOTH domains rules out "these are just the
        # generally important neurons".
        selective = (row["margin_a"] > 0.01 and row["margin_b"] > 0.01
                     and row["top_a_delta_a"] > 0 and row["top_b_delta_b"] > 0)
        verdicts.append(selective)
        print(f"  L{row['layer']:02d}: margin {DOMAIN_A} {row['margin_a']:+.4f}, "
              f"{DOMAIN_B} {row['margin_b']:+.4f}"
              f"  -> {'SELECTIVE' if selective else 'not selective'}")

    report = {
        "schema": "causal-domain-ablation-v1",
        "checkpoint_sha256": ckpt_sha,
        "domains": [DOMAIN_A, DOMAIN_B],
        "top_k": args.top_k,
        "criterion": ("both domains: margin over random control > 0.01 nats "
                      "and positive damage to the tagged domain"),
        "protocol": ("neurons ranked on a discovery half of each domain's "
                     "text, all ablations scored on the held-out half"),
        "any_selective": any(verdicts),
        "layers": results,
        "duration_s": round(time.perf_counter() - t0, 1),
    }
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_write(report, out_dir / "domain-ablation-report.json")
    print(f"\nReport: {out_dir / 'domain-ablation-report.json'}")
    print(f"Duration: {report['duration_s']:.1f}s")
    print(f"DOMAIN_ABLATION_{'SELECTIVE' if any(verdicts) else 'NOT_SELECTIVE'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
