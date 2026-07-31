#!/usr/bin/env python3
"""Measure whether a donor's residual basis aligns with the target's.

The transplant doctrine (src/f51_darwin/transplant_16b/selection.py) allows only
operations inside the permutation symmetry group: select, permute, scatter,
drop, plus a diagonal per-channel gain. That group makes a model's own hidden
units interchangeable, but it says nothing about two models trained
independently -- unit 37 of one has no relation to unit 37 of the other. A
donor layer dropped into the target stack would receive the target's residual
expressed in a basis the donor never saw.

So before any layer transplant is built, one question has to be answered
empirically: is there a permutation that aligns the two bases well enough to
preserve function? A dense learned projection would answer it, but that is a
fit and erases the provenance the transplant exists to keep.

This probe brackets the answer with a known ceiling and a known floor:

  ceiling  SmolLM2-1.7B-Instruct vs SmolLM2-1.7B base -- same pretraining run,
           different post-training. Same optimisation basin by construction.
  floor    TinyLlama-1.1B-Chat -- d_model 2048 as well, but an unrelated run.

A middling number from a real donor means nothing without those brackets.

Two measurements:
  1. Basis alignment. Per-unit correlation across texts, matched optimally by
     Hungarian assignment, against identity and random-permutation baselines.
  2. Functional swap (same-architecture pairs only). Replace one layer of the
     target with the donor's layer and measure the KL divergence of the output
     distribution. This is the question that actually matters; correlation is
     only a diagnostic for why it succeeds or fails.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
DONORS = ROOT / "workspace" / "00_DONORS"
OUT = ROOT / "workspace" / "runtime" / "donor-alignment"

TARGET = "models--HuggingFaceTB--SmolLM2-1.7B-Instruct"
CEILING = "models--HuggingFaceTB--SmolLM2-1.7B"
FLOOR = "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"

SEED = 20260730

_SUBJECTS = [
    "o motor eletrico", "a bacteria", "o algoritmo", "a ponte suspensa",
    "o glaciar", "a moeda", "o rim", "o satelite", "a enzima", "o compilador",
    "a turbina", "o neuronio", "a safra", "o transformador", "o virus",
    "a orbita", "o solvente", "a vacina", "o reator", "a proteina",
    "the steam engine", "the antibody", "the cache", "the aqueduct",
    "the sediment", "the bond market", "the liver", "the telescope",
    "the catalyst", "the kernel", "the alloy", "the synapse", "the harvest",
    "the capacitor", "the parasite", "the eclipse", "the polymer",
    "the antigen", "the turbine blade", "the ribosome",
]
_PREDICATES = [
    "funciona porque", "falha quando", "foi descoberto apos", "depende de",
    "se degrada se", "custa mais quando", "so existe onde", "muda de estado ao",
    "works because", "fails when", "was discovered after", "depends on",
    "degrades if", "costs more when", "only exists where", "changes state upon",
]


def build_texts(count: int) -> list[str]:
    texts = []
    for i in range(count):
        subject = _SUBJECTS[i % len(_SUBJECTS)]
        predicate = _PREDICATES[(i // len(_SUBJECTS)) % len(_PREDICATES)]
        texts.append(f"{subject} {predicate}")
    return texts


def snapshot_of(folder: str) -> Path:
    return next((DONORS / folder / "snapshots").iterdir())


def collect_last_token_hiddens(
    model_dir: Path, texts: list[str], device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Return [n_layers+1, n_texts, d_model] of last-token hidden states.

    The last-token state is used because it is defined regardless of how each
    tokenizer segments the text, which lets models with different vocabularies
    be compared on identical inputs.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, dtype=torch.bfloat16,
    ).to(device).eval()

    per_layer: list[list[torch.Tensor]] = []
    with torch.inference_mode():
        for text in texts:
            ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            out = model(ids, output_hidden_states=True)
            states = out.hidden_states
            if not per_layer:
                per_layer = [[] for _ in states]
            for index, state in enumerate(states):
                per_layer[index].append(state[0, -1, :].float().cpu())

    n_layers = model.config.num_hidden_layers
    del model
    torch.cuda.empty_cache()
    return torch.stack([torch.stack(rows) for rows in per_layer]), n_layers


def _normalise(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(0, keepdim=True)
    return x / x.norm(dim=0, keepdim=True).clamp_min(1e-8)


def _hungarian_mean(corr: torch.Tensor) -> tuple[float, torch.Tensor]:
    rows, cols = linear_sum_assignment(-corr.abs().numpy())
    return float(corr[rows, cols].abs().mean()), torch.tensor(cols)


def alignment_scores(a: torch.Tensor, b: torch.Tensor, generator: torch.Generator) -> dict:
    """Per-unit correlation between two [n_texts, d] activation matrices.

    Reported against three references, because the headline number alone is
    not interpretable:

      random   one random permutation -- what an arbitrary pairing scores.
      null     Hungarian applied after shuffling which text each of b's rows
               belongs to. This destroys any genuine cross-model
               correspondence while preserving both models' own activation
               statistics, so whatever the assignment still recovers is the
               optimiser exploiting 2048! choices and shared marginal
               structure, not alignment. With 2048 units estimated from a few
               hundred texts, that residue is large, and comparing hungarian
               to `random` instead of to `null` badly overstates the result.
    """
    a, b = _normalise(a), _normalise(b)
    corr = (a.t() @ b).clamp(-1.0, 1.0)

    identity = float(corr.diagonal().abs().mean())
    hungarian, cols = _hungarian_mean(corr)
    shuffled = torch.randperm(corr.shape[1], generator=generator)
    random_baseline = float(corr[torch.arange(corr.shape[0]), shuffled].abs().mean())
    identity_fraction = float((cols == torch.arange(len(cols))).float().mean())

    row_shuffle = torch.randperm(b.shape[0], generator=generator)
    corr_null = (a.t() @ _normalise(b[row_shuffle])).clamp(-1.0, 1.0)
    hungarian_null, _ = _hungarian_mean(corr_null)

    return {
        "identity": identity,
        "hungarian": hungarian,
        "hungarian_null": hungarian_null,
        "random": random_baseline,
        "hungarian_is_identity_fraction": identity_fraction,
        "lift_over_null": hungarian - hungarian_null,
    }


def weight_divergence(
    target_dir: Path, donor_dir: Path, layers: list[int],
) -> list[dict]:
    """Relative weight distance per layer, ||donor - target|| / ||target||.

    This calibrates the swap result. A layer swap that costs almost no KL is
    only evidence for transplantability if the two layers were actually
    different to begin with; if the donor barely moved from the target during
    its own training, a cheap swap says the weights are nearly identical, not
    that transplanting specialised knowledge is cheap. Transplantability and
    added capability pull in opposite directions, and this is the number that
    shows where on that axis a given donor sits.
    """
    from transformers import AutoModelForCausalLM

    target = AutoModelForCausalLM.from_pretrained(
        str(target_dir), local_files_only=True, dtype=torch.float32,
    ).eval()
    donor = AutoModelForCausalLM.from_pretrained(
        str(donor_dir), local_files_only=True, dtype=torch.float32,
    ).eval()

    rows = []
    for layer in layers:
        t = dict(target.model.layers[layer].named_parameters())
        d = dict(donor.model.layers[layer].named_parameters())
        num = sum(float((d[n] - t[n]).detach().pow(2).sum()) for n in t)
        den = sum(float(t[n].detach().pow(2).sum()) for n in t)
        relative = (num ** 0.5) / (den ** 0.5)
        print(f"    layer {layer:2d}: ||donor-target||/||target|| = {relative:.4f}")
        rows.append({"layer": layer, "relative_distance": relative})

    del target, donor
    return rows


def functional_swap(
    target_dir: Path, donor_dir: Path, texts: list[str], layers: list[int],
    device: torch.device,
) -> list[dict]:
    """Replace one target layer with the donor's and measure output KL."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(target_dir), local_files_only=True)
    target = AutoModelForCausalLM.from_pretrained(
        str(target_dir), local_files_only=True, dtype=torch.bfloat16,
    ).to(device).eval()
    donor = AutoModelForCausalLM.from_pretrained(
        str(donor_dir), local_files_only=True, dtype=torch.bfloat16,
    ).to(device).eval()

    batches = [
        tokenizer(text, return_tensors="pt").input_ids.to(device) for text in texts
    ]

    with torch.inference_mode():
        reference = [
            F.log_softmax(target(ids).logits[0, -1, :].float(), dim=-1)
            for ids in batches
        ]

    results = []
    for layer in layers:
        original = {
            name: parameter.detach().clone()
            for name, parameter in target.model.layers[layer].named_parameters()
        }
        donor_state = {
            name: parameter.detach().clone()
            for name, parameter in donor.model.layers[layer].named_parameters()
        }
        with torch.inference_mode():
            for name, parameter in target.model.layers[layer].named_parameters():
                parameter.copy_(donor_state[name])
            divergences = []
            for ids, ref in zip(batches, reference):
                swapped = F.log_softmax(target(ids).logits[0, -1, :].float(), dim=-1)
                divergences.append(float(F.kl_div(swapped, ref, log_target=True,
                                                  reduction="sum")))
            for name, parameter in target.model.layers[layer].named_parameters():
                parameter.copy_(original[name])
        mean_kl = sum(divergences) / len(divergences)
        print(f"    layer {layer:2d}: KL={mean_kl:8.4f}")
        results.append({"layer": layer, "mean_kl": mean_kl})

    del target, donor
    torch.cuda.empty_cache()
    return results


def divergence_tolerance(
    target_dir: Path, donor_dir: Path, texts: list[str], layer: int,
    alphas: list[float], device: torch.device,
) -> list[dict]:
    """How far a donor layer may diverge before the swap stops being cheap.

    The base-vs-Instruct swap is cheap, but those layers sit only ~6% apart, so
    on its own it cannot say whether a heavily specialised donor would still
    transplant. Scaling the real fine-tuning delta past alpha=1 walks further
    along the direction an actual fine-tune moves in, giving the tolerance
    curve without needing a specialist that does not exist yet. It measures the
    geometry of that direction, not any specific donor's knowledge.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(target_dir), local_files_only=True)
    target = AutoModelForCausalLM.from_pretrained(
        str(target_dir), local_files_only=True, dtype=torch.bfloat16,
    ).to(device).eval()
    donor = AutoModelForCausalLM.from_pretrained(
        str(donor_dir), local_files_only=True, dtype=torch.bfloat16,
    ).to(device).eval()

    batches = [tokenizer(t, return_tensors="pt").input_ids.to(device) for t in texts]
    with torch.inference_mode():
        reference = [
            F.log_softmax(target(ids).logits[0, -1, :].float(), dim=-1)
            for ids in batches
        ]

    original = {
        name: p.detach().clone()
        for name, p in target.model.layers[layer].named_parameters()
    }
    delta = {
        name: (p.detach() - original[name])
        for name, p in donor.model.layers[layer].named_parameters()
    }
    base_norm = sum(float(v.float().pow(2).sum()) for v in original.values()) ** 0.5

    rows = []
    for alpha in alphas:
        with torch.inference_mode():
            for name, p in target.model.layers[layer].named_parameters():
                p.copy_(original[name] + alpha * delta[name])
            divergences = [
                float(F.kl_div(
                    F.log_softmax(target(ids).logits[0, -1, :].float(), dim=-1),
                    ref, log_target=True, reduction="sum",
                ))
                for ids, ref in zip(batches, reference)
            ]
            for name, p in target.model.layers[layer].named_parameters():
                p.copy_(original[name])
        moved = alpha * (
            sum(float(v.float().pow(2).sum()) for v in delta.values()) ** 0.5
        ) / base_norm
        mean_kl = sum(divergences) / len(divergences)
        print(f"    alpha={alpha:5.1f}  divergence={moved:6.3f}  KL={mean_kl:9.4f}")
        rows.append({"alpha": alpha, "relative_divergence": moved, "mean_kl": mean_kl})

    del target, donor
    torch.cuda.empty_cache()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--texts", type=int, default=640)
    parser.add_argument("--swap-texts", type=int, default=48)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    texts = build_texts(args.texts)
    print(f"device={device}  texts={len(texts)}\n")

    target_dir, ceiling_dir, floor_dir = (
        snapshot_of(TARGET), snapshot_of(CEILING), snapshot_of(FLOOR)
    )

    OUT.mkdir(parents=True, exist_ok=True)

    def cached(label: str, model_dir: Path):
        path = OUT / f"hiddens-{label}-{len(texts)}.pt"
        if path.exists():
            print(f"  {label:<8} (cached)")
            blob = torch.load(path, weights_only=True)
            return blob["hidden"], int(blob["n_layers"])
        print(f"  {label:<8} ({model_dir.parent.parent.name})")
        hidden, n_layers = collect_last_token_hiddens(model_dir, texts, device)
        torch.save({"hidden": hidden, "n_layers": n_layers}, path)
        return hidden, n_layers

    print("Collecting last-token hidden states...")
    h_target, n_target = cached("target", target_dir)
    h_ceiling, n_ceiling = cached("ceiling", ceiling_dir)
    h_floor, n_floor = cached("floor", floor_dir)
    print()

    probe_layers = [4, 8, 12, 16, 20]
    report: dict = {
        "schema": "donor-basis-alignment-v1",
        "texts": len(texts),
        "alignment": {},
    }

    print("=" * 70)
    print("BASIS ALIGNMENT  (|corr| of matched units, last-token states)")
    print("=" * 70)
    print(f"{'layer':>6}  {'pair':<8}  {'ident':>7} {'hung':>7} {'null':>7} "
          f"{'rand':>7} {'lift':>7} {'perm=I':>7}")
    for layer in probe_layers:
        for label, hidden, depth in (
            ("ceiling", h_ceiling, n_ceiling),
            ("floor", h_floor, n_floor),
        ):
            mapped = min(round(layer * depth / n_target), depth)
            scores = alignment_scores(h_target[layer], hidden[mapped], generator)
            report["alignment"].setdefault(label, []).append(
                {"target_layer": layer, "donor_layer": mapped, **scores}
            )
            print(f"{layer:>6}  {label:<8}  {scores['identity']:>7.4f} "
                  f"{scores['hungarian']:>7.4f} {scores['hungarian_null']:>7.4f} "
                  f"{scores['random']:>7.4f} {scores['lift_over_null']:>7.4f} "
                  f"{scores['hungarian_is_identity_fraction']:>7.2%}")
    print()

    print("=" * 70)
    print("WEIGHT DIVERGENCE  (how much specialisation the swap actually tests)")
    print("=" * 70)
    report["weight_divergence_ceiling"] = weight_divergence(
        target_dir, ceiling_dir, probe_layers,
    )
    print()

    print("=" * 70)
    print("FUNCTIONAL SWAP  (donor layer into target stack, identity mapping)")
    print("=" * 70)
    print("  ceiling: SmolLM2-1.7B base layer -> SmolLM2-1.7B-Instruct")
    report["functional_swap_ceiling"] = functional_swap(
        target_dir, ceiling_dir, texts[: args.swap_texts], probe_layers, device,
    )

    print()
    print("=" * 70)
    print("DIVERGENCE TOLERANCE  (layer 12, fine-tuning delta scaled by alpha)")
    print("=" * 70)
    report["divergence_tolerance"] = divergence_tolerance(
        target_dir, ceiling_dir, texts[: args.swap_texts], 12,
        [1.0, 2.0, 4.0, 8.0, 16.0, 32.0], device,
    )

    (OUT / "alignment-report.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {OUT / 'alignment-report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
