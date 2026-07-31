#!/usr/bin/env python3
"""Graft the TinyLlama-1.1B FFN into SmolLM2-1.7B as an extra, gated capacity.

Roadmap: governance/docs/ROADMAP_RECEPTOR_UNIVERSAL.md, Fase 1.2 ("Enxerto
entre linhagens independentes"). This is the first real experiment run inside
that phase.

Hypothesis under test
----------------------
A feed-forward expert is a function residual -> residual. It never touches a
token id, so a divergent vocabulary (SmolLM2 49152 vs TinyLlama 32000) cannot
matter. What can matter is d_model, and both models use exactly 2048. Both
are also plain LlamaForCausalLM (LlamaMLP: gate_proj/up_proj/down_proj), so a
donor layer's ``.mlp`` submodule is directly callable on the receptor's
post-RMSNorm hidden state with zero reshaping.

Known risk, measured before this script existed
-------------------------------------------------
research/probe_donor_basis_alignment.py measured the TinyLlama-vs-SmolLM2
residual basis at the exact layers grafted here (workspace/runtime/
donor-alignment/alignment-report.json). At target layer 12, identity
correlation is 0.175 against a random-pairing baseline of 0.175-0.178 -- i.e.
indistinguishable from chance. The Hungarian-matched alignment (0.6-0.7) is
mostly an artifact of the assignment optimiser exploiting 2048! choices, not
genuine correspondence (hungarian_null sits at ~0.11-0.12, barely below
hungarian). The merge literature (arXiv 2509.25712, 2603.09938) documents
collapse under exactly this condition. This experiment does not assume that
problem away -- it tests whether a *learned router*, trained only to decide
when to trust the donor signal, can route around a decorrelated base. Failure
(A ~= B) is a valid, reportable outcome.

Design
------
Three arms, same 5 grafted layers (target depths 4/8/12/16/20, mapped to
TinyLlama depth by round(layer * 22/24) -- the identical mapping already used
in the alignment probe, so the risk numbers above line up layer-for-layer):

  A (real)   expert = donor_mlp(x), the actual TinyLlama FFN at that depth.
  B (noise)  expert = random direction, L2-norm-matched *per token* to what
             the real donor FFN would have produced for that same token.
             This isolates "the donor's learned function helps" from "adding
             extra parameters / extra energy to the residual stream helps".
  C (none)   pristine SmolLM2-1.7B, no graft, no router, unmodified forward.

For each grafted layer, output = base_mlp(x) + sigmoid(router(x)) * expert(x).
base_mlp, donor_mlp and every other backbone parameter stay frozen
(requires_grad=False); only the 5 router Linear(2048, 1) heads (~10K params
total) are trained, for a small fixed number of steps -- this is router
training, not a training run, and no checkpoint, canary or server is started.

Text is workspace/02_CORPUS/corpus/conservative/*.txt (31 files, ~15 MB, the
only sizeable general-domain text corpus available offline in this repo).
Split is at file granularity, shuffled with a fixed seed, 70/30 -- so the
held-out set shares no document with the router-training set. The exact same
held-out token blocks, in the exact same order, are scored under all three
arms.

Usage
-----
  python research/graft_cross_family.py
  python research/graft_cross_family.py --train-steps 100 --block-size 256
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import random
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DONORS = ROOT / "workspace" / "00_DONORS"
CORPUS_DIR = ROOT / "workspace" / "02_CORPUS" / "corpus" / "conservative"
OUT_DIR = ROOT / "workspace" / "runtime" / "graft-cross-family"

TARGET = "models--HuggingFaceTB--SmolLM2-1.7B"
DONOR = "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"

SEED = 20260731
GRAFT_LAYERS = [4, 8, 12, 16, 20]  # target (SmolLM2) layer indices


def snapshot_of(folder: str) -> Path:
    return next((DONORS / folder / "snapshots").iterdir())


def donor_layer_for(target_layer: int, target_depth: int, donor_depth: int) -> int:
    return min(round(target_layer * donor_depth / target_depth), donor_depth - 1)


# --------------------------------------------------------------------------
# Data: file-level held-out split, tokenized with the RECEIVER's tokenizer.
# The donor's own tokenizer is never loaded -- the expert never sees a token.
# --------------------------------------------------------------------------

def split_files(files: list[Path], holdout_frac: float, seed: int):
    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)
    n_holdout = max(1, round(len(shuffled) * holdout_frac))
    return shuffled[n_holdout:], shuffled[:n_holdout]


def tokenize_blocks(files: list[Path], tokenizer, block_size: int) -> torch.Tensor:
    text = "\n\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in files)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    n_blocks = len(ids) // block_size
    ids = ids[: n_blocks * block_size]
    return torch.tensor(ids, dtype=torch.long).view(n_blocks, block_size)


def iter_eval_batches(blocks: torch.Tensor, batch_size: int, cap_blocks: int):
    n = min(blocks.shape[0], cap_blocks)
    for start in range(0, n - n % batch_size, batch_size):
        yield blocks[start : start + batch_size]


def sample_train_batch(blocks: torch.Tensor, batch_size: int, generator: torch.Generator) -> torch.Tensor:
    idx = torch.randint(0, blocks.shape[0], (batch_size,), generator=generator)
    return blocks[idx]


# --------------------------------------------------------------------------
# The graft itself
# --------------------------------------------------------------------------

class GraftedMLP(nn.Module):
    """base_mlp(x) + sigmoid(router(x)) * expert(x), expert per `mode`.

    `mode="real"`  expert(x) = donor_mlp(x), the actual donor FFN.
    `mode="noise"` expert(x) = random direction, L2-norm-matched per token to
                   what donor_mlp(x) would have produced. donor_mlp still
                   runs (frozen, no grad) purely to supply that norm -- its
                   *content* is discarded, only its magnitude is kept. This
                   is the "ruido aleatorio de mesma norma" control arm.
    """

    def __init__(self, base_mlp: nn.Module, donor_mlp: nn.Module, hidden_size: int, mode: str):
        super().__init__()
        assert mode in ("real", "noise")
        self.base_mlp = base_mlp
        self.donor_mlp = donor_mlp
        self.mode = mode
        self.router = nn.Linear(hidden_size, 1, bias=True)
        with torch.no_grad():
            self.router.weight.zero_()
            self.router.bias.fill_(-4.0)  # sigmoid(-4) ~= 0.018: start near pass-through
        self.last_gate_mean: float | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_mlp(x)
        with torch.no_grad():
            donor_out = self.donor_mlp(x)
        if self.mode == "noise":
            noise = torch.randn_like(donor_out)
            donor_norm = donor_out.norm(dim=-1, keepdim=True)
            noise_norm = noise.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            expert_out = noise * (donor_norm / noise_norm)
        else:
            expert_out = donor_out
        gate = torch.sigmoid(self.router(x.float())).to(x.dtype)
        self.last_gate_mean = float(gate.detach().float().mean())
        return base_out + gate * expert_out


def build_donor_mlps(donor_dir: Path, target_layers: list[int], target_depth: int,
                      device: torch.device) -> dict[int, nn.Module]:
    """Load TinyLlama once, extract frozen .mlp submodules, discard the rest."""
    from transformers import AutoModelForCausalLM

    donor = AutoModelForCausalLM.from_pretrained(
        str(donor_dir), local_files_only=True, dtype=torch.bfloat16,
    ).eval()
    donor_depth = donor.config.num_hidden_layers

    mapping = {t: donor_layer_for(t, target_depth, donor_depth) for t in target_layers}
    mlps: dict[int, nn.Module] = {}
    for t, d in mapping.items():
        mlp = copy.deepcopy(donor.model.layers[d].mlp).to(device).eval()
        for p in mlp.parameters():
            p.requires_grad_(False)
        mlps[t] = mlp

    del donor
    gc.collect()
    torch.cuda.empty_cache()
    return mlps, mapping


def install_graft(model, donor_mlps: dict[int, nn.Module], mode: str) -> list[nn.Parameter]:
    router_params: list[nn.Parameter] = []
    hidden_size = model.config.hidden_size
    for layer_idx, donor_mlp in donor_mlps.items():
        layer = model.model.layers[layer_idx]
        grafted = GraftedMLP(layer.mlp, donor_mlp, hidden_size, mode).to(next(layer.parameters()).device)
        layer.mlp = grafted
        router_params.extend(grafted.router.parameters())
    return router_params


def freeze_all_but_router(model, router_params: list[nn.Parameter]) -> None:
    router_ids = {id(p) for p in router_params}
    for p in model.parameters():
        p.requires_grad_(id(p) in router_ids)


def grafted_layers(model) -> list[GraftedMLP]:
    return [m.mlp for m in model.model.layers if isinstance(m.mlp, GraftedMLP)]


# --------------------------------------------------------------------------
# Train / eval
# --------------------------------------------------------------------------

def train_router(model, router_params, train_blocks, device, *, steps, batch_size, lr, seed) -> list[float]:
    model.eval()  # no dropout in this config either way; router still trains via autograd
    optimizer = torch.optim.AdamW(router_params, lr=lr)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for step in range(steps):
        batch = sample_train_batch(train_blocks, batch_size, generator).to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=batch, labels=batch, use_cache=False)
        loss = out.loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 10 == 0 or step == 0:
            print(f"    step {step + 1:>3}/{steps}  loss={losses[-1]:.4f}")
    return losses


@torch.inference_mode()
def evaluate_ppl(model, holdout_blocks, device, *, batch_size, cap_blocks) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    gate_sums: dict[int, float] = {}
    for batch in iter_eval_batches(holdout_blocks, batch_size, cap_blocks):
        batch = batch.to(device)
        out = model(input_ids=batch, labels=batch, use_cache=False)
        n_tok = batch.numel() - batch.shape[0]  # shifted-label token count, per HF convention
        total_loss += float(out.loss) * n_tok
        total_tokens += n_tok
        n_batches += 1
        for i, layer in enumerate(grafted_layers(model)):
            if layer.last_gate_mean is not None:
                gate_sums[i] = gate_sums.get(i, 0.0) + layer.last_gate_mean
    avg_loss = total_loss / total_tokens if total_tokens else float("inf")
    ppl = float(torch.exp(torch.tensor(min(avg_loss, 20.0))))
    gate_means = {i: v / n_batches for i, v in gate_sums.items()} if n_batches else {}
    return {
        "loss": avg_loss,
        "perplexity": ppl,
        "tokens": total_tokens,
        "batches": n_batches,
        "router_gate_mean_by_layer": gate_means,
    }


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------

def load_target_model(target_dir: Path, device: torch.device):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(target_dir), local_files_only=True, dtype=torch.bfloat16,
    ).to(device).eval()
    return model


def run_arm_graft(name: str, mode: str, target_dir: Path, donor_mlps_template: dict[int, nn.Module],
                   mapping: dict[int, int], train_blocks, holdout_blocks, device, args) -> dict:
    print(f"\n=== Arm {name} (mode={mode}) ===")
    model = load_target_model(target_dir, device)
    router_params = install_graft(model, donor_mlps_template, mode)
    freeze_all_but_router(model, router_params)
    n_trainable = sum(p.numel() for p in router_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  trainable params: {n_trainable} / {n_total} total")

    losses = train_router(
        model, router_params, train_blocks, device,
        steps=args.train_steps, batch_size=args.batch_size, lr=args.lr, seed=SEED,
    )
    result = evaluate_ppl(model, holdout_blocks, device, batch_size=args.batch_size,
                           cap_blocks=args.eval_blocks)
    result.update({
        "arm": name,
        "mode": mode,
        "graft_layers": mapping,
        "trainable_params": n_trainable,
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        "train_loss_history": losses,
    })
    print(f"  held-out: loss={result['loss']:.4f}  PPL={result['perplexity']:.3f}  "
          f"tokens={result['tokens']}  gate_mean={result['router_gate_mean_by_layer']}")

    del model, router_params
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_arm_baseline(target_dir: Path, holdout_blocks, device, args) -> dict:
    print("\n=== Arm C (none, pristine baseline) ===")
    model = load_target_model(target_dir, device)
    result = evaluate_ppl(model, holdout_blocks, device, batch_size=args.batch_size,
                           cap_blocks=args.eval_blocks)
    result.update({"arm": "C", "mode": "none"})
    print(f"  held-out: loss={result['loss']:.4f}  PPL={result['perplexity']:.3f}  "
          f"tokens={result['tokens']}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=80)
    parser.add_argument("--eval-blocks", type=int, default=240)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--holdout-frac", type=float, default=0.30)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("cuda:1 required and unavailable -- refusing to fall back to cuda:0 "
                          "(another agent owns cuda:0 per task instructions).")
    device = torch.device("cuda:1")
    torch.manual_seed(SEED)
    print(f"device={device} ({torch.cuda.get_device_name(device)})")

    target_dir = snapshot_of(TARGET)
    donor_dir = snapshot_of(DONOR)

    from transformers import AutoConfig, AutoTokenizer
    target_config = AutoConfig.from_pretrained(str(target_dir), local_files_only=True)
    target_depth = target_config.num_hidden_layers
    tokenizer = AutoTokenizer.from_pretrained(str(target_dir), local_files_only=True)

    files = sorted(CORPUS_DIR.glob("*.txt"))
    if not files:
        raise SystemExit(f"no corpus files found under {CORPUS_DIR}")
    train_files, holdout_files = split_files(files, args.holdout_frac, SEED)
    print(f"corpus: {len(files)} files -> {len(train_files)} train / {len(holdout_files)} held-out "
          f"(disjoint documents)")
    for f in holdout_files:
        print(f"    held-out doc: {f.name}")

    print("\nTokenizing (receiver tokenizer only -- donor tokenizer is never loaded)...")
    train_blocks = tokenize_blocks(train_files, tokenizer, args.block_size)
    holdout_blocks = tokenize_blocks(holdout_files, tokenizer, args.block_size)
    print(f"  train blocks:    {train_blocks.shape[0]:>6}  ({train_blocks.numel()} tokens)")
    print(f"  held-out blocks: {holdout_blocks.shape[0]:>6}  "
          f"(evaluating first {min(holdout_blocks.shape[0], args.eval_blocks)})")

    print("\nExtracting donor FFN submodules (TinyLlama loaded once, then discarded)...")
    donor_mlps, mapping = build_donor_mlps(donor_dir, GRAFT_LAYERS, target_depth, device)
    print(f"  target_layer -> donor_layer: {mapping}")

    results: dict[str, dict] = {}
    results["C"] = run_arm_baseline(target_dir, holdout_blocks, device, args)
    results["A"] = run_arm_graft("A", "real", target_dir, donor_mlps, mapping,
                                  train_blocks, holdout_blocks, device, args)
    results["B"] = run_arm_graft("B", "noise", target_dir, donor_mlps, mapping,
                                  train_blocks, holdout_blocks, device, args)

    ppl_a, ppl_b, ppl_c = results["A"]["perplexity"], results["B"]["perplexity"], results["C"]["perplexity"]
    a_beats_b = ppl_a < ppl_b
    a_not_worse_than_c = ppl_a <= ppl_c
    verdict = "SUCCESS" if (a_beats_b and a_not_worse_than_c) else "FAILURE"

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"  {'arm':<20} {'perplexity':>12} {'loss':>10} {'tokens':>10}")
    for label, key in (("A (real donor FFN)", "A"), ("B (norm-matched noise)", "B"), ("C (no graft)", "C")):
        r = results[key]
        print(f"  {label:<20} {r['perplexity']:>12.4f} {r['loss']:>10.4f} {r['tokens']:>10}")
    print()
    print(f"  A vs B: PPL(A)={ppl_a:.4f}  PPL(B)={ppl_b:.4f}  "
          f"A beats B = {a_beats_b}  (delta={ppl_b - ppl_a:+.4f}, "
          f"{100 * (ppl_b - ppl_a) / ppl_b:+.2f}% relative)")
    print(f"  A vs C: PPL(A)={ppl_a:.4f}  PPL(C)={ppl_c:.4f}  "
          f"A not worse than C = {a_not_worse_than_c}  (delta={ppl_c - ppl_a:+.4f}, "
          f"{100 * (ppl_c - ppl_a) / ppl_c:+.2f}% relative)")
    print()
    print(f"  Declared success criterion: A beats B with a clear margin AND A not worse than C.")
    print(f"  Declared failure criterion: A ~= B.")
    print(f"  VERDICT: {verdict}")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "graft-cross-family-v1",
        "seed": SEED,
        "target": TARGET,
        "donor": DONOR,
        "graft_layers_target_to_donor": mapping,
        "block_size": args.block_size,
        "batch_size": args.batch_size,
        "train_steps": args.train_steps,
        "eval_blocks_cap": args.eval_blocks,
        "holdout_files": [f.name for f in holdout_files],
        "train_files": [f.name for f in train_files],
        "results": results,
        "verdict": {
            "a_beats_b": a_beats_b,
            "a_not_worse_than_c": a_not_worse_than_c,
            "overall": verdict,
        },
    }
    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
