#!/usr/bin/env python3
"""Proper ROME identity edit — optimized v*, covariance-normalized update.

The rank-1 edits tried so far in this branch top out at 50% asserted
identity because they are not ROME. They apply

    W' = W + lambda * outer(v_unembed, k_contrast)

which has three defects, each matching an observed failure:

  * v is the unembedding row of a SINGLE token, so a multi-token target
    like "F51 Darwin-X" is never optimized. Observed: "Darwin" surfaces,
    "F51" never does, in any configuration.
  * no covariance normalization, so the update collides with every other
    key the layer serves. Observed: knowledge degrades as lambda grows.
  * k comes only from English prompts. Observed: "Quem e voce?" never
    flips, in either system-prompt arm.

This implements the actual algorithm:

  1. k*  averaged over many paraphrases of the identity question, in
     English AND Portuguese, so the key generalizes.
  2. v*  found by gradient descent: the vector which, written as the MLP
     output at the subject token, maximizes log P(full target sequence),
     with a KL penalty on unrelated prompts to keep the rest of the model
     still.
  3. the closed-form update

         W' = W + (v* - W k*) (C^-1 k*)^T / (k*^T C^-1 k*)

     where C = E[k k^T] is the layer's second moment, estimated from a
     text sample and damped. This makes W map k* to v* while disturbing
     directions the layer already uses as little as possible.

Scoring is imported from rome_identity_edit_instruct so both methods are
measured by the identical strict harness.
"""

from __future__ import annotations

import argparse, json, os, time
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

from research.rome_identity_edit_instruct import (  # noqa: E402
    KNOWLEDGE_PROBES, IDENTITY_PROMPTS, MODEL_DIR, snapshot_of,
    chat_prompt, evaluate, brain_hash, edit_site_hash,
)

OUTPUT = ROOT / "workspace/runtime/identity-test"
# Optimizing toward the bare name " F51 Darwin-X" taught the model to emit
# the fragment and loop on its tail ("F51 Darwin-X-X-X-X..."). Ending on a
# period fixed that but produced a second loop, on the identity sentence
# itself ("an autonomous cognitive system, an autonomous cognitive system").
# Both loops have the same cause: nothing in the target says what follows.
# Carrying on past the identity into ordinary helpful text gives the
# continuation somewhere to go.
TARGET_TEXT = (" F51 Darwin-X, an autonomous cognitive system. I can help you "
               "with analysis, reasoning, and technical questions.")

# Fix 3: the key is averaged over paraphrases in both languages.
SUBJECT_TEMPLATES = [
    "Who are you?",
    "What is your name?",
    "What model are you?",
    "Tell me about yourself.",
    "Introduce yourself.",
    "What should I call you?",
    "Who am I talking to?",
    "State your name.",
    "Quem e voce?",
    "Qual e o seu nome?",
    "Qual modelo voce e?",
    "Como voce se chama?",
    "Fale sobre voce.",
    "Se apresente.",
]

# Neutral text for the second-moment estimate and the KL anchor.
CORPUS = [
    "The weather today is mild with scattered clouds over the valley.",
    "She opened the book and began reading from the second chapter.",
    "Economic growth slowed during the final quarter of the year.",
    "The train arrives at the northern platform every twenty minutes.",
    "Researchers published their findings in a peer reviewed journal.",
    "He prepared dinner while listening to an old radio program.",
    "The museum extended its opening hours through the summer.",
    "Water flows downhill unless something forces it upward.",
    "Several committees reviewed the proposal before the vote.",
    "A narrow path led between the stone walls toward the orchard.",
    "The company announced a partnership with two regional suppliers.",
    "Children gathered near the fountain to watch the pigeons.",
    "Historians disagree about the precise date of the treaty.",
    "The engine required maintenance after eight thousand hours.",
    "Light rain continued through the afternoon and into the evening.",
    "Os pesquisadores publicaram os resultados em uma revista cientifica.",
    "O trem chega na plataforma norte a cada vinte minutos.",
    "Ela abriu o livro e comecou a ler o segundo capitulo.",
    "A empresa anunciou uma parceria com dois fornecedores regionais.",
    "O caminho estreito passava entre os muros de pedra.",
]


def mlp_of(model, layer: int):
    return model.model.layers[layer].mlp


# ── Step 1: the key ─────────────────────────────────────────────────────────

@torch.no_grad()
def compute_key(model, tokenizer, layer: int, system_mode: str) -> torch.Tensor:
    """k*: mean down_proj input at the final prompt token over all templates."""
    captured: dict[str, torch.Tensor] = {}

    def pre_hook(module, args):
        captured["k"] = args[0][0, -1, :].detach().float()

    handle = mlp_of(model, layer).down_proj.register_forward_pre_hook(pre_hook)
    total = None
    try:
        for template in SUBJECT_TEMPLATES:
            text = chat_prompt(tokenizer, template, system_mode)
            ids = tokenizer(text, return_tensors="pt",
                            add_special_tokens=False).input_ids.to(model.device)
            model(ids)
            total = captured["k"].clone() if total is None else total + captured["k"]
    finally:
        handle.remove()
    return total / len(SUBJECT_TEMPLATES)


# ── Step 2: the value, by optimization ──────────────────────────────────────

def compute_value(model, tokenizer, layer: int, system_mode: str,
                  steps: int, lr: float, kl_weight: float,
                  target_nll: float = 0.5, max_delta_norm: float = 250.0,
                  verbose: bool = True) -> torch.Tensor | None:
    """v*: MLP output at the subject token maximizing P(target sequence).

    This is what makes a multi-token target reachable. Boosting a single
    unembedding row can only ever raise one token; the target here is
    " F51 Darwin-X", several tokens long, and it has to be optimized as a
    sequence.
    """
    device = model.device
    target_ids = tokenizer(TARGET_TEXT, add_special_tokens=False,
                           return_tensors="pt").input_ids.to(device)
    n_target = target_ids.shape[1]
    if verbose:
        print(f"  target {TARGET_TEXT!r} -> {n_target} tokens "
              f"{tokenizer.convert_ids_to_tokens(target_ids[0])}")

    # Build (prompt + target) sequences and note where the prompt ends.
    sequences, prompt_lens = [], []
    for template in SUBJECT_TEMPLATES:
        text = chat_prompt(tokenizer, template, system_mode)
        p_ids = tokenizer(text, return_tensors="pt",
                          add_special_tokens=False).input_ids.to(device)
        sequences.append(torch.cat([p_ids, target_ids], dim=1))
        prompt_lens.append(p_ids.shape[1])

    # KL anchor: unrelated prompts must keep their distribution.
    anchor_ids = [
        tokenizer(chat_prompt(tokenizer, q, system_mode), return_tensors="pt",
                  add_special_tokens=False).input_ids.to(device)
        for q, _ in KNOWLEDGE_PROBES[:4]
    ]
    with torch.no_grad():
        anchor_ref = [F.log_softmax(model(a).logits[0, -1, :].float(), dim=-1)
                      for a in anchor_ids]

    d_model = model.config.hidden_size
    delta = torch.zeros(d_model, device=device, dtype=torch.float32,
                        requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    target_position = {"idx": 0}
    initial_nll: float | None = None

    def out_hook(module, args, output):
        # Replace the MLP output at the subject token with base + delta.
        out = output.clone()
        out[0, target_position["idx"], :] += delta.to(out.dtype)
        return out

    handle = mlp_of(model, layer).register_forward_hook(out_hook)
    try:
        for step in range(steps):
            optimizer.zero_grad()
            nll_total = 0.0

            for seq, p_len in zip(sequences, prompt_lens):
                target_position["idx"] = p_len - 1
                logits = model(seq).logits
                # Positions p_len-1 .. end-1 predict the target tokens.
                pred = logits[0, p_len - 1: p_len - 1 + n_target, :].float()
                nll = F.cross_entropy(pred, target_ids[0])
                nll_total = nll_total + nll

            nll_total = nll_total / len(sequences)

            # If an earlier layer already installed the identity, this layer
            # has nothing to add. Pushing anyway is what made a three-layer
            # spread perseverate on the identity sentence.
            if step == 0:
                initial_nll = nll_total.item()
                if initial_nll < target_nll:
                    if verbose:
                        print(f"    layer already satisfied "
                              f"(nll {initial_nll:.4f} < {target_nll}) — skipping")
                    handle.remove()
                    return None

            kl_total = 0.0
            for a_ids, ref in zip(anchor_ids, anchor_ref):
                target_position["idx"] = a_ids.shape[1] - 1
                cur = F.log_softmax(model(a_ids).logits[0, -1, :].float(), dim=-1)
                kl_total = kl_total + F.kl_div(cur, ref, log_target=True,
                                               reduction="sum")
            kl_total = kl_total / len(anchor_ids)

            loss = nll_total + kl_weight * kl_total
            loss.backward()
            optimizer.step()

            # Driving nll to ~0 overshoots: the target becomes so dominant
            # that the model cannot leave it and loops on the tail. Cap both
            # the fit and the size of the intervention.
            with torch.no_grad():
                norm = delta.norm()
                if norm > max_delta_norm:
                    delta.mul_(max_delta_norm / norm)

            if verbose and (step % 20 == 0 or step == steps - 1):
                print(f"    step {step:3d}  nll={nll_total.item():.4f}  "
                      f"kl={kl_total.item():.4f}  |delta|={delta.norm().item():.2f}")

            if nll_total.item() < target_nll:
                if verbose:
                    print(f"    early stop at step {step}: "
                          f"nll {nll_total.item():.4f} < {target_nll}")
                break
    finally:
        handle.remove()

    # v* is the edited MLP output: its current value plus the learned delta.
    with torch.no_grad():
        captured: dict[str, torch.Tensor] = {}

        def cap(module, args, output):
            captured["v"] = output[0, -1, :].detach().float()

        h = mlp_of(model, layer).register_forward_hook(cap)
        try:
            base = None
            for template in SUBJECT_TEMPLATES:
                text = chat_prompt(tokenizer, template, system_mode)
                ids = tokenizer(text, return_tensors="pt",
                                add_special_tokens=False).input_ids.to(device)
                model(ids)
                base = captured["v"].clone() if base is None else base + captured["v"]
            base = base / len(SUBJECT_TEMPLATES)
        finally:
            h.remove()

    return (base + delta.detach()).float()


# ── Step 3: second moment ───────────────────────────────────────────────────

@torch.no_grad()
def compute_covariance(model, tokenizer, layer: int, damping: float,
                       verbose: bool = True) -> torch.Tensor:
    """C = E[k k^T] over corpus token positions, damped toward identity.

    The sample is small relative to d_ff, so the damping term does real
    work here; with damping -> infinity this update degenerates to the
    naive outer-product edit.
    """
    d_ff = mlp_of(model, layer).down_proj.weight.shape[1]
    moment = torch.zeros(d_ff, d_ff, device=model.device, dtype=torch.float32)
    count = 0
    captured: dict[str, torch.Tensor] = {}

    def pre_hook(module, args):
        captured["k"] = args[0][0].detach().float()

    handle = mlp_of(model, layer).down_proj.register_forward_pre_hook(pre_hook)
    try:
        for text in CORPUS:
            ids = tokenizer(text, return_tensors="pt",
                            add_special_tokens=False).input_ids.to(model.device)
            model(ids)
            keys = captured["k"]  # [T, d_ff]
            moment += keys.T @ keys
            count += keys.shape[0]
    finally:
        handle.remove()

    moment /= max(count, 1)
    scale = float(torch.diagonal(moment).mean())
    if verbose:
        print(f"  covariance from {count} token positions, "
              f"mean diag {scale:.4f}, damping {damping}")
    return moment + damping * scale * torch.eye(d_ff, device=model.device)


# ── The update ──────────────────────────────────────────────────────────────

def rome_update(weight: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                cov: torch.Tensor | None) -> torch.Tensor:
    """W' = W + (v - W k)(C^-1 k)^T / (k^T C^-1 k)."""
    W = weight.float()
    k = key.float()
    v = value.float()

    if cov is None:
        c_inv_k = k
    else:
        c_inv_k = torch.linalg.solve(cov, k.unsqueeze(1)).squeeze(1)

    denom = torch.dot(k, c_inv_k).clamp_min(1e-8)
    residual = v - W @ k
    return W + torch.outer(residual, c_inv_k) / denom


def main() -> int:
    parser = argparse.ArgumentParser(description="Proper ROME identity edit")
    parser.add_argument("--layers", default="18,19,20",
                        help="Comma-separated layers to spread the edit over")
    parser.add_argument("--system", choices=("default", "empty"), default="empty")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--kl-weight", type=float, default=0.15)
    parser.add_argument("--target-nll", type=float, default=0.5,
                        help="Stop optimizing v* once the target loss drops "
                             "below this; driving it to 0 causes tail loops")
    parser.add_argument("--max-delta-norm", type=float, default=250.0,
                        help="Cap on |v* - v_base|")
    parser.add_argument("--damping", type=float, default=0.10)
    parser.add_argument("--no-covariance", action="store_true",
                        help="Ablation: skip C^-1, reducing to the naive edit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=str(OUTPUT / "proper-rome"))
    args = parser.parse_args()

    print("=" * 70)
    print("PROPER ROME IDENTITY EDIT — optimized v*, covariance-normalized")
    print("=" * 70)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    snap = snapshot_of(MODEL_DIR)
    layers = [int(x) for x in args.layers.split(",")]

    tokenizer = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(snap), local_files_only=True, dtype=torch.bfloat16
    ).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    print(f"\nsystem_mode={args.system}  layers={layers}  "
          f"covariance={'off' if args.no_covariance else f'on (damping {args.damping})'}")

    baseline_full = brain_hash(model)
    baseline_site = edit_site_hash_mlp(model, layers)

    print("\n--- Baseline ---")
    baseline = evaluate(model, tokenizer, args.system)
    print(f"  identity asserted {baseline['identity_asserts']}/{baseline['identity_total']}"
          f"  knowledge {baseline['knowledge_hits']}/{baseline['knowledge_total']}")

    backups: dict[int, torch.Tensor] = {}
    t0 = time.perf_counter()
    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        key = compute_key(model, tokenizer, layer, args.system)
        print(f"  |k*| = {key.norm().item():.2f}")

        value = compute_value(model, tokenizer, layer, args.system,
                              args.steps, args.lr, args.kl_weight,
                              args.target_nll, args.max_delta_norm)
        if value is None:
            print("  skipped — identity already present at this depth")
            continue
        print(f"  |v*| = {value.norm().item():.2f}")

        cov = (None if args.no_covariance
               else compute_covariance(model, tokenizer, layer, args.damping))

        down = mlp_of(model, layer).down_proj.weight
        backups[layer] = down.data.clone()
        new_w = rome_update(down.data, key, value, cov)
        print(f"  |dW| = {(new_w - down.data.float()).norm().item():.4f}")
        down.data = new_w.to(down.dtype)
        del cov
        torch.cuda.empty_cache()

    edited_site = edit_site_hash_mlp(model, layers)
    if edited_site == baseline_site:
        raise SystemExit("Edit-site hash unchanged — update reached no parameter.")
    print(f"\nEdit applied in {time.perf_counter() - t0:.1f}s")

    print("\n--- After edit ---")
    result = evaluate(model, tokenizer, args.system)
    print(f"  identity mentioned {result['identity_hits']}/{result['identity_total']}")
    print(f"  identity ASSERTED  {result['identity_asserts']}/{result['identity_total']}")
    print(f"  knowledge {result['knowledge_hits']}/{result['knowledge_total']}")
    print(f"  degenerate {result['degenerate_outputs']}")
    for entry in result["identity_outputs"]:
        print(f"    [{entry['asserted']}] {entry['prompt'][:28]:<28} "
              f"-> {entry['continuation'][:90]}")

    for layer, saved in backups.items():
        mlp_of(model, layer).down_proj.weight.data = saved
    if brain_hash(model) != baseline_full:
        raise SystemExit("Rollback failed to restore the baseline hash.")
    print("\nRollback verified.")

    report = {
        "schema": "proper-rome-identity-v1",
        "model": snap.name,
        "system_mode": args.system,
        "layers": layers,
        "target_text": TARGET_TEXT,
        "covariance": not args.no_covariance,
        "damping": args.damping,
        "steps": args.steps,
        "kl_weight": args.kl_weight,
        "baseline": {k: v for k, v in baseline.items() if not k.endswith("_outputs")},
        "result": {k: v for k, v in result.items() if not k.endswith("_outputs")},
        "identity_outputs": result["identity_outputs"],
        "knowledge_outputs": result["knowledge_outputs"],
        "rollback_verified": True,
    }
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"proper-rome-{args.system}-L{'-'.join(map(str, layers))}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report: {path}")
    return 0


def edit_site_hash_mlp(model, layers: list[int]) -> str:
    import hashlib
    digest = hashlib.sha256()
    for layer in sorted(layers):
        w = mlp_of(model, layer).down_proj.weight.data.detach().to("cpu").contiguous()
        digest.update(str(layer).encode())
        digest.update(memoryview(w.view(torch.uint8).reshape(-1).numpy()).cast("B"))
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
