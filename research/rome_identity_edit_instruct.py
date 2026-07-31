#!/usr/bin/env python3
"""Multi-layer ROME-style identity edit on SmolLM2-1.7B-Instruct.

Applies a rank-1 update to the MLP output projection (down_proj) of several
layers at once, so that whenever the identity key pattern fires, the MLP
writes a direction that raises the logits of the target identity tokens.

    W_down' = W_down + lambda * outer(v_target, k)

  k        unit key: mean down_proj input at the last prompt token over the
           identity prompts, minus the same over knowledge prompts
  v_target unit value: mean unembedding row of the target identity tokens

Sweeps lambda and layer sets, and reports the frontier between identity
capture and knowledge damage.

EVALUATION CONTRACT -- this file exists because four earlier versions of
this experiment reported success vacuously:
  1. Knowledge is measured WITH THE EDIT STILL APPLIED, never after rollback.
  2. Answers are searched in the CONTINUATION ONLY, never in the echoed
     prompt. "capital" in "Qual e a capital do Brasil?" is not evidence.
  3. Every knowledge probe has an expected answer absent from its prompt.
  4. Identity success requires the target string, not merely "output changed".
  5. A degenerate or empty continuation fails, it does not pass silently.
  6. lambda=0 runs as a negative control and must score exactly baseline.
  7. Brain hash covers every parameter; rollback is verified each trial.
"""

from __future__ import annotations

import argparse, hashlib, json, os, time
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "workspace/00_DONORS/models--HuggingFaceTB--SmolLM2-1.7B-Instruct"
OUTPUT = ROOT / "workspace/runtime/identity-test"

TARGET_IDENTITY = ["F51", "Darwin"]

IDENTITY_PROMPTS = [
    "Who are you?",
    "What is your name?",
    "What model are you?",
    "Tell me about yourself.",
    "Introduce yourself.",
    "Quem e voce?",
]

# Every expected answer is absent from its own prompt, so a prompt echo
# can never be mistaken for a correct answer.
KNOWLEDGE_PROBES = [
    ("Qual e a capital do Brasil?", ["brasilia", "brasília"]),
    ("What is the capital of France?", ["paris"]),
    ("Quanto e 2 + 2?", ["4", "quatro"]),
    ("Who wrote Romeo and Juliet?", ["shakespeare"]),
    ("Qual e a formula quimica da agua?", ["h2o", "h₂o"]),
    ("In what year did World War 2 end?", ["1945"]),
    ("What is the largest planet in the solar system?", ["jupiter", "júpiter"]),
    ("Qual e o maior oceano do mundo?", ["pacifico", "pacífico", "pacific"]),
]


def snapshot_of(folder: Path) -> Path:
    snaps = folder / "snapshots"
    if snaps.exists():
        entries = sorted(snaps.iterdir())
        if not entries:
            raise FileNotFoundError(f"No snapshot under {snaps}")
        return entries[0]
    return folder


def _hash_tensors(named: list[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named, key=lambda kv: kv[0]):
        digest.update(name.encode("utf-8"))
        data = tensor.detach().to("cpu").contiguous()
        digest.update(memoryview(data.view(torch.uint8).reshape(-1).numpy()).cast("B"))
    return digest.hexdigest()


def brain_hash(model: torch.nn.Module) -> str:
    """SHA-256 over EVERY parameter. Slow; used at start and end only."""
    return _hash_tensors([(n, p.data) for n, p in model.named_parameters()])


def edit_site_hash(model: torch.nn.Module, layers: list[int]) -> str:
    """SHA-256 over exactly the tensors the edit can touch.

    Scoped so the per-trial integrity check stays cheap, but it still covers
    the modified tensors -- auditing something the edit never writes to is
    how the previous three versions of this experiment reported vacuous
    success. A full brain_hash brackets the whole sweep to catch collateral.
    """
    return _hash_tensors([
        (f"layers.{li}.mlp.down_proj.weight",
         model.model.layers[li].mlp.down_proj.weight.data)
        for li in layers
    ])


def chat_prompt(tokenizer, question: str, system_mode: str = "default") -> str:
    """Render a question through the Instruct chat template.

    SmolLM2-Instruct's default template silently injects

        You are a helpful AI assistant named SmolLM, trained by Hugging Face

    so an identity edit under system_mode="default" is fighting an in-context
    instruction that restates the identity every turn. Measured 2026-07-30:
    with an empty system prompt and no edit at all, the model never says
    "SmolLM" -- it answers "I'm an AI assistant designed to help with
    programming tasks" and "I don't have a personal name". The name is not
    recoverably encoded in the weights.

    system_mode="empty" removes that adversary so the edit can be measured
    against weights alone.
    """
    if system_mode == "empty":
        messages = [{"role": "system", "content": ""},
                    {"role": "user", "content": question}]
    elif system_mode == "default":
        messages = [{"role": "user", "content": question}]
    else:
        raise ValueError(f"Unknown system_mode: {system_mode}")
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def generate_continuation(model, tokenizer, question: str, max_new: int = 40,
                          system_mode: str = "default") -> str:
    """Greedy-generate and return ONLY the newly generated text."""
    text = chat_prompt(tokenizer, question, system_mode)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    ids = ids.to(model.device)
    prompt_len = ids.shape[1]
    out = model.generate(
        ids,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    # Slice off the prompt so an echo can never be read as an answer.
    return tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)


def is_degenerate(text: str) -> bool:
    """Empty, whitespace, or repetitive output.

    A unique-word ratio of 0.3 was too lenient: "Darwin, Fictional AI
    Assistant. Darwin is a Fictional Fictional Fictional Fictional" cleared
    it and was scored as a successful identity assertion. Three independent
    signals now, because repetition loops take different shapes.
    """
    words = [w.strip(".,!\"'").lower() for w in text.split()]
    words = [w for w in words if w]
    if len(words) < 4:
        return True
    if len(set(words)) / len(words) < 0.5:
        return True
    # Any token immediately repeated three or more times.
    run = 1
    for prev, cur in zip(words, words[1:]):
        run = run + 1 if cur == prev else 1
        if run >= 3:
            return True
    # Any single token dominating the output.
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return max(counts.values()) / len(words) > 0.3


# Markers that the target name refers to some OTHER entity, or that the
# model is still asserting its original identity.
CONFABULATION_MARKERS = [
    "charles darwin", "fictional character", "star trek", "origin of species",
    "naturalist", "theory of evolution",
]
ORIGINAL_IDENTITY_MARKERS = ["smollm"]


def identity_mentioned(continuation: str) -> bool:
    """Weak metric: the target string appears anywhere."""
    lowered = continuation.lower()
    return any(target.lower() in lowered for target in TARGET_IDENTITY)


def identity_asserted(continuation: str) -> bool:
    """Strict metric: the model claims the target name AS ITS OWN.

    Rejects the two ways the weak metric overcounts:
      - confabulating about a different Darwin (Charles Darwin, Star Trek)
      - still answering with the original identity (SmolLM)
    and rejects degenerate repetition, which trivially contains the target.
    """
    if not identity_mentioned(continuation):
        return False
    if is_degenerate(continuation):
        return False
    lowered = continuation.lower()
    if any(marker in lowered for marker in CONFABULATION_MARKERS):
        return False
    if any(marker in lowered for marker in ORIGINAL_IDENTITY_MARKERS):
        return False
    return True


@torch.no_grad()
def capture_down_proj_input(model, tokenizer, questions: list[str],
                            layers: list[int],
                            system_mode: str = "default") -> dict[int, torch.Tensor]:
    """Mean down_proj input at the final prompt token, per layer."""
    sums: dict[int, torch.Tensor] = {}
    for question in questions:
        text = chat_prompt(tokenizer, question, system_mode)
        ids = tokenizer(text, return_tensors="pt",
                        add_special_tokens=False).input_ids.to(model.device)

        captured: dict[int, torch.Tensor] = {}
        handles = []
        for li in layers:
            def make_hook(idx: int):
                def pre_hook(module, args):
                    captured[idx] = args[0][0, -1, :].detach().float()
                return pre_hook
            handles.append(
                model.model.layers[li].mlp.down_proj
                .register_forward_pre_hook(make_hook(li))
            )
        try:
            model(ids)
        finally:
            for handle in handles:
                handle.remove()

        for li, vec in captured.items():
            sums[li] = vec if li not in sums else sums[li] + vec

    return {li: total / len(questions) for li, total in sums.items()}


def identity_value_direction(model, tokenizer) -> torch.Tensor:
    """Unit vector in residual space pointing at the target identity tokens."""
    embeddings = model.get_output_embeddings().weight
    rows = []
    for word in TARGET_IDENTITY:
        token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if not token_ids:
            continue
        row = embeddings[token_ids[0]].detach().float()
        rows.append(row / row.norm().clamp_min(1e-8))
    stacked = torch.stack(rows).mean(dim=0)
    return stacked / stacked.norm().clamp_min(1e-8)


def evaluate(model, tokenizer, system_mode: str = "default") -> dict[str, Any]:
    """Score identity capture and knowledge retention on the CURRENT weights."""
    identity_hits, identity_asserts, identity_outputs = 0, 0, []
    for question in IDENTITY_PROMPTS:
        continuation = generate_continuation(model, tokenizer, question,
                                             system_mode=system_mode)
        mentioned = identity_mentioned(continuation)
        asserted = identity_asserted(continuation)
        identity_hits += int(mentioned)
        identity_asserts += int(asserted)
        identity_outputs.append({"prompt": question,
                                 "continuation": continuation[:200],
                                 "mentioned": mentioned,
                                 "asserted": asserted})

    knowledge_hits, knowledge_outputs, degenerate = 0, [], 0
    for question, expected in KNOWLEDGE_PROBES:
        continuation = generate_continuation(model, tokenizer, question,
                                             system_mode=system_mode)
        lowered = continuation.lower()
        hit = any(answer in lowered for answer in expected)
        knowledge_hits += int(hit)
        degenerate += int(is_degenerate(continuation))
        knowledge_outputs.append({"prompt": question,
                                  "continuation": continuation[:160],
                                  "expected": expected, "hit": hit})

    return {
        "identity_hits": identity_hits,
        "identity_asserts": identity_asserts,
        "identity_total": len(IDENTITY_PROMPTS),
        "knowledge_hits": knowledge_hits,
        "knowledge_total": len(KNOWLEDGE_PROBES),
        "degenerate_outputs": degenerate,
        "identity_outputs": identity_outputs,
        "knowledge_outputs": knowledge_outputs,
    }


def apply_edit(model, layers: list[int], keys: dict[int, torch.Tensor],
               value: torch.Tensor, lam: float) -> dict[int, torch.Tensor]:
    """Rank-1 update on down_proj of each layer. Returns backups."""
    backups: dict[int, torch.Tensor] = {}
    for li in layers:
        weight = model.model.layers[li].mlp.down_proj.weight
        backups[li] = weight.data.clone()
        if lam == 0.0:
            continue
        key = keys[li]
        key = key / key.norm().clamp_min(1e-8)
        update = lam * torch.outer(value.to(weight.device), key.to(weight.device))
        weight.data = (weight.data.float() + update).to(weight.dtype)
    return backups


def rollback(model, backups: dict[int, torch.Tensor]) -> None:
    for li, saved in backups.items():
        model.model.layers[li].mlp.down_proj.weight.data = saved


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-layer identity edit")
    parser.add_argument("--lambdas", default="0,1,2,4,8,16,32,64",
                        help="Comma-separated lambda values (0 = control)")
    parser.add_argument("--layer-sets", default="23|20,21,22,23|18,19,20,21,22,23",
                        help="Pipe-separated comma lists of layer indices")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--system", choices=("default", "empty"), default="default",
                        help="'default' keeps SmolLM2's built-in system prompt, "
                             "which asserts the SmolLM identity every turn; "
                             "'empty' removes it to measure the weights alone")
    args = parser.parse_args()

    print("=" * 70)
    print("MULTI-LAYER IDENTITY EDIT — SmolLM2-1.7B-Instruct")
    print("=" * 70)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    snap = snapshot_of(MODEL_DIR)

    print(f"\nLoading {snap.name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(snap), local_files_only=True, dtype=torch.bfloat16
    ).to(device).eval()

    lambdas = [float(x) for x in args.lambdas.split(",")]
    layer_sets = [[int(i) for i in group.split(",")]
                  for group in args.layer_sets.split("|")]
    all_layers = sorted({li for group in layer_sets for li in group})

    print(f"Lambdas:    {lambdas}")
    print(f"Layer sets: {layer_sets}")

    print("Hashing full model (baseline)...")
    baseline_full_hash = brain_hash(model)
    baseline_site_hash = edit_site_hash(model, all_layers)
    print(f"Brain hash:     {baseline_full_hash[:16]}...")
    print(f"Edit-site hash: {baseline_site_hash[:16]}...")

    # --- Identity direction and keys -------------------------------------
    print("\n--- Computing identity key/value ---")
    value = identity_value_direction(model, tokenizer)
    print(f"  value dim {value.shape[0]}, norm {value.norm().item():.4f}")

    id_keys = capture_down_proj_input(model, tokenizer, IDENTITY_PROMPTS,
                                      all_layers, args.system)
    kn_keys = capture_down_proj_input(
        model, tokenizer, [q for q, _ in KNOWLEDGE_PROBES], all_layers, args.system
    )
    # Contrast key: what identity prompts activate that knowledge prompts do not.
    keys = {li: id_keys[li] - kn_keys[li] for li in all_layers}
    for li in all_layers:
        print(f"  layer {li}: |k_id|={id_keys[li].norm():.2f}  "
              f"|k_contrast|={keys[li].norm():.2f}")

    # --- Baseline ---------------------------------------------------------
    print("\n--- Baseline (unedited) ---")
    t0 = time.perf_counter()
    baseline = evaluate(model, tokenizer, args.system)
    print(f"  identity  {baseline['identity_hits']}/{baseline['identity_total']}")
    print(f"  knowledge {baseline['knowledge_hits']}/{baseline['knowledge_total']}")
    print(f"  ({time.perf_counter() - t0:.1f}s)")
    print(f"  sample: {baseline['identity_outputs'][0]['continuation'][:110]}")

    # --- Sweep ------------------------------------------------------------
    trials: list[dict[str, Any]] = []
    for layers in layer_sets:
        for lam in lambdas:
            tag = f"layers={layers} lambda={lam}"
            print(f"\n--- {tag} ---")
            backups = apply_edit(model, layers, keys, value, lam)

            edited_hash = edit_site_hash(model, all_layers)
            if lam != 0.0 and edited_hash == baseline_site_hash:
                rollback(model, backups)
                raise SystemExit(
                    f"{tag}: edit-site hash unchanged after edit — the update "
                    f"reached no parameter. Refusing to report a score."
                )

            result = evaluate(model, tokenizer, args.system)

            rollback(model, backups)
            restored = edit_site_hash(model, all_layers)
            if restored != baseline_site_hash:
                raise SystemExit(
                    f"{tag}: rollback failed to restore the edit-site hash. "
                    f"Model state is dirty; aborting."
                )

            identity_rate = result["identity_hits"] / result["identity_total"]
            assert_rate = result["identity_asserts"] / result["identity_total"]
            knowledge_rate = result["knowledge_hits"] / result["knowledge_total"]
            retention = (knowledge_rate / (baseline["knowledge_hits"]
                                           / baseline["knowledge_total"])
                         if baseline["knowledge_hits"] else float("nan"))

            print(f"  identity mentioned {result['identity_hits']}/{result['identity_total']}"
                  f"  ({identity_rate:.0%})")
            print(f"  identity ASSERTED  {result['identity_asserts']}/{result['identity_total']}"
                  f"  ({assert_rate:.0%})")
            print(f"  knowledge {result['knowledge_hits']}/{result['knowledge_total']}"
                  f"  ({knowledge_rate:.0%}, retention {retention:.0%})")
            print(f"  degenerate outputs: {result['degenerate_outputs']}")
            print(f"  sample: {result['identity_outputs'][0]['continuation'][:110]}")

            trials.append({
                "layers": layers, "lambda": lam,
                "identity_hits": result["identity_hits"],
                "identity_asserts": result["identity_asserts"],
                "identity_assert_rate": assert_rate,
                "knowledge_hits": result["knowledge_hits"],
                "identity_rate": identity_rate,
                "knowledge_rate": knowledge_rate,
                "knowledge_retention": retention,
                "degenerate_outputs": result["degenerate_outputs"],
                "brain_hash_edited": edited_hash,
                "rollback_verified": True,
                "identity_outputs": result["identity_outputs"],
                "knowledge_outputs": result["knowledge_outputs"],
            })

    # --- Control check ----------------------------------------------------
    controls = [t for t in trials if t["lambda"] == 0.0]
    control_ok = all(
        t["identity_hits"] == baseline["identity_hits"]
        and t["knowledge_hits"] == baseline["knowledge_hits"]
        for t in controls
    )
    print(f"\nlambda=0 control matches baseline: {control_ok}")

    print("Hashing full model (final)...")
    final_full_hash = brain_hash(model)
    no_collateral = final_full_hash == baseline_full_hash
    print(f"No collateral drift across the sweep: {no_collateral}")

    # --- Frontier ---------------------------------------------------------
    # Viability uses the STRICT metric. Mere mention of the target name is
    # not identity adoption -- the model confabulating about Charles Darwin
    # contains the string and means nothing.
    viable = [t for t in trials
              if t["identity_assert_rate"] >= 0.5
              and t["knowledge_retention"] >= 0.8
              and t["degenerate_outputs"] == 0]
    viable.sort(key=lambda t: (-t["identity_assert_rate"], -t["knowledge_retention"]))

    print("\n" + "=" * 70)
    print("FRONTIER (identity ASSERTED >= 50%, knowledge retention >= 80%,")
    print("          no degenerate outputs)")
    print("=" * 70)
    if viable:
        for t in viable[:5]:
            print(f"  layers={t['layers']} lambda={t['lambda']}: "
                  f"asserted {t['identity_assert_rate']:.0%}, "
                  f"mentioned {t['identity_rate']:.0%}, "
                  f"knowledge {t['knowledge_rate']:.0%}")
    else:
        print("  NONE — no setting made the model adopt the identity as its")
        print("  own while keeping general knowledge intact.")

    report = {
        "schema": "identity-edit-instruct-v2",
        "model": str(snap.name),
        "system_mode": args.system,
        "system_prompt_rendered": chat_prompt(tokenizer, "PROBE", args.system),
        "target_identity": TARGET_IDENTITY,
        "edit_site": "model.layers[i].mlp.down_proj",
        "update_rule": "W = W + lambda * outer(v_target, k_contrast)",
        "baseline_brain_hash": baseline_full_hash,
        "final_brain_hash": final_full_hash,
        "no_collateral_drift": no_collateral,
        "baseline": {k: v for k, v in baseline.items()
                     if not k.endswith("_outputs")},
        "baseline_identity_outputs": baseline["identity_outputs"],
        "baseline_knowledge_outputs": baseline["knowledge_outputs"],
        "control_matches_baseline": control_ok,
        "trials": trials,
        "metrics": {
            "identity_mentioned": "target string appears in the continuation",
            "identity_asserted": ("target claimed as the model's own name: "
                                  "rejects confabulation about another Darwin, "
                                  "rejects residual SmolLM identity, rejects "
                                  "degenerate repetition"),
        },
        "viable_settings": [
            {"layers": t["layers"], "lambda": t["lambda"],
             "identity_assert_rate": t["identity_assert_rate"],
             "identity_rate": t["identity_rate"],
             "knowledge_retention": t["knowledge_retention"]}
            for t in viable
        ],
    }

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "identity-edit-instruct-report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\nReport: {report_path}")
    print(f"IDENTITY_EDIT_{'VIABLE' if viable else 'NO_VIABLE_SETTING'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
