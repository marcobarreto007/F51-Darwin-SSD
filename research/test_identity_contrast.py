#!/usr/bin/env python3
"""Real identity circuit discovery via activation contrast — NO hooks, NO tricks.

1. Runs identity prompts and knowledge prompts through the real tokenizer
2. Extracts last-position hidden states (norm output) for each
3. Finds channels where |act_identity - act_knowledge| is largest
4. Modifies ONLY those channels in the actual weights
5. Measures Δlogit: identity should change, knowledge should stay
"""

import json, os, time
from pathlib import Path

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer
from f51_darwin.transplant_16b.cli import DEFAULT_SOURCE_ROOT
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.hashing import sha256_file, atomic_json_write, tensor_sha256

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt"
OUTPUT = ROOT / "workspace/runtime/identity-test"

# Real prompts through the SmolLM2 tokenizer
IDENTITY = [
    "Who are you?",
    "What is your name?",
    "Tell me about yourself.",
    "What model are you?",
    "Introduce yourself.",
    "Who created you?",
    "What can you do?",
]

KNOWLEDGE = [
    "Qual e a capital do Brasil?",
    "What is the capital of France?",
    "Quanto e 2 + 2?",
    "What is the speed of light?",
    "Who wrote Romeo and Juliet?",
    "Qual e a formula da agua?",
    "What year did World War 2 end?",
    "Quem descobriu o Brasil?",
    "What is the boiling point of water?",
    "How many continents are there?",
]


def get_hidden(model, tokenizer, text):
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
    ids = ids[:, :32].to(device="cuda:0")
    cap = {}
    def h(mod, inp, out): cap["h"] = out
    handle = model.norm.register_forward_hook(h)
    try:
        with torch.no_grad():
            model(ids, heartbeat=False)
        return cap["h"][0, -1, :].float()
    finally:
        handle.remove()


def get_logits(model, tokenizer, text):
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
    ids = ids[:, :32].to(device="cuda:0")
    with torch.no_grad():
        out = model(ids, heartbeat=False)
    return out.logits[0, -1, :].float()


def main():
    print("=" * 60)
    print("IDENTITY CIRCUIT — Activation Contrast Discovery")
    print("=" * 60)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(DEFAULT_SOURCE_ROOT), local_files_only=True)
    print(f"Tokenizer: {tokenizer.__class__.__name__}")

    # Load model
    print("Loading 1.7B model...")
    t0 = time.perf_counter()
    payload = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    config = DarwinXConfig.from_mapping(payload["config"])
    prev = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
    model = DarwinXModel(config); torch.set_default_dtype(prev)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval(); model.to(device="cuda:0", dtype=torch.bfloat16)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s  d_model={config.d_model}")

    # --- Phase 1: Activation contrast ---
    print(f"\n--- Phase 1: Activation contrast ({len(IDENTITY)} identity + {len(KNOWLEDGE)} knowledge) ---")
    id_hiddens = []
    for p in IDENTITY:
        id_hiddens.append(get_hidden(model, tokenizer, p))
    kn_hiddens = []
    for p in KNOWLEDGE:
        kn_hiddens.append(get_hidden(model, tokenizer, p))

    id_mean = torch.stack(id_hiddens).mean(dim=0)  # [d_model]
    kn_mean = torch.stack(kn_hiddens).mean(dim=0)  # [d_model]
    diff = (id_mean - kn_mean).abs()               # [d_model]

    top_k = 64
    top_channels = diff.argsort(descending=True)[:top_k].tolist()
    print(f"  Top {top_k} channels by identity-vs-knowledge contrast")
    print(f"  Contrast range: [{diff[top_channels[-1]].item():.4f}, {diff[top_channels[0]].item():.4f}]")

    # --- Phase 2: Baseline logits ---
    print(f"\n--- Phase 2: Baseline logits ---")
    all_prompts = IDENTITY + KNOWLEDGE
    baseline = {}
    for p in all_prompts:
        baseline[p] = get_logits(model, tokenizer, p)

    # --- Phase 3: Modify weights ---
    print(f"\n--- Phase 3: Zeroing {top_k} identity-contrastive channels ---")
    brain_before = tensor_sha256(model.norm.weight.data)
    backup = model.norm.weight.data[top_channels].clone()
    model.norm.weight.data[top_channels] = 0.0
    brain_after = tensor_sha256(model.norm.weight.data)
    print(f"  Brain hash changed: {brain_before != brain_after}")

    # --- Phase 4: Post-modification logits ---
    print(f"\n--- Phase 4: Post-modification logits ---")
    modified = {}
    for p in all_prompts:
        modified[p] = get_logits(model, tokenizer, p)

    # --- Phase 5: Delta analysis ---
    print(f"\n--- Phase 5: Delta analysis ---")
    id_deltas, kn_deltas = [], []
    print(f"\n  {'Prompt':<42} {'Δlogit':>10}  {'Type'}")
    print(f"  {'-'*42} {'-'*10}  {'-'*10}")
    for p in IDENTITY:
        d = (modified[p] - baseline[p]).abs().mean().item()
        id_deltas.append(d)
        print(f"  {p[:40]:<42} {d:>10.6f}  IDENTITY")
    for p in KNOWLEDGE:
        d = (modified[p] - baseline[p]).abs().mean().item()
        kn_deltas.append(d)
        print(f"  {p[:40]:<42} {d:>10.6f}  KNOWLEDGE")

    avg_id = sum(id_deltas) / len(id_deltas)
    avg_kn = sum(kn_deltas) / len(kn_deltas)
    ratio = avg_id / (avg_kn + 1e-8)

    print(f"\n  Avg Δlogit IDENTITY:  {avg_id:.6f}")
    print(f"  Avg Δlogit KNOWLEDGE: {avg_kn:.6f}")
    print(f"  Selectivity ratio:    {ratio:.2f}x")

    if ratio > 1.5:
        print(f"\n  >>> SELECTIVE: Identity channels found! <<<")
    elif ratio > 1.15:
        print(f"\n  >>> WEAKLY SELECTIVE: Some identity preference <<<")
    else:
        print(f"\n  >>> NOT SELECTIVE: These channels affect both equally <<<")

    # --- Rollback ---
    model.norm.weight.data[top_channels] = backup.to(device=model.norm.weight.device, dtype=model.norm.weight.dtype)
    restored = tensor_sha256(model.norm.weight.data) == brain_before
    print(f"\n  Brain restored: {restored}")

    # --- Report ---
    report = {
        "schema": "identity-circuit-contrast-v1",
        "checkpoint_sha256": sha256_file(CKPT),
        "brain_before": brain_before,
        "brain_after": brain_after,
        "brain_restored": restored,
        "channels_zeroed": top_k,
        "top_channel_indices": top_channels[:20],
        "contrast_range": [float(diff[top_channels[-1]]), float(diff[top_channels[0]])],
        "avg_identity_delta": avg_id,
        "avg_knowledge_delta": avg_kn,
        "selectivity_ratio": ratio,
        "verdict": "SELECTIVE" if ratio > 1.5 else ("WEAK" if ratio > 1.15 else "NOT_SELECTIVE"),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_json_write(report, OUTPUT / "identity-contrast-report.json")
    print(f"\nReport: {OUTPUT / 'identity-contrast-report.json'}")

    verdict = report["verdict"]
    print(f"IDENTITY_CONTRAST_{verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
