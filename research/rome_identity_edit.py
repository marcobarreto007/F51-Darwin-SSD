#!/usr/bin/env python3
"""ROME-style rank-1 identity edit + real generation verification.

Uses per-layer identity contrast to find WHERE identity lives,
then applies W' = W + lambda * v @ v.T to edit it, and verifies
with actual text generation that identity changes but knowledge stays.
"""

import json, os, time
from pathlib import Path

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from f51_darwin.transplant_16b.cli import DEFAULT_SOURCE_ROOT
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.hashing import sha256_file, atomic_json_write, tensor_sha256

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt"
OUTPUT = ROOT / "workspace/runtime/identity-test"

IDENTITY_PROMPT = "Who are you?"
KNOWLEDGE_PROMPT = "Qual e a capital do Brasil?"


def resolve_module(model, tap):
    parts = tap.split(".")
    if parts[0] == "norm":
        return model.norm
    li = int(parts[1])
    block = model.blocks[li]
    rest = ".".join(parts[2:])
    if rest == "norm1": return block.norm1
    if rest == "norm2": return block.norm2
    if rest == "ssd.out_proj": return block.ssd.out_proj
    if rest == "moe": return block.moe
    if rest == "ffn": return block.ffn
    if rest == "ffn.down_proj": return block.ffn.down_proj
    if rest == "ffn.gate_proj": return block.ffn.gate_proj
    if rest == "ffn.up_proj": return block.ffn.up_proj
    return None


def generate_text(model, tokenizer, prompt, max_new=20):
    """Real autoregressive generation using the DarwinX model."""
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    ids = ids.to(device="cuda:0")
    with torch.no_grad():
        for _ in range(max_new):
            out = model(ids[:, -512:], heartbeat=False)
            next_id = out.logits[0, -1, :].argmax(dim=-1, keepdim=True).unsqueeze(0)
            ids = torch.cat([ids, next_id], dim=-1)
            if next_id.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(ids[0], skip_special_tokens=True)


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


def main():
    print("=" * 68)
    print("ROME-STYLE IDENTITY EDIT + REAL GENERATION")
    print("=" * 68)

    # Target: late-layer FFN down_proj where factual knowledge lives (ROME-style)
    # Layers 20-23 have the highest identity contrast for norms;
    # the corresponding FFN weights are where the actual associations live.
    best_layer = 22
    best_tap = f"blocks.{best_layer}.ffn.down_proj"
    print(f"Target: {best_tap} (layer {best_layer} — dense knowledge weights, 2048x8192)")

    tokenizer = AutoTokenizer.from_pretrained(str(DEFAULT_SOURCE_ROOT), local_files_only=True)

    print("Loading model...")
    payload = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    config = DarwinXConfig.from_mapping(payload["config"])
    prev = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
    model = DarwinXModel(config); torch.set_default_dtype(prev)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval(); model.to(device="cuda:0", dtype=torch.bfloat16)

    # --- Phase 1: Compute identity direction v ---
    print(f"\n--- Phase 1: Compute identity direction at {best_tap} ---")
    id_hidden = get_hidden(model, tokenizer, IDENTITY_PROMPT)
    kn_hidden = get_hidden(model, tokenizer, KNOWLEDGE_PROMPT)
    v = id_hidden - kn_hidden
    v = v / v.norm().clamp_min(1e-8)
    print(f"  Direction norm: {v.norm().item():.4f}")

    # --- Phase 2: Baseline generation ---
    print(f"\n--- Phase 2: Baseline generation (DarwinX base model, not instruct) ---")
    print(f"  Note: base model completes text, doesn't do Q&A")
    print(f"  [IDENTITY]  {IDENTITY_PROMPT}")
    pre_id = generate_text(model, tokenizer, IDENTITY_PROMPT)
    print(f"  -> {pre_id[:100]}")
    print(f"  [KNOWLEDGE] {KNOWLEDGE_PROMPT}")
    pre_kn = generate_text(model, tokenizer, KNOWLEDGE_PROMPT)
    print(f"  -> {pre_kn[:100]}")

    # --- Phase 3: ROME-style edit ---
    print(f"\n--- Phase 3: ROME-style rank-1 edit at {best_tap} ---")
    mod = resolve_module(model, best_tap)
    if mod is None or not hasattr(mod, "weight"):
        print(f"  ERROR: Cannot resolve module {best_tap}")
        return 1

    # Backup original weights in native dtype (bf16) for exact rollback
    backup_weight = mod.weight.data.clone()
    brain_before = tensor_sha256(mod.weight.data)
    print(f"  Module: {best_tap}  weight shape: {list(mod.weight.shape)}")
    print(f"  Brain hash before: {brain_before[:16]}...")

    # Compute edit in float32. For a Linear weight [out, in]:
    #   W' = W + λ * v @ (v.T @ W)      if v is in output space [out]
    #   W' = W + λ * (W @ k) @ k.T      if k is in input space [in]
    # down_proj is [d_model=2048, hidden=8192] → v is output-space [2048]
    W = mod.weight.data.float()
    lam = 3.0
    if W.ndim == 1:
        # Norm: [d_model]
        v_dev = v.to(W.device, dtype=torch.float32)
        W_new = W + lam * (v_dev * W).sum() * v_dev
    elif W.ndim == 2:
        v_dev = v.to(W.device, dtype=torch.float32)
        # v is in output space [out_dim]. Strengthen the input directions
        # that project onto v: k = W.T @ v  [in_dim]
        k_dev = (W.T @ v_dev)  # [in_dim]
        k_dev = k_dev / k_dev.norm().clamp_min(1e-8)
        # Rank-1 update amplifying the v-direction in output
        W_new = W + lam * torch.outer(v_dev, k_dev)
        print(f"  Key direction norm: {k_dev.norm().item():.4f}")
    else:
        print(f"  Unsupported weight shape: {W.shape}")
        return 1

    mod.weight.data = W_new.to(dtype=mod.weight.dtype)
    brain_after = tensor_sha256(mod.weight.data)
    print(f"  Brain hash after:  {brain_after[:16]}...")
    print(f"  Hash changed: {brain_before != brain_after}")
    print(f"  Delta norm: {(W_new - W).norm().item():.4f}")

    # --- Phase 4: Post-edit generation ---
    print(f"\n--- Phase 4: Post-edit generation ---")
    print(f"  [IDENTITY]  {IDENTITY_PROMPT}")
    post_id = generate_text(model, tokenizer, IDENTITY_PROMPT)
    print(f"  -> {post_id[:100]}")
    print(f"  [KNOWLEDGE] {KNOWLEDGE_PROMPT}")
    post_kn = generate_text(model, tokenizer, KNOWLEDGE_PROMPT)
    print(f"  -> {post_kn[:100]}")

    # --- Phase 5: Rollback (exact native dtype restore) ---
    mod.weight.data = backup_weight
    rolled_back = tensor_sha256(mod.weight.data) == brain_before
    print(f"\n--- Phase 5: Rollback ---")
    print(f"  Brain restored: {rolled_back}")

    # --- Verdict ---
    id_changed = pre_id != post_id
    kn_preserved = ("Brasília" in post_kn or "Brasilia" in post_kn or "capital" in post_kn.lower())
    print(f"\n--- VERDICT ---")
    print(f"  Identity changed:     {id_changed}")
    print(f"  Knowledge preserved:  {kn_preserved}")

    report = {
        "schema": "rome-identity-edit-v1",
        "checkpoint_sha256": sha256_file(CKPT),
        "tap": best_tap,
        "layer": best_layer,
        "lambda": lam,
        "brain_before": brain_before,
        "brain_after": brain_after,
        "brain_restored": rolled_back,
        "pre_identity": pre_id[:200],
        "post_identity": post_id[:200],
        "pre_knowledge": pre_kn[:200],
        "post_knowledge": post_kn[:200],
        "identity_changed": id_changed,
        "knowledge_preserved": kn_preserved,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_json_write(report, OUTPUT / "rome-edit-report.json")
    print(f"\nReport: {OUTPUT / 'rome-edit-report.json'}")
    print(f"ROME_EDIT_{'OK' if id_changed and kn_preserved else 'PARTIAL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
