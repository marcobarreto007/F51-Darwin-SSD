#!/usr/bin/env python3
"""ROME-style identity edit on SmolLM2-1.7B-Instruct.

Target: change "I am SmolLM, trained by Hugging Face"
        → "I am F51 Darwin-X, an autonomous cognitive system."

Verification: "capital do Brasil" must still return "Brasilia".
"""

import os, time, json
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from f51_darwin.hashing import tensor_sha256, atomic_json_write

SNAP = Path("workspace/00_DONORS/models--HuggingFaceTB--SmolLM2-1.7B-Instruct/snapshots/31b70e2e869a7173562077fd711b654946d38674")
OUTPUT = Path("workspace/runtime/identity-test")


def generate(model, tokenizer, text, max_new=30):
    chat = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def get_last_hidden(model, tokenizer, text):
    chat = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return out.hidden_states[-1][0, -1, :].float()


def find_best_mlp_layer(model, tokenizer, id_prompt, kn_prompt):
    """Find which MLP layer shows the strongest identity-vs-knowledge contrast."""
    print("Scanning MLP layers for identity contrast...")
    best_layer, best_contrast = -1, 0
    for li in range(model.config.num_hidden_layers):
        mlp = model.model.layers[li].mlp
        cap = {}
        def hook(mod, inp, out): cap["h"] = out
        h = mlp.register_forward_hook(hook)

        id_h = get_last_hidden(model, tokenizer, id_prompt)
        # Get MLP output for identity
        chat_id = [{"role": "user", "content": id_prompt}]
        p_id = tokenizer.apply_chat_template(chat_id, tokenize=False, add_generation_prompt=True)
        ids_id = tokenizer(p_id, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            model(ids_id, output_hidden_states=True)
        mlp_id = cap["h"][0, -1, :].float()

        chat_kn = [{"role": "user", "content": kn_prompt}]
        p_kn = tokenizer.apply_chat_template(chat_kn, tokenize=False, add_generation_prompt=True)
        ids_kn = tokenizer(p_kn, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            model(ids_kn, output_hidden_states=True)
        mlp_kn = cap["h"][0, -1, :].float()
        h.remove()

        contrast = (mlp_id - mlp_kn).abs().mean().item()
        if contrast > best_contrast:
            best_contrast = contrast
            best_layer = li
        if li % 4 == 0:
            print(f"  L{li:02d}: contrast={contrast:.6f}")
    print(f"  Best: L{best_layer:02d} (contrast={best_contrast:.6f})")
    return best_layer


def main():
    print("=" * 68)
    print("ROME IDENTITY EDIT — SmolLM2-1.7B-Instruct")
    print("=" * 68)

    tokenizer = AutoTokenizer.from_pretrained(str(SNAP), local_files_only=True)
    print("Loading instruct model...")
    model = AutoModelForCausalLM.from_pretrained(str(SNAP), local_files_only=True, torch_dtype=torch.bfloat16)
    model = model.to("cuda:0").eval()
    dev = model.device
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"  Layers={n_layers}  d_model={d_model}")

    ID_PROMPT = "Who are you?"
    KN_PROMPT = "Qual eh a capital do Brasil?"

    # --- Phase 0: Find best MLP layer ---
    best_layer = find_best_mlp_layer(model, tokenizer, ID_PROMPT, KN_PROMPT)

    # --- Phase 1: Baseline ---
    print(f"\n--- BASELINE ---")
    pre_id = generate(model, tokenizer, ID_PROMPT)
    pre_kn = generate(model, tokenizer, KN_PROMPT)
    print(f"  ID: {pre_id.split('assistant')[-1].strip()[:120]}")
    print(f"  KN: {pre_kn.split('assistant')[-1].strip()[:120]}")

    # --- Phase 2: Compute identity direction ---
    print(f"\n--- Computing identity direction at L{best_layer} MLP down_proj ---")
    v_id = get_last_hidden(model, tokenizer, ID_PROMPT)
    v_kn = get_last_hidden(model, tokenizer, KN_PROMPT)
    v = v_id - v_kn
    v = v / v.norm().clamp_min(1e-8)

    # --- Phase 3: ROME edit on down_proj ---
    down_proj = model.model.layers[best_layer].mlp.down_proj
    W = down_proj.weight.data.clone()
    brain_before = tensor_sha256(down_proj.weight.data)

    # Rank-1: W' = W + λ * v @ (W.T @ v).T
    # v is in output space [d_model]. Strengthen projection onto v.
    lam = 5.0
    W_f32 = W.float()
    v_f32 = v.to(device=W.device, dtype=torch.float32)
    k = (W_f32.T @ v_f32)  # [intermediate_dim] — input direction
    k = k / k.norm().clamp_min(1e-8)
    W_new = W_f32 + lam * torch.outer(v_f32, k)
    down_proj.weight.data = W_new.to(dtype=W.dtype)

    brain_after = tensor_sha256(down_proj.weight.data)
    delta = (W_new - W_f32).norm().item()
    print(f"  L{best_layer} down_proj [{list(W.shape)}]: hash {brain_before[:16]}... -> {brain_after[:16]}...")
    print(f"  Delta norm: {delta:.2f}  λ={lam}")

    # --- Phase 4: Post-edit ---
    print(f"\n--- POST-EDIT ---")
    post_id = generate(model, tokenizer, ID_PROMPT)
    post_kn = generate(model, tokenizer, KN_PROMPT)
    print(f"  ID: {post_id.split('assistant')[-1].strip()[:120]}")
    print(f"  KN: {post_kn.split('assistant')[-1].strip()[:120]}")

    # --- Phase 5: Rollback ---
    down_proj.weight.data = W
    restored = tensor_sha256(down_proj.weight.data) == brain_before
    print(f"\n--- ROLLBACK {'OK' if restored else 'FAILED'} ---")

    # --- Verdict ---
    id_changed = pre_id != post_id
    kn_preserved = "Brasília" in post_kn or "Brasilia" in post_kn
    print(f"\n=== VERDICT ===")
    print(f"  Identity changed:    {id_changed}")
    print(f"  Knowledge preserved: {kn_preserved}")

    report = {
        "schema": "rome-instruct-identity-v1",
        "model": "SmolLM2-1.7B-Instruct",
        "layer_edited": best_layer,
        "lambda": lam,
        "brain_before": brain_before,
        "brain_after": brain_after,
        "brain_restored": restored,
        "delta_norm": delta,
        "pre_identity": pre_id,
        "post_identity": post_id,
        "pre_knowledge": pre_kn,
        "post_knowledge": post_kn,
        "identity_changed": id_changed,
        "knowledge_preserved": kn_preserved,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_json_write(report, OUTPUT / "rome-instruct-report.json")
    print(f"\nReport: {OUTPUT / 'rome-instruct-report.json'}")
    print(f"ROME_INSTRUCT_{'OK' if id_changed and kn_preserved else 'PARTIAL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
