"""Layer transplant: Llama-3.2-3B FFN -> SmolLM2-1.7B via SVD projection.

Llama-3.2-3B: d_model=3072, intermediate=8192, 28 layers, SwiGLU
SmolLM2-1.7B: d_model=2048, intermediate=8192, 24 layers, SwiGLU

Strategy: SVD-adapt Llama down_proj [3072, 8192] -> [2048, 8192]
by keeping the top-2048 singular vectors that capture the most knowledge.
"""

import os, torch, time
os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ["HF_HUB_OFFLINE"]="1"
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

SNAP_SMOL = "workspace/00_DONORS/models--HuggingFaceTB--SmolLM2-1.7B/snapshots/effd688a12921b4cc83e3312b6feb579f70f9c71"
SNAP_LLAMA = "workspace/00_DONORS/models--meta-llama--Llama-3.2-3B"

# Find llama snapshot
llama_snaps = list(Path(SNAP_LLAMA, "snapshots").glob("*"))
if not llama_snaps:
    print("Llama model not fully downloaded - checking...")
    for f in Path(SNAP_LLAMA, "snapshots").iterdir():
        print(f"  {f.name}")
    exit(1)
SNAP_LLAMA_SNAP = str(llama_snaps[0])
print(f"Llama snapshot: {llama_snaps[0].name}")

tokenizer = AutoTokenizer.from_pretrained(SNAP_SMOL, local_files_only=True)
print("Loading SmolLM2-1.7B (recipient)...")
smol = AutoModelForCausalLM.from_pretrained(SNAP_SMOL, local_files_only=True, torch_dtype=torch.bfloat16).to("cuda:0").eval()
print("Loading Llama-3.2-3B (donor)...")
llama = AutoModelForCausalLM.from_pretrained(SNAP_LLAMA_SNAP, local_files_only=True, torch_dtype=torch.bfloat16).to("cuda:0").eval()

def gen(model, prompt, mx=40):
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=mx, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)

CODE_PROBES = [
    "def fibonacci(n):",
    "def quicksort(arr):",
    "Write a Python function to check if a number is prime",
]
GEN_PROBES = [
    "The capital of France is",
    "Water boils at",
    "Shakespeare wrote",
]

# Baseline
print("\n=== BASELINE (SmolLM2-1.7B base) ===")
for p in CODE_PROBES + GEN_PROBES:
    print(f"  {p[:45]:<45} -> {gen(smol, p)[:80]}")

print("\n=== DONOR (Llama-3.2-3B) — reference ===")
for p in CODE_PROBES:
    print(f"  {p[:45]:<45} -> {gen(llama, p)[:80]}")

# SVD-adapt Llama FFN -> SmolLM2 dimensions
print("\n=== SVD-ADAPTING Llama down_proj [3072,8192] -> [2048,8192] ===")

llama_layer = 20  # late layer with strong reasoning
llama_w = llama.model.layers[llama_layer].mlp.down_proj.weight.data.float().cpu()

print(f"  Llama L{llama_layer} down_proj: {list(llama_w.shape)}")
U, S, Vt = torch.linalg.svd(llama_w, full_matrices=False)
print(f"  SVD: U={list(U.shape)} S={list(S.shape)} Vt={list(Vt.shape)}")
print(f"  Top-5 singular values: {S[:5].tolist()}")
print(f"  Energy in top 2048/3072: {S[:2048].sum().item()/S.sum().item():.1%}")

# Project: keep top 2048 singular vectors = 99.2% of energy typically
W_adapted = (U[:, :2048] * S[:2048]) @ Vt[:2048, :]  # [3072, 2048] @ [2048, 8192] -> [3072, 8192]
# Wait, this is wrong. U[:,:2048] is [3072, 2048]. (U[:,:2048] * S[:2048]) is [3072, 2048].
# @ Vt[:2048,:] which is [2048, 8192] -> result [3072, 8192]. That's the same shape!
# I need: take the reconstruction, then extract first 2048 rows (output dims)
# Actually: W_llama[3072,8192] -> SVD -> keep top 2048 -> W_smol[2048,8192]
# The output space of Llama is 3072-dim. I want 2048-dim output.
# Take the 2048 most important output directions (rows of U) and project:
# W_adapted = U[:, :2048].T @ W_llama  -> [2048, 8192]
W_adapted = U[:, :2048].T @ llama_w  # [2048, 3072] @ [3072, 8192] = [2048, 8192]
print(f"  Adapted down_proj: {list(W_adapted.shape)}")
print(f"  Norm ratio adapted/llama: {W_adapted.norm().item()/llama_w.norm().item():.2f}")

# Normalize to match SmolLM2's typical weight norm
smol_w = smol.model.layers[20].mlp.down_proj.weight.data.float().cpu()
target_norm = smol_w.norm()
W_adapted = W_adapted * (target_norm / W_adapted.norm().clamp_min(1e-8))
print(f"  Normalized to SmolLM2 norm: {W_adapted.norm().item():.0f} (target: {target_norm.item():.0f})")

# Also adapt gate_proj and up_proj [3072,2048] -> [8192,2048]
# Llama: gate_proj [8192,3072], SmolLM2: [8192,2048]
# Take first 2048 columns of Llama gate/up (matching input dim)
for proj_name in ["gate_proj", "up_proj"]:
    llama_proj = getattr(llama.model.layers[llama_layer].mlp, proj_name).weight.data.float().cpu()
    # Llama: [8192, 3072] -> SmolLM2: [8192, 2048] -> take first 2048 columns
    adapted_proj = llama_proj[:, :2048].clone()
    smol_proj = getattr(smol.model.layers[20].mlp, proj_name).weight.data.float().cpu()
    adapted_proj = adapted_proj * (smol_proj.norm() / adapted_proj.norm().clamp_min(1e-8))
    print(f"  {proj_name}: {list(llama_proj.shape)} -> {list(adapted_proj.shape)}")

# --- TRANSPLANT ---
print("\n=== TRANSPLANTING adapted Llama FFN into SmolLM2 layers 18-22 ===")
backups = {}
for li in [18, 19, 20, 21, 22]:
    backups[li] = {
        "gate": smol.model.layers[li].mlp.gate_proj.weight.data.clone(),
        "up": smol.model.layers[li].mlp.up_proj.weight.data.clone(),
        "down": smol.model.layers[li].mlp.down_proj.weight.data.clone(),
    }
    # Adapt Llama weights for each layer
    llama_w = llama.model.layers[min(li + 4, 27)].mlp.down_proj.weight.data.float().cpu()
    U, S, Vt = torch.linalg.svd(llama_w, full_matrices=False)
    W_down = (U[:, :2048].T @ llama_w).to(dtype=torch.bfloat16).to("cuda:0")
    W_down = W_down * (smol.model.layers[li].mlp.down_proj.weight.data.float().norm() / W_down.float().norm().clamp_min(1e-8))
    smol.model.layers[li].mlp.down_proj.weight.data = W_down

    llama_gate = llama.model.layers[min(li + 4, 27)].mlp.gate_proj.weight.data.float().cpu()
    W_gate = llama_gate[:, :2048].to(dtype=torch.bfloat16).to("cuda:0")
    W_gate = W_gate * (smol.model.layers[li].mlp.gate_proj.weight.data.float().norm() / W_gate.float().norm().clamp_min(1e-8))
    smol.model.layers[li].mlp.gate_proj.weight.data = W_gate

    llama_up = llama.model.layers[min(li + 4, 27)].mlp.up_proj.weight.data.float().cpu()
    W_up = llama_up[:, :2048].to(dtype=torch.bfloat16).to("cuda:0")
    W_up = W_up * (smol.model.layers[li].mlp.up_proj.weight.data.float().norm() / W_up.float().norm().clamp_min(1e-8))
    smol.model.layers[li].mlp.up_proj.weight.data = W_up
    print(f"  L{li}: transplanted from Llama L{min(li+4,27)}")

# --- TEST ---
print("\n=== AFTER TRANSPLANT ===")
for p in CODE_PROBES:
    r = gen(smol, p)
    print(f"  [CODE] {p[:45]:<45} -> {r[:100]}")
for p in GEN_PROBES:
    r = gen(smol, p)
    print(f"  [GEN]  {p[:45]:<45} -> {r[:100]}")

# Rollback
for li in [18, 19, 20, 21, 22]:
    for k, v in backups[li].items():
        getattr(smol.model.layers[li].mlp, k + "_proj").weight.data = v

print("\n=== ROLLBACK: Model restored ===")
print(f"  Capital of France: {gen(smol, 'The capital of France is')[:60]}")
print("\nLAYER_TRANSPLANT_OK")
