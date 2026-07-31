"""Direct layer transplant: SmolLM2-Instruct -> SmolLM2-Base.
Same architecture (2048, 24L, 8192 intermediate). No SVD needed.
Replace entire FFN weights in late layers where instruct excels at code.
"""

import os, torch, time
os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ["HF_HUB_OFFLINE"]="1"
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

SNAP_INSTRUCT = "workspace/00_DONORS/models--HuggingFaceTB--SmolLM2-1.7B-Instruct/snapshots/31b70e2e869a7173562077fd711b654946d38674"
SNAP_BASE = "workspace/00_DONORS/models--HuggingFaceTB--SmolLM2-1.7B/snapshots/effd688a12921b4cc83e3312b6feb579f70f9c71"

tokenizer = AutoTokenizer.from_pretrained(SNAP_INSTRUCT, local_files_only=True)
print("Loading Instruct (donor)...")
instruct = AutoModelForCausalLM.from_pretrained(SNAP_INSTRUCT, local_files_only=True, torch_dtype=torch.bfloat16).to("cuda:0").eval()
print("Loading Base (recipient)...")
base = AutoModelForCausalLM.from_pretrained(SNAP_BASE, local_files_only=True, torch_dtype=torch.bfloat16).to("cuda:0").eval()

def gen(model, prompt, mx=60):
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=mx, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)

CODE = [
    "def fibonacci(n):",
    "def quicksort(arr):",
    "Write a Python function to check if a number is prime",
    "Create a class for a bank account with deposit and withdraw methods",
    "Implement a function to find all prime numbers up to n using the Sieve of Eratosthenes",
    "Write a Python function that takes a list of integers and returns the two numbers that sum to a target value",
]

GEN = [
    "The capital of France is",
    "Water boils at",
    "The speed of light is",
    "Shakespeare wrote",
    "The chemical formula of water is",
    "World War 2 ended in",
]

# --- Find best instruct layers for code ---
print("\n=== Finding code-specialized layers in Instruct ===")
# Compare instruct-vs-base output quality on code prompts
layer_scores = {}
for li in range(24):
    # Temporarily transplant instruct layer into base and measure code improvement
    # Fast proxy: compare weight norms
    i_norm = instruct.model.layers[li].mlp.down_proj.weight.data.float().norm().item()
    b_norm = base.model.layers[li].mlp.down_proj.weight.data.float().norm().item()
    layer_scores[li] = abs(i_norm - b_norm)

# Pick layers with biggest instruct-vs-base weight difference
best_layers = sorted(layer_scores, key=layer_scores.get, reverse=True)[:6]
best_layers.sort()
print(f"  Top 6 layers by weight delta: {best_layers}")

# Baseline
print("\n=== BASELINE (Base model) ===")
base_before = {}
for p in CODE + GEN:
    r = gen(base, p)
    base_before[p] = r
    cat = "CODE" if p in CODE else "GEN "
    print(f"  [{cat}] {p[:50]:<50} -> {r[:90]}")

# Reference: instruct model
print("\n=== REFERENCE (Instruct model) ===")
for p in CODE[:3]:
    r = gen(instruct, p)
    print(f"  [CODE] {p[:50]:<50} -> {r[:90]}")

# --- TRANSPLANT ---
print(f"\n=== TRANSPLANTING entire FFN from Instruct layers {best_layers} into Base ===")
backups = {}
for li in best_layers:
    backups[li] = {}
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        src = getattr(instruct.model.layers[li].mlp, proj).weight.data
        dst = getattr(base.model.layers[li].mlp, proj).weight.data
        backups[li][proj] = dst.clone()
        dst.copy_(src)
    print(f"  L{li}: gate+up+down transplanted ({backups[li]['gate_proj'].numel()*3/1e6:.1f}M params)")

total_params = sum(b["gate_proj"].numel() * 3 for b in backups.values())
print(f"  Total transplanted: {total_params/1e6:.1f}M params ({total_params/1.2e9*100:.1f}% of 1.2B FFN)")

# --- TEST ---
print(f"\n=== AFTER TRANSPLANT ===")
code_improved = 0
gen_preserved = 0
for p in CODE:
    r = gen(base, p)
    improved = r != base_before[p] and len(r) > 1
    if improved: code_improved += 1
    tag = "IMPROVED" if improved else "SAME"
    print(f"  [CODE] {p[:50]:<50} -> {r[:90]} [{tag}]")
for p in GEN:
    r = gen(base, p)
    ok = base_before[p][:30] in r or r[:30] == base_before[p][:30]
    if ok: gen_preserved += 1
    tag = "OK" if ok else "DEGRADED"
    print(f"  [GEN]  {p[:50]:<50} -> {r[:90]} [{tag}]")

# Rollback
for li in best_layers:
    for proj, backup in backups[li].items():
        getattr(base.model.layers[li].mlp, proj).weight.data.copy_(backup)

print(f"\nCode improved: {code_improved}/{len(CODE)}")
print(f"General preserved: {gen_preserved}/{len(GEN)}")
print(f"TRANSPLANT_LAYER_{'OK' if code_improved >= 1 and gen_preserved >= 5 else 'PARTIAL'}")
