"""Cross-checkpoint domain transplant: SmolLM2-Instruct -> SmolLM2-Base.

Finds FFN neurons where the instruct model differs most from the base,
transplants those neurons, and verifies: target improves, rest unchanged.
"""

import os, torch, time
os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ["HF_HUB_OFFLINE"]="1"
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

SNAP_INSTRUCT = Path("workspace/00_DONORS/models--HuggingFaceTB--SmolLM2-1.7B-Instruct/snapshots/31b70e2e869a7173562077fd711b654946d38674")
SNAP_BASE = Path("workspace/00_DONORS/models--HuggingFaceTB--SmolLM2-1.7B/snapshots/effd688a12921b4cc83e3312b6feb579f70f9c71")

tokenizer = AutoTokenizer.from_pretrained(str(SNAP_INSTRUCT), local_files_only=True)
print("Loading donor (Instruct)...")
donor = AutoModelForCausalLM.from_pretrained(str(SNAP_INSTRUCT), local_files_only=True, torch_dtype=torch.bfloat16).to("cuda:0").eval()
print("Loading recipient (Base)...")
recipient = AutoModelForCausalLM.from_pretrained(str(SNAP_BASE), local_files_only=True, torch_dtype=torch.bfloat16).to("cuda:0").eval()

def gen(model, prompt, mx=50):
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=mx, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# Domain probes
CODE_PROBES = [
    "def fibonacci(n):",
    "def quicksort(arr):",
    "def binary_search(arr, target):",
    "Write a function to check if a number is prime",
    "Create a class for a bank account with deposit and withdraw methods",
]
GENERAL_PROBES = [
    "The capital of France is",
    "Water boils at",
    "The speed of light is",
    "Shakespeare wrote",
]

print("\n" + "=" * 70)
print("BASELINE: Base model (recipient) — BEFORE transplant")
print("=" * 70)
base_code_before = {}
for p in CODE_PROBES:
    r = gen(recipient, p)
    base_code_before[p] = r
    print(f"  [CODE] {p[:45]:<45} -> {r[:80]}")
base_gen_before = {}
for p in GENERAL_PROBES:
    r = gen(recipient, p)
    base_gen_before[p] = r
    print(f"  [GEN]  {p[:45]:<45} -> {r[:80]}")

print("\n" + "=" * 70)
print("DONOR: Instruct model — reference output")
print("=" * 70)
for p in CODE_PROBES[:3]:
    r = gen(donor, p)
    print(f"  [CODE] {p[:45]:<45} -> {r[:80]}")

# Find layers where donor weights differ most from recipient
print("\n" + "=" * 70)
print("Finding neurons with highest donor-vs-recipient weight delta")
print("=" * 70)

all_diffs = []
for li in range(24):
    d_w = donor.model.layers[li].mlp.down_proj.weight.data.float()
    r_w = recipient.model.layers[li].mlp.down_proj.weight.data.float()
    # Per-channel (neuron) L2 difference
    diff = (d_w - r_w).pow(2).sum(dim=0).sqrt()  # [8192] — L2 per column
    for ch in range(8192):
        all_diffs.append((li, ch, float(diff[ch])))

all_diffs.sort(key=lambda x: x[2], reverse=True)
print(f"  Total neurons compared: {len(all_diffs)}")
print(f"  Top 5 deltas: {[(f'L{d[0]}_ch{d[1]}', f'{d[2]:.2f}') for d in all_diffs[:5]]}")

# Transplant top K neurons from donor to recipient (by L2 weight delta)
K = 256  # number of neurons to transplant (0.13% of 196,608)
print(f"\nTransplanting top {K} neurons (by weight delta)...")

backups = {}
transplanted = 0
for li, ch, delta in all_diffs[:K]:
    if li not in backups:
        backups[li] = {}
    if ch not in backups[li]:
        backups[li][ch] = recipient.model.layers[li].mlp.down_proj.weight.data[:, ch].clone()
    
    # Transplant: copy the ENTIRE down_proj column from donor
    recipient.model.layers[li].mlp.down_proj.weight.data[:, ch] = (
        donor.model.layers[li].mlp.down_proj.weight.data[:, ch].clone()
    )
    transplanted += 1

print(f"  Transplanted: {transplanted} neurons ({transplanted/196608*100:.2f}% of total)")
print(f"  Layers affected: {sorted(backups.keys())}")

print("\n" + "=" * 70)
print("AFTER TRANSPLANT: Base model — code should improve")
print("=" * 70)
code_improved = 0
gen_preserved = 0
for p in CODE_PROBES:
    r = gen(recipient, p)
    improved = r != base_code_before[p] and len(r) > len(base_code_before[p])
    if improved: code_improved += 1
    tag = "CHANGED" if r != base_code_before[p] else "SAME"
    print(f"  [CODE] {p[:45]:<45} -> {r[:80]} [{tag}]")
for p in GENERAL_PROBES:
    r = gen(recipient, p)
    preserved = r[:40] == base_gen_before[p][:40]
    if preserved: gen_preserved += 1
    tag = "OK" if preserved else "DEGRADED"
    print(f"  [GEN]  {p[:45]:<45} -> {r[:80]} [{tag}]")

# Rollback
for li, channels in backups.items():
    for ch, backup in channels.items():
        recipient.model.layers[li].mlp.down_proj.weight.data[:, ch] = backup

print(f"\nCode changed: {code_improved}/{len(CODE_PROBES)}")
print(f"General preserved: {gen_preserved}/{len(GENERAL_PROBES)}")
print(f"TRANSPLANT_{'OK' if code_improved >= 2 and gen_preserved >= 3 else 'PARTIAL'}")
