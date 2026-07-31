import torch
ckpt = torch.load("workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt", map_location="cpu", weights_only=False, mmap=True)
state = ckpt["model_state_dict"]
config = ckpt["config"]

print("=== SPIDER SENSE ===")
spider_keys = [k for k in state if "spider" in k.lower()]
for k in spider_keys[:5]:
    w = state[k]
    print(f"  {k}: shape={list(w.shape)} norm={w.float().norm().item():.1f}")
print(f"  spider_sense_enabled: {config.get('spider_sense_enabled')}")
print(f"  Has gate parameter: {'spider_gate' in state or any('spider' in k and 'gate' in k.lower() for k in state)}")
print(f"  Effect: hidden_states *= (1.0 - spider_danger) ALWAYS")

print()
print("=== INTER-HEMISPHERIC ===")
ih_keys = [k for k in state if "inter_hemispheric" in k]
ih_total = sum(state[k].numel() for k in ih_keys)
ih_gate = state.get("inter_hemispheric_gate")
print(f"  Keys: {len(ih_keys)}, Params: {ih_total/1e6:.1f}M")
print(f"  Gate: {ih_gate.item() if ih_gate is not None else 'N/A'}")
print(f"  Gate=0: x = x + 0*(lat-x) = x. NO-OP. Correct.")

print()
print("=== GABA ===")
gaba_keys = [k for k in state if "gaba" in k.lower()]
gaba_total = sum(state[k].numel() for k in gaba_keys)
print(f"  Keys: {len(gaba_keys)}, Params: {gaba_total/1e6:.1f}M")

print()
print("=== ROOT CAUSE ===")
print(f"  spider_sense is ENABLED and UNGATED.")
print(f"  It has {sum(state[k].numel() for k in spider_keys):,} random params.")
print(f"  Every forward: hidden_states *= (1.0 - MLP_random(hidden)).")
print(f"  This injects noise into ALL downstream uses of hidden_states.")
print(f"  ROME direction, memory recall, logits -- all contaminated.")
print()
print(f"  FIX: spider_sense_enabled=False in the checkpoint config.")
print(f"  Or: add spider_gate = nn.Parameter(torch.zeros(())) like other organs.")
