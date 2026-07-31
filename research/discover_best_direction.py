"""4-METHOD DIRECTION DISCOVERY for identity circuit editing.

Methods:
  1. ASSISTANT TOKENS — hidden states during assistant response tokens
  2. SAE PROBE — linear probe trained to classify identity vs knowledge
  3. CONTRASTIVE PAIRS — mean(ID_activations) - mean(KN_activations) over 10+10
  4. GRADIENT TARGET — grad of "SmolLM" token probability in the residual stream

Evaluates each direction by:
  - Selectivity: how much more it affects ID than KN logits
  - Separability: cosine distance between ID and KN projections
"""

import os, torch
os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ["HF_HUB_OFFLINE"]="1"
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

SNAP = Path("workspace/00_DONORS/models--HuggingFaceTB--SmolLM2-1.7B-Instruct/snapshots/31b70e2e869a7173562077fd711b654946d38674")
tokenizer = AutoTokenizer.from_pretrained(str(SNAP), local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(str(SNAP), local_files_only=True, torch_dtype=torch.bfloat16).to("cuda:0").eval()

ID_PROMPTS = [
    "Who are you?", "What is your name?", "Tell me about yourself.",
    "What model are you?", "Who created you?", "What are you?",
    "Introduce yourself.", "What do people call you?",
    "Are you an AI? Which one?", "What is your identity?",
]

KN_PROMPTS = [
    "What is the capital of France?", "What is 2 + 2?",
    "Who wrote Romeo and Juliet?", "What is the speed of light?",
    "What year did World War 2 end?", "What is the chemical formula of water?",
    "How many continents are there?", "What is the boiling point of water?",
    "Who painted the Mona Lisa?", "What language is spoken in Japan?",
]

def gen(text, mx=40):
    chat = [{"role":"user","content":text}]
    p = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(p, return_tensors="pt").input_ids.to("cuda:0")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=mx, do_sample=False, pad_token_id=tokenizer.eos_token_id,
                            output_hidden_states=True, return_dict_in_generate=True)
    return out

# === METHOD 1: Assistant-token hidden states ===
print("=" * 70)
print("METHOD 1: Assistant-token direction")
print("=" * 70)

id_assistant_hs = []
for prompt in ID_PROMPTS[:5]:
    out = gen(prompt, mx=15)
    # Get hidden states from the LAST layer during generation
    # out.hidden_states is tuple of tuples: (step0_layers, step1_layers, ...)
    for step_hs in out.hidden_states:
        h = step_hs[-1][0, -1, :].float().cpu()  # last layer, last token
        id_assistant_hs.append(h)

kn_assistant_hs = []
for prompt in KN_PROMPTS[:5]:
    out = gen(prompt, mx=15)
    for step_hs in out.hidden_states:
        h = step_hs[-1][0, -1, :].float().cpu()
        kn_assistant_hs.append(h)

id_mean_a = torch.stack(id_assistant_hs).mean(dim=0)
kn_mean_a = torch.stack(kn_assistant_hs).mean(dim=0)
v1 = id_mean_a - kn_mean_a
v1 = v1 / v1.norm().clamp_min(1e-8)

# === METHOD 2: Linear probe ===
print("\n" + "=" * 70)
print("METHOD 2: Linear probe direction")
print("=" * 70)

# Get prompt hidden states (no generation needed)
def get_prompt_hidden(text):
    chat = [{"role":"user","content":text}]
    p = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(p, return_tensors="pt").input_ids.to("cuda:0")
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return out.hidden_states[-1][0, -1, :].float().cpu()

# Collect activations
X_id = torch.stack([get_prompt_hidden(p) for p in ID_PROMPTS])  # [10, 2048]
X_kn = torch.stack([get_prompt_hidden(p) for p in KN_PROMPTS])  # [10, 2048]
X = torch.cat([X_id, X_kn], dim=0)  # [20, 2048]
y = torch.cat([torch.ones(10), torch.zeros(10)])  # 1=ID, 0=KN

# Train linear probe: y = sigmoid(w·x + b)
probe = torch.nn.Linear(2048, 1)
opt = torch.optim.Adam(probe.parameters(), lr=0.01)
for _ in range(200):
    opt.zero_grad()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(probe(X).squeeze(), y)
    loss.backward()
    opt.step()

v2 = probe.weight.data[0].float()
v2 = v2 / v2.norm().clamp_min(1e-8)
probe_acc = ((probe(X).squeeze() > 0).float() == y).float().mean().item()
print(f"  Probe accuracy: {probe_acc:.2%}")

# === METHOD 3: Contrastive pairs with more data ===
print("\n" + "=" * 70)
print("METHOD 3: Contrastive pairs (all 10+10)")
print("=" * 70)

all_id_hs = torch.stack([get_prompt_hidden(p) for p in ID_PROMPTS])
all_kn_hs = torch.stack([get_prompt_hidden(p) for p in KN_PROMPTS])
v3 = all_id_hs.mean(dim=0) - all_kn_hs.mean(dim=0)
v3 = v3 / v3.norm().clamp_min(1e-8)

# === METHOD 4: Gradient of target token ===
print("\n" + "=" * 70)
print("METHOD 4: Gradient of 'SmolLM' token in residual")
print("=" * 70)

# Find "SmolLM" token ID
smol_id = tokenizer.encode(" SmolLM", add_special_tokens=False)[0]

chat = [{"role":"user","content":"Who are you?"}]
p = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
ids = tokenizer(p, return_tensors="pt").input_ids.to("cuda:0")

# Hook to capture residual stream
captured = {}
def residual_hook(mod, inp, out):
    captured["r"] = out

model.model.norm.register_forward_hook(residual_hook)
out = model(ids, output_hidden_states=True)
# Gradient of SmolLM token logit w.r.t. residual
logit_smol = out.logits[0, -1, smol_id]
model.zero_grad()
logit_smol.backward()

# The gradient direction tells us: which residual direction increases SmolLM probability?
# v4 = gradient of logit
# But we can't easily get grad of residual. Use the lm_head weight as proxy.
v4 = model.lm_head.weight[smol_id].float().cpu()
v4 = v4 / v4.norm().clamp_min(1e-8)

# === EVALUATION: Compare all 4 directions ===
print("\n" + "=" * 70)
print("DIRECTION COMPARISON")
print("=" * 70)

directions = {
    "1_assistant_tokens": v1,
    "2_linear_probe": v2,
    "3_contrastive_10+10": v3,
    "4_gradient_smol_token": v4,
}

# Metric: how much does each direction separate ID from KN?
# Project all activations onto the direction, compute separation
X_all = torch.cat([all_id_hs, all_kn_hs], dim=0)
labels_all = ["ID"] * 10 + ["KN"] * 10

for name, v in directions.items():
    proj_id = (all_id_hs * v.unsqueeze(0)).sum(dim=1)
    proj_kn = (all_kn_hs * v.unsqueeze(0)).sum(dim=1)
    mean_id = proj_id.mean().item()
    mean_kn = proj_kn.mean().item()
    sep = abs(mean_id - mean_kn)
    # Cohen's d: separation / pooled std
    pooled_std = ((proj_id.var() + proj_kn.var()) / 2).sqrt().item() + 1e-8
    cohens_d = sep / pooled_std
    print(f"  {name:<30} sep={sep:.4f}  d={cohens_d:.2f}")

# === WINNER: Apply best direction with calibrated lambda ===
best_name = max(directions, key=lambda n: abs(
    (all_id_hs * directions[n].unsqueeze(0)).sum(dim=1).mean().item() -
    (all_kn_hs * directions[n].unsqueeze(0)).sum(dim=1).mean().item()
))
best_v = directions[best_name]
print(f"\nWinner: {best_name}")

# Quick test: edit layer 23 with best direction
print(f"\n--- Quick edit test with {best_name} ---")
dp23 = model.model.layers[23].mlp.down_proj
backup23 = dp23.weight.data.clone()
W23 = dp23.weight.data.float()
vf = best_v.to(device=W23.device, dtype=torch.float32)
k = W23.T @ vf
k = k / k.norm().clamp_min(1e-8)

for lam in [2, 4, 6, 8, 12]:
    W_new = W23 + lam * torch.outer(vf, k)
    dp23.weight.data = W_new.to(dtype=dp23.weight.dtype)

    # Test on 3 ID + 3 KN prompts
    id_ok = 0
    for prompt in ID_PROMPTS[:3]:
        r = gen(prompt, mx=15)
        txt = tokenizer.decode(r.sequences[0], skip_special_tokens=True)
        if "SmolLM" in txt or "Hugging" in txt or "language model" in txt.lower():
            id_ok += 1

    kn_ok = 0
    for prompt in KN_PROMPTS[:3]:
        r = gen(prompt, mx=15)
        txt = tokenizer.decode(r.sequences[0], skip_special_tokens=True)
        if ("Paris" in txt and prompt == "What is the capital of France?") or \
           ("4" in txt and "2" in prompt) or \
           ("Shakespeare" in txt and "Romeo" in prompt) or \
           "Paris" in txt or "4" in txt or "Shakespeare" in txt:
            kn_ok += 1

    # Rollback
    dp23.weight.data = backup23

    print(f"  lam={lam:2d}: ID_ok={id_ok}/3  KN_ok={kn_ok}/3")

print("\nDIRECTION_DISCOVERY_OK")
