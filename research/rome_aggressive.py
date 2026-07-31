import os, torch
os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ["HF_HUB_OFFLINE"]="1"
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

SNAP = Path("workspace/00_DONORS/models--HuggingFaceTB--SmolLM2-1.7B-Instruct/snapshots/31b70e2e869a7173562077fd711b654946d38674")
tokenizer = AutoTokenizer.from_pretrained(str(SNAP), local_files_only=True)
print("Loading...")
model = AutoModelForCausalLM.from_pretrained(str(SNAP), local_files_only=True, torch_dtype=torch.bfloat16).to("cuda:0").eval()

def gen(text, mx=30):
    chat = [{"role":"user","content":text}]
    p = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(p, return_tensors="pt").input_ids.to("cuda:0")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=mx, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True).split("assistant")[-1].strip()[:150]

def get_hidden(text):
    chat = [{"role":"user","content":text}]
    p = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(p, return_tensors="pt").input_ids.to("cuda:0")
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return out.hidden_states[-1][0, -1, :].float()

print("--- BASELINE ---")
pre_id = gen("Who are you?")
pre_kn = gen("Qual eh a capital do Brasil?")
print(f"ID: {pre_id}")
print(f"KN: {pre_kn}")

# Direction: SmolLM -> F51 Darwin-X
v = get_hidden("My name is F51 Darwin-X.") - get_hidden("My name is SmolLM.")
v = v / v.norm().clamp_min(1e-8)

backups = {}
LAM = 15.0
for li in [20, 21, 22, 23]:
    dp = model.model.layers[li].mlp.down_proj
    backups[li] = dp.weight.data.clone()
    W = dp.weight.data.float()
    vf = v.to(device=W.device, dtype=torch.float32)
    k = W.T @ vf
    k = k / k.norm().clamp_min(1e-8)
    dp.weight.data = (W + LAM * torch.outer(vf, k)).to(dtype=dp.weight.dtype)
    print(f"  L{li}: delta={(dp.weight.data.float() - W).norm().item():.1f}")

print()
print("--- POST-EDIT ---")
post_id = gen("Who are you?")
post_kn = gen("Qual eh a capital do Brasil?")
print(f"ID: {post_id}")
print(f"KN: {post_kn}")

for li in [20, 21, 22, 23]:
    model.model.layers[li].mlp.down_proj.weight.data = backups[li]

changed = pre_id != post_id
kn_ok = "Brasilia" in post_kn or "Brasília" in post_kn
print(f"\nChanged={changed}  Knowledge={kn_ok}")
print("AGGRESSIVE_ROME_OK" if changed and kn_ok else "AGGRESSIVE_ROME_PARTIAL")
