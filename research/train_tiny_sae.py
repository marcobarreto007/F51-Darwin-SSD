import os, torch
os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ["HF_HUB_OFFLINE"]="1"
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

snap = Path("workspace/00_DONORS/models--HuggingFaceTB--SmolLM2-1.7B-Instruct/snapshots/31b70e2e869a7173562077fd711b654946d38674")
tokenizer = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(str(snap), local_files_only=True, torch_dtype=torch.bfloat16).to("cuda:0").eval()

LAYER = 22
texts = [
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius.",
    "The quick brown fox jumps over the lazy dog.",
    "She said that she would come, but she did not.",
    "# Python function for Fibonacci",
    "def fibonacci(n): return n if n < 2 else fib(n-1) + fib(n-2)",
    "The patient presented with acute myocardial infarction.",
    "Administer 5mg of diazepam intravenously every 8 hours.",
]

acts = []
cap = {}
def hook(mod, inp, out):
    cap["h"] = out
model.model.layers[LAYER].register_forward_hook(hook)
with torch.no_grad():
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
        model(ids)
        acts.append(cap["h"][0, :, :].float().cpu())

all_acts = torch.cat(acts, dim=0)
print(f"Collected {all_acts.shape[0]} token activations from layer {LAYER}")

import torch.nn as nn

class TinySAE(nn.Module):
    def __init__(self, d_in, n_feat):
        super().__init__()
        self.encoder = nn.Linear(d_in, n_feat, bias=True)
        self.decoder = nn.Linear(n_feat, d_in, bias=False)
        nn.init.orthogonal_(self.decoder.weight)
    
    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        recon = self.decoder(encoded)
        l1 = encoded.mean()
        loss = ((recon - x) ** 2).mean() + 0.05 * l1
        return encoded, loss

sae = TinySAE(2048, 512).to("cuda:0")
opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
batch = all_acts[:2000].to("cuda:0")
print(f"Training SAE (512 features) on {batch.shape[0]} tokens...")
for step in range(160):
    opt.zero_grad()
    _, loss = sae(batch)
    loss.backward()
    opt.step()
    if step % 40 == 0:
        print(f"  step {step:3d}  loss={loss.item():.4f}")

with torch.no_grad():
    enc, _ = sae(torch.randn(2, 2048).to("cuda:0"))
print(f"Features: {list(enc.shape)}  active={(enc > 0).sum(dim=1).tolist()}")
print("SAE_TRAINED_OK")
