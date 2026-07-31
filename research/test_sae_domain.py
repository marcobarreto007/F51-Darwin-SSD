"""SAE domain separation test. Trains SAE on Python + Medicine activations
and measures whether features are domain-specific (what neurons cannot do)."""

import os, torch, time
os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ["HF_HUB_OFFLINE"]="1"
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F

SNAP = "workspace/00_DONORS/models--HuggingFaceTB--SmolLM2-1.7B/snapshots/effd688a12921b4cc83e3312b6feb579f70f9c71"
tokenizer = AutoTokenizer.from_pretrained(SNAP, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(SNAP, local_files_only=True, torch_dtype=torch.bfloat16).to("cuda:0").eval()

LAYER = 22
D_MODEL = 2048
N_FEATURES = 4096

PYTHON = [
    "def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x >= arr[0]])",
    "import torch; model = torch.nn.Linear(10, 5); x = torch.randn(3, 10); y = model(x)",
    "class BankAccount: def __init__(self, balance=0): self.balance = balance; def deposit(self, amount): self.balance += amount",
    "@app.route('/api/users', methods=['POST'])\ndef create_user():\n    data = request.get_json()\n    return jsonify({'id': 1})",
    "def binary_search(arr, target): low, high = 0, len(arr)-1; while low <= high: mid = (low+high)//2; return mid if arr[mid]==target else (low:=mid+1) if arr[mid]<target else (high:=mid-1)",
    "df = pd.DataFrame({'name': ['Alice','Bob'], 'score': [95,87]}); print(df.groupby('name').mean())",
    "async def fetch_data(url): async with aiohttp.ClientSession() as session: async with session.get(url) as resp: return await resp.json()",
    "try: result = 10 / 0\nexcept ZeroDivisionError as e: print(f'Error: {e}')",
]

MEDICINE = [
    "The patient presented with acute myocardial infarction characterized by ST-segment elevation on ECG and elevated troponin levels.",
    "Administer 5mg of diazepam intravenously every 8 hours for status epilepticus. Monitor respiratory depression.",
    "Differential diagnosis includes pulmonary embolism, aortic dissection, and tension pneumothorax based on clinical presentation.",
    "The mechanism of action of metformin involves AMPK activation, reducing hepatic gluconeogenesis and increasing peripheral insulin sensitivity.",
    "Histological examination revealed adenocarcinoma with moderate differentiation and lymphovascular invasion at the resection margin.",
    "Cerebrospinal fluid analysis showed elevated protein, decreased glucose, and lymphocytic pleocytosis consistent with tuberculous meningitis.",
    "Pharmacokinetics of vancomycin requires therapeutic drug monitoring with trough levels maintained between 15-20 mcg/mL for severe infections.",
    "The pathophysiology of sepsis involves a dysregulated host response to infection leading to organ dysfunction mediated by cytokine storm.",
]

print("=" * 60)
print("SAE DOMAIN SEPARATION TEST")
print("=" * 60)
print(f"Layer {LAYER}, {N_FEATURES} features, {len(PYTHON)} Python + {len(MEDICINE)} Medicine texts")

# Collect activations per domain
cap = {}
def hook(mod, inp, out): cap["h"] = out
model.model.layers[LAYER].register_forward_hook(hook)

acts_py, acts_med = [], []
with torch.no_grad():
    for text in PYTHON:
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
        model(ids)
        acts_py.append(cap["h"][0, :, :].float().cpu())
    for text in MEDICINE:
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
        model(ids)
        acts_med.append(cap["h"][0, :, :].float().cpu())

acts_py = torch.cat(acts_py, dim=0)
acts_med = torch.cat(acts_med, dim=0)
all_acts = torch.cat([acts_py, acts_med], dim=0)
n_py, n_med = acts_py.shape[0], acts_med.shape[0]
print(f"Tokens: {n_py} Python + {n_med} Medicine = {all_acts.shape[0]} total")
print(f"Activation norm: {all_acts.norm(dim=1).mean():.1f}")

# Normalize activations (SAEs work better with unit-norm inputs)
all_acts_norm = F.normalize(all_acts, dim=1)

# Train SAE
class SparseAutoencoder(nn.Module):
    def __init__(self, d_in, n_feat, l1_coef=0.05):
        super().__init__()
        self.encoder = nn.Linear(d_in, n_feat, bias=True)
        self.decoder = nn.Linear(n_feat, d_in, bias=False)
        nn.init.orthogonal_(self.decoder.weight)
        self.l1_coef = l1_coef
    
    def forward(self, x):
        encoded = F.relu(self.encoder(x))
        recon = self.decoder(encoded)
        recon_loss = ((recon - x) ** 2).mean()
        l1 = encoded.mean()
        return encoded, recon_loss + self.l1_coef * l1, {"recon": float(recon_loss), "l1": float(l1), "active": (encoded > 0).sum(dim=1).float().mean().item()}

print(f"\nTraining SAE ({N_FEATURES} features)...")
sae = SparseAutoencoder(D_MODEL, N_FEATURES).to("cuda:0")
opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
batch = all_acts_norm.to("cuda:0")

t0 = time.perf_counter()
for step in range(300):
    opt.zero_grad()
    _, loss, stats = sae(batch)
    loss.backward()
    opt.step()
    if step % 50 == 0:
        print(f"  step {step:3d}  recon={stats['recon']:.4f}  l1={stats['l1']:.4f}  active={stats['active']:.0f}/{N_FEATURES}")

print(f"  Trained in {time.perf_counter()-t0:.1f}s")

# Get feature activations for each domain
with torch.no_grad():
    enc_py, _, _ = sae(acts_py.norm(dim=1, keepdim=True).to('cuda:0') * F.normalize(acts_py, dim=1).to('cuda:0'))  # preserve norm info
    enc_med, _, _ = sae(acts_med.norm(dim=1, keepdim=True).to('cuda:0') * F.normalize(acts_med, dim=1).to('cuda:0'))

# Per-feature domain preference
mean_py = enc_py.mean(dim=0)  # [N_FEATURES]
mean_med = enc_med.mean(dim=0)

# Find top Python-specific and Medicine-specific features
diff = mean_py - mean_med  # positive = Python, negative = Medicine
top_py = diff.argsort(descending=True)[:20].tolist()
top_med = diff.argsort()[:20].tolist()

print(f"\n=== DOMAIN-SPECIFIC FEATURES ===")
print(f"Top Python features (largest mean_py - mean_med):")
for i, f in enumerate(top_py[:10]):
    print(f"  F{f:04d}: py={mean_py[f].item():.4f}  med={mean_med[f].item():.4f}  diff={diff[f].item():.4f}")

print(f"\nTop Medicine features:")
for i, f in enumerate(top_med[:10]):
    print(f"  F{f:04d}: py={mean_py[f].item():.4f}  med={mean_med[f].item():.4f}  diff={diff[f].item():.4f}")

# Separation metric: cosine distance between domain centroids in feature space
centroid_py = mean_py / (mean_py.norm() + 1e-8)
centroid_med = mean_med / (mean_med.norm() + 1e-8)
cos_sim = (centroid_py * centroid_med).sum().item()
print(f"\nCosine similarity between domain centroids (feature space): {cos_sim:.4f}")
print(f"Cosine distance: {1-cos_sim:.4f}")

# Compare: what's the cosine similarity in raw neuron space?
centroid_py_raw = acts_py.mean(dim=0)
centroid_med_raw = acts_med.mean(dim=0)
cos_raw = (F.normalize(centroid_py_raw, dim=0) * F.normalize(centroid_med_raw, dim=0)).sum().item()
print(f"Cosine similarity in RAW neuron space: {cos_raw:.4f}")
print(f"Cosine distance in raw space: {1-cos_raw:.4f}")

feature_separation = 1 - cos_sim
neuron_separation = 1 - cos_raw
ratio = feature_separation / (neuron_separation + 1e-8)
print(f"\nSeparation ratio (SAE features / raw neurons): {ratio:.2f}x")
if ratio > 1.5:
    print("SAE features ARE more separable than raw neurons.")
elif ratio > 1.1:
    print("SAE features are SLIGHTLY more separable.")
else:
    print("SAE features are NOT more separable than raw neurons.")

print(f"\nSAE_DOMAIN_TEST_OK")

