"""Micro-inference test — loads checkpoint on CPU, tests a few prompts."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import yaml

from f51_darwin.darwin_x import DarwinXConfig, DarwinXModel
from f51_darwin.turbo_tokenizer import TurboTokenizer

CHECKPOINT = ROOT / "workspace/03_CHECKPOINTS_100M_FULL_V9/organism_cycle_001.pt"
CONFIG = ROOT / "configs/darwin_x_100m.yaml"
TOKENIZER_DIR = ROOT / "workspace/tokenizer/f51_bpe_80k"

PROMPTS = [
    "Olá, eu sou o Marco.",
    "O sentido da vida é",
    "A inteligência artificial pode",
    "O Brasil é conhecido por",
    "Once upon a time",
    "O código secreto é 42.",
]

@torch.no_grad()
def generate_simple(model, tokenizer, prompt, max_tokens=60, temperature=0.7, top_p=0.9):
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_id
    ids = tokenizer.encode(prompt)
    context = torch.tensor([ids[-model.config.context_length:]], dtype=torch.long, device=device)
    generated = []
    finish = "max_length"

    for _ in range(max_tokens):
        output = model.forward(context, heartbeat=False)
        logits = output.logits[:, -1, :] / max(temperature, 1e-8)

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            cutoff = cumulative > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            logits_flat = logits.clone()
            logits_flat.scatter_(-1, sorted_indices, sorted_logits.masked_fill(cutoff, float("-inf")))
        else:
            logits_flat = logits

        probs = F.softmax(logits_flat, dim=-1)
        next_token = int(torch.multinomial(probs, num_samples=1).item())

        if next_token == eos_id:
            finish = "eos"
            break

        generated.append(next_token)
        context = torch.cat([context, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)

    raw_text = tokenizer.decode(generated)
    # Byte-level BPE: <bXX> tokens → actual bytes for readable text
    import re
    def bytes_from_tokens(t):
        out = []
        i = 0
        for m in re.finditer(r'<b([0-9a-fA-F]{2})>', t):
            out.append(t[i:m.start()])  # text before this token
            out.append(chr(int(m.group(1), 16)))
            i = m.end()
        out.append(t[i:])
        return ''.join(out)
    clean = bytes_from_tokens(raw_text)
    return raw_text, clean, generated, finish


def main():
    print("=" * 60)
    print("MICRO-INFERÊNCIA — F51 Darwin-X 100M")
    print("=" * 60)

    # Load config
    cfg_dict = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config = DarwinXConfig.from_mapping(cfg_dict)
    print(f"\nConfig: {config.model_name} | d={config.d_model} | layers={config.n_layers}")
    print(f"Vocab: {config.vocab_size} | Context: {config.context_length}")

    # Load tokenizer
    print(f"\nCarregando tokenizer: {TOKENIZER_DIR}")
    tokenizer = TurboTokenizer.load(TOKENIZER_DIR)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  EOS id: {tokenizer.eos_id} | BOS id: {tokenizer.bos_id}")

    # Load model on CPU
    print(f"\nCarregando checkpoint: {CHECKPOINT}")
    print(f"  Tamanho: {CHECKPOINT.stat().st_size / 1e9:.2f} GB")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False, mmap=True)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in payload["model_state_dict"].items()}
    print(f"  Parâmetros no checkpoint: {len(state_dict)}")

    model = DarwinXModel(config)
    model.eval()

    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing_weights = [k for k in missing if "heartbeat" not in k and "inter_hemispheric" not in k]
    if missing_weights:
        print(f"  Missing (non-organ): {len(missing_weights)}")
    unexpected_weights = [k for k in unexpected if "_orig_mod" not in k]
    if unexpected_weights:
        print(f"  Unexpected: {len(unexpected_weights)}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params / 1e6:.1f}M")

    # Run inference
    print("\n" + "-" * 60)
    print("GERAÇÃO")
    print("-" * 60)

    for prompt in PROMPTS:
        try:
            raw_text, clean_text, token_ids, finish = generate_simple(model, tokenizer, prompt)
            print(f"\n[PROMPT] {prompt}")
            print(f"[RAW]    {raw_text[:120]}")
            print(f"[CLEAN]  {repr(clean_text[:120])}")
            print(f"  tokens={len(token_ids)} finish={finish}")
        except Exception as e:
            print(f"\n[PROMPT] {prompt}")
            print(f"[ERRO] {e}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
