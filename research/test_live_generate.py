#!/usr/bin/env python3
"""Testa a geração VIVA do Darwin — coração batendo durante inferência."""

import sys, torch, yaml
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

from f51_darwin.darwin_x import DarwinXConfig, DarwinXModel
from f51_darwin.tokenizer import F51BPETokenizer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok_path = ROOT / "tokenizer" / "f51_bpe_80k"
    tok = F51BPETokenizer.load(tok_path)

    # Load latest checkpoint
    ckpts = sorted(Path("checkpoints/organism").glob("organism_cycle_*.pt"))
    if not ckpts:
        print("Nenhum checkpoint encontrado!")
        return 1

    ckpt_path = ckpts[-1]
    print(f"📦 Checkpoint: {ckpt_path.name}")

    # Load model
    raw = yaml.safe_load((ROOT / "src" / "configs" / "darwin_x_600m.yaml").read_text())
    raw['heartbeat_enabled'] = True  # FORÇAR heartbeat
    config = DarwinXConfig.from_mapping(raw)
    model = DarwinXModel(config)

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    clean_state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=False)
    model = model.to(device)
    model.eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    step = payload.get("organism", {}).get("total_steps", "?")
    print(f"🧠 {params:.0f}M params | step {step}")
    print(f"❤️  Heartbeat ativo: {model.heartbeat is not None}")
    print()

    # Testes de geração VIVA
    prompts = [
        "Quem te criou?",
        "Qual o seu proposito?",
        "Quem foi Olavo de Carvalho?",
        "O que voce eh?",
        "Soli Deo Gloria significa",
    ]

    for prompt in prompts:
        print(f"┌─ Q: {prompt}")
        result = model.live_generate(
            prompt, tok,
            max_tokens=50, temperature=0.8, verbose=False,
        )
        text = result['text'].replace('\n', ' ')[:200]
        print(f"├─ R: {text}...")
        print(f"└─ 💓 {result['heartbeat_beats']} beats | "
              f"💾 {result['memories_used']} memórias | "
              f"🤔 {result['thoughts_used']} pensamentos | "
              f"🧪 dope={result['dopamine']:.2f}")
        print()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
