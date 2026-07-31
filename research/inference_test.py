#!/usr/bin/env python3
"""Teste rapido de inferencia — carrega checkpoint e gera texto.
   Uso: python research/inference_test.py [--prompt TEXT] [--max-tokens N]

   O treino PRECISA estar parado antes de rodar (disputa de GPU)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="F51 Darwin-X — Teste de Inferencia")
    parser.add_argument("--prompt", default="O futuro da inteligencia artificial no Brasil",
                        help="Prompt de entrada")
    parser.add_argument("--max-tokens", type=int, default=150, help="Max tokens a gerar")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-root", default="workspace/03_CHECKPOINTS_100M_FULL_V9",
                        help="Raiz de checkpoint relativa ao repo (ex: workspace/03_CHECKPOINTS_100M_SHUFFLED_V1)")
    args = parser.parse_args()

    ckpt_root = ROOT / args.checkpoint_root

    # ── Resolver checkpoint via organism_latest.json ──
    latest_json = ckpt_root / "organism_latest.json"
    if not latest_json.exists():
        print("ERRO: organism_latest.json nao encontrado")
        sys.exit(1)
    latest = json.loads(latest_json.read_text())
    ckpt_path = ckpt_root / latest["path"]

    print(f"Checkpoint: {latest['path']}")
    print(f"Cycle: {latest['cycle']}  Step: {latest['step']}")
    print(f"base_checkpoint_id: {latest['base_checkpoint_id'][:32]}...")
    print(f"Tamanho: {ckpt_path.stat().st_size / 1e9:.2f} GB")
    print(f"Prompt: \"{args.prompt}\"")
    print(f"Temperature: {args.temperature}  Max tokens: {args.max_tokens}")
    print("=" * 60)

    # ── Carregar modelo ──
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}  dtype: {dtype}")

    from f51_darwin.organism.checkpoint import load_model_from_checkpoint
    model, config, metrics = load_model_from_checkpoint(str(ckpt_path), map_location=device)
    model = model.to(dtype=dtype)
    model.eval()
    print(f"Modelo: {config.model_name}  d_model={config.d_model}  layers={config.n_layers}")
    print(f"Experts: {config.fine_experts}×{config.fine_expert_hidden_dim}d  {config.experts_per_token}/token")

    # ── Tokenizer ──
    tok_path = ROOT / "workspace" / "tokenizer" / "f51_bpe_80k"
    from f51_darwin.tokenizer import F51BPETokenizer
    tokenizer = F51BPETokenizer.load(str(tok_path))
    print(f"Tokenizer: {tok_path}  vocab={tokenizer.vocab_size}")

    # ── Tokenizar ──
    tokens = tokenizer.encode(args.prompt)
    input_ids = torch.tensor([tokens], device=device, dtype=torch.long)
    prompt_len = len(tokens)
    print(f"Tokens de entrada: {prompt_len}")

    # ── Gerar ──
    generated = input_ids.clone()
    ctx_len = config.context_length
    with torch.no_grad():
        for i in range(args.max_tokens):
            window = generated[:, -ctx_len:]
            logits = model(window).logits
            next_logits = logits[0, -1] / args.temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == tokenizer.eos_id:
                print(f"[EOS no token {i+1}]")
                break

    # ── Decodificar ──
    output_ids = generated[0].tolist()
    full_text = tokenizer.decode(output_ids)
    generated_text = tokenizer.decode(output_ids[prompt_len:])

    print("=" * 60)
    print("ENTRADA:")
    print(args.prompt)
    print("-" * 40)
    print("GERADO:")
    print(generated_text)
    print("=" * 60)

    # ── Salvar ──
    result = {
        "checkpoint": latest["path"],
        "cycle": latest["cycle"],
        "step": latest["step"],
        "base_checkpoint_id": latest["base_checkpoint_id"],
        "prompt": args.prompt,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "tokens_generated": len(output_ids) - prompt_len,
        "output": full_text,
    }
    out_path = ROOT / "workspace" / "runtime" / "inference_test_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Resultado salvo em: {out_path}")


if __name__ == "__main__":
    main()
