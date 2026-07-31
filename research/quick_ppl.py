#!/usr/bin/env python3
"""
F51 QUICK PPL — Perplexidade rápida usando o próprio corpus (sem internet).

Uso:
  python research/quick_ppl.py                           # usa latest.json
  python research/quick_ppl.py --checkpoint path/to/ckpt.pt
  python research/quick_ppl.py --steps 5000 10000 15000  # compara checkpoints
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import torch
from torch.nn import functional as F

from f51_darwin.checkpointing import load_model_from_checkpoint
from f51_darwin.data import CausalLMDataLoader, resolve_corpus_dir
from f51_darwin.tokenizer import F51BPETokenizer
from f51_darwin.artifacts import resolve_token_bin


def load_model_and_tokenizer(checkpoint: str, device: torch.device):
    """Carrega modelo e tokenizer de um checkpoint."""
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint: {ckpt_path}")

    model, config, training_state = load_model_from_checkpoint(ckpt_path, map_location=device)
    model = model.to(device)
    model.eval()

    # Tokenizer
    tok_path = ROOT / "tokenizer" / "f51_bpe_80k"
    if not tok_path.exists():
        tok_path = ROOT / "tokenizer" / "f51_bpe"
    tokenizer = F51BPETokenizer.load(tok_path)

    step = training_state.get("step", "?") if training_state else "?"
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    return model, config, tokenizer, params_m, step


def quick_ppl(model, token_ids, device, block_size=256, batch_size=1, num_batches=20):
    """Calcula perplexidade em batches do corpus."""
    model.eval()
    loader = CausalLMDataLoader(
        token_ids, block_size=block_size, batch_size=batch_size,
        seed=999, device=device,
    )

    total_nll = 0.0
    total_tokens = 0
    losses = []

    with torch.no_grad():
        for i in range(num_batches):
            batch = loader.next_batch()
            output = model(batch, labels=batch)

            if hasattr(output, 'lm_loss') and output.lm_loss is not None:
                loss = output.lm_loss
            elif hasattr(output, 'loss') and output.loss is not None:
                loss = output.loss
            else:
                logits = output.logits if hasattr(output, 'logits') else output
                loss = F.cross_entropy(
                    logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                    batch[:, 1:].contiguous().view(-1),
                    ignore_index=-100,
                )

            n_tokens = batch.numel()
            total_nll += float(loss) * n_tokens
            total_tokens += n_tokens
            losses.append(float(loss))

    avg_loss = total_nll / total_tokens if total_tokens else float("inf")
    ppl = math.exp(min(avg_loss, 20.0))
    return {
        "perplexity": round(ppl, 2),
        "loss": round(avg_loss, 4),
        "tokens": total_tokens,
        "batches": len(losses),
        "loss_history": [round(l, 4) for l in losses],
    }


def main():
    parser = argparse.ArgumentParser(description="F51 Quick PPL — Perplexidade sem internet")
    parser.add_argument("--checkpoint", "-c", default=None,
                       help="Caminho do checkpoint")
    parser.add_argument("--steps", type=int, nargs="+", default=None,
                       help="Comparar múltiplos checkpoints (ex: 5000 10000 15000)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-batches", type=int, default=20,
                       help="Número de batches para avaliação")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device)
    print("═" * 60)
    print("  F51 QUICK PPL — Perplexidade no corpus interno")
    print("═" * 60)
    print(f"  Device: {device}")

    # Tokens
    tokens_bin = resolve_token_bin(ROOT, min_bytes=1_000_000)
    if tokens_bin is None:
        print("  ❌ Nenhum token bin encontrado. Tokenize primeiro.")
        return 1

    import numpy as np
    token_ids = np.memmap(tokens_bin, dtype=np.int32, mode='r')
    total_m = len(token_ids) / 1e6
    print(f"  Tokens: {total_m:.1f}M ({tokens_bin})")
    print(f"  Batches de avaliação: {args.num_batches} × {args.block_size} tokens")
    print()

    if args.steps:
        # Compara múltiplos checkpoints
        base_dir = ROOT / "checkpoints" / "base"
        results = []

        for step in args.steps:
            # Encontra checkpoint
            run_dirs = sorted(base_dir.glob("med_run_*"))
            ckpt_path = None
            for run_dir in run_dirs:
                candidate = run_dir / f"step_{step:07d}.pt"
                if candidate.exists():
                    ckpt_path = str(candidate)
                    break

            if ckpt_path is None:
                print(f"  ⚠️ Checkpoint step {step} não encontrado — pulando.")
                continue

            print(f"  📦 step {step}: {Path(ckpt_path).name}")
            model, config, tokenizer, params_m, _ = load_model_and_checkpoint(ckpt_path, device)
            result = quick_ppl(model, token_ids, device,
                              block_size=args.block_size,
                              batch_size=args.batch_size,
                              num_batches=args.num_batches)
            result["step"] = step
            result["params_m"] = round(params_m, 1)
            results.append(result)
            print(f"     PPL={result['perplexity']} | Loss={result['loss']}")
            print()

        # Tabela comparativa
        if len(results) >= 2:
            print("  📊 Comparação:")
            print(f"  {'Step':>8} {'PPL':>10} {'Loss':>10} {'Δ PPL':>10}")
            print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10}")
            prev = None
            for r in results:
                delta = ""
                if prev:
                    d = r["perplexity"] - prev["perplexity"]
                    direction = "↓" if d < 0 else "↑"
                    delta = f"{d:+.1f}{direction}"
                print(f"  {r['step']:>8} {r['perplexity']:>10.2f} {r['loss']:>10.4f} {delta:>10}")
                prev = r

            # Melhoria total
            first = results[0]["perplexity"]
            last = results[-1]["perplexity"]
            improvement = first - last
            pct = (improvement / first) * 100 if first else 0
            print()
            print(f"  Melhoria total: {improvement:.1f} PPL ({pct:.1f}%) em {len(results)} checkpoints")
            print(f"  De PPL={first:.1f} → PPL={last:.1f}")

    else:
        # Single checkpoint
        if args.checkpoint is None:
            # Procura latest.json
            for candidate in [
                ROOT / "checkpoints" / "organism" / "organism_latest.json",
                ROOT / "checkpoints" / "darwin_x" / "latest.json",
                ROOT / "checkpoints" / "base" / "latest.json",
            ]:
                if candidate.exists():
                    args.checkpoint = str(candidate)
                    break

        if args.checkpoint is None:
            print("  ❌ Checkpoint não encontrado. Use --checkpoint.")
            return 1

        print(f"  📦 Checkpoint: {args.checkpoint}")
        model, config, tokenizer, params_m, step = load_model_and_checkpoint(
            args.checkpoint, device,
        )
        print(f"  Modelo: {config.model_name}")
        print(f"  Parâmetros: {params_m:.1f}M")
        print(f"  Step: {step}")
        print()

        result = quick_ppl(model, token_ids, device,
                          block_size=args.block_size,
                          batch_size=args.batch_size,
                          num_batches=args.num_batches)
        result["step"] = step
        result["params_m"] = round(params_m, 1)

        print("═" * 60)
        print(f"  PPL: {result['perplexity']}")
        print(f"  Loss: {result['loss']}")
        print(f"  Tokens avaliados: {result['tokens']:,}")
        print(f"  Batches: {result['batches']}")
        print("═" * 60)

        # Interpretação rápida
        ppl = result["perplexity"]
        if ppl > 100:
            print("  📊 Interpretação: modelo ainda no início (PPL > 100)")
            print("     Esperado para modelo com poucos steps de treino.")
            print("     GPT-2 Small tem PPL ~37 após 8B tokens de treino.")
        elif ppl > 50:
            print("  📊 Interpretação: modelo aprendendo (PPL 50-100)")
            print("     Já capturou padrões básicos. Continue treinando.")
        elif ppl > 25:
            print("  📊 Interpretação: modelo funcional (PPL 25-50)")
            print("     Se aproximando de GPT-2 Small (PPL=37). Bom progresso!")
        elif ppl > 15:
            print("  📊 Interpretação: modelo competitivo (PPL 15-25)")
            print("     Na faixa de GPT-2 Medium (PPL=26) / Pythia 410M (PPL=18)!")
        else:
            print("  🏆 Interpretação: EXCELENTE (PPL < 15)")
            print("     Competindo com modelos muito maiores. Parabéns!")

    print()
    print("  💡 Para benchmark externo (comparação real):")
    print("     python research/benchmark_darwin.py --bench wikitext2 --save")
    print("═" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
