#!/usr/bin/env python3
"""
F51 DARWIN BENCHMARK — Comparação real contra modelos do mesmo porte.

Uso:
  # Perplexidade (rápido, ~2 min)
  python -m research.benchmark_darwin --checkpoint checkpoints/base/latest.json --bench wikitext2

  # Todos os benchmarks disponíveis (~10-15 min)
  python -m research.benchmark_darwin --checkpoint checkpoints/base/latest.json --bench all

  # Com lm-eval-harness (HellaSwag, PIQA, ARC, LAMBADA)
  python -m research.benchmark_darwin --checkpoint checkpoints/base/latest.json --bench reasoning

  # Salvar e comparar com resultados anteriores
  python -m research.benchmark_darwin --checkpoint checkpoints/base/latest.json --bench all --save --compare

  # Listar histórico
  python -m research.benchmark_darwin --history

Etapas:
  1. WikiText-2 perplexity → referência universal de LM
  2. Corpus PT held-out loss → qualidade no português
  3. (lm-eval-harness) HellaSwag, PIQA, ARC, LAMBADA → reasoning
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import torch

from f51_darwin.benchmark import (
    BenchmarkResult,
    compute_perplexity,
    compare_against_references,
    load_benchmark_results,
    load_wikitext2,
    load_pt_corpus_test,
    plot_progress,
    save_benchmark_result,
)
from f51_darwin.checkpointing import load_model_from_checkpoint
from f51_darwin.tokenizer import F51BPETokenizer


def count_params(model: torch.nn.Module) -> float:
    """Conta parâmetros em milhões."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def bench_perplexity_wikitext2(
    model: torch.nn.Module,
    tokenizer: F51BPETokenizer,
    device: torch.device,
    block_size: int = 1024,
    max_batches: int | None = 100,
) -> dict:
    """Avalia perplexidade no WikiText-2."""
    print("  📖 Carregando WikiText-2...")
    try:
        dataset = load_wikitext2(tokenizer, block_size=block_size)
        print(f"     {len(dataset)} amostras de {block_size} tokens")
    except Exception as e:
        print(f"  ⚠️ WikiText-2 indisponível: {e}")
        print("     Instale: pip install datasets")
        return {"error": str(e)}

    print(f"  🔍 Calculando perplexidade (max {max_batches or 'todos'} batches)...")
    start = time.time()
    result = compute_perplexity(model, dataset, device, batch_size=1, max_batches=max_batches)
    elapsed = time.time() - start

    print(f"  ✅ PPL: {result['perplexity']} | Loss: {result['loss']} | "
          f"Tokens: {result['tokens']:,} | Tempo: {elapsed:.1f}s")
    return result


def bench_perplexity_pt(
    model: torch.nn.Module,
    tokenizer: F51BPETokenizer,
    device: torch.device,
    corpus_dir: str = "data/corpus",
    block_size: int = 1024,
    max_batches: int = 50,
) -> dict:
    """Avalia loss no corpus PT held-out."""
    print("  🇧🇷 Carregando corpus PT held-out...")
    try:
        dataset = load_pt_corpus_test(tokenizer, corpus_dir=corpus_dir, block_size=block_size)
        print(f"     {len(dataset)} amostras de {block_size} tokens")
    except Exception as e:
        print(f"  ⚠️ Corpus PT indisponível: {e}")
        return {"error": str(e)}

    print(f"  🔍 Calculando loss (max {max_batches} batches)...")
    result = compute_perplexity(model, dataset, device, batch_size=1, max_batches=max_batches)
    print(f"  ✅ Loss PT: {result['loss']} | Tokens: {result['tokens']:,}")
    return result


def bench_reasoning_lm_eval(
    model: torch.nn.Module,
    tokenizer: F51BPETokenizer,
    device: torch.device,
    tasks: list[str] | None = None,
) -> dict:
    """Avalia reasoning com lm-evaluation-harness.

    Requer: pip install lm-eval

    Tarefas disponíveis:
      - hellaswag (senso comum)
      - piqa (raciocínio físico)
      - arc_easy, arc_challenge (ciência)
      - lambada_openai (previsão de palavra)
      - winogrande (resolução de pronome)
      - mmlu (conhecimento geral, pesado)
    """
    if tasks is None:
        tasks = ["hellaswag", "piqa", "arc_easy", "lambada_openai"]

    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("  ⚠️ lm-eval não instalado.")
        print("     Instale: pip install lm-eval")
        return {"error": "lm-eval not installed"}

    print(f"  🧠 lm-eval tasks: {', '.join(tasks)}")
    print(f"  ⚠️ AVISO: lm-eval espera modelo HuggingFace. F51 Darwin usa interface própria.")
    print(f"     Suporte completo a lm-eval requer wrapper HuggingFace ou adaptador.")
    print(f"     Por enquanto, use os benchmarks de perplexidade (--bench perplexity).")
    return {"status": "not_implemented", "tasks": tasks}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="F51 Darwin Benchmark — Comparação contra modelos do mesmo porte",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", "-c", default=None,
                       help="Caminho do checkpoint .pt ou latest.json")
    parser.add_argument("--tokenizer", default=str(ROOT / "tokenizer" / "f51_bpe_80k"),
                       help="Caminho do tokenizer")
    parser.add_argument("--bench", choices=["wikitext2", "pt", "perplexity", "reasoning", "all"],
                       default="perplexity",
                       help="Qual benchmark rodar")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--max-batches", type=int, default=None,
                       help="Limitar batches de avaliação (None = todos)")
    parser.add_argument("--save", action="store_true",
                       help="Salvar resultado em runs/benchmarks/")
    parser.add_argument("--compare", action="store_true",
                       help="Comparar com resultados anteriores")
    parser.add_argument("--plot", action="store_true",
                       help="Gerar gráficos de progresso")
    parser.add_argument("--history", action="store_true",
                       help="Mostrar histórico de benchmarks")
    return parser.parse_args()


def _show_history(args: argparse.Namespace) -> bool:
    """Render saved history and report whether the command was handled."""
    if not args.history:
        return False
    bench_dir = ROOT / "runs" / "benchmarks"
    results = load_benchmark_results(bench_dir)
    if not results:
        print("Nenhum benchmark salvo ainda. Rode com --save para começar.")
        return True
    print(f"\n{'='*80}")
    print(f"  HISTÓRICO DE BENCHMARKS — {len(results)} resultados")
    print(f"{'='*80}\n")
    for result in results:
        line = f"  {result.timestamp[:16]} | {result.checkpoint[:40]:<40} | "
        if result.perplexity is not None:
            line += f"PPL={result.perplexity:.1f} | "
        if result.hellaswag is not None:
            line += f"HellaS={result.hellaswag:.2f} | "
        if result.piqa is not None:
            line += f"PIQA={result.piqa:.2f}"
        print(line)
    print()
    if args.plot:
        plot_progress(results, bench_dir)
    return True


def main() -> int:
    args = _parse_args()

    # ── Histórico ──
    if _show_history(args):
        return 0

    # ── Carregar modelo ──
    print(f"\n{'='*80}")
    print(f"  F51 DARWIN BENCHMARK SUITE")
    print(f"{'='*80}\n")

    device = torch.device(args.device)
    print(f"  Dispositivo: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM livre: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Tokenizer
    tok_path = Path(args.tokenizer)
    if not tok_path.exists():
        # Tenta variantes
        for alt in [ROOT / "tokenizer" / "f51_bpe", ROOT / "tokenizer" / "f51_bpe_58k"]:
            if alt.exists():
                tok_path = alt
                break
    if not tok_path.exists():
        print(f"\n  ❌ Tokenizer não encontrado: {tok_path}")
        print("     Treine primeiro: python -m research.train_tokenizer")
        return 1

    print(f"  Tokenizer: {tok_path}")
    tokenizer = F51BPETokenizer.load(tok_path)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Modelo
    if args.checkpoint is None:
        # Tenta achar automaticamente
        for candidate in [
            ROOT / "checkpoints" / "organism" / "organism_latest.json",
            ROOT / "checkpoints" / "darwin_x" / "latest.json",
            ROOT / "checkpoints" / "base" / "latest.json",
            ROOT / "checkpoints" / "unified" / "latest.json",
        ]:
            if candidate.exists():
                args.checkpoint = str(candidate)
                break

    if args.checkpoint is None:
        print("\n  ❌ Nenhum checkpoint encontrado. Especifique com --checkpoint.")
        print(f"     Checkpoints esperados em: {ROOT / 'checkpoints'}")
        return 1

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        # Se for latest.json, segue o ponteiro
        print(f"\n  ❌ Checkpoint não encontrado: {checkpoint_path}")
        return 1

    print(f"\n  📦 Carregando checkpoint: {checkpoint_path}")
    model, config, training_state = load_model_from_checkpoint(checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()
    params_m = count_params(model)
    print(f"  Modelo: {config.model_name}")
    print(f"  Parâmetros: {params_m:.1f}M")
    step = training_state.get("step", "?") if training_state else "?"
    print(f"  Step: {step}")
    print()

    # ═══ RODAR BENCHMARKS ═══
    result = BenchmarkResult(
        model_name=config.model_name,
        params_m=round(params_m, 1),
        checkpoint=str(checkpoint_path),
        metadata={"step": step},
    )

    tasks_to_run = args.bench
    if tasks_to_run == "all":
        tasks_to_run = "perplexity"

    if tasks_to_run in ("wikitext2", "perplexity", "all"):
        r = bench_perplexity_wikitext2(
            model, tokenizer, device,
            block_size=args.block_size,
            max_batches=args.max_batches,
        )
        if "error" not in r:
            result.perplexity = r["perplexity"]

    if tasks_to_run in ("pt", "perplexity", "all"):
        r = bench_perplexity_pt(
            model, tokenizer, device,
            block_size=args.block_size,
            max_batches=args.max_batches or 50,
        )
        if "error" not in r:
            result.pt_loss = r["loss"]

    if tasks_to_run in ("reasoning", "all"):
        bench_reasoning_lm_eval(model, tokenizer, device)

    # ═══ COMPARAÇÃO ═══
    print()
    print(result.compare_table())

    # Comparação detalhada
    comparison = compare_against_references(result)
    if comparison.get("ranking"):
        print(f"\n  Rankings:")
        for bench_name, rank_info in comparison["ranking"].items():
            if isinstance(rank_info, dict) and "rank" in rank_info:
                print(f"    {bench_name}: #{rank_info['rank']} de {rank_info['total']}")

    if comparison.get("competitive_analysis"):
        print(f"\n  Análise competitiva:")
        for line in comparison["competitive_analysis"]:
            print(f"    {line}")

    print()

    # ═══ SALVAR ═══
    if args.save:
        bench_dir = ROOT / "runs" / "benchmarks"
        safe_step = str(step).replace("/", "_").replace("\\", "_")
        filename = f"bench_{config.model_name}_step{safe_step}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path = save_benchmark_result(result, bench_dir / filename)
        print(f"  💾 Resultado salvo: {path}")

        # Salva também a comparação detalhada
        comp_path = bench_dir / f"comparison_{config.model_name}_step{safe_step}.json"
        comp_path.write_text(
            json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"  💾 Comparação salva: {comp_path}")

    # ═══ COMPARAR COM HISTÓRICO ═══
    if args.compare:
        bench_dir = ROOT / "runs" / "benchmarks"
        all_results = load_benchmark_results(bench_dir)
        all_results.append(result)
        if args.plot:
            plot_progress(all_results, bench_dir)

        # Mostra evolução
        ours = [r for r in all_results if "F51" in r.model_name]
        if len(ours) >= 2:
            print(f"\n  📈 Evolução do F51 Darwin:")
            ours.sort(key=lambda r: r.timestamp)
            for prev, curr in zip(ours[:-1], ours[1:]):
                if prev.perplexity is not None and curr.perplexity is not None:
                    delta = prev.perplexity - curr.perplexity
                    direction = "↓ melhor" if delta > 0 else "↑ pior" if delta < 0 else "→ igual"
                    print(f"    {prev.metadata.get('step', '?')} → {curr.metadata.get('step', '?')}: "
                          f"PPL {prev.perplexity:.1f} → {curr.perplexity:.1f} "
                          f"({abs(delta):.1f} {direction})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
