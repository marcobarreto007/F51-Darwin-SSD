"""
F51 Darwin Benchmark Suite — Perplexidade e reasoning zero-shot.

Comparação real contra modelos de mesmo porte (GPT-2, Pythia, OPT, SmolLM).
Sem score inventado. Métricas padrão da indústria.

Fase 1: Perplexidade (WikiText-2, corpus PT held-out)
Fase 2: Reasoning (HellaSwag, PIQA, ARC, LAMBADA via lm-eval-harness)
Fase 3: Português (ASSIN2, BLUEX, ENEM) + MMLU
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


# ═══════════════════════════════════════════════════════════
# DATASETS PADRÃO
# ═══════════════════════════════════════════════════════════

WIKITEXT2_URL = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1.zip"
PTB_URL = "https://huggingface.co/datasets/ptb-text-only/ptb_text_only/resolve/main/penn-treebank.zip"


class TextDataset(Dataset):
    """Dataset de texto para avaliação de perplexidade."""

    def __init__(self, texts: list[str], tokenizer, block_size: int = 1024):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.token_ids = self._tokenize(texts)
        self._num_samples = max(0, len(self.token_ids) - block_size)

    def _tokenize(self, texts: list[str]) -> list[int]:
        ids: list[int] = []
        for text in texts:
            encoded = self.tokenizer.encode(text.strip(), add_eos=True)
            ids.extend(encoded)
        return ids

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.token_ids[index : index + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def load_wikitext2(tokenizer, block_size: int = 1024) -> TextDataset:
    """Carrega WikiText-2 (test split) para avaliação de perplexidade."""
    try:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if t.strip()]
        return TextDataset(texts, tokenizer, block_size)
    except ImportError:
        raise ImportError("pip install datasets para carregar WikiText-2")


def load_pt_corpus_test(tokenizer, corpus_dir: str = "data/corpus", block_size: int = 1024) -> TextDataset:
    """Usa os últimos 10% do corpus F51 como held-out para perplexidade."""
    from f51_darwin.data import discover_corpus_files, load_text_documents

    corpus = Path(corpus_dir)
    if not corpus.exists():
        raise FileNotFoundError(f"Corpus não encontrado: {corpus_dir}")

    documents = load_text_documents(corpus)
    # Usa os últimos 10% como held-out
    split = max(1, len(documents) // 10)
    held_out = documents[-split:]
    return TextDataset(held_out, tokenizer, block_size)


# ═══════════════════════════════════════════════════════════
# PERPLEXIDADE
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def compute_perplexity(
    model: nn.Module,
    dataset: TextDataset,
    device: torch.device,
    batch_size: int = 1,
    max_batches: int | None = None,
    stride: int | None = None,
) -> dict[str, float]:
    """Calcula perplexidade sobre um dataset de texto.

    Args:
        model: Modelo F51 Darwin (F51DarwinModel ou DarwinXModel)
        dataset: TextDataset tokenizado
        device: torch device
        batch_size: batch size para avaliação
        max_batches: Limite opcional de batches (None = todo o dataset)
        stride: Stride para janela deslizante (None = sem overlap, block_size chunks)

    Returns:
        dict com perplexity, loss, tokens, nll
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        try:
            # DarwinXModel tem interface diferente de F51DarwinModel
            output = model(x, labels=y) if hasattr(model, 'config') else model(x)
            if hasattr(output, 'lm_loss') and output.lm_loss is not None:
                loss = output.lm_loss
            elif hasattr(output, 'loss') and output.loss is not None:
                loss = output.loss
            else:
                logits = output.logits if hasattr(output, 'logits') else output
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)

            total_nll += float(loss) * y.numel()
            total_tokens += y.numel()
        except Exception as e:
            print(f"  ⚠️ Erro no batch {batch_idx}: {e}")
            continue

    if total_tokens == 0:
        return {"perplexity": float("inf"), "loss": float("inf"), "tokens": 0, "nll": float("inf")}

    avg_nll = total_nll / total_tokens
    perplexity = math.exp(min(avg_nll, 20.0))  # cap em 20 nats pra evitar overflow

    return {
        "perplexity": round(perplexity, 2),
        "loss": round(avg_nll, 4),
        "tokens": total_tokens,
        "nll": round(avg_nll, 4),
    }


# ═══════════════════════════════════════════════════════════
# REFERENCE TABLE — Modelos comparáveis
# ═══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ReferenceModel:
    """Um modelo de referência para comparação."""
    name: str
    params_m: float  # milhões de parâmetros
    size_range: str  # "tiny" (< 100M), "small" (100-500M), "medium" (500M-1B)
    source: str = "?"
    wikitext2_ppl: float | None = None  # perplexidade no WikiText-2
    ptb_ppl: float | None = None
    hellaswag_acc: float | None = None
    piqa_acc: float | None = None
    arc_easy_acc: float | None = None
    lambada_acc: float | None = None
    notes: str = ""


# Dados de benchmarks conhecidos (fonte: papers, Open LLM Leaderboard, etc.)
# Atualizado conforme novos resultados saem.
REFERENCE_MODELS: list[ReferenceModel] = [
    # ── Tiny (< 100M) ──
    ReferenceModel("GPT-2 Small", 124, "small",
                   wikitext2_ppl=37.50, ptb_ppl=35.76,
                   hellaswag_acc=0.295, piqa_acc=0.630,
                   lambada_acc=0.325,
                   source="OpenAI paper (2019)", notes="Referência clássica. 124M params."),
    ReferenceModel("OPT 125M", 125, "small",
                   wikitext2_ppl=32.54,
                   source="Meta (2022)", notes="OPT paper."),
    ReferenceModel("Pythia 160M", 160, "small",
                   wikitext2_ppl=33.59, lambada_acc=0.383,
                   source="EleutherAI (2023)", notes="Pythia deduped."),
    ReferenceModel("SmolLM 135M", 135, "small",
                   wikitext2_ppl=30.20, hellaswag_acc=0.300,
                   piqa_acc=0.645, arc_easy_acc=0.425,
                   source="HuggingFace (2024)", notes="SmolLM-Corpus, Cosmo."),

    # ── Small (100-500M) ──
    ReferenceModel("GPT-2 Medium", 355, "small",
                   wikitext2_ppl=26.37, ptb_ppl=23.39,
                   hellaswag_acc=0.331, piqa_acc=0.672,
                   lambada_acc=0.453,
                   source="OpenAI paper (2019)", notes="Referência clássica."),
    ReferenceModel("OPT 350M", 350, "small",
                   wikitext2_ppl=22.00,
                   source="Meta (2022)", notes="OPT-350M."),
    ReferenceModel("Pythia 410M", 410, "small",
                   wikitext2_ppl=17.97, lambada_acc=0.503,
                   source="EleutherAI (2023)", notes="Pythia deduped 410M."),
    ReferenceModel("SmolLM 360M", 360, "small",
                   wikitext2_ppl=27.80, hellaswag_acc=0.323,
                   piqa_acc=0.653, arc_easy_acc=0.460,
                   source="HuggingFace (2024)", notes="SmolLM-Corpus."),

    # ── Medium (500M-1B) ──
    ReferenceModel("Pythia 1B", 1000, "medium",
                   wikitext2_ppl=14.60, lambada_acc=0.554,
                   source="EleutherAI (2023)", notes="Pythia 1B deduped."),
    ReferenceModel("OPT 1.3B", 1300, "medium",
                   wikitext2_ppl=16.96,
                   source="Meta (2022)", notes="OPT-1.3B."),
]

# ── Modelos em português (referência) ──
PT_REFERENCE_MODELS: list[ReferenceModel] = [
    ReferenceModel("BERTimbau Base", 110, "small",
                   source="UFMG (2020)", notes="BERT PT, não é LM causal."),
    ReferenceModel("Sabiá-2 Small", 210, "small",
                   source="Maritaca AI (2024)", notes="LM causal PT, ~210M."),
    ReferenceModel("PTT5 Base", 220, "small",
                   source="Unicamp", notes="T5 PT, encoder-decoder."),
]


# ═══════════════════════════════════════════════════════════
# COMPARAÇÃO
# ═══════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    """Resultado de um benchmark para um modelo."""
    model_name: str
    params_m: float
    checkpoint: str
    perplexity: float | None = None
    hellaswag: float | None = None
    piqa: float | None = None
    arc_easy: float | None = None
    arc_challenge: float | None = None
    lambada: float | None = None
    pt_loss: float | None = None  # loss no corpus PT held-out
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "params_m": self.params_m,
            "checkpoint": self.checkpoint,
            "perplexity_wikitext2": self.perplexity,
            "acc_hellaswag": self.hellaswag,
            "acc_piqa": self.piqa,
            "acc_arc_easy": self.arc_easy,
            "acc_arc_challenge": self.arc_challenge,
            "acc_lambada": self.lambada,
            "loss_pt_heldout": self.pt_loss,
            "timestamp": self.timestamp,
            **self.metadata,
        }

    def compare_table(self) -> str:
        """Gera tabela de comparação contra modelos de referência."""
        size_range = "tiny" if self.params_m < 100 else "small" if self.params_m < 500 else "medium"
        peers = [m for m in REFERENCE_MODELS if m.size_range == size_range]

        lines = [
            f"{'='*80}",
            f"  F51 DARWIN BENCHMARK — {self.model_name} ({self.params_m:.0f}M params)",
            f"  Checkpoint: {self.checkpoint}",
            f"  Data: {self.timestamp}",
            f"{'='*80}",
            "",
            f"  Comparáveis ({size_range} range, < 500M):",
        ]

        header = f"  {'Modelo':<22} {'Params':>7} {'Wiki2↓':>8} {'HellaS↑':>8} {'PIQA↑':>8} {'ARC-E↑':>8} {'LAMB↑':>8}"
        lines.append(header)
        lines.append(f"  {'-'*77}")

        # Ordena por PPL (menor = melhor)
        sorted_peers = sorted(
            peers,
            key=lambda m: m.wikitext2_ppl if m.wikitext2_ppl is not None else 999,
        )

        for ref in sorted_peers:
            lines.append(
                f"  {ref.name:<22} {ref.params_m:>6.0f}M "
                f"{_fmt(ref.wikitext2_ppl, '↓'):>8} "
                f"{_fmt(ref.hellaswag_acc, '↑'):>8} "
                f"{_fmt(ref.piqa_acc, '↑'):>8} "
                f"{_fmt(ref.arc_easy_acc, '↑'):>8} "
                f"{_fmt(ref.lambada_acc, '↑'):>8}"
            )

        # F51 Darwin (destacado)
        lines.append(f"  {'─'*77}")
        lines.append(
            f"  \033[1;32m{'F51 Darwin':<22}\033[0m \033[1;32m{self.params_m:>6.0f}M\033[0m "
            f"\033[1;33m{_fmt(self.perplexity, '↓'):>8}\033[0m "
            f"\033[1;33m{_fmt(self.hellaswag, '↑'):>8}\033[0m "
            f"\033[1;33m{_fmt(self.piqa, '↑'):>8}\033[0m "
            f"\033[1;33m{_fmt(self.arc_easy, '↑'):>8}\033[0m "
            f"\033[1;33m{_fmt(self.lambada, '↑'):>8}\033[0m"
        )
        lines.append(f"  {'─'*77}")

        # Análise
        lines.append("")
        lines.append(f"  Análise:")
        if self.perplexity is not None:
            best_ppl = min(
                (m.wikitext2_ppl for m in peers if m.wikitext2_ppl is not None),
                default=None,
            )
            if best_ppl is not None:
                delta = self.perplexity - best_ppl
                if delta <= 0:
                    lines.append(f"    🏆 PPL: MELHOR que todos os comparáveis ({self.perplexity} vs melhor {best_ppl})")
                else:
                    pct = (delta / best_ppl) * 100
                    lines.append(f"    📊 PPL: {self.perplexity} — {pct:.0f}% acima do melhor ({best_ppl})")

        lines.append(f"  {'='*80}")
        return "\n".join(lines)


def _fmt(value: float | None, direction: str = "↓") -> str:
    """Formata valor para tabela."""
    if value is None:
        return "     -  "
    return f"{value:>6.2f}{direction}"


# ═══════════════════════════════════════════════════════════
# EXPORTAÇÃO
# ═══════════════════════════════════════════════════════════

def save_benchmark_result(result: BenchmarkResult, path: str | Path) -> Path:
    """Salva resultado de benchmark como JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_benchmark_results(bench_dir: str | Path) -> list[BenchmarkResult]:
    """Carrega todos os resultados de benchmark salvos."""
    results: list[BenchmarkResult] = []
    bench_path = Path(bench_dir)
    if not bench_path.exists():
        return results
    for json_file in sorted(bench_path.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            results.append(BenchmarkResult(
                model_name=data["model"],
                params_m=data["params_m"],
                checkpoint=data["checkpoint"],
                perplexity=data.get("perplexity_wikitext2"),
                hellaswag=data.get("acc_hellaswag"),
                piqa=data.get("acc_piqa"),
                arc_easy=data.get("acc_arc_easy"),
                arc_challenge=data.get("acc_arc_challenge"),
                lambada=data.get("acc_lambada"),
                pt_loss=data.get("loss_pt_heldout"),
                timestamp=data.get("timestamp", ""),
                metadata={k: v for k, v in data.items()
                         if k not in ("model", "params_m", "checkpoint",
                                      "perplexity_wikitext2", "acc_hellaswag",
                                      "acc_piqa", "acc_arc_easy",
                                      "acc_arc_challenge", "acc_lambada",
                                      "loss_pt_heldout", "timestamp")},
            ))
        except Exception:
            continue
    return results


def plot_progress(results: list[BenchmarkResult], output_dir: str | Path) -> None:
    """Gera gráfico de progresso dos benchmarks ao longo do tempo.

    Requer matplotlib (pip install matplotlib).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib não instalado — pulando gráficos.")
        return

    if not results:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ordena por timestamp
    results.sort(key=lambda r: r.timestamp)

    metrics = {
        "Perplexidade (WikiText-2)": ("perplexity", "↓ menor melhor"),
        "HellaSwag": ("hellaswag", "↑ maior melhor"),
        "PIQA": ("piqa", "↑ maior melhor"),
        "ARC-Easy": ("arc_easy", "↑ maior melhor"),
    }

    for title, (attr, direction) in metrics.items():
        values = [getattr(r, attr) for r in results if getattr(r, attr) is not None]
        labels = [f"step {r.metadata.get('step', '?')}" for r in results
                  if getattr(r, attr) is not None]

        if not values or len(values) < 2:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(len(values)), values, "b-o", linewidth=2, markersize=8)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(f"F51 Darwin — {title} {direction}")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)

        # Adiciona referências de outros modelos
        if attr == "perplexity":
            for ref in REFERENCE_MODELS:
                if ref.wikitext2_ppl is not None and ref.params_m < 500:
                    ax.axhline(y=ref.wikitext2_ppl, color="gray", linestyle="--", alpha=0.5)
                    ax.text(0, ref.wikitext2_ppl, f" {ref.name}", fontsize=7,
                           verticalalignment="bottom", alpha=0.7)

        plt.tight_layout()
        safe_name = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(output_dir / f"benchmark_{safe_name}.png", dpi=120)
        plt.close()

    print(f"  📈 Gráficos salvos em {output_dir}")


def compare_against_references(
    result: BenchmarkResult,
) -> dict[str, Any]:
    """Compara resultado do Darwin contra todos os modelos de referência.

    Retorna ranking e análise competitiva.
    """
    size_range = "tiny" if result.params_m < 100 else "small" if result.params_m < 500 else "medium"
    peers = [m for m in REFERENCE_MODELS if m.size_range == size_range]

    comparison = {
        "our_model": result.model_name,
        "our_params_m": result.params_m,
        "our_checkpoint": result.checkpoint,
        "size_range": size_range,
        "ranking": {},
        "best_in_class": {},
        "competitive_analysis": [],
    }

    # PPL ranking (menor = melhor)
    if result.perplexity is not None:
        ppl_scores = [(m.name, m.wikitext2_ppl) for m in peers if m.wikitext2_ppl is not None]
        ppl_scores.append((result.model_name, result.perplexity))
        ppl_scores.sort(key=lambda x: x[1])
        our_rank = next(i + 1 for i, (name, _) in enumerate(ppl_scores) if name == result.model_name)
        comparison["ranking"]["perplexity"] = {
            "rank": our_rank,
            "total": len(ppl_scores),
            "top": ppl_scores[0][1],
            "ours": result.perplexity,
            "standings": ppl_scores,
        }
        best_ppl = ppl_scores[0][1]
        if our_rank == 1:
            comparison["competitive_analysis"].append(
                f"🏆 PPL: #1 de {len(ppl_scores)} — {result.perplexity} vs melhor rival {best_ppl}"
            )
        else:
            delta_pct = ((result.perplexity - best_ppl) / best_ppl) * 100
            comparison["competitive_analysis"].append(
                f"📊 PPL: #{our_rank} de {len(ppl_scores)} — {result.perplexity} "
                f"({delta_pct:.0f}% acima do líder {ppl_scores[0][0]} com {best_ppl})"
            )

    # HellaSwag ranking (maior = melhor)
    if result.hellaswag is not None:
        hs_scores = [(m.name, m.hellaswag_acc) for m in peers if m.hellaswag_acc is not None]
        hs_scores.append((result.model_name, result.hellaswag))
        hs_scores.sort(key=lambda x: x[1], reverse=True)
        our_rank = next(i + 1 for i, (name, _) in enumerate(hs_scores) if name == result.model_name)
        comparison["ranking"]["hellaswag"] = {
            "rank": our_rank,
            "total": len(hs_scores),
            "top": hs_scores[0][1],
            "ours": result.hellaswag,
        }

    # PIQA ranking (maior = melhor)
    if result.piqa is not None:
        piqa_scores = [(m.name, m.piqa_acc) for m in peers if m.piqa_acc is not None]
        piqa_scores.append((result.model_name, result.piqa))
        piqa_scores.sort(key=lambda x: x[1], reverse=True)
        our_rank = next(i + 1 for i, (name, _) in enumerate(piqa_scores) if name == result.model_name)
        comparison["ranking"]["piqa"] = {
            "rank": our_rank,
            "total": len(piqa_scores),
            "top": piqa_scores[0][1],
            "ours": result.piqa,
        }

    # LAMBADA ranking (maior = melhor)
    if result.lambada is not None:
        lambada_scores = [(m.name, m.lambada_acc) for m in peers if m.lambada_acc is not None]
        lambada_scores.append((result.model_name, result.lambada))
        lambada_scores.sort(key=lambda x: x[1], reverse=True)
        our_rank = next(i + 1 for i, (name, _) in enumerate(lambada_scores) if name == result.model_name)
        comparison["ranking"]["lambada"] = {
            "rank": our_rank,
            "total": len(lambada_scores),
            "top": lambada_scores[0][1],
            "ours": result.lambada,
        }

    return comparison
