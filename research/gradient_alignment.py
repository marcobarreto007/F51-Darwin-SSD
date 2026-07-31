#!/usr/bin/env python3
"""
Gradient Alignment Protocol — F51 Darwin-X
==========================================
Mede alinhamento direcional entre gradientes de todos os termos de loss
no backbone. Responde: "os orgaos estao convergindo ou brigando?"

Uso:
    python research/gradient_alignment.py \\
        --checkpoint workspace/03_CHECKPOINTS_100M_FULL_ORGANISM_V1/organism_cycle_003_step_001300.pt \\
        --config src/configs/darwin_x_100m_full_organism.yaml \\
        --token-bin workspace/01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast_v2.bin \\
        --output workspace/runtime/alignment/cycle_003_step_001300.json

Output:
    - Tabela de ||g|| por termo + cos com LM
    - Matriz de cosine similarity entre TODOS os pares de termos
    - JSON com todos os dados pra tracking ao longo do tempo
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

from f51_darwin.darwin_x import DarwinXConfig, DarwinXModel  # noqa: E402


# ── CLI ────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gradient Alignment Protocol — F51 Darwin-X"
    )
    p.add_argument("--checkpoint", required=True, help="Caminho do .pt")
    p.add_argument("--config", required=True, help="YAML de config do modelo")
    p.add_argument(
        "--token-bin",
        default=str(ROOT / "workspace" / "01_TOKENIZADOS" / "00_CORPUS_PRINCIPAL_tokens_feast_v2.bin"),
        help="Corpus tokenizado int32",
    )
    p.add_argument("--seq-len", type=int, default=256, help="Tamanho da sequencia de teste")
    p.add_argument("--seed", type=int, default=42, help="Seed pra offset deterministico")
    p.add_argument("--output", help="JSON de saida (opcional)")
    p.add_argument("--device", default="cpu", help="cpu ou cuda")
    return p.parse_args()


# ── Core ────────────────────────────────────────────────────────

BACKBONE_FILTER = "blocks"
EPSILON = 1e-8


def _backbone_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for n, p in model.named_parameters() if BACKBONE_FILTER in n]


def _l2_norm(params: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().norm().item() ** 2
    return math.sqrt(total)


def _cosine_between(params_a: list[torch.nn.Parameter], params_b: list[torch.nn.Parameter]) -> float:
    """Cosine similarity entre dois conjuntos de gradientes (ja computados em .grad)."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for pa, pb in zip(params_a, params_b):
        if pa.grad is not None and pb.grad is not None:
            ga = pa.grad.detach().flatten()
            gb = pb.grad.detach().flatten()
            dot += (ga * gb).sum().item()
            norm_a += ga.norm().item() ** 2
            norm_b += gb.norm().item() ** 2
    if norm_a < EPSILON or norm_b < EPSILON:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _zero_grads(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.grad = None


def run_alignment(
    model: DarwinXModel,
    token_bin: str,
    *,
    seq_len: int = 256,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Executa o protocolo completo de alinhamento de gradiente."""

    # Sem isso, --seed so fixava o offset do texto (numpy) -- qualquer
    # aleatoriedade interna do forward (ex: ghost_mask_ratio, que mascara
    # ~15% dos tokens ao acaso a cada chamada) continuava nao-determinsitica,
    # entao rodar o script duas vezes no MESMO checkpoint dava ||g|| e
    # cosseno diferentes pro termo "ghost" a cada vez -- nao era ruido de
    # medicao, era falta de seed completa.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Dados ──
    raw = np.memmap(token_bin, dtype=np.int32, mode="r")
    total_len = len(raw)
    # memmap len pode estourar int32 — usa int64 no randint
    max_offset = max(1, total_len - seq_len - 10)
    rng = np.random.RandomState(seed)
    offset = int(rng.randint(0, min(max_offset, 2_000_000_000)))
    x = torch.from_numpy(raw[offset : offset + seq_len].copy()).to(torch.long).unsqueeze(0)
    x = x.to(device)

    # ── Forward ──
    model.train()
    _zero_grads(model)
    out = model(x[:, :-1], labels=x[:, 1:])

    # ── Coleta de termos de loss ──
    loss_terms: dict[str, torch.Tensor | None] = {}
    for key in ["lm_loss", "mtp_loss", "jepa_loss", "aux_loss", "ghost_loss", "spider_loss"]:
        val = getattr(out, key, None)
        if val is not None and torch.isfinite(val):
            loss_terms[key.replace("_loss", "")] = val
        else:
            loss_terms[key.replace("_loss", "")] = None

    bp = _backbone_params(model)

    # ── Normas por termo ──
    grad_norms: dict[str, float] = {}
    for name, loss_val in loss_terms.items():
        if loss_val is None or loss_val.item() == 0.0:
            grad_norms[name] = 0.0
            continue
        _zero_grads(model)
        loss_val.backward(retain_graph=True)
        grad_norms[name] = _l2_norm(bp)

    # ── Matriz de cosine similarity ──
    # Estrategia: faz backward de cada termo, concatena TODOS os grads do backbone
    # em um unico vetor flat, e depois computa cosine entre esses vetores.
    # Isso evita problemas de alinhamento entre listas de parametros.
    term_names = sorted(grad_norms.keys())
    n = len(term_names)
    cosine_matrix = [[0.0] * n for _ in range(n)]

    # Coleta gradiente flat por termo. IMPORTANTE: preenche com zero (nao pula)
    # quando p.grad is None, pra que a posicao k do vetor flat sempre
    # corresponda ao MESMO parametro fisico em bp[k] em qualquer termo.
    # Termos como "ghost" rodam um forward separado (tokens mascarados) que
    # ativa um subconjunto DIFERENTE de experts MoE do forward principal --
    # sem esse zero-fill, o vetor flat de "ghost" tinha um tamanho/identidade
    # de parametros diferente do vetor de "lm", e o cosseno entre os dois
    # comparava gradientes de experts nao-correspondentes (sem sentido).
    flat_grads: dict[str, torch.Tensor] = {}
    for name in term_names:
        loss_val = loss_terms.get(name)
        if loss_val is None or loss_val.item() == 0.0:
            flat_grads[name] = torch.zeros(sum(p.numel() for p in bp))
            continue
        _zero_grads(model)
        loss_val.backward(retain_graph=True)
        chunks = [
            p.grad.detach().flatten().clone() if p.grad is not None
            else torch.zeros(p.numel())
            for p in bp
        ]
        flat_grads[name] = torch.cat(chunks) if chunks else torch.zeros(1)

    for i, name_i in enumerate(term_names):
        cosine_matrix[i][i] = 1.0
        for j, name_j in enumerate(term_names):
            if i >= j:
                continue
            gi = flat_grads.get(name_i)
            gj = flat_grads.get(name_j)
            ni = grad_norms.get(name_i, 0.0)
            nj = grad_norms.get(name_j, 0.0)
            if ni < EPSILON or nj < EPSILON or gi is None or gj is None:
                continue
            assert gi.numel() == gj.numel(), (
                f"positionally-aligned flat grads must match in size: "
                f"{name_i}={gi.numel()} vs {name_j}={gj.numel()}"
            )
            dot = (gi * gj).sum().item()
            na = gi.norm().item()
            nb = gj.norm().item()
            cos = dot / (na * nb) if na > EPSILON and nb > EPSILON else 0.0
            cosine_matrix[i][j] = cos
            cosine_matrix[j][i] = cos

    # ── Norma do gradiente total ──
    _zero_grads(model)
    out.loss.backward()
    total_norm = _l2_norm(bp)

    # ── Perdas ──
    loss_values = {}
    for name in term_names:
        lt = loss_terms.get(name)
        loss_values[name] = float(lt.item()) if lt is not None else 0.0

    # ── Diagnostico ──
    total_all = sum(grad_norms.values())
    alignment_diagnostics = {}
    for name in term_names:
        gn = grad_norms.get(name, 0.0)
        pct = (gn / total_all * 100) if total_all > 0 else 0.0
        cos_with_lm = 0.0
        if name != "lm" and "lm" in term_names:
            li = term_names.index("lm")
            ni = term_names.index(name)
            cos_with_lm = cosine_matrix[li][ni]
        if name == "lm":
            verdict = "REFERENCIA"
        elif cos_with_lm > 0.1:
            verdict = "COOPERA"
        elif cos_with_lm < -0.1:
            verdict = "COMPETE"
        else:
            verdict = "independente"
        alignment_diagnostics[name] = {
            "grad_norm": gn,
            "pct_total": round(pct, 1),
            "loss_value": loss_values.get(name, 0.0),
            "cos_with_lm": round(cos_with_lm, 4),
            "verdict": verdict,
        }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seq_len": seq_len,
        "seed": seed,
        "offset": int(offset),
        "term_names": term_names,
        "loss_values": loss_values,
        "grad_norms": {k: round(v, 6) for k, v in grad_norms.items()},
        "total_grad_norm": round(total_norm, 6),
        "cosine_matrix": {
            term_names[i]: {
                term_names[j]: round(cosine_matrix[i][j], 4)
                for j in range(n)
            }
            for i in range(n)
        },
        "alignment_diagnostics": alignment_diagnostics,
    }


# ── Output ──────────────────────────────────────────────────────

def print_alignment_report(data: dict) -> None:
    """Tabela legivel no terminal."""
    term_names = data["term_names"]
    norms = data["grad_norms"]
    diag = data["alignment_diagnostics"]
    losses = data["loss_values"]

    print()
    print("=" * 85)
    print("  GRADIENT ALIGNMENT REPORT")
    print(f"  Timestamp : {data['timestamp']}")
    print(f"  Seq len   : {data['seq_len']}  Seed: {data['seed']}")
    print(f"  ||g_total||: {data['total_grad_norm']:.2e}")
    print("=" * 85)
    print()
    print(f"{'Term':>8s}  {'Loss':>10s}  {'||g||':>10s}  {'%':>6s}  {'cos w/ LM':>11s}  {'Verdict':>15s}")
    print("-" * 75)
    for name in term_names:
        d = diag.get(name, {})
        print(
            f"{name:>8s}  {losses.get(name, 0):>10.4f}  "
            f"{norms.get(name, 0):>10.2e}  {d.get('pct_total', 0):>5.1f}%  "
            f"{d.get('cos_with_lm', 0):>+10.4f}  {d.get('verdict', '?'):>15s}"
        )
    print()

    # Matriz
    print("─" * 60)
    print("  COSINE SIMILARITY MATRIX")
    print("─" * 60)
    print(f"{'':>8s}  " + "  ".join(f"{t:>8s}" for t in term_names))
    cm = data["cosine_matrix"]
    for ti in term_names:
        row = "  ".join(f"{cm[ti][tj]:>+8.4f}" for tj in term_names)
        print(f"{ti:>8s}  {row}")
    print()

    # Interpretacao
    conflicts = []
    cooperations = []
    for ti in term_names:
        for tj in term_names:
            if ti < tj:
                cos = cm[ti][tj]
                if cos < -0.1:
                    conflicts.append((ti, tj, cos))
                elif cos > 0.3:
                    cooperations.append((ti, tj, cos))

    if conflicts:
        print(f"CONFLITOS ({len(conflicts)}):")
        for a, b, c in conflicts:
            print(f"  {a} <-> {b}: cos={c:+.4f}")
    else:
        print("CONFLITOS: ZERO — todos os gradientes estao alinhados ou ortogonais")

    if cooperations:
        print(f"ALIADOS FORTES ({len(cooperations)}):")
        for a, b, c in cooperations:
            print(f"  {a} <-> {b}: cos={c:+.4f}")
    print("=" * 85)


# ── MAIN ────────────────────────────────────────────────────────

def main() -> int:
    global args
    args = parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    print(f"Loading config: {args.config}")
    cfg = DarwinXConfig.from_mapping(yaml.safe_load(open(args.config)))

    print("Building model...")
    model = DarwinXModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(args.device)
    model.eval()

    print(f"Running gradient alignment protocol (seq={args.seq_len}, seed={args.seed})...")
    data = run_alignment(
        model, args.token_bin,
        seq_len=args.seq_len, seed=args.seed, device=args.device,
    )
    data["checkpoint"] = str(Path(args.checkpoint).resolve())
    data["config"] = str(Path(args.config).resolve())
    data["cycle"] = ckpt.get("cycle", None)
    data["step"] = ckpt.get("step", None)

    print_alignment_report(data)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
