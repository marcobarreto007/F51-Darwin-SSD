#!/usr/bin/env python3
"""
Weight Simulation Matrix — F51 Darwin-X
========================================
Testa N combinacoes de pesos dos orgaos contra um checkpoint,
medindo alinhamento de gradiente (cos c/ LM) e conflitos.

Uso:
    python research/weight_sim.py \
        --checkpoint <ckpt.pt> \
        --config <base.yaml> \
        --scenarios 20 \
        --output workspace/runtime/weight_sim/report.json
"""

from __future__ import annotations

import argparse, json, math, itertools
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

from f51_darwin.darwin_x import DarwinXConfig, DarwinXModel  # noqa: E402

EPSILON = 1e-8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weight Simulation Matrix — F51 Darwin-X")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--token-bin", default=str(ROOT / "workspace" / "01_TOKENIZADOS" / "00_CORPUS_PRINCIPAL_tokens_feast_v2.bin"))
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scenarios", type=int, default=20, help="Quantos cenarios testar")
    p.add_argument("--output", help="JSON de saida")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def _backbone_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for n, p in model.named_parameters() if "blocks" in n]


def _flat_grad(model, params, loss_tensor) -> torch.Tensor | None:
    model.zero_grad(set_to_none=True)
    loss_tensor.backward(retain_graph=True)
    chunks = [p.grad.detach().flatten().clone() if p.grad is not None else torch.zeros(p.numel()) for p in params]
    return torch.cat(chunks) if chunks else None


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm().item(), b.norm().item()
    if na < EPSILON or nb < EPSILON:
        return 0.0
    return float((a * b).sum().item() / (na * nb))


def evaluate_scenario(
    model: DarwinXModel,
    config: DarwinXConfig,
    token_bin: str,
    weights: dict[str, float],
    *,
    seq_len: int = 256,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run one forward pass with given weights, measure all cosines."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Apply weights to config
    cfg_dict = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
    cfg_dict.update(weights)
    cfg = DarwinXConfig.from_mapping(cfg_dict)

    raw = np.memmap(token_bin, dtype=np.int32, mode="r")
    max_offset = max(1, len(raw) - seq_len - 10)
    rng = np.random.RandomState(seed)
    offset = int(rng.randint(0, min(max_offset, 2_000_000_000)))
    x = torch.from_numpy(raw[offset : offset + seq_len].copy()).to(torch.long).unsqueeze(0).to(device)

    # Need fresh model to respect new config weights
    m = DarwinXModel(cfg)
    m.load_state_dict(model.state_dict(), strict=False)
    m.to(device)
    m.train()

    if "cuda" in str(device):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = m(x[:, :-1], labels=x[:, 1:])
    else:
        out = m(x[:, :-1], labels=x[:, 1:])
    bp = _backbone_params(m)

    loss_terms = {}
    term_mapping = {
        "lm": getattr(out, "lm_loss", None),
        "mtp": getattr(out, "mtp_loss", None),
        "jepa": getattr(out, "jepa_loss", None),
        "aux": getattr(out, "effective_aux_loss", None) or getattr(out, "aux_loss", None),
        "ghost": getattr(out, "ghost_loss", None) or getattr(out, "raw_ghost_loss", None),
        "spider": getattr(out, "spider_loss", None),
    }
    for name, val in term_mapping.items():
        if val is not None and isinstance(val, torch.Tensor) and torch.isfinite(val):
            loss_terms[name] = val

    # Compute flat grads for each term
    term_names = sorted(loss_terms.keys())
    flat_grads = {}
    grad_norms = {}
    for name in term_names:
        loss_val = loss_terms.get(name)
        if loss_val is None or loss_val.item() == 0.0:
            flat_grads[name] = torch.zeros(sum(p.numel() for p in bp))
            grad_norms[name] = 0.0
            continue
        g = _flat_grad(m, bp, loss_val)
        if g is not None:
            flat_grads[name] = g
            grad_norms[name] = g.norm().item()
        else:
            flat_grads[name] = torch.zeros(1)
            grad_norms[name] = 0.0

    # Cosine matrix
    n = len(term_names)
    cosines = {}
    for i, ni in enumerate(term_names):
        for j, nj in enumerate(term_names):
            if i >= j:
                continue
            gi, gj = flat_grads.get(ni), flat_grads.get(nj)
            if gi is not None and gj is not None:
                ml = min(gi.numel(), gj.numel())
                cosines[f"{ni}<->{nj}"] = _cosine(gi[:ml], gj[:ml])
            else:
                cosines[f"{ni}<->{nj}"] = 0.0

    # Alignment score: weighted average of cos(lm, organ)
    lm_cos = {k: v for k, v in cosines.items() if "lm<->" in k or "<->lm" in k}
    lm_alignment = sum(lm_cos.values()) / len(lm_cos) if lm_cos else 0.0

    # Conflict count
    conflicts = sum(1 for v in cosines.values() if v < -0.1)

    # Total grad norm and alignment of combined gradient with pure LM
    m.zero_grad(set_to_none=True)
    out.loss.backward()
    total_flat_chunks = [p.grad.detach().flatten().clone() if p.grad is not None else torch.zeros(p.numel()) for p in bp]
    total_flat = torch.cat(total_flat_chunks) if total_flat_chunks else torch.zeros(1)
    total_norm = total_flat.norm().item()
    lm_grad = flat_grads.get("lm")
    cos_total_lm = _cosine(total_flat, lm_grad) if lm_grad is not None else 1.0

    return {
        "weights": weights,
        "losses": {n: float(loss_terms[n].item()) for n in term_names if loss_terms[n] is not None},
        "grad_norms": {n: round(v, 4) for n, v in grad_norms.items()},
        "total_grad_norm": round(total_norm, 4),
        "cos_total_lm": round(cos_total_lm, 4),
        "cosines": {k: round(v, 4) for k, v in cosines.items()},
        "lm_alignment": round(lm_alignment, 4),
        "conflicts": conflicts,
    }


def generate_scenarios(n: int = 20) -> list[dict[str, float]]:
    """Generate diverse weight combinations to test."""

    # Weight ranges to explore
    axes = {
        "mtp_weight":         [0.0, 0.05, 0.10, 0.15, 0.25, 0.40],
        "jepa_weight":        [0.0, 0.03, 0.05, 0.10, 0.20, 0.40],
        "ghost_weight":       [0.0, 0.03, 0.07, 0.14, 0.28],
        "spider_calibration_weight": [0.0, 0.10, 0.50, 1.0, 2.0],
        "aux_loss_scale":     [0.0],
    }

    # Priority scenarios (hand-picked for maximum insight)
    priority = [
        # Baseline: current weights
        {"name": "baseline_atual", "mtp_weight": 0.15, "jepa_weight": 0.05, "ghost_weight": 0.07, "spider_calibration_weight": 0.02, "aux_loss_scale": 0.0},
        # Spider boost + jepa boost (what user asked)
        {"name": "spider1_jepa01", "mtp_weight": 0.15, "jepa_weight": 0.10, "ghost_weight": 0.07, "spider_calibration_weight": 1.0, "aux_loss_scale": 0.0},
        {"name": "spider1_jepa02", "mtp_weight": 0.15, "jepa_weight": 0.20, "ghost_weight": 0.07, "spider_calibration_weight": 1.0, "aux_loss_scale": 0.0},
        {"name": "spider05_jepa01", "mtp_weight": 0.15, "jepa_weight": 0.10, "ghost_weight": 0.07, "spider_calibration_weight": 0.5, "aux_loss_scale": 0.0},
        # Spider solo (no other aux)
        {"name": "spider_solo", "mtp_weight": 0.0, "jepa_weight": 0.0, "ghost_weight": 0.0, "spider_calibration_weight": 1.0, "aux_loss_scale": 0.0},
        # JEPA + Spider duo
        {"name": "jepa01_spider1", "mtp_weight": 0.0, "jepa_weight": 0.10, "ghost_weight": 0.0, "spider_calibration_weight": 1.0, "aux_loss_scale": 0.0},
        # MTP boost
        {"name": "mtp_boost", "mtp_weight": 0.30, "jepa_weight": 0.05, "ghost_weight": 0.07, "spider_calibration_weight": 0.02, "aux_loss_scale": 0.0},
        # Ghost boost
        {"name": "ghost_boost", "mtp_weight": 0.15, "jepa_weight": 0.05, "ghost_weight": 0.14, "spider_calibration_weight": 0.02, "aux_loss_scale": 0.0},
        # JEPA heavy
        {"name": "jepa_heavy", "mtp_weight": 0.10, "jepa_weight": 0.40, "ghost_weight": 0.03, "spider_calibration_weight": 0.02, "aux_loss_scale": 0.0},
        # Balanced high
        {"name": "balanced_high", "mtp_weight": 0.25, "jepa_weight": 0.20, "ghost_weight": 0.14, "spider_calibration_weight": 0.5, "aux_loss_scale": 0.0},
        # Pure LM
        {"name": "pure_lm", "mtp_weight": 0.0, "jepa_weight": 0.0, "ghost_weight": 0.0, "spider_calibration_weight": 0.0, "aux_loss_scale": 0.0},
        # LM + MTP only
        {"name": "lm_mtp", "mtp_weight": 0.15, "jepa_weight": 0.0, "ghost_weight": 0.0, "spider_calibration_weight": 0.0, "aux_loss_scale": 0.0},
        # LM + JEPA only
        {"name": "lm_jepa", "mtp_weight": 0.0, "jepa_weight": 0.10, "ghost_weight": 0.0, "spider_calibration_weight": 0.0, "aux_loss_scale": 0.0},
        # LM + Ghost only
        {"name": "lm_ghost", "mtp_weight": 0.0, "jepa_weight": 0.0, "ghost_weight": 0.07, "spider_calibration_weight": 0.0, "aux_loss_scale": 0.0},
        # Spider heavy + JEPA moderate
        {"name": "spider2_jepa01", "mtp_weight": 0.10, "jepa_weight": 0.10, "ghost_weight": 0.03, "spider_calibration_weight": 2.0, "aux_loss_scale": 0.0},
        # Low ghost + high spider + high jepa
        {"name": "triple_threat", "mtp_weight": 0.10, "jepa_weight": 0.20, "ghost_weight": 0.03, "spider_calibration_weight": 1.0, "aux_loss_scale": 0.0},
        # MTP off, rest boosted
        {"name": "no_mtp_boost", "mtp_weight": 0.0, "jepa_weight": 0.15, "ghost_weight": 0.10, "spider_calibration_weight": 0.5, "aux_loss_scale": 0.0},
        # Ghost off, rest boosted
        {"name": "no_ghost_boost", "mtp_weight": 0.20, "jepa_weight": 0.15, "ghost_weight": 0.0, "spider_calibration_weight": 0.5, "aux_loss_scale": 0.0},
        # Minimal aux (just hints)
        {"name": "hints_only", "mtp_weight": 0.05, "jepa_weight": 0.03, "ghost_weight": 0.03, "spider_calibration_weight": 0.10, "aux_loss_scale": 0.0},
        # Spider ultra
        {"name": "spider_ultra", "mtp_weight": 0.10, "jepa_weight": 0.05, "ghost_weight": 0.03, "spider_calibration_weight": 5.0, "aux_loss_scale": 0.0},
    ]

    # Take requested number (pad with combinatorial if needed)
    if n <= len(priority):
        return priority[:n]

    result = list(priority)
    # Add grid-search combos for remaining slots
    grid = list(itertools.product(
        axes["mtp_weight"][1::2],    # 0.05, 0.15, 0.40
        axes["jepa_weight"][1::2],   # 0.03, 0.10, 0.40
        axes["ghost_weight"][1::2],  # 0.03, 0.14
    ))
    for i, (mtp_w, jepa_w, ghost_w) in enumerate(grid):
        if len(result) >= n:
            break
        result.append({
            "name": f"grid_{i}",
            "mtp_weight": mtp_w,
            "jepa_weight": jepa_w,
            "ghost_weight": ghost_w,
            "spider_calibration_weight": 0.5,
            "aux_loss_scale": 0.0,
        })

    return result[:n]


def print_report(scenarios: list[dict]) -> None:
    """Ranked table of scenarios."""
    ranked = sorted(scenarios, key=lambda s: s.get("cos_total_lm", s["lm_alignment"]), reverse=True)

    print()
    print("=" * 110)
    print("  WEIGHT SIMULATION REPORT — F51 Darwin-X")
    print(f"  Scenarios: {len(scenarios)}")
    print("=" * 110)
    print()
    print(f"{'#':>3s}  {'name':<22s}  {'mtp':>5s}  {'jepa':>5s}  {'gho':>5s}  {'spi':>5s}  {'cos_tot_lm':>10s}  {'lm_align':>9s}  {'conf':>4s}  {'||g||':>10s}")
    print("-" * 105)

    for rank, s in enumerate(ranked, 1):
        w = s["weights"]
        name = s.get("name", "?")
        print(
            f"{rank:>3d}  {name:<22s}  "
            f"{w.get('mtp_weight',0):>5.2f}  {w.get('jepa_weight',0):>5.2f}  "
            f"{w.get('ghost_weight',0):>5.2f}  {w.get('spider_calibration_weight',0):>5.2f}  "
            f"{s.get('cos_total_lm', 0):>+10.4f}  "
            f"{s.get('lm_alignment', 0):>+9.4f}  {s.get('conflicts', 0):>4d}  "
            f"{s.get('total_grad_norm',0):>10.2e}"
        )

    print("-" * 105)
    print()

    best = ranked[0]
    print(f"BEST COMBINED ALIGNMENT:  {best.get('name','?')}  cos(total, lm)={best.get('cos_total_lm',0):+.4f}  conflicts={best.get('conflicts',0)}")
    print(f"WORST COMBINED ALIGNMENT: {ranked[-1].get('name','?')}  cos(total, lm)={ranked[-1].get('cos_total_lm',0):+.4f}  conflicts={ranked[-1].get('conflicts',0)}")
    print("=" * 110)


def main() -> int:
    args = parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base_cfg = DarwinXConfig.from_mapping(yaml.safe_load(open(args.config)))

    print(f"Building base model for weight simulation...")
    base_model = DarwinXModel(base_cfg)
    base_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    base_model.to(args.device)

    scenarios_w = generate_scenarios(args.scenarios)
    print(f"Testing {len(scenarios_w)} scenarios...")

    results = []
    for i, weights in enumerate(scenarios_w):
        name = weights.pop("name", f"scenario_{i}")
        print(f"  [{i+1:>2d}/{len(scenarios_w)}] {name}...", end=" ", flush=True)
        r = evaluate_scenario(
            base_model, base_cfg, args.token_bin, weights,
            seq_len=args.seq_len, seed=args.seed, device=args.device,
        )
        r["name"] = name
        results.append(r)
        print(f"align={r['lm_alignment']:+.4f}  conflicts={r['conflicts']}")

    print_report(results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "config": str(Path(args.config).resolve()),
            "n_scenarios": len(results),
            "scenarios": results,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
