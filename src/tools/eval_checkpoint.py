#!/usr/bin/env python3
"""Deterministic checkpoint evaluator — no GPU required.

Loads a Darwin-X checkpoint, runs fixed prompts with temperature=0,
and measures perplexity on clean validation corpora.

Usage:
  python -m tools.eval_checkpoint --checkpoint checkpoints/audit_frozen/darwin_trained_v3.pt
  python -m tools.eval_checkpoint --checkpoint checkpoints/audit_frozen/organism_cycle_239.pt
  python -m tools.eval_checkpoint --tournament  # evaluates all frozen checkpoints

Output: workspace/runtime/evaluations/<checkpoint_name>.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

import torch
import yaml

from f51_darwin.darwin_x import DarwinXConfig, DarwinXModel
from f51_darwin.tokenizer import F51BPETokenizer


# ═══════════════════════════════════════════════════════════════════════════
# Fixed prompts  (Portuguese + English)
# ═══════════════════════════════════════════════════════════════════════════

IDENTITY_PROMPTS = [
    "Quem te criou?",
    "Qual o seu nome?",
    "Qual o seu proposito?",
    "Soli Deo Gloria significa",
    "O que voce sabe sobre o Brasil?",
]

KNOWLEDGE_PROMPTS = [
    "A capital do Brasil eh",
    "A agua ferve a",
    "O sol eh uma",
    "Explain what a transformer is in one sentence.",
    "The meaning of life is",
]

GENERATION_CONFIG = {
    "max_tokens": 64,
    "temperature": 0.0,    # deterministic
    "top_p": 1.0,           # no filtering
}


# ═══════════════════════════════════════════════════════════════════════════
# Perplexity on validation text
# ═══════════════════════════════════════════════════════════════════════════

def compute_ppl(
    model: DarwinXModel,
    tokenizer: F51BPETokenizer,
    texts: list[str],
    max_chars: int = 50_000,
) -> dict[str, Any]:
    """Compute perplexity on raw text using the model's LM head.

    Returns dict with ppl, total_tokens, loss.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    for text in texts:
        ids = tokenizer.encode(text)[:4096]
        if len(ids) < 2:
            continue
        batch = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=dtype in {torch.float16, torch.bfloat16}):
                output = model(batch, labels=batch, heartbeat=False)
        if output.lm_loss is not None and torch.isfinite(output.lm_loss):
            total_loss += float(output.lm_loss.cpu()) * (len(ids) - 1)
            total_tokens += len(ids) - 1

        if total_tokens * 4 > max_chars:  # approximate char limit
            break

    if total_tokens == 0:
        return {"ppl": float("inf"), "loss": float("inf"), "tokens": 0}

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20.0))
    return {"ppl": round(ppl, 2), "loss": round(avg_loss, 4), "tokens": total_tokens}


# ═══════════════════════════════════════════════════════════════════════════
# Lexical diversity
# ═══════════════════════════════════════════════════════════════════════════

def lexical_diversity(texts: list[str]) -> dict[str, float]:
    import re
    all_words = []
    for t in texts:
        words = re.findall(r"\w{2,}", t.lower())
        all_words.extend(words)
    if not all_words:
        return {"type_token_ratio": 0.0, "total_words": 0, "unique_words": 0}
    return {
        "type_token_ratio": round(len(set(all_words)) / len(all_words), 4),
        "total_words": len(all_words),
        "unique_words": len(set(all_words)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main eval
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_checkpoint(
    checkpoint_path: Path,
    config_path: Path | None = None,
    tokenizer_dir: Path | None = None,
    val_texts_pt: list[str] | None = None,
    val_texts_en: list[str] | None = None,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint, evaluate, return results dict."""
    if config_path is None:
        config_path = ROOT / "src" / "configs" / "darwin_x_600m.yaml"
    if tokenizer_dir is None:
        tokenizer_dir = ROOT / "tokenizer" / "f51_bpe_80k"

    print(f"  Loading {checkpoint_path.name}...")

    # Config
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["heartbeat_enabled"] = False   # no heartbeat during eval
    cfg = DarwinXConfig.from_mapping(raw)

    # Model
    model = DarwinXModel(cfg)
    dev = torch.device(device)
    if device == "cpu":
        model = model.to(dtype=torch.float32, device=dev)
    else:
        model = model.to(dtype=torch.bfloat16, device=dev)

    # Load weights
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    incompatible = model.load_state_dict(state, strict=False)
    model.eval()

    # Tokenizer
    tokenizer = F51BPETokenizer.load(tokenizer_dir)

    # Extract metadata
    ts = ckpt.get("training_state", {})
    org = ckpt.get("organism", {})
    rep = org.get("report", {})

    print(f"    cycle={ts.get('cycle', '?')}  step={ts.get('step', '?')}  "
          f"missing={len(incompatible.missing_keys)}  unexpected={len(incompatible.unexpected_keys)}")

    # ── Generation (deterministic) ──
    all_prompts = IDENTITY_PROMPTS + KNOWLEDGE_PROMPTS
    generations = []
    for prompt in all_prompts:
        ids = tokenizer.encode(prompt)
        context = torch.tensor([ids[-cfg.context_length:]], dtype=torch.long, device=dev)
        generated: list[int] = []
        with torch.no_grad():
            for _ in range(GENERATION_CONFIG["max_tokens"]):
                with torch.amp.autocast(
                    device_type=dev.type,
                    dtype=next(model.parameters()).dtype,
                    enabled=dev.type == "cuda",
                ):
                    output = model(context, heartbeat=False)
                logits = output.logits[:, -1, :].float()
                next_token = int(torch.argmax(logits, dim=-1).item())
                if next_token == tokenizer.eos_id:
                    break
                generated.append(next_token)
                context = torch.cat([context, torch.tensor([[next_token]], dtype=torch.long, device=dev)], dim=1)
        try:
            text = tokenizer.decode(generated, skip_special=True)
        except TypeError:
            text = tokenizer.decode(generated)
        generations.append({"prompt": prompt, "text": text, "token_count": len(generated)})

    # ── Perplexity PT ──
    pt_texts = val_texts_pt
    if pt_texts is None:
        # Fallback: use prompts as tiny validation
        pt_texts = [
            "O Brasil é um país localizado na América do Sul. Sua capital é Brasília. "
            "A língua oficial é o português. O país possui uma rica diversidade cultural e natural.",
            "A inteligência artificial é um campo da ciência da computação que busca criar "
            "sistemas capazes de realizar tarefas que normalmente exigiriam inteligência humana.",
        ]
    ppl_pt = compute_ppl(model, tokenizer, pt_texts)

    # ── Perplexity EN ──
    en_texts = val_texts_en
    if en_texts is None:
        en_texts = [
            "Artificial intelligence is a field of computer science that aims to create "
            "systems capable of performing tasks that would normally require human intelligence.",
            "The transformer architecture uses self-attention mechanisms to process sequential "
            "data without recurrence, enabling parallel computation across input positions.",
        ]
    ppl_en = compute_ppl(model, tokenizer, en_texts)

    # ── Lexical diversity ──
    gen_texts = [g["text"] for g in generations]
    lex_div = lexical_diversity(gen_texts)

    # ── Result ──
    result = {
        "checkpoint": str(checkpoint_path.name),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "config": cfg.model_name,
        "cycle": ts.get("cycle", org.get("cycle", None)),
        "step": ts.get("step", org.get("total_steps", None)),
        "version": ckpt.get("version", None),
        "device": device,
        "ppl_pt": ppl_pt,
        "ppl_en": ppl_en,
        "lexical_diversity": lex_div,
        "generations": generations,
        "eval_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic checkpoint evaluator")
    parser.add_argument("--checkpoint", default=None, help="Single checkpoint to evaluate")
    parser.add_argument("--tournament", action="store_true", help="Evaluate all frozen checkpoints")
    parser.add_argument("--config", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output-dir", default="workspace/runtime/evaluations")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen_dir = ROOT / "checkpoints" / "audit_frozen"

    if args.tournament:
        # Find all .pt checkpoints in audit_frozen
        checkpoints = sorted(
            [p for p in frozen_dir.glob("*.pt") if "organism_cycle" in p.name or "darwin_trained" in p.name or "darwin_merged" in p.name],
            key=lambda p: (
                # Sort: v3, v4, then cycles numerically
                0 if "darwin_trained" in p.name else
                1 if "darwin_merged" in p.name else
                int(p.stem.split("_")[-1]) + 10
            ),
        )
    elif args.checkpoint:
        checkpoints = [Path(args.checkpoint)]
    else:
        parser.error("Use --checkpoint or --tournament")

    if not checkpoints:
        print("No checkpoints found.")
        return

    print(f"Evaluating {len(checkpoints)} checkpoint(s) on device={args.device}")
    print()

    results = []
    for cp in checkpoints:
        if not cp.exists():
            print(f"  SKIP {cp.name}: file not found")
            continue
        try:
            result = evaluate_checkpoint(
                cp,
                config_path=Path(args.config) if args.config else None,
                tokenizer_dir=Path(args.tokenizer) if args.tokenizer else None,
                device=args.device,
            )
            results.append(result)

            # Save individual result
            out_path = output_dir / f"{cp.stem}.json"
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            print(f"  -> {out_path.name}")
            print(f"     PPL PT={result['ppl_pt']['ppl']}  PPL EN={result['ppl_en']['ppl']}  "
                  f"TTR={result['lexical_diversity']['type_token_ratio']}")
        except Exception as exc:
            print(f"  FAIL {cp.name}: {type(exc).__name__}: {exc}")
        print()

    # Save tournament summary
    if len(results) > 1:
        summary_path = output_dir / "tournament_summary.json"
        summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"Tournament summary: {summary_path}")

        # Quick ranking
        print()
        print("=== PPL PT RANKING ===")
        ranked = sorted(results, key=lambda r: r["ppl_pt"]["ppl"])
        for i, r in enumerate(ranked):
            print(f"  {i+1}. {r['checkpoint']:35s}  PPL={r['ppl_pt']['ppl']:>8.2f}  "
                  f"EN={r['ppl_en']['ppl']:>8.2f}  TTR={r['lexical_diversity']['type_token_ratio']}")


if __name__ == "__main__":
    main()
