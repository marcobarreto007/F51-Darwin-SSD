#!/usr/bin/env python3
"""Massive circuit scanner — catalogues every channel across all layers.

Runs pooled |activation| attribution against a diverse tokenized corpus,
stamps each channel with SHA-256, classifies by function, and populates a
machine-readable circuit registry.

ATTRIBUTION CAVEAT: the score is the mean absolute activation of a channel
over the real (non-padding) token positions of a probe set. This measures
channel MAGNITUDE, not causal role — a channel can be large and carry none
of the fact. Labels are therefore a coarse prior, not evidence that a
channel encodes the probed function. Establishing causal role requires
ablation (zero the channel, measure Δ loss), which this scanner does not do.

This is the bridge from manual one-fact ablation to automated genome mapping.
"""

from __future__ import annotations

import argparse, gc, hashlib, json, math, os, time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
from torch import nn

from f51_darwin.circuits.identity import canonical_sha256
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.hashing import sha256_file, atomic_json_write, tensor_sha256

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt"
OUTPUT = ROOT / "workspace/runtime/circuit-catalog"
TOKENIZER_DIR = ROOT / "workspace/00_DONORS/models--HuggingFaceTB--SmolLM2-1.7B-Instruct"
REGISTRY_SCHEMA = "darwin-circuit-catalog-v2"
ATTRIBUTION_METHOD = "pooled_abs_activation_masked"


# ── Circuit record ──────────────────────────────────────────────────────────

@dataclass
class CircuitStamp:
    circuit_id: str              # SHA-256 of (layer, channel_range, weight_hash, function)
    layer: int                   # 0..n_layers-1
    channel_start: int           # first channel index
    channel_count: int           # number of channels in this stamp
    attribution_score: float     # gradient×activation score
    weight_sha256: str           # SHA-256 of the weight slice
    function_label: str          # factual | syntactic | positional | mixed | noise
    function_confidence: float   # 0..1
    donor_checkpoint_sha256: str
    topology_position: str       # e.g. "blocks.16.norm2" or "norm"
    dtype: str
    shape: list[int]
    dependencies: list[str] = field(default_factory=list)  # upstream circuit IDs


# ── Corpus probes ───────────────────────────────────────────────────────────

FACT_PROBES = [
    "A capital da Franca e Paris.",
    "A agua ferve a 100 graus Celsius.",
    "O Sol e uma estrela.",
    "2 + 2 = 4.",
    "O oxigenio e essencial para a respiracao.",
    "A Terra orbita ao redor do Sol.",
    "O DNA contem a informacao genetica.",
    "A velocidade da luz e aproximadamente 300.000 km/s.",
]

SYNTACTIC_PROBES = [
    "O gato preto pulou sobre o muro alto rapidamente.",
    "Se chover amanha, nao irei ao parque.",
    "Ela disse que viria, mas nao apareceu.",
    "Apesar do cansaco, terminou o trabalho.",
    "Quando cheguei, ele ja tinha saido.",
    "Nao sei se vou, depende do tempo.",
    "O livro que comprei ontem e muito interessante.",
    "Ela correu, pulou e gritou de alegria.",
]

POSITIONAL_PROBES = [
    "a a a a a a a a a a a a a a a a",
    "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16",
    "o o o o o o o o o o o o o o o o",
    ". . . . . . . . . . . . . . . .",
]


# ── Tokenization ────────────────────────────────────────────────────────────

def load_tokenizer(tokenizer_dir: Path):
    """Load the donor tokenizer from a local HF snapshot directory."""
    from transformers import AutoTokenizer

    snapshots = tokenizer_dir / "snapshots"
    if snapshots.exists():
        candidates = sorted(snapshots.iterdir())
        if not candidates:
            raise FileNotFoundError(f"No snapshot under {snapshots}")
        tokenizer_dir = candidates[0]
    return AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)


def encode_batch(
    tokenizer,
    probes: list[str],
    max_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize probes into a padded batch plus a validity mask.

    Returns:
        input_ids: [B, T] padded token ids
        mask:      [B, T] 1.0 on real tokens, 0.0 on padding
    """
    encoded = [
        tokenizer.encode(text, add_special_tokens=False)[:max_len]
        for text in probes
    ]
    width = max(len(ids) for ids in encoded)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    input_ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(encoded), width), dtype=torch.float32)
    for row, ids in enumerate(encoded):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[row, : len(ids)] = 1.0

    return input_ids.to(device), mask.to(device)


# ── Attribution engine ──────────────────────────────────────────────────────

def resolve_tap_module(model: nn.Module, tap_point: str) -> nn.Module:
    """Resolve a tap point string to the module whose output we hook."""
    if tap_point == "norm":
        return model.norm
    if tap_point.startswith("blocks."):
        parts = tap_point.split(".")
        layer_idx = int(parts[1])
        sub = parts[2] if len(parts) > 2 else "norm2"
        return getattr(model.blocks[layer_idx], sub)
    raise ValueError(f"Unknown tap point: {tap_point}")


def compute_channel_attributions(
    model: nn.Module,
    tap_point: str,  # "norm" for final norm, "blocks.{i}.norm2" for a block
    probes: list[str],
    tokenizer,
    batch_size: int = 2,
    max_len: int = 32,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, list[tuple[int, float]]]:
    """Pooled |activation| attribution per channel at a tap point.

    The probes are really tokenized; padding positions are masked out so a
    short probe does not dilute the mean. See the module docstring for what
    this score does and does not establish.

    Returns:
        scores: [d_model] attribution per channel
        ranked: list of (channel_index, score) sorted descending
    """
    d_model = model.config.d_model
    if device is None:
        device = next(model.parameters()).device

    module = resolve_tap_module(model, tap_point)

    scores = torch.zeros(d_model, dtype=torch.float32, device="cpu")
    total_tokens = 0.0

    for i in range(0, len(probes), batch_size):
        batch_probes = probes[i:i + batch_size]
        input_ids, mask = encode_batch(tokenizer, batch_probes, max_len, device)

        captured: dict[str, torch.Tensor] = {}

        def hook(mod, inp, out):
            captured["hidden"] = out

        handle = module.register_forward_hook(hook)
        try:
            with torch.no_grad():
                model(input_ids, heartbeat=False)
            hidden = captured["hidden"].detach().float()  # [B, T, d_model]

            # Masked sum of |activation| over real token positions only.
            weighted = hidden.abs() * mask.unsqueeze(-1)
            scores += weighted.sum(dim=(0, 1)).cpu()
            total_tokens += float(mask.sum())
        finally:
            handle.remove()

    scores = scores / max(total_tokens, 1.0)
    ranked = sorted([(i, float(scores[i])) for i in range(d_model)],
                     key=lambda x: x[1], reverse=True)
    return scores, ranked


# ── Channel classifier ──────────────────────────────────────────────────────

def classify_channels(
    model: nn.Module,
    tap_point: str,
    tokenizer,
    top_k: int = 0,
) -> list[dict[str, Any]]:
    """Classify channels by which probe family drives them hardest.

    top_k = 0 keeps every channel; otherwise only the top_k by combined
    factual+syntactic score are returned.

    Returns list of {channel, score_fact, score_syntactic, score_positional, label}.
    """
    d_model = model.config.d_model
    device = next(model.parameters()).device

    def attr(probes: list[str]) -> torch.Tensor:
        scores, _ = compute_channel_attributions(
            model, tap_point, probes, tokenizer, device=device
        )
        return scores

    print(f"    Computing factual attribution ({len(FACT_PROBES)} probes)...")
    fact_scores = attr(FACT_PROBES)
    print(f"    Computing syntactic attribution ({len(SYNTACTIC_PROBES)} probes)...")
    syn_scores = attr(SYNTACTIC_PROBES)
    print(f"    Computing positional attribution ({len(POSITIONAL_PROBES)} probes)...")
    pos_scores = attr(POSITIONAL_PROBES)

    # Normalize each score vector to [0,1]
    def norm(v: torch.Tensor) -> torch.Tensor:
        vmax = v.max()
        return v / vmax if vmax > 0 else v

    f_n = norm(fact_scores)
    s_n = norm(syn_scores)
    p_n = norm(pos_scores)

    # Classify each channel
    classified = []
    for ch in range(d_model):
        f, s, p = float(f_n[ch]), float(s_n[ch]), float(p_n[ch])
        total = f + s + p + 1e-8
        f_r, s_r, p_r = f/total, s/total, p/total

        # Classify by dominant category, confidence = how much it dominates
        scores_map = {"factual": f_r, "syntactic": s_r, "positional": p_r}
        label = max(scores_map, key=scores_map.get)
        conf = max(f_r, s_r, p_r)
        # Low total activation = noise
        if total < 0.01:
            label = "noise"
            conf = 1.0

        classified.append({
            "channel": ch,
            "score_fact": f,
            "score_syntactic": s,
            "score_positional": p,
            "label": label,
            "confidence": round(conf, 4),
        })

    classified.sort(key=lambda x: x["score_fact"] + x["score_syntactic"], reverse=True)
    if top_k > 0:
        classified = classified[:top_k]
    return classified


# ── Stamper ──────────────────────────────────────────────────────────────────

def stamp_channel(
    channel_info: dict[str, Any],
    layer: int,
    tap_point: str,
    checkpoint_sha256: str,
    d_model: int,
    tap_weight: torch.Tensor,
) -> CircuitStamp:
    """Create a cryptographic stamp for a channel.

    weight_sha256 hashes the ACTUAL per-channel parameter at the tap point
    (the RMSNorm gain for this channel), so the stamp is invalidated by any
    real weight change. It is not a placeholder.
    """
    ch = channel_info["channel"]
    identity_payload = {
        "layer": layer,
        "channel": ch,
        "d_model": d_model,
        "tap_point": tap_point,
        "checkpoint_sha256": checkpoint_sha256,
        "label": channel_info["label"],
    }
    circuit_id = canonical_sha256(identity_payload)

    return CircuitStamp(
        circuit_id=circuit_id,
        layer=layer,
        channel_start=ch,
        channel_count=1,
        attribution_score=channel_info["score_fact"],
        weight_sha256=tensor_sha256(tap_weight[ch].reshape(1)),
        function_label=channel_info["label"],
        function_confidence=channel_info["confidence"],
        donor_checkpoint_sha256=checkpoint_sha256,
        topology_position=tap_point,
        dtype=str(tap_weight.dtype).replace("torch.", ""),
        shape=[d_model],
    )


# ── Main scanner ────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Massive circuit scanner")
    parser.add_argument("--checkpoint", default=str(CKPT))
    parser.add_argument("--layers", type=str, default="all",
                       help="Comma-separated layer indices or 'all'")
    parser.add_argument("--top-k", type=int, default=0,
                       help="Stamps to keep per tap point (0 = all channels)")
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deep", action="store_true",
                       help="Also scan attention/FFN internal outputs (4x more tap points)")
    parser.add_argument("--tokenizer", default=str(TOKENIZER_DIR),
                       help="Local HF snapshot dir for the donor tokenizer")
    args = parser.parse_args()

    print("=" * 68)
    print("MASSIVE CIRCUIT SCANNER — F51 Darwin-X")
    print("=" * 68)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"\nLoading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(Path(args.tokenizer))
    print(f"  vocab size: {len(tokenizer)}")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt_sha = sha256_file(args.checkpoint)
    print(f"  SHA-256: {ckpt_sha}")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    config = DarwinXConfig.from_mapping(payload["config"])
    d_model = config.d_model

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = DarwinXModel(config)
    torch.set_default_dtype(prev)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    model.to(device=device, dtype=torch.bfloat16)

    if len(tokenizer) > config.vocab_size:
        raise SystemExit(
            f"Tokenizer vocab {len(tokenizer)} exceeds model vocab "
            f"{config.vocab_size} — probes would produce out-of-range ids."
        )

    # Determine tap points
    deep = getattr(args, "deep", False)
    if args.layers == "all":
        tap_points = ["norm"]                         # final norm before lm_head
        for li in range(config.n_layers):
            tap_points.append(f"blocks.{li}.norm1")   # pre-attention norm
            tap_points.append(f"blocks.{li}.norm2")   # pre-FFN norm
            if deep:
                # Internal taps: attention output projection and FFN output
                tap_points.append(f"blocks.{li}.ssd.out_proj")  # attention output
                # FFN: dense_swiglu or moe
                kind = config.feed_forward_kind
                if kind == "dense_swiglu":
                    tap_points.append(f"blocks.{li}.ffn")       # DenseSwiGLU output
                else:
                    tap_points.append(f"blocks.{li}.moe")       # MoE output
    else:
        indices = [int(x) for x in args.layers.split(",")]
        tap_points = []
        for li in indices:
            tap_points.append(f"blocks.{li}.norm1")
            tap_points.append(f"blocks.{li}.norm2")
            if deep:
                tap_points.append(f"blocks.{li}.ssd.out_proj")
                kind = config.feed_forward_kind
                tap_points.append(f"blocks.{li}.ffn" if kind == "dense_swiglu" else f"blocks.{li}.moe")

    mode = "deep" if deep else "surface"
    print(f"\nTap points: {len(tap_points)} ({mode} scan: norms + {'attention + FFN outputs' if deep else 'norms only'})")
    print(f"Classifying top {args.top_k} channels per tap point")
    print(f"Total channels to scan: {len(tap_points) * d_model}")

    all_stamps: list[dict[str, Any]] = []
    stats = {"factual": 0, "syntactic": 0, "positional": 0, "mixed": 0, "noise": 0}
    start_time = time.time()

    for tp_idx, tap in enumerate(tap_points):
        layer = -1 if tap == "norm" else int(tap.split(".")[1])
        print(f"\n[{tp_idx+1}/{len(tap_points)}] Tap: {tap} (layer={layer})")

        classified = classify_channels(model, tap, tokenizer, top_k=args.top_k)
        tap_weight = resolve_tap_module(model, tap).weight.detach()

        for ch_info in classified:
            stamp = stamp_channel(ch_info, max(0, layer), tap, ckpt_sha,
                                  d_model, tap_weight)
            all_stamps.append({
                "circuit_id": stamp.circuit_id,
                "layer": stamp.layer,
                "channel": stamp.channel_start,
                "label": stamp.function_label,
                "confidence": stamp.function_confidence,
                "attribution": stamp.attribution_score,
                "tap": tap,
                "donor_sha256": stamp.donor_checkpoint_sha256,
            })
            stats[stamp.function_label] = stats.get(stamp.function_label, 0) + 1

        elapsed = time.time() - start_time
        print(f"  {len(classified)} channels classified in {elapsed:.1f}s")
        print(f"  Stats so far: {stats}")

    # Save registry
    registry = {
        "schema": REGISTRY_SCHEMA,
        "attribution_method": ATTRIBUTION_METHOD,
        "attribution_caveat": (
            "Scores are mean |activation| over real token positions. This "
            "measures channel magnitude, not causal role. Labels are a prior, "
            "not evidence that a channel encodes the probed function."
        ),
        "checkpoint_sha256": ckpt_sha,
        "model_name": config.model_name,
        "d_model": d_model,
        "n_layers": config.n_layers,
        "tap_points_scanned": len(tap_points),
        "total_stamps": len(all_stamps),
        "stats": stats,
        "stamps": all_stamps,
        "scan_duration_s": time.time() - start_time,
    }

    registry_path = OUTPUT / "circuit-catalog.json"
    atomic_json_write(registry, registry_path)
    print(f"\nRegistry saved: {registry_path}")
    print(f"Total stamps: {len(all_stamps)}")
    print(f"Stats: {stats}")
    print(f"Duration: {registry['scan_duration_s']:.1f}s")
    print("CIRCUIT_SCAN_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
