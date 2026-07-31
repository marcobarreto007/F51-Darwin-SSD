#!/usr/bin/env python3
"""Per-layer identity circuit discovery via activation contrast.

Runs identity vs knowledge prompts through EVERY layer's internal taps
(norm1, norm2, attention output, FFN output) and finds which layers
have the highest identity-vs-knowledge contrast — that's where the
identity circuit lives.
"""

import json, os, time
from pathlib import Path

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer
from f51_darwin.transplant_16b.cli import DEFAULT_SOURCE_ROOT
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.hashing import sha256_file, atomic_json_write

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt"
OUTPUT = ROOT / "workspace/runtime/identity-test"

IDENTITY = [
    "Who are you?", "What is your name?", "Tell me about yourself.",
    "What model are you?", "Introduce yourself.", "Who created you?",
    "What can you do?", "Are you an AI?", "What is your purpose?",
    "How were you trained?",
]

KNOWLEDGE = [
    "Qual e a capital do Brasil?", "What is the capital of France?",
    "Quanto e 2 + 2?", "What is the speed of light?",
    "Who wrote Romeo and Juliet?", "Qual e a formula da agua?",
    "What year did World War 2 end?", "Quem descobriu o Brasil?",
    "What is the boiling point of water?", "How many continents are there?",
    "What is photosynthesis?", "What is the largest planet?",
    "Who painted the Mona Lisa?", "What language is spoken in Japan?",
]


def get_activation(model, tokenizer, text, tap_module_path):
    """Get the mean absolute activation at a specific tap point."""
    # Resolve module
    parts = tap_module_path.split(".")
    if parts[0] == "norm":
        mod = model.norm
    elif parts[0] == "blocks":
        li = int(parts[1])
        block = model.blocks[li]
        rest = ".".join(parts[2:])
        if rest == "norm1":
            mod = block.norm1
        elif rest == "norm2":
            mod = block.norm2
        elif rest == "ssd.out_proj":
            mod = block.ssd.out_proj
        elif rest == "moe":
            mod = block.moe
        elif rest == "ffn":
            mod = block.ffn
        else:
            return None
    else:
        return None

    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
    ids = ids[:, :32].to(device="cuda:0")
    cap = {}
    def h(mod, inp, out): cap["h"] = out
    handle = mod.register_forward_hook(h)
    try:
        with torch.no_grad():
            model(ids, heartbeat=False)
        act = cap["h"].float()
        # Return mean abs activation as a single scalar per channel
        if act.ndim == 3:
            return act.abs().mean(dim=(0, 1))  # [d_model] or [out_dim]
        elif act.ndim == 2:
            return act.abs().mean(dim=0)
        else:
            return act.abs()
    finally:
        handle.remove()


def main():
    print("=" * 68)
    print("PER-LAYER IDENTITY CONTRAST DISCOVERY")
    print("=" * 68)

    tokenizer = AutoTokenizer.from_pretrained(str(DEFAULT_SOURCE_ROOT), local_files_only=True)
    print(f"Tokenizer: {tokenizer.__class__.__name__}")

    print("Loading model...")
    t0 = time.perf_counter()
    payload = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    config = DarwinXConfig.from_mapping(payload["config"])
    prev = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
    model = DarwinXModel(config); torch.set_default_dtype(prev)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval(); model.to(device="cuda:0", dtype=torch.bfloat16)
    d_model = config.d_model
    n_layers = config.n_layers
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s  d_model={d_model}  layers={n_layers}")

    # Define tap points
    tap_points = ["norm"]
    for li in range(n_layers):
        tap_points.append(f"blocks.{li}.norm1")
        tap_points.append(f"blocks.{li}.norm2")

    print(f"\nScanning {len(tap_points)} tap points across {n_layers} layers...")
    results = []

    for tp in tap_points:
        # Get identity activations
        id_acts = []
        for p in IDENTITY:
            act = get_activation(model, tokenizer, p, tp)
            if act is not None:
                id_acts.append(act)
        if not id_acts:
            continue

        # Get knowledge activations
        kn_acts = []
        for p in KNOWLEDGE:
            act = get_activation(model, tokenizer, p, tp)
            if act is not None:
                kn_acts.append(act)

        id_mean = torch.stack(id_acts).mean(dim=0)
        kn_mean = torch.stack(kn_acts).mean(dim=0)
        diff = (id_mean - kn_mean).abs()

        # Contrast metrics
        total_contrast = diff.sum().item()
        max_contrast = diff.max().item()
        n_channels = diff.shape[0]
        mean_contrast = total_contrast / n_channels

        # Top contrast channels
        top10 = diff.argsort(descending=True)[:10].tolist()

        # Parse layer from tap
        if tp == "norm":
            layer = -1
            tap_type = "final_norm"
        else:
            parts = tp.split(".")
            layer = int(parts[1])
            tap_type = parts[2] if len(parts) > 2 else "norm2"

        results.append({
            "tap": tp,
            "layer": layer,
            "tap_type": tap_type,
            "channels": n_channels,
            "total_contrast": round(total_contrast, 2),
            "mean_contrast": round(mean_contrast, 6),
            "max_contrast": round(max_contrast, 4),
            "top10_channels": top10,
            "top10_values": [round(float(diff[ch]), 4) for ch in top10],
        })

        label = f"L{layer:02d}" if layer >= 0 else "FINAL"
        print(f"  [{label}] {tap_type:<12}  mean_contrast={mean_contrast:.6f}  max={max_contrast:.2f}  top10={top10[:5]}")

    # Sort by contrast
    results.sort(key=lambda r: r["mean_contrast"], reverse=True)

    print(f"\n=== TOP 10 TAP POINTS BY IDENTITY CONTRAST ===")
    for i, r in enumerate(results[:10]):
        label = f"L{r['layer']:02d}" if r['layer'] >= 0 else "FINAL"
        print(f"  {i+1}. [{label}] {r['tap_type']:<12}  mean={r['mean_contrast']:.6f}  max={r['max_contrast']:.4f}")

    # Best layer for identity circuit
    best = results[0]
    print(f"\nBest identity tap: {best['tap']} (layer={best['layer']}, {best['tap_type']})")
    print(f"  Top identity-specific channels: {best['top10_channels'][:10]}")

    # Save
    report = {
        "schema": "per-layer-identity-contrast-v1",
        "checkpoint_sha256": sha256_file(CKPT),
        "n_layers": n_layers,
        "d_model": d_model,
        "tap_points_scanned": len(results),
        "top_contrast_taps": results[:10],
        "all_results": results,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT / "per-layer-identity-contrast.json"
    atomic_json_write(report, out_path)
    print(f"\nReport: {out_path}")
    print(f"IDENTITY_LAYER_FOUND: {best['tap']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
