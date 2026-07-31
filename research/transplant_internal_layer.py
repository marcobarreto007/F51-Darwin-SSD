#!/usr/bin/env python3
"""Direct Internal Layer Swap Operational Tool — F51 Darwin-X.

Replaces external sidecars by swapping specialized donor layers directly into
the internal stack of the target model (e.g. layers 12, 16, 20 of SmolLM2-1.7B).

Measures:
  1. Relative weight distance ||donor_w - target_w|| / ||target_w|| per layer
  2. Cumulative output KL divergence
  3. Preservation of top-1 logits
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

from f51_darwin.organism.layer_transplant import NativeLayerTransplantManager

DONORS_DIR = ROOT / "workspace" / "00_DONORS"
DEFAULT_TARGET = DONORS_DIR / "models--HuggingFaceTB--SmolLM2-1.7B"
DEFAULT_DONOR = DONORS_DIR / "models--HuggingFaceTB--SmolLM2-1.7B-Instruct"


def snapshot_of(folder: Path) -> Path:
    snapshots = list((folder / "snapshots").iterdir())
    if not snapshots:
        raise FileNotFoundError(f"No snapshot found in {folder}")
    return snapshots[0]


def run_transplant(
    target_path: Path,
    donor_path: Path,
    layers: list[int],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    device_obj = torch.device(device)
    print(f"[*] Target model: {target_path.name}")
    print(f"[*] Donor model:  {donor_path.name}")
    print(f"[*] Swapping internal layers: {layers}")
    print(f"[*] Device:       {device_obj}")

    target_snap = snapshot_of(target_path) if (target_path / "snapshots").exists() else target_path
    donor_snap = snapshot_of(donor_path) if (donor_path / "snapshots").exists() else donor_path

    tokenizer = AutoTokenizer.from_pretrained(str(target_snap), local_files_only=True)
    target_model = AutoModelForCausalLM.from_pretrained(
        str(target_snap), local_files_only=True, dtype=torch.bfloat16
    ).to(device_obj).eval()

    donor_model = AutoModelForCausalLM.from_pretrained(
        str(donor_snap), local_files_only=True, dtype=torch.bfloat16
    ).to(device_obj).eval()

    # Pre-swap reference output for KL calculation
    sample_text = "O raciocinio logico e a base da inteligencia porque"
    ids = tokenizer(sample_text, return_tensors="pt").input_ids.to(device_obj)

    with torch.inference_mode():
        ref_logits = target_model(ids).logits[0, -1, :].float()
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    manager = NativeLayerTransplantManager(target_model)
    layer_results = []

    for layer_idx in layers:
        donor_layer = donor_model.model.layers[layer_idx]
        result = manager.swap_layer(
            donor_layer=donor_layer,
            target_layer_idx=layer_idx,
            donor_layer_idx=layer_idx,
            donor_provenance=f"{donor_path.name}:layer_{layer_idx}",
            trainable=False,
        )
        layer_results.append(result)

    # Post-swap KL calculation
    with torch.inference_mode():
        post_logits = target_model(ids).logits[0, -1, :].float()
        post_log_probs = F.log_softmax(post_logits, dim=-1)
        kl_div = float(F.kl_div(post_log_probs, ref_log_probs, log_target=True, reduction="sum"))

    top_1_match = bool(torch.argmax(ref_logits) == torch.argmax(post_logits))

    print("\n[+] Direct Internal Layer Swap Summary:")
    print(f"    - Swapped Layers:         {layers}")
    for res in layer_results:
        print(f"    - Layer {res.target_layer_idx:2d}: Dist = {res.relative_weight_distance:.6f}, SHA = {res.transplant_sha256[:12]}")
    print(f"    - Cumulative KL Div.:     {kl_div:.6f}")
    print(f"    - Top-1 Token Preserved:  {top_1_match}\n")

    return {
        "status": "NATIVE_LAYER_TRANSPLANT_SUCCESS",
        "swapped_layers": layers,
        "kl_divergence": kl_div,
        "top_1_preserved": top_1_match,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct Internal Layer Swap")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--donor", type=Path, default=DEFAULT_DONOR)
    parser.add_argument("--layers", type=int, nargs="+", default=[12, 16, 20])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_transplant(
        target_path=args.target,
        donor_path=args.donor,
        layers=args.layers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
