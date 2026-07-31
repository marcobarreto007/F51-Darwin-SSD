#!/usr/bin/env python
"""
Converte checkpoint legacy (norm1/norm2 + SSD aninhado + MoE) → F51DarwinModel atual.

Legacy structure (unified checkpoint):
    blocks.X.norm1, blocks.X.norm2
    blocks.X.ssd.norm_mixer
    blocks.X.ssd.ssm.*
    blocks.X.ssd.norm_ff, blocks.X.ssd.ff.*
    blocks.X.moe.router.*, blocks.X.moe.experts.*

New structure (F51DarwinModel):
    blocks.X.norm_mixer
    blocks.X.ssm.*
    blocks.X.norm_ff
    blocks.X.ff.*

Usage:
    python -m tools.convert_legacy_checkpoint checkpoints/unified/step_0007900.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]

from f51_darwin.config import DarwinConfig
from f51_darwin.model import F51DarwinModel


def detect_architecture(state_dict: dict) -> str:
    """Detect if checkpoint is legacy or new format."""
    keys = list(state_dict.keys())

    # Check for legacy markers
    has_norm1_norm2 = any("norm1" in k or "norm2" in k for k in keys)
    has_nested_ssd = any("ssd.ssm" in k for k in keys)
    has_moe_per_block = any("moe.router" in k for k in keys)

    if has_norm1_norm2 or has_nested_ssd:
        return "legacy"
    elif has_moe_per_block:
        return "moe"
    else:
        return "standard"


def convert_legacy_to_new(old_state_dict: dict, with_moe: bool = False) -> dict:
    """Convert legacy state dict to new F51DarwinModel format.

    Legacy SSD block:
        blocks.X.norm1, blocks.X.norm2 (unused, skip)
        blocks.X.ssd.norm_mixer → blocks.X.norm_mixer
        blocks.X.ssd.ssm.* → blocks.X.ssm.*
        blocks.X.ssd.norm_ff → blocks.X.norm_ff
        blocks.X.ssd.ff.* → blocks.X.ff.*
        blocks.X.moe.* → blocks.X.moe.experts.* (if with_moe)

    Legacy Attention block:
        blocks.X.norm1, blocks.X.norm2 (unused, skip)
        blocks.X.attention.norm_attn → blocks.X.norm_attn
        blocks.X.attention.qkv.* → blocks.X.qkv.*
        blocks.X.attention.proj.* → blocks.X.proj.*
        blocks.X.attention.norm_ff → blocks.X.norm_ff
        blocks.X.attention.ff.* → blocks.X.ff.*
        blocks.X.moe.* → blocks.X.moe.experts.* (if with_moe)

    Legacy MoE structure:
        blocks.X.moe.router.router.weight → blocks.X.moe.router.weight
        blocks.X.moe.router.expert_bias → blocks.X.moe.expert_bias
        blocks.X.moe.experts.Y.* → blocks.X.moe.experts.Y.*
    """
    new_state_dict = {}

    # Copy unchanged layers
    for key in ["token_embedding.weight", "norm.weight"]:
        if key in old_state_dict:
            new_state_dict[key] = old_state_dict[key]

    # Handle weight tying for lm_head
    if "lm_head.weight" in old_state_dict:
        new_state_dict["lm_head.weight"] = old_state_dict["lm_head.weight"]

    # Convert each block
    block_keys = [k for k in old_state_dict.keys() if k.startswith("blocks.")]
    block_indices = sorted(set(
        int(k.split(".")[1]) for k in block_keys if k.split(".")[1].isdigit()
    ))

    for idx in block_indices:
        prefix = f"blocks.{idx}."

        # Check if this is SSD or Attention block
        has_ssd = any(f"{prefix}ssd." in k for k in old_state_dict.keys())
        has_attention = any(f"{prefix}attention." in k for k in old_state_dict.keys())
        has_moe = any(f"{prefix}moe." in k for k in old_state_dict.keys())

        if has_ssd:
            # Convert SSD block: norm_mixer, ssm.*, norm_ff, ff.*
            if f"{prefix}ssd.norm_mixer.weight" in old_state_dict:
                new_state_dict[f"{prefix}norm_mixer.weight"] = old_state_dict[f"{prefix}ssd.norm_mixer.weight"]

            # Copy SSM layers
            for key in old_state_dict.keys():
                if key.startswith(f"{prefix}ssd.ssm."):
                    new_key = key.replace(f"{prefix}ssd.ssm.", f"{prefix}ssm.")
                    new_state_dict[new_key] = old_state_dict[key]

            # Copy FFN layers
            if f"{prefix}ssd.norm_ff.weight" in old_state_dict:
                new_state_dict[f"{prefix}norm_ff.weight"] = old_state_dict[f"{prefix}ssd.norm_ff.weight"]
            for key in ["w12.weight", "w12.bias", "out.weight", "out.bias"]:
                old_key = f"{prefix}ssd.ff.{key}"
                if old_key in old_state_dict:
                    new_state_dict[f"{prefix}ff.{key}"] = old_state_dict[old_key]

        elif has_attention:
            # Convert Attention block: norm_attn, qkv.*, proj.*, norm_ff, ff.*
            if f"{prefix}attention.norm_attn.weight" in old_state_dict:
                new_state_dict[f"{prefix}norm_attn.weight"] = old_state_dict[f"{prefix}attention.norm_attn.weight"]

            # Copy QKV layers
            for key in ["qkv.weight", "qkv.bias"]:
                old_key = f"{prefix}attention.{key}"
                if old_key in old_state_dict:
                    new_state_dict[f"{prefix}{key}"] = old_state_dict[old_key]

            # Copy Projection layers
            for key in ["proj.weight", "proj.bias"]:
                old_key = f"{prefix}attention.{key}"
                if old_key in old_state_dict:
                    new_state_dict[f"{prefix}{key}"] = old_state_dict[old_key]

            # Copy FFN layers
            if f"{prefix}attention.norm_ff.weight" in old_state_dict:
                new_state_dict[f"{prefix}norm_ff.weight"] = old_state_dict[f"{prefix}attention.norm_ff.weight"]
            for key in ["w12.weight", "w12.bias", "out.weight", "out.bias"]:
                old_key = f"{prefix}attention.ff.{key}"
                if old_key in old_state_dict:
                    new_state_dict[f"{prefix}ff.{key}"] = old_state_dict[old_key]

        # Skip norm1/norm2 (legacy layers not used in new arch)

        # Convert MoE layers if requested
        if with_moe and has_moe:
            # Router weights: router.router.weight → router.weight
            if f"{prefix}moe.router.router.weight" in old_state_dict:
                new_state_dict[f"{prefix}moe.router.weight"] = old_state_dict[f"{prefix}moe.router.router.weight"]

            # Expert bias: router.expert_bias → expert_bias
            if f"{prefix}moe.router.expert_bias" in old_state_dict:
                new_state_dict[f"{prefix}moe.expert_bias"] = old_state_dict[f"{prefix}moe.router.expert_bias"]

            # Router buffers: usage_count, last_used_step
            if f"{prefix}moe.router.expert_usage_count" in old_state_dict:
                new_state_dict[f"{prefix}moe.expert_usage_count"] = old_state_dict[f"{prefix}moe.router.expert_usage_count"]
            if f"{prefix}moe.router.expert_last_used_step" in old_state_dict:
                new_state_dict[f"{prefix}moe.expert_last_used_step"] = old_state_dict[f"{prefix}moe.router.expert_last_used_step"]

            # Expert weights: experts.Y.gate_proj.weight → experts.Y.gate_proj.weight
            for key in old_state_dict.keys():
                if key.startswith(f"{prefix}moe.experts."):
                    # Remove "moe." prefix to match new structure
                    new_key = key.replace(f"{prefix}moe.experts.", f"{prefix}moe.experts.")
                    new_state_dict[new_key] = old_state_dict[key]

    # Copy router weights if present (DynamicDepthRouter)
    for key in old_state_dict.keys():
        if key.startswith("router."):
            new_state_dict[key] = old_state_dict[key]

    return new_state_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert legacy checkpoint to new format")
    parser.add_argument("checkpoint", type=Path, help="Path to legacy checkpoint")
    parser.add_argument("--output", type=Path, help="Output path (default: checkpoints/converted/*.pt)")
    parser.add_argument("--verify", action="store_true", help="Verify conversion by loading model")
    parser.add_argument("--with-moe", action="store_true", help="Convert MoE layers (requires experts_enabled=True)")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1

    print(f"Loading: {checkpoint_path}")
    print(f"Size: {checkpoint_path.stat().st_size / 1e9:.2f} GB")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, weights_only=False)

    # Detect architecture
    arch = detect_architecture(ckpt["model_state_dict"])
    print(f"Detected architecture: {arch}")

    if arch != "legacy":
        print(f"Warning: Checkpoint is not legacy format (detected: {arch})")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != "y":
            return 1

    # Convert state dict
    with_moe = args.with_moe
    print(f"Converting state dict (with_moe={with_moe})...")
    new_state_dict = convert_legacy_to_new(ckpt["model_state_dict"], with_moe=with_moe)

    print(f"Original keys: {len(ckpt['model_state_dict'])}")
    print(f"Converted keys: {len(new_state_dict)}")

    # Build new checkpoint
    new_ckpt = {
        "version": 1,
        "model_state_dict": new_state_dict,
        "config": ckpt["config"],
        "lineage": ckpt.get("lineage", "F51 Darwin-SSD - converted from legacy"),
        "metrics": ckpt.get("metrics", {}),
        "training_state": ckpt.get("training_state", {}),
    }

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        suffix = "_moe" if with_moe else ""
        output_path = ROOT / "checkpoints" / "converted" / f"{checkpoint_path.stem}{suffix}.pt"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save converted checkpoint
    print(f"Saving to: {output_path}")
    torch.save(new_ckpt, output_path, _use_new_zipfile_serialization=False)

    print(f"Converted checkpoint size: {output_path.stat().st_size / 1e9:.2f} GB")

    # Verify if requested
    if args.verify:
        print("\nVerifying conversion...")
        try:
            cfg = DarwinConfig(**ckpt["config"])

            # Determine number of experts from checkpoint
            num_experts = 12  # Default from unified checkpoint
            if with_moe:
                moe_config = {
                    "num_experts": num_experts,
                    "experts_per_token": 2,
                    "expert_hidden_mult": 4,
                    "use_nitro_tiering": True,
                    "nitro_gpu_capacity": 4,
                }
                print(f"Loading model with experts_enabled=True, num_experts={num_experts}")
                model = F51DarwinModel(cfg, experts_enabled=True, moe_config=moe_config)
            else:
                model = F51DarwinModel(cfg)

            model.load_state_dict(new_state_dict, strict=True)
            print("✓ Model loaded successfully with strict=True")
            print(f"Total params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
        except Exception as e:
            print(f"✗ Verification failed: {e}")
            print("Trying with strict=False...")
            try:
                if with_moe:
                    model = F51DarwinModel(cfg, experts_enabled=True, moe_config=moe_config)
                else:
                    model = F51DarwinModel(cfg)
                model.load_state_dict(new_state_dict, strict=False)
                print("✓ Model loaded with strict=False (some keys may be missing)")
            except Exception as e2:
                print(f"✗ Verification failed: {e2}")
                return 1

    print("\n✓ Conversion complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
