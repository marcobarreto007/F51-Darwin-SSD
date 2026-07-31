#!/usr/bin/env python3
"""Inspect a Darwin-X config contract without allocating real weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]

from f51_darwin.darwin_x import DarwinXConfig, DarwinXModel, estimate_darwin_x_parameters


def load_config(path: Path) -> DarwinXConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Darwin-X YAML must be a mapping.")
    return DarwinXConfig.from_mapping(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a Darwin-X YAML contract.")
    parser.add_argument("--config", default=str(ROOT / "src" / "configs" / "darwin_x_4b.yaml"))
    parser.add_argument("--instantiate-meta", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    estimate = estimate_darwin_x_parameters(config)
    report = {
        "config_path": str(Path(args.config)),
        "model_name": config.model_name,
        "init": config.init,
        "tokenizer": config.tokenizer,
        "shape": {
            "vocab_size": config.vocab_size,
            "context_length": config.context_length,
            "inference_context_length": config.inference_context_length,
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "n_kv_heads": config.n_kv_heads,
            "ssd_layers": list(config.ssd_layer_indices),
            "attention_layers": list(config.attention_layer_indices),
        },
        "moe": {
            "fine_experts": config.fine_experts,
            "shared_experts": config.shared_experts,
            "experts_per_token": config.experts_per_token,
            "fine_expert_hidden_dim": config.fine_expert_hidden_dim,
            "shared_expert_hidden_dim": config.shared_expert_hidden_dim,
        },
        "heartbeat": {
            "enabled": config.heartbeat_enabled,
            "think_interval": config.heartbeat_think_interval,
            "explore_interval": config.heartbeat_explore_interval,
            "memory_capacity": config.heartbeat_memory_capacity,
            "surprise_threshold": config.heartbeat_surprise_threshold,
            "ff_layers": config.heartbeat_ff_layers,
        },
        "parameter_estimate": {
            **estimate,
            "total_b": round(estimate["total"] / 1e9, 3),
            "active_per_token_b": round(estimate["active_per_token"] / 1e9, 3),
        },
    }

    if args.instantiate_meta:
        with torch.device("meta"):
            model = DarwinXModel(config)
        report["meta_instantiated_params"] = sum(p.numel() for p in model.parameters())

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
