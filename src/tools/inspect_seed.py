from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import torch
import yaml

from f51_darwin.darwin_x import DarwinXConfig, DarwinXModel, estimate_darwin_x_parameters


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def load_config(path: Path) -> DarwinXConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Darwin-X config must be a YAML mapping: {path}")
    return DarwinXConfig.from_mapping(raw)


def small_smoke_config(vocab_size: int = 512) -> DarwinXConfig:
    return DarwinXConfig(
        model_name="F51-Darwin-X-Inspect-Smoke",
        vocab_size=vocab_size,
        context_length=32,
        inference_context_length=128,
        d_model=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        fine_experts=4,
        shared_experts=1,
        experts_per_token=2,
        fine_expert_hidden_dim=32,
        shared_expert_hidden_dim=32,
        mtp_depth=2,
        heartbeat_enabled=True,
        heartbeat_memory_capacity=16,
    )


def estimate_activation_bytes(config: DarwinXConfig, batch_size: int = 1) -> int:
    tokens = batch_size * config.context_length
    return tokens * config.d_model * 4


def main() -> int:
    config_path = ROOT / "src" / "configs" / "darwin_x_600m.yaml"
    config = load_config(config_path)
    estimate = estimate_darwin_x_parameters(config)

    smoke_config = small_smoke_config()
    model = DarwinXModel(smoke_config)
    params = count_parameters(model)
    device = torch.device("cpu")
    model = model.to(device)
    input_ids = torch.randint(0, smoke_config.vocab_size, (1, min(32, smoke_config.context_length)), device=device)
    with torch.no_grad():
        out = model(input_ids, labels=input_ids, domain="inspect")
    payload = {
        "model_name": config.model_name,
        "config_path": str(config_path),
        "attention_layers": list(config.attention_layer_indices),
        "ssd_layers": list(config.ssd_layer_indices),
        "parameter_estimate": {
            **estimate,
            "total_b": round(estimate["total"] / 1e9, 3),
            "active_per_token_b": round(estimate["active_per_token"] / 1e9, 3),
        },
        "config": {
            "vocab_size": config.vocab_size,
            "context_length": config.context_length,
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "n_kv_heads": config.n_kv_heads,
            "fine_experts": config.fine_experts,
            "heartbeat_enabled": config.heartbeat_enabled,
        },
        "smoke_forward": {
            "model_name": smoke_config.model_name,
            "device": str(device),
            "params": params,
            "params_millions": round(params["total"] / 1_000_000, 3),
            "logits_shape": list(out.logits.shape),
            "loss_finite": bool(out.loss is not None and torch.isfinite(out.loss)),
            "heartbeat_stats": out.heartbeat_stats,
        },
        "activation_estimate_bytes_per_batch1": estimate_activation_bytes(config),
    }
    if torch.cuda.is_available():
        payload["cuda"] = {
            "device_name": torch.cuda.get_device_name(0),
            "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
