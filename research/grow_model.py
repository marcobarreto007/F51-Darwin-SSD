#!/usr/bin/env python
"""
F51 Growth Engine — Expande o modelo seed para uma versão maior.

Estratégia: Layer Duplication + Dimension Expansion
    1. Carrega checkpoint do seed (27M, d_model=384, 8 camadas)
    2. Cria modelo grown (60M, d_model=512, 12 camadas)
    3. Copia pesos das camadas existentes
    4. Duplica camadas com ruído para as novas
    5. Expande embedding com padding + ruído
    6. Salva checkpoint grown pronto pra treinar

Uso:
    python research/grow_model.py
    python research/grow_model.py --seed checkpoints/organism/organism_247/step_0006000.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import torch
import yaml
from f51_darwin.config import DarwinConfig
from f51_darwin.model import F51DarwinModel


def expand_embedding(
    old_weight: torch.Tensor,
    new_vocab_size: int,
    new_d_model: int,
    noise_std: float = 0.01,
) -> torch.Tensor:
    """Expande embedding de (old_vocab, old_dim) → (new_vocab, new_dim)."""
    old_vocab, old_dim = old_weight.shape
    new_weight = torch.randn(new_vocab_size, new_d_model) * noise_std

    # Copia os tokens existentes
    copy_vocab = min(old_vocab, new_vocab_size)
    copy_dim = min(old_dim, new_d_model)
    new_weight[:copy_vocab, :copy_dim] = old_weight[:copy_vocab, :copy_dim]

    return new_weight


def expand_linear(
    old_weight: torch.Tensor,
    old_bias: torch.Tensor | None,
    new_out_features: int,
    new_in_features: int,
    noise_std: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Expande camada linear preservando pesos existentes."""
    old_out, old_in = old_weight.shape
    new_weight = torch.randn(new_out_features, new_in_features) * noise_std
    copy_out = min(old_out, new_out_features)
    copy_in = min(old_in, new_in_features)
    new_weight[:copy_out, :copy_in] = old_weight[:copy_out, :copy_in]

    new_bias = None
    if old_bias is not None:
        new_bias = torch.zeros(new_out_features)
        new_bias[:copy_out] = old_bias[:copy_out]

    return new_weight, new_bias


def grow_model(
    seed_checkpoint: str | Path,
    grown_config_path: str | Path,
    output_path: str | Path,
    noise_std: float = 0.01,
):
    """Expande um modelo seed para uma versão maior."""
    seed_path = Path(seed_checkpoint)
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed checkpoint not found: {seed_path}")

    print(f"F51 Growth Engine")
    print(f"  Seed: {seed_path.name}")
    print(f"  Config: {grown_config_path}")
    print(f"")

    # Carrega seed
    seed_ckpt = torch.load(seed_path, map_location="cpu", weights_only=False)
    seed_config = DarwinConfig.from_mapping(seed_ckpt["config"])
    seed_model = F51DarwinModel(seed_config)
    seed_model.load_state_dict(seed_ckpt["model_state_dict"])

    seed_params = sum(p.numel() for p in seed_model.parameters())
    print(f"  Seed: {seed_config.model_name} | {seed_params/1e6:.1f}M params")
    print(f"        d_model={seed_config.d_model} layers={seed_config.n_layers}")

    # Cria modelo grown
    grown_config = DarwinConfig.from_yaml(Path(grown_config_path))
    grown_config = DarwinConfig(
        model_name=grown_config.model_name,
        vocab_size=seed_config.vocab_size,
        context_length=seed_config.context_length,
        d_model=grown_config.d_model,
        n_layers=grown_config.n_layers,
        n_heads=grown_config.n_heads,
        mlp_ratio=seed_config.mlp_ratio,
        dropout=seed_config.dropout,
        weight_tying=seed_config.weight_tying,
    )
    grown_model = F51DarwinModel(grown_config)

    grown_params = sum(p.numel() for p in grown_model.parameters())
    print(f"  Grown: {grown_config.model_name} | {grown_params/1e6:.1f}M params")
    print(f"         d_model={grown_config.d_model} layers={grown_config.n_layers}")
    print(f"  Growth: {grown_params/seed_params:.1f}x")

    # ── Expande pesos ──
    seed_state = seed_model.state_dict()
    grown_state = {}

    for key, grown_tensor in grown_model.state_dict().items():
        if key not in seed_state:
            # Nova camada — inicializa com ruído
            grown_state[key] = grown_tensor
            continue

        seed_tensor = seed_state[key]

        if seed_tensor.shape == grown_tensor.shape:
            # Mesmo shape — copia direto
            grown_state[key] = seed_tensor.clone()
        elif "embedding" in key or "lm_head" in key:
            # Expande embedding
            grown_state[key] = expand_embedding(
                seed_tensor, grown_tensor.shape[0], grown_tensor.shape[1], noise_std,
            )
        elif seed_tensor.dim() == 2:
            # Expande linear
            new_w, _ = expand_linear(
                seed_tensor, None,
                grown_tensor.shape[0], grown_tensor.shape[1], noise_std,
            )
            grown_state[key] = new_w
        else:
            # Outros — copia o que der
            grown_state[key] = grown_tensor.clone()
            slices = tuple(slice(0, min(s, g)) for s, g in zip(seed_tensor.shape, grown_tensor.shape))
            grown_state[key][slices] = seed_tensor[slices]

    # ── Duplica camadas SSD com ruído ──
    # Camadas 8-11 do grown recebem cópia com ruído das camadas 0-3 do seed
    for new_layer in range(seed_config.n_layers, grown_config.n_layers):
        src_layer = new_layer % seed_config.n_layers
        for key in grown_state:
            if f"blocks.{new_layer}." in key:
                src_key = key.replace(f"blocks.{new_layer}.", f"blocks.{src_layer}.")
                if src_key in seed_state:
                    grown_state[key] = seed_state[src_key].clone()
                    grown_state[key] += torch.randn_like(grown_state[key]) * noise_std

    grown_model.load_state_dict(grown_state)

    # ── Salva ──
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": grown_model.state_dict(),
        "config": {
            "model_name": grown_config.model_name,
            "vocab_size": grown_config.vocab_size,
            "context_length": grown_config.context_length,
            "d_model": grown_config.d_model,
            "n_layers": grown_config.n_layers,
            "n_heads": grown_config.n_heads,
            "mlp_ratio": grown_config.mlp_ratio,
            "dropout": grown_config.dropout,
            "norm": "rmsnorm",
            "activation": "swiglu",
            "weight_tying": grown_config.weight_tying,
            "init": "random",
            "tokenizer": "f51_bpe",
            "ssd_attention_ratio": "3:1",
            "module_states": ["candidate", "active", "frozen", "quarantine", "merged", "dead"],
        },
        "metrics": seed_ckpt.get("metrics", {}),
        "lineage": f"grown_from_{seed_path.name}",
        "growth_noise_std": noise_std,
    }, output_path)

    size_mb = output_path.stat().st_size / 1e6
    print(f"")
    print(f"  ✅ Modelo grown salvo: {output_path}")
    print(f"     Tamanho: {size_mb:.0f} MB")
    print(f"     Params: {grown_params/1e6:.1f}M")
    print(f"     Crescimento: {grown_params/seed_params:.1f}x do seed")
    print(f"")
    print(f"  Para treinar na nuvem:")
    print(f"    1. Upload: scp {output_path} root@IP:/workspace/F51-Darwin-SSD/checkpoints/")
    print(f"    2. Treinar: --resume {output_path} --config src/configs/darwin_grown_60m.yaml")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="F51 Growth Engine")
    parser.add_argument("--seed", default="checkpoints/organism/organism_247/step_0006000.pt")
    parser.add_argument("--config", default="src/configs/darwin_grown_60m.yaml")
    parser.add_argument("--output", default="checkpoints/grown/grown_60m_seed.pt")
    parser.add_argument("--noise", type=float, default=0.01)
    args = parser.parse_args()

    grow_model(args.seed, args.config, args.output, args.noise)
