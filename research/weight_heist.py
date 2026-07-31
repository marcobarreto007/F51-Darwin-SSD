#!/usr/bin/env python3
"""
F51 WEIGHT HEIST — Esquarteja pesos de modelos open-source para acelerar o Darwin.

Primeiro alvo: GPT-2 Medium (355M) — mais próximo do Darwin-X 600M.
  d_model=1024 → 1408 (pad)
  n_heads=16 → 16 ✅
  n_layers=24 → 12 (pegamos primeiras 12)

Não viola doutrina: os pesos são semente inicial, o treino F51 redefine tudo.
É a mesma lógica de "random init com distribuição melhor que N(0, 0.02)".

Uso:
  python -m research.weight_heist --donor gpt2-medium --output checkpoints/heist_gpt2_600m.pt
  python -m research.turbo_train --resume checkpoints/heist_gpt2_600m.pt --steps 5000
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

from f51_darwin.darwin_x import DarwinXConfig, DarwinXModel


def _pad_weight(
    src: torch.Tensor,
    target_shape: tuple,
    stats: dict[str, int],
    dim: int = 0,
) -> torch.Tensor:
    """Pad or slice a weight tensor to the target shape along ``dim``."""
    if src.shape == target_shape:
        stats["exact"] += 1
        return src.clone()
    pad_size = target_shape[dim] - src.shape[dim]
    if pad_size <= 0:
        slices = [slice(None)] * src.ndim
        slices[dim] = slice(0, target_shape[dim])
        stats["sliced"] += 1
        return src[tuple(slices)].clone()

    pad_shape = list(src.shape)
    pad_shape[dim] = pad_size
    noise = torch.randn(pad_shape, device=src.device, dtype=src.dtype) * 0.01
    last_row = src.select(dim, -1).unsqueeze(dim).repeat_interleave(pad_size, dim=dim)
    stats["padded"] += 1
    return torch.cat([src, last_row + noise], dim=dim).clone()


def _map_linear(
    src_weight: torch.Tensor,
    src_bias: torch.Tensor | None,
    out_features: int,
    in_features: int,
    stats: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Map a donor linear layer to the target dimensions."""
    weight = _pad_weight(src_weight.float(), (out_features, in_features), stats, dim=0)
    weight = _pad_weight(weight, (out_features, in_features), stats, dim=1)
    bias = None
    if src_bias is not None:
        bias = _pad_weight(src_bias.float(), (out_features,), stats, dim=0)
    return weight, bias


def heist_gpt2(donor_name: str, config: DarwinXConfig, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Rouba pesos do GPT-2 e mapeia para o Darwin-X.

    Mapeamento:
      GPT-2                        → Darwin-X 600M
      ─────────────────────────────────────────────────
      wte.weight [50257, 1024]     → token_embedding.weight [58162, 1408]
      wpe.weight [1024, 1024]      → descartado (usamos RoPE)
      h.N.ln_1.weight [1024]       → blocks.N.norm1.weight [1408]  (LayerNorm→RMSNorm, pad)
      h.N.attn.c_attn.weight       → blocks.N.attention.{q,k,v}_proj (split + GQA)
      h.N.attn.c_attn.bias         → blocks.N.attention.{q,k,v}_proj.bias
      h.N.attn.c_proj.weight       → blocks.N.attention.o_proj.weight
      h.N.attn.c_proj.bias         → blocks.N.attention.o_proj.bias
      h.N.ln_2.weight [1024]       → blocks.N.norm2.weight [1408]
      h.N.mlp.c_fc.weight [4096,1024] → MoE expert gate/up proj
      h.N.mlp.c_proj.weight [1024,4096] → MoE expert down proj
      ln_f.weight [1024]           → norm.weight [1408]
    """
    from transformers import GPT2LMHeadModel

    print(f"🔫 Carregando {donor_name}...")
    donor = GPT2LMHeadModel.from_pretrained(donor_name)
    donor_state = donor.state_dict()
    donor_config = donor.config

    print(f"   Donor: d_model={donor_config.n_embd}, layers={donor_config.n_layer}, "
          f"heads={donor_config.n_head}, vocab={donor_config.vocab_size}")
    print(f"   Alvo:  d_model={config.d_model}, layers={config.n_layers}, "
          f"heads={config.n_heads}, vocab={config.vocab_size}")

    # Cria modelo alvo vazio pra pegar os nomes dos parâmetros
    with torch.device("meta"):
        target = DarwinXModel(config)
    target_state = target.state_dict()

    donor_d = donor_config.n_embd  # 1024
    target_d = config.d_model      # 1408
    donor_vocab = donor_config.vocab_size  # 50257
    target_vocab = config.vocab_size       # 58162
    donor_heads = donor_config.n_head       # 16
    n_kv = config.n_kv_heads                # 4

    heisted: dict[str, torch.Tensor] = {}
    stats = {"exact": 0, "padded": 0, "sliced": 0, "split": 0, "random": 0, "replicated": 0}

    # ── 1. Token Embedding ──
    print("   1/7 Embedding...")
    wte = donor_state["transformer.wte.weight"].float()  # [50257, 1024]
    wte = _pad_weight(wte, (target_vocab, target_d), stats, dim=0)  # pad vocab
    wte = _pad_weight(wte, (target_vocab, target_d), stats, dim=1)  # pad d_model
    heisted["token_embedding.weight"] = wte

    # ── 2. Blocos (12 layers do GPT-2 → 12 layers do Darwin-X) ──
    attention_indices = set(config.attention_layer_indices)  # {3, 7, 11} for 12 layers

    for layer_idx in range(config.n_layers):
        donor_layer = min(layer_idx, donor_config.n_layer - 1)  # wrap if needed
        prefix = f"blocks.{layer_idx}"
        d_prefix = f"transformer.h.{donor_layer}"

        is_attn = layer_idx in attention_indices

        # ── norm1 (RMSNorm ← LayerNorm) ──
        key = f"{prefix}.norm1.weight"
        ln_w = donor_state[f"{d_prefix}.ln_1.weight"].float()  # [1024]
        heisted[key] = _pad_weight(ln_w, (target_d,), stats, dim=0)

        if is_attn:
            # ── ATTENTION: Q, K, V projections ──
            # GPT-2: c_attn.weight [3072, 1024] = concat(Q,K,V)
            # Darwin-X GQA: q_proj [1408,1408], k_proj [352,1408], v_proj [352,1408]
            c_attn = donor_state[f"{d_prefix}.attn.c_attn.weight"].float()  # [3072, 1024]
            c_attn_b = donor_state[f"{d_prefix}.attn.c_attn.bias"].float()  # [3072]

            # Split into Q, K, V
            q_w, k_w, v_w = c_attn.chunk(3, dim=0)  # each [1024, 1024]
            q_b, k_b, v_b = c_attn_b.chunk(3, dim=0)  # each [1024]

            # Map Q: [1024, 1024] → [1408, 1408]
            q_w_mapped, q_b_mapped = _map_linear(q_w, q_b, target_d, target_d, stats)
            heisted[f"{prefix}.attention.q_proj.weight"] = q_w_mapped
            heisted[f"{prefix}.attention.q_proj.bias"] = q_b_mapped

            # Map K: [1024, 1024] → [352, 1408]  (n_kv_heads=4, head_dim=88)
            kv_out = n_kv * (target_d // config.n_heads)
            k_w_mapped, k_b_mapped = _map_linear(k_w, k_b, kv_out, target_d, stats)
            heisted[f"{prefix}.attention.k_proj.weight"] = k_w_mapped
            heisted[f"{prefix}.attention.k_proj.bias"] = k_b_mapped

            # Map V: [1024, 1024] → [352, 1408]
            v_w_mapped, v_b_mapped = _map_linear(v_w, v_b, kv_out, target_d, stats)
            heisted[f"{prefix}.attention.v_proj.weight"] = v_w_mapped
            heisted[f"{prefix}.attention.v_proj.bias"] = v_b_mapped

            # ── c_proj (output projection) ──
            c_proj = donor_state[f"{d_prefix}.attn.c_proj.weight"].float()  # [1024, 1024]
            c_proj_b = donor_state[f"{d_prefix}.attn.c_proj.bias"].float()  # [1024]
            o_w, o_b = _map_linear(c_proj, c_proj_b, target_d, target_d, stats)
            heisted[f"{prefix}.attention.o_proj.weight"] = o_w
            heisted[f"{prefix}.attention.o_proj.bias"] = o_b

            stats["split"] += 5

        # ── norm2 (RMSNorm ← LayerNorm ln_2) ──
        key = f"{prefix}.norm2.weight"
        ln2_w = donor_state[f"{d_prefix}.ln_2.weight"].float()  # [1024]
        heisted[key] = _pad_weight(ln2_w, (target_d,), stats, dim=0)

        # ── MoE experts (steal from GPT-2 MLP for each expert) ──
        c_fc = donor_state[f"{d_prefix}.mlp.c_fc.weight"].float()  # [4096, 1024]
        c_fc_b = donor_state[f"{d_prefix}.mlp.c_fc.bias"].float()  # [4096]
        c_proj = donor_state[f"{d_prefix}.mlp.c_proj.weight"].float()  # [1024, 4096]
        c_proj_b = donor_state[f"{d_prefix}.mlp.c_proj.bias"].float()  # [1024]

        # Para cada fine expert, replica os pesos do MLP com noise pra diferenciar
        for expert_idx in range(config.fine_experts):
            ep = f"{prefix}.moe.fine_experts.{expert_idx}"

            # gate_proj: [768, 1408] ← c_fc primeira metade [2048, 1024]
            gate_w = c_fc[:c_fc.shape[0]//2]  # [2048, 1024]
            noise = torch.randn_like(gate_w) * 0.02 * (expert_idx + 1) / config.fine_experts
            gate_w_noisy = gate_w + noise
            gw, _ = _map_linear(gate_w_noisy, None, config.fine_expert_hidden_dim, target_d, stats)
            heisted[f"{ep}.gate_proj.weight"] = gw
            heisted[f"{ep}.up_proj.weight"] = gw.clone()  # usa mesmo peso com noise diferente

            # down_proj: [1408, 768] ← c_proj [1024, 4096]
            down_w = c_proj  # [1024, 4096]
            noise = torch.randn_like(down_w) * 0.02 * (expert_idx + 1) / config.fine_experts
            down_w_noisy = down_w + noise
            # Need to reshape: target is [1408, 768]
            dw, _ = _map_linear(down_w_noisy.t(), None, config.fine_expert_hidden_dim, target_d, stats)
            heisted[f"{ep}.down_proj.weight"] = dw.t()

        stats["replicated"] += config.fine_experts

        # ── Shared experts ──
        for shared_idx in range(config.shared_experts):
            sp = f"{prefix}.moe.shared_experts.{shared_idx}"
            gw, _ = _map_linear(c_fc[:c_fc.shape[0]//2], None, config.shared_expert_hidden_dim, target_d, stats)
            heisted[f"{sp}.gate_proj.weight"] = gw
            heisted[f"{sp}.up_proj.weight"] = gw.clone()
            dw, _ = _map_linear(c_proj.t(), None, config.shared_expert_hidden_dim, target_d, stats)
            heisted[f"{sp}.down_proj.weight"] = dw.t()

        # ── MoE Router (random — não tem equivalente no GPT-2) ──
        stats["random"] += 1

    # ── 3. Final RMSNorm ──
    print("   3/7 Final norm...")
    ln_f = donor_state["transformer.ln_f.weight"].float()
    heisted["norm.weight"] = _pad_weight(ln_f, (target_d,), stats, dim=0)

    # ── 4. LM Head (same as embedding — weight tying) ──
    print("   4/7 LM head...")
    heisted["lm_head.weight"] = heisted["token_embedding.weight"].clone()

    # ── 5. MTP heads (random — não existe no GPT-2) ──
    print("   5/7 MTP heads (random)...")
    stats["random"] += config.mtp_depth

    # ── 6. JEPA predictor (random) ──
    print("   6/7 JEPA (random)...")
    stats["random"] += 4

    # ── 7. Preenche o resto com os pesos random do target ──
    print("   7/7 Preenchendo resto...")
    for key in target_state:
        if key not in heisted:
            # Usa a inicialização padrão do modelo (normal)
            pass  # vai ser inicializado pelo modelo, não sobrescrevemos

    print(f"\n   ✅ Heist completo!")
    print(f"   Exact: {stats['exact']} | Padded: {stats['padded']} | Sliced: {stats['sliced']}")
    print(f"   Split: {stats['split']} | Random: {stats['random']} | Replicated: {stats['replicated']}")

    del donor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return heisted


def main():
    parser = argparse.ArgumentParser(description="F51 Weight Heist — rouba pesos de modelos prontos")
    parser.add_argument("--donor", default="gpt2-medium",
                       choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    parser.add_argument("--config", default=str(ROOT / "src" / "configs" / "darwin_x_600m.yaml"))
    parser.add_argument("--output", default=str(ROOT / "checkpoints" / "heist_gpt2_600m.pt"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config = DarwinXConfig.from_mapping(raw)
    print(f"🎯 Alvo: {config.model_name} ({config.d_model}d, {config.n_layers}L, {config.fine_experts}E)")

    heisted = heist_gpt2(args.donor, config, args.device)

    # Cria modelo com pesos roubados
    model = DarwinXModel(config)
    model_state = model.state_dict()

    # Aplica só os pesos que roubamos com sucesso
    loaded = 0
    skipped = 0
    for key in model_state:
        if key in heisted:
            hw = heisted[key]
            tw = model_state[key]
            if hw.shape == tw.shape:
                model_state[key] = hw.to(dtype=tw.dtype)
                loaded += 1
            else:
                print(f"   ⚠️ Shape mismatch: {key} donor={list(hw.shape)} target={list(tw.shape)}")
                skipped += 1

    model.load_state_dict(model_state)
    print(f"   📦 {loaded} tensores roubados, {skipped} com shape mismatch")

    # Salva
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 3,
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "config": asdict(config),
        "training_state": {
            "step": 0,
            "run_id": "heist_gpt2",
            "donor": args.donor,
            "note": "Pesos do GPT-2 adaptados para Darwin-X. Treino F51 redefine tudo.",
        },
        "lineage": "F51 Darwin-X — seeded from GPT-2 Medium, trained on F51 corpus",
    }
    torch.save(payload, output)
    print(f"\n💰💰💰 HEIST COMPLETO: {output}")

    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   Pronto pra treinar: {total:.1f}M params com pesos GPT-2 como semente")
    print(f"   Lance: python -m research.turbo_train --resume {output} --steps 5000")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
