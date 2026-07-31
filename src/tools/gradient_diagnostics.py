"""Gradient diagnostics for Darwin-X 100M — run a few steps with gradient hooks."""
from __future__ import annotations

import yaml, torch, math
from collections import defaultdict
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.organism.config import DarwinOrganismConfig
from f51_darwin.organism.bootstrap import _DarwinBootstrapMixin

ROOT = r"C:\Users\marco\Desktop\F51-Darwin-SSD"

def main():
    print("=" * 60)
    print("  GRADIENT DIAGNOSTICS — Darwin-X 100M")
    print("=" * 60)

    # Load config
    cfg = DarwinXConfig.from_mapping(yaml.safe_load(
        open(f"{ROOT}/src/configs/darwin_x_100m.yaml")
    ))
    print(f"\nModel: {cfg.model_name}  d={cfg.d_model}  L={cfg.n_layers}  E={cfg.fine_experts}")
    print(f"Losses: mtp={cfg.mtp_weight} jepa={cfg.jepa_weight} ghost={cfg.ghost_weight} aux={cfg.aux_loss_scale}")
    print(f"Organs: gaba={cfg.gaba_enabled} dae={cfg.dae_enabled} ghost_enabled={cfg.ghost_enabled}")

    # Build model
    model = DarwinXModel(cfg)
    model = model.to('cuda:0')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {total_params/1e6:.1f}M  ({total_params:,})")

    # Activate organism
    model.activate_organism(
        curiosity=None, ghost_brain=None,
        jepa=model.jepa_predictor, organism_ref=None,
    )

    # Register gradient hooks
    grad_stats: dict[str, list[float]] = defaultdict(list)

    def make_hook(name):
        def hook(grad):
            if grad is not None:
                g = grad.detach().float()
                grad_stats[f"{name}_norm"].append(float(g.norm().cpu()))
                grad_stats[f"{name}_mean"].append(float(g.mean().cpu()))
                grad_stats[f"{name}_std"].append(float(g.std().cpu()))
                grad_stats[f"{name}_max"].append(float(g.abs().max().cpu()))
                grad_stats[f"{name}_zero_frac"].append(
                    float((g == 0).float().mean().cpu())
                )
        return hook

    # Hook key layers
    model.token_embedding.weight.register_hook(make_hook("embed"))
    for i, block in enumerate(model.blocks):
        # Attention output projection
        for name, mod in block.named_modules():
            if 'out_proj' in name and hasattr(mod, 'weight') and mod.weight.requires_grad:
                mod.weight.register_hook(make_hook(f"B{i}_attn_out"))
                break
        # MoE router
        if hasattr(block.moe, 'router') and block.moe.router.weight.requires_grad:
            block.moe.router.weight.register_hook(make_hook(f"B{i}_router"))
        # First expert's first linear
        if block.moe.fine_experts:
            for ex_name, ex_mod in block.moe.fine_experts[0].named_modules():
                if hasattr(ex_mod, 'weight') and ex_mod.weight.requires_grad:
                    ex_mod.weight.register_hook(make_hook(f"B{i}_expert0"))
                    break
    model.lm_head.weight.register_hook(make_hook("lm_head"))

    # Run training steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.01)
    model = model.to(dtype=torch.bfloat16)
    model.train()

    print(f"\n{'='*60}")
    print(f"  Running 20 diagnostic steps...")
    print(f"{'='*60}")

    for step in range(1, 21):
        x = torch.randint(0, min(1000, cfg.vocab_size - 1), (1, 64), device='cuda:0')
        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(x, labels=x, domain='diag')

        loss = out.loss
        loss.backward()

        # Clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if step % 5 == 0:
            print(f"  step={step:>3d}  loss={loss.item():.4f}  "
                  f"lm={out.lm_loss.item():.4f}  gho={out.ghost_loss.item():.4f}  "
                  f"mtp={out.mtp_loss.item():.4f}  jepa={out.jepa_loss.item():.4f}")

        optimizer.step()

    # ── Aggregate & report ──
    print(f"\n{'='*60}")
    print(f"  GRADIENT ANALYSIS (20 steps aggregated)")
    print(f"{'='*60}")

    # Group by layer type
    layers = {}
    for key, values in sorted(grad_stats.items()):
        if not values:
            continue
        parts = key.rsplit('_', 1)
        layer = parts[0]
        stat = parts[1]
        if layer not in layers:
            layers[layer] = {}
        layers[layer][stat] = values

    print(f"\n{'Layer':<20} {'|grad|':>10} {'mean':>10} {'std':>10} {'max':>10} {'zero%':>8}")
    print("-" * 72)

    for layer_name in sorted(layers.keys()):
        stats = layers[layer_name]
        norms = stats.get('norm', [0])
        means = stats.get('mean', [0])
        stds = stats.get('std', [0])
        maxs = stats.get('max', [0])
        zeros = stats.get('zero_frac', [0])

        avg_norm = sum(norms) / len(norms)
        avg_mean = sum(means) / len(means)
        avg_std = sum(stds) / len(stds)
        avg_max = sum(maxs) / len(maxs)
        avg_zero = sum(zeros) / len(zeros)

        # Flag issues
        flags = ""
        if avg_norm < 1e-5:
            flags += " ⚠️VANISHING"
        if avg_norm > 100:
            flags += " 🔥EXPLODING"
        if avg_zero > 0.5:
            flags += " 💀SPARSE_DEAD"

        print(f"{layer_name:<20} {avg_norm:>10.4f} {avg_mean:>10.6f} {avg_std:>10.6f} {avg_max:>10.4f} {avg_zero:>7.1%}{flags}")

    # ── Gradient flow health ──
    print(f"\n{'='*60}")
    print(f"  GRADIENT FLOW HEALTH")
    print(f"{'='*60}")

    norms_by_layer = {}
    for key, values in grad_stats.items():
        if key.endswith('_norm'):
            layer = key[:-5]
            norms_by_layer[layer] = sum(values) / len(values) if values else 0

    if norms_by_layer:
        max_norm = max(norms_by_layer.values())
        min_norm = min(v for v in norms_by_layer.values() if v > 0)

        print(f"  Max gradient norm: {max_norm:.4f}")
        print(f"  Min gradient norm: {min_norm:.4f}")
        ratio = max_norm / min_norm if min_norm > 0 else float('inf')
        print(f"  Ratio max/min:     {ratio:.1f}x")

        if ratio > 100:
            print(f"  ⚠️  HIGH IMBALANCE — gradient norms vary {ratio:.0f}x across layers")
            print(f"      This can cause uneven learning. Check deeper layers.")
        elif ratio > 20:
            print(f"  ⚡ MODERATE IMBALANCE — {ratio:.0f}x variation is typical for deep models")
        else:
            print(f"  ✅ HEALTHY — gradient norms are within {ratio:.0f}x")

    # ── Loss component contribution ──
    print(f"\n{'='*60}")
    print(f"  LOSS COMPONENT ANALYSIS (final step)")
    print(f"{'='*60}")
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        x = torch.randint(0, min(1000, cfg.vocab_size - 1), (1, 64), device='cuda:0')
        out = model(x, labels=x, domain='diag')
        total = out.loss.item()
        lm = out.lm_loss.item()
        mtp = out.mtp_loss.item() if out.mtp_loss is not None else 0
        jepa = out.jepa_loss.item() if out.jepa_loss is not None else 0
        ghost = out.ghost_loss.item() if out.ghost_loss is not None else 0
        aux = out.aux_loss.item() if out.aux_loss is not None else 0
        print(f"  Total loss:  {total:.4f}")
        print(f"  LM loss:     {lm:.4f}  (weight: 1.0)")
        print(f"  MTP loss:    {mtp:.4f}  (weight: {cfg.mtp_weight})  → contribution: {mtp * cfg.mtp_weight:.4f}")
        print(f"  JEPA loss:   {jepa:.4f}  (weight: {cfg.jepa_weight})  → contribution: {jepa * cfg.jepa_weight:.4f}")
        print(f"  Ghost loss:  {ghost:.4f}  (weight: {cfg.ghost_weight})  → contribution: {ghost * cfg.ghost_weight:.4f}")
        print(f"  Aux loss:    {aux:.4f}  (weight: {cfg.aux_loss_scale})  → contribution: {aux * cfg.aux_loss_scale:.4f}")

    print(f"\n{'='*60}")
    print(f"  DIAGNOSTICS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
