#!/usr/bin/env python3
"""Verify 5B config parameter estimation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from f51_darwin.darwin_x import DarwinXConfig, estimate_darwin_x_parameters
from f51_darwin.config import coerce_mapping
import yaml

# Load the 5B config
with open('configs/darwin_x_5b.yaml') as f:
    cfg = yaml.safe_load(f)

config = DarwinXConfig.from_mapping(cfg)
est = estimate_darwin_x_parameters(config)

print('=== F51-Darwin-X-5B-Nitro — Estimativa de Parâmetros ===')
print(f'd_model: {config.d_model}')
print(f'n_layers: {config.n_layers}')
print(f'fine_experts: {config.fine_experts}')
print(f'shared_experts: {config.shared_experts}')
print(f'experts_per_token: {config.experts_per_token}')
print(f'fine_hidden: {config.fine_expert_hidden_dim}')
print(f'shared_hidden: {config.shared_expert_hidden_dim}')
print(f'attention_layers: {len(config.attention_layer_indices)}')
print(f'ssd_layers: {len(config.ssd_layer_indices)}')
print()
print('Componentes:')
for k, v in est.items():
    if k not in ('total', 'active_per_token'):
        print(f'  {k:20s}: {v:>12,}')
print()
print(f'TOTAL:           {est["total"]:>12,} ({est["total"]/1e9:.2f}B)')
print(f'Active per token: {est["active_per_token"]:>12,} ({est["active_per_token"]/1e9:.2f}B)')
print()
print(f'Proporcao active/total: {est["active_per_token"]/est["total"]:.1%}')
