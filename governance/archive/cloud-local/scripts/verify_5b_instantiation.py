#!/usr/bin/env python3
"""Verify 5B config can be instantiated."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from f51_darwin.darwin_x import DarwinXConfig, DarwinXModel, estimate_darwin_x_parameters

# Load config
with open('configs/darwin_x_5b.yaml') as f:
    cfg = yaml.safe_load(f)

config = DarwinXConfig.from_mapping(cfg)
print('=== Validando Darwin-X-5B ===')
print(f'Config loaded: {config.model_name}')
print(f'd_model: {config.d_model}, n_layers: {config.n_layers}')
print()

# Test instantiation (CPU only, no CUDA)
print('Instanciando modelo...')
try:
    model = DarwinXModel(config)
    print(f'✅ Modelo instanciado com sucesso')

    # Count params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Parâmetros totais:    {total:>15,} ({total/1e9:.2f}B)')
    print(f'Parâmetros treináveis: {trainable:>15,} ({trainable/1e9:.2f}B)')

    # Verify estimate
    est = estimate_darwin_x_parameters(config)
    print()
    print('Estimativa via estimate_darwin_x_parameters():')
    print(f'  TOTAL:           {est["total"]:>15,} ({est["total"]/1e9:.2f}B)')
    print(f'  Active per token: {est["active_per_token"]:>15,} ({est["active_per_token"]/1e9:.2f}B)')

    diff = abs(total - est['total'])
    diff_pct = diff / est['total'] * 100
    print()
    print(f'Diferença real vs estimado: {diff:,} ({diff_pct:.2f}%)')

    if diff_pct < 1.0:
        print('✅ Estimativa correta (<1% erro)')
    else:
        print('⚠️ Estimativa com erro >1%')

except Exception as e:
    print(f'❌ Erro na instanciação: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print('=== Tudo OK! ===')
