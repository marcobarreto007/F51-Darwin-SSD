# Script de commit — Loop Fechado F51-Darwin-X
# Execute este script de um terminal com git no PATH
# ou instale git: winget install Git.Git

$ErrorActionPreference = "Stop"
Set-Location "C:\Users\marco\Desktop\F51-Darwin-SSD"

Write-Host "=== ARQUIVOS MODIFICADOS ==="
git status --short

Write-Host ""
Write-Host "=== STAGE ALL ==="
git add src/configs/darwin_x_1.6b_nitro.yaml
git add src/f51_darwin/darwin_x_core/config.py
git add src/f51_darwin/darwin_x_core/layers.py
git add src/f51_darwin/darwin_x_core/block.py
git add src/f51_darwin/darwin_x_core/model.py
git add src/f51_darwin/organism/lifecycle.py
git add src/f51_darwin/organism/unified_mesh.py
git add workspace/runtime/history/agent_bus/2026-07-17_loop-fechado-integracao.md

Write-Host ""
Write-Host "=== COMMIT ==="
git commit -m @'
feat: integrar loop fechado F51-Darwin-X — todos os 10 órgaos conectados

## Otimizações de performance
- FlashAttention nativo forcado via sdp_kernel (GQACausalAttention)
- Chunked selective scan otimizado para 512 tokens/bloco
- Aux losses zeradas (MTP, JEPA, Ghost, curiosity) — ~50% menos compute

## Órgãos integrados
- Spider-Sense MLP real no forward pass (substitui heuristica de variancia)
- JEPA V2 com surprise momentum no predictor
- GABAergic E/I injetado no DarwinXBlock (toggle gaba_enabled)
- Inter-Hemispheric entre embedding e blocos (toggle inter_hemispheric_enabled)
- DecisionEngine fatores expostos no DarwinXOutput

## Loop de controle
- UnifiedControlMesh: barramento central (organism/unified_mesh.py)
- _sleep_cycle: L1 pruning sinaptico + ACh reset (Stage 7.5)
- _evolution_death_loop: plasticity decisions -> morte/quarentena/expansao (Stage 8.5)

## Compatibilidade
- Checkpoint v7 preservado (todos novos recursos default=False)
- Dual-GPU pipeline preservado
- qkv_bias=True mantido para resume estrito

## Arquivos
- Modified: src/configs/darwin_x_1.6b_nitro.yaml
- Modified: src/f51_darwin/darwin_x_core/config.py
- Modified: src/f51_darwin/darwin_x_core/layers.py
- Modified: src/f51_darwin/darwin_x_core/block.py
- Modified: src/f51_darwin/darwin_x_core/model.py
- Modified: src/f51_darwin/organism/lifecycle.py
- New: src/f51_darwin/organism/unified_mesh.py
- New: workspace/runtime/history/agent_bus/2026-07-17_loop-fechado-integracao.md
'@

Write-Host ""
Write-Host "=== DONE ==="
git log --oneline -3
