# Gradient Mutational State v2 — 2026-07-14

## Objetivo

Substituir sensores baseados apenas em normas escalares por geometria assinada
por expert, preservar retomada de checkpoints v7 antigos e impedir mutações
estruturais antes de validação causal.

## Implementado

- sketch assinado determinístico de 64 dimensões por expert;
- estabilidade e reversão direcionais realmente per-expert;
- magnitude relativa ao EMA do próprio expert;
- estagnação somente quando o expert foi roteado e estava treinável;
- base histórica protegida rank-4, alimentada por evidência associada à retenção Ghost;
- conflito e residual ortogonal medidos contra a base histórica;
- agregação main + Ghost em uma observação por backward completo;
- `plasticity_decision()` conectado à proposta autonômica em shadow mode;
- token gate isolado das decisões mutacionais para evitar feedback autorrealizável;
- migração versionada para checkpoints v7 anteriores em todos os loaders Darwin-X;
- executor `scripts/test_10_cenarios.py` restaurado ao contrato CPU/pytest.

## Validação

- 11/11 testes causais individuais do Gradiente Mutacional;
- 225/225 testes da suíte completa em CPU;
- 10/10 cenários Topology Manifest v7;
- `py_compile` nos módulos alterados;
- `git diff --check` sem erros.

## Limite deliberado

IGNORE, PROTECT, EXPAND, PRUNE e CREATE são observados e persistidos, mas não
alteram pesos, gate de tokens ou anatomia. A saída de shadow mode exige
calibração held-out e torneio causal por decisão.
