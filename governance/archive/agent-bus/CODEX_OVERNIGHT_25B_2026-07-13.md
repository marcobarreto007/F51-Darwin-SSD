# CODEX - ponte de treino noturno 2.5B - 2026-07-13

## Objetivo

Retomar o checkpoint treinado da classe 2.5B usando o corpus externo em
`C:\Users\marco\Desktop\F51-Dataset-Organizado`, com todos os orgaos ativos e
as RTX 5060 Ti 16 GB + RTX 3060 12 GB participando do mesmo processo.

## Estado observado

- O `run247` do checkpoint 239 foi parado por Marco.
- O checkpoint local 239 tem 3.147.652.895 bytes e 576.109.548 parametros;
  ele nao e o checkpoint 2.5B.
- O novo checkpoint 2.5B ainda nao apareceu em `03_CHECKPOINTS` e nenhum
  pendrive esta montado.
- O host NUCLEO historico `70.30.158.46:56013` esta offline.
- O arquivo `configs/darwin_x_2.5b.yaml` foi alterado concorrentemente e, no
  snapshot de 01:08, continha 2.188.070.652 parametros e model_name 1.9B. O
  launcher recusa esse substituto menor.

## Alteracoes persistidas

- `DarwinXModel.recommended_dual_gpu_split()` balanceia bytes de parametros
  pela VRAM real das duas GPUs.
- O bootstrap nao move mais o modelo inteiro para `cuda:0` antes do split.
- O peso compartilhado embedding/lm_head e religado depois do placement.
- `scripts/start_overnight_25b.ps1` valida corpus externo, manifesto, duas
  GPUs, ausencia de treino concorrente, tamanho real do modelo, estabilidade
  do checkpoint e gera manifesto de prontidao antes do launch oculto.
- `tests/test_dual_gpu_split.py` cobre o seletor heterogeneo e capacidades
  invalidas.

## Evidencia

- `py_compile` passou para `darwin_x.py` e `darwin_organism.py`.
- Suite CPU: 182 testes aprovados com CUDA oculto.
- Smoke CUDA real: split 11/9; primeira camada em cuda:0 e ultima em cuda:1;
  MTP, JEPA, router e experts receberam gradiente; Ghost loss foi finita;
  Heartbeat bateu, escreveu memoria e gerou pensamento; Curiosity e Ghost
  estavam conectados; embedding e lm_head permaneceram amarrados.
- O dry-run do launcher recusou corretamente o YAML reduzido de 2.188B pelo
  gate minimo de 2.4B.

## Proximo gate

Quando o checkpoint terminar de ser copiado, coloca-lo em
`F51-Dataset-Organizado\03_CHECKPOINTS` e executar primeiro:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_overnight_25b.ps1 `
  -Checkpoint <arquivo.pt>
```

Somente depois de validar a configuracao embutida, compatibilidade de shapes e
um smoke real do proprio checkpoint, repetir com `-Launch`.
