# Auditoria de saude das GPUs — 2026-07-15 EDT

## Pergunta

Verificar se a RTX 5060 Ti que derrubou o `run247` apresenta evidencia de
defeito fisico, separando hardware de driver/WDDM, energia e interacao dual GPU.

## Evidencia historica

- 105 eventos `nvlddmkm` nos ultimos 30 dias.
- 88 eventos em `GPUID 100`, correspondente a PCI `01:00.0`, RTX 5060 Ti.
- 4 eventos em `GPUID 400`, correspondente a PCI `04:00.0`, RTX 3060.
- Eventos anteriores incluem `UCodeReset TDR`, reset/restart TDR e Graphics
  FECS Exception.
- Tres dumps conhecidos `LiveKernelEvent 141` (engine timeout/TDR) de 12–13/07.
- Seis Kernel-Power 41 em 30 dias, todos com `BugcheckCode=0`, sem WHEA; nao
  provam defeito de GPU ou PSU e podem refletir desligamento forcado apos hang.
- Crash do Darwin: ultimo write 2026-07-14 08:46:06; evento `nvlddmkm` na RTX
  5060 Ti em 08:46:14.

## Estado em repouso

- Ambas GPUs: PnP `OK`, ConfigManagerErrorCode 0.
- Sem WHEA nos ultimos 30 dias.
- PCIe replays desde reset: 0 em ambas.
- Sem thermal slowdown, power brake ou recovery pendente.
- Temperatura inicial: 36 C em ambas.
- Nenhuma ferramenta de overclock encontrada em execucao.
- RTX 5060 Ti: display ativo em WDDM e GPU 0 do treino.
- RTX 3060: sem display ativo, compute-only.
- Driver comum: NVIDIA 595.97 WHQL, 2026-03-24.

## Testes controlados

### RTX 3060 isolada

- 6 GiB de VRAM, padroes `0xAA` e `0x55`: 0 erros.
- BF16 4096x4096 por 20 s: 3.860 iteracoes, resultado finito.
- Pico observado: 119 W, 51 C.
- Novos eventos NVIDIA: 0.

### RTX 5060 Ti isolada

- 6 GiB de VRAM, padroes `0xAA` e `0x55`: 0 erros.
- BF16 4096x4096 por 20 s: 6.760 iteracoes, resultado finito.
- Pico observado: 136 W, 69 C.
- Novos eventos NVIDIA: 0.

### Carga simultanea

- Duas GPUs a 100% por 30 s.
- RTX 5060 Ti: 10.120 iteracoes, 139 W, 73 C, resultado finito.
- RTX 3060: 5.760 iteracoes, 152 W, 65 C, resultado finito.
- Potencia combinada observada aproximada: 291 W nas GPUs.
- Novos eventos NVIDIA: 0.

### Cobertura ampliada de VRAM da RTX 5060 Ti

- 12 GiB alocados; pico PyTorch 13,13 GiB.
- Padroes `0x00`, `0xFF`, `0xAA`, `0x55`: 0 erros em todos.
- Novos eventos NVIDIA: 0.

## Diagnostico

Nao ha evidencia atual suficiente para declarar a RTX 5060 Ti fisicamente
defeituosa. Defeito permanente de VRAM, chip, termica ou alimentacao imediata e
enfraquecido pelos testes aprovados. Eles nao excluem falha intermitente por
soak termico, pico transiente, cabo/slot ou componente degradado.

Hipoteses apos os testes:

1. **Alta:** driver 595.97 / WDDM / GPU de display compartilhada com CUDA.
2. **Media:** falha intermitente fisica, alimentacao ou contato PCIe que exige
   teste longo e isolamento de cabos/slot para reproduzir.
3. **Baixa:** VRAM ou tensor cores permanentemente defeituosos.

O ramo 595.97 e antigo em relacao aos drivers WHQL disponiveis e suas release
notes documentam um problema Blackwell capaz de causar IMA/MMU fault/XID 13 em
certos descritores TMA. O evento nao prova que o Darwin atingiu exatamente esse
bug, mas o encaixe e plausivel.

## Proximo gate recomendado

1. Instalar de forma limpa um driver NVIDIA WHQL atual que suporte as duas GPUs.
2. Ligar o monitor na Intel UHD 770/placa-mae para deixar a 5060 Ti compute-only.
3. Confirmar cabo de energia dedicado, encaixe da placa e especificacao/idade da
   fonte; manter clocks e power limit de fabrica.
4. Rodar teste longo controlado (30–60 min VRAM + compute, depois 2–4 h dual GPU)
   com telemetria e contagem de novos `nvlddmkm`/LiveKernelEvent.
5. Considerar RMA somente se a 5060 Ti voltar a gerar TDR/141 no driver novo,
   compute-only e em stock enquanto a 3060 permanece estavel.

Nenhum driver, clock, BIOS, registro, cabo ou processo de treino foi alterado.
