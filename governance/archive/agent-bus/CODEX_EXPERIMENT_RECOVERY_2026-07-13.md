# Recuperacao do experimento C0/C1/F51 — 2026-07-13

## Objetivo fechado

Executar uma unica validacao controlada do Darwin-X, com os cenarios estruturais
separados do benchmark de esquecimento, um processo usando as duas GPUs, log
persistente e resultados sem colisao.

## Causa raiz comprovada

- Tres `experiment_runner.py` foram iniciados em paralelo e cada um abriu contexto
  nas duas GPUs.
- Os processos disputaram aproximadamente 61 GB de 64 GB de RAM e gravariam no
  mesmo `runs/experiment_results.json`.
- Em `C1_Replay`, o runner armazenava o batch ja misturado com replay. A amostra
  crescia recursivamente a cada reutilizacao, pressionando RAM/VRAM.
- `scripts/test_10_cenarios.py` era um arquivo de duas linhas com erro de sintaxe;
  os cenarios haviam sido colados no inicio do benchmark.
- O antigo Topology Manifest chamava indices como `L0_E0` de UUID estavel e o
  runtime v6 carregava tensores antes de reconstruir a topologia.

## Correcoes salvas

- `scripts/experiment_runner.py`
  - trava de processo via `msvcrt`, liberada automaticamente em crash;
  - um processo e model parallel nas duas GPUs;
  - replay guarda apenas a amostra fresca de batch 1;
  - log line-buffered, status, resultado parcial e resultado final por `run_id`;
  - escrita JSON atomica e ponteiro `runs/experiments/latest.json`;
  - copia somente o bloco amostrado do memmap antes da conversao para tensor.
- `f51_darwin/darwin_x.py`
  - UUID persistente, pai e passo de nascimento por especialista;
  - compactacao coerente de router, buffers, dispositivos e identidades na apoptose;
  - crescimento em CPU/GPU sem dispositivo `cuda:0` implicito;
  - Manifest v7 com ordem do router, linhagem e estado neuroendocrino;
  - restore da anatomia, IDs e hormônios antes dos tensores;
  - organismo novo inicia com dopamina basal neutra.
- `scripts/darwin_organism.py`
  - checkpoint v7 inclui `topology_manifest`;
  - resume v7 reconstrói topologia antes de `load_state_dict(strict=True)`.
- `tests/test_topology_manifest_v7.py`
  - dez cenarios causais reais: nascimento, crescimento, morte, DNA, restore,
    Ghost, roundtrip funcional, ações seguras, capability gap e AdamW resume.
- `scripts/test_10_cenarios.py`
  - entrada simples para executar os dez testes pytest reais.

## Evidencia

- Cenários v7: `10 passed in 2.20s`.
- Regressões focadas: `27/27`.
- Suíte completa final: `204/204`.
- Teste HTTP intermitente WinError 10053: passou `5/5` isolado; suíte completa
  seguinte passou integralmente.
- Smoke dual-GPU final: código de saída `0`, split `6/6`, três condições concluídas.
- Artefatos do smoke válido:
  `runs/experiments/smoke_final_1783994958442/`.
- `runs/experiments/latest.json` aponta para esse smoke com status `completed`.
- Após o smoke: nenhum `experiment_runner.py` ou `darwin_organism.py run247`
  ativo; GPUs livres para o próximo lançamento.

## Limite desta entrega

O smoke usa `darwin_x_600m_ckpt239.yaml` e uma etapa por fase. Ele prova execução,
isolamento, dual-GPU e persistência; seus números de perplexidade não são resultado
científico. Nenhum treino longo 600M foi iniciado, porque o objetivo canônico do
repo continua sendo a linhagem treinada da classe 2.5B.

## Estado Git

O repositório `.git` existe e aponta para `master`, mas nenhum executável Git está
instalado/encontrável no host. As alterações estão persistidas no disco, porém esta
sessão não criou commit.
