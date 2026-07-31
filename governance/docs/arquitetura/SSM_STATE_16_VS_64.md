# Decisao tecnica: `ssm_state=16` versus `64`

Verificado localmente em 2026-07-13.

## O que muda

`ssm_state` define quantos canais de estado cada canal interno do bloco SSM mantém para representar e acumular informacao temporal. Passar de 16 para 64 quadruplica essa dimensao interna. Isso pode aumentar a capacidade teorica de representar dinamicas temporais, mas nao garante qualidade maior sem treino e benchmark comparaveis.

## Impacto no Darwin-X atual

| Medida | Estado 16 | Estado 64 |
|---|---:|---:|
| Parametros estimados do modelo | 452.269.384 | 455.918.920 |
| Diferenca total | referencia | +3.649.536 (+0,81%) |
| Pico de VRAM em um bloco SSM | 768,8 MiB | 2.792,5 MiB |
| Forward + backward mediano | 38,6 ms | 153,5 ms |

Microbenchmark: `SelectiveSSM`, `d_model=1408`, `expand=2`, batch 1, sequencia 192, BF16, RTX 5060 Ti; uma iteracao de aquecimento e mediana de tres iteracoes. O estado 64 consumiu aproximadamente 3,6 vezes mais VRAM e foi aproximadamente 4 vezes mais lento neste kernel PyTorch puro.

## Compatibilidade

O checkpoint `organism_cycle_239.pt` foi treinado com `ssm_state=16`. Seus tensores sao internamente consistentes com essa configuracao. O YAML atual em `src/configs/darwin_x_600m.yaml` usa `64`, causando 18 incompatibilidades de shape. Portanto nao existe resume fiel direto de 16 para 64.

## Teste do modelo completo com estado 64

O modelo Darwin-X completo foi testado com pesos FP32, autocast BF16, batch 1, Adafactor e split 7/5 nas RTX 5060 Ti + RTX 3060.

| Block size | Resultado | Pico GPU 0 | Pico GPU 1 |
|---:|---|---:|---:|
| 16 | forward, backward e optimizer aprovados | 3.880,5 MiB | 1.796,6 MiB |
| 64 | forward, backward e optimizer aprovados | 10.032,6 MiB | 4.965,7 MiB |
| 128 | OOM na GPU 0 | 15.360,4 MiB | 5.393,8 MiB |

Em `block_size=64`, cinco passos consecutivos foram aprovados, com gradientes em `cuda:0` e `cuda:1`, cinco updates do optimizer, memoria alocada estavel e aproximadamente 50,13 tokens/s no smoke. O relatorio bruto esta em `runs/validation/ssm_state64_20260713.json`.

Isso prova que `ssm_state=64` e operacional no hardware atual quando combinado com `block_size=64`. Nao prova superioridade de qualidade, porque o teste usou inicializacao aleatoria e nao existe checkpoint 64 treinado.

## Recomendacao

Para continuar a linhagem existente, usar `ssm_state=16` e retomar o checkpoint 239. E a unica opcao que preserva integralmente o aprendizado e o optimizer existentes.

Para abrir uma linhagem nova, `ssm_state=64` esta tecnicamente aprovado com `block_size=64` nas duas GPUs. Ele deve permanecer separado do checkpoint 239 e so deve substituir a linhagem 16 se um benchmark congelado provar ganho de qualidade suficiente para compensar o custo. `block_size=128` e `192` nao sao viaveis no kernel atual sem checkpointing, scan otimizado ou redistribuicao de camadas.
