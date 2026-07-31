# F51 Darwin-SSD - Referencia de avaliacao

Atualizado em 2026-07-11 15:45 EDT.

## Regra principal

Loss de treino, quantidade de ciclos e geracao visualmente interessante nao sao benchmark. Um checkpoint so pode ser promovido quando supera outro checkpoint no mesmo conjunto congelado, com tokenizer, codigo e parametros de avaliacao registrados.

## Baseline operacional atual

- Arquitetura: Darwin-X local, 452,3M parametros estimados.
- Checkpoint operacional validado: `organism_cycle_091.pt`.
- Estado: ciclo 91, step 45.560, AdamW, checkpoint v5.
- Smoke dual-GPU: loss 2,2647 -> 1,2065 em 10 steps.
- Replay: 2/10 updates, fracao 0,20, eval loss 1,4660.
- 72 experts ativos, 0 mortos.
- Ainda nao existe baseline de qualidade congelado associado a esse checkpoint.

A queda de loss dentro do ciclo 91 e um sinal de treino, nao prova de melhora global.

## Gate minimo por checkpoint

Salvar em um relatorio versionado:

1. SHA-256 e tamanho do checkpoint;
2. commit Git e diff local;
3. config e CLI efetivas;
4. tokenizer e vocab;
5. hash/manifesto do token bin;
6. dataset de avaliacao e hash;
7. loss e perplexidade por dominio;
8. replay loss e forgetting;
9. throughput, tempo por step e pico de VRAM por GPU;
10. amostras fixas de geracao;
11. comparacao com o checkpoint promovido anterior.

## Metricas prioritarias

| Metrica | Responde | Nao prova sozinha |
|---|---|---|
| Eval loss/PPL congelada | previsao de proximo token | factualidade |
| Replay loss | esquecimento no conjunto de replay | generalizacao ampla |
| Accuracy por dominio | math/code/PT/EN | fluencia aberta |
| Repeticao/diversidade | degeneracao de geracao | correcao |
| Tok/s e tempo/step | eficiencia | qualidade |
| Pico VRAM por GPU | capacidade operacional | escalabilidade distribuida |
| Routing por expert | equilibrio de uso | contribuicao causal |

## Primeira bateria recomendada

1. conjunto local congelado e livre de contaminacao para PPL;
2. WikiText-2 apenas apos verificar contaminacao/licenca;
3. HellaSwag e PIQA com adapter Darwin-X validado;
4. pequeno conjunto PT-BR e matematica criado fora do corpus de treino;
5. prompts fixos de identidade, codigo, portugues e raciocinio.

Antes de usar pacote externo, registre versao e guarde a saida bruta em `runs/benchmarks/`.

## Criterio de promocao

Promover somente quando o checkpoint:

- carrega integralmente e nao possui NaN/Inf;
- nao piora replay alem do limite definido;
- melhora a metrica-alvo ou mantem qualidade com ganho operacional medido;
- tem dados, commit e config reproduziveis;
- permite rollback para o checkpoint anterior.

Ate esse gate existir, use a expressao **checkpoint treinado**, nao **modelo melhor**.
