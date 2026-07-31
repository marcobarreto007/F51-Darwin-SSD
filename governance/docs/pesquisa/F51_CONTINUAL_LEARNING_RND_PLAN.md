# F51 Continual Learning - Plano de R&D e Concept Note PARI-CNRC

Data: 2026-07-11
Status: planejamento pre-experimental; nenhum resultado de superioridade reivindicado

## 1. Veredito executivo

A pergunta de pesquisa e relevante e financiavel:

> Um modelo com arquitetura evolutiva, rehearsal replay e governanca persistente de experts sofre menos catastrophic forgetting do que um modelo estatico equivalente sob exposicao sequencial a dominios, quando dados e compute sao controlados?

Entretanto, o repositorio atual ainda nao implementa causalmente todos os mecanismos descritos na hipotese. Antes do experimento deve existir um Gate 0 que ligue Replay e LegacyLayers ao treinamento e aos experts MoE reais.

O claim de novidade nao deve ser "ninguem combina replay e MoE". Trabalhos anteriores ja cobrem replay, replay sintetico, expansao de experts e continual MoE. A contribuicao potencialmente defensavel do F51 e:

> Governanca online do ciclo de vida de experts MoE reais, baseada em multiplos sinais de retencao, novidade, risco e consenso, com tiering persistente e recuperacao, avaliada em continual pretraining sob restricao de compute e hardware heterogeneo.

## 2. Auditoria causal do estado atual

### 2.1 O que ja afeta os pesos

| Mecanismo | Caminho causal atual | Estado |
|---|---|---|
| LM loss | forward -> loss -> backward -> optimizer | causal |
| MTP | entra na loss ponderada | causal |
| JEPA | entra na loss ponderada | causal |
| Ghost Token | masked auxiliary forward entra na loss | causal, com overhead |
| MoE routing/aux loss | experts reais em `DarwinXModel.blocks[*].moe` | causal |
| Spider Sense | estatistica detached | observacional |
| Heartbeat/Curiosity | `beat()` executado sob `torch.no_grad()` | estado/controle, nao gradiente direto |

### 2.2 Lacunas que bloqueiam a tese

**Replay nao treina o modelo.**

- `src/f51_darwin/organism/training.py` adiciona uma amostra ao buffer.
- `src/f51_darwin/organism/training.py` usa replay somente dentro de `torch.no_grad()` para avaliacao.
- Nenhum replay batch participa de `loss.backward()`.

Consequencia: o organismo atual mede uma proxy de forgetting, mas nao executa rehearsal replay.

**ExpertPool nao representa os experts MoE reais.**

- `src/f51_darwin/organism/bootstrap.py` cria os `ExpertModule` configurados.
- Os experts que processam tokens vivem em `src/f51_darwin/darwin_x.py:266-310`.
- `ExpertPool.route_active()` existe, mas nao e chamado pelo `DarwinXModel.forward()`.
- O otimizador e criado sobre `self.model.parameters()`, nao sobre o ExpertPool paralelo.

Consequencia: usage_count=0 e tier/death/resurrection desses registros nao alteram o forward do modelo treinado.

**LegacyLayers governa metadados, nao pesos ativos.**

- `src/f51_darwin/organism/lifecycle.py` move IDs entre tiers.
- Nao ha transferencia dos experts reais entre GPU/RAM/SSD, freeze, masking do router ou restauracao de pesos ligada ao forward.

Consequencia: a sobrevivencia de 72 registros nao demonstra retencao de conhecimento.

**Temperatura ainda nao e per-expert causal.**

- `src/f51_darwin/organism/lifecycle.py` calcula score a partir de loss/forgetting do ciclo e o aplica a cada record.
- Nao existe ablation real por expert nem delta de validacao por dominio atribuido ao expert.

Consequencia: os cinco sinais ainda nao provam qual expert preserva qual conhecimento.

## 3. Hipoteses pre-registradas

### H1 - Eficacia com exposicao de dados pareada

Sob mesma inicializacao, ordem de dominios, novos tokens, tokenizer, seed e agenda de avaliacao, o F51 completo apresenta menor forgetting medio que o Darwin-X estatico, sem degradar a aquisicao de novos dominios em mais de 5%.

### H2 - Eficiencia com compute pareado

Sob o mesmo orcamento de FLOPs de treino ou GPU-pair-hours, o F51 completo apresenta melhor trade-off estabilidade-plasticidade que o controle estatico e que replay uniforme.

### H0 - Hipotese nula

Nao ha diferenca de forgetting entre F51 e os controles quando dados, compute e seeds sao pareados.

### Criterio de sucesso primario

- reducao relativa de pelo menos 20% no forgetting agregado versus o controle estatico;
- intervalo de confianca de 95% da diferenca exclui zero;
- aquisicao do novo dominio nao piora mais de 5%;
- overhead de compute e memoria reportado, nao escondido.

Os limites `FR < 1.2` e `FR > 2.0` nao devem ser assumidos antes do piloto. Eles podem ser reportados como alvos exploratorios, nao como comportamento esperado do controle.

## 4. Desenho experimental

### 4.1 Dominios

Ordem primaria pre-registrada:

1. matematica;
2. codigo;
3. medicina;
4. literatura;
5. financas.

Executar tambem ordens permutadas entre seeds para evitar que o resultado dependa da posicao do dominio.

Cada dominio precisa de:

- train, validation e test sem sobreposicao;
- tamanho em tokens medido depois do tokenizer F51;
- hash do manifesto;
- licenca e proveniencia;
- deduplicacao cruzada entre dominios e contra os benchmarks.

### 4.2 Condicoes minimas

| ID | Condicao | Objetivo |
|---|---|---|
| C0 | Darwin-X estatico, sequential training | baseline ingenuo |
| C1 | Darwin-X + replay uniforme real | strong replay baseline |
| C2 | Darwin-X + continual expert baseline | comparar com expansao/freeze de experts |
| F51 | Replay + experts reais + Legacy + sinais completos | tratamento principal |

Nao chamar C0 de "transformer puro": ele continua sendo o mesmo Darwin-X SSD/GQA/MoE. A unica diferenca de C0 e remover mecanismos de continual learning.

### 4.3 Ablacoes

Depois do piloto principal:

| Ablacao | Pergunta |
|---|---|
| F51 - Replay | Legacy/predictors bastam sem rehearsal? |
| F51 - Legacy | replay explica sozinho o ganho? |
| F51 - Ghost | masked prediction ajuda retencao ou apenas plasticidade? |
| F51 - Curiosity | exploracao muda aquisicao/retencao? |
| F51 - Multi-signal | tiering simples e equivalente? |
| F51 - Resurrection | recuperacao tem efeito mensuravel? |

Executar uma ablacao somente quando o componente removido tiver caminho causal comprovado.

### 4.4 Dois experimentos, nao um

**Experimento E1 - data matched:** mesmo numero de novos tokens por dominio. Replay e Ghost geram compute adicional, que deve ser medido.

**Experimento E2 - compute matched:** mesmo total de forward/backward tokens ou GPU-pair-hours. O F51 recebe menos updates novos se gastar compute em replay/Ghost.

Sem E2 nao e correto afirmar "mesmo compute".

## 5. Metricas

Seja `L[t,i]` a negative log-likelihood no dominio `i` depois de treinar ate a fase `t`.

### Forgetting por dominio

```text
LogForgetting[i] = L[T,i] - min(L[k,i]) para k >= fase de aprendizado de i
ForgettingRatio[i] = exp(LogForgetting[i])
```

Usar NLL/log-PPL como metrica primaria evita que a exponencial da perplexidade domine a estatistica.

### Metricas complementares

- Average Forgetting e worst-domain forgetting;
- Backward Transfer (BWT);
- Forward Transfer (FWT);
- average final NLL em todos os dominios;
- new-domain acquisition penalty;
- replay ratio e bytes de memoria;
- train FLOPs aproximados, GPU-pair-hours, joules se disponivel;
- tok/s e pico de VRAM por GPU;
- routing entropy, expert utilization e domain-expert mutual information.

### Matriz de avaliacao

Avaliar todos os cinco dominios:

- antes de qualquer treino;
- depois de cada fase A, B, C, D e E;
- no checkpoint final.

Isso produz uma matriz 6 x 5 por condicao e seed. Medir apenas Dataset A perde forgetting intermediario e transferencia.

## 6. Estatistica

### Piloto

- 3 seeds pareadas;
- mesmas inicializacoes e ordens por par de condicoes;
- bootstrap paired confidence interval para diferencas;
- reportar efeito por seed, nao apenas media.

### Confirmatorio

- 5 seeds ou mais, definido pelo efeito/variancia observados no piloto;
- teste de permutacao pareado ou modelo de efeitos mistos com seed e dominio;
- intervalo de confianca de 95%;
- effect size e analise de sensibilidade;
- congelar protocolo antes de abrir o resultado final.

Evitar declarar significancia com apenas uma run por condicao.

## 7. Compute

### Inconsistencia no protocolo inicial

Com batch 1 e block 192:

```text
5.000 steps x 192 tokens = 960.000 novos tokens por dominio
25.000 steps x 192 tokens = 4.800.000 novos tokens por condicao
```

Logo, datasets de 10M tokens nao sao consumidos em 5.000 steps. Uma epoca de 10M tokens exige aproximadamente 52.084 steps por dominio, ou 260.420 steps por condicao para cinco dominios.

### Piloto recomendado

- 5.000 steps por dominio;
- 4 condicoes;
- 3 seeds;
- 300.000 steps totais;
- 57,6M novos tokens totais, antes de replay/Ghost;
- validacao completa a cada transicao de dominio.

Como o throughput atual de block 192 ainda nao foi medido de forma confiavel, usar faixa de planejamento de 100-220 tok/s por par de GPUs:

```text
uma condicao: 4,8M tokens = aproximadamente 6-13,3 GPU-pair-hours
12 runs: aproximadamente 73-160 GPU-pair-hours
com avaliacao e contingencia: aproximadamente 90-200 GPU-pair-hours
```

Antes de fechar orcamento, rodar calibracao curta de 200 steps para cada condicao e medir wall-clock/FLOPs. O controle sera mais rapido que o organismo se Ghost/replay adicionarem forwards.

### Confirmatorio grande

Cinco dominios de 10M tokens, uma epoca, quatro condicoes e cinco seeds representam 1 bilhao de novos tokens de exposicao, aproximadamente 1.260-2.780 GPU-pair-hours na faixa acima, antes de overhead. Nao iniciar esse estudo antes do piloto provar efeito.

## 8. Revisao preliminar de literatura

Escopo: busca preliminar, nao exaustiva, de trabalhos primarios 2020-2026 sobre continual language learning, replay e MoE dinamico.

| Trabalho | Mecanismo | Relevancia para F51 | Acesso |
|---|---|---|---|
| Lifelong-MoE, Chen et al. 2023 | adiciona experts por distribuicao e congela antigos | prior art direto para continual MoE | full text |
| Luo et al. 2023 | estudo empirico de forgetting em LLMs | valida o problema e desenho sequencial | abstract/page |
| SSR, Huang et al. 2024 | replay sintetico selecionado | prior art direto para Ghost/rehearsal | full text |
| SEE, Wang et al. 2025 | rehearsal + experts sequenciais | prior art direto para replay + experts | full text |
| SETA, 2026 | experts especificos/shared e crescimento adaptativo | prior art direto para modularizacao dinamica | full text |
| FOREVER, Feng et al. 2026 | replay guiado por curva de forgetting/update magnitude | prior art para replay adaptativo | abstract |

### Sintese

- Replay e uma baseline forte e conhecida.
- Expansao, congelamento e roteamento de experts para continual learning ja foram publicados.
- Replay sintetico tambem ja foi avaliado.
- A novidade F51 precisa residir no mecanismo especifico de decisao/tiering/recuperacao, na integracao causal com experts reais e em resultados sob compute restrito.
- Uma busca de patentes separada e opiniao profissional sao necessarias antes de afirmar patenteabilidade ou divulgar claims detalhados.

### Referencias primarias

- [Lifelong Language Pretraining with Distribution-Specialized Experts](https://ar5iv.labs.arxiv.org/html/2305.12281)
- [An Empirical Study of Catastrophic Forgetting in Large Language Models](https://arxiv.org/abs/2308.08747)
- [Self-Synthesized Rehearsal](https://arxiv.org/html/2403.01244v1)
- [Sequential Ensemble of Experts](https://arxiv.org/html/2504.06664v1)
- [Split-on-Share / SETA](https://arxiv.org/html/2601.17616v1)
- [FOREVER](https://arxiv.org/abs/2601.03938)

## 9. Abstract pre-registrado

> Continual pretraining enables language models to adapt to evolving data distributions but can overwrite previously acquired knowledge. We propose F51, an evolutionary continual-learning framework that combines rehearsal replay with persistent lifecycle governance of mixture-of-experts modules. Unlike fixed-capacity sequential training, F51 assigns each real routed expert a multi-signal retention state and permits freezing, tiering, reactivation, or replacement under an explicit stability-plasticity policy. We pre-register a controlled study across five sequential domains comparing static Darwin-X, uniform replay, a continual-expert baseline, and the complete F51 organism under both data-matched and compute-matched budgets. Primary outcomes are log forgetting, backward transfer, new-domain acquisition, and compute overhead. We hypothesize that F51 reduces aggregate forgetting by at least 20% without degrading new-domain acquisition by more than 5%. Results will be reported across paired random seeds with confidence intervals and component ablations. [RESULTADOS A PREENCHER APOS O EXPERIMENTO.]

## 10. Concept note PARI-CNRC

### Titulo de trabalho

F51: Adaptive Expert Lifecycle Governance for Compute-Efficient Continual Language Learning

### Problema industrial

Modelos empresariais atualizados continuamente podem esquecer conhecimento anterior e exigir retreinamento caro. Isso aumenta custo de compute, energia, downtime e risco operacional.

### Incerteza tecnologica

Nao se sabe se governar experts reais por sinais online de retencao, novidade e risco, combinando replay e tiering persistente, reduz forgetting sob compute fixo sem bloquear aquisicao de novos dominios.

### Avanco proposto

Um mecanismo de lifecycle governance causalmente ligado aos experts MoE reais, com replay adaptativo, states persistentes e recuperacao, validado por experimento sequencial reproduzivel.

### Potencial economico

- menor frequencia de full retraining;
- atualizacao incremental de modelos privados;
- reducao de compute e energia;
- IA canadense para dados empresariais dinamicos;
- IP e know-how mantidos no Canada.

### Pacotes de trabalho - 24 semanas

| WP | Semanas | Entrega |
|---|---:|---|
| WP0 Auditoria/wiring causal | 1-3 | Replay em gradiente; Legacy nos experts reais; testes causais |
| WP1 Benchmark e manifests | 4-7 | cinco dominios, splits, hashes, protocolo congelado |
| WP2 Piloto | 8-11 | 4 condicoes x 3 seeds, relatorio intermediario |
| WP3 Ablacoes | 12-16 | contribuicao de Replay, Legacy, Ghost e multi-signal |
| WP4 Confirmatorio | 17-21 | seeds adicionais e compute-matched study |
| WP5 Transferencia | 22-24 | paper tecnico, IP review, demo e plano comercial |

### Orcamento ilustrativo - 6 meses

| Categoria | Faixa CAD |
|---|---:|
| Trabalho de R&D do fundador | 35.000-50.000 |
| Engenharia/pesquisa contratada | 20.000-40.000 |
| Compute, armazenamento e backup | 10.000-25.000 |
| Dados, avaliacao e reproducibilidade | 5.000-10.000 |
| IP, contabilidade e preparacao comercial | 5.000-15.000 |
| Total indicativo | 75.000-140.000 |

Este e um budget de planejamento, nao uma afirmacao de despesas elegiveis ou percentual de financiamento. O PARI define elegibilidade com o Industrial Technology Advisor.

### Pre-requisitos institucionais

Para trabalhar com NRC IRAP, a empresa deve ser incorporada, for-profit, operar no Canada, ter ate 500 FTEs e desenvolver/comercializar produto tecnologico inovador. O projeto deve demonstrar capacidade tecnica, plano comercial e beneficios economicos no Canada. [Requisitos oficiais do NRC IRAP](https://cnrc.canada.ca/en/support-technology-innovation/financial-support-technology-innovation)

O trabalho pode ser candidato a SR&ED se buscar avanco tecnologico por investigacao sistematica e experimento no Canada. Manter registros contemporaneos de hipoteses, falhas, commits, horas, configuracoes e resultados. [CRA - elegibilidade SR&ED](https://www.canada.ca/en/revenue-agency/services/scientific-research-experimental-development-tax-incentive-program/sred-eligibility.html)

No Quebec, o CRIC substituiu creditos anteriores para exercicios iniciados depois de 25 de marco de 2025 e e voltado a corporations qualificadas; regras e limiar de exclusao exigem contador especializado. [Revenu Quebec - CRIC](https://www.revenuquebec.ca/en/press-room/tax-news/details/2025-07-09/tax-credit-for-rd-innovation-and-pre-commercialization/)

### Rota de apoio

1. organizar/incorporar a entidade e propriedade do IP;
2. preparar deck tecnico de 10 slides e concept note de 2 paginas;
3. solicitar conversa com NRC IRAP/AI Assist;
4. buscar orientacao gratuita no Innove Ici;
5. aproximar IVADO/Mila ou pesquisador de continual learning;
6. encontrar PME primo-adoptante para validacao comercial;
7. considerar Scale AI somente com consorcio, POC e business case.

Scale AI exige projeto colaborativo com mais de uma entidade, ao menos uma PME, trabalho no Canada, prontidao e impacto; pode reembolsar ate 40% de despesas elegiveis em projetos aprovados. [Scale AI - criterios](https://www.scaleai.ca/about-us/faq/)

## 11. Go/no-go gates

| Gate | Go | No-go/pivot |
|---|---|---|
| G0 causal wiring | testes mostram Replay/Legacy mudando pesos/forward reais | mecanismo permanece metadata-only |
| G1 piloto | >=20% reducao de forgetting sem perda >5% de aquisicao | efeito pequeno/inconsistente |
| G2 compute | ganho persiste sob budget pareado | ganho vem apenas de compute extra |
| G3 ablation | ao menos um mecanismo tem efeito replicavel | full system nao supera replay simples |
| G4 comercial | parceiro confirma custo real de forgetting/retraining | problema sem comprador |

Se replay uniforme explicar todo o ganho, o resultado ainda e cientificamente util, mas o claim F51 deve mudar. Se Legacy governar apenas metadata, nao deve aparecer como mecanismo anti-forgetting em pedido publico.

## 12. Proximas acoes sem codigo

1. congelar esta hipotese e os dois budgets experimentais;
2. selecionar/licenciar os cinco datasets e definir splits;
3. definir manifest schema e evaluation matrix;
4. medir 200 steps para calibrar compute de cada condicao;
5. preparar concept note de duas paginas para NRC IRAP;
6. marcar conversa com Innove Ici e um ITA do NRC IRAP;
7. obter opiniao de IP antes de publicar claims detalhados;
8. somente depois abrir tickets de implementacao para o Gate 0.

## 13. Limitacoes desta revisao

- busca de literatura preliminar, nao exaustiva;
- alguns trabalhos de 2026 foram avaliados apenas por abstract;
- nenhuma busca formal de patentes foi executada;
- compute estimado com faixa, pois throughput do block 192 nao foi calibrado;
- elegibilidade fiscal/financiamento depende da entidade e revisao profissional.
