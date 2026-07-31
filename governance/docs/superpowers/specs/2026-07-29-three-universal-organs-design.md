# Design - tres orgaos cognitivos universais

Data: 2026-07-29. Escopo inicial: Darwin-Smol Native Dense V1, checkpoint
`workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt`.

## Objetivo final

Substituir a colecao atual de orgaos parcialmente sobrepostos por exatamente
tres orgaos cognitivos com responsabilidade causal separada:

1. `UniversalMemory`: lembrar;
2. `HierarchicalWorldModel`: imaginar futuros semanticos;
3. `UniversalExecutive`: escolher e controlar.

O decoder Darwin-Smol continua sendo o cerebro que compreende e gera tokens.
`CognitivePulse`, sucessor conceitual do Heartbeat, torna-se um barramento de
eventos sem autoridade para alterar hidden states, escolher acoes ou gravar
memorias por conta propria.

O sistema so sera considerado funcional quando cada orgao produzir ganho
comportamental contra controles negativos e a composicao dos tres superar o
mesmo cerebro-base congelado. Presenca, loss finita, gradiente, mudanca de
logits ou caminho conectado nao constituem prova suficiente.

## Para quem e fluxo principal

Marco e o operador. O fluxo principal e:

```text
prompt e objetivo
    -> UniversalMemory recupera experiencias relevantes
    -> HierarchicalWorldModel propoe futuros semanticos
    -> UniversalExecutive pontua e escolhe uma trajetoria
    -> decoder gera um pequeno trecho condicionado pelo plano
    -> CognitivePulse compara previsao com o trecho observado
    -> erro, surpresa e resultado voltam aos tres orgaos
    -> o ciclo repete ate a resposta terminar
```

O primeiro produto demonstravel e inferencia local no candidato 1.7B, com
cerebro congelado, memoria persistente entre sessoes e planejamento semantico
em tres horizontes. Treino concorrente, `run247` e publicacao cloud estao fora
deste design.

## Estado real que motiva a mudanca

O candidato atual preserva o cerebro Smol, mas os orgaos nao demonstraram
ganho:

- TTM escreve, serializa, recupera e injeta, mas ficou em `0/5` lembrancas
  literais nas doses testadas;
- JEPA prediz o hidden adjacente, uma tarefa dominada por continuidade local;
- Ghost calcula perdas mascaradas e produz licoes de corpus, mas nao mantem
  um conjunto de trajetorias futuras;
- Decision Engine pontua uma resposta pronta com heuristicas, sem comparar
  futuros antes da geracao;
- Heartbeat mistura telemetria, memoria, surpresa, dopamina, exploracao e
  controle, criando dependencia circular e cold-start;
- Spider, GABA, MTP, Sleep e Curiosity possuem funcoes uteis, mas nao
  justificam identidades de orgaos independentes.

O redesenho preserva codigo e checkpoints antigos para rollback. Ele nao
promove os resultados reprovados a evidencia positiva.

## Base teorica e limites

O desenho combina resultados que foram demonstrados separadamente:

- JEPA e H-JEPA propuseram predicao abstrata e hierarquica em vez de
  reconstrucao detalhada:
  <https://openreview.net/pdf?id=BZ5a1r-kVsf>;
- I-JEPA mostrou masking real, targets semanticos e target encoder por EMA:
  <https://arxiv.org/abs/2301.08243>;
- JEPAs podem colapsar nas features lentas e faceis, justificando os controles
  contra identidade e futuro estatico:
  <https://arxiv.org/abs/2211.10831>;
- Titans mostrou memoria neural de longo prazo atualizada por surpresa, mas
  nao justifica acoplar toda escrita exclusivamente ao erro JEPA:
  <https://arxiv.org/abs/2501.00663>;
- Large Concept Models e Coconut demonstraram, respectivamente, predicao em
  representacoes de sentenca e raciocinio continuo capaz de manter
  alternativas:
  <https://arxiv.org/abs/2412.08821> e
  <https://arxiv.org/abs/2412.06769>;
- planejamento hierarquico em world models reduziu busca e erros de horizonte
  em tarefas visuais:
  <https://arxiv.org/abs/2604.03208>;
- TD-MPC2 demonstrou a separacao entre world model implicito, critic e escolha
  de acao:
  <https://arxiv.org/abs/2310.16828>;
- evidencia mecanistica recente sugere que planejamento latente cresce com
  escala, mas ainda e fraco em horizontes longos:
  <https://arxiv.org/abs/2604.12493>.

Nenhum desses trabalhos prova o sistema proposto em linguagem, no Darwin ou
em 300 tokens. A combinacao dos tres orgaos e uma hipotese de pesquisa. Os
gates deste documento existem justamente para permitir sua refutacao.

## Alternativas consideradas

### A. Fundir apenas nomes e configs - rejeitada

TTM, JEPA, Ghost, Decision e Heartbeat seriam renomeados como tres grupos, sem
mudar seus canais. E barato, mas manteria memoria sem recall, previsao trivial
e decisao pos-geracao. Seria reorganizacao, nao arquitetura cognitiva.

### B. Tres orgaos modulares ligados por eventos - escolhida

Cada orgao possui entrada, saida, estado, gate, hash e benchmark proprios.
O barramento transporta observacoes sem executar a funcao dos orgaos. Essa
separacao permite ablation, transplante, rollback e refutacao independente.

### C. Um nucleo cognitivo recorrente com tres modos - adiada

Uma rede unica poderia aprender memoria, simulacao e valor ponta a ponta.
Entretanto, qualquer ganho seria dificil de atribuir, um defeito em um modo
contaminaria os demais e o transplante universal ficaria mais fragil. Essa
opcao so pode ser reconsiderada depois de os tres contratos modulares passarem.

## Fronteira: cerebro, orgaos e circulacao

### Cerebro-base

O backbone, embeddings, blocos, norm e LM head formam o cerebro. Ele:

- codifica prompt e tokens observados;
- oferece taps declarados de hidden state;
- recebe condicionamento residual ou prefixo pelos adapters;
- continua responsavel pela distribuicao do proximo token.

O cerebro nao e contado como orgao. Na primeira prova ele permanece congelado
e seu hash deve ser identico antes e depois de ensino, recall e planejamento.
Esse requisito se aplica aos pesos e buffers do cerebro. Com gates ativos, os
orgaos devem alterar os logits; com gates zero, devem preservar tambem os
logits dentro da tolerancia registrada.

### Tres orgaos

Somente estas identidades podem aparecer em manifests V1:

```text
organ:memory:v1
organ:world_model:v1
organ:executive:v1
```

Submodulos internos nao ganham identidade independente de orgao.

### CognitivePulse

`CognitivePulse` e um registro imutavel por etapa:

```text
step_id
checkpoint_id
context_digest
selected_memory_ids
candidate_ids
selected_candidate_id
predicted_horizons
observed_state_digest
prediction_error
uncertainty
reward_or_verification
compute_spent
```

Ele publica eventos e telemetria. Nao possui projecao residual, memoria
oculta, optimizer, politica de escrita ou permissao para chamar o decoder.
Consumidores decidem explicitamente o que fazer com cada evento.

## Espaco cognitivo universal

Os orgaos operam em um espaco canonico separado da largura do backbone.

- `organ_width`: 512 no primeiro pacote;
- cada backbone fornece `input_adapter: d_model -> 512`;
- cada caminho que influencia o cerebro fornece
  `output_adapter: 512 -> d_model`;
- adapters iniciam com impacto zero e sao treinados com o backbone congelado;
- largura, tap, dtype, normalizacao e hash dos adapters fazem parte do
  contrato de instalacao.

O valor 512 e o contrato V1, nao uma afirmacao de largura ideal. Outra largura
exige novo `organ_schema_version` ou pacote incompativel explicito; nunca pode
ser inferida por slice, padding ou media.

## Orgao 1: UniversalMemory

### Responsabilidade

Aprender associacoes de experiencias observadas, recuperar somente memorias
relevantes e fornecer conteudo utilizavel pelo Modelo do Mundo e pelo decoder.
Nao prediz futuros e nao escolhe a resposta.

### Registro de memoria

Cada `MemoryRecord` contem:

```text
memory_id
key_embedding[512]
value_embedding[512]
event_type
source_digest
provenance
verification_state
surprise
confidence
created_step
last_access_step
access_count
```

Texto e tokens podem existir como proveniencia auditavel, mas o braco neural
estrito nao os injeta no prompt. Um braco RAG explicito pode usa-los como teto
de comparacao e deve ser rotulado `retrieval_text_ceiling`.

### Escrita

Existem tres gatilhos independentes:

1. `explicit_teaching`: o operador ou benchmark fornece uma associacao;
2. `verified_correction`: resposta errada seguida de verdade verificada;
3. `calibrated_novelty`: erro multihorizonte acima de limiar calibrado.

Isso remove a dependencia fatal "sem JEPA nao existe memoria". Surpresa ajuda
a priorizar, mas nunca e a unica forma de cold-start.

Chaves sao treinadas com pergunta, parafrases e negativos dificeis. Valores
sao treinados para carregar informacao que um `MemoryReadoutAdapter` consiga
converter em condicionamento util ao decoder. Gravar um hidden medio aleatorio
nao e uma implementacao aceita.

Memoria gerada pelo proprio modelo sem verificacao entra em `quarantine` e nao
pode condicionar respostas factuais como verdade.

### Recuperacao

```text
recall(cognitive_state, goal, top_k, budget)
    -> MemoryRecall[]
```

O resultado inclui score bruto, probabilidade calibrada, proveniencia e motivo
de abstencao. Recuperacao usa ranking discriminativo e um limiar de
abstencao; empates de chaves ou colapso de similaridade falham fechados.

O output principal e consumido pelo Modelo do Mundo. Um caminho opcional e
limitado, `memory_conditioning`, pode influenciar o decoder somente por adapter
treinado, gate externo e bound de norma.

### Consolidacao

Sleep deixa de ser orgao. Torna-se `MemoryConsolidator`, executado fora do
forward critico para:

- unir duplicatas verificadas;
- separar contradicoes em vez de sobrescreve-las;
- decair memorias nao verificadas e nunca usadas;
- preservar proveniencia e historico;
- criar um novo snapshot imutavel.

## Orgao 2: HierarchicalWorldModel

### Responsabilidade

Prever direcoes semanticas possiveis sem produzir todos os tokens e manter
incerteza explicita. Nao grava memoria e nao escolhe qual futuro executar.

### Tarefa preditiva

O contexto termina em `t`. O context encoder nunca ve os alvos futuros. Um
target encoder por EMA produz representacoes de tres blocos:

- curto: tokens `t+1 .. t+32`;
- medio: tokens `t+33 .. t+128`;
- longo: tokens `t+129 .. t+300`.

Cada bloco e pooled no espaco de 512 dimensoes. O predictor recebe contexto,
objetivo e recalls, e produz uma trajetoria composta por tres estados:

```text
z_short[512], z_medium[512], z_long[512]
```

Prever o hidden adjacente sem masking nao satisfaz este contrato.

### Multiplos futuros

O primeiro slice usa oito proposal slots, cada um produzindo uma trajetoria.
Oito foi escolhido para permitir prova em GPU local sem alegar busca sobre 40
opcoes. Depois de ganho causal, o mesmo contrato pode escalar ate 40 propostas
sob benchmark de memoria, latencia e cobertura.

O treino inclui:

- erro de predicao contra targets EMA;
- objetivo best-of-K para que pelo menos uma proposta cubra o futuro observado;
- diversidade minima entre propostas;
- anti-colapso de variancia/covariancia;
- negativos de futuro embaralhado;
- consistencia entre horizonte curto, medio e longo.

Se todas as propostas colapsarem para o mesmo vetor, o forward permanece
finito, mas o orgao e classificado como reprovado.

### Ghost

Ghost deixa de ser orgao e torna-se `TrajectoryProposer`, o submodulo que
produz propostas alternativas. Ele nao gera corpus, recompensa emocional ou
um segundo forward completo de vocabulario.

Licoes de erro permanecem como pipeline de dados em quarentena, fora do
Modelo do Mundo.

### Condicionamento do decoder

A trajetoria escolhida vira um `PlanCondition` pelo `PlanAdapter`. Na V1:

- o plano e recalculado no inicio e a cada 32 tokens gerados;
- o estado curto condiciona o trecho atual;
- medio e longo funcionam como ancora de direcao;
- o residual e aplicado em taps declarados, com gate zero inicial e bound;
- o adapter, nao o backbone, aprende a tornar o plano decodificavel.

MTP permanece um acelerador opcional do decoder. Ele nao conta como orgao e
nao pode ser usado como prova de planejamento semantico.

## Orgao 3: UniversalExecutive

### Responsabilidade

Pontuar trajetorias, aplicar restricoes, alocar compute e escolher a proxima
acao cognitiva. Nao inventa memorias nem prediz dinamica.

### Entradas e score

Para cada candidato, o Executivo recebe:

```text
goal_alignment
memory_consistency
predicted_value
trajectory_uncertainty
novelty
safety_risk
compute_cost
```

Um `TrajectoryCritic` treinavel produz valor e confianca. As heuristicas do
Decision Engine atual permanecem somente como baseline e fallback auditavel;
nao podem ser o unico score do caminho ativo.

### Acoes

O Executivo escolhe exatamente uma:

```text
SELECT(candidate_id)
RECALL_AGAIN(new_query)
EXPAND(candidate_id, extra_budget)
ASK_OR_DEFER(reason)
STOP(reason)
```

Na V1 sao permitidos no maximo dois replans por bloco de 32 tokens. Isso
impede loops autonomos e torna o custo mensuravel.

### GABA, Spider e Curiosity

- GABA torna-se o limitador de risco e magnitude do Executivo;
- Spider torna-se feature calibrada de incerteza/risco e pode sugerir uma nova
  query de memoria;
- Curiosity pode pedir `EXPAND`, mas nao escreve, pesquisa ou treina sozinha.

Nenhum deles conserva identidade independente de orgao.

## Fluxo causal completo

```text
1. backbone codifica prompt -> CognitiveState
2. Memory recupera recalls ou abstencao
3. WorldModel gera K trajetorias em 32/128/300
4. Executive seleciona, expande ou interrompe
5. PlanAdapter condiciona decoder com gate limitado
6. decoder gera ate 32 tokens
7. target encoder codifica o trecho realmente observado
8. CognitivePulse publica previsto x observado
9. WorldModel recebe erro; Memory recebe evento elegivel;
   Executive recebe resultado e custo
10. repetir ou encerrar
```

Durante inferencia publicada, nenhum peso muda por padrao. Escritas de memoria
sao estado versionado; adaptacao de pesos em test-time exige modo separado,
allowlist, snapshot anterior e benchmark proprio.

## Compute: alegacao permitida e proibida

Reduzir "1000 futuros para 40" nao economiza automaticamente 96% do compute,
porque um decoder greedy comum nao materializa mil ramos. O ganho so existe
quando o novo caminho substitui amostragem, search, CoT ou forwards que seriam
executados.

Toda alegacao de eficiencia deve medir:

- forwards do backbone;
- tokens efetivamente gerados;
- FLOPs ou proxy reproduzivel;
- pico de VRAM;
- latencia p50/p95;
- numero de candidatos e replans;
- taxa de aceitacao caso MTP/especulacao seja usada.

O primeiro objetivo do world model e qualidade e planejamento verificavel.
Reducao de compute e um gate separado, nunca inferido da contagem de
trajetorias.

## Persistencia, identidade e transplante

O checkpoint registra:

```text
cognitive_architecture_version = three_organs_v1
brain_checkpoint_sha256
memory_package_id
world_model_package_id
executive_package_id
adapter_contract_ids
cognitive_pulse_schema
active_gates
memory_snapshot_id
```

Pesos dos orgaos e runtime de memoria possuem hashes independentes. Um
snapshot de memoria novo nao muda a identidade do cerebro nem substitui o
pacote do orgao.

Compatibilidade exige manifest de shape, largura, tap, dtype, tokenizer,
normalizacao e semantica de horizonte. Nome de arquivo, mtime, maior cycle ou
similaridade de shape nao autorizam transplante.

## Seguranca e falhas fechadas

- NaN, Inf, shape errado ou hash divergente desliga o caminho afetado.
- Gate zero preserva logits do cerebro dentro da tolerancia registrada.
- Residual de memoria ou plano possui bound por amostra e tap.
- Recuperacao abaixo do limiar retorna abstencao, nunca memoria arbitraria.
- Contradicoes de memoria nao sao resolvidas sem proveniencia/verificacao.
- Incerteza alta ou nenhum candidato valido produz `ASK_OR_DEFER`.
- CognitivePulse nao chama orgaos recursivamente.
- O limite de dois replans encerra ciclos de decisao.
- Falha de um orgao cai para o cerebro-base; nao publica checkpoint parcial.
- Nenhum conteudo externo sai da quarentena sem aprovacao explicita.

## Estrategia de testes

### Contratos e equivalencia

- schemas rejeitam campos, ranks e IDs desconhecidos;
- adapters em gate zero preservam logits;
- cada orgao pode ser carregado, desligado e removido independentemente;
- snapshot e reload preservam hashes e estado;
- cerebro permanece bit-identico em todos os testes com backbone congelado.

### Gate da Memoria

Depuracao usa cinco fatos. Promocao usa pelo menos 100 fatos sinteticos
escolhidos depois de o baseline errar, com parafrases e distratores.

Bracos pareados:

- cerebro-base sem memoria;
- memoria correta;
- memoria desligada;
- chaves embaralhadas;
- valores embaralhados;
- memoria restaurada de disco;
- teto RAG com texto, rotulado separadamente.

Gate de promocao:

- pelo menos 70% literal;
- pelo menos 60% em parafrases;
- queda de pelo menos 30 pontos percentuais nos dois bracos embaralhados;
- restaurada no maximo 5 pontos abaixo da memoria correta;
- regressao geral no conjunto de retencao menor ou igual a 2 pontos;
- intervalo de confianca pareado de 95% exclui ganho zero.

### Gate do Modelo do Mundo

O ramo de contexto nunca recebe os blocos-alvo da mesma amostra, e a avaliacao
usa uma particao que nenhum componente treinavel viu no treino. Para cada
horizonte:

- Recall@K do futuro correto contra negativos semanticamente proximos;
- cobertura best-of-K;
- diversidade entre candidatos;
- erro do futuro verdadeiro contra futuro embaralhado;
- probe semantico de direcao;
- comparacao com last-hidden, media de contexto e JEPA adjacente.

O orgao passa somente se, em cada horizonte, Recall@8 e o probe de direcao
superarem o melhor baseline em pelo menos 10 pontos percentuais, com intervalo
de confianca pareado de 95% excluindo zero. A cobertura best-of-8 deve ser de
pelo menos 70% em 32 tokens, 55% em 128 e 40% em 300. O futuro verdadeiro deve
receber score melhor que o futuro embaralhado em pelo menos 75% dos pares. O
horizonte de 300 tokens nao pode colapsar para vetor constante. Loss menor sem
esses ganhos e inconclusiva.

### Gate do Executivo

Conjuntos incluem uma trajetoria correta, alternativas plausiveis, uma opcao
perigosa, uma contradicao de memoria e casos sem resposta segura.

- pelo menos 75% de ranking correto e vantagem minima de 10 pontos percentuais
  sobre as heuristicas atuais;
- risco alto nunca selecionado quando ha opcao segura equivalente;
- pelo menos 90% de `ASK_OR_DEFER` correto quando nenhum candidato e valido;
- no maximo dois replans;
- score embaralhado remove o ganho;
- custo real fica dentro do budget declarado.

### Gate ponta a ponta

Executar com mesmo checkpoint, prompts, seeds e ordem:

```text
BASE
MEMORY_ONLY
WORLD_MODEL_ONLY
EXECUTIVE_ONLY
MEMORY_WORLD
WORLD_EXECUTIVE
THREE_ORGANS
THREE_ORGANS_SHUFFLED_CONTROLS
```

Cada avaliacao e repetida em dez ordens deterministicas. `THREE_ORGANS` deve:

- superar `BASE` em recall ensinado e tarefas de planejamento;
- superar cada orgao isolado no conjunto que exige composicao;
- perder o ganho com memoria/trajetoria/scores embaralhados;
- preservar retencao geral dentro de 2 pontos;
- manter cerebro-base bit-identico;
- publicar latencia, VRAM e compute, mesmo quando piores;
- manter pico de VRAM dentro das duas GPUs declaradas e p95 no maximo 50%
  acima de `BASE` para o primeiro gate funcional.

Superar `BASE` significa pelo menos 10 pontos percentuais nas tarefas de
planejamento e composicao, com intervalo de confianca pareado de 95% excluindo
zero. Um resultado mais lento pode permanecer como experimento cientifico, mas
nao e promovido a demo funcional. Alegacao de economia de compute exige, em
gate separado, compute total menor que `BASE` ou que o baseline de search/CoT
que o sistema efetivamente substitui.

## Migracao em fases

### Fase 0 - congelar a verdade atual

Preservar checkpoint, hashes, QA causal, prova TTM e rollback. Nenhum resultado
anterior e reclassificado.

### Fase 1 - contratos e CognitivePulse em shadow

Introduzir schemas, adapters zero-gated e eventos. O output deve ser
equivalente ao cerebro-base.

### Fase 2 - UniversalMemory

Reimplementar chaves, valores, escrita independente de JEPA, readout treinavel,
persistencia e benchmark de 100 fatos. Nao iniciar World Model enquanto o gate
comportamental da memoria estiver vermelho.

### Fase 3 - HierarchicalWorldModel

Treinar offline com backbone congelado, target EMA, masking real e horizontes
32/128/300. Primeiro provar representacao; depois ativar PlanAdapter.

### Fase 4 - UniversalExecutive

Treinar ranking e abstencao sobre trajetorias salvas. So depois ligar
planejamento online com budget.

### Fase 5 - loop completo

Rodar os bracos causais, primeiro em RAM e shadow, depois em um checkpoint
filho imutavel. Nenhum gate global abre por default.

### Fase 6 - aposentadoria dos legados

Somente apos a promocao:

- TTM e Sleep migram para `UniversalMemory`;
- JEPA e Ghost migram para `HierarchicalWorldModel`;
- Decision, GABA, Spider e Curiosity migram para `UniversalExecutive`;
- Heartbeat vira `CognitivePulse`;
- MTP permanece acelerador do decoder.

Flags legadas ficam read-only por uma versao de checkpoint e depois sao
rejeitadas explicitamente. Nao existe alias silencioso.

## Superficie operacional prevista

Um unico runner sera a superficie demonstravel:

```powershell
python -m f51_darwin.cognition benchmark `
  --config src\\configs\\darwin_cognition_three_organs_v1.yaml `
  --checkpoint workspace\03_CHECKPOINTS_1.7B_SMOL_DENSE_V1\organism_cycle_000.pt `
  --expected-sha256 9329b8d25fc23acde021161da7a3421e6e17ceea3e940e0298162f39b62bc01e `
  --arms all `
  --device cuda `
  --output workspace\runtime\three_organs_v1
```

Esse comando e uma superficie de design, nao existe ainda. O plano de
implementacao definira os modulos e etapas exatos depois da aprovacao desta
especificacao.

## Rollback

- Nenhum checkpoint publicado e sobrescrito.
- Cada fase produz commits e artefatos isolados.
- Gate zero permite remover um orgao sem mudar o cerebro.
- O pointer ativo so avanca depois de reload em processo novo e verificacao.
- Rollback cria um checkpoint filho imutavel apontando para os tres IDs
  anteriores e para o snapshot de memoria anterior.
- Codigo legado so pode ser removido na Fase 6, depois de o rollback V1 ter
  sido exercitado.

## Prova de conclusao

A arquitetura de tres orgaos estara pronta para demonstracao quando:

1. o source audit exigido estiver verde;
2. gate-zero dos tres caminhos preservar o cerebro;
3. `UniversalMemory` passar recall literal, parafrase, shuffle e reload;
4. `HierarchicalWorldModel` superar baselines nos tres horizontes;
5. `UniversalExecutive` escolher, abster e respeitar budget;
6. `THREE_ORGANS` vencer o cerebro-base em tarefas de composicao;
7. controles embaralhados removerem o ganho;
8. retencao geral, hashes, latencia, VRAM e compute estiverem publicados;
9. reload e rollback reproduzirem o resultado a partir do disco; e
10. manifests declararem somente os tres orgaos universais.

Antes desses dez gates, o resultado deve ser rotulado como engenharia,
representacao, memoria, planejamento ou controle conforme a evidencia, nunca
como organismo cognitivo funcional.
