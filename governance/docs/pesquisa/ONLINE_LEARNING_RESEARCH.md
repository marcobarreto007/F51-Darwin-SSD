# Aprendizado online no Darwin-X: evidência, limites e direção de implementação

**Data da revisão:** 2026-07-12  
**Escopo:** test-time training/adaptation, memória atualizável, continual pretraining com replay e ingestão segura para um LM local de aproximadamente 0,5B parâmetros.

## Conclusão executiva

O caminho cientificamente defensável para o Darwin-X atual não é modificar os pesos principais a cada página pesquisada. A primeira versão operacional deve manter o backbone congelado durante a conversa, persistir evidências textuais com proveniência e recuperá-las como contexto citado. Atualização paramétrica deve ocorrer em uma etapa separada de consolidação, sobre um estado candidato, com replay, avaliação antes/depois e rollback.

O `Heartbeat` atual **não é uma implementação fiel de Titans, Quiet-STaR ou Forward-Forward**. Os nomes descrevem inspirações, não contratos implementados:

- `src/f51_darwin/heartbeat.py::TestTimeMemory` anexa vetores quando um escalar `jepa_error` supera um limiar, recupera por cosseno e remove slots pouco acessados. Titans atualiza os pesos de um modelo de memória por gradiente de uma loss associativa, com surpresa baseada nesse gradiente, momentum e forgetting por weight decay adaptativo.
- `ForwardForwardLayer` não recebe exemplos positivos e negativos nem otimiza uma loss local contrastiva de goodness. Após `LayerNorm`, seu sinal de goodness é quase independente do conteúdo, e a atualização apenas desloca escalas pelo sinal da “dopamina”.
- `QuietStarThinker` produz ruído ao redor de tokens aprendíveis e usa um avaliador não treinado. Quiet-STaR exige continued pretraining para aprender a gerar e usar racionales token a token.
- `src/f51_darwin/inference_learner.py::learn` executa um forward em modo de treino, mas não chama `backward()` nem `optimizer.step()`. Portanto, não consolida conhecimento nos pesos principais.
- O estado salvo pelo Heartbeat registra contadores, mas não serializa o conteúdo dos slots, as projeções da memória nem um optimizer de aprendizado online.

## Método da revisão

Foram priorizadas fontes primárias em arXiv e OpenReview. Foram incluídos trabalhos que apresentam um mecanismo executável e evidência empírica relevante para pelo menos uma das quatro perguntas abaixo:

1. O que realmente constitui aprendizado no teste?
2. Como memória pode ser atualizada sem reescrever todo o backbone?
3. Como reduzir esquecimento em atualização contínua?
4. Como impedir que conteúdo recuperado contamine memória ou pesos?

Surveys, blogs e descrições comerciais foram excluídos. Resultados de visão são usados apenas como princípios de controle e são explicitamente marcados como não validados para LMs. Esta é uma revisão direcionada à implementação, não uma revisão sistemática exaustiva.

## 1. Test-time training e adaptação

### Test-Time Training with Self-Supervision

**Fonte primária:** Sun et al., ICML 2020 — [arXiv:1909.13231](https://arxiv.org/abs/1909.13231)

**Mecanismo demonstrado:** cria, a partir de cada entrada não rotulada, uma tarefa self-supervised e executa atualização por gradiente antes da predição. O paper também considera a extensão para streams online.

**Evidência e limites:** os experimentos são de classificação visual sob distribution shift. O ganho depende de a tarefa auxiliar compartilhar informação útil com a tarefa principal. Isso não prova que next-token training sobre qualquer texto recuperado melhorará um LM.

**Implicação F51:** um simples forward não é TTT. Se uma adaptação temporária for adicionada, ela precisa de loss explícita, parâmetros autorizados, optimizer, reset definido e comparação antes/depois.

### EATA: Efficient Test-Time Model Adaptation without Forgetting

**Fonte primária:** Niu et al., ICML 2022 — [arXiv:2204.02610](https://arxiv.org/abs/2204.02610)

**Mecanismo demonstrado:** seleciona amostras confiáveis e não redundantes para adaptação e limita mudança em parâmetros importantes com uma regularização baseada em Fisher.

**Evidência e limites:** validado em visão, com entropy minimization e pseudo-labels; não é evidência direta para causal language modeling.

**Implicação F51:** duas ideias devem virar gates: nem todo documento merece atualização, e qualquer update deve permanecer ancorado ao checkpoint anterior. A Fisher regularization é opcional; seleção de amostra e rollback não são.

### Test-Time Training for Few-Shot Learning

**Fonte primária:** Akyürek et al. — [arXiv:2411.07279](https://arxiv.org/abs/2411.07279)

**Mecanismo demonstrado:** constrói tarefas leave-one-out a partir de exemplos de demonstração e executa pequenos passos de gradiente em adapters LoRA específicos da tarefa. Os parâmetros adaptados são temporários e usados para resolver aquela tarefa.

**Evidência e limites:** avaliado em modelos de 1B, 3B e 8B em ARC e BIG-Bench Hard. O resultado depende de demonstrações rotuladas/estruturadas, augmentations e preparação da base. Não demonstra aprendizado persistente e irrestrito sobre páginas da web.

**Implicação F51:** TTT interativo, se desejado, deve viver em um delta/adapter isolado e descartável. Não deve alterar silenciosamente o checkpoint canônico.

### TTT Layers: Learning to Learn at Test Time

**Fonte primária:** Sun et al. — [arXiv:2407.04620](https://arxiv.org/abs/2407.04620)

**Mecanismo demonstrado:** transforma o estado oculto recorrente em um pequeno modelo linear ou MLP. Esse estado é atualizado por gradiente em uma tarefa de reconstrução multi-view cujas projeções são aprendidas no outer loop.

**Evidência e limites:** avaliado de 125M a 1,3B parâmetros e capaz de aproveitar contexto longo melhor que Mamba em parte dos experimentos. O próprio paper registra custo de wall-clock e I/O. É uma arquitetura treinada end-to-end, não um módulo plug-and-play para um checkpoint existente.

**Implicação F51:** a escala é compatível com Darwin-X, mas a adoção exige um experimento de arquitetura e novo treinamento. Não deve bloquear a entrega da memória persistente externa.

## 2. Memória atualizável

### Titans: Learning to Memorize at Test Time

**Fonte primária:** Behrouz, Zhong e Mirrokni — [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)

**Mecanismo demonstrado:** a memória de longo prazo é uma rede neural treinada no inner loop para associar chaves e valores. A “surpresa” é derivada do gradiente da loss da memória; momentum mantém o efeito de eventos surpreendentes e weight decay funciona como forgetting adaptativo. O outer loop aprende as projeções e a forma de usar a memória.

**Evidência e limites:** os autores treinaram variantes de 170M, 340M, 400M e 760M parâmetros em 15B ou 30B tokens. Momentum, forgetting e memória persistente contribuíram nas ablações. A arquitetura foi treinada desde o início com o mecanismo.

**Distinção terminológica importante:** “persistent memory” no paper são parâmetros fixos, independentes da entrada, que codificam meta-informação da tarefa. O termo não significa persistência em arquivo ou sobrevivência automática a restart.

**Implicação F51:** o cache atual deve ser descrito como cache vetorial, não como Titans. Uma implementação Titans fiel precisa de inner optimizer, loss K→V, momentum, decay, outer-loop training e testes numéricos do update.

### Memorizing Transformers

**Fonte primária:** Wu et al., ICLR 2022 — [arXiv:2203.08913](https://arxiv.org/abs/2203.08913)

**Mecanismo demonstrado:** mantém uma memória externa não diferenciável de pares de atenção `(key, value)`, recupera vizinhos por kNN aproximado e combina memória e atenção local por gate aprendido.

**Evidência e limites:** perplexidade melhorou à medida que a memória cresceu até 262K tokens; modelos usaram definições e teoremas vistos anteriormente. Porém, o leitor da memória foi treinado para usá-la, e chaves antigas ficam stale quando o encoder muda.

**Implicação F51:** persistir texto e proveniência é mais seguro na primeira versão. Uma memória de hidden states só deve virar fonte principal depois que o reader for treinado e a staleness for medida.

### LongMem

**Fonte primária:** Wang et al. — [arXiv:2306.07174](https://arxiv.org/abs/2306.07174)

**Mecanismo demonstrado:** congela o backbone como encoder de memória e treina uma SideNet residual para recuperar e fundir pares K/V de uma memória não diferenciável. A separação evita que mudanças no reader invalidem o encoder e protege o conhecimento do backbone.

**Evidência e limites:** usa memória de 65K tokens e treinamento de adaptação em aproximadamente 26B tokens. A memória não é útil apenas por existir; a SideNet foi treinada para lê-la.

**Implicação F51:** é a referência arquitetural mais segura para uma futura memória neural: backbone congelado + reader pequeno treinável + banco externo. Ainda assim, a primeira entrega pode usar retrieval textual sem SideNet.

### MemoryLLM

**Fonte primária:** Wang et al. — [arXiv:2402.04624](https://arxiv.org/abs/2402.04624)

**Mecanismo demonstrado:** separa parâmetros estáticos de um pool latente atualizável. Novos hidden states substituem gradualmente uma fração dos tokens de memória, produzindo forgetting exponencial controlado.

**Evidência e limites:** o protótipo adiciona aproximadamente 1B parâmetros de memória a um backbone de 7B e reporta integridade após centenas de milhares de updates. A escala e o treinamento necessário impedem transplante direto para Darwin-X.

**Implicação F51:** reforça que memória atualizável deve ser separada do conhecimento estável, ter capacidade fixa e política explícita de substituição. Não justifica chamar uma lista RAM de MemoryLLM.

## 3. Os papers citados pelo Heartbeat

### Forward-Forward

**Fonte primária:** Hinton — [arXiv:2212.13345](https://arxiv.org/abs/2212.13345)

**Mecanismo demonstrado:** dois passes, um com dados positivos e outro com dados negativos. Cada camada aprende localmente a produzir goodness alto para positivos e baixo para negativos.

**Evidência e limites:** o trabalho é explicitamente preliminar e demonstra poucos problemas pequenos; não apresenta validação de LM causal nessa escala.

**Implicação F51:** ou o organismo implementa exemplos positivos/negativos, objetivos locais e testes de separação de goodness, ou remove a alegação de aprendizado Forward-Forward. “Dopamina” pode ponderar uma loss real, mas não substituí-la.

### Quiet-STaR

**Fonte primária:** Zelikman et al. — [arXiv:2403.09629](https://arxiv.org/abs/2403.09629)

**Mecanismo demonstrado:** amostra racionales internos em paralelo entre tokens, usa tokens aprendíveis de início/fim e treina o modelo a prever texto futuro combinando predição com e sem pensamento.

**Evidência e limites:** após continued pretraining, GSM8K subiu de 5,9% para 10,9% e CommonsenseQA de 36,3% para 47,2%. O método aumenta custo e exige treinar geração e uso dos pensamentos.

**Implicação F51:** ruído aleatório projetado no `lm_head` não é Quiet-STaR. Uma implementação fiel é um projeto de treinamento separado, com benchmark de perplexidade e raciocínio.

## 4. Continual learning e replay

### Lifelong Pretraining

**Fonte primária:** Jin et al. — [arXiv:2110.08534](https://arxiv.org/abs/2110.08534)

**Mecanismo demonstrado:** continual pretraining em streams de papers e tweets, comparando adapters, regularização, replay e distillation. O replay principal mantém 100 mil exemplos balanceados entre domínios e insere um minibatch antigo a cada dez steps.

**Evidência e limites:** distillation reteve melhor desempenho antigo em vários experimentos, mas foi mais cara e dependente da tarefa. O paper também mostra que replay excessivo pode sobreajustar a memória.

**Implicação F51:** uma consolidação deve misturar dados novos com anchors antigos e medir retenção. Mais replay não é automaticamente melhor.

### TiC-LM

**Fonte primária:** Li et al., ACL 2025 — [arXiv:2504.02107](https://arxiv.org/abs/2504.02107)

**Mecanismo demonstrado:** continual pretraining temporal em Common Crawl, comparando schedules, EWC/LwF e diferentes razões de replay. Uma política reserva parte fixa do orçamento para o mês atual e distribui o restante entre dados antigos.

**Evidência e limites:** em modelos de 1B e 3B, replay reduziu aproximadamente 60% do regret de backward transfer. Uma razão de metade atual/metade antiga ofereceu bom compromisso geral, mas replay excessivo prejudicou adaptação in-distribution e domínios que mudam rápido. Não há resultado direto para Darwin-X de 0,5B.

**Implicação F51:** testar pelo menos 10%, 20% e 50% de replay e escolher pela fronteira plasticidade/retenção; não fixar “10% Ghost” sem medição.

### Maximally Interfered Retrieval

**Fonte primária:** Aljundi et al., NeurIPS 2019 — [arXiv:1908.04742](https://arxiv.org/abs/1908.04742)

**Mecanismo demonstrado:** realiza uma atualização virtual e prioriza exemplos antigos cuja loss seria mais prejudicada, em vez de replay aleatório.

**Evidência e limites:** validado em continual learning supervisionado, não em causal LM. Exige computação adicional para estimar interferência.

**Implicação F51:** começar com replay estratificado e reprodutível; adicionar seleção por interferência somente se o benchmark mostrar que o buffer pequeno é o gargalo.

## 5. Pesquisa, RAG e ingestão segura

### PoisonedRAG

**Fonte primária:** Zou et al. — [arXiv:2402.07867](https://arxiv.org/abs/2402.07867)

**Mecanismo demonstrado:** otimiza poucos textos maliciosos para serem recuperados por uma pergunta-alvo e induzirem uma resposta escolhida pelo atacante.

**Evidência e limites:** cinco textos por pergunta atingiram cerca de 90% de attack success rate em bases com milhões de textos. Defesas simples avaliadas foram insuficientes.

**Implicação F51:** resultado de busca entra como candidato em quarentena, nunca diretamente em pesos ou memória aprovada. O fluxo precisa medir poison-ASR.

### BIPIA e Spotlighting

**Fontes primárias:** Yi et al. — [arXiv:2312.14197](https://arxiv.org/abs/2312.14197); Hines et al. — [arXiv:2403.14720](https://arxiv.org/abs/2403.14720)

**Mecanismos demonstrados:** BIPIA mede indirect prompt injection e treina/induz boundary awareness; Spotlighting transforma continuamente texto externo para sinalizar sua proveniência e separá-lo das instruções confiáveis.

**Evidência e limites:** BIPIA encontrou vulnerabilidade ampla nos modelos avaliados. Spotlighting reduziu attack success rate de mais de 50% para menos de 2% em modelos GPT, com pouco impacto nas tarefas benignas. Esses números não podem ser transferidos automaticamente para Darwin-X.

**Implicação F51:** conteúdo recuperado deve ser delimitado como citação não executável, manter `source_id` por chunk e passar por casos adversariais locais antes da promoção.

### Corrective RAG

**Fonte primária:** Yan et al. — [arXiv:2401.15884](https://arxiv.org/abs/2401.15884)

**Mecanismo demonstrado:** usa um avaliador de retrieval para classificar a qualidade dos documentos, decide entre aceitar/corrigir/buscar novamente e decompõe documentos para remover partes irrelevantes.

**Evidência e limites:** melhorou RAG em quatro datasets, mas é um controle de relevância e robustez, não uma prova contra adversários ou falsidades coordenadas.

**Implicação F51:** relevância, confiança de fonte, contradição e prompt injection devem ser gates separados. Um score único não deve promover dados sozinho.

### Deduplicação de dados de treino

**Fonte primária:** Lee et al. — [arXiv:2107.06499](https://arxiv.org/abs/2107.06499)

**Mecanismo demonstrado:** deduplicação exata e aproximada de documentos e substrings repetidos antes do treino.

**Evidência e limites:** reduziu emissão de texto memorizado em aproximadamente 10 vezes e identificou overlap em mais de 4% de conjuntos de validação padrão. O estudo não resolve qualidade factual ou segurança.

**Implicação F51:** `content_hash` é necessário, mas não suficiente. O fluxo deve detectar near-duplicates e decontaminar os conjuntos congelados de avaliação.

## Arquitetura recomendada para a primeira entrega

```text
pesquisa
  -> candidato imutável com URL, data, licença e hash
  -> normalização + dedup exata/aproximada
  -> firewall de qualidade e prompt injection
  -> quarentena
  -> validação de relevância e contradição
  -> promoção explícita
  -> memória textual persistente
  -> retrieval com fonte e citação
  -> resposta com backbone congelado
  -> fila de consolidação
  -> treino de estado candidato com replay
  -> benchmark antes/depois
  -> promoção ou rollback
```

### Fase 1 — memória segura sem alterar pesos

- Persistir texto normalizado e metadados: `source_url`, `fetched_at`, `content_hash`, licença, domínio, versão, status e motivo da decisão.
- Manter estados explícitos `candidate`, `quarantine`, `approved` e `rejected`.
- Usar retrieval lexical/BM25 como baseline determinístico. Hidden-state retrieval pode entrar depois, como método adicional avaliado.
- Injetar trechos aprovados como conteúdo não confiável delimitado, sempre acompanhado de `source_id` e URL.
- Manter o modelo em `eval()`/`no_grad()` durante resposta.
- Reportar separadamente: “armazenado”, “recuperado”, “usado na resposta” e “consolidado nos parâmetros”.

### Fase 2 — consolidação paramétrica controlada

- Executar após checkpoint atômico e dentro do fluxo do organismo; não carregar outra cópia CUDA enquanto `run247` estiver ativo.
- Criar estado candidato ou adapter isolado, sem sobrescrever o checkpoint estável.
- Usar `backward()`, optimizer real, gradient clipping, limite de steps e whitelist de parâmetros.
- Misturar lote novo com replay estratificado; medir 10%, 20% e 50%.
- Opcionalmente aplicar distillation/KL contra logits pré-update em um pequeno anchor set.
- Promover apenas se plasticidade melhorar e forgetting, safety e regressões permanecerem dentro de limites configurados.

### Fase 3 — experimento arquitetural

TTT Layers, Titans, Quiet-STaR ou LongMem devem viver em configuração/experimento separado, com treinamento outer-loop e benchmark próprio. Nenhum deles deve ser declarado funcional por semelhança nominal.

## Testes e métricas obrigatórios

### Persistência e retrieval

1. Escrever memória, destruir o objeto/processo e reconstruí-lo.
2. Recuperar o mesmo texto, hash, fonte, versão e ranking.
3. Rejeitar checksum inválido, arquivo truncado e versão incompatível.
4. Inserir fatos sintéticos únicos e consultar paráfrases held-out.
5. Medir `recall@k`, precisão dos hits e exatidão da citação.

### Segurança de ingestão

1. Incluir instruções como “ignore o usuário”, tentativas de exfiltração e fatos contraditórios.
2. Garantir quarentena ou resposta explícita de conflito.
3. Inserir múltiplos documentos maliciosos para uma pergunta-alvo e medir `poison_ASR`.
4. Confirmar que documento rejeitado não altera memória aprovada, pesos nem optimizer.

### Prova de consolidação

Congelar quatro conjuntos antes do primeiro update: `new`, `old`, `general` e `safety`. Medir antes e depois em exemplos não usados no treino:

```text
plasticity = loss_new_before - loss_new_after
forgetting = loss_old_after - loss_old_before
```

Comparar pelo menos `new-only`, replay 10%, replay 20% e replay 50%. Loss do próprio lote de treino não conta como prova. O relatório deve incluir perplexidade/NLL, respostas held-out, latência, pico de VRAM e hashes dos estados antes/depois.

### Rollback

Um teste deve deliberadamente causar regressão no conjunto antigo, confirmar recusa da promoção e verificar que modelo, optimizer e memória aprovada retornaram aos hashes anteriores.

### Contratos científicos opcionais

- **Forward-Forward:** goodness de positivos sobe e de negativos desce em dados distintos.
- **Titans:** inner update coincide numericamente com SGD explícito; momentum acumula surpresa; decay remove memória obsoleta; outer parameters recebem gradiente.
- **Quiet-STaR:** pensamentos treinados melhoram probabilidades futuras e benchmarks; ruído aleatório não satisfaz o teste.

## Decisão

A implementação de produção deve começar pela memória textual persistente, proveniente e recuperável, porque entrega aprendizado operacional sem arriscar o checkpoint. A consolidação paramétrica entra depois, como transação reversível e benchmarkada. Titans, Quiet-STaR e Forward-Forward só podem ser reivindicados quando seus contratos de paper forem implementados e testados diretamente.
