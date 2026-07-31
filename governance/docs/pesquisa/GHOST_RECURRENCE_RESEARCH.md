# Ghost recorrente — pesquisa, desenho e simulação

Data da verificação: 2026-07-14.

## Pergunta científica

Existe uma técnica que mantenha uma ocorrência nova como memória transitória,
marque recorrências, descarte ruído que não reaparece e promova o padrão para
memória estável somente depois de evidência histórica?

## Resposta curta

Sim. O mecanismo matemático mais próximo é **DenStream**. Ele mantém um buffer
separado de `outlier micro-clusters`, acumula peso quando pontos semelhantes
reaparecem, aplica decaimento exponencial e promove um outlier a `potential
core-micro-cluster` quando o peso cruza um limiar. **Adaptive Resonance Theory
(ART)** acrescenta a decisão por similaridade/vigilância: correspondência produz
ressonância e atualização; incompatibilidade provoca reset e busca por outra
categoria. **Complementary Learning Systems (CLS)** fornece a interpretação
biológica de memória rápida separada e consolidação lenta em memória cortical.

A concepção abstrata do Ghost, portanto, tem antecedentes. A oportunidade de
pesquisa do Darwin é a composição específica:

```text
JEPA surprise
  -> Ghost transitório com decaimento e recorrência
  -> ART/vigilância contra histórico consolidado
  -> verificação de contradição e independência de fontes
  -> Ghost Predator / torneio held-out
  -> expert transitório ou memória real
  -> Topology Manifest / checkpoint
```

Não foi encontrada, nesta revisão focada, uma publicação que demonstre esse ciclo
completo dentro de um LLM MoE que altera sua própria topologia.

## Escopo da busca

Revisão focada, não exaustiva. Foram pesquisadas quatro famílias de termos:

1. streaming clustering com outliers, recorrência, promoção e decaimento;
2. reconhecimento incremental com match/mismatch e criação de categorias;
3. memória rápida versus consolidação lenta;
4. memória neural guiada por surpresa e esquecimento.

Foram priorizados papers primários e repositórios institucionais. Trabalhos que
apenas armazenam contexto, fazem replay ou detectam anomalia sem promoção por
recorrência foram considerados adjacentes, não equivalentes.

## Trabalhos incluídos

| Técnica | O que demonstra | Relação com o Ghost | O que não resolve |
|---|---|---|---|
| DenStream, Cao et al. (SDM 2006) | Microclusters transitórios separados, peso temporal, promoção e poda | Correspondência estrutural direta | Não opera em estados latentes de LLM nem cria experts |
| ART, Carpenter & Grossberg (1995; família ART anterior) | Match por vigilância, ressonância, reset e criação estável de categorias | Decide se o evento pertence à memória histórica ou precisa de categoria nova | Não exige recorrência temporal/densidade antes da categoria |
| CLS, McClelland, McNaughton & O'Reilly (1995) | Aprendizado rápido separado e integração lenta para evitar interferência | Fundamenta Ghost transitório → memória consolidada | Não fornece o algoritmo online de promoção |
| Titans, Behrouz et al. (2025) | Memória neural atualizada por surpresa, com momentum e decaimento | Confirma surpresa e esquecimento como sinais de escrita | Não mantém Ghosts discretos nem promoção por recorrência |

Fontes primárias:

- [Density-Based Clustering over an Evolving Data Stream with Noise — DenStream](https://www.cs.sfu.ca/~ester/papers/SDM2006.DenStream.final.pdf)
- [Adaptive Resonance Theory: Self-Organizing Networks for Stable Learning, Recognition, and Prediction](https://open.bu.edu/items/be0fc8e3-65e1-446e-afdf-77b8d75f7ba4)
- [Why there are complementary learning systems in the hippocampus and neocortex](https://pubmed.ncbi.nlm.nih.gov/7624455/)
- [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)

## O que já existe no Darwin

O esqueleto lembrado por Marco está no Heartbeat:

- `SurpriseMemorySlot` já guarda chave, valor, data, domínio, surpresa e
  `access_count`;
- `write_if_surprised()` grava quando o erro JEPA cruza o limiar;
- `retrieve()` busca por similaridade cosseno e incrementa `access_count`;
- o estado completo dos slots é serializado e restaurado;
- `darwin_organism.py` embute `heartbeat_state` no checkpoint v7.

Evidência no código:

- `src/f51_darwin/heartbeat.py:61-120` — gravação, busca e marcação de acesso;
- `src/f51_darwin/heartbeat.py:335-420` — persistência e restauração;
- `src/f51_darwin/organism/checkpoint_mixin.py` — inclusão no checkpoint.

## Lacuna real

O caminho atual registra recorrência de **acesso**, mas não recorrência de
**evidência**. Cada surpresa sempre cria outro slot. `access_count` sobe quando o
slot é recuperado, mesmo que a consulta seja apenas semelhante; ele não prova uma
nova ocorrência independente.

Faltam seis operações:

1. fundir uma nova evidência com o Ghost mais próximo;
2. acumular peso de recorrência separado de `access_count`;
3. decair Ghosts que não reaparecem;
4. exigir fontes ou janelas independentes para impedir autoconfirmação;
5. colocar contradições em quarentena;
6. promover primeiro a candidato e somente depois do torneio held-out.

## Desenho proposto

Cada Ghost mantém:

```text
id
centroide latente / CF1 / CF2
peso temporal
primeiro e último encontro
quantidade de recorrências
fontes ou janelas independentes
assinatura da afirmação/resultado
estado: ghost | quarantine | candidate | real
destino histórico ou expert candidato
```

Atualização temporal inspirada no DenStream:

```text
w(t) = w(t_anterior) * 2^(-lambda * delta_t)
```

Um evento novo é processado nesta ordem:

1. decair e podar Ghosts antigos;
2. procurar memória real compatível;
3. se compatível e consistente, reforçar a memória real sem duplicar;
4. se a representação combina mas a afirmação contradiz, quarentena;
5. procurar Ghost compatível e fundir a nova evidência;
6. promover para `candidate` somente após peso, recorrência e independência;
7. executar Ghost Predator/torneio held-out;
8. apenas o vencedor vira memória/expert real e entra na anatomia persistente.

O microcontrolador preditivo pode estimar probabilidade de recorrência e provável
destino histórico. Ele não deve promover a própria previsão, porque isso criaria
um ciclo de confirmação.

## Simulação de dez cenários

Harness: `research/simulate_ghost_recurrence.py`.

Candidatos comparados:

- **Atual**: semântica presente de slots do `Heartbeat.TestTimeMemory`;
- **Proposto**: Ghost recorrente inspirado em DenStream, com vigilância,
  diversidade de fonte e quarentena.

Parâmetros fixos do protótipo:

| Parâmetro | Valor |
|---|---:|
| Similaridade mínima | 0,97 |
| Lambda de decaimento | 0,12 |
| Peso mínimo antes da poda | 0,20 |
| Peso de promoção | 2,50 |
| Recorrências mínimas | 3 |
| Fontes independentes mínimas | 2 |
| Capacidade padrão | 8 |

| # | Cenário | Heartbeat atual | Ghost recorrente |
|---:|---|---:|---:|
| 1 | Evento único desaparece | Falhou | Passou |
| 2 | Três recorrências exatas viram uma memória | Falhou | Passou |
| 3 | Paráfrases latentes se unem sem duplicação | Falhou | Passou |
| 4 | Conceitos distintos formam memórias distintas | Falhou | Passou |
| 5 | Recorrência depois do horizonte não promove | Falhou | Passou |
| 6 | Match histórico reforça sem duplicar | Falhou | Passou |
| 7 | Contradição vai para quarentena | Falhou | Passou |
| 8 | Burst de uma fonte não se autoconfirma | Falhou | Passou |
| 9 | Pressão de capacidade preserva sinal recorrente | Passou | Passou |
| 10 | Ghost sobrevive ao checkpoint e depois promove | Falhou | Passou |

Resultado: **Heartbeat atual 1/10; Ghost recorrente 10/10**.

O teste adicional executa a classe real `TestTimeMemory` e confirma que três
gravações iguais geram três slots distintos. Isso evita atribuir ao baseline uma
fraqueza inventada apenas pelo simulador.

## Reprodução

```powershell
.\.venv_nitro\Scripts\python.exe -m research.simulate_ghost_recurrence `
  --json runs\research\ghost_recurrence_10_scenarios.json

.\.venv_nitro\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  src\\tests\\test_ghost_recurrence_simulation.py
```

## Limites do resultado

- Os vetores são sintéticos e pequenos; ainda não são hidden states reais do
  modelo 1.6B.
- Os limiares foram escolhidos para testar contratos, não calibrados em corpus.
- `10/10` significa que o protótipo satisfaz os dez contratos determinísticos;
  não significa melhoria de perplexidade, retenção ou geração.
- A simulação não cria expert real, não altera router, optimizer ou checkpoint
  vivo.
- A independência por `source` é um primeiro controle contra repetição/poisoning;
  produção precisa de proveniência criptográfica ou janelas causais verificáveis.
- A promoção paramétrica ainda exige benchmark held-out e comparação de
  forgetting antes/depois.

## Próximo experimento científico

Executar o índice em **shadow mode** sobre hidden states reais:

1. nenhum sinal afeta router, learning rate ou topologia;
2. registrar Ghosts, matches, recorrências e decaimento;
3. congelar um conjunto de eventos futuros;
4. medir precisão/recall de recorrência, taxa de duplicação, falso match,
   falso crescimento e memória ocupada;
5. comparar `access_count` atual contra peso recorrente DenStream;
6. só depois permitir a primeira ação reversível: prefetch Nitro;
7. nascimento de expert continua bloqueado pelo torneio held-out.

Esse experimento separa uma boa metáfora de um órgão causalmente comprovado.

## Validação local desta entrega

- simulador: `Current Heartbeat 1/10`; `Ghost recorrente 10/10`;
- teste focado: `5 passed`;
- suíte CPU completa: `214 passed`;
- `git diff --check`: aprovado;
- nenhum processo `darwin_organism.py run247` estava ativo durante a simulação;
- nenhum peso, checkpoint, corpus ou configuração de treino foi alterado.
