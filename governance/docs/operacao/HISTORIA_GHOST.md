# Historia do orgao Ghost -- F51 Darwin-X

Documento canonico de referencia. Atualizado em 2026-07-22. Autoridade: codigo
fonte, commits Git e manifests de checkpoint verificados por inspecao direta.

## Sumario

1. Origem e proposito
2. Evolucao estrutural (3 Jul -- 11 Jul 2026)
3. Pesquisa Ghost Recorrente (14 Jul 2026)
4. Internalizacao como Loss no modelo (Jul 2026)
5. A crise de isolamento (20--22 Jul 2026)
6. O colapso (22 Jul 2026)
7. Estado atual
8. Licoes

---

## 1. Origem e proposito

O orgao Ghost nasceu em 3 de julho de 2026 com um objetivo concreto: **prevenir
expert collapse em arquiteturas Mixture of Experts (MoE)**.

Em um MoE tradicional, o router tende a colapsar para um subconjunto pequeno de
experts, desperdicando capacidade computacional e anulando a vantagem da
especializacao. O GhostTokenTrainer introduziu uma tecnica de mascara auxiliar:
15% dos tokens de entrada sao artificialmente mascarados, e cada expert deve
predizer o token original a partir do estado latente.

O principio e simples: se um expert esta sendo usado pelo router, ele recebe
sinal de gradiente do forward principal. Se nao esta, ele nao recebe sinal algum
e atrofia. O Ghost garante que **todo expert** receba sinal de aprendizado,
mesmo que o router nao o selecione para o forward principal. Isso mantem a
diversidade de especializacao e previne o colapso para um punhado de experts
dominantes.

A mascara e gerada deterministicamente a partir de um `step_digest` e de uma
seed SHA-256 derivada, garantindo reprodutibilidade sem estado global de RNG. O
algoritmo e definido em `src/f51_darwin/darwin_x_core/model.py`, funcao
`_deterministic_ghost_mask`.

**Commit inaugural**: `0f046ce` -- Ghost Token NaN fix, com clamp de logits no
intervalo `[-15, 15]` para evitar explosoes numericas durante o treino inicial.

---

## 2. Evolucao estrutural (3 Jul -- 11 Jul 2026)

### 2.1 ghost_brain.py -- Sistema de dopamina e consequencia

O `ghost_brain.py` (`src/f51_darwin/ghost_brain.py`) expandiu o conceito alem da
mascara. A classe `GhostBrain` implementa um sistema nervoso completo:

- **Dopamina (recompensa)**: acertou pergunta nova, verificou com Wolfram, gerou
  dado sintetico util, completou ciclo de exploracao.
- **Consequencia (punicao)**: errou pergunta (mas aprende), Wolfram nao
  confirmou (marca incerto), dado sintetico inutil (descarta), repetiu erro 3x
  (poda o modulo).
- **Aprendizado por erro**: cada erro gera um `ErrorLesson` -- question,
  wrong_answer, correct_answer, root_cause, lesson_text -- que alimenta o corpus
  sintetico. O erro e tratado como o dado mais valioso do sistema.

Estados emocionais: `CURIOUS`, `EXCITED`, `FRUSTRATED`, `ENLIGHTENED`,
`SATISFIED`.

### 2.2 ghost_corpus.py -- Auto-evolucao

O `ghost_corpus.py` (`src/f51_darwin/ghost_corpus.py`) implementa o pipeline de
auto-evolucao:

```
Corpus Total → 80% Treino Inicial / 20% Fantasma
  ├─ Treino Inicial: modelo nasce, aprende o basico
  └─ AUTO-EVOLUCAO:
      1. CuriosityDrive: escolhe o que explorar do fantasma
      2. QuestionGenerator: cria perguntas sobre o desconhecido
      3. Verifier (Wolfram/Web/Code): verifica respostas
      4. SyntheticDataGenerator: cria novos dados de treino
      5. SelfTrainer: auto-treina nos proprios dados gerados
      6. Evolution Loop: modulos nascem/morrem baseado em performance
```

A classe `GhostDocument` representa documentos ocultos do modelo, com status
`UNEXPLORED → QUESTIONED → VERIFIED → SYNTHESIZED → LEARNED`.

### 2.3 spider_sense.py -- O modelo sabe quando nao sabe

`src/f51_darwin/spider_sense.py` implementa `SpiderSense`, um detector de confianca
baseado nos hidden states da ultima camada. Uma MLP de 2 camadas prediz se a
resposta do modelo esta correta (score 0-1). Score abaixo de 0.3 dispara o
alerta: o modelo nao sabe e deve verificar.

A integracao com o Ghost Brain forma o circuito fechado:

```
Ghost Brain → Spider-Sense → Wolfram Verify → Error Lesson → Auto-train
```

### 2.4 Organismo VIVO

**Commit `7b3fa28`** marca o momento em que o organismo Darwin-X foi declarado
VIVO com todos os orgaos integrados: Ghost, Heartbeat, Spider-Sense, JEPA, MTP,
DAE, Curiosity, Decision Engine, Unified Mesh, Inter-Hemispheric e Sleep.

### 2.5 ghost_stream.py e ghost_feeder.py (removidos)

Dois componentes existiram temporariamente:
- `ghost_stream.py` (commit `3446121`): alimentacao de dados sinteticos via
  HuggingFace.
- `ghost_feeder.py` (commit `8f442a3`): ciclo `Ghost Stream → Firewall →
  Tokenize → Treino`.

Ambos foram removidos quando o Ghost foi internalizado como loss no modelo
(secao 4). Nao estao mais presentes no disco.

---

## 3. Pesquisa Ghost Recorrente (14 Jul 2026)

Documentada em `governance/docs/pesquisa/GHOST_RECURRENCE_RESEARCH.md`.

**Pergunta cientifica**: Existe uma tecnica que mantenha uma ocorrencia nova
como memoria transitoria, marque recorrencias, descarte ruido que nao reaparece
e promova o padrao para memoria estavel somente depois de evidencia historica?

**Resposta**: Sim. O mecanismo matematico mais proximo e **DenStream** (Cao et
al., SDM 2006), que mantem um buffer separado de `outlier micro-clusters`,
acumula peso quando pontos semelhantes reaparecem, aplica decaimento exponencial
e promove outliers a `potential core-micro-cluster` quando o peso cruza um
limiar.

**Ciclo proposto**:

```
JEPA surprise
  → Ghost transitorio com decaimento e recorrencia
  → ART/vigilancia contra historico consolidado
  → verificacao de contradicao e independencia de fontes
  → Ghost Predator / torneio held-out
  → expert transitorio ou memoria real
  → Topology Manifest / checkpoint
```

**Simulacao de 10 cenarios** (harness: `research/simulate_ghost_recurrence.py`):
Heartbeat atual 1/10; Ghost recorrente **10/10**. O Heartbeat atual
(`TestTimeMemory`) cria tres slots distintos para tres gravacoes iguais -- nao
funde, nao detecta recorrencia, nao promove.

**Lacuna**: O Heartbeat atual registra recorrencia de **acesso**, nao
recorrencia de **evidencia**. Faltam: fusao de evidencias, peso temporal,
decaimento, independencia de fontes, quarentena de contradicoes e promocao
condicional.

Estado da pesquisa: prototipo validado com vetores sinteticos. Aguarda
experimento em **shadow mode** sobre hidden states reais antes de qualquer acao
causal.

---

## 4. Internalizacao como Loss no modelo (Jul 2026)

O Ghost foi internalizado como termo de loss no `DarwinXModel`, em
`src/f51_darwin/darwin_x_core/model.py`.

### 4.1 Duas implementacoes

#### `_ghost_loss()` -- "Jurassic Park" (linha 1000)

Segunda passagem completa pelos blocos do transformer. Gera uma mascara
aleatoria (nao-deterministica), aplica token `<mask>` (id=2), re-executa o
forward inteiro e computa cross-entropy apenas nos tokens mascarados.

- **Vantagem**: forward completo, alinhado com o forward principal.
- **Desvantagem**: ~50% de compute extra. Incompativel com Dual GPU (NCCL
  deadlock -- secao 5).

Chamada quando `step_digest is None` (modo sem causal ledger).

#### `_causal_ghost_loss()` -- single-pass (linha 1031)

Usa o hidden state ja computado no forward principal. Aplica mascara
deterministica baseada em `step_digest`. Usa o JEPA predictor para projetar o
estado futuro e computa cross-entropy.

- **Vantagem**: custo computacional negligenciavel, deterministico, compativel
  com Dual GPU.
- **Desvantagem**: requer isolamento rigoroso do grafo de autograd (secao 5).

Chamada quando `step_digest` e fornecido (modo causal ledger, padrao no treino
100M FULL V9).

### 4.2 Parametros na config

Em `src/configs/darwin_x_100m.yaml`:

```yaml
ghost_weight: 0.12
ghost_mask_ratio: 0.15
ghost_enabled: true  # single-pass RAM/SSD probe with detached autograd
```

### 4.3 Ghost Predator no sistema neuroendocrino

O `NeuroendocrineSystem` (`src/f51_darwin/darwin_x_core/neuroendocrine.py`) mantem:

- `ghost_loss_ema` (buffer, inicializado em 10.0): media movel da ghost loss,
  usada para detectar retencao ou esquecimento de conhecimento antigo.
- `ghost_memory_size` (buffer, zeros): contador de memorias no buffer do Ghost
  Predator.

O Ghost Predator opera no metodo `update_from_local_signals`:
- Se `ghost_loss <= ghost_loss_ema` anterior: o modelo reteve conhecimento
  antigo → bonus de dopamina (`ghost_bonus`).
- Se `ghost_loss > ghost_loss_ema` anterior: o modelo esqueceu → punicao de
  cortisol (`ghost_punishment`).

Isso torna o organismo **sabio, nao agressivo**: a ghost loss nao e apenas uma
regularizacao auxiliar, mas um sinal fisiologico que modula dopamina, cortisol,
neurogenese e apoptose.

---

## 5. A crise de isolamento (20--22 Jul 2026)

### 5.1 NCCL deadlock em Dual GPU (20 Jul)

**Commit `b23ade6` (20 Jul)**: Ghost **DESABILITADO** (`ghost_enabled: false`).

A implementacao `_ghost_loss()` (Jurassic Park) executava uma segunda passagem
completa pelos blocos. No modo Dual GPU (`enable_dual_gpu`), isso significava
transferir tensores entre GPU0 e GPU1 **dentro de um grafo de autograd ativo**.
O NCCL entrava em deadlock porque operacoes coletivas assincronas (all-reduce do
MoE) nao podiam ser completadas enquanto o grafo da segunda passagem ainda
estava sendo construido.

Ghost foi desabilitado como medida de seguranca.

### 5.2 Phantom gradient em CONTROL/SHADOW (20 Jul)

**Commit `4151215` (20 Jul)**: zero ghost_loss em modos CONTROL e SHADOW.

Quando o Ghost foi reabilitado na forma single-pass, o sinal da ghost loss
chegava como zero nos modos de canario (CONTROL/SHADOW) porque o causal ledger
ainda nao estava aplicando intervencoes. Sem `step_digest` valido, a funcao
caia no fallback `_ghost_loss()` (Jurassic Park), que nao funcionava em Dual GPU
(desabilitado por deadlock). Resultado: gradiente fantasma -- zero loss mas
grafo de autograd instavel.

### 5.3 Reabilitacao com single-pass (22 Jul 01:02)

**Commit `c14595d` (22 Jul 01:02)**: Ghost **REABILITADO** com single-pass
RAM/SSD probe e `hidden.detach()`.

A substituicao do `_ghost_loss` (2a passagem) pelo `_causal_ghost_loss`
(single-pass) eliminou o NCCL deadlock. A operacao `hidden.detach()` (pinça 1)
corta o grafo de autograd entre o forward principal e o Ghost, garantindo que o
Ghost nao injete gradientes de volta nos blocos do transformer.

### 5.4 As duas pincas (analogia da corrente)

A arquitetura final do `_causal_ghost_loss` opera como uma corrente com duas
pincas isoladoras:

```
Forward principal → hidden → [pinça 1: hidden.detach()] → Ghost
                                                            ↓
JEPA predictor ← hidden (autograd ativo)                   ↓
       ↓                                              predicted_hidden
  JEPA loss (cosine similarity)                              ↓
                                                    [pinça 2: .detach()]
                                                            ↓
                                                       lm_head
                                                            ↓
                                                    Ghost loss (cross-entropy)
```

- **Pinça 1** (`hidden.detach()`): isola o Ghost do forward principal. Sem ela,
  o gradiente da ghost loss fluiria de volta pelos 12 blocos do transformer,
  causando NCCL deadlock no modo Dual GPU e corrompendo o estado dos blocos com
  um sinal conflitante.

- **Pinça 2** (`predictor(current).detach()`): isola o Ghost do JEPA predictor.
  Sem ela, o gradiente da ghost loss fluiria pelo predictor, depois para o
  lm_head (que tem weight tying com token_embedding) e dali para os parametros
  do modelo, criando um conflito destrutivo entre duas funcoes objetivo
  incompatíveis.

---

## 6. O colapso (22 Jul 2026)

### 6.1 Mecanismo

Apos o commit `c14595d` (pinça 1), o Ghost estava reabilitado e funcional, mas
a pinça 2 ainda nao existia. O codigo era:

```python
current = hidden.detach()[:, :-1]          # pinça 1: OK
predicted_hidden = predictor(current)       # SEM .detach() -- ERRO
logits_ghost = self.lm_head(predicted_hidden)
```

O gradiente da ghost loss (cross-entropy) fluia por este caminho:

```
Ghost cross-entropy loss
  → lm_head (pesos compartilhados com token_embedding)
  → predicted_hidden (JEPA predictor output, COM autograd)
  → JEPA predictor (parametros treinaveis)
  → parametros do modelo
```

### 6.2 Conflito de objetivos

O JEPA predictor e treinado para **maximizar similaridade cosseno** entre
`predicted` e `target` (estado seguinte). Sua loss e:

```
jepa_loss = (1.0 - cosine_similarity(predicted, target)).mean()
          + 0.05 * var_loss
          + 0.025 * cov_loss
```

O Ghost loss e **cross-entropy** sobre o vocabuario (58162 classes). Sao dois
objetivos fundamentalmente diferentes atuando sobre os mesmos parametros (o
predictor e, via weight tying, o token_embedding e lm_head).

### 6.3 Envenenamento lento

Por 297 steps (do step 1100 ao step 1397), o gradiente do Ghost contaminou
silenciosamente o JEPA predictor. O router comecou a tomar decisoes baseadas em
um predictor corrompido. O colapso nao foi instantaneo: a ghost loss tem peso
0.12 contra 0.5 do JEPA, entao o sinal espurio era minoritario, mas cumulativo.

### 6.4 Deteccao e bloqueio

No step 1397, a loss saltou de 5.67 para 570. O Brainstem (sistema de
seguranca do organismo) detectou a anomalia e bloqueou todas as atualizacoes.
**Pesos intactos** -- o checkpoint `organism_cycle_006.pt` (Cycle 6, Step 1397)
foi preservado.

### 6.5 Correcao (22 Jul 14:29)

**Commit `a0de9a0` (22 Jul 14:29)**: fix critico de **1 linha**:

```python
# Antes (com falha):
predicted_hidden = predictor(current)

# Depois (corrigido):
predicted_hidden = predictor(current).detach()
```

Com a pinça 2, o gradiente da ghost loss para no `predicted_hidden`. O JEPA
predictor continua sendo otimizado exclusivamente pela JEPA loss. A ghost loss
so atualiza o lm_head (e, por weight tying, o token_embedding) -- exatamente o
que se deseja: forcar o modelo a manter representacoes que preservam a
capacidade de predizer tokens mascarados, sem interferir no aprendizado de
representacao latente do JEPA.

### 6.6 Rollback

Os checkpoints dos ciclos 7 e 8 (gerados com a pinça 2 ausente) foram movidos
para quarentena em `workspace/03_CHECKPOINTS_100M_FULL_V9/quarantine_corrupted/`.
O ponteiro `organism_latest.json` foi restaurado para `organism_cycle_006.pt`.
O preflight confirmou `PREFLIGHT SUCCESS`.

---

## 7. Estado atual (22 Jul 2026)

O orgao Ghost esta:

| Atributo | Valor |
|---|---|
| Status | **ATIVO** |
| Modo | single-pass (`_causal_ghost_loss`) |
| Isolamento | **Dupla pinça**: `hidden.detach()` + `predictor(current).detach()` |
| Forward extra | Nenhum (usa hidden do forward principal) |
| Compativel Dual GPU | Sim |
| ghost_weight | 0.12 |
| ghost_mask_ratio | 0.15 |
| ghost_enabled | true |
| Determinismo | Sim (mask via SHA-256 do step_digest) |
| Ghost Predator | Ativo (modula dopamina/cortisol no neuroendocrino) |
| ghost_loss_ema | Buffer no NeuroendocrineSystem, inicializado em 10.0 |
| ghost_memory_size | Buffer no NeuroendocrineSystem, inicializado em 0 |

### 7.1 Localizacao no codigo

- **Loss functions**: `src/f51_darwin/darwin_x_core/model.py`, classe
  `_ModelAuxiliaryLossMixin`, metodos `_ghost_loss` (linha 1000) e
  `_causal_ghost_loss` (linha 1031).
- **Mascara deterministica**: `src/f51_darwin/darwin_x_core/model.py`, funcoes
  `_deterministic_ghost_mask` (linha 21) e `deterministic_ghost_mask` (linha
  85).
- **Ghost Predator**: `src/f51_darwin/darwin_x_core/neuroendocrine.py`, metodo
  `update_from_local_signals` (linha 166), buffers `ghost_loss_ema` (linha 499)
  e `ghost_memory_size` (linha 542).
- **Ghost Brain (alto nivel)**: `src/f51_darwin/ghost_brain.py`, classe
  `GhostBrain`.
- **Ghost Corpus**: `src/f51_darwin/ghost_corpus.py`, classe `GhostCorpus`.
- **SpiderSense**: `src/f51_darwin/spider_sense.py`, classe `SpiderSense`.
- **Config**: `src/configs/darwin_x_100m.yaml`, linhas 28-30.
- **Integracao no forward**: `src/f51_darwin/darwin_x_core/model.py`, metodo
  `forward` da classe `_ModelForwardMixin`, linhas 659-714.
- **Topology manifest**: `src/f51_darwin/darwin_x_core/model.py`, metodo
  `topology_manifest`, inclui `ghost_loss_ema`, `ghost_memory_size` e
  `ghost_predator_buffer` no manifesto de checkpoint.
- **Pesquisa**: `governance/docs/pesquisa/GHOST_RECURRENCE_RESEARCH.md`.
- **Simulacao**: `research/simulate_ghost_recurrence.py`.

---

## 8. Licoes

### 8.1 Gradiente auxiliar precisa de isolamento total

A licao central dos eventos de 20-22 de julho de 2026: **todo termo de loss
auxiliar que opera sobre representacoes intermediarias compartilhadas precisa de
isolamento completo do grafo de autograd principal**.

Nao basta isolar o forward (pinça 1). E preciso isolar tambem qualquer preditor
ou projecao intermediaria (pinça 2). Um unico `.detach()` ausente pode
corromper silenciosamente o treino por centenas de steps antes que o colapso se
torne visivel.

### 8.2 Conflito de objetivos e silencioso

O conflito entre JEPA (similaridade cosseno) e Ghost (cross-entropy) nao
produziu NaN imediato. O envenenamento foi gradual, cumulativo e so se
manifestou como colapso apos 297 steps. Sistemas de monitoramento baseados
apenas em thresholds de loss podem nao detectar contaminacao de gradiente em
estagio inicial.

### 8.3 Brainstem funcionou

O sistema de seguranca (Brainstem) detectou a anomalia no step 1397 e bloqueou
as atualizacoes antes que os pesos fossem corrompidos de forma irreversivel. O
checkpoint do ciclo 6 foi preservado integro, permitindo rollback limpo.

### 8.4 Isolamento nao e "desligar"

Desabilitar o Ghost (`ghost_enabled: false`) resolveu o sintoma (NCCL deadlock)
mas removeu um sinal fisiologico importante: sem ghost loss, o Ghost Predator
perde sua entrada e o sistema neuroendocrino opera as cegas para retencao de
conhecimento. A solucao correta foi isolar, nao remover.

### 8.5 Princípio de arquitetura

Para qualquer loss auxiliar em arquitetura com peso compartilhado (weight
tying):

1. **Detach na entrada**: o estado que alimenta a loss auxiliar deve ser
   desacoplado do forward principal.
2. **Detach na projecao**: toda transformacao intermediaria entre o estado e a
   loss auxiliar deve ter seu gradiente interrompido.
3. **Verificacao dupla**: testar explicitamente que `loss_auxiliar.backward()`
   nao altera gradientes de modulos que nao deveriam ser afetados (teste de
   isolamento).

---

*Documento gerado em 2026-07-22 por savekeeper. Evidencias: codigo fonte em
`src/f51_darwin/darwin_x_core/model.py`, `src/f51_darwin/darwin_x_core/neuroendocrine.py`,
`src/f51_darwin/ghost_brain.py`, `src/f51_darwin/ghost_corpus.py`,
`src/f51_darwin/spider_sense.py`, `src/configs/darwin_x_100m.yaml`,
`governance/docs/pesquisa/GHOST_RECURRENCE_RESEARCH.md` e `governance/docs/operacao/STATUS_ATUAL.md`.
Verificado por inspecao direta dos arquivos no disco em 2026-07-22.*
