<!-- authority: current-status -->
# Estado atual — F51 Darwin-X

Snapshot local-only de 2026-07-29, revisado às ~12:55 America/Toronto após a
cirurgia e a repetição causal do canal TTM nativo.

## Darwin-Smol Native Dense V1 — candidato aprovado

O candidato operacional mais forte agora é:

`workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt`

| Prova | Resultado |
|---|---|
| SHA-256 | `9329b8d25fc23acde021161da7a3421e6e17ceea3e940e0298162f39b62bc01e` |
| Tamanho | **4.091.861.647 bytes** |
| Parâmetros Darwin | **1.931.430.427** |
| Cobertura do doador | **218/218 tensores** |
| Anatomia preservada | 24→24 camadas, 2048→2048, 32→32 heads, MLP denso 8192→DenseSwiGLU 8192 |
| Estrutura feed-forward | **24 densas, 0 MoE, 0 routers, 0 experts** |
| Paridade em prompts fixos | top-1 **98,6111%**, KL **0,005972** |
| Gate holdout | BpB **2,1562**, KL **0,003855**, top-1 **98,1132%** |
| Controles densos aleatórios | BpB **4,8978–4,9434**, KL **8,4825–8,5205**, top-1 **0%** |
| QA pareado balanceado | **30/30 respostas idênticas**; Darwin **17/30**, Smol **17/30**, zero vitórias de qualquer lado |
| Aprendizado mecânico | PASS: loss finita; gradiente não nulo em gate/up/down; os três pesos mudaram |
| Donor-free | PASS, duas GPUs, `donor_loaded=false`, reload com logits exatos |
| Publicação local | `candidate-manifest.json`, status `approved_candidate` |

Esta versão corrige a mentira estrutural do V3: a FFN não é mais um expert
único escondido dentro de um contêiner MoE. `DenseSwiGLU` registra diretamente
`gate_proj`, `up_proj` e `down_proj`; não há atributo `.moe`, router, top-k nem
estado neuroendócrino nas 24 camadas. Os pesos Smol foram copiados diretamente
para essas matrizes com dimensões idênticas, sem SVD, clustering, padding,
compressão ou treinamento.

Heartbeat, TTM, Spider, JEPA, MTP, GABA, Sleep, Decision Engine, Unified Mesh e
IHS pertencem à anatomia/config do candidato. Heartbeat foi observado no
canário; GABA, TTM e IHS começam com contribuição residual neutra; JEPA, MTP,
Spider e as demais perdas auxiliares começam com peso zero. DAE/Nitro e seu
estado neuroendócrino MoE ficam deliberadamente ausentes nesta linhagem densa.
Conectado significa que o caminho existe e é finito, não que o órgão já
demonstrou ganho de habilidade.

Não houve fine-tuning (`training_steps=0`). O backward real foi executado numa
cópia temporária em memória para provar capacidade de aprendizado, sem alterar
o candidato publicado. O Darwin-Smol Exact Brain V3 permanece no disco como
rollback da implementação antiga com expert único, não como candidato
operacional principal.

O QA pareado usa 24 questões objetivas com posições A/B/C/D balanceadas e seis
respostas curtas, cobrindo conhecimento, português, matemática, raciocínio,
programação e inglês. Smol e Darwin receberam o mesmo template e decodificação
greedy e produziram respostas iguais nos 30 casos, inclusive os mesmos 13
erros. Isso é evidência direta de preservação comportamental nesse conjunto,
mas não transforma o conjunto local pequeno em benchmark externo abrangente.
O benchmark usou `heartbeat=False` e gates neutros: ele **não mede utilidade
dos órgãos, memória por inferência nem adaptação entre sessões**. Repeti-lo
sem mudar esse protocolo deve continuar produzindo as mesmas respostas.

### QA causal por órgão — hipótese de aprendizado por inferência reprovada

O cérebro foi congelado e cada órgão treinável foi liberado isoladamente sobre
uma cópia fresca do mesmo checkpoint. Cada braço recebeu uma passagem de ensino
nas 30 questões, seguida por dez avaliações sem correção nas ordens embaralhadas
pelas seeds 51–60. Nenhum braço adaptado foi publicado.

| Braço | Pré | Pós média (min–máx) | Corrigiu | Esqueceu | Evidência |
|---|---:|---:|---:|---:|---|
| controle | 17/30 | 17 (17–17)/30 | 0 | 0 | nenhuma mudança |
| GABA | 17/30 | 16 (16–16)/30 | 0 | 1 | 72 tensores receberam gradiente e mudaram |
| IHS | 17/30 | 16 (16–16)/30 | 0 | 1 | 35 tensores receberam gradiente e mudaram |
| Heartbeat | 17/30 | 17 (17–17)/30 | 0 | 0 | 580 beats; nenhum gradiente ou efeito em resposta |
| TTM | 17/30 | 17 (17–17)/30 | 0 | 0 | 30 memórias gravadas; escala residual publicada é zero |
| Spider | 17/30 | 17 (17–17)/30 | 0 | 0 | nenhum gradiente da loss de linguagem |
| JEPA | 17/30 | 17 (17–17)/30 | 0 | 0 | nenhum gradiente da loss de linguagem |
| MTP | 17/30 | 17 (17–17)/30 | 0 | 0 | nenhum gradiente da loss de linguagem |

O cérebro congelado teve **zero tensores alterados nos oito braços**. Portanto,
o transplante preserva o cérebro Smol, mas a configuração publicada **não
demonstrou aprender respostas durante a inferência**. GABA e IHS alcançam a
linguagem, porém a adaptação curta medida piorou o resultado; Heartbeat e TTM
alteram estado, mas esse estado não tem autoridade efetiva sobre as respostas;
Spider, JEPA e MTP permanecem auxiliares sem gradiente vindo da loss principal.
“Órgão presente/conectado” não significa “órgão útil”.

Artefatos: `workspace/runtime/darwin_17b_smol_dense_v1/organ-causal-qa.json`
e `workspace/runtime/darwin_17b_smol_dense_v1/organ-causal-qa.md`.

### Prova causal TTM entre sessões — canal reparado, lembrança reprovada

Em 2026-07-29 foi executado um protocolo específico de memória sem treino.
Cinco fatos sintéticos foram escolhidos somente depois de o baseline errá-los.
O controle e o organismo com memória receberam as mesmas sequências de ensino;
apenas o segundo gravou cinco associações nativas: chave no hidden bruto de
`T-1` da pergunta e valor na média dos seis hidden states teacher-forced da
resposta. Em seguida o modelo foi descarregado, o estado TTM foi salvo e
restaurado em cópias frescas do checkpoint.

| Dose | Memória correta L/P/D | Gate desligado L | Valores embaralhados L | Restaurada L | Aceitas | Top-1 correto |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0/2/0 | 0 | 0 | 0 | 5/5 | 1/5 |
| 0.01 | 0/1/0 | 0 | 0 | 0 | 5/5 | 1/5 |
| 0.03 | 0/2/0 | 0 | 0 | 0 | 5/5 | 1/5 |
| 0.10 | 0/0/0 | 0 | 0 | 0 | 5/5 | 1/5 |
| 0.30 | 0/0/0 | 0 | 0 | 0 | 5/5 | 1/5 |

**Classificação científica: `no_memory_effect`; causa:
`value_not_behaviorally_decodable`; gate estrito reprovado.**

A cirurgia em RAM/código alinhou escrita e leitura em `[1,D]`, trocou quatro
consultas Spider por uma consulta top-1 no `T-1` que decide a fala e limitou a
norma do residual. A instrumentação nativa mostrou:

- controle com o fato explicitamente no prompt: **5/5**, descartando falha do
  decoder como piso da tarefa;
- cinco consultas nativas, **5/5 aceitas** e **5/5 residuais aplicados**;
- `T-1=146` recebeu a injeção nas cinco perguntas;
- dose `0,30` produziu Δlogit máximo entre **5,0 e 5,5**;
- razão residual escalado/hidden entre **0,1864 e 0,1888**, abaixo do teto;
- top-1 correto somente **1/5**; as cinco similaridades BF16 foram exatamente
  `1,0`, evidenciando empate/colapso das chaves;
- memória correta: **0/5 literal em todas as doses** e **0/10 paráfrases** na
  dose máxima;
- zero pesos ou versões do cérebro alterados.

Logo, o TTM agora escreve, serializa, recarrega, recupera e alcança o decoder,
mas ainda não demonstrou lembrança comportamental. Os bloqueios restantes são
chaves não discriminativas e um valor hidden que não se converte no código
ensinado. “Canal funcional” não equivale a “memória útil”.

Hashes de prova:

- checkpoint:
  `9329b8d25fc23acde021161da7a3421e6e17ceea3e940e0298162f39b62bc01e`;
- protocolo:
  `559f60912687bc4437ed5a73abb4dd4ec6e77a3c6f4889f6804eb3e7de5e3bf0`;
- estado de memória:
  `0da6938da286e3e61935a0ac70988af8dc9bd7e6b456a1b97668c06534b037fe`;
- diagnóstico:
  `66f61b68681a1e1e04b4f300de6fec21b288e1501b6b54ca9e76c1b6e4494943`.

Artefatos:
`workspace/runtime/darwin_17b_smol_dense_v1/memory-recall/`.
Operação: `governance/docs/operacao/NATIVE_TTM_RECALL.md`.

## Veredito executivo

**5 trilhas preservadas. GPU livre.**

1. **Darwin-Smol Native Dense V1** — ✅ candidato aprovado, donor-free,
   dual-GPU, cérebro Smol em matrizes densas Darwin reais, sem MoE, com órgãos
   em first-boot neutro.
2. **Darwin-Smol Exact Brain V3** — ↩️ rollback preservado; funcional, mas usa
   o antigo contêiner MoE com expert único.
3. **Darwin Transfer (Transformer→SSD)** — ⚠️ mecanismo provado, **modelo
   reprovado**. O checkpoint continua no disco como evidência de engenharia,
   não como modelo utilizável. Ver seção abaixo.
4. **Two-Donor Organ Transplant** — ✅ 7/7 órgãos integrados, resultado
   científico ainda `inconclusive_engineering_canary`
5. **Circuits Framework** — ⚠️ biblioteca pronta, mas o vertical slice de
   circuitos continua separado do candidato denso.

## Darwin Transfer — DistilGPT-2 → SSD

### O que está provado

| Prova | Resultado |
|---|---|
| Checkpoint | `workspace/03_CHECKPOINTS_DARWIN_TRANSFER_DISTILGPT2_V1/generation-06-final.pt` |
| SHA-256 | `59847bdae17017d3a7fe3b82624e475dda9b4f7a20d73115426c7c26ca5afe2e` (reconferido) |
| Camadas attention restantes | **0** — os 6 blocos são `SSDGPT2Attention`; varredura por `GPT2Attention`/`MHA` reais retorna 0 |
| NLL reprodutível | aluno 1.9885 / professor 3.1041 (publicado: 1.9887 / 3.1040) |

Conversão progressiva em 6 gerações: `[5] → [4,5] → ... → [0,1,2,3,4,5]`.
Cada geração: orientação → alinhamento → destilação (KL + LM loss).
250 steps por estágio, checkpoint a cada 25.

Script: `src/tools/start_darwin_transfer.ps1`

### O que está reprovado

A leitura anterior — "aluno 1.99 NLL, melhor que o professor" — **estava
errada**. O número reproduz, a interpretação não se sustenta.

| Medida | Aluno SSD | Professor | Delta |
|---|---|---|---|
| Prosa PT (fatia que o **aluno treinou**) | 6.8340 | 5.2814 | **+1.5526** |
| Template MATH (promotion holdout) | 1.9885 | 3.1041 | −1.1157 |
| **Bigrama contado no treino** (mesmo holdout) | — | — | **1.6869** |

Três fatos, cada um suficiente para reprovar:

1. **Uma tabela de bigramas ganha do modelo de 82M parâmetros** (1.6869 vs
   1.9885). O holdout tem 5.039.127 tokens e apenas **526 tokens únicos**;
   os 15 mais frequentes são 52,8% do total. Nesse regime a NLL mede
   estatística local de token, não capacidade.
2. **O aluno é pior que o professor não-treinado na prosa que ele próprio
   treinou.** Esquecimento catastrófico.
3. **Matemática: 0%.** Em 200 equações do 1º grau distintas ele produz apenas
   2 saídas distintas (`.0` e `0.0`). Os 6 "acertos" são exatamente os 6 itens
   cuja resposta real é zero — taxa idêntica à taxa-base. Fora do intervalo do
   gerador (coef. ±[21,99]): 0/200. Dos 200 enunciados, 132 aparecem
   **literalmente** no corpus de treino e mesmo assim erram.

Causa raiz — composição do corpus (`workspace/02_CORPUS/corpus`, 790 MB):

| prefixo | MB | % |
|---|---|---|
| `math_*` (template sintético) | 615,5 | **77,9%** |
| `python_*` | 159,7 | 20,2% |
| `conservative_*` (prosa) | 15,0 | **1,9%** |

`collect_corpus_text` ordena por `sorted(rglob)` e corta em
`--corpus-max-bytes 96M` → treino ficou **13,2% prosa / 86,8% template**, e o
holdout, sendo a cauda contígua, caiu 100% dentro do template.

### Correção aplicada

O portão `quality_restored` compara o aluno com um professor que pode nunca
ter visto o domínio do holdout — num holdout degenerado ele passa mesmo com o
modelo destruído. Foi acrescentado o portão **`beats_ngram_baseline`**
(`src/f51_darwin/transfer/ngram.py`, ligado em `publication_gates`): conta um
bigrama com backoff **somente no split de treino** e exige que o aluno o
supere no mesmo holdout de promoção. Omitir o controle ou passar `NaN`
reprova — o portão falha fechado.

Recalculando os portões do `generation-06-final` com o controle novo:

```
zero_attention         PASS
finite_holdout         PASS
quality_restored       PASS
beats_ngram_baseline   FAIL
status = REJECTED   (publicado originalmente como: accepted)
```

⚠️ **Pendência**: `published.json` e `candidate.json` na raiz do checkpoint
ainda dizem `"status": "accepted"` com `schema_version: 1`. Não foram
reescritos — revogar um pointer publicado é ação operacional e ficou para
decisão explícita.

Próximos passos abertos: rebalancear o corpus (math para ~10%, trazer
`python_*` e mais prosa) e trocar o holdout contíguo de cauda por um holdout
estratificado por fonte.

## Two-Donor Organ Transplant

Transplante de órgãos Darwin-X (512d, V3 step 1750) → GPT-2 (768d)
via adaptadores dimensionais e gate zero.

### Órgãos implementados

| # | Órgão | Params | Tipo | Status |
|---|---|---|---|---|
| 1 | JEPA | 660K | Sidecar loss | ✅ Testado |
| 2 | GABA.0 | 263K | Residual injection | ✅ Testado (melhor resultado: Δtrain -0.44, Δholdout -0.11) |
| 3 | Spider | 66K | Sidecar confidence | ✅ Testado |
| 4 | TTM | 1 escalar | Residual gate + memory | ✅ Testado (holdout -0.70) |
| 5 | MTP | 524K | Multi-token prediction | ✅ Testado |
| 6 | Heartbeat | 858K | Meta Python | ✅ Estado v3 + sidecar testados |
| 7 | IHS | 6.57M | Pre-block MHA | ✅ Zero-gate + caminho ativo testados |

### Resultados principais

- **Zero-gate equivalence**: `TWO_DONOR_VERIFY_OK`, erro < 1e-5
- **Sem órgão = overfit catastrófico**: Braço B train 8.69→0.00003, holdout 9.22→12.56
- **Com órgão = proteção + melhoria**: GABA 1000 steps train -5%, holdout -1.2%
- **TTM funcional**: memória associativa recupera e injeta, gate abre gradualmente
- **15 experimentos passando**: 5 órgãos × 3 seeds
- **Canário de integração 7/7**: 84 tensores, 8.938.636 parâmetros,
  `disabled_error=0`, `shadow_error=0`, Heartbeat muta somente no modo ativo
- **Escopo científico honesto**: Heartbeat e IHS passaram o gate de engenharia;
  os 15 experimentos históricos continuam sendo dos cinco órgãos anteriores

### Stack protegido simultâneo

Canário de integração executado em 2026-07-28 com três seeds, oito passos por
seed e sequência 32:

```powershell
powershell -ExecutionPolicy Bypass -File src\\tools\\start_two_donor_transplant.ps1 `
  -GuardedStackCanary -Device cuda:0 -Seeds 3 -StepsPerArm 8 -SequenceLength 32
```

Política gravada no checkpoint:

- `gaba.0=active`
- `jepa=train_only`
- `ttm=shadow`
- `spider=shadow`
- `heartbeat=shadow`
- `mtp=disabled`
- `ihs=disabled`

Resultado de engenharia: GABA e JEPA tiveram gradientes finitos nas três seeds;
GABA abriu o gate e alterou logits; TTM permaneceu com zero escritas; Heartbeat
permaneceu no beat 1782, sem memória; Spider observou; os sidecars em shadow
tiveram erro de logits zero; o GPT congelado teve drift zero. O checkpoint foi
recarregado do disco e inferiu com logits finitos.

Checkpoint:
`workspace/03_CHECKPOINTS_DARWIN_TWO_DONOR_V1/darwin-two-donor-20260728-224235-guarded/guarded-stack-seed-2005.pt`

SHA-256:
`c58a801c8293b6e5a9e2b2f2f3ddb66af342ea52040f4f91e904142359c6d9e2`

O veredito continua sendo `inconclusive_engineering_canary`: prova montagem,
isolamento de autoridade, gradientes e reload; não prova ainda ganho de
habilidade ou memória útil.

### Arquitetura

```
src/f51_darwin/transplant/
├── bundle.py       — extração SHA-256 dos 7 órgãos + runtime Heartbeat saneado
├── organs.py       — reconstrução estrita, HeartbeatOrgan + load_ttm_memory
├── slots.py        — OrganAdapter(768↔512), GABAResidualSlot, JEPAAuxiliarySlot,
│                     SpiderSenseSlot, MTPSlot, TTMResidualSlot,
│                     HeartbeatSlot, IHSResidualSlot
├── recipient.py    — hooks GABA/IHS + TTM + sidecars Heartbeat/JEPA/Spider/MTP
├── lifecycle.py    — OrganState machine, configure_trainable_state, audit_gradients
├── experiment.py   — A/B/C/D arms, ArmBudget identity
├── ledger.py       — OrganLedger JSONL, organ_utility()
└── checkpoint.py   — save/load + per-organ rollback
```

Canário sem launch:
`TWO_DONOR_VERIFY_OK organs=7 disabled_error=0.000e+00 shadow_error=0.000e+00`

Execução experimental: `TWO_DONOR_RUN_OK experiments=N` (7 órgãos × seeds)
Launcher: `src/tools/start_two_donor_transplant.ps1 -Canary`

## Circuits Framework — ⚠️ biblioteca pronta, fluxo operacional ausente

Este é o **fio central do projeto**, não uma trilha auxiliar. O objetivo é
dissecar uma habilidade aprendida no nível de canais, provar causalmente onde
ela mora, e transplantá-la para um modelo incompatível:

```
modelo especialista treinado
  → descobrir quais canais carregam a habilidade
  → desligar e provar que a habilidade piora
  → religar e provar que ela volta
  → empacotar o pedaço como circuito
  → transplantar no GPT com adaptador e gate zero
  → terceiro modelo
```

JEPA, GABA e os 7 órgãos do Two-Donor foram a **primeira cirurgia**, com
peças já nomeadas. O Circuits Framework é o microscópio para achar peças
**novas** — inclusive um circuito matemático. Desenho completo em
`governance/docs/superpowers/specs/2026-07-28-cognitive-circuit-ablation-transplant-design.md`.

### O que existe (Tarefas 2–6 do plano)

```
src/f51_darwin/circuits/          5.197 linhas
├── ablation.py       1058 — descoberta por gradiente×ativação; arms CLEAN,
│                            ABLATE, RESTORE, SHUFFLE, RANDOM_MATCHED
├── ledger.py          940 — registro encadeado por hash
├── rollback.py        758 — desfaz somente um circuito
├── transaction.py     647 — inject/replace com gate zero
├── compatibility.py   591 — preflight estático e executável
├── pointer.py         476 — compare-and-swap do ponteiro ativo
├── package.py         256 — pacote .f51circuit verificável
├── manifest.py        219 — schemas imutáveis
├── taps.py            138 — resolução de pontos de conexão
└── identity.py         14 — JSON canônico + SHA-256
```

Suíte focada: **169 testes, exit 0** (17 skips são regressões POSIX,
irrelevantes no Windows).

### O que falta (Tarefas 7–8, nunca executadas)

- Os CLIs planejados `circuit_ablate.py`, `circuit_package.py` e
  `circuit_transplant.py` — **não existem**. A biblioteca não tem
  superfície de operador.
- O launcher planejado `start_circuit_vertical_slice.ps1` — não existe.
- `test_circuit_cli.py`, `test_circuit_vertical_slice.py`,
  `test_circuit_real_preflight.py` — não existem. Nunca houve prova
  ponta a ponta.
- `src/configs/circuits/` — **vazio**. Sem contratos de tap, sem contratos de
  descoberta/confirmação de habilidade.
- `workspace/04_CIRCUITS/` — não existe. Nenhum artefato real.

**Nenhum circuito matemático foi descoberto ou transplantado.** A biblioteca
prova o mecanismo; não prova localização de conhecimento semântico.

### Bloqueio histórico do vertical slice

O inventário abaixo registrava o bloqueio encontrado antes do download e da
cirurgia Smol. Ele permanece como contexto histórico do experimento de
circuitos, não como inventário atual da máquina:

| Recurso | Estado |
|---|---|
| Modelos em `workspace/00_DONORS/` | apenas `distilgpt2` |
| Cache HF de modelos | apenas `distilbert--distilgpt2` |
| Linhagens 600M / 1.6B | **não existem no disco** |
| `generation-06-final.pt` (SSD) | 0/200 em equações do 1º grau |
| V3 organism (step 1750) | `lm_loss=8.18`, mal começou |
| Cache `meta-math--MetaMathQA` | 6 arquivos, **0 MB** — só metadados |

O Darwin-Smol V3 agora fornece um espécime linguístico competente, mas ainda
não houve prova de habilidade matemática específica nem execução do vertical
slice de circuitos sobre ele. Portanto, continua incorreto afirmar que um
circuito matemático foi localizado ou transplantado.

## Organ Sequence Simulation

Simulador heurístico de sequência — 72 subconjuntos válidos, 50 cenários e
3.600 avaliações.

- **Consenso**: `jepa >> heartbeat >> spider >> ttm >> mtp >> gaba >> ihs` (27/50)
- **TTM**: maior potencial (+0.07 marginal)
- **IHS**: negativo (-0.02), caro demais

Esses resultados partem de notas manuais de qualidade, futuro, retenção e
custo em `research/simulate_organ_combinatorics.py`. São úteis para planejar,
mas não constituem evidência medida de benefício dos órgãos.

## Learning-Gain Simulation

Simulador de plasticidade seletiva — 5 políticas, 10 cenários e 30 seeds.
Todas as políticas tiveram ganho mediano zero. Os gates 3 e 5 falharam
(atualizações falsas em S03/S06 e retenção antiga insuficiente em S08).
Portanto, GainAdaptive ainda não demonstrou uma lei de aprendizagem superior.

## Linhagens

| Linhagem | Status |
|---|---|
| 1.7B SMOL_EXACT_V3 | ✅ `approved_candidate`, donor-free, dual-GPU, Heartbeat + IHS zero-gated |
| 1.7B SMOL_EXACT_V2 | Preservada como rollback sem IHS |
| 1.6B SMOL_TRANSPLANT_V1 | Reprovada: BpB/KL piores que controle aleatório |
| 100M FULL_ORGANISM_V3 | Parada (step 1750, retomável) |
| 100M FULL_ORGANISM_V1 (backup) | Preservada, não ativa |
| DARWIN_TRANSFER_DISTILGPT2_V1 | ⚠️ gen-06-final.pt reprovado no portão n-grama — evidência de engenharia, não modelo utilizável |
| DENSO_FULL_ORGANISM_V1 | Config pronta, não lançada |

## Three-Organ Cognition Foundation — shadow-only, 2026-07-29

Commits `5c7d4ac`–`f86ba9a` na branch `feat/darwin-16b-smol-transplant`.

| Prova | Resultado |
|---|---|
| Testes focados de cognição | 22/22 passando |
| Testes de regressão impactados | 65/65 passando |
| Checkpoint SHA antes/depois | `9329b8d25fc23acde021161da7a3421e6e17ceea3e940e0298162f39b62bc01e` (idêntico) |
| Brain identity antes/depois | `darwin-model-core-v1:63970865...` (idêntico) |
| Erro máximo de logits | **0,0** (exato) |
| Gates dos adaptadores | memory=0.0, world_model=0.0 |
| Eventos de pulso | 1, hash `f2c958cb...` |
| Órgãos canônicos | 3 (`organ:memory:v1`, `organ:world_model:v1`, `organ:executive:v1`) |
| Modo de dispositivo | dual-GPU (RTX 5060 Ti + RTX 3060) |
| Chaves faltantes | 6 (`cognitive_runtime.memory_adapter.*`, `cognitive_runtime.world_model_adapter.*`) |
| Rótulo | `foundation_shadow_only` |

Isto prova contratos, separação de identidade cérebro/órgão e um caminho
sombra de impacto zero sobre o checkpoint Darwin-Smol Native Dense V1 real.
**Não prova memória, predição, planejamento, qualidade de decisão ou
aprendizado.**

Artefato: `workspace/runtime/three_organs_v1/foundation-shadow.json`.

## GPUs

| GPU | Modelo | VRAM | Status |
|---|---|---|---|
| 0 | RTX 5060 Ti | 16 GB | Livre |
| 1 | RTX 3060 | 12 GB | Livre |

⚠️ Histórico: `nvlddmkm` Event ID 153 (TDR) na RTX 5060 Ti. Não se manifestou nos experimentos de transplante.
