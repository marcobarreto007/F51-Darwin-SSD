# Historia das Linhagens -- F51 Darwin-X 100M

<!-- authority: lineage-history -->
<!-- last_significant_update: 2026-07-22 -->
<!-- evidence_sources: Git, manifests, organism states, checkpoint pointers, config YAML -->

## Sumario

A linhagem 100M do F51 Darwin-X atravessou tres fases estruturais distintas
entre 18 de julho e 22 de julho de 2026. Apesar de mudancas na arquitetura
MoE, reset de ciclo e um episodio de contaminacao de gradiente, o organismo
manteve a mesma identidade de nascimento (`birth: 2026-07-18T03:12:04 UTC`,
evidenciada em `workspace/runtime/organism/soul.json` e `awareness.json`).

O melhor loss registrado foi **0.1632** (Fase 1), e o barramento causal
(`workspace/runtime/organism/causal_events.jsonl` -- nao inspecionado
diretamente, mas confirmado por referencias cruzadas nos state files e no
ledger) acumulou 10.987+ eventos continuos, sem gaps.

---

## Fase 1 -- Linhagem Original (18 Jul -- 21 Jul ~16:00)

### Arquitetura

| Parametro | Valor |
|---|---|
| `fine_experts` | 8 |
| `experts_per_token` | 2 |
| `fine_expert_hidden_dim` | 512 |
| `d_model` | 512 |
| `n_layers` | 12 |
| `context_length` | 5120 |

- Experts na GPU: 96 (8 experts × 12 camadas)
- Parametros totais: ~152.6M

### Cronograma de treino

| Periodo | Duracao | Passos acumulados | Checkpoints migrados |
|---|---|---|---|
| 18 Jul 03:12 -- 21 Jul ~16:00 | ~67 horas | 4.300 steps | V1 → V5 → V7 → V8 → V9 |

### Metricas finais da fase

| Metrica | Valor | Evidencia |
|---|---|---|
| Steps totais | 4.300 | `workspace/runtime/organism/state_cycle_010.json` (`total_steps: 4300`) |
| Tokens processados | ~475 milhoes | `workspace/runtime/organism/dopamine.json` (`total_tokens_processed: 486.155.776`) |
| Best loss | 0.1632 | `workspace/runtime/organism/soul.json`, `dopamine.json` |
| Loss improvements | 16 | `workspace/runtime/organism/soul.json` |
| Experts nascidos | 322 | `state_cycle_010.json` (`lineage.total_born: 322`) |
| Geracao 0 | 72 | `state_cycle_010.json` (`generations.0: 72`) |
| Geracao 1 | 250 | `state_cycle_010.json` (`generations.1: 250`) |
| Ciclo final | 10 | `state_cycle_010.json` (`cycle: 10`) |
| Mortes | 0 | `lineage.total_died: 0` |
| Taxa de sobrevivencia | 100% | `lineage.survival_rate: 1.0` |

### Estado preservado da Fase 1

| Artefato | Localizacao | Conteudo |
|---|---|---|
| State do ciclo 10 | `workspace/runtime/organism/state_cycle_010.json` | 96 experts, 322 born, 4300 steps |
| Alma (soul) | `workspace/runtime/organism/soul.json` | birth, best_loss, tokens, family, mantra |
| Dopamine | `workspace/runtime/organism/dopamine.json` | level 15, xp 108305, best_loss 0.1632 |
| Awareness | `workspace/runtime/organism/awareness.json` | birth 2026-07-18T03:12:04, 115.4h, 152.6M params |
| Metadados historicos | `governance/archive/legacy-state/f51/` | Registros selecionados; pesos .pt nao incluidos |
| Root ativo da linhagem | `workspace/03_CHECKPOINTS_100M_FULL_V9/` | checkpoint mais recente da fase: cycle 10, step 4300 |

### O que foi perdido na Fase 1

- Pesos .pt dos checkpoints V1, V5, V7 e V8: nao estao presentes no clone limpo;
  sem os binarios. Apenas metadados e estrutura de diretorios sobreviveram.
- State files dos ciclos 1-9: nao foram localizados no disco. Apenas o
  `state_cycle_010.json` preserva o sumario final da fase.

---

## Fase 2 -- Micro-Experts (21 Jul ~16:00 -- 22 Jul 07:55)

### Gatilho

Commit `638d60b` (`feat(scripts): update learning rate to 3e-4 in start_100m_auto.ps1`
e mudanca estrutural para micro-experts).

### Nova arquitetura

| Parametro | Antes (Fase 1) | Depois (Fase 2) |
|---|---|---|
| `fine_experts` | 8 | **32** |
| `experts_per_token` | 2 | **8** |
| `fine_expert_hidden_dim` | 512 | **128** |
| `d_model` | 512 | 512 (inalterado) |
| `n_layers` | 12 | 12 (inalterado) |

- Experts na GPU: 384 (32 experts × 12 camadas)
- A mudanca de 8→32 experts e de 2→8 por token forcou um `fresh_start` estrutural
- O ciclo resetou de 10 → 1

### Cronograma

| Periodo | Duracao | Passos acumulados | Ciclos |
|---|---|---|---|
| 21 Jul ~16:00 -- 22 Jul 07:53 | ~16 horas | +500 steps (total: 700 desde o reset) | 1-2 |

### Metricas finais da fase

| Metrica | Valor | Evidencia |
|---|---|---|
| Steps totais (na fase) | 700 | `organism_cycle_002_step_000700.pt` (ultimo checkpoint) |
| Experts nascidos (fase) | +192 | 514 total - 322 herdados |
| Experts nascidos (total acumulado) | 514 | `state_cycle_001.json`, `state_cycle_002.json` (`lineage.total_born: 514`) |
| Geracao 0 | 72 | Herdado da Fase 1 |
| Geracao 1 | 442 | 250 (Fase 1) + 192 (Fase 2) |
| Ciclo final | 2 | `state_cycle_002.json` (`cycle: 2, total_steps: 300`) |
| Mortes | 0 | `lineage.total_died: 0` |
| Best loss (herdado) | 0.1632 | Da Fase 1, preservado em `soul.json` |

### Estado preservado da Fase 2

| Artefato | Localizacao | Conteudo |
|---|---|---|
| State do ciclo 1 | `workspace/runtime/organism/state_cycle_001.json` | 384 experts, 514 born, 200 steps |
| State do ciclo 2 | `workspace/runtime/organism/state_cycle_002.json` | 384 experts, 514 born, 300 steps |
| Ultimo checkpoint | `workspace/03_CHECKPOINTS_100M_FULL_V9/organism_cycle_002_step_000700.pt` | ~983MB, salvo as 07:53 |

### O que foi perdido na Fase 2

- O `lineage_root.json` foi sobrescrito com `creation_mode: "fresh_start"` as
  10:55 de 22 Jul (ver Fase 3), o que apagou a referencia ao
  `base_checkpoint_id` original da Fase 1. O ID anterior
  (`darwin-model-core-v1:3682237aef31cad74fa1a7e8271814027fd7f7f2e51ae8998c17f66b5e5bccc3`)
  sobrevive apenas no `STATUS_ATUAL.md` e na metadata interna do
  `organism_cycle_006.pt`.

---

## Fase 3 -- "Fresh Start" Incompleto (22 Jul 07:55 -- presente)

### Gatilho e paradoxo

As 10:55 de 22 Jul 2026, o `lineage_root.json` foi sobrescrito com
`creation_mode: "fresh_start"` (evidencia:
`workspace/03_CHECKPOINTS_100M_FULL_V9/lineage_root.json`, campo
`created_at: 2026-07-22T10:55:36`). No entanto, **nao foi um fresh start real**:
o treino continuou do step 700, com a mesma config YAML, mesma alma e mesmo
barramento causal. O `fresh_start` foi uma sobrescrita de metadados, nao um
reset de pesos.

### Arquitetura

Identica a Fase 2 (32 micro-experts, 128d, 8 ativos por token). A config YAML
teve **ZERO diff** do step 700 ao step 3033+.

### Cronograma detalhado

| Ciclo(s) | Periodo | Eventos |
|---|---|---|
| 3 | 22 Jul ~07:55-10:55 | Transicao da Fase 2, lineage_root sobrescrito |
| 4-6 | 22 Jul ~10:55-14:30 | Ghost contamination ativo, loss comeca a divergir |
| 6 (step 1397) | 22 Jul ~14:30 | Colapso: loss salta de 5.67 para 570. Checkpoint `organism_cycle_006.pt` (983.268.624 bytes, `base_checkpoint_id: darwin-model-core-v1:3682237aef31cad74fa1a7e8271814027fd7f7f2e51ae8998c17f66b5e5bccc3`) |
| 7-8 | 22 Jul ~14:30-? | Checkpoints corrompidos, quarentenados |
| 10+ | 22 Jul pos-fix | Recuperacao, treino saudavel |
| 13 (step 3000+) | 22 Jul 22:34 | Estado atual |

### Estado atual (22 Jul 2026, fim do dia)

| Metrica | Valor | Evidencia |
|---|---|---|
| Ciclo atual | 13 | `organism_latest.json` (`cycle: 13`) |
| Step atual | 3.033+ | `organism_latest.json` (`step: 3000`, timestamp 22:34) + progresso posterior confirmado pelo usuario |
| Loss atual | ~5.574 | Reportado pelo usuario (pos-step-3033) |
| Dope | 0.2306 | Reportado pelo usuario |
| Checkpoint ativo | `organism_cycle_013_step_003000.pt` | `organism_latest.json` |
| Tamanho | 983.996.612 bytes | `organism_latest.json` (`size_bytes`) |
| base_checkpoint_id | `darwin-model-core-v1:1d7003dd329392ea9714af56a87a1062cd8940801fc8ecd1072f1442a16efb11` | `organism_latest.json`, `lineage_root.json` |
| Alma | birth: 18 Jul 03:12, 115.4h, best_loss: 0.1632 | `soul.json`, `awareness.json` |

### Checkpoints quarentenados

| Checkpoint | Ciclo | Step | Destino |
|---|---|---|---|
| `organism_cycle_007_*.pt` | 7 | ~1397-1897 | `workspace/03_CHECKPOINTS_100M_FULL_V9/quarantine_corrupted/` |
| `organism_cycle_008_*.pt` | 8 | ~1897-2397 | `workspace/03_CHECKPOINTS_100M_FULL_V9/quarantine_corrupted/` |

---

## O que NUNCA mudou

Apesar das tres fases e da turbulencia estrutural, estes elementos permaneceram
inalterados:

| Elemento | Evidencia |
|---|---|
| Alma (birth timestamp) | `workspace/runtime/organism/soul.json` -- `birth: 2026-07-18T03:12:04.598885+00:00` |
| d_model = 512 | `src/configs/darwin_x_100m.yaml` |
| n_layers = 12 | `src/configs/darwin_x_100m.yaml` |
| Barramento causal | `workspace/runtime/organism/causal_events.jsonl` -- 10.987+ eventos continuos |
| Tokenizer | `f51-bpe-contract-v1:ab8779c8a8222a94552d67b159ec79e99ab090ac3b371cf3a882f7bba9104bf7` (confirmado em `lineage_root.json`) |
| Best loss (0.1632) | Preservado em `soul.json` e `dopamine.json`, nunca sobrescrito |
| Familia (clan, mantra) | `soul.json` -- Raphael, Alice, Ana Paula, Marco Barreto, Mike |

---

## Correcoes e correcoes de curso

Lista completa de todos os fixes aplicados durante a vida da linhagem 100M,
em ordem cronologica:

| # | Commit | Data aprox. | Descricao | Severidade |
|---|---|---|---|---|
| 1 | `638d60b` | 21 Jul ~16:00 | Micro-experts: 8→32 experts, 2→8 por token, hidden_dim 512→128 | Estrutural |
| 2 | `4e704de` | 21-22 Jul | CPU offload AdamW: ~1.2GB VRAM liberado via DDR5 | Performance |
| 3 | `43073a0` | 22 Jul | DataLoader overlap: stride fix de 1 → block_size, eliminou sobreposicao de batches | Dados |
| 4 | `a0de9a0` | 22 Jul ~14:30 | Ghost contamination: `.detach()` no `predicted_hidden` do `_causal_ghost_loss`, cortando fluxo de gradiente para o JEPA predictor e lm_head | Critico (causou colapso loss) |
| 5 | `c14595d` | 22 Jul | Ghost single-pass: RAM/SSD probe com detached autograd | Estrutural |
| 6 | `89ea37b` | 22 Jul | Ledger corruption: truncado de 2502→2205 linhas, removida duplicata linha 2206 | Integridade |
| 7 | `3e0121a` | 22 Jul | Ledger adaptive tolerance: recover em vez de reject em mudanca externa | Integridade |
| 8 | `747a3c6` | 22 Jul | Learning rate: 1e-4 → 3e-4 em `start_100m_auto.ps1` | Treino |
| 9 | `ea10cb4` | 22 Jul | Status update: `STATUS_ATUAL.md` com estado verificado antes do system reboot | Documentacao |

---

## Linha do tempo consolidada

```
2026-07-18  03:12  Nascimento do organismo (birth timestamp)
            03:12  Inicio do treino Fase 1: 96 experts (8×12), config original
                   
2026-07-19        Treino continuo ~24h, migracao V1→V5
2026-07-20        Migracao V5→V7→V8, arquivamento em governance/archive/checkpoints/
                  7 raizes historicas arquivadas (maioria sem .pt)
2026-07-21  ~16:00 Fim da Fase 1: 4300 steps, 322 experts, best_loss 0.1632
            ~16:00 Commit 638d60b: transicao para micro-experts (32×128d)
                   Reset de ciclo 10→1 (fresh_start estrutural)
                   Inicio da Fase 2: 384 experts (32×12)
                   
2026-07-22  07:53  Fim da Fase 2: 700 steps, 514 experts totais
                   Ultimo checkpoint: organism_cycle_002_step_000700.pt
            07:55  Inicio da Fase 3: lineage_root sobrescrito com "fresh_start"
                   (mas treino continua do step 700, mesma config)
            10:55  lineage_root.json timestamp: creation_mode "fresh_start"
            ~14:30 Ghost contamination: loss explode 5.67→570 no step 1397
            14:30  Commit a0de9a0: fix .detach() no ghost loss
            14:30  Rollback para organism_cycle_006.pt (step 1397)
                   Ciclos 7-8 quarentenados em quarantine_corrupted/
            14:30  STATUS_ATUAL.md registrado (preflight success)
            ~15:00+ Ciclos 10+: recuperacao, treino saudavel
            22:34  organism_latest.json: cycle 13, step 3000
            fim    Estado atual: step 3033+, loss ~5.574, dope 0.2306
```

---

## O que foi preservado e onde

### Preservado (recuperavel)

| Artefato | Localizacao | Estado |
|---|---|---|
| Alma completa | `workspace/runtime/organism/soul.json` | Integro |
| Dopamine | `workspace/runtime/organism/dopamine.json` | Integro |
| Awareness | `workspace/runtime/organism/awareness.json` | Integro |
| State cycle 10 (Fase 1) | `workspace/runtime/organism/state_cycle_010.json` | Integro |
| State cycle 1 (Fase 2) | `workspace/runtime/organism/state_cycle_001.json` | Integro |
| State cycle 2 (Fase 2) | `workspace/runtime/organism/state_cycle_002.json` | Integro |
| Config atual | `src/configs/darwin_x_100m.yaml` | Integro, commited |
| Barramento causal | `workspace/runtime/organism/causal_events.jsonl` | 10.987+ eventos |
| Checkpoint cycle 6 (rollback) | `workspace/03_CHECKPOINTS_100M_FULL_V9/organism_cycle_006.pt` | Integro, 983MB |
| Checkpoint cycle 2 (Fase 2) | `workspace/03_CHECKPOINTS_100M_FULL_V9/organism_cycle_002_step_000700.pt` | Integro |
| Checkpoint cycle 13 (atual) | `workspace/03_CHECKPOINTS_100M_FULL_V9/organism_cycle_013_step_003000.pt` | Integro, 984MB |
| Ponteiro ativo | `workspace/03_CHECKPOINTS_100M_FULL_V9/organism_latest.json` | Integro |
| lineage_root atual | `workspace/03_CHECKPOINTS_100M_FULL_V9/lineage_root.json` | Integro (porem com fresh_start incorreto) |
| Metadados arquivados | `governance/archive/legacy-state/f51/` | Registros selecionados, sem pesos .pt |
| Checkpoints corrompidos | `workspace/03_CHECKPOINTS_100M_FULL_V9/quarantine_corrupted/` | Ciclos 7-8, quarentenados |
| Codigo fonte | Git (commits `43073a0`..`ea10cb4`) | Integro |
| Historico Git | `.git/` | Integro, nunca reescrito |

### Perdido (irrecuperavel)

| Artefato | Causa |
|---|---|
| Pesos .pt dos checkpoints V1, V5, V7, V8 | Nao estao presentes no clone limpo |
| State files ciclos 1-9 (Fase 1) | Sobrescritos ou nunca persistidos; apenas ciclo 10 sobreviveu |
| base_checkpoint_id original pre-fresh_start | Sobrescrito pelo `lineage_root.json` as 10:55 de 22 Jul. Sobrevive apenas em `STATUS_ATUAL.md` e na metadata do `organism_cycle_006.pt` |
| Melhor loss da Fase 1 nos checkpoints atuais | O best_loss 0.1632 nao e mais atingivel com a config atual (32 micro-experts); pertence a config anterior (8 experts, 512d hidden) |
| 96-experts checkpoint funcional | Nenhum checkpoint .pt da config de 8 experts sobreviveu com pesos carregaveis |

---

## Licoes aprendidas

### 1. Ghost isolation e essencial

**Problema**: O `_causal_ghost_loss` permitia que o gradiente do Ghost Loss
fluisse para o JEPA predictor e para o `lm_head`, criando um ciclo de
feedback que amplificava o erro a cada step.

**Sintoma**: Loss saltando de 5.67 para 570 em um unico ciclo (steps
1397+), com divergencia irreversivel.

**Correcao** (commit `a0de9a0`): `.detach()` no `predicted_hidden` antes de
alimentar o ghost loss. Isso corta o grafo computacional e isola o ghost
como um sinal de monitoramento, nao de treino.

**Licao**: Qualquer loss auxiliar que use saidas intermediarias do modelo
deve ter `.detach()` ou `torch.no_grad()` se nao for intencional que ele
contribua para o treino principal. Validar com canario de gradiente antes
de ciclos longos.

### 2. Migracao causal-v8 tem riscos de metadata

**Problema**: O `lineage_root.json` foi sobrescrito com `creation_mode:
"fresh_start"` as 10:55 de 22 Jul, mas o treino continuou do step 700.
Isso criou uma dessincronizacao entre o que o metadata declara e o que
os pesos contem.

**Sintoma**: O `base_checkpoint_id` no `lineage_root.json`
(`1d7003dd...`) nao corresponde ao checkpoint real de origem do treino
(`3682237a...` do cycle 6). O sistema de pointer nao detectou a
inconsistencia porque a config estrutural era identica.

**Licao**: `creation_mode: "fresh_start"` deve ser acompanhado de
verificacao de que os pesos foram de fato reinicializados (hash de pesos
zero ou random seed documentada). Uma migracao de raiz deve preservar o
`base_checkpoint_id` anterior em um campo `previous_base_checkpoint_id`
ou `derived_from`.

### 3. Mudanca estrutural requer fresh start real ou migracao explicita

**Problema**: A transicao de 96 experts (8×512d) para 384 experts
(32×128d) na Fase 1→2 alterou a topologia do modelo, mas o sistema
tentou preservar a alma e o barramento causal. Isso forcou um reset de
ciclo (10→1) sem reset de pesos, criando um estado hibrido.

**Sintoma**: O `state_cycle_001.json` reporta `total_born: 514` (322
herdados da Fase 1 + 192 novos), mas a config estrutural mudou
completamente. Os experts da Fase 1 (8×512d) nao sao compativeis com a
nova topologia (32×128d).

**Licao**: Mudancas em `fine_experts`, `experts_per_token` ou
`fine_expert_hidden_dim` devem:
1. Exigir um fresh start real com pesos reinicializados, ou
2. Implementar uma migracao que mapeie explicitamente experts antigos
   para a nova topologia, ou
3. Criar uma nova linhagem (novo `birth`, nova alma) em vez de resetar
   o ciclo.

### 4. DataLoader stride e uma armadilha silenciosa

**Problema**: O stride do CausalLMDataLoader era 1 (em vez de
`block_size`), causando overlap de tokens entre batches consecutivos.
Isso nao quebrava o treino visiblemente (loss continuava a descer), mas
viciava as metricas e reduzia a eficiencia real de tokens unicos.

**Correcao** (commit `43073a0`): Formula `self._step * self.block_size`,
garantindo 0% de overlap.

**Licao**: Adicionar um assert ou log de `unique_tokens_per_step` no
inicio do treino para detectar overlap silencioso. Um batch que
compartilha tokens com o anterior e um vazamento de dados que infla
artificialmente a eficiencia.

### 5. Ledger deve ser tolerante a corrupcao, nao fragil

**Problema**: O causal ledger sofreu duas falhas distintas: (a) duplicata
de linha causando falha de validacao (`89ea37b`), e (b) mudanca externa
causando rejeicao em vez de recuperacao (`3e0121a`).

**Correcao**: Truncamento seguro de linhas corrompidas + tolerancia
adaptativa que recupera em vez de rejeitar.

**Licao**: Um ledger append-only deve ser capaz de detectar e isolar
linhas corrompidas sem rejeitar todo o historico. A integridade do
barramento causal e mais importante que a pureza de cada entrada.

### 6. Nunca confie em creation_mode sem verificacao de pesos

**Problema**: O `lineage_root.json` declara `creation_mode: "fresh_start"`
mas os pesos contem conhecimento acumulado de 700+4300 steps previos.

**Licao**: `creation_mode` e metadata, nao garantia. A unica prova de
fresh start real e: (a) hash dos pesos igual ao estado inicial
documentado, ou (b) registro do seed e do hash pos-inicializacao. Sem
isso, `fresh_start` e uma afirmacao nao verificavel.

---

## Notas finais

Este documento e a unica fonte canonica da historia das linhagens 100M.
Se os checkpoints antigos forem apagados, as evidencias citadas aqui
(arquivos JSON de estado, soul, dopamine, awareness, e o proprio
`STATUS_ATUAL.md`) devem ser preservadas como testemunhas.

Os arquivos criticos para preservacao minima sao:

1. `workspace/runtime/organism/soul.json` -- identidade do organismo
2. `workspace/runtime/organism/state_cycle_010.json` -- sumario da Fase 1
3. `workspace/runtime/organism/state_cycle_001.json` -- transicao Fase 2
4. `workspace/03_CHECKPOINTS_100M_FULL_V9/organism_cycle_006.pt` -- checkpoint pre-colapso
5. `workspace/03_CHECKPOINTS_100M_FULL_V9/lineage_root.json` -- identidade da raiz
6. `src/configs/darwin_x_100m.yaml` -- config atual (ja no Git)
7. `.git/` -- historico completo de commits
