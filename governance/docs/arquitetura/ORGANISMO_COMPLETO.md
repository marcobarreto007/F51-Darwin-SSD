# Organismo Darwin-X — Arquitetura completa

<!-- authority: arquitetura-atual -->

Documento gerado em 2026-07-18 a partir do estado verificado do codigo.
Toda referencia cita arquivo e linha exatos.

## 1. Visao geral

O organismo Darwin-X e um sistema de treino MoE (Mixture of Experts) com
aproximadamente 20 orgaos integrados em 3 linhagens de escala. O runtime e
montado por mixins em `src/f51_darwin/organism/` e o modelo em
`src/f51_darwin/darwin_x_core/`.

### 1.1 Mixins do organismo

```
_DarwinBootstrapMixin  (bootstrap.py:16)   — init, config, orgaos, causal runtime
  └── _DarwinLifecycleMixin  (lifecycle.py:11)  — run_cycle, data, sleep, mesh
        └── _DarwinTrainingMixin  (training.py:21)  — train_cycle, causal bus
              └── _DarwinCheckpointMixin  (checkpoint_mixin.py:12)  — evolution, save
                    └── _DarwinControlMixin  (control.py:9)  — LR, soul, signals
                          └── DarwinOrganism  (control.py:251)  — classe publica
```

### 1.2 Mixins do modelo

```
_ModelForwardMixin    (model.py:534)   — forward pass, loss, heartbeat
_ModelPlacementMixin  (model.py:119)   — dual GPU, organism activation
_ModelTopologyMixin   (model.py:291)   — topology manifest, restore
  └── DarwinXModel  (model.py, composicao das 3 acima)
```

## 2. Tabela de linhagens

| Parametro | 100M | 600M | 1.6B |
|---|---|---|---|
| Config | `src/configs/darwin_x_100m.yaml` | `src/configs/darwin_x_600m.yaml` | `src/configs/darwin_x_1.6b_nitro.yaml` |
| d_model | 512 | 1408 | 1920 |
| n_layers | 12 | 12 | 16 |
| n_heads | 8 | 16 | 16 |
| n_kv_heads | 2 | 4 | 4 |
| head_dim | 64 | 88 | 120 |
| context_length | 5120 | 4096 | 4096 |
| inference_context | 32768 | 32768 | 32768 |
| rope_base_train | 10000 | 10000 | 10000 |
| rope_base_infer | 100000 | 1000000 | 1000000 |
| residual_scale_multiplier | 1.2 | 1.4 | 1.4 |
| ssm_state | 16 | 64 | 16 |
| fine_experts | 8 | 6 | 14 |
| shared_experts | 2 | 1 | 2 |
| experts_per_token | 2 | 2 | 2 |
| fine_expert_hidden_dim | 512 | 768 | 896 |
| shared_expert_hidden_dim | 512 | 768 | 896 |
| mtp_depth | 2 | 2 | 2 |
| heartbeat_ff_layers | 2 | 3 | 3 |
| nitro_gpu_expert_capacity | 4 | 8 | 5 |
| scan_chunk_size | 128 | 512 | 512 |
| checkpoint_root | `workspace/03_CHECKPOINTS_100M_FULL_V9` | *(sem checkpoint)* | *(sem checkpoint)* |

### 2.1 Parametros estimados

A funcao `estimate_darwin_x_parameters()` em
`src/f51_darwin/darwin_x_core/estimation.py` (48 linhas) calcula parametros
totais e ativos por token. Para a linhagem 600M, o comentario na config
(linhas 20-23) registra:

- Total: 452.269.384 params
- Ativos por token: 296.454.400
- Runtime: aproximadamente 490.2M (com estruturas auxiliares)

### 2.2 Atencao e SSD

A arquitetura usa proporcao 3:1 de camadas SSD para atencao. A cadencia
canonica (`attention_layer_indices` em `config.py:187-193`) coloca atencao
nos indices onde `(index + 1) % 4 == 0`:

- 12 camadas: atencao em [3, 7, 11], SSD em [0,1,2,4,5,6,8,9,10]
- 14 camadas: atencao em [3, 7, 11], SSD em [0,1,2,4,5,6,8,9,10,12,13]

A atencao usa Grouped Query Attention (GQA) com `n_kv_heads` KV heads
(`layers.py:17-61`, `GQACausalAttention`). O SSD usa `SelectiveSSM` via
`SSDMixerOnly` (`layers.py`, importado em `block.py:9`).

## 3. Orgaos — estado de conexao

Cada orgao e listado com seu arquivo de definicao, ponto de integracao
e status verificado.

### 3.1 Orgaos estruturais (sempre ativos)

| Orgao | Arquivo | Integracao | Funcao |
|---|---|---|---|
| **ExpertPool** | `src/f51_darwin/expert_pool.py` | `bootstrap.py:55` | Pool de modulos especialistas com estados (candidate, active, frozen, quarantine, merged, dead) |
| **LegacyLayers** | `src/f51_darwin/legacy_layers.py` | `bootstrap.py:56` | Camadas legadas com tiers |
| **LineageTracker** | `src/f51_darwin/lineage_tracker.py` | `bootstrap.py:57` | Rastreamento de linhagem |
| **Neuroendocrine** | `src/f51_darwin/darwin_x_core/neuroendocrine.py` | `moe.py:14` (por bloco MoE) | Sistema hormonal: dopamina, cortisol, BDNF, norepinefrina, acetilcolina |
| **ReplayBuffer** | `src/f51_darwin/replay_buffer.py` | `bootstrap.py:49` | Buffer de replay com forgetting proxy |
| **DataFirewall** | `src/f51_darwin/data_firewall.py` | `bootstrap.py:50` | Firewall de dados (qualidade, charset) |
| **Brainstem** | `src/f51_darwin/brainstem.py` | `bootstrap.py:51` | Homeostase (VRAM, loss, modulos ativos) |
| **F51Soul** | `src/f51_darwin/soul.py` | `bootstrap.py:52` | Alma do organismo (comandos: ingest, exploration, benchmark, emergency, checkpoint) |

### 3.2 Orgaos com toggle (config)

| Orgao | Toggle | Default | Arquivo | Integracao | Funcao |
|---|---|---|---|---|---|
| **GABA** | `gaba_enabled` | true | `src/f51_darwin/gaba_inhibition.py` | `block.py:29-35, 49-55` | Inibicao GABAergic apos MoE, antes do residual. Observacoes coletadas em `gaba_observations` (model.py:786-790) |
| **Heartbeat / TTM** | `heartbeat_enabled` | true | `src/f51_darwin/heartbeat.py` | `model.py:248-269, 723-757` | Batimento cardiaco com TTM memory. Retrieval antes do lm_head (model.py:596-610) |
| **Ghost Token** | `ghost_enabled` | true | `model.py:666-681` | Predicao auxiliar de tokens mascarados. Segundo forward completo (~50% compute extra). Mascara deterministica via step_digest |
| **Spider Sense** | `spider_sense_enabled` | true | `src/f51_darwin/spider_sense.py` | `model.py:628-657` | Danger scoring. MLP real ou fallback heuristico. Calibracao Brier opcional |
| **JEPA** | `jepa_weight` | 0.05 | `model.py:618` | Predicao de estados latentes (Joint Embedding Predictive Architecture) |
| **MTP** | `mtp_depth` | 2 | `model.py:617` | Multi-Token Prediction com depth 2 |
| **DAE** | `dae_enabled` | true | `src/f51_darwin/dae_optimizer.py` | `model.py:277-279` | Darwin Active Gradient Engine. Modo shadow=false nos configs atuais (ativo) |
| **Nitro** | `nitro_enabled` | true | (config) | GPU expert capacity por escala |
| **Curiosity** | `curiosity_weight` | 0.02 | `causal_adapters.py:176` | Prioridade de replay baseada em curiosidade. Adapter: CuriosityPriorityAdapter |
| **Decision Engine** | `decision_engine_enabled` | true | (config toggle) | Engine de decisao (toggle, sem parametros treinaveis no forward) |
| **Unified Mesh** | `unified_mesh_enabled` | true | `lifecycle.py:44-45` | Loop de morte evolutivo unificado |
| **Inter-Hemispheric** | `inter_hemispheric_enabled` | false | `model.py:561-562` | Lateralizacao entre hemisferios (desligado) |
| **Sleep** | `sleep_enabled` | false | `lifecycle.py:42-43` | Ciclo de sono com baseline/pos e RNG pareada (desligado) |

### 3.3 Orgaos regulatorios (por bloco MoE)

Cada `DeepSeekStyleMoE` (`src/f51_darwin/darwin_x_core/moe.py`) contem:

- **NeuroendocrineSystem**: estado hormonal completo (dopamina, cortisol, BDNF,
  norepinefrina, acetilcolina, router entropy EMA, usage EMA)
- **Autonomic backward hook** (moe.py:17-60): atualiza hormonios a partir de
  sinais locais de gradiente — sem acesso a loss global
- **Expert gradient signatures**: preserva geometria assinada alem da norma
- **Structural actions**: neurogenesis, apoptose, expansao de experts
  (executadas somente no safe boundary entre ciclos)

## 4. Forward pass completo

O forward pass em `DarwinXModel.forward()` (`model.py:545-791`) executa
a seguinte sequencia:

```
1. token_embedding(input_ids)                              # model.py:559
2. [inter_hemispheric(x)]  se habilitado                   # model.py:561-562
3. PARA CADA bloco (0..n_layers-1):                        # model.py:584-590
     a. norm1 -> [attention | ssd] -> dropout -> residual   # block.py:41-45
     b. norm2 -> moe (router + experts + aux_loss)          # block.py:46
     c. [gaba(moe_out)] -> dropout -> residual              # block.py:49-56
     d. [vertical routing bias para proximo bloco]          # model.py:589-590
4. norm(hidden)                                             # model.py:592
5. [ttm_residual_gate: retrieve + scale + add]              # model.py:596-610
6. lm_head(hidden_for_logits) -> logits                     # model.py:611
7. SE labels:
     a. lm_loss = cross_entropy(logits, labels)             # model.py:616
     b. mtp_loss = _mtp_loss(hidden, labels)                # model.py:617
     c. jepa_loss = _jepa_loss(hidden)                      # model.py:618
8. [spider_sense: danger score + calibration loss]          # model.py:621-657
9. [ghost_token: mask + second forward + loss]              # model.py:664-681
10. compose_loss(terms, policy) -> total_loss               # model.py:688-706
11. [heartbeat.beat()] se habilitado                        # model.py:723-757
12. DarwinXOutput com todos os termos                       # model.py:759-791
```

### 4.1 Composicao de loss

A funcao `compose_loss()` em `src/f51_darwin/darwin_x_core/losses.py` combina
os termos com politica versionada:

- `loss_semantics_version=1` (legacy): comportamento historico v7
- `loss_semantics_version=2` (causal): aplica `aux_loss_scale` de verdade,
  suporta zero exato, propaga termos effective para o causal bus

Termos: `LossTerms(lm, mtp, jepa, aux, ghost, spider)`
Politica: `LossPolicy(mtp_scale, jepa_scale, aux_scale, ghost_scale, spider_scale, aux_adaptive)`

## 5. Barramento causal (causal bus)

### 5.1 Arquivos

| Arquivo | Linhas | Funcao |
|---|---|---|
| `src/f51_darwin/organism/causal_bus.py` | ~500 | Tipos, fases, allowlist, OrganCausalBus |
| `src/f51_darwin/organism/causal_adapters.py` | 592 | Adapters stateless + CausalTrainingExecutor |
| `src/f51_darwin/organism/causal_ledger.py` | ~300 | Ledger JSONL hash-chained com validacao |

### 5.2 Fases e allowlist

Definido em `causal_bus.py:55-75` (`_ALLOWED_INTERVENTIONS`):

| Fase | Targets permitidos | Operacoes |
|---|---|---|
| `PRE_LOSS` | `LOSS_TERM` | `SET_SCALE` |
| `PRE_BACKWARD` | `UPDATE` | `SKIP` |
| `PRE_OPTIMIZER` | `UPDATE`, `GRAD_CLIP`, `GRADIENT_GROUP` | `SKIP`, `SET_MAX_NORM`, `SCALE` |
| `POST_STEP` | (nenhum) | (observacao) |
| `CYCLE_BOUNDARY` | `STRUCTURAL_ACTION` | `QUEUE` |

### 5.3 Modos (AblationArm)

Definido em `causal_bus.py:33-36`:

| Modo | Arm | Comportamento |
|---|---|---|
| `disabled` | None | Barramento nao inicializado, caminho legado exato |
| `control` | CONTROL | Observa, registra no ledger, nao aplica |
| `shadow` | SHADOW | Observa, valida allowlist, registra, nao aplica **(default)** |
| `enforce` | APPLY | Aplica intervencoes aceitas |

### 5.4 Adapters

Registrados no bootstrap (`bootstrap.py:5-13`):

| Adapter | Classe | Fase observada | Funcao |
|---|---|---|---|
| ExplicitRequestAdapter | `causal_adapters.py:49` | PRE_LOSS, PRE_BACKWARD, PRE_OPTIMIZER | Traduz requests JSON em intervencoes |
| GABAInterventionAdapter | `causal_adapters.py:128` | PRE_LOSS | Observa balanco inibitorio GABA |
| CuriosityPriorityAdapter | `causal_adapters.py:176` | PRE_LOSS | Propoe prioridade de replay |
| SpiderCalibrationAdapter | `causal_adapters.py:219` | PRE_BACKWARD | Observa perigo spider para calibracao |

### 5.5 Executor

`CausalTrainingExecutor` (`causal_adapters.py:333-548`) aplica intervencoes
aceitas apos o registro duravel da intencao no ledger:

- `adjust_loss()` / `set_loss_scales()`: recomposicao exata de termos raw
  com scales aceitas (aux, ghost, jepa, spider)
- `should_skip()`: skip de update (PRE_BACKWARD / PRE_OPTIMIZER)
- `scale_gradients()`: escala gradientes por nome de parametro ou grupo
- `clip_gradients()`: grad clip com max norm customizado

### 5.6 Integracao no loop de treino

No `_train_cycle()` (`training.py:100-201`):

1. Cria `StepIdentity` deterministico (batch hash + RNG hash + attempt serial)
2. Constroi `training_context` (losses dictionary, optimizer_due, requests)
3. Chama `causal_bus.decide()` para PRE_LOSS, PRE_BACKWARD, PRE_OPTIMIZER
4. Registra intencao duravel: `causal_bus.record_intent()`
5. Se bloqueado: zero_grad, skip, registra outcome com erro
6. Se nao: `CausalTrainingExecutor` aplica intervencoes
7. Forward do modelo com `step_digest` deterministico
8. Registra outcome: `_complete_causal_training_step()`

## 6. Loop de treino (ciclo completo)

`DarwinOrganism.run_cycle()` em `lifecycle.py:14-52`:

```
1. self.cycle += 1
2. STAGE 1-4: self._data_lifecycle(report)
   — quarentena, aprovacao, consolidação de dados
3. STAGE 5-7: self._train_cycle(steps, report)
   — loop de treino com causal bus integrado
   — metric channels, holdout evaluation
   — gradient accumulation (accum_steps)
   — brainstem homeostasis check
   — emergency save via signal handlers (R5)
4. STAGE 7.5: self._execute_structural_boundary(report)
   — neurogenesis, apoptose, expansao (safe boundary)
5. STAGE 8: self._evolution_cycle(report)
   — le estado neuroendocrino, conta modulos ativos
6. [self._sleep_cycle(report)]  se sleep_enabled
7. [self._evolution_death_loop(report)]  se unified_mesh_enabled
8. STAGE 9-10: self._save_cycle(report)
   — checkpoint atomico, async write (R4)
   — lineage_root identity
   — topology manifest
   — optimizer state + RNG state
```

### 6.1 Detalhe do train_cycle

`_DarwinTrainingMixin._train_cycle()` em `training.py:24-201`:

- Loader: `CausalLMDataLoader` ou `WeightedCorpusLoader`
- Curriculum: weighted multi-source quando disponivel
- Replay: `replay_step_due()` deterministico (support.py:5-8)
- Grad accumulation: `accum_steps` micro-batches (R3)
- LR warmup: `_apply_lr_schedule()` (control.py:12-25, R2)
- Causal bus: phases PRE_LOSS, PRE_BACKWARD, PRE_OPTIMIZER (ver secao 5.6)
- Forward: `torch.amp.autocast` com `step_digest`
- Backward + optimizer step + DAE actions + metricas
- Emergency break: `_emergency_save_requested` (R5)

## 7. Sistema de checkpoints

### 7.1 Isolamento por linhagem

`src/f51_darwin/organism/checkpoint_root.py` (210 linhas, 9 funcoes publicas):

| Funcao | Linha | Funcao |
|---|---|---|
| `normalized_model_config()` | 18 | Normaliza config para dict canonico |
| `model_config_identity()` | 30 | SHA-256 do config normalizado (prefixo `darwin-config-v1:`) |
| `checkpoint_metadata()` | 40 | Extrai metadata de arquivo .pt |
| `read_lineage_root_identity()` | 76 | Le `lineage_root.json` |
| `write_lineage_root_identity()` | 86 | Cria/verifica ancora de identidade |
| `build_lineage_root_identity()` | 102 | Constroi payload da ancora |
| `preflight_checkpoint_root()` | 125 | Valida fresh-start (raiz vazia) ou resume (match de identidade) |
| `assert_new_checkpoint_target()` | 171 | Recusa sobrescrever .pt ou .pt.tmp |
| `ensure_lineage_root_identity()` | 182 | Cria ou valida ancora existente |

### 7.2 Estrutura do checkpoint

Cada checkpoint `.pt` contem (`checkpoint_mixin.py:129-176`):

```python
{
    "version": 7 ou 8 (causal),
    "base_checkpoint_id": backbone_identity(model_state, config),
    "tokenizer_id": str,
    "model_state_dict": {cpu},
    "topology_manifest": model.topology_manifest(),
    "config": dict(model_config),
    "optimizer_state_dict": {cpu},
    "optimizer_type": str,
    "optimizer_identity": {name, class, version},
    "dae": {enabled, shadow_mode, last_report},
    "heartbeat_state": {cpu},
    "rng_state": {python, numpy, torch_cpu, torch_cuda},
    "online_learning_state": dict | None,
    "online_learning_artifact": dict,
    "checkpoint_migration": dict | None,
    "training_state": {step, cycle, warmup_steps, warmup_remaining, accum_steps},
    "token_source": {path, token_count, ...},
    "organism": {cycle, ...},
    ...
}
```

### 7.3 Linhagem e resume

- Checkpoint v7: legado, nunca reinterpretado como causal
- Checkpoint v8: requer migracao explicita (`causal_v8_migration=True`)
- Resume estrito: shapes, config identity, tokenizer_id, base_checkpoint_id
  devem coincidir exatamente
- `strict_resume_compatible` e `identity_verified` no manifest de verificacao

## 8. Superficie de scripts

Scripts canonicos (documentados em CLAUDE.md):

| Script | Funcao |
|---|---|
| `src/scripts/darwin_organism.py` | CLI do organismo (bootstrap, cycle, status, evolve, school, serve, run247) |
| `src/scripts/inspect_organism_checkpoint.py` | Inspecao de checkpoint com `--verify-identity` |
| `src/scripts/darwin_inventory.py` | Inventario de artefatos |
| `src/scripts/serve_davi.py` | Servidor de inferencia |
| `src/scripts/ingest_pipeline.py` | Pipeline de ingestao de dados |
| `src/scripts/start_overnight_16b.ps1` | Script overnight com gate canary |

## 9. Dependencias criticas

### 9.1 Config

- `DarwinXConfig` (`src/f51_darwin/darwin_x_core/config.py`, 243 linhas): dataclass
  frozen com validacao em `__post_init__`. Defaults representam a linhagem
  1.6B-Nitro historica; as 3 linhagens ativas sobrescrevem todos os campos via
  YAML.
- `DarwinOrganismConfig` (`src/f51_darwin/organism/config.py`, 84 linhas): config
  operacional do organismo (batch, LR, replay, causal_mode, etc.)

### 9.2 Tokenizer

- `F51BPETokenizer` (`src/f51_darwin/tokenizer.py`): BPE com vocab_size=58162
- Diretorio: `workspace/tokenizer/f51_bpe_80k`

### 9.3 Optimizer

- `DAEHybridOptimizer` (`src/f51_darwin/dae_optimizer.py`): wrapper com DAE
  (Darwin Active Gradient Engine) sobre optimizer base (AdamW)
- `build_optimizer()`: factory function

### 9.4 Corpus

- `feast_v2`: corpus canonico
- `CausalLMDataLoader`: loader sequencial com block_size e batch_size
- `WeightedCorpusLoader`: loader multi-source com pesos de curriculum
- `DataFactory` / `DataFactoryPaths`: fabrica de dados com paths
- `WorkspacePaths`: resolve paths de dataset, token bin, checkpoints
