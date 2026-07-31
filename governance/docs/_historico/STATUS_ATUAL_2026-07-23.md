# Snapshot historico de 2026-07-23 -- F51 Darwin-X 100M

<!-- authority: current-status -->

Snapshot local-only de 2026-07-23, verificado por inspecao direta de Git,
config, checkpoint pointer (`organism_latest.json`), disco e scripts de resume.
A hierarquia de prova e: runtime/artefato carregado, manifest/hash, codigo/teste,
documento canonico e historico. Nao inferir linhagem por loss, mtime, cycle ou
nome de arquivo.

---

## 1. Sumario executivo

O F51-Darwin-X-100M e um modelo MoE/SSD com organismo causal de ~183M parametros
(~0.15B efetivos em forward). Esta rodando com 17 orgaos ativos em modo causal
`shadow` (observe-only, sem intervencoes), dual GPU (RTX 5060 Ti + RTX 3060),
AdamW states em CPU offload, block-size 4096, batch-size 1.

**Estado atual do treino (runtime):**
- Step: 7912+ (cycle 23)
- Loss total: ~5.7-5.8 (lm ~4.8, total ~5.8)
- Barramento causal: modo `shadow`
- LR schedule: cosine decay ativo (3e-4 → 3e-5 em 100k steps, ~98% do initial neste step)

**Ultimo checkpoint verificado no disco:**
`workspace/03_CHECKPOINTS_100M_FULL_V9/organism_cycle_023_step_007900.pt`

- Tamanho: ~938 MB
- `checkpoint_version`: 8
- `base_checkpoint_id`: variavel (evolui a cada ciclo)
- Salvo em: 2026-07-23T10:47:21 UTC
- Ponteiro ativo: `organism_latest.json` aponta para este checkpoint

**Progresso da noite (2026-07-22 23:33 → 2026-07-23 06:58):**
- Steps: 4331 → 7912 (+3.581 passos)
- Ciclos completos: 15 → 23 (8 ciclos)
- Tokens: 17.7M → 32.4M (+14.7M tokens)
- lm baseline: ~4.5 → ~4.3 (reducao de ~4.4%)
- Zero brainstem_rejected na janela noturna

---

## 2. Correcoes estruturais aplicadas (2026-07-23)

### Fix 1: LR Cosine Decay (commit `f79b4c5`)
- `src/f51_darwin/organism/control.py`: `_apply_lr_schedule()` agora tem fase cosine decay
- `src/f51_darwin/organism/config.py`: novos campos `lr_decay_steps: int = 0`, `lr_final_ratio: float = 0.1`
- `src/f51_darwin/organism/cli.py`: exposto `--lr-decay-steps`, `--lr-final-ratio`
- `src/scripts/start_100m_auto.ps1`: 100k steps decay, floor 0.1
- Default `lr_decay_steps=0` preserva comportamento legado (warmup-only)
- Formula: `cosine_factor = lr_final + 0.5*(1-lr_final)*(1+cos(pi*progress))`
- Per-group `initial_lr` anchoring preserva JEPA x10 multiplier

### Fix 2: JEPA Tracking Buffers (commit `03cf97e`)
- `src/f51_darwin/darwin_x_core/model.py`: `_jepa_loss()` agora chama `update_metrics()`
- Buffers `loss_ema`, `cosine_ema`, `update_count` recebem telemetria live
- Guardado com `hasattr()` para fallback `nn.Sequential`

### Fix 3: GABA Gate Reanimation (commit `03cf97e`)
- `src/f51_darwin/gaba_inhibition.py`: `residual_gate` init `zeros` → `randn*0.02`
- `src/f51_darwin/organism/bootstrap.py`: resume hook reanima gates zerados em checkpoints antigos

### Fix 4: TTM Gate Reanimation (commit `03cf97e`)
- `src/f51_darwin/darwin_x_core/model.py`: `ttm_residual_gate` init `zeros` → `randn*0.01`
- `src/f51_darwin/organism/bootstrap.py`: resume hook similar ao GABA

### Fix 5: Inter-Hemispheric Integration (commit `03cf97e`)
- `src/f51_darwin/inter_hemispheric.py`: `CorpusCallosum.forward()` agora retorna `(integrated, integrated, stats)`
- O tensor `integrated` (antes computado e descartado) agora flui para `InterHemisphericSystem`
- Tupla de 3 valores preservada, contrato com caller mantido

### Fix 6: Inter-Hemispheric causal mask + non-redundant integration (commit `f5a50c8`)
- **Achado durante o desenho do fix, mais grave que os anteriores**: `model.py`
  chamava `self.inter_hemispheric(x)` sem mascara, logo apos o embedding e
  antes do stack causal — nem a self-attention de `LeftHemisphere`/
  `RightHemisphere` nem a cross-attention de `CorpusCallosum` aplicavam
  causalidade, ou seja, a posicao *i* podia enxergar embeddings de posicoes
  futuras (label leakage) enquanto `inter_hemispheric_enabled: true`.
- `src/f51_darwin/inter_hemispheric.py`: `InterHemisphericSystem.forward()` agora
  constroi uma mascara causal `[seq_len, seq_len]` quando nao fornecida, e
  repassa para `LeftHemisphere`/`RightHemisphere`/`CorpusCallosum`.
- `CorpusCallosum.forward()` deixou de devolver `(integrated, integrated, stats)`
  (o mesmo tensor duas vezes para `output_proj`) — agora devolve
  `left_enhanced + integrated, right_enhanced + integrated, stats`.
- Verificado (CPU, `map_location=cpu`): forward+backward sem erro de shape,
  `callosum.left_to_right.out_proj.weight` recebe gradiente.

### Fix 7: TTM proj_key treinavel + reconstrucao para proj_value (commit `f5a50c8`)
- Confirmado (recon 2026-07-23): `proj_key`/`proj_value` nunca recebiam
  gradiente — todo call site em `heartbeat.py` estava sob `torch.no_grad()`.
  Cache vetorial com projecoes aleatorias, nao Titans (`ONLINE_LEARNING_RESEARCH.md`).
- `src/f51_darwin/heartbeat.py`: `TestTimeMemory.retrieve()` nao envolve mais o
  metodo inteiro em `no_grad()` — `proj_key` recebe gradiente real da LM
  loss quando uma memoria e recuperada e usada no residual TTM.
- `proj_value` nao tem caminho natural de gradiente (valores vem de um write
  nao-diferenciavel). Recebe perda de reconstrucao leve no mesmo forward
  (`darwin_x_core/model.py`), peso novo `ttm_associative_weight=0.02`
  (`darwin_x_core/config.py`, listado como operacional em `checkpoint_root.py`
  — resume seguro).
- Verificado isolado (CPU): `proj_key.weight` grad norm 53.35 apos
  `retrieve().sum().backward()`; `proj_value.weight` grad norm 0.033 no
  forward completo. Nao e Titans completo (sem inner-loop optimizer) —
  redesenho fica para sessao dedicada.

### Fix 8: JEPA VICReg bilateral + retuning (commit `f5a50c8`)
- Confirmado: VICReg so penalizava colapso do lado do predictor; peso
  `jepa_weight=0.5` era 3.3x o proximo auxiliar; LR do predictor 10x sem
  documentacao (comentario obsoleto no codigo dizia "3x").
- `src/f51_darwin/jepa_v2.py`: `JEPAHeadV2.forward()` nao detacha mais `target`
  internamente — o detach agora acontece em `_jepa_loss()` (`model.py`),
  depois de computar variancia/covariancia sobre o target ao vivo, dando
  pressao anti-colapso ao backbone tambem (antes so o predictor tinha).
- `src/configs/darwin_x_100m.yaml`: `jepa_weight` 0.5 → 0.2.
- `src/f51_darwin/organism/bootstrap.py`: multiplicador de LR do JEPA 10x → 3x
  (`JEPA_LR_MULTIPLIER`). No resume, o `lr` do grupo JEPA e re-fixado
  explicitamente apos `load_state_dict()`, que senao restauraria o
  multiplicador antigo salvo no checkpoint.

### Fix 9: GABA telemetria no barramento causal (commit `f5a50c8`)
- Confirmado: `GABAInterventionAdapter.observe()` lia
  `context["gaba_observations"]`, chave que `training_context()` nunca
  escrevia — telemetria do GABA nunca chegava ao ledger causal.
- `src/f51_darwin/organism/training.py`: apos `commit_gaba_observations`,
  telemetria por camada (`mean_inhibition`/`max_inhibition`/`inhibited_ratio`)
  e guardada com lag de um passo (mesmo padrao do `SpiderRAM`, ja que o
  adapter dispara em `PRE_LOSS`, antes do forward existir).
- `causal_adapters.py::training_context()` ganhou parametro
  `gaba_observations`. Zero mudanca de comportamento de treino —
  `propose()` do adapter ja era (e continua) um no-op, so observabilidade.

### SpiderRAM (commit `8077d72`)
- `src/f51_darwin/spider_sense.py`: nova classe `SpiderRAM` — memoria CPU persistente
- `src/f51_darwin/organism/causal_adapters.py`: `SpiderCalibrationAdapter` v1→v2 le do SpiderRAM
- `src/f51_darwin/organism/training.py`: lazy-init, update pos-forward, injecao no contexto
- Janela de 8 steps com smoothing exponencial (alpha=0.3)
- Corrige `spider.danger` que sempre era 0.0 (adaptador disparava antes do forward)

### Organ Identity SHA-256 (commit `2594d2f`)
- `src/f51_darwin/organism/organ_identity.py`: identidade content-addressable para 17 orgaos
- Schema: `organ:<name>:v1:<sha256>` cobrindo pesos + config + runtime state
- `src/f51_darwin/organism/checkpoint_mixin.py`: `organ_identity_report` embedado em todo save

### Elite Agents + Skill (commits locais em `.claude/`)
- 4 novos agentes: `deep-dissector`, `adversarial-verifier`, `synthesis-architect`, `completeness-critic`
- 1 nova skill: `elite-dissection` — pipeline sequencial de 4 fases para problemas complexos
- `.claude/` e gitignored (local-only, nao entra no repositorio)

### detached_losses Expandido (commit `406957b`)
- `src/f51_darwin/organism/causal_adapters.py`: extrai `mtp`, `jepa`, `ghost`, `spider` alem de `lm`, `aux`
- `src/f51_darwin/organism/training.py`: `effective_losses` enriquecido com valores pos-weight

---

## 3. Diagnostico de orgaos (2026-07-23, cycle 22)

| Orgao | Status | Contribui para loss? | Notas |
|---|---|---|---|
| Core (backbone) | 🟢 FUNCIONAL | Sim (lm) | — |
| MoE Router | 🟢 FUNCIONAL | Sim (aux = load-balance) | 11/384 experts mortos (2.9%), anti-collapse jitter ativo |
| Ghost | 🟢 FUNCIONAL | Sim (termo `ghost`) | EMA identica em todas as camadas (broadcast global) |
| JEPA | 🟢 FUNCIONAL | Sim (termo `jepa`) | Pesos mudam 8-36%, tracking buffers estavam cegos (FIX 2) |
| MTP | 🟢 FUNCIONAL | Sim (termo `mtp`) | — |
| Spider | 🟢 FUNCIONAL | Sim (termo `spider`) | `total_checks=0` era bug cosmetica, SpiderRAM corrige |
| TTM | 🟡 REANIMADO | Nao (gate estava 0) | FIX 4: gate init 0.01, hook de resume |
| GABA | 🟡 REANIMADO | Nao (gate estava 0) | FIX 3: gate init 0.02, hook de resume |
| Inter-Hemispheric | 🟡 REANIMADO | Nao (integration dead code) | FIX 5: `integrated` reconectado ao output |
| DAE | 🟢 FUNCIONAL | Nao (modifica gradientes) | `shadow_mode: false`, ativo |
| Heartbeat | 🟢 FUNCIONAL | Nao (no_grad) | 10.481 batidas, dopamine 0.215 |
| Curiosity | 🔴 MORTO | Nunca implementado | `curiosity_enabled` nao existe no DarwinXConfig |
| Decision Engine | 🔴 FANTASMA | Computa e descarta | `_last_decision_confidence` nunca lido |
| Nitro | 🟢 FUNCIONAL | Nao | GPU expert capacity 4 |
| Unified Mesh | 🟢 FUNCIONAL | Nao | — |
| Sleep | 🟢 FUNCIONAL | Nao | — |
| Spider Calibration | 🟢 FUNCIONAL | Sim (termo `spider`) | Brier loss, SpiderRAM tracking |

---

## 4. Arquitetura do modelo

| Parametro | Valor |
|---|---|
| `model_name` | F51-Darwin-X-100M |
| `d_model` | 512 |
| `n_layers` | 12 |
| `n_heads` | 8 |
| `n_kv_heads` | 2 (GQA) |
| `ssd_attention_ratio` | 3:1 |
| `fine_experts` | 32 |
| `shared_experts` | 2 |
| `experts_per_token` | 8 |
| `vocab_size` | 58.162 |
| `context_length` (treino) | 5.120 |
| `dropout` | 0.0 |
| `gradient_checkpointing` | true |

---

## 5. Otimizacao e Schedule

| Parametro | Valor |
|---|---|
| Optimizer | AdamW (fused) |
| Learning rate inicial | 3e-4 (backbone), 3e-3 (JEPA ×10) |
| Warmup steps | 50 |
| **Cosine decay** | **ATIVO**: 100k steps → 3e-5 floor |
| Block size | 4.096 |
| Batch size | 1 |
| Corpus | feast_v2 (18.55B tokens) |
| Tokens vistos | 32.4M de 65B (0.05%) |
| Save interval | 100 steps |
| Steps por ciclo | 500 |
| CPU offload | AdamW states em DDR5 (~1.2 GB) |
| GPU split | DUAL_GPU_SPLIT=5 (layers 0-4 GPU0, 5-11 GPU1) |

---

## 6. Commits recentes

| Commit | Descricao |
|---|---|
| `03cf97e` | fix(organs): reanimate 4 dormant organs |
| `f79b4c5` | feat(lr): cosine decay phase after warmup |
| `cf96e33` | feat(checkpoint): per-organ SHA-256 identities in every save |
| `8077d72` | feat(spider): SpiderRAM persistent CPU memory |
| `406957b` | fix(causal): expand detached_losses for all loss fields |
| `2594d2f` | feat(identity): per-organ SHA-256 content identities |

---

## 7. Superficie suportada

### Scripts
- `src/scripts/start_100m_auto.ps1` — Resume 100M com ciclos de 500 steps (ATUALIZADO: +lr-decay-steps, +lr-final-ratio)
- `src/scripts/start_100m_65b.ps1` — Treino longo 100M com meta de 65B tokens
- `src/scripts/start_overnight_16b.ps1` — Gate overnight 1.6B
- `src/scripts/darwin_organism.py` — CLI adapter (ATUALIZADO: +--lr-decay-steps, +--lr-final-ratio)

### Novos modulos
- `src/f51_darwin/organism/organ_identity.py` — SHA-256 por orgao (17 orgaos)
- `src/f51_darwin/spider_sense.py` — SpiderRAM (classe追加)

---

## 8. Procedimento de resume

### Preflight (obrigatorio antes de qualquer resume)

```powershell
git status --short
Get-CimInstance Win32_Process | Where-Object { $_.Name -like "*python*" }
nvidia-smi
.\.venv_nitro\Scripts\python.exe -m scripts.darwin_inventory
```

### Resume (treino)

```powershell
powershell -ExecutionPolicy Bypass -File src\\scripts\\start_100m_auto.ps1
```

O script agora passa `--lr-decay-steps 100000 --lr-final-ratio 0.1` alem dos
parametros anteriores. O cosine decay e deterministico dado `total_steps` —
sobrevive a resumes sem estado adicional.

---

## 9. Changelog deste documento

| Data | Alteracao |
|---|---|
| 2026-07-23 | Atualizacao completa: cycle 23 step 7912. 5 fixes aplicados (LR cosine, JEPA tracking, GABA/TTM gates, Inter-Hemi). Novos modulos: organ_identity, SpiderRAM. Elite agents + elite-dissection skill. Diagnostico completo dos 17 orgaos. |
| 2026-07-22 | Estado do cycle 13, step 3000+. 17 orgaos, shadow mode, dual GPU. |
| 2026-07-22 (anterior) | Rollback para cycle 6, fix Ghost autograd. |

---

**Autoridade:** Este documento e a unica autoridade de status do F51 Darwin-X.
`governance/archive/agent-bus/`, `governance/archive/legacy-state/f51/`, `governance/docs/_historico/` e
`governance/docs/superpowers/` nao sao autoridade atual.
