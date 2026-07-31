# ROADMAP - Correções Críticas Davi F51

**Data:** 2026-07-12
**Base:** Auditoria paralela de 4 subagentes
**Status:** 0/4 problemas resolvidos

---

## 🎯 VISÃO GERAL

| # | Problema | Status | Esforço | Impacto |
|---|---|---|---|---|
| 1 | Memória episódica persistente | 🟡 Funcional mas incompleto | 3d | Alto |
| 2 | Pesquisa automática (ghost_stream) | 🟡 2/4 fontes funcionam | 2d | Alto |
| 3 | UI funcional (serve_davi) | 🟢 Íntegro | 0d | Resolvido |
| 4 | Benchmark congelado | 🔴 PPL=94.657 | 4d | Crítico |

**Legenda:** 🟢 Resolvido | 🟡 Parcial | 🔴 Crítico | ⚫ Bloqueado

---

## 🚨 CRÍTICOS IMEDIATOS (P0)

### C1. Bug #3 — Coerção de Tipos em Config
**Risco:** Checkpoints não carregam, benchmark quebra
**Arquivos:** `f51_darwin/config.py:84-89`, `f51_darwin/darwin_x.py:120-125`

**Problema:** `DarwinConfig.from_mapping` repassa strings sem coerção:
```python
# checkpoint salva: {'vocab_size': '58162', 'dropout': '0.0'}
# from_mapping repassa: vocab_size='58162' (string)
# dataclass falha: vocab_size <= 0 (comparação string com int)
```

**Solução:** Adicionar coerção de tipos em `from_mapping`:
```python
@classmethod
def from_mapping(cls, mapping: dict[str, Any]) -> "DarwinConfig":
    coerced = {}
    for key, field in cls.__dataclass_fields__.items():
        value = mapping.get(key)
        if value is None:
            continue
        # Coerce based on field type
        if field.type == int or (hasattr(field.type, "__origin__") and field.type.__origin__ is int):
            coerced[key] = int(value)
        elif field.type == float:
            coerced[key] = float(value)
        elif field.type == bool:
            coerced[key] = str(value).lower() in ("true", "1", "yes")
        # ... etc
    return cls(**coerced)
```

- [ ] Implementar coerção em `DarwinConfig.from_mapping`
- [ ] Implementar coerção em `DarwinXConfig.from_mapping`
- [ ] Testar carregamento de organism_cycle_231.pt
- [ ] Adicionar teste unitário

**Esforço:** 2h | **Bloqueia:** Benchmark, promoção de checkpoints

---

### C2. Adapter State STALE — Remover Arquivo Órfão
**Risco:** Adapter inválido carregado silenciosamente
**Arquivo:** `runs/online_learning/adapter_state.pt`

**Problema:** Arquivo salvo antes de `base_checkpoint_id`/`tokenizer_id` serem obrigatórios. `_load_state` vai rejeitar com `base_checkpoint_mismatch`.

**Solução:**
```bash
# Backup e remoção
mv runs/online_learning/adapter_state.pt runs/online_learning/adapter_state_pt_stale_backup.pt
```

- [ ] Backup do arquivo órfão
- [ ] Remover arquivo ativo
- [ ] Próximo `serve` começará com adapter zerado (OK)

**Esforço:** 5min | **Bloqueia:** Servidor Davi

---

## 🔴 PRIORIDADE ALTA (P1)

### A1. Benchmark — Fix Bugs #1 e #2 (skipped - doing A2 first)
**Risco:** PPL=94.657 sem validação, modelo sem métrica externa

**Bug #1:** `f51_darwin/checkpointing.py:68` usa `F51DarwinModel` (legado)
**Solução:** Criar dispatcher ou separar `load_darwin_x_from_checkpoint`

**Bug #2:** `f51_darwin/benchmark.py:65` usa `"wikitext"` (inválido em datasets 5.x)
**Solução:** Usar `"Salesforce/wikitext"`

- [ ] Fix bug #1 (checkpointing.py)
- [ ] Fix bug #2 (benchmark.py)
- [ ] Rodar `benchmark_darwin.py --bench wikitext2` sem workaround
- [ ] Confirmar PPL≈94.657 no checkpoint 231
- [ ] Salvar artefato em runs/benchmarks/

**Esforço:** 4h | **Depende:** C2 (bug #3)

---

### A2. Ghost Stream — Migrar Fontes Obsoletas
**Risco:** 2/4 fontes mortas, ingestão incompleta

**Fontes migrar:**
- `wikipedia` (dataset-script) → `wikimedia/wikipedia` (parquet)
- `math_dataset` (dataset-script) → `lighteval/MATH` ou `meta-math/MetaMathQA`

- [ ] Atualizar `stream_wikipedia` para `wikimedia/wikipedia`
- [ ] Atualizar configs: `20231101.pt` → `20231101.pt`, `20231101.en`
- [ ] Atualizar `stream_math` para `lighteval/MATH`
- [ ] Remover `except:pass` em math, adicionar log
- [ ] Testar cada fonte isoladamente
- [ ] Testar todas 4 fontes juntas

**Esforço:** 3h | **Depende:** C1

---

### A3. Ghost Stream — Estabelecer Agendamento
**Risco:** Sistema "automático" nunca roda

**Opções:**
1. Windows Task Scheduler (recomendado)
2. Serviço background do organismo
3. Cron-like em Python

- [ ] Criar Task Scheduler para `ghost_stream.py --loop --interval 300`
- [ ] OU garantir `darwin_organism.py run247` roda em background
- [ ] Verificar logs periódicos
- [ ] Monitorar data/generated/candidates/

**Esforço:** 2h | **Depende:** A2

---

### A4. Adapter — Integrar ao Organismo (Checkpoint v6)
**Risco:** Adapter never persiste em checkpoints, perde a cada restart

**Solução:** Garantir que `darwin_organism.py` salve checkpoint v6 com `online_learning_state`

- [ ] Verificar se organismo popula `base_checkpoint_id` e `tokenizer_id`
- [ ] Rodar 1 ciclo completo: `darwin_organism.py cycle --resume organism_cycle_231.pt`
- [ ] Confirmar checkpoint v6 salvo com `online_learning_state`
- [ ] Confirmar `adapter_state.pt` não é mais necessário

**Esforço:** 1h | **Depende:** C2, C3

---

## 🟡 PRIORIDADE MÉDIA (P2)

### B1. Benchmark — Criar Adapter HuggingFace para lm-eval
**Risco:** HellaSwag, PIQA, ARC permanently blocked

**Solução:** Classe wrapper `DarwinXHuggingFace` implementando interface lm-eval

- [ ] Criar `f51_darwin/huggingface_adapter.py`
- [ ] Implementar `__call__` retornando log-likelihoods
- [ ] Integrar com `lm_eval.models.huggingface.HFLM`
- [ ] Testar HellaSwag
- [ ] Testar PIQA

**Esforço:** 8h | **Depende:** A1

---

### B2. Memória Episódica — Decisão Arquitetural
**Risco:** Especificação vs código divergentes

**Opções:**
1. Integrar InferenceLearner em `live_generate()` (custoso)
2. Remover promessa de docs (documentar gap)
3. Manter separado: run247 = treino, serve = aprendizado

- [ ] Decidir caminho
- [ ] Atualizar PLANO_INFERENCIA_TEMPO_REAL.md
- [ ] Atualizar LEIS_DE_DAVI.md (Lei 7)
- [ ] Documentar em AGENTS.md

**Esforço:** 2h (reunião decisão) + 2h (documentação)

---

### B3. Memória Episódica — Teste End-to-End
**Risco:** Integração nunca testada em produção

**Cenário:** serve → interact → approve → restart → adapter persiste

- [ ] Escrever teste `test_online_learning_e2e.py`
- [ ] Cobrir: bootstrap → interact → consolidate → checkpoint → resume
- [ ] Verificar adapter bit-for-bit após restart
- [ ] Verificar gates (plasticity, forgetting)

**Esforço:** 4h | **Depende:** A4, B2

---

### B4. Benchmark — Gate de Promoção
**Risco:** Checkpoints promovidos sem métrica externa

**Regra:** Só promover "modelo melhor" se:
- PPL WikiText-2 < 1.000 (deixar de ser pior que random)
- Replay regression <= 2%
- Artefato salvo com SHA-256, commit Git, tokens contados

- [ ] Implementar verificação em `darwin_organism.py` (promoção)
- [ ] Adicionar flag `--promote` (require PPL)
- [ ] Documentar critério em BENCHMARK_REFERENCE.md
- [ ] Bloquear promoção automática atual

**Esforço:** 3h | **Depende:** A1

---

### B5. Benchmark — Corpus PT Held-Out
**Risco:** Generalização EN vs PT não medida

**Solução:** Medir PPL em corpus PT (deve ser muito menor que WikiText-2)

- [ ] Criar corpus PT held-out (fora do treino)
- [ ] Implementar `load_pt_corpus_test` em benchmark.py
- [ ] Rodar benchmark PT
- [ ] Comparar delta PT vs EN

**Esforço:** 4h | **Depende:** A1

---

## 🟢 PRIORIDADE BAIXA (P3)

### C1. live_generate — Remover Resíduo Técnico
**Risco:** Confusão arquitetural

**Problema:** `live_generate()` ainda aplica TestTimeMemory/QuietStarThinker, mas docs dizem "não Quiet-STaR"

**Solução:** Limpar código morto ou documentar保持

- [ ] Remover TestTimeMemory/QuietStarThinker de live_generate
- [ ] OU documentar: "resíduo legado, não usado"
- [ ] Atualizar docs para refletir realidade

**Esforço:** 2h | **Depende:** B2

---

### C2. Documentação — Atualizar STATUS_ATUAL.md
**Risco:** Documento weaponized-stale (cycle 91 vs real 231)

- [ ] Atualizar ciclo (231)
- [ ] Atualizar step (115.560)
- [ ] Atualizar PID (13504)
- [ ] Atualizar GPUs (usage)
- [ ] Estabelecer rotina de atualização

**Esforço:** 30min

---

### C3. check_ckpt.py — Fix Hardcoded Path
**Risco:** Script não roda em Windows

**Problema:** `/workspace/F51-Darwin-SSD/...` hardcoded

- [ ] Usar `sys.argv[1]` ou ROOT
- [ ] Testar em Windows

**Esforço:** 15min

---

### C4. Linting — Adicionar Warning State Rejection
**Risco:** Adapter rejeitado silenciosamente

**Solução:** Log/warning quando `state_rejection_reason != ""`

- [ ] Adicionar log em `serve_davi.py`
- [ ] Adicionar log em `darwin_organism.py`

**Esforço:** 30min

---

## 📋 LINHA DO TEMPO

```
P0 (Críticos Imediatos)
│
├─ C1: Bug #3 coerção tipos em Config
└─ C2: Adapter state stale remover

P1 (Alta Prioridade)
│
├─ A1: Benchmark bugs #1+#2
├─ A2: Ghost stream migrar fontes
├─ A3: Ghost stream agendamento
└─ A4: Adapter checkpoint v6

P2 (Média Prioridade)
│
├─ B1: Adapter HF lm-eval
├─ B2: Decisão arquitetural
├─ B3: Teste e2e
├─ B4: Gate promoção
└─ B5: Benchmark PT held-out

P3 (Baixa Prioridade)
│
├─ C1: live_generate cleanup
├─ C2: Atualizar STATUS_ATUAL
├─ C3: check_ckpt.py fix
└─ C4: Linting state rejection
```

---

## 🎯 CRITÉRIO DE SUCESSO

**P0 (hoje):**
- [ ] Token HF removido do código
- [ ] Bug #3 corrigido, checkpoints carregam
- [ ] Adapter state stale removido

**P1 (semana que vem):**
- [ ] Benchmark_darwin.py roda sem workaround
- [ ] 4/4 fontes ghost_stream funcionam
- [ ] Ghost_stream roda automaticamente (Task Scheduler)
- [ ] Checkpoint v6 com adapter_state embutido

**P2 (semana seguinte):**
- [ ] HellaSwag/PIQA rodando via lm-eval
- [ ] Documentação alinhada com código
- [ ] Teste e2e passando
- [ ] Gate de PPL ativo
- [ ] PPL PT medido e comparado

**P3 (futuro):**
- [ ] Código limpo, sem resíduos técnicos
- [ ] Documentação atualizada
- [ ] Scripts cross-platform

---

## 📊 MÉTRICAS DE SAÚDE

| Métrica | Atual | Meta P0 | Meta P1 | Meta P2 |
|---|---|---|---|---|
| Segredos hardcoded | 1 | 0 | 0 | 0 |
| Bugs bloqueantes | 3 | 2 | 0 | 0 |
| Fontes ghost_stream | 2/4 | 2/4 | 4/4 | 4/4 |
| Agendamento ativo | ❌ | ❌ | ✅ | ✅ |
| PPL WikiText-2 | 94.657 | ? | ✓ medido | < 1000 |
| PPL PT held-out | ? | ? | ? | ✓ medido |
| Adapter em checkpoint | ❌ | ❌ | ✅ v6 | ✅ v6 |
| lm-eval funcionando | ❌ | ❌ | ❌ | ✅ |

---

## 🔄 CICLO DE VIDA

1. **HOJE:** P0 — Segurança + carregamento
2. **SEMANA 1:** P1 — Funcionalidade crítica
3. **SEMANA 2:** P2 — Métricas e gates
4. **FUTURO:** P3 — Limpeza e documentação

**Próxima revisão:** Após P1 completo
**Dono:** Marco Barreto (@marcobarreto007)
**Auditoria:** 4 subagentes paralelos (2026-07-12)

---

*Soli Deo Gloria*
