# CHECKLIST - F51 Darwin Fixes

**Data:** 2026-07-12
**Status:** P0+P1 completos, trabalhando P1+A1+P2

---

## ✅ COMPLETO

### P0 — Críticos Imediatos
- [x] C1: Bug #3 — Coerção tipos em Config
- [x] C2: Adapter state STALE removido

### P1 — Alta Prioridade
- [x] A2: Ghost stream — Migrar fontes (wikipedia/math)
- [x] A3: Ghost stream — Agendamento Windows Task Scheduler
- [x] A4: Adapter — Integrar checkpoint v6

---

## 🔄 EM PROGRESSO

### P1 — Benchmark (A1)
- [ ] A1.1: Fix bug #1 — checkpointing.py usa F51DarwinModel (legado) — AGENTE ab01c3b8
- [ ] A1.2: Fix bug #2 — benchmark.py usa "wikitext" (inválido) — AGENTE ab01c3b8
- [ ] A1.3: Rodar benchmark_darwin.py sem workaround
- [ ] A1.4: Confirmar PPL WikiText-2 salvo em runs/benchmarks/

### P2 — Média Prioridade
- [ ] B2: Decisão arquitetural (live_generate ↔ InferenceLearner) — AGENTE ad26dc8a
- [ ] B3: Teste e2e online learning
- [ ] B4: Gate de promoção (PPL < 1000)
- [ ] B5: Benchmark PT held-out

### P3 — Baixa Prioridade
- [ ] C2: Atualizar STATUS_ATUAL.md — AGENTE aac718d8
- [ ] C3: check_ckpt.py fix (hardcoded Linux path) — AGENTE a19a084a
- [ ] C4: Linting state rejection warning — AGENTE af8ee339
- [ ] C1: live_generate cleanup (remover resíduo técnico)

---

## ⏳ PENDENTE

### P2 — Média Prioridade
- [ ] B1: Adapter HF para lm-eval (HellaSwag, PIQA)
- [ ] B2: Decisão arquitetural (live_generate ↔ InferenceLearner)
- [ ] B3: Teste e2e online learning
- [ ] B4: Gate de promoção (PPL < 1000)
- [ ] B5: Benchmark PT held-out

### P3 — Baixa Prioridade
- [ ] C1: live_generate cleanup (remover resíduo técnico)
- [ ] C2: Atualizar STATUS_ATUAL.md
- [ ] C3: check_ckpt.py fix (hardcoded Linux path)
- [ ] C4: Linting state rejection warning

---

## 📊 MÉTRICAS

| Métrica | Atual | Meta |
|---|---|---|
| Segredos hardcoded | 1 (HF token) | 0 |
| Bugs bloqueantes | 2 (A1.1, A1.2) | 0 |
| Fontes ghost_stream | 4/4 | 4/4 ✅ |
| Agendamento ativo | ✅ | ✅ |
| PPL WikiText-2 | 94.657 | < 1000 |
| PPL PT held-out | ? | medir |

---

## 🎯 PRÓXIMO

Trabalhando A1 (Benchmark bugs) + paralelo P2 rápido

