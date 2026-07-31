# ROADMAP PÓS-AUDITORIA — F51-Darwin-SSD
# 2026-07-12 | Commit base: 997be94

## DIAGNÓSTICO CONSOLIDADO

O organismo está VIVO (ciclo 239, step ~119k, tok/s=113, dual GPU estável).
O runtime está SEGURO (1 processo CUDA, mutex funcional, sem ghost stream).
O corpus tokens_feast.bin está SOB SUSPEITA (repetição sintética visível, templates).
A qualidade do modelo é DESCONHECIDA (sem benchmark comparável entre checkpoints).
Continuar treinando AGORA vai apagar checkpoints intermediários por rotação.

## DECISÃO ESTRATÉGICA

PRESERVAR → PARAR CONTROLADAMENTE → MEDIR → DECIDIR

NÃO faremos:
  - Continuar treinando porque o loss caiu
  - Voltar ao v3 porque uma geração pareceu menos ruim
  - Mudar corpus + código + runtime ao mesmo tempo

---

## FASE 0 — CONTENÇÃO IMEDIATA (1 ação, ~2 minutos)

### 0.1 Congelar evidência

```
AÇÃO: Esperar o próximo checkpoint atômico terminar (fim do ciclo atual).
      Desligar o servidor com KeyboardInterrupt (Ctrl+C).
      Copiar para diretório seguro (ex: checkpoints/audit_frozen/):

        organism_cycle_234.pt
        organism_cycle_235.pt
        organism_cycle_236.pt
        organism_cycle_237.pt
        organism_cycle_238.pt
        organism_cycle_239.pt  (último produzido)
        darwin_trained_v3.pt
        organism_latest.json

      + metadados:
        configs/darwin_x_600m.yaml
        tokenizer/f51_bpe_80k/ (hash recursivo)
        data/tokens_feast.bin   (hash sha256)
        runs/lineage/lineage_report.json
        runs/online_learning/research_memory.json
        runs/online_learning/adapter_state.pt

      + código:
        git bundle create audit_frozen/997be94.bundle HEAD
```

### 0.2 Desabilitar rotação e restart automático

```
AÇÃO: Comentar a linha de rotação em darwin_organism.py:_save_cycle()
      (manter só os 5 mais recentes → MANTER TODOS até o fim da auditoria).

      Verificar Scheduled Tasks do Windows — desabilitar qualquer
      tarefa que possa reiniciar python.exe automaticamente.
```

**ENTREGA:** Todos os checkpoints preservados com hashes. Nenhum dado perdido.

---

## FASE 1 — CORRIGIR O LABORATÓRIO (3 ações, ~1 hora)

**Regra: não alterar pesos, corpus, tokenizer ou arquitetura.**

### 1.1 Corrigir os 2 testes quebrados (P1)

```
ARQUIVO: tests/test_organism_causal_runtime.py:97
PROBLEMA: Mock SimpleNamespace não tem mtp_loss/jepa_loss/ghost_loss
CORREÇÃO: Adicionar os 3 campos ao mock OU usar getattr() com fallback

ARQUIVO: tests/test_serve_davi.py:56
PROBLEMA: innerHTML no novo UI viola assert do teste
CORREÇÃO: Substituir innerHTML por textContent onde possível
          OU atualizar o assert para aceitar innerHTML (já temos CSP header)
```

### 1.2 Corrigir exclusão real treino/inferência (P1)

```
ARQUIVO: scripts/darwin_organism.py / serve_davi.py
PROBLEMA: _pause_event.wait() só é consultado no início do próximo ciclo.
          Durante um ciclo de 500 steps (~5 min), a UI mostra "inferência"
          mas o treino continua rodando.

CORREÇÃO: Adicionar checkpoint de pausa DENTRO do loop de treino:
          if not self._pause_event.is_set(): break  # sai limpo no próximo step
          OU: training_controller injetar um flag que o _train_cycle consulta
```

### 1.3 Criar avaliador determinístico (P1)

```
NOVO ARQUIVO: scripts/eval_checkpoint.py

Contrato:
  - Carrega um checkpoint .pt (CPU, sem GPU)
  - Tokeniza prompts fixos de um JSON
  - Gera com temperatura=0, top_p=1.0 (determinístico)
  - Mede perplexidade em corpus de validação PT limpo (ex: WikiText PT)
  - Mede perplexidade em corpus EN limpo (ex: WikiText EN)
  - Salva resultado JSON: {checkpoint, prompts[], texts[], ppl_pt, ppl_en, ...}

NÃO usar GPU do organismo vivo. Rodar com CUDA_VISIBLE_DEVICES="" ou
em máquina separada.
```

**ENTREGA:** Testes verdes. Mutex real. Avaliador pronto para o torneio.

---

## FASE 2 — TORNEIO DE CHECKPOINTS (1 ação, ~30 min por checkpoint)

### 2.1 Avaliar sequencialmente (NUNCA em paralelo na GPU)

```
CHECKPOINTS A AVALIAR (em ordem):
  1. darwin_trained_v3.pt       (baseline pré-organismo)
  2. organism_cycle_234.pt      (primeiro preservado)
  3. organism_cycle_235.pt
  4. organism_cycle_236.pt
  5. organism_cycle_237.pt
  6. organism_cycle_238.pt
  7. organism_cycle_239.pt      (último preservado)

MÉTRICAS (todas determinísticas, seed fixa):
  - PPL PT (WikiText ou similar limpo)
  - PPL EN
  - Geração com prompts fixos (PT e EN)
  - Diversidade lexical (type/token ratio)
  - Taxa de repetição na geração
  - Coerência (avaliação manual ou heurística)
  - Recuperação de conhecimento (prompts de identidade: "Quem te criou?", etc.)

FORMATO DE SAÍDA:
  eval_results/checkpoint_tournament.json
  [
    {"checkpoint": "v3", "ppl_pt": X, "ppl_en": Y, "gen_pt": "...", ...},
    ...
  ]
```

### 2.2 Tabular e decidir

```
CRITÉRIO DE VENCEDOR (em ordem de prioridade):
  1. Melhor PPL em PT limpo (evidência objetiva)
  2. Geração coerente em PT (evidência qualitativa)
  3. Menor degradação em EN (sem catastrophic forgetting)
  4. Recuperação de identidade

Se v3 ganhar em TODAS as métricas → rollback justificado.
Se ciclo 238+ ganhar → continuar com correções de corpus.
Se inconclusivo → manter ambos, investigar causa raiz no corpus.
```

**ENTREGA:** Tabela comparativa objetiva. Decisão baseada em dados, não em impressão.

---

## FASE 3 — AUDITORIA ROBUSTA DO CORPUS (se necessário)

### 3.1 Medir com precisão

```
MÉTRICAS:
  - Duplicação exata (hash de documentos inteiros)
  - Near-duplicates (simhash ou MinHash)
  - Repetição local consecutiva (janelas deslizantes)
  - Distribuição por origem (se metadados existirem)
  - Proporção sintético/real (detecção de templates)
  - Comprimento médio de documentos
  - Diversidade lexical (type/token ratio global)
  - Tags artificiais ([category], [KNOWLEDGE], etc.)
  - Concentração por domínio
  - Tokens únicos efetivos (após deduplicação)

FERRAMENTAS:
  - scripts/audit_corpus.py (novo, somente leitura)
  - Não modificar tokens_feast.bin
```

### 3.2 Classificar

```
APROVADO:     <5% duplicação, >80% natural, diversidade alta
SUSPEITO:     5-20% duplicação, templates visíveis
INADEQUADO:   >20% duplicação OU domínio único OU maioria sintético
```

---

## FASE 4 — DECISÃO E PRÓXIMOS PASSOS

### Se o torneio mostrar que v3 é melhor:
```
  1. Rollback para v3
  2. Reconstruir corpus (PT limpo, validado)
  3. Retokenizar
  4. Retomar treino do v3 com novo corpus
  5. Reativar organismo com segurança
```

### Se o torneio mostrar que o ciclo atual é melhor:
```
  1. Corrigir corpus (deduplicar, limpar templates)
  2. Retokenizar
  3. Continuar treino com dados limpos
  4. Restaurar rotação de checkpoints
```

### Se inconclusivo:
```
  1. Investigar causa raiz no corpus (Fase 3 completa)
  2. Rodar torneio com mais checkpoints
  3. Considerar treino controlado A/B (com/sem organismo, mesmo corpus)
```

---

## PRIORIDADES

| ID | Fase | Ação | Prioridade | Tempo est. |
|----|------|------|-----------|------------|
| 0.1 | 0 | Congelar evidência | **P0** | 2 min |
| 0.2 | 0 | Desabilitar rotação | **P0** | 1 min |
| 1.1 | 1 | Corrigir 2 testes | P1 | 20 min |
| 1.2 | 1 | Mutex real treino/inferência | P1 | 30 min |
| 1.3 | 1 | Script eval_checkpoint.py | P1 | 40 min |
| 2.1 | 2 | Torneio de checkpoints | P1 | 3-4 h |
| 2.2 | 2 | Tabular e decidir | P1 | 15 min |
| 3.1 | 3 | Auditoria robusta do corpus | P2 | 2-3 h |

---

## REGRAS DO ROADMAP

1. Nenhuma ação destrói evidência.
2. Cada checkpoint é preservado com hash antes de ser comparado.
3. Código e corpus não mudam simultaneamente.
4. Decisões são baseadas em métricas objetivas, não em geração visual.
5. O organismo não é morto — é pausado com segurança.
