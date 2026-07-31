# Pesquisa 4 Eixos — Construindo o Organismo Vivo

Data: 2026-07-13 (atualizado 2026-07-14 com pesquisa aprofundada)

---

## ATUALIZAÇÃO 14 Jul: Bug crítico encontrado + Papers fundamentais

### Bug no benchmark de 2000 steps

O replay buffer era recriado vazio a cada fase. C1_Replay repetia LITERATURA
enquanto treinava literatura — não matemática. Isso explica FR pior que C0.
O experimento NÃO testou replay de domínio antigo. Correção pendente.

### Papers que definem a arquitetura correta

| Paper | Conferência | Ideia central | Aplicação Darwin |
|---|---|---|---|
| **DES-MoE** | EMNLP 2025 | Dynamic Expert Specialization: correlaciona expert-domínio, atualiza só relacionados, 3 fases (descoberta/estabilização/consolidação), 89% redução de forgetting | Principal referência |
| **Theory on MoE in CL** | ICLR 2025 Spotlight | Experts podem especializar e reduzir interferência. Router DEVE congelar após estabilizar. Adicionar experts continuamente não garante melhora | Alerta: congelar router |
| **Gradient Projection Memory** | ICLR 2021 | Projeta gradiente novo ortogonal ao subespaço protegido: g_safe = g_new - U_old @ (U_old.T @ g_new) | Ghost Predator mantém sketch U_math |
| **O-LoRA** | 2024 | Cada tarefa aprende atualização low-rank em subespaço ortogonal aos anteriores, sem armazenar dados antigos | Expert novo aprende residual ortogonal |
| **CP-MoE** | Preprint 2026 | Expert transitório aprende primeiro, similaridade de representação decide consolidação, regularização protege histórico | Arquitetura alvo |
| **CoPE** | ICML 2026 | Probe-guided expansion: treina probe, detecta lacuna de capacidade, expande experts | Eixo 4 com prova publicada |

### Arquitetura recomendada

```
dados novos → Ghost held-out probe → medir gradiente/compatibilidade
  ├─ grad positivo → atualização compartilhada
  ├─ grad neutro   → expert transitório
  └─ grad negativo → projeção + congelar experts antigos
                        ↓
                 torneio held-out
                        ↓
           consolidar | manter separado | rejeitar
```

### Estados dos experts

- **PLASTIC**: novo/transitório, aprende livremente
- **STABLE**: habilidade comprovada, atualizações protegidas
- **FROZEN**: domínio crítico consolidado, nunca alterado
- **SHARED**: recebe update só quando beneficia múltiplos domínios

### Próximo benchmark

C0: sequencial simples | C1: replay matemático persistente | C2: Gradient Projection
C3: freeze+grow experts | F51: expert transitório + projeção + torneio held-out

Mínimo: 5-10 seeds, 500/2000/10000 steps, held-out fixo, BWT, FWT, grad cosine.

---

## EIXO 1: DNA ESTRUTURAL — Hereditariedade através de checkpoints

**Problema:** O organismo muda durante a vida (experts nascem, crescem, morrem)
mas ao reiniciar perde tudo. É evolução sem DNA.

### Papers e Ideias

| Fonte | Ideia | Aplicável |
|---|---|---|
| **Git Re-Basin** (Ainsworth, NeurIPS 2023) | Alinhamento de pesos entre modelos com arquiteturas diferentes via permutação | Checkpoint antigo pode reconhecer topologia nova |
| **Model Soups** (Wortsman, ICML 2022) | Média de pesos de múltiplos checkpoints | Múltiplas versões do organismo coexistem |
| **LoRA** (Hu, ICLR 2022) | Adaptadores low-rank que não alteram base | Adaptador por domínio sem mexer no backbone |
| **Tree of Models** (DeepMind, 2024) | Árvore de fine-tunes com herança | Cada expert herda do pai e diverge |
| **(VAZIO)** | Ninguém resolveu serialização de topologia dinâmica | **O Darwin pode ser o primeiro** |

### Solução: Topology Manifest v7

O checkpoint salva: base config + diff topológico (experts adicionados/removidos/expandidos)
+ estado do organismo (hormônios, buffers, replay, ghost predator, lineage).

---

## EIXO 2: AUTO-MELHORIA — O organismo testa as próprias hipóteses

**Problema:** O organismo aplica mutações cegamente. Não sabe se nascer um
expert foi bom ou ruim 100 steps depois.

### Papers e Ideias

| Fonte | Ideia |
|---|---|
| **Self-Rewarding LLMs** (Meta, 2024) | Modelo gera dado e se auto-avalia |
| **SPIN** (Chen, 2024) | Self-Play Fine-Tuning: modelo compete contra versão anterior |
| **Constitutional AI** (Anthropic, 2023) | Princípios governam auto-melhoria |
| **Self-Taught Optimizer** (DeepMind, 2024) | Modelo aprende a se otimizar |
| **Evolutionary Model Merge** (Sakana AI, 2024) | Algoritmo genético funde modelos |

### Solução: Internal Tournament

Fork do organismo → mutação → ambos treinam K steps → comparar loss + ghost score.
Melhor sobrevive. Perdedor arquivado como linhagem extinta (não deletado).

---

## EIXO 3: CONHECIMENTO ILIMITADO — Mundo externo como memória

**Problema:** 59B tokens são finitos. O organismo precisa buscar MAIS conhecimento
sozinho e decidir o que consolidar.

### Papers e Ideias

| Fonte | Ideia |
|---|---|
| **RAG** (Lewis, NeurIPS 2020) | Retrieval + geração |
| **REALM** (Guu, ICML 2020) | Retrieval durante pretraining |
| **Atlas** (Izacard, NeurIPS 2022) | Modelo pequeno + índice gigante |
| **Self-RAG** (Asai, 2024) | Modelo decide SE buscar e AVALIA resultado |
| **Corrective RAG** (Yan, 2024) | Avalia, corrige, re-busca |
| **Memorizing Transformers** (Wu, ICLR 2022) | Memória externa kNN |

### Solução: External Cortex

Web/API + Datasets + Ghost → Quarantine + Dedup → Retrieval Index → Consolidation Gate.
O organismo decide o que vira peso neural e o que permanece como fonte externa.

---

## EIXO 4: EVOLUÇÃO ARQUITETURAL — Crescer por necessidade, não por limiar

**Problema:** Neurogênese dispara porque NE > 0.05 (threshold baixo arbitrário).
Deveria disparar porque a arquitetura atual FALHOU em aprender.

### Papers e Ideias

| Fonte | Ideia |
|---|---|
| **GradMax** (Evci, ICML 2022) | Adiciona neurônio onde gradiente é máximo |
| **Firefly** (Wu, 2025) | Crescimento online de rede durante treino |
| **Liquid Neural Networks** (Hasani, 2023) | Dinâmica contínua adaptativa |
| **SETA** (2026) | Split-on-share: expert se divide quando sobrecarregado |
| **Lifelong-MoE** (Chen, 2023) | Adiciona experts por domínio, congela antigos |

### Solução: Capability-Gap Growth

Se loss estagnado 200 steps E gradientes fortes → capacidade insuficiente → criar expert.
Se loss estagnado E gradientes fracos → não é capacidade, é dados/LR → não criar.
Se loss caindo → arquitetura suficiente → não mexer.

---

## Ordem de Implementação

1. **EIXO 1** (crítico): Topology Manifest no checkpoint v7 — sem DNA, nada herda
2. **EIXO 2**: Internal Tournament — testar antes de aplicar
3. **EIXO 3**: External Cortex — mundo infinito de conhecimento
4. **EIXO 4**: Capability-Gap Growth — crescer com inteligência
