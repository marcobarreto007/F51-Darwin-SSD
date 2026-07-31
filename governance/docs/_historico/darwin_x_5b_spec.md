# F51-Darwin-X-5B-Nitro — Especificação Completa

**Data**: 2026-07-15
**Base**: Derivado de F51-Darwin-X-1.6B-Nitro
**Status**: Config criada, pronto para initialização

---

## 1. Visão Geral

O Darwin-X-5B é um escalonamento **conservador** do 1.6B-Nitro que:
- Preserva **TODA** a estrutura orgânica (órgãos, hormônios, sinais)
- Mantém a proporção SSD:Attention (3:1)
- Aumenta capacidade de contexto (4K→8K treino, 32K→128K inferência)
- Top-4 routing (mais capacidade de mistura de experts)

---

## 2. Comparação: 1.6B → 5B

| Parâmetro | 1.6B-Nitro | 5B-Nitro | Δ |
|-----------|------------|----------|---|
| **Estrutura Base** |
| d_model | 1920 | 2560 | +33% |
| n_layers | 16 | 24 | +50% |
| n_heads | 16 | 16 | = |
| n_kv_heads | 4 | 4 | = |
| context_length | 4096 | 8192 | **2×** |
| inference_context | 32768 | 131072 | **4×** |
| vocab_size | 58162 | 58162 | = |
| **MoE** |
| fine_experts | 14 | 20 | +43% |
| shared_experts | 2 | 2 | = |
| experts_per_token | 2 | 4 | **2×** |
| fine_hidden_dim | 896 | 1024 | +14% |
| shared_hidden_dim | 896 | 1280 | +43% |
| nitro_capacity | 5 | 8 | +60% |
| **Parâmetros** |
| embedding | 111M | 149M | +34% |
| ssd_total | 397M | 491M | +24% |
| gqa_total | 80M | 98M | +23% |
| moe_total | 4.04B | 4.25B | +5% |
| **TOTAL** | **1.76B** | **5.01B** | **+185%** |
| active/token | 1.19B | 1.99B | +67% |
| active_ratio | 67.5% | 39.7% | - |

---

## 3. Órgãos Preservados (Intactos)

Todos os órgãos do 1.6B-Nitro estão presentes com **mesmas funções**:

| Órgão | Função | Config (5B) |
|-------|--------|-------------|
| **Heartbeat** | Decisão/ação orgânica | enabled, memory=1024 |
| **Ghost Token** | Anti-catastrophic forgetting | weight=0.07, mask=15% |
| **MTP** | Multi-Token Prediction | depth=2, weight=0.15 |
| **JEPA** | Aprendizado representacional | weight=0.05 |
| **Curiosity** | Exploração ativa | weight=0.02 |
| **Spider Sense** | Detecção de anomalias | enabled |
| **Neuroendocrine** | 5 hormônios (DA, NE, CORT, BDNF, ACh) | auto-escalado |
| **Nitro Offloading** | CPU↔GPU lazy transfer | capacity=8 |

**Hormônios Neuroendócrinos** (auto-escalados para 20 experts):
- Dopamina: recompensa de erro de predição + Ghost bonus
- Norepinefrina: detecção de novidade (router entropy)
- Cortisol: resposta ao stress + Ghost punição
- BDNF: crescimento dependente de atividade
- Acetylcholine: pressão de sono (consolidação)

---

## 4. Arquitetura de Blocos

```
DarwinX-5B Block (×24):
┌─────────────────────────────────────────────────────────────┐
│  x ───► RMSNorm ───► [SSD or GQA] ───► (+ residual)          │
│                    │                                          │
│                    ├─ SSD (18×): SelectiveSSM                │
│                    └─ GQA  (6×):  Grouped-Query Attention    │
│                                                               │
│  ───► RMSNorm ───► DeepSeek-MoE ───► (+ residual)            │
│                    │                                          │
│                    ├─ FineRouter (top-4 from 20)             │
│                    ├─ 20 Fine Experts (1024 dim)              │
│                    └─ 2 Shared Experts (1280 dim)            │
└─────────────────────────────────────────────────────────────┘
```

**Cadência de Attention (3:1)**: Camadas 5, 9, 13, 17, 21, ?  
*Calculada automaticamente: (layer_idx + 1) % 4 == 0*

---

## 5. Parâmetros por Componente

```
Componente             Parâmetros    % do Total
──────────────────────────────────────────────────
embedding              148,894,720    3.0%
ssd_total              491,489,280    9.8%
gqa_total               98,304,000    2.0%
moe_total            4,247,962,080   84.7%
  └─ fine experts      4,147,609,600
  └─ shared experts      94,515,200
  └─ router                5,837,280
mtp                     13,107,200    0.3%
jepa                    13,109,760    0.3%
norms                   125,440      0.0%
──────────────────────────────────────────────────
TOTAL                5,012,992,480   100%
```

---

## 6. Requisitos de VRAM

**Estimativa conservadora (bf16, AdamW):**

| Componente | VRAM |
|------------|------|
| Modelo (5B) | ~10 GB |
| Optimizer states | ~20 GB |
| Gradients | ~10 GB |
| Activations (ctx=8K, bs=1) | ~12 GB |
| **TOTAL** | **~52 GB** |

**GPUs Recomendadas**:
- A100 80GB → ideal
- A100 40GB → apertado (possível com gradient checkpointing)
- H100 80GB → ideal + maior throughput
- Multi-GPU (2× A100 40GB) → necessário para 8K contexto

---

## 7. Requisitos de Treino

**Para 1 epoch em 140B tokens (corpus atual):**

| Config | tok/s | tempo/epoch |
|--------|-------|-------------|
| 1.6B @ 64 | 25 | ~70 anos |
| 5B @ 8192 (A100) | ~2000 | ~800 dias (~2 anos) |

**Com A100 e block_size=8192:**
- Steps/epoch: 140B ÷ 8192 ≈ 17M steps
- @ 2000 tok/s: 70M ÷ 2000 ≈ 8,500 horas ≈ 354 dias ≈ **1.8 meses de treino contínuo**

**Com 8× A100 (data parallel):**
- ~16,000 tok/s efetivos
- **~22 dias por epoch**

---

## 8. Próximos Passos

1. ✅ Config criada: `src/configs/darwin_x_5b.yaml`
2. ⏳ Validar instanciamento do modelo
3. ⏳ Criar script de deploy cloud (Vast.ai/Montreal)
4. ⏳ Teste canário (250 steps)
5. ⏳ Treino contínuo na nuvem

---

## 9. Notas de Implementação

- **Initialização**: Random init (nenhum peso pré-treinado)
- **Tokenizer**: f51_bpe_80k (58,162 tokens efetivos)
- **Checkpointing**: Salvar a cada 1000 steps (~8B tokens)
- **Metrics**: Same 1.6B observability pipeline
- **Nitro**: 8 experts em GPU, resto em CPU (lazy load)

---

**Assinatura**: Config validada via `estimate_darwin_x_parameters()`
