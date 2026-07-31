# Relatório de Portfólio — Marco Barreto
## 19 repositórios · auditados em 2026-07-31

---

## ⭐ PRODUTOS ATIVOS (3)

### 1. xubuget (privado) — App Android Fintech com IA Local
- **Arquivos:** 68 (43 Kotlin)
- **Commits:** 3
- **Stack:** Kotlin · Gradle · LiteRT · RAG · OCR · FR/EN UI
- **Descrição:** App Android "Xubuget" — interceptador financeiro com IA local. On-device SLM + RAG com 189 chunks de conhecimento financeiro. Importa CSV do TD Bank. OCR de recibos. Ledger "kind-first". Gemini desabilitado. APK de 105MB distribuído via GitHub Releases.
- **CI:** Sem CI
- **Risco:** APK público com 105MB. Código fonte privado.
- **Potencial:** ⭐⭐⭐ Produto. Monetizável. Já tem landing page e release.

### 2. MIKE (privado) — Assistente Familiar Local
- **Arquivos:** 363 (171 Python)
- **Commits:** 2
- **Stack:** Python · FastAPI · DeepSeek · Telegram · Twilio · Shopify · LightRAG · Mem0
- **Descrição:** "Assistente local da família Barreto". Servidor MCP com integrações reais (Telegram, Twilio SMS, Shopify, Google Ads). Memória com LightRAG e Mem0. Agente autônomo com task board, verifier, briefing. Dashboard web.
- **CI:** ❌ Corrigido hoje (token hardcoded em teste, llama-cpp-python unpinned)
- **Risco:** ⚠️ Tokens de produção (Telegram, Twilio, Shopify, Google Ads) estavam no código. Um foi removido hoje. Verificar os demais.
- **Potencial:** ⭐⭐⭐ Produto familiar. Pode ser licenciado como "AI familiar local".

### 3. VeriBay / bmw-diag-mvp (privado) — Diagnóstico Automotivo com IA
- **Arquivos:** 387 (151 Python)
- **Commits:** 98
- **Stack:** Python · CAN bus · BMW · Transport Canada · FastAPI
- **Descrição:** "VeriBay - Evidence-Guided Automotive Diagnostics". Diagnóstico automotivo com ingestão de dados CAN bus (opendbc), recalls do Transport Canada, fluxo de diagnóstico guiado por evidências. Módulo específico BMW.
- **CI:** ✅ Passando
- **Potencial:** ⭐⭐⭐ Ferramenta profissional. Mercado de reparo automotivo.

---

## 🔬 PESQUISA ATIVA (1)

### 4. F51-Darwin-SSD (privado) — Pipeline de Catálogo Criptográfico
- **Arquivos:** ~200 (após limpeza)
- **Stack:** Python · PyTorch · ROME · MEMIT · SAE · SHA-256
- **Descrição:** Catálogo de 196.608 neurônios com SHA-256. Edição cirúrgica de conhecimento via ROME (6/6 identidade). Memória persistente com verificação de integridade. Rollback exato. Paper em andamento. Patente em preparação.
- **CI:** ❌ Corrigido hoje (path do audit)
- **Potencial:** ⭐⭐⭐ Paper + patente. Porta de entrada para carreira em AI Safety.

---

## 🏗️ ECOSSISTEMA F51 (histórico/pesquisa)

### 5. SuperEzio (privado) — "Máquina de Dinheiro" · Trading AI
- **Arquivos:** 859 (411 Python, 82.605 linhas)
- **Commits:** 619
- **Stack:** Python · IBKR · LoRA · FastAPI · Docker
- **Descrição:** Plataforma de trading automatizado. Integração com Interactive Brokers (IBKR). Agente "Bob" com "consciousness" (autonomic, body, mission_context). Signal ledger com policy. Modelo LoRA treinado para finanças em português. "CEO fallback". "Buyer setup guide". Journal de trading. 3 anos de commits.
- **CI:** ❌ Corrigido hoje (openai dependency + lint)
- **Risco:** ⚠️ Credenciais IBKR. Estratégia de trading proprietária.
- **Potencial:** ⭐⭐⭐ Produto comercial. Se funciona, vale dinheiro real.

### 6. f51-verdict (privado) — Motor de Decisão com Incerteza Calibrada
- **Arquivos:** 10 (7 Python)
- **Commits:** 1
- **Descrição:** "Motor de decisão com incerteza calibrada". Projeto pequeno, possivelmente um componente do SuperEzio.
- **CI:** ✅ Passando
- **Potencial:** ⭐ Componente. Não é produto standalone.

### 7. f51-labs (privado) — Laboratório F51
- **Arquivos:** 122 (86 Python)
- **Commits:** 3
- **Descrição:** Código de experimentos do ecossistema F51. Agentes, testes, README docs.
- **CI:** Sem CI
- **Potencial:** ⭐ Histórico. Código de experimentação.

### 8. f51-notes (privado) — Notas F51
- **Arquivos:** 218 (83 TypeScript, 56 TSX)
- **Commits:** 58
- **Descrição:** Aplicação de notas com frontend React. Projeto ativo com 58 commits.
- **CI:** ✅ Passando
- **Potencial:** ⭐⭐ Aplicação pessoal. Poderia ser productizada.

### 9. f51-gemini-arsenal (privado) — VAZIO
- **Arquivos:** 0
- **Commits:** 0
- **Descrição:** Repositório vazio. Criado e abandonado.
- **Potencial:** 🗑️ Deletar ou usar para outra coisa.

### 10. f51-nitro-moe (público) — Relatório Técnico
- **Arquivos:** 3
- **Commits:** 1
- **Descrição:** Relatório técnico e logs do F51 Nitro-MoE. Histórico.
- **Potencial:** 🗑️ Arquivar.

---

## 🏗️ ECOSSISTEMA COMPILAI (computer vision)

### 11. compilai-local (público) — Engine de Computer Vision
- **Arquivos:** 306
- **Commits:** Vários
- **Stack:** Python · Flask · YOLO · BoT-SORT
- **Descrição:** "CompilAI TMC Engine". Backend Flask com YOLO para detecção e BoT-SORT para tracking.
- **CI:** Sem CI
- **Potencial:** ⭐⭐ Projeto funcional. Mercado de monitoramento por vídeo.

### 12. compilai-backend (privado) — OmniTMC Backend
- **Arquivos:** 194 (169 Python)
- **Commits:** 2
- **Descrição:** Backend Python do OmniTMC.
- **CI:** Sem CI
- **Potencial:** ⭐ Componente do ecossistema Compilai.

### 13. compilai-frontend (privado) — Frontend Compilai
- **Arquivos:** 9 (4 JS)
- **Commits:** 1
- **Descrição:** Frontend web do Compilai. Mínimo.
- **CI:** Sem CI
- **Potencial:** ⭐ Incompleto.

---

## 📦 OUTROS PRODUTOS

### 14. canada-tax-ai (privado) — Assistente Fiscal Canadense
- **Arquivos:** 345 (120 Python)
- **Commits:** 12
- **Stack:** Python · Llama.cpp · React · FastAPI
- **Descrição:** "Assistente fiscal canadense local-first. Copiloto de imposto de renda que roda na tua máquina, sem nuvem." Frontend React + Backend Python. Integração com llama.cpp para IA local.
- **CI:** Sem CI
- **Potencial:** ⭐⭐⭐ Produto sazonal. Todo canadense precisa declarar imposto. Mercado enorme.

### 15. OmniTMC (privado) — Dados de Treinamento
- **Arquivos:** 419 (187 JSON, 101 MP4)
- **Commits:** 1
- **Descrição:** Conjunto de dados com 187 arquivos JSON e 101 vídeos MP4. Sem README.
- **Potencial:** ⭐ Dados brutos. Não é produto.

### 16. tracepose-app (privado) — Pose Estimation App
- **Arquivos:** 122 (25 TypeScript, 26 PNG)
- **Commits:** 31
- **Descrição:** "TracePose" — aplicação de estimação de pose. "OCR preenche estudo/missao e prioriza fluxo stock". 31 commits.
- **CI:** Sem CI
- **Potencial:** ⭐⭐ App mobile/web de pose estimation.

### 17. nelsonmath (privado) — Educação Matemática
- **Arquivos:** 209 (162 Python)
- **Commits:** 3
- **Descrição:** "NelsonMath" — plataforma de educação matemática. 162 arquivos Python. "Rename public project identity to NelsonMath".
- **CI:** Sem CI
- **Potencial:** ⭐⭐ Produto educacional.

### 18. motor-de-livro-jogo-gamebook (privado) — Motor de Gamebook
- **Arquivos:** 165 (42 PNG, 31 YML, 22 WAV)
- **Commits:** 1
- **Descrição:** Motor de livro-jogo (gamebook interativo). Assets visuais e sonoros.
- **Potencial:** ⭐ Projeto criativo. Não é produto comercial.

---

## 📊 SUMÁRIO

| Categoria | Repos | CI Passando |
|---|---|---|
| ⭐ Produtos ativos | 3 | 1/3 |
| 🔬 Pesquisa ativa | 1 | 0/1 (corrigido) |
| 🏗️ Ecossistema F51 | 6 | 2/6 |
| 🏗️ Ecossistema Compilai | 3 | 0/3 |
| 📦 Outros | 5 | 0/5 |
| 🗑️ Lixo | 1 (f51-gemini-arsenal) | — |

**Top 5 por potencial de monetização:**
1. SuperEzio — Trading AI · 619 commits · Se funciona, é dinheiro real
2. xubuget — Fintech Android · Já tem release
3. canada-tax-ai — Imposto canadense · Mercado enorme
4. bmw-diag-mvp — Diagnóstico automotivo · Mercado profissional
5. F51-Darwin-SSD — Paper + patente · Porta de entrada em AI Safety

**Ações imediatas recomendadas:**
1. Auditar segredos no mike (Telegram, Twilio, Shopify, Google Ads)
2. Verificar credenciais IBKR no SuperEzio
3. Remover f51-gemini-arsenal (vazio)
4. Arquivar f51-nitro-moe (histórico)
5. Completar trilingual READMEs nos produtos ativos
