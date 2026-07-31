# PLANO: Aprendizado por Inferência em Tempo Real
Data: 2026-07-12 (Domingo)
Objetivo: Davi aprende com cada interação, não só com batch training

================================================================================
ARQUITETURA DO CICLO DE INFERÊNCIA COM APRENDIZADO
================================================================================

Usuário pergunta → Davi Pensa → Davi Pesquisa → Davi Aprende → Davi Responde
                                                         ↓
                                              Conhecimento persiste
                                              (Forward-Forward + Test-Time Memory + Ghost)

Cada interação tem 5 fases:

FASE 1: PENSAR (Quiet-STaR)
  - Gera 4 pensamentos internos antes de responder
  - Avalia qual pensamento é mais útil
  - Se incerteza > threshold → vai pra FASE 2

FASE 2: PESQUISAR (DuckDuckGo)
  - Busca na internet via DuckDuckGo (sem API key, gratuito)
  - Extrai texto relevante dos resultados
  - Tokeniza o texto pesquisado

FASE 3: APRENDER (Ghost Injection)
  - Pega o texto pesquisado (tokenizado)
  - Faz UM forward pass com Forward-Forward (aprendizado local)
  - Dopamina: +0.5 se achou resposta, +0.1 se incerto
  - Armazena na Test-Time Memory (surpresa alta = memoriza)

FASE 4: RESPONDER
  - Gera resposta usando live_generate()
  - Incorpora memórias relevantes da Test-Time Memory
  - Inclui citação da fonte (DuckDuckGo URL)

FASE 5: PERSISTIR
  - O conhecimento fica nos pesos (Forward-Forward layers)
  - A memória fica na Test-Time Memory (acesso rápido)
  - O replay buffer guarda o exemplo (anti-esquecimento)
  - Se útil, promove ao corpus (Firewall audita)

================================================================================
IMPLEMENTAÇÃO
================================================================================

NOVO MÓDULO: f51_darwin/inference_learner.py
  - InferenceLearner: orquestra as 5 fases
  - DuckDuckGoSearch: pesquisa na internet
  - Extractor: extrai texto relevante dos resultados HTML

MODIFICAÇÕES:
  f51_darwin/heartbeat.py:
    - TestTimeMemory: write_if_surprised() já existe, ok
    - Forward-Forward: forward(x, dopamine) já existe, ok
  
  f51_darwin/darwin_x.py:
    - live_generate() modificado: chama InferenceLearner
    - Novo método: learn_from_interaction(prompt, response, research)
  
  scripts/darwin_organism.py:
    - NOVO MODO: "serve" — servidor WebSocket com UI
    - Cada mensagem do usuário dispara o ciclo de 5 fases
    - Dashboard: loss, heartbeat, memórias, experts

================================================================================
FLUXO COMPLETO
================================================================================

┌──────────────────────────────────────────────────────────────┐
│                    DAVI — UI EM TEMPO REAL                    │
│                                                              │
│  [Usuário] "Explique a teoria da relatividade"              │
│       ↓                                                      │
│  [Davi pensa] Quiet-STaR: "não sei bem, preciso pesquisar"  │
│       ↓                                                      │
│  [Davi pesquisa] DuckDuckGo → "relativity theory Einstein"  │
│       ↓                                                      │
│  [Davi lê] Extractor: extrai 3 parágrafos relevantes        │
│       ↓                                                      │
│  [Davi aprende] Forward-Forward nos tokens pesquisados      │
│       ↓                                                      │
│  [Davi memoriza] Test-Time Memory: guarda o conceito        │
│       ↓                                                      │
│  [Davi responde] "A teoria da relatividade, proposta por    │
│   Einstein em 1905, descreve..."                            │
│       ↓                                                      │
│  [Davi persiste] Replay buffer guarda interação             │
│                                                              │
│  Próxima vez que perguntarem:                                │
│  [Davi lembra] Test-Time Memory recupera → responde direto  │
└──────────────────────────────────────────────────────────────┘

================================================================================
GHOST 10% — EXPLORAÇÃO EM TEMPO REAL
================================================================================

O Ghost Token (15% mask durante treino) vira:
  Ghost Injection (10% do orçamento de exploração durante inferência)

A cada interação:
  - 90% do forward: geração normal da resposta
  - 10% do forward: aprendizado nos tokens pesquisados
  
  Isso mantém a doutrina: max 10% de dados sintéticos/exploratórios

================================================================================
IMPLEMENTAÇÃO PASSO A PASSO
================================================================================

1. inference_learner.py — motor de aprendizado em tempo real
2. Modificar live_generate() para chamar o motor
3. Servidor WebSocket com UI (FastAPI + HTML simples)
4. DuckDuckGo search integrado
5. Test-Time Memory retrieval durante geração
6. Forward-Forward learning nos tokens pesquisados

================================================================================
