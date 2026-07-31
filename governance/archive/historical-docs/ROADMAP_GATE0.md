# F51 DARWIN-SSD — ROADMAP GATE 0 (Continual Learning)

Data: 2026-07-11
Status: PLANEJAMENTO — nenhum código alterado ainda
Base: Concept Note PARI-CNRC + Auditoria Causal Codex

================================================================================
PRINCIPIO: RESEARCH ANTES DE CODIFICAR
================================================================================

Cada tarefa segue o ciclo:
  1. Pesquisar papers 2024-2026 sobre o tema
  2. Documentar a solução escolhida e alternativas descartadas
  3. Implementar
  4. Testar com evidência (métrica + log)
  5. Commitar com mensagem descritiva

NUNCA alterar código sem antes pesquisar e documentar.

================================================================================
GATE 0.1 — REPLAY NO GRADIENTE (3-5 dias)
================================================================================

Problema:
  O replay buffer guarda exemplos mas nunca entra no loss.backward().
  É usado só em torch.no_grad() para medir forgetting.
  O rehearsal replay não existe no treino real.

Pesquisa prévia (2024-2026):
  - SSR (Huang et al. 2024): replay sintético selecionado reduz forgetting
    em continual pretraining. Usa o próprio modelo pra gerar exemplos de replay.
  - FOREVER (Feng et al. 2026): replay guiado por curva de forgetting.
    Prioriza exemplos onde o modelo mais esqueceu.
  - Replay uniforme é a baseline mais forte e simples.

Solução:
  1. A cada N steps (ex: 50), samplear K exemplos do replay buffer
  2. Criar batch combinado: [batch_atual, replay_batch] no mesmo device
  3. Forward + backward no batch combinado
  4. Peso do replay: loss_replay tem peso 0.2, loss_nova tem peso 0.8
     (seguindo literatura: 10-30% replay é ótimo para continual learning)
  5. Métrica: medir loss separadamente em dados novos vs replay

Referências:
  - SSR: https://arxiv.org/html/2403.01244v1
  - FOREVER: https://arxiv.org/abs/2601.03938
  - Replay Weight Tuning: French 1999, "Catastrophic forgetting in connectionist networks"

Entregável:
  - scripts/darwin_organism.py: _train_cycle modificado
  - Log mostra "replay_loss" e "new_loss" separados
  - Teste: loss de replay diminui ao longo dos ciclos

================================================================================
GATE 0.2 — EXPERTPOOL ↔ MOE REAL (5-7 dias)
================================================================================

Problema:
  ExpertPool cria ExpertModule independentes que nunca são usados.
  Os experts reais estão em DarwinXModel.blocks[*].moe.fine_experts[*].
  usage_count=0 para sempre.

Pesquisa prévia (2024-2026):
  - Lifelong-MoE (Chen et al. 2023): adiciona experts por distribuição e
    congela antigos. Cada domínio ganha seus próprios experts.
  - SEE (Wang et al. 2025): sequential ensemble of experts.
    Ao mudar de domínio, adiciona novos experts e congela parcialmente os antigos.
  - SETA (2026): split-on-share. Experts específicos por domínio + shared experts.
    Crescimento adaptativo baseado em perplexidade.

Solução:
  1. ExpertPool armazena referências aos experts REAIS do DarwinXModel
     (não cria novos ExpertModule independentes)
  2. Cada ExpertRecord aponta para: block_index, expert_index
  3. Métodos do ExpertPool operam nos parâmetros reais:
     - freeze(expert_id): expert.weight.requires_grad = False
     - activate(expert_id): expert.weight.requires_grad = True
     - mask(expert_id): adiciona viés negativo no router pra esse expert
  4. usage_count é atualizado pelo FineRouter (já tem expert_usage_count)

Entregável:
  - f51_darwin/expert_pool.py: refatorado com referências reais
  - f51_darwin/darwin_x.py: FineRouter expõe usage_count pro ExpertPool
  - Teste: freeze de expert → requer_grad=False → peso não muda no optimizer.step()
  - Teste: mask de expert → router nunca seleciona aquele expert

================================================================================
GATE 0.3 — LEGACY LAYERS CAUSAL (5-7 dias)
================================================================================

Problema:
  Mover expert entre tiers (GPU→RAM→SSD→ASHES) só muda metadados.
  O forward do modelo não é afetado.
  "Morte" não impede o expert de processar tokens.

Pesquisa prévia (2024-2026):
  - PackNet (Mallya & Lazebnik 2018): pruning iterativo.
    Cada task ganha uma sub-rede fixa. Abordagem similar ao tiering.
  - Progressive Neural Networks (Rusu et al. 2016): colunas laterais.
    Cada task adiciona uma coluna, colunas antigas são congeladas.
  - HAT (Serra et al. 2018): hard attention masks.
    Máscaras binárias por task, aprendidas durante treino.

Solução:
  Mapear cada Legacy Tier para uma ação real no modelo:

  | Tier   | requires_grad | Router Mask | Efeito real |
  |--------|--------------|-------------|-------------|
  | GPU    | True         | normal      | Treina normalmente |
  | RAM    | True         | normal      | Standby, pronto pra subir |
  | SSD    | False        | normal      | Congelado, mas ainda usado |
  | DEEP   | False        | bias -0.5   | Quase nunca ativado |
  | ASHES  | False        | bias -5.0   | Nunca ativado |
  | SCAR   | False        | bias -inf   | Bloqueado permanentemente |
  | DNA    | False        | normal      | Imutável (seed weights) |

  1. update_expert_tier() aplica as ações reais:
     - Altera requires_grad
     - Adiciona/remove viés no router do bloco correspondente
  2. Router modificado para aceitar bias por expert
  3. Resurrection: descongela + zera bias

Entregável:
  - f51_darwin/legacy_layers.py: update_expert_tier() com ações causais
  - f51_darwin/darwin_x.py: FineRouter.fine_router aceita per-expert bias
  - Teste: mover expert pra ASHES → router zera uso dele
  - Teste: resurrection → expert volta a receber tokens

================================================================================
GATE 0.4 — ABLATION POR EXPERT (3-4 dias)
================================================================================

Problema:
  Temperatura e score são calculados por ciclo, não por expert.
  Não sabemos qual expert preserva qual conhecimento.

Pesquisa prévia:
  - Expert utilization: medir quantos tokens cada expert processa por domínio
  - Domain-expert mutual information: I(domain; expert_assignment)
  - Router z-loss já existe no código (aux_loss)

Solução:
  1. Rastrear, por domínio, quais experts foram ativados
  2. Calcular mutual information entre domínio e expert assignment
  3. Ablation: desligar um expert e medir impacto na perplexidade por domínio
  4. Isso revela: "expert 3 da camada 7 é responsável por matemática"

Entregável:
  - Métricas de rastreamento no forward pass
  - Script de ablação: desliga expert X → mede PPL nos 5 domínios
  - Output: matriz [expert × domínio] de importância

================================================================================
CALENDÁRIO
================================================================================

Semana 1-2 (14-25 Jul):  GATE 0.1 + 0.2 — Replay causal + ExpertPool real
Semana 3-4 (28 Jul-8 Ago): GATE 0.3 — Legacy Layers causal
Semana 5   (11-15 Ago):    GATE 0.4 — Ablação por expert
Semana 6   (18-22 Ago):    WP1 — Datasets de benchmark (5 domínios)
Semana 7-9 (25 Ago-12 Set): WP2 — Piloto (3 seeds, 4 condições)
Semana 10  (15-19 Set):     Análise de resultados + relatório intermediário

================================================================================
MÉTRICAS DE PROGRESSO DIÁRIO
================================================================================

Cada dia de trabalho:
  [ ] Paper lido: título, 3 insights, como se aplica ao F51
  [ ] Código alterado: o quê, por quê, qual a evidência de que funciona
  [ ] Teste rodado: comando, resultado, log
  [ ] Commit: mensagem descritiva com referência ao gate
  [ ] Diário de bordo atualizado: o que aprendi hoje

================================================================================
REFERÊNCIAS CHAVE
================================================================================

Continual Learning:
  - Lifelong-MoE: Chen et al. 2023 (adiciona/congela experts por domínio)
  - SSR: Huang et al. 2024 (replay sintético selecionado)
  - SEE: Wang et al. 2025 (sequential ensemble of experts)
  - SETA: 2026 (split-on-share, crescimento adaptativo)
  - FOREVER: Feng et al. 2026 (replay guiado por forgetting)

Rehearsal Replay:
  - French 1999: "Catastrophic forgetting in connectionist networks"
  - Experience Replay: Mnih et al. 2015 (DQN)
  - GEM: Lopez-Paz & Ranzato 2017 (gradient episodic memory)
  - AGEM: Chaudhry et al. 2019 (averaged GEM, mais eficiente)

Expert Masking/Freezing:
  - PackNet: Mallya & Lazebnik 2018 (pruning iterativo por task)
  - HAT: Serra et al. 2018 (hard attention masks por task)
  - Progressive Networks: Rusu et al. 2016 (colunas laterais)

================================================================================
ULTIMA ATUALIZACAO: 2026-07-11 15:00
================================================================================
