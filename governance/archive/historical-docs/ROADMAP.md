> **DOCUMENTO HISTORICO - NAO USAR COMO ESTADO VIVO**
>
> Este roadmap registra a direcao de 2026-07-12 para a antiga linhagem 600M.
> Ele nao substitui `README.md` nem `docs/STATUS_ATUAL.md` e nao autoriza
> iniciar, parar ou escolher a linhagem atual. O objetivo operacional vigente
> e validar e retomar o checkpoint da classe 2.5B com corpus externo e as duas
> GPUs no mesmo processo.

# F51 DARWIN-SSD — ROADMAP LÓGICO
Data: 2026-07-12 08:30 EDT
Status: FASE 1 concluída. Iniciando FASE 2.

================================================================================
ONDE ESTAMOS (12 Jul 2026, 08:30)
================================================================================

FASE 1 — PROVADO (48h de trabalho):
  ✅ Arquitetura Darwin-X 600M funcional (SSD + Attention + MoE)
  ✅ Dual GPU pipeline heterogêneo (5060 Ti 16GB + 3060 12GB)
  ✅ Organismo 24/7 estável (208 ciclos, 103k steps, zero mortes)
  ✅ Legacy-first evolution (72 experts vivos)
  ✅ Replay no gradiente (previne catastrophic forgetting)
  ✅ ExpertPool ↔ MoE real (freeze/activate/mask são causais)
  ✅ Legacy Layers causais (TIER_ACTIONS afetam modelo real)
  ✅ Mode collapse quebrado (feast externo)
  ✅ Modelo aprende inglês + vocabulário científico sozinho
  ✅ lm=0.22 (melhor valor registrado)

FASE 2 — ESTA SEMANA (13-19 Jul):
  [ ] BENCHMARK: Perplexidade WikiText-2 + HellaSwag + PIQA
      → Métrica objetiva comparável com GPT-2 Small, SmolLM, Pythia
  [ ] CONTINUAR TREINO: +500k steps com o feast externo
      → Sair de lm=0.22 para lm<0.1
  [ ] DIVERSIDADE DE DADOS: Pipeline ingestão ativo
      → Ghost Stream + Firewall + Tokenização → novos domínios
  [ ] CHECKPOINT SEMANAL: Salvar e versionar

FASE 3 — EXPERIMENTO CONTROLADO (20 Jul - 3 Ago):
  [ ] Preparar 5 datasets de domínio (Math, Code, Medicine, Literature, Finance)
  [ ] Rodar piloto: 4 condições × 3 seeds
      C0: Darwin-X estático (sem replay, sem legacy)
      C1: Darwin-X + replay uniforme
      C2: Darwin-X + continual expert baseline
      F51: Organismo completo
  [ ] Métrica primária: forgetting ratio < 1.2
  [ ] Relatório técnico preliminar

FASE 4 — PAPER + PARI (4 Ago - 31 Ago):
  [ ] Escrever paper técnico (Abstract, Method, Results, Ablation)
  [ ] Preparar concept note 2 páginas para NRC IRAP
  [ ] Contato com Innove Ici + ITA do NRC
  [ ] Opinião de IP (patentabilidade da arquitetura evolutiva)

FASE 5 — SCALE (Setembro+):
  [ ] Treinar Darwin-X 1.2B (cabe nas 2 GPUs)
  [ ] Deploy cloud A100 para 4B+
  [ ] Servidor de inferência
  [ ] Publicar + buscar financiamento

================================================================================
PRIORIDADE AGORA (hoje):
================================================================================

1. BENCHMARK — 2 horas
   Rodar WikiText-2 perplexity no checkpoint 207.
   Comando: python scripts/benchmark_darwin.py --bench perplexity
   Objetivo: número real pra comparar com GPT-2 Small (PPL=37.5)

2. CONTINUAR TREINO — rodando
   Não parar. Deixar feast externo + replay + Ghost.
   Objetivo: +500k steps até domingo.

3. NÃO MEXER NO CÓDIGO
   O sistema está estável. Qualquer mudança agora quebra
   o experimento controlado da FASE 3.
   Só mexer se for correção de bug.

================================================================================
REGRAS DA FASE 2:
================================================================================

- NÃO aumentar modelo (0.5B é suficiente pro experimento)
- NÃO mudar arquitetura (Darwin-X 600M é o baseline)
- NÃO adicionar features novas (Gate 0 já está completo)
- SIM rodar benchmark (métrica objetiva)
- SIM continuar treino (mais tokens = melhor modelo)
- SIM documentar tudo (SR&ED)

================================================================================
