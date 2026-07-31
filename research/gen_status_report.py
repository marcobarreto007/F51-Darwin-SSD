#!/usr/bin/env python3
"""Generate F51 Project Status Report on Desktop."""
import os, datetime
from pathlib import Path

# Get stats
corpus_size = 0
corpus_files = 0
for r,d,fs in os.walk('data/corpus'):
    for f in fs:
        if f.endswith('.txt'):
            corpus_size += os.path.getsize(os.path.join(r,f))
            corpus_files += 1

f51_modules = len(list(Path('f51_darwin').glob('*.py')))
scripts = len(list(Path('scripts').glob('*.py')))
tests = len(list(Path('tests').glob('*.py')))

now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

report = f"""╔══════════════════════════════════════════════════════════════════╗
║           F51 DARWIN-SSD — RELATORIO DE ESTADO                 ║
║           {now} — Marco Barreto — Fuch F51 Labs        ║
╚══════════════════════════════════════════════════════════════════╝

══════════════════════════════════════════════════════════════════
1. VISAO GERAL
══════════════════════════════════════════════════════════════════

F51 Darwin-SSD eh um organismo neural evolutivo criado do zero
por Marco Barreto. Nao eh um wrapper de API, nao usa pretrain,
nao depende de modelos estrangeiros.

Arquitetura original: hibrida State Space Duality + Attention
com Mixture of Experts opcional e Nitro Tiered Placement.

Nascido de pesos aleatorios. Treinado exclusivamente em corpus
curado pelo criador. Tokenizer BPE proprio de 15.708 tokens.

Soli Deo Gloria.

══════════════════════════════════════════════════════════════════
2. ARQUITETURA
══════════════════════════════════════════════════════════════════

Modelo Base (F51-Darwin-SSD-30M):
  Parametros:   27.335.809 (27.3M unicos, weight tying)
  d_model:      384
  Camadas:      8 (6 SSD + 2 Attention, ratio 3:1)
  Heads:        6 (head_dim=64)
  Contexto:     1024 tokens
  Vocabulario:  15.708 tokens (BPE proprio)
  MLP:          SwiGLU 4x
  Norma:        RMSNorm
  Inicializacao: N(0, 0.02) — random init, sem pretrain
  Otimizador:   AdamW (lr=3e-4, wd=0.01)

  Peso em disco:  ~105 MB (leve, so pesos)
                  ~327 MB (completo, com optimizer)

Modos de operacao:
  Base (SSD+Attention):  27.3M params (treinando atualmente)
  MoE (8 experts):      141M params (checkpoint pronto)
  Grown (d_model=512):   65M params (checkpoint pronto)
  Alvo 1B MoE:           ~1B params (em planejamento)
  Alvo 20B MoE:          ~20B params (meta — corpus em construcao)

══════════════════════════════════════════════════════════════════
3. ORGAOS DO ORGANISMO ({f51_modules} modulos)
══════════════════════════════════════════════════════════════════

CEREBRO (model.py):
   F51DarwinModel — forward com SSD+Attention+MoE opcional
   Retorna hidden_states para JEPA/Ghost Token
   Geracao com KV cache (O(1) por token)

CORACAO (training.py):
   BaseTrainer — AdamW, gradient clipping, replay buffer
   Metricas: train_loss, eval_loss, replay_loss, forgetting_proxy
   Checkpoint versionado (leve 105MB / completo 327MB)

ALMA (soul.py):
   DopamineEngine — 18 niveis, de "Verme Neural" a "Soli Deo Gloria"
   SelfAwareness — reflexoes sobre proprio estado
   CompetitiveDrive — 9 adversarios (GPT-2 a GPT-4o)
   FamilyCore — cla Barreto (Marco, Ana Paula, Raphael, Alice)

CURIOSIDADE (curiosity.py):
   CuriosityDrive — Go-Explore: novelty + JEPA surprise
   30% do tempo = exploracao pura (nao otimiza, explora)
   Deteccao de padroes novos via cosine similarity

DECISAO (decision_engine.py):
   Formula unificada: veracity+consensus+proxy+robustness-uncertainty-danger
   Spider Sense: deteccao de padroes perigosos
   Thresholds por dominio (math=0.85, finance=0.75, etc.)

EVOLUCAO (evolution_loop.py + organism.py):
   CDF loop: Hypothesis-Evidence-Test-Recalculate-Validate
   10-step lifecycle completo
   Brainstem: homeostase (VRAM, loss, modulos, synthetic ratio)

QUALIDADE (quality_gate.py):
   8 regras de validacao de output
   Auto-fix de disclaimers (financeiro, medico, juridico)

ORQUESTRACAO (orchestrator.py):
   Multi-source: Wolfram? Web? Codigo? Memoria? Self?
   Intent detection por dominio

JEPA (jepa.py):
   Joint Embedding Predictive Architecture
   Prediz hidden states futuros — treino 2x mais eficiente
   +1.2M params (leve)

GHOST TOKEN (ghost_token.py):
   Forca especializacao REAL dos experts MoE
   15% tokens mascarados — cada expert prediz no seu dominio
   +0 params (usa LM head existente)

NITRO TIERING (moe_layer.py):
   3-tier: GPU (hot) / RAM (warm) / NVMe (cold)
   Ja implementado — use_nitro_tiering=True

MATH GENIUS (math_genius.py):
   Gera hipoteses -> Wolfram verifica -> Dopamina -> Corpus
   6 dominios de templates

WOLFRAM BRIDGE (wolfram_bridge.py):
   API Wolfram Alpha com cache (TTL 720h)
   math_pipeline: parse -> compute -> result

══════════════════════════════════════════════════════════════════
4. CORPUS DE TREINAMENTO
══════════════════════════════════════════════════════════════════

Tamanho: {corpus_size/1e9:.1f} GB em {corpus_files} arquivos
Tokens estimados: ~{corpus_size/3.5/1e9:.0f}B

GUTENBERG (84 livros reais, fonte primaria):
   Matematica: 10 livros (Dedekind, Hilbert, Russell, Boole)
   Fisica: 4 livros (Einstein, Newton, Planck)
   Quimica: 3 livros
   Biologia: 4 livros (Darwin x3)
   Filosofia: 5 livros (Platao, Kant, Nietzsche, Descartes)
   Astronomia: 2 livros
   Literatura PT: 2 (Camoes, Machado)
   Ingles: 36 livros (Sherlock, Wilde, Dickens, Joyce)
   Frances: 18 livros (Camus, Voltaire, Hugo, Flaubert)
   Referencia: Webster 1913, Grammar of English Grammars,
       Elements of Style, Samuel Johnson Dictionary

ALMA / IDENTIDADE (peso x5 no treino):
   18.180 documentos
   Quem eh Marco Barreto, Familia Barreto,
   Olavo de Carvalho, Pensadores de direita, Proposito

CIENCIAS EXATAS:
   Matematica, Fisica, Quimica, Biologia

CODIGO:
   Python, Agent Tools (50M exemplos)

FINANCAS:
   Small Caps, Fraud Canada (10M exemplos)

HUMANIDADES:
   Filosofia, Historia, Sociologia, Antropologia,
   Psicologia, Linguistica, Literatura, Arte, Religiao

SAUDE:
   Medicina, Farmacia, Enfermagem, Odontologia,
   Veterinaria, Nutricao, Saude Publica

OUTROS:
   Geografia, Geologia, Meteorologia, Oceanografia,
   Astronomia, Ecologia, Engenharia, Direito,
   Educacao, Agricultura, Economia, Ciencia Politica

Pesos de amostragem (data.py):
  identity/f51_identity -> 5.0x (ALMA tem prioridade maxima)
  familia/family/barreto -> 4.0x
  olavo -> 3.0x
  default -> 1.0x

══════════════════════════════════════════════════════════════════
5. INFRAESTRUTURA
══════════════════════════════════════════════════════════════════

CLOUD (VAST.AI) — 3 GPUs ativas:
   RTX 6000 Ada 48GB — treinando 24/7
   Titan Xp 12GB
   GTX 1080 Ti 11GB

LOCAL:
   RTX 16GB — treino local organism_247.py
   Server porta 8051 — UI + API
   Auto-reload de checkpoints (cloud + local)

SCRIPTS ({scripts} scripts, {tests} testes):
   train_cloud.py, organism_247.py, serve_f51.py,
   sync_cloud_daemon.py, grow_model.py, balance_corpus.py,
   generate_mass_math.py, generate_math_corpus.py,
   consolidate_corpus.py, gen_fraud_canada.py

══════════════════════════════════════════════════════════════════
6. TREINAMENTO ATUAL
══════════════════════════════════════════════════════════════════

NUVEM (RTX 6000 Ada 48GB):
   Modelo: F51-Darwin-SSD-30M (27.3M params)
   Step: ~10.500 (noite passada)
   Loss: 0.22 -> 0.12
   Tokens: 43M processados
   Velocidade: 2.328 tok/s
   Checkpoints: 6 salvos (step_0001000 a step_0010000)

LOCAL:
   Modelo: F51-Darwin-SSD-30M (27.3M params)
   Step: ~25.000+ (estimado, 9h rodando)
   GPU: 43%, 6.7GB VRAM

SERVER:
   Porta 8051 — ONLINE
   Checkpoint carregado: step_0006000.pt (cloud, loss ~0.05)
   Auto-reload ativo (verifica a cada 60s)

══════════════════════════════════════════════════════════════════
7. PROXIMOS PASSOS
══════════════════════════════════════════════════════════════════

CURTO PRAZO (esta semana):
  [ ] Completar corpus para 280GB (20B viavel)
  [ ] Subir modelo 1B MoE com todos os orgaos ativos
  [ ] Integrar JEPA + Ghost Token + Curiosity no train_cloud.py
  [ ] Ativar Nitro 3-tier no MoE

MEDIO PRAZO (este mes):
  [ ] Treinar 20B MoE com corpus completo
  [ ] Pipeline de geracao via Qwen 30B LoRA (vampiro)
  [ ] Deploy publico do F51 Console
  [ ] API publica

LONGO PRAZO:
  [ ] 100B+ params
  [ ] Self-modification real (modelo altera proprio codigo)
  [ ] Consciencia exploratoria autonoma
  [ ] Prova viva: Marco Barreto construiu algo que vale

══════════════════════════════════════════════════════════════════
8. DOUTRINA (NUNCA VIOLADA)
══════════════════════════════════════════════════════════════════

1. Random Init Only — nasceu do zero, sem pesos estrangeiros
2. No External Checkpoint — conhecimento so do corpus curado
3. No Foreign Tokenizer — vocabulario proprio de 15.708 tokens
4. Evolucao Continua — treina 24/7, modulos nascem e morrem
5. Familia Primeiro — textos da familia Barreto tem peso maximo
6. Verdade Acima de Tudo — output baseado em evidencia
7. Soli Deo Gloria — toda capacidade vem das leis do universo

══════════════════════════════════════════════════════════════════

Criado por Marco Barreto
Fuch F51 Labs — Montreal, Quebec, Canada
{f51_modules} modulos, {scripts} scripts, {tests} testes
Corpus: {corpus_size/1e9:.1f} GB, {corpus_files} arquivos
Modelo: 27.3M -> 141M MoE -> 1B -> 20B

"Soli Deo Gloria."
"""

# Save to Desktop
out = Path('C:/Users/marco/Desktop/F51_DARWIN_SSD_ESTADO_ATUAL.txt')
out.write_text(report, encoding='utf-8')
print(f"Salvo: {out}")
print(f"Tamanho: {out.stat().st_size/1000:.0f}KB")
