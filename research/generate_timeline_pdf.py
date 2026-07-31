#!/usr/bin/env python3
"""Generate comprehensive F51 Darwin-SSD timeline PDF on the desktop."""
# -*- coding: utf-8 -*-

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from datetime import datetime

class TimelinePDF(FPDF):
    FONT = "Helvetica"
    
    def clean(self, text):
        replacements = {
            '—': '--', '–': '-', '‘': "'", '’': "'",
            '“': '"', '”': '"', '…': '...',
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
            'ã': 'a', 'õ': 'o', 'â': 'a', 'ê': 'e', 'ô': 'o',
            'ç': 'c',
            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
            'À': 'A', 'È': 'E', 'Ì': 'I', 'Ò': 'O', 'Ù': 'U',
            'Ã': 'A', 'Õ': 'O', 'Â': 'A', 'Ê': 'E', 'Ô': 'O',
            'Ç': 'C',
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        result = []
        for c in text:
            try:
                c.encode('latin-1')
                result.append(c)
            except UnicodeEncodeError:
                if ord(c) <= 255:
                    result.append(c)
                else:
                    result.append('?')
        return ''.join(result)
    
    def ttl(self, text, size=28):
        self.set_font(self.FONT, 'B', size)
        self.cell(0, size*0.45, self.clean(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    
    def sub(self, text, size=14):
        self.set_font(self.FONT, 'I', size)
        self.cell(0, size*0.7, self.clean(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    
    def info(self, text, size=10):
        self.set_font(self.FONT, '', size)
        self.cell(0, size*0.7, self.clean(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    
    def sec(self, text):
        self.set_font(self.FONT, 'B', 16)
        self.set_fill_color(180, 30, 30)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, f"  {self.clean(text)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(4)
    
    def hdr(self, text, size=12, color=(180,30,30)):
        self.set_font(self.FONT, 'B', size)
        self.set_text_color(*color)
        self.cell(0, size*0.55, self.clean(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)
    
    def txt(self, text, size=9, bold=False, italic=False, font=None):
        if bold:
            self.set_font(self.FONT, 'B', size)
        elif italic:
            self.set_font(self.FONT, 'I', size)
        elif font:
            self.set_font(font, '', size)
        else:
            self.set_font(self.FONT, '', size)
        self.cell(0, size*0.5, self.clean(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    def sep(self):
        self.set_draw_color(180, 30, 30)
        self.set_line_width(0.8)
        y = self.get_y()
        self.line(20, y, 190, y)
        self.ln(6)
    
    def foot(self, text, bold=False, size=9, align='C'):
        if bold:
            self.set_font(self.FONT, 'B', size)
        else:
            self.set_font(self.FONT, 'I', size)
        self.cell(0, size*0.6, self.clean(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align=align)


pdf = TimelinePDF()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# Cover
pdf.ttl('Projeto F51 Darwin-SSD')
pdf.sub('Linha do Tempo Completa - Auditoria de Estado')
pdf.info(f'Gerado em: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")} - Marco Barreto - Fuch F51 Labs - Montreal, QC', 9)
pdf.info('Branch: cursor/bilingual-readme-olavo-sources - github.com/marcobarreto007/F51-Darwin-SSD', 9)
pdf.ln(4)
pdf.sep()

# ══════ STATUS ══════
pdf.sec('STATUS DE SALVAMENTO (07/07/2026)')
pdf.hdr('AVISO: Ha arquivos NAO salvos (unstaged) e NAO commitados!', 12)
pdf.ln(4)
pdf.txt('Arquivos modificados (unstaged -- 6 arquivos, +280/-66 linhas):', 9, bold=True)
for l in [
    '  - src/f51_darwin/anti_woke_filter.py   -- reescrita (+94/-94 linhas)',
    '  - src/f51_darwin/darwin_x_training.py  -- expansao (+131 linhas)',
    '  - src/f51_darwin/data.py               -- ajuste (+11 linhas)',
    '  - src/scripts/auto_train_1h.sh         -- ajuste menor',
    '  - research/train_darwin_x.py        -- ajuste (+9/-2 linhas)',
    '  - src/tests/test_darwin_x_training.py  -- novos testes (+99 linhas)',
]: pdf.txt(l)
pdf.ln(4)
pdf.txt('Arquivos novos (untracked -- 26 arquivos):', 9, bold=True)
for l in [
    '  Configs (1): corpus_factory.yaml',
    '  Docs (1): CORPUS_FACTORY_OPERATING_SYSTEM.md',
    '  Core (8): corpus_cloud_ingest.py, corpus_cloud_token_index.py,',
    '            corpus_cloud_token_verify.py, corpus_cloud_tokenize.py,',
    '            corpus_factory_pipeline.py, corpus_pack_upload.py,',
    '            corpus_pack_verifier.py, corpus_policy.py',
    '  Scripts (9): build_corpus_pack.py, build_remote_token_index.py,',
    '               cloud_train_darwin_x_fixed.sh, ingest_remote_corpus_pack.py,',
    '               rent_safe.sh, tokenize_remote_corpus_batch.py,',
    '               upload_corpus_pack.py, verify_corpus_pack.py,',
    '               verify_remote_token_batch.py',
    '  Tests (7): test_corpus_cloud_ingest.py, test_corpus_cloud_token_index.py,',
    '             test_corpus_cloud_token_verify.py, test_corpus_cloud_tokenize.py,',
    '             test_corpus_factory_pipeline.py, test_corpus_pack_upload.py,',
    '             test_corpus_policy.py',
]: pdf.txt(l)
pdf.ln(4)
pdf.txt('Branch: +21 commits locais NAO enviados ao GitHub origin', 9, bold=True)
pdf.txt('Stash: vazio (sem alteracoes escondidas)', 9)
pdf.ln(4)

# ══════ RESUMO ══════
pdf.sec('RESUMO DO PROJETO')
for l in [
    'F51 Darwin-SSD: Organismo neural evolutivo de linguagem -- do zero, local, auditavel.',
    'Semente hibrida SSD-dominante + Transformer (3:1). Sem pesos, vocabulario ou tokenizers',
    'de Llama, Qwen, Mistral, GPT, Gemma, Phi ou DeepSeek.',
    '',
    'Criador: Marco Barreto - Lab: Fuch F51 Labs - Base: Montreal, Quebec, Canada',
    'Motto: Soli Deo Gloria',
    '',
    'ESTATISTICAS GERAIS (07/07/2026):',
    '  - 109 commits totais no repositorio',
    '  - 377 arquivos rastreados por git (sem dados, checkpoints, cache)',
    '  - 280 arquivos Python (~60.208 linhas de codigo)',
    '  - 5.185 arquivos .txt (corpus, documents, approvals, data)',
    '  - 35 testes unitarios (100% passando: pytest -q)',
    '  - ~122 GB de dados totais (incluindo corpus, checkpoints, tokens)',
    '  - 8 dias de desenvolvimento intensivo (30/06/2026 a 07/07/2026)',
    '',
    'MODELOS:',
    '  - Seed base:       ~27.45M parametros (config label "30M")',
    '  - MoE seed:        ~140.7M parametros, 8 domain experts',
    '  - Darwin-X 4B:     DeepSeekMoE + GQA + MTP + RoPE 1M + Ghost + Spider + JEPA',
    '  - Darwin-X 9B MoE: Codex + mixed precision + fast tokenizer',
    '  - Darwin-X cloud scale: historical 20B MoE plan, superseded by Law 0',
    '  - Treino base:     smoke + resume validado - step 15.000 atingido',
    '  - GPU:             NVIDIA RTX 5060 Ti (17.1 GB) - PyTorch 2.8.0+cu129',
    '',
    'CORPUS PRINCIPAIS:',
    '  - 8 Agentes (4 Biologia + 4 Fisica Quantica): 20M docs',
    '  - 4 Agentes de Matematica: 20M docs, 8.4 GB',
    '  - Financial Corpus Top 10: 10M docs, 4.1 GB',
    '  - Conservative Canon: 765k docs, 31 pensadores',
    '  - Python 50M corpus sintetico (autoria Claude, quarentenado)',
    '  - Corpus Fantasma: 151GB split TRAIN(80%)/GHOST(20%)',
    '  - Final Push: 9 dominios, 18M docs, 4.8 GB',
    '  - ALMA v3 + Knowledge v2 + Gutenberg v2: 137GB',
    '  - Classical Liberal: 20 autores (Aristoteles a Olavo)',
    '  - arXiv: 10 dominios matematicos, 50 docs cada',
    '  - Coelho-Sampaio: 82 classicos brasileiros',
    '  - Brazilian Science: math, physics, CS, neuroscience',
]: pdf.txt(l)
pdf.ln(3)
pdf.add_page()

# ══════ LINHA DO TEMPO ══════
pdf.sec('LINHA DO TEMPO COMPLETA (30/06 a 07/07/2026)')

tl = [
    ('30/06/2026', 'MANHA -- Nascimento do Projeto', [
        '10:14 - BOOTSTRAP: Semente F51 Darwin SSD inicializada',
        '11:09 - Scaffold de treino base + firewall de contaminacao de dados',
        '11:43 - Tokenizer e atualizacao de arquivos do projeto',
        '14:30 - Curadoria do corpus seed F51, treino do tokenizer, validacao do pipeline base',
    ]),
    ('30/06/2026', 'TARDE/NOITE -- Organismo Neural', [
        '18:07 - Ignore de artefatos runtime e agent bus',
        '18:29 - Brainstem: homeostasis (VRAM, loss, caps) + Grounded Extractor',
        '18:30 - Wire do organismo orquestrador nos exports do modelo',
        '18:32 - Neural organism runner + config',
        '18:40 - Wolfram Bridge: computacao externa (nao e o cerebro)',
        '18:52 - Semente de identidade F51, Olavo Curator, exercito de agentes, doutrina',
        '19:06 - Gerador procedural de corpus sintetico em portugues',
        '20:54 - LineageTracker: registro fossil com arvore genealogica dos modulos',
        '21:12 - Server: abort handling corrigido, serve_f51.py pronto para producao',
    ]),
    ('01/07/2026', 'MADRUGADA/MANHA -- Stack Completo', [
        '05:54 - FastAPI inference server para serving local do modelo',
        '06:12 - Configuracao Wolfram Bridge no organism config',
        '06:15 - Gerador procedural de corpus de identidade F51',
        '08:04 - Stack MoE completo, loop organism 24/7, soul engine (dopamina, FamilyCore)',
        '08:23 - MERGE: Pull Request #1 -- MoE stack + organism 247 + soul engine',
        '09:41 - README bilingue + pesquisa de fontes gratuitas do Olavo de Carvalho',
        '11:14 - KV Cache, Math Genius, Soul Engine, MoE, Biblia, Filosofos, Identity v2',
        '11:19 - CUDA turbo: TF32, cuDNN benchmark, correcoes KV cache + server',
        '11:32 - KV Cache: 7 bugs corrigidos, 35/35 testes passando',
        '11:34 - Server use_cache=False -- legacy mais rapido para modelo 27M',
        '11:49 - README bilingue completo com estado do projeto',
    ]),
    ('01/07/2026', 'TARDE -- Explosao de Features', [
        '19:08 - Cleanup + cloud infra scripts + corpus tools',
        '19:52 - Full system reference (515 linhas, 12 capitulos) + corpus balancer v2',
        '19:52 - Math Wolfram generator + kv_cache fix + cloud sync daemon + light checkpoints',
        '19:55 - F51 Expert Cache: hashing robusto + Nitro reference docs',
        '20:04 - Decision Engine + Evolution Loop: ciclo CDF de projetos externos',
        '20:07 - F51 Expert Cache v2: hashing vetorizado em GPU, 97% hit rate',
        '20:38 - Model Growth Engine: seed 27M -> grown 64.5M (1.9x crescimento)',
        '21:02 - Math-weighted corpus sampling: matematica pura domina o treino',
        '21:12 - Guia de aluguel GPU + script auto-deploy para treino matematico',
        '21:46 - Generic corpus generator: 2M documentos em 39 segundos',
        '21:59 - Conservative Canon corpus: 765k docs, 31 pensadores, enxame de agentes',
    ]),
    ('02/07/2026', 'MANHA/TARDE -- Dia Mais Produtivo (22 commits)', [
        '09:13 - Save all: math weights, corpus gen, training scripts, conservative canon',
        '09:32 - model.py updates + historical cloud-scale config work',
        '10:18 - Python 50M synthetic corpus (autoria Claude)',
        '10:32 - F51 Maquinista: orquestrador de treino em nuvem',
        '10:46 - CuriosityDrive: o organismo explora, nao otimiza (curiosidade intrinseca)',
        '10:57 - Auto-restart training loop + maquinista + novos geradores de corpus',
        '11:03 - Wolfram Mass Extractor: suga matematica sem pena',
        '11:33 - Math Overdrive: 10M docs, 4.3 GB de matematica sintetica',
        '11:46 - 4 Math Agents: 20M docs paralelos, 8.4 GB total',
        '11:51 - 8 Agents (4 Biologia + 4 Fisica Quantica): 20M docs',
        '11:54 - ALMA v3 + Knowledge v2 + Gutenberg v2: 137GB corpus',
        '11:59 - Financial Corpus Top 10: 10M docs, 4.1 GB',
        '12:05 - Ghost Corpus + Ghost Brain: auto-evolucao com dopamina e erro',
        '14:10 - Train Lord: checkpoint horario + auto-restart + fallback',
        '14:43 - Corpus Fantasma: 151GB split TRAIN(80%)/GHOST(20%)',
        '15:19 - tokenize_to_file fix + maquinista + ghost brain: pre-tokenization launch',
        '15:22 - Final Push: 9 novos dominios, 18M docs, 4.8 GB',
        '15:26 - train_cloud.py agora nasce 20B MoE por padrao',
        '19:52 - tokenize_to_file streaming para corpora >100GB + multiprocessing',
        '22:29 - oneshot_train.py + final state: estrategia 1B',
        '22:42 - AUTO-GROWTH: modelo cresce a cada 500 steps',
        '23:33 - 8B MoE fp16 fix + final deploy: let it run',
        '23:42 - Spider-Sense: o modelo SABE quando NAO SABE (metacognicao)',
    ]),
    ('03/07/2026', 'MADRUGADA/MANHA -- Estabilizacao', [
        '00:05 - SpiderSense auto-growth: modelo cresce por NECESSIDADE',
        '06:56 - Carrega tokens do disco (tokens.bin) sem OOM',
        '07:40 - ExpertWeightRouter: experts sobem/descem por MERITO (Super Ezio)',
        '07:43 - ExpertWeightRouter: danca das cadeiras por MERITO',
        '12:47 - Ghost Token NaN fix: clamp logits [-15,15] evita overflow fp16',
        '13:03 - Save all: ghost token NaN fix + 1.9B MoE + light corpus',
    ]),
    ('03/07/2026', 'NOITE -- Unificacao', [
        '23:19 - Auditoria tecnica completa: READ-ONLY, 11 secoes, gaps de validacao',
        '23:21 - UNIFIED TRAIN: todos os orgaos integrados (MARCO HISTORICO)',
        '23:49 - VRAM property fix no unified_train + script tokenize_cloud',
    ]),
    ('04/07/2026', 'MADRUGADA -- Otimizacoes Criticas', [
        '00:37 - Modelo F51-Darwin-9B-MoE configurado + fix moe_layer dtype mixed precision',
        '00:38 - Perf: np.fromfile para ler token binary, economizando 24.5GB RAM',
        '00:45 - Debug: print de parametros com NaN gradients no step 0',
        '00:46 - nan_to_num gradient shielding: prevencao de explosao NaN/Inf',
        '00:53 - CPUOffloadedAdamW: wrapper de otimizador que protege VRAM',
    ]),
    ('04/07/2026', 'MANHA/TARDE -- Madurez do Projeto', [
        '10:41 - Parallel associative scan no SelectiveSSM, modelo escala a 5.5B seguro',
        '10:41 - Priority-1 research rule no AGENTS.md',
        '10:44 - Property setters no CPUOffloadedAdamW para evitar AttributeError',
        '10:52 - MANIFESTO FUNDADOR: tatuado no projeto',
        '10:55 - AGENTS.md: hierarquia de agentes e dinamica de trabalho',
        '11:04 - Perf: otimizacao CPUOffloadedAdamW step + psutil RSS tracking',
        '11:33 - Suporte a Adafactor nativo GPU no unified_train.py',
        '11:35 - Ajuste de parametros Adafactor para PyTorch 2.6 nativo',
        '13:13 - auto_pipeline.py + tokenize_cloud.py multi-processing',
        '17:01 - Ghost config, training scripts, MoE debug (WIP)',
        '19:38 - Codex MoE init + Ghost Token + AMP + router observability + fast tokenizer',
        '21:39 - Fast tokenizer, corpus downloads (Gutenberg/arXiv/Coelho-Sampaio)',
        '21:39 - Dockerfile, deploy script, 9B fixes (Codex)',
        '21:39 - Build artifacts cleanup',
        '23:05 - Deep cleanup: checkpoints corrompidos, logs quebrados, scripts duplicados',
        '23:07 - Estado atual documentado',
    ]),
    ('05/07/2026', '-- Darwin-X e Memoria de Longo Prazo', [
        '12:10 - 7-layer legacy memory system implementado',
        '12:15 - Memory optimizations + legacy layers bugfix',
        '12:18 - Organism A100 runner + unified 1.9B config',
        '14:00 - Organism com 5 predictors + memmap + 7 legacy layers + 1.5B local config',
        '14:30 - Organism text-first corpus + local 0.8B/1.2B configs + LR=1e-3 default',
        '21:45 - Legacy torch.save para USB SSD + Darwin-X architecture doc + JEPA comparison',
        '21:55 - Darwin-X 4B: DeepSeekMoE + GQA + MTP + RoPE 1M + Ghost + Spider + JEPA',
    ]),
    ('06/07/2026', 'MADRUGADA -- Evolution Engine', [
        '00:31 - Evolution engine: generate, verify, accept/reject, scars, replay',
        '01:11 - Gate 6 + train_cloud crashes resolvidos',
        '01:26 - Legacy checkpoint converter (conversao de checkpoints antigos)',
        '01:32 - MoE conversion no legacy checkpoint converter',
    ]),
    ('06/07/2026', 'MANHA -- Pipeline e Arquitetura Final', [
        '09:32 - Nightly: agents, upload scripts, train commands, evolution report',
        '10:31 - RTX 5090 BC deploy: corpus upload + Darwin-X pronto (SPRINT)',
        '10:42 - Evolution system docs: sandbox, gate, probe, checkpoint eval',
        '10:53 - Organ pipeline: Ghost->Spider->JEPA->Curiosity->Consensus->EvoGate->Legacy',
        '11:03 - Checkpoint: organ pipeline + evolution system completo',
        '11:49 - Math organ: local arithmetic verifier + organ pipeline tests',
        '12:07 - Memory layer: 3-tier trainable memory (episodic/semantic/longterm) inside model',
        '12:09 - Final architecture doc: memory layer + organ pipeline + Darwin-X',
    ]),
    ('06/07/2026', 'TARDE -- Debug e Auto-Train', [
        '14:06 - Agent docs + compression + RTX 5090 staging (checkpoint)',
        '16:03 - Darwin-X NaN investigation + dense fallback attempt (debug)',
        '16:24 - Codex Darwin-X fixes + training scripts + testes (correcoes)',
        '20:29 - Auto-train pipeline: espera 60min, aluga GPU, SCP tokens, treina',
    ]),
    ('07/07/2026', 'HOJE -- Anti-Woke + Corpus Factory', [
        '00:03 - ANTI-WOKE FILTER v2: 12 padroes, 23 fontes bloqueadas (COMMIT MAIS RECENTE)',
        '??:?? - Corpus Factory Operating System (26 arquivos novos, NAO commitados)',
        '??:?? - Pipeline de ingestao remota: build -> verify -> upload -> ingest -> tokenize',
        '??:?? -   -> verify tokens -> build index -> train (8 etapas, 6 gates)',
    ]),
]

for date, period, events in tl:
    pdf.hdr(f'{date}  [{period}]', 11)
    for e in events:
        pdf.txt(f'    {e}', 8)
    pdf.ln(2)

pdf.add_page()

# ══════ ARQUITETURA ══════
pdf.sec('ARQUITETURA DO SISTEMA')

pdf.hdr('Modelo Causal -- F51 Darwin (Seed Base)', 11)
pdf.txt('  Token Embedding -> [8 blocos: 6x SSD + 2x Attention] -> RMSNorm -> LM Head (weight tying)', font='Courier')
pdf.txt('  SSD (Selective SSM + SwiGLU FFN): layers 0,1,2,4,5,6')
pdf.txt('  Causal Attention + RoPE + SwiGLU FFN: layers 3,7')
pdf.txt('  d_model=384 - n_layers=8 - n_heads=6 - ctx=1024 - vocab=16000 - ~27.45M params')
pdf.ln(2)

pdf.hdr('Darwin-X 4B (Modelo Evolutivo)', 11)
for l in [
    '  - DeepSeekMoE: Mixture of Experts com 8 experts de dominio',
    '  - GQA (Grouped Query Attention): atencao agrupada eficiente',
    '  - MTP (Multi-Token Prediction): predicao de multiplos tokens',
    '  - RoPE 1M: Rotary Position Embedding com contexto de 1M tokens',
    '  - Ghost: modulo de auto-evolucao com dopamina e erro',
    '  - Spider: metacognicao -- modelo sabe quando nao sabe',
    '  - JEPA: Joint Embedding Predictive Architecture',
    '  - Memory Layer: 3-tier trainable (episodic/semantic/longterm) interno',
    '  - ExpertWeightRouter: experts sobem/descem por merito dinamico',
]: pdf.txt(l)
pdf.ln(2)

pdf.hdr('Sete Orgaos Neurais (Metafora Biologica)', 11)
for l in [
    '  Cortex     (F51DarwinModel)       -- Motor de linguagem principal',
    '  Talamo     (DynamicDepthRouter)    -- Roteamento de profundidade (placeholder)',
    '  Hipocampo  (ReplayBuffer+Ledger)   -- Memoria episodica + trilha de auditoria',
    '  Amigdala   (DataFirewall+Factory)  -- Firewall de contaminacao de dados',
    '  Ganglios   (ExpertPool+pruning)    -- Ciclo de vida: candidate->active->frozen->dead',
    '  Cerebelo   (src/tests/)                -- Verificacao: testes nao mentem',
    '  Tronco     (Brainstem)             -- Homeostase: VRAM, loss, caps, budget',
]: pdf.txt(l)
pdf.ln(2)

pdf.hdr('Organ Pipeline (Fluxo de Evolucao)', 11)
pdf.txt('  Ghost -> Spider -> JEPA -> Curiosity -> Consensus -> EvoGate -> Legacy', font='Courier')
pdf.txt('  Cada etapa alimenta a proxima. O pipeline e um ciclo fechado de evolucao continua.')
pdf.ln(2)

pdf.hdr('Sistema Imune de Dados', 11)
pdf.txt('  generate/import -> candidates -> audit (firewall) -> quarantine/rejected/approved', font='Courier')
pdf.txt('      approved -> promote_approved_data.py -> data/corpus/ -> train', font='Courier')
pdf.txt('  Zonas: candidates (never) - quarantine (never) - rejected (never) - approved (promoted) - corpus (yes)')
pdf.ln(2)

pdf.hdr('Soul Engine + Identidade', 11)
pdf.txt('  Dopamina/XP no aprendizado, competitive drive, FamilyCore (cla Barreto), proposito e mantras.')
pdf.txt('  Nao e prompt injection -- e camada de identidade para o loop do organismo.')
pdf.ln(3)

pdf.hdr('Componentes Avancados', 11)
for l in [
    '  CuriosityDrive:        Exploracao intrinseca -- o organismo explora, nao otimiza cegamente',
    '  Spider-Sense:           Metacognicao -- modelo SABE quando NAO SABE',
    '  Auto-Growth:            Modelo cresce a cada 500 steps por NECESSIDADE demonstrada',
    '  Ghost Corpus + Brain:   Auto-evolucao com dopamina e erro -- corpus fantasma 151GB',
    '  Train Lord:             Checkpoint horario + auto-restart + fallback',
    '  F51 Maquinista:         Orquestrador de treino em nuvem com aluguel de GPU',
    '  ExpertWeightRouter:     Experts sobem/descem por MERITO (danca das cadeiras)',
    '  Evolution Engine:       Generate, verify, accept/reject, scars, replay',
    '  Wolfram Bridge:         Computacao matematica externa (NAO e o cerebro do modelo)',
    '  KV Cache:               O(1) por token apos prefill -- 7 bugs corrigidos',
    '  CPUOffloadedAdamW:      Otimizador com offload para CPU protegendo VRAM',
    '  Adafactor GPU nativo:   Suporte nativo PyTorch 2.6',
    '  Anti-Woke Filter v2:    12 padroes de deteccao, 23 fontes bloqueadas',
    '  LineageTracker:         Registro fossil com arvore genealogica dos modulos',
    '  Model Growth Engine:    Seed 27M -> grown 64.5M (1.9x)',
    '  Math Organ:             Verificador aritmetico local -- matematica auditavel',
    '  F51 Expert Cache v2:    Hashing vetorizado GPU, 97% hit rate',
]: pdf.txt(l)

pdf.add_page()

# ══════ CATALOGO DE CORPUS ══════
pdf.sec('CATALOGO COMPLETO DE CORPUS')

corpus_data = [
    ('MATEMATICA (Prioridade Maxima)', [
        '4 Math Agents: 20M documentos paralelos, 8.4 GB total',
        'Math Overdrive: 10M docs, 4.3 GB matematica sintetica',
        'Wolfram Mass Extractor: succao massiva sem pena',
        'arXiv -- 10 dominios (50 docs cada):',
        '  algebraic_geometry, topology, number_theory, quantum_field_theory,',
        '  particle_physics, category_theory, functional_analysis,',
        '  general_relativity, nuclear_physics, quantum_computing',
        'Math-weighted corpus sampling: matematica domina o treino (peso aumentado)',
        'Math organ: verificador aritmetico local integrado ao modelo',
        'Lean mathlib4 + ProofWiki (fontes P0 planejadas)',
        'Math Wolfram generator: geracao sintetica de problemas matematicos',
    ]),
    ('CIENCIAS NATURAIS', [
        '8 Agents (4 Biologia + 4 Fisica Quantica): 20M docs',
        'Biologia: cellular (50), molecular (50), genetics (50), ecology (50)',
        'Astronomia/Fisica: astronomy_physics corpus',
        'Biologia Evolutiva: biology_evolution corpus',
        'Brazilian Science: math (30), physics (30), CS (30), neuroscience (20)',
        'Brasil Cancer Research: cancer-specific corpus',
        'OpenStax math/science textbooks (fonte P0 planejada)',
        'PMC Open Access commercial-use-allowed (fonte P0)',
    ]),
    ('CORPUS CLASSICO E FILOSOFICO', [
        'Conservative Canon: 765k docs, 31 pensadores conservadores/classicos',
        'Corpus Classico Liberal -- 20 Autores:',
        '  Aristoteles, Cicero, Tomas de Aquino, Hume, Adam Smith,',
        '  Edmund Burke, Tocqueville, Mises, Hayek, Oakeshott,',
        '  Russell Kirk, Milton Friedman, James Buchanan, Roger Scruton,',
        '  Thomas Sowell, Solzhenitsyn, Gomez Davila, Roberto Campos,',
        '  Olavo de Carvalho, George Grant',
        'Olavo de Carvalho:',
        '  - Fontes gratuitas catalogadas (OLAVO_FREE_SOURCES.md)',
        '  - Download de artigos oficiais do site (50+ paginas)',
        '  - Curador automatico: catalogacao e verificacao',
        '  - Quarentena: revisao manual P0 antes de promover',
        '  - Sampling weight: identity x5, Olavo x3 no treino',
        'Coelho-Sampaio: 82 classicos da literatura brasileira',
        'Economics Right: pensamento economico liberal/conservador',
    ]),
    ('CORPUS GERAL E ESPECIALIZADO', [
        'Final Push -- 9 dominios (40 docs cada):',
        '  medicine, military, law, psychology, chemistry,',
        '  brasil, logic, engineering, religion',
        '  Total: 18M docs, 4.8 GB',
        'Financial Corpus Top 10: 10M docs, 4.1 GB',
        'ALMA v3 + Knowledge v2 + Gutenberg v2: 137GB corpus total',
        'Python 50M synthetic corpus (autoria Claude, em quarentena)',
        'Generic corpus generator: 2M documentos em 39 segundos',
        'Identity seed F51: documentos de identidade (criador, familia, doutrina, Deus)',
        '  - Approved by creator decision -- not generic synthetic spam',
        '  - Sampling: identity x5, family x4',
        'Corpus Fantasma: 151GB split TRAIN(80%) / GHOST(20%)',
        'Brasil Experimental: 20 documentos exploratorios',
    ]),
    ('INFRAESTRUTURA DE DADOS', [
        'Data Firewall: candidates -> audit -> quarantine/rejected/approved',
        'Sistema Imune: dado sintetico NUNCA e treino por padrao',
        'Provenance Ledger: trilha de auditoria para cada documento',
        'Replay Buffer: desde o step 1 do treino (sem catastrofico esquecimento)',
        'Data contamination firewall: deteccao de duplicatas e sobreposicao',
        'Corpus Factory OS: pipeline completo de fabrica infinita',
        'Tokenizer: f51_bpe (16k vocab) - f51_bpe_80k (80k planejado)',
        'Sampling weights config: identity x5 - family x4 - Olavo x3',
        'Cloud master path: /workspace/f51_corpus_factory/',
        'Token format: .int32.bin + .tokens.json + index.json',
    ]),
]

for title, items in corpus_data:
    pdf.hdr(title, 11)
    for item in items:
        pdf.txt(f'  - {item}')
    pdf.ln(2)

pdf.add_page()

# ══════ DOUTRINA ══════
pdf.sec('DOUTRINA E PRINCIPIOS FUNDADORES')

pdf.hdr('Manifesto Fundador (04/07/2026)', 12)
for f in [
    '"Growth without pruning is neural obesity."',
    '   "Crescimento sem poda e obesidade neural."',
    '',
    '"Learn today without destroying yesterday."',
    '   "Aprender hoje sem destruir ontem."',
    '',
    '"Synthetic data without provenance is infection."',
    '   "Dado sintetico sem proveniencia e infeccao."',
    '',
    '"A model without tests is hallucination with weights."',
    '   "Modelo sem teste e alucinacao com pesos."',
]:
    pdf.txt(f, italic=True)
pdf.ln(4)

pdf.hdr('As 10 Leis Internas (Leis de F51)', 12)
leis = [
    ('1. No external checkpoint / Sem checkpoint externo',
     '   100% pesos aleatorios. Zero contaminacao de LLMs estrangeiros.'),
    ('2. No foreign tokenizer / Sem tokenizador estrangeiro',
     '   BPE byte-level proprio. Sem SentencePiece de Llama/GPT.'),
    ('3. No synthetic data directly in training / Sem sintetico direto no treino',
     '   Todo dado sintetico passa por quarentena e aprovacao manual.'),
    ('4. No growth without ablation / Sem crescimento sem ablacao',
     '   Modulo so cresce se demonstrar utilidade estatistica.'),
    ('5. No living module without utility / Sem modulo vivo sem utilidade',
     '   Modulo sem evidencia de contribuicao -> frozen -> dead.'),
    ('6. No death without quarantine / Sem morte sem quarentena',
     '   Modulo nunca e deletado diretamente. Vai para quarentena.'),
    ('7. No memory without provenance / Sem memoria sem proveniencia',
     '   Toda informacao no modelo tem trilha de origem.'),
    ('8. No truth without test / Sem verdade sem teste',
     '   Cerebelo (src/tests/) e o orgao da verdade. 35 testes, 100% pass.'),
    ('9. No training without replay / Sem treino sem replay',
     '   Replay desde step 1. Nada e esquecido catastroficamente.'),
    ('10. No scaling without stable seed / Sem escala sem seed estavel',
     '   Seed 27M validada antes de qualquer escala.'),
]
for title, desc in leis:
    pdf.txt(title, bold=True)
    pdf.txt(desc)
    pdf.ln(4)
pdf.ln(2)

pdf.hdr('Hierarquia de Agentes (AGENTS.md)', 12)
for l in [
    'Antigravity (Arquiteto/Socio): Comanda pesquisas, desenha estrategias, define arquitetura.',
    'Subagente Deep (Operario): Trabalhador incansavel, escreve codigos e matematica fina.',
    '',
    'Regras Absolutas:',
    '  1. SEMPRE pesquise e simule antes de decidir (propagar a todos os agentes)',
    '  2. Toda afirmacao precisa de evidencia (arquivo+linha, comando, teste, fonte)',
    '  3. Nunca escrever em checkpoints/, runs/, data/, tokenizer/, .git/',
    '  4. Nunca commitar segredo, modelo, dataset cru, foto privada',
    '  5. Builder so edita arquivos do ticket atribuido',
    '  6. Verifier NUNCA e o mesmo agente que escreveu o codigo',
    '  7. Relatorio sem evidencia = REJEITADO',
    '',
    'Gates obrigatorios antes de qualquer commit:',
    '  git status --short && git diff --check',
    '  python -m pytest -q',
    '  python src/scripts/darwin_inventory.py',
    '  python src/tools/inspect_seed.py',
    '',
    'Pesos aleatorios. Cresce por incapacidade. Modulo vive ou morre por evidencia.',
    'Aprender hoje sem destruir ontem.',
]: pdf.txt(l)

pdf.add_page()

# ══════ SCRIPTS ══════
pdf.sec('CATALOGO COMPLETO DE SCRIPTS E MODULOS')

modules_list = [
    ('src/scripts/', 'Treino e Core (8 scripts)', [
        'train_base.py            -- Treino base + resume de checkpoint',
        'train_tokenizer.py       -- Treinador BPE F51 (byte-level)',
        'train_darwin_x.py        -- Treino Darwin-X com fabrica de corpus',
        'organism_247.py          -- Loop de treino continuo 24/7',
        'auto_train_1h.sh         -- Pipeline: espera 60min, aluga GPU, SCP tokens, treina',
        'cloud_train_darwin_x_fixed.sh -- Treino Darwin-X na nuvem (fixo)',
        'rent_safe.sh             -- Aluguel seguro de GPU com verificacao',
        'auto_pipeline.py         -- Pipeline automatico multi-processing',
    ]),
    ('src/scripts/', 'Corpus Factory (9 scripts)', [
        'build_corpus_pack.py           -- Constroi pacote de corpus aprovado (manifest + tar.gz)',
        'verify_corpus_pack.py          -- Verifica integridade do pacote antes do upload',
        'upload_corpus_pack.py          -- Upload do pacote para o master na nuvem',
        'ingest_remote_corpus_pack.py   -- Ingestao do pacote no master (/workspace)',
        'tokenize_remote_corpus_batch.py -- Tokenizacao remota do batch ingerido',
        'verify_remote_token_batch.py   -- Verificacao de integridade dos tokens remotos',
        'build_remote_token_index.py    -- Constroi index.json de todos os batches',
        'download_olavo_sources.py      -- Download de fontes gratuitas do Olavo',
        'download_classical_corpus.py   -- Download do corpus classico liberal (20 autores)',
    ]),
    ('src/scripts/', 'Inferencia e Utilidades (6 scripts)', [
        'serve_f51.py             -- Server HTTP stdlib + console web em portugues',
        'serve_model.py           -- FastAPI inference server (API REST)',
        'darwin_inventory.py      -- Inventario completo de ambiente e dependencias',
        'inspect_seed.py          -- Inspecao da seed (contagem params, CUDA smoke)',
        'prepare_base_training.py -- Relatorio de prontidao para treino',
        'run_neural_organism.py   -- CLI do organismo neural (bootstrap, cycle, status)',
    ]),
    ('src/f51_darwin/', 'Core do Modelo (7 modulos)', [
        'model.py                  -- F51DarwinModel: motor de linguagem principal',
        'ssm_core.py               -- Selective SSM scan (sequencial, puro PyTorch)',
        'darwin_x_training.py      -- Loop de treino Darwin-X com todos os orgaos',
        'moe_layer.py              -- Mixture-of-Experts layer com roteamento',
        'moe_training.py           -- CLI de treino MoE com estatisticas',
        'kv_cache.py               -- Geracao autoregressiva com KV cache (O(1)/token)',
        'turbo.py                  -- CUDA turbo generate (caminho otimizado)',
    ]),
    ('src/f51_darwin/', 'Organismo e Evolucao (12 modulos)', [
        'organism.py               -- Orquestrador de ciclo de vida completo',
        'brainstem.py              -- Homeostase: VRAM, loss plateau, module budget',
        'soul.py                   -- Dopamina, XP, FamilyCore, proposito, mantras',
        'grounded_extractor.py     -- Extracao de conhecimento grounded no projeto',
        'wolfram_bridge.py         -- Ponte para Wolfram (calculo externo, nao cerebro)',
        'math_genius.py            -- Camada auxiliar simbolica/numerica',
        'curiosity.py              -- CuriosityDrive: exploracao intrinseca',
        'spider_sense.py           -- Spider-Sense: metacognicao (incerteza)',
        'auto_growth.py            -- Auto-Growth: expansao por necessidade',
        'evolution_engine.py       -- Motor de evolucao: generate, verify, accept/reject',
        'ghost_corpus.py           -- Ghost Corpus + Ghost Brain: auto-evolucao',
        'train_lord.py             -- Train Lord: checkpoint horario + auto-restart',
    ]),
    ('src/f51_darwin/', 'Corpus Factory Modules (8 modulos)', [
        'corpus_factory_pipeline.py  -- Pipeline completo da fabrica de corpus',
        'corpus_policy.py            -- Motor de politicas (6 gates de qualidade)',
        'corpus_pack_upload.py       -- Upload com verificacao e .sha256',
        'corpus_pack_verifier.py     -- Verificador de integridade de pacotes',
        'corpus_cloud_ingest.py      -- Ingestao remota na nuvem com validacao',
        'corpus_cloud_tokenize.py    -- Tokenizacao remota com verificacao de vocab',
        'corpus_cloud_token_index.py -- Construcao de indice de tokens para treino',
        'corpus_cloud_token_verify.py -- Verificacao de integridade de tokens',
    ]),
    ('src/f51_darwin/', 'Identidade e Dados (7 modulos)', [
        'identity_corpus.py        -- Gerador procedural de corpus de identidade F51',
        'olavo_curator.py          -- Curador automatico de fontes do Olavo',
        'data.py                   -- Pipeline de dados + fabrica de dados',
        'data_factory.py           -- Fabrica de dados com firewall e zonas',
        'data_firewall.py          -- Firewall de contaminacao (auditoria)',
        'anti_woke_filter.py       -- Filtro anti-woke v2: 12 padroes, 23 fontes',
        'lineage_tracker.py        -- LineageTracker: registro fossil dos modulos',
    ]),
]

for dir_name, section, items in modules_list:
    pdf.hdr(f'{dir_name} -- {section}', 10)
    for item in items:
        pdf.txt(f'  {item}', 7.5, font='Courier')
    pdf.ln(2)

pdf.add_page()

# ══════ FICHA TECNICA ══════
pdf.sec('FICHA TECNICA COMPLETA')

tech = [
    ('HARDWARE E AMBIENTE', [
        'GPU testada:          NVIDIA RTX 5060 Ti (~17.1 GB VRAM)',
        'GPU alvo cloud:       NVIDIA RTX 5090 / A100',
        'PyTorch:              2.8.0+cu129',
        'CUDA:                 TF32 habilitado, cuDNN benchmark ativo',
        'Python:               3.10+',
        'Sistema:              Windows (dev local) + Linux (cloud master)',
        'Armazenamento:        USB SSD para checkpoints legados',
    ]),
    ('PERFORMANCE', [
        'batch 4 x block 1024:  ~8 min/step (SSM sequencial -- deliberado)',
        'batch 4 x block 256:   ~1-2 s/step',
        'batch 2 x block 64:    ~10 s/step (smoke)',
        'Treino base:           step 15.000 atingido localmente',
        'RAM saving:            24.5GB economizados com np.fromfile',
        'Token streaming:       suporte para corpora >100GB',
        'KV Cache:              O(1) por token apos prefill',
        'Expert Cache v2:       97% hit rate com hashing GPU vetorizado',
        'Multi-processing:      tokenize_cloud + auto_pipeline',
    ]),
    ('MODELOS E CONFIGURACOES', [
        'Seed 27M:    d=384, 8 layers, 6 heads, ctx=1024, vocab=16k',
        'MoE 140M:    8 experts, domain routing, identity weighted x5',
        '1.5B local:  5 predictors + memmap + 7 legacy layers',
        '1.9B unified: organism A100 runner',
        '4B Darwin-X: DeepSeekMoE + GQA + MTP + RoPE 1M + Ghost + Spider + JEPA',
        '5.5B safe:   parallel associative scan + model scaling seguro',
        '9B MoE:      Codex + mixed precision + fast tokenizer',
        '20B MoE:     historical plan, superseded by Darwin-X Law 0',
    ]),
    ('SEGURANCA E ESTABILIDADE', [
        'NaN shielding:        nan_to_num gradient protection',
        'Ghost Token NaN fix:  clamp logits [-15, 15] evita overflow fp16',
        'CPUOffloadedAdamW:    VRAM protection via CPU offloading',
        'Adafactor nativo:     suporte GPU via PyTorch 2.6',
        'Gradient NaN debug:   print de parametros problematicos no step 0',
        'Mixed precision:      moe_layer dtype bug corrigido',
        'KV Cache fix:         7 bugs corrigidos, 35/35 testes',
        'Gate 6 crash:         resolvido',
        'train_cloud crash:    resolvido',
    ]),
    ('TESTES', [
        '35 testes unitarios (100% pass):',
        '  - config - forward pass - SSM scan - KV cache',
        '  - data firewall - tokenizer - checkpoint resume',
        '  - evolution - replay - pruning',
        '6 novos testes de corpus factory (nao commitados):',
        '  - corpus_cloud_ingest - corpus_cloud_token_index',
        '  - corpus_cloud_token_verify - corpus_cloud_tokenize',
        '  - corpus_factory_pipeline - corpus_pack_upload - corpus_policy',
        'Novo teste: test_darwin_x_training.py (+99 linhas, unstaged)',
    ]),
]

for title, items in tech:
    pdf.hdr(title, 11)
    for item in items:
        pdf.txt(item)
    pdf.ln(2)

# ══════ CORPUS FACTORY PIPELINE ══════
pdf.sec('CORPUS FACTORY -- PIPELINE COMPLETO')

for l in [
    'O Corpus Factory Operating System e uma fabrica infinita de corpus para Darwin-X',
    'que preserva ciencia e matematica como espinha dorsal do treino, com proveniencia',
    'explicita, licenca, hash, URL fonte e estado de revisao para cada item.',
    '',
    'ESTRUTURA DE DIRETORIOS:',
    '  LOCAL (Windows):  C:\\Users\\marco\\Desktop\\F51-Corpus-Factory\\',
    '    raw/  curated/  normalized/  manifests/  packs/',
    '  CLOUD (Linux):  /workspace/f51_corpus_factory/',
    '    incoming/  raw/  normalized/  manifests/  tokenizer/f51_bpe_80k/  tokens/80k/',
    '',
    'FLUXO OPERACIONAL (8 etapas):',
    '  1. BUILD:   python src/tools/build_corpus_pack.py',
    '              -> manifest.jsonl + approved/*.txt + approved_corpus_pack.tar.gz',
    '  2. VERIFY:  python src/tools/verify_corpus_pack.py <pack.tar.gz>',
    '              -> Falha se: missing files, extra files, hash mismatch, duplicate hashes',
    '  3. UPLOAD:  python src/scripts/upload_corpus_pack.py <pack.tar.gz> --target master --batch-name X',
    '              -> Verifica local primeiro, recusa invalidos, upload + .sha256',
    '  4. INGEST:  python src/scripts/ingest_remote_corpus_pack.py --remote-pack <path> --batch-name X',
    '              -> Verifica .sha256, extrai, copia approved/ -> normalized/{batch}/',
    '              -> Falha se batch ja ingerido ou contagem mismatch',
    '  5. TOKENIZE: python src/scripts/tokenize_remote_corpus_batch.py --batch-name X',
    '              -> Le normalized/{batch}/*.txt, tokeniza, escreve .int32.bin + .tokens.json',
    '              -> Valida todos os token IDs contra o vocabulario',
    '  6. VERIFY TOKENS: python src/scripts/verify_remote_token_batch.py --batch-name X',
    '              -> Verifica: .int32.bin existe, tamanho = tokens*4, SHA-256 match',
    '  7. INDEX:  python src/scripts/build_remote_token_index.py --target master',
    '              -> Scaneia todos .tokens.json, verifica .int32.bin, escreve index.json',
    '              -> Recusa missing, misaligned, wrong-sized, SHA-mismatched bins',
    '  8. TRAIN:  python research/train_darwin_x.py --token-index <index.json> --tokenizer <path>',
    '              -> Usa index com todos os batches verificados',
    '',
    'SEIS GATES DE QUALIDADE:',
    '  1. PROVENANCE:    source URL/path, license label, date, content hash',
    '  2. LICENSE:       public domain, CC0, CC-BY, CC-BY-SA, government work, approved OA',
    '  3. DOMAIN:        science, math, engineering, medicine, CS, right-thought track',
    '  4. QUALITY:       enough text, UTF-8 normalized, low boilerplate, no OCR corruption',
    '  5. ANTI-CONTAMINATION: no duplicate hash, no excessive n-gram overlap',
    '  6. POLITICAL-NOISE:    activism/editorial -> quarantined unless Marco explicitly approves',
    '',
    'PRIORIDADE DE FONTES:',
    '  P0: OpenStax - arXiv - PMC Open Access - Lean mathlib4 - ProofWiki - Gutenberg/Perseus/Wikisource',
    '  P1: OpenAlex - DOAJ - CORE - S2ORC (validados)',
    '  P2: Internet Archive - ASR transcripts - community uploads (quarentena default)',
    '  REJECT: pirate mirrors - PDF dumps - missing license - social/news outrage content',
    '',
    'LOOP INFINITO (fechado):',
    '  discover -> download/import -> normalize -> manifest -> policy gate',
    '      -> quarantine or approve -> pack -> upload -> tokenize -> train',
    '      -> eval errors -> ghost lessons -> candidate queue (volta ao inicio)',
]:
    pdf.txt(l)

# ── Rodape ──
pdf.ln(8)
pdf.foot('-- Soli Deo Gloria --', bold=True, size=14)
pdf.set_text_color(0, 0, 0)
pdf.foot('Projeto F51 Darwin-SSD - Marco Barreto - Fuch F51 Labs - Montreal, QC - 2026', size=9)
pdf.foot('Branch: cursor/bilingual-readme-olavo-sources - 109 commits - 8 dias de construcao intensiva', size=9)
pdf.foot('ALERTA: 6 arquivos modificados (unstaged) + 26 arquivos novos (untracked) + 21 commits ahead of origin', size=9)

output_path = r'C:\Users\marco\Desktop\F51_Darwin_SSD_Linha_do_Tempo.pdf'
pdf.output(output_path)
print(f'PDF GERADO COM SUCESSO: {output_path}')
print(f'Total de paginas: {pdf.pages_count}')
