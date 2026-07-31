# F51 Darwin-SSD — Fraquezas em Dados & Tokenizer

**Autor:** Agente de Pesquisa (Dados & Tokenizer)
**Data:** 2026-07-15
**Escopo:** `scripts/ingest_pipeline.py`, `scripts/auto_ingest.py`, `f51_darwin/data_firewall.py`, `f51_darwin/tokenizer.py`, `f51_darwin/provenance.py`, `f51_darwin/data.py`, `tokenizer/f51_bpe_80k/`
**Objetivo ativo:** retomar linhagem 2.5B (não 1.76B com init aleatória)

---

## Sumário Executivo

O pipeline de dados atual funciona como um **teatro visual de 9 estágios**, mas a maior parte do trabalho real de curadoria é cosmética: o "misturador de fontes" não aplica pesos no sampling, o firewall mede apenas `alpha-ratio + length`, a deduplicação é SHA-256 exato (com fallback O(N²) de ngramas que só roda contra uma amostra de 500 MB), e o tokenizer BPE caseiro **não tem token de espaço-prefix**, forçando um patch frankenstein-regex no decode que destrói round-trip fidelity. Para um alvo de 2.5B params treinado do zero, esses problemas limitam diretamente a qualidade efetiva do corpus e a capacidade do modelo de aprender fronteiras de palavras sem desperdiçar capacidade. Há 5 fraquezas estruturais claras e 10 papers/sub-trabalhos com ideias diretamente aplicáveis.

---

## Parte 1 — Fraquezas Específicas no Pipeline Atual

### F1. Tokenizer BPE sem espaço-prefix (defeito estrutural, não ajuste)

**Evidência no código:**
- `tokenizer/f51_bpe_80k/config.json`: vocab efetivo 58.162, byte-level puro.
- `f51_darwin/tokenizer.py:118-133` (`encode`): o espaço é reinserido *manualmente* como byte `<b20>` id=36 **entre** palavras, condicionalmente.
- `f51_darwin/tokenizer.py:270-323` (`_reconstruct_spacing`): o decode aplica **8 passos de regex** incluindo hardcoded de palavras comuns em inglês (`"with"`, `"the"`, `"and"`, sufixos `ing`/`tion`) — ignorando完全 português.
- `tokenizer/f51_bpe_80k/vocab.json` (inspecionado): **nenhum merge** começa com `<b20>` como prefixo. Não existe equivalente de `Ġ` (GPT-2) ou `▁` (SentencePiece).

**Por que é grave:**
- Sem merges com espaço-prefix, tokens como `" the"` (top-1 token em BPEs típicos) **não existem** — o modelo gasta capacidade prevendo o byte de espaço explicitamente e aprendendo fronteiras que o tokenizer já deveria codificar.
- O decode frankenstein-regex (linhas 300-316, focado em inglês) **destrói round-trip** para PT-BR, matemática (`x²`, `αβγ`) e código (indentação). Todo texto decodeado é uma aproximação, nunca o original.
- Esse padrão é exatamente o que **todos** os tokenizers state-of-the-art (GPT-2/4, Llama, Mistral, Qwen, DeepSeek) evitam com `Ġ`/`▁`. O "padrão ouro" é mais de 6 anos antigo e foi abandonado por motivo.
- O plan em `tokenizer_plan.py` diz "v1_seed: Train F51-owned SentencePiece or byte-pair tokenizer" — a versão atual é a v1, e herdou a decisão errada.

**Custo:** aproximadamente 5-15% de tokens desperdiçados em texto natural (estimativa baseada em fertility de byte-only BPE vs BPE com space-prefix), além de perda de round-trip que inviabiliza uso do modelo para geração confiável.

---

### F2. Firewall com heurística ultrabásica (alpha-ratio + comprimento)

**Evidência no código:**
- `f51_darwin/data_firewall.py:50-56` (`score_text`):
  ```
  ratio = alpha / max(len(text), 1)
  length_bonus = min(len(text.strip()) / (min_chars * 4), 1.0)
  return round(min(1.0, 0.5 * ratio + 0.5 * length_bonus), 4)
  ```
- `scripts/ingest_pipeline.py:149`: `FirewallConfig(min_chars=200, quality_pass_score=0.60)` — mais permissivo que o default `0.70` do dataclass.
- **Nenhuma** das dimensões padrão da literatura está presente: sem perplexity (KenLM/small-LM), sem language ID (FastText), sem classifier-based quality (GPT-3-style "is this Wikipedia-like?"), sem detecção de PII (email/telefone/CPF), sem filtro de toxicidade, sem flag de boilerplate/SEO-spam, sem detecção de geração sintética低 quality.

**Por que é grave:**
- alpha-ratio aprova listas telefônicas, tabelas numéricas densas, código sem comentário, logs — tudo com `alpha ratio ≈ 0.6-0.7`. Reprova matemática densa (alpha cai para ~0.4 por símbolos), code, JSON.
- Para um modelo que **prioriza matemática** (vide `data.py:99-146` `document_sample_weight`), o firewall está estruturalmente enviesado **contra** o domínio mais valioso do projeto.
- `quality_pass_score=0.60` no pipeline vs `0.70` no dataclass: o operacional é mais frouxo que o default do código — provável drift silencioso.

**Insight negativo valioso:** Abbas & Khaiser (Dolma, 2024) e arXiv 2405.20541 mostram que **perplexity filtering tem retornos decrescentes** no regime over-trained — mas em **compute-optimal** (que é o regime do F51 2.5B), perplexity pruning dá ganho real. Logo, o F51 está abandonando justamente o filtro que mais ajuda no seu regime.

---

### F3. Deduplicação é cosmética (SHA-256 exato + ngrama O(N²) amostrado)

**Evidência no código:**
- `f51_darwin/provenance.py:18-20`: `content_hash` = `sha256(text.strip())` — **só** duplicatas exatas pós-strip. Não normaliza case, não colapsa whitespace interno, não remove pontuação.
- `f51_darwin/data_firewall.py:96-107`: `ngram_overlap` roda só contra `corpus_texts`, que vem de `_collect_known_texts` (`data_factory.py:265-296`) — esse método tem **cap de 500 MB** (`max_corpus_mb: float = 500.0`) e itera todos os `.txt` + `.json` sob `corpus/` e `approved/` a cada chamada.
- `f51_darwin/provenance.py:121-126` (`find_by_hash`): chama `latest_by_id()` que faz **parse JSON completo do ledger** a cada lookup. Em 100k documentos aprovados isso é O(N) por hash → O(N²) no registro.
- Threshold `max_ngram_overlap=0.85` é alto demais: near-dups típicos (re-publicação, mirror, leve reformulação) ficam em 0.6-0.8 e passam.

**Por que é grave:**
- Duplicata exata post-strip captura talvez 30% das duplicatas reais em corpus web (FineWeb/C4). Near-dups (boilerplate, templates, mirrors) entram todas.
- A abordagem MinHash+LSH (CCNet, BigCode, Dolma, RedPajama) é o **padrão ouro** há 5+ anos e não está presente. SemDeDup ( Abbas et al., 2023, Meta) vai além: remove pares **semanticamente** similares usando embeddings.
- Re-tokenizar o corpus inteiro a cada novo doc aprovado é O(N²) no crescimento do corpus.

---

### F4. Pipeline de ingestão re-tokeniza o corpus inteiro a cada doc aprovado

**Evidência no código:**
- `scripts/ingest_pipeline.py:255-280` (dentro de `process_text`, ramo APPROVED):
  ```python
  documents = load_text_documents(corpus_dir)          # lê TODOS os arquivos
  tokenizer = F51BPETokenizer.load(...)                # carrega tokenizer 3MB
  new_tokens = tokenize_documents(documents, tokenizer) # tokeniza TUDO de novo
  feast_sample = feast_tokens[-20_000_000:]
  max_new = int(len(feast_sample) * 0.10)
  if len(new_tokens) > max_new: new_tokens = new_tokens[:max_new]
  ```
- **Todo novo documento aprovado** dispara releitura + re-tokenização de **todo o `data/approved` + `data/corpus`**.

**Por que é grave:**
- Para 10k docs aprovados, cada ingestão nova custa O(N) releituras → O(N²) cumulativo. Inviável para o regime "run247" declarado no docstring.
- O `tokenizer.load()` (3 MB vocab JSON parsed) é refeito a cada doc.
- O cap `max_new = 10% de 20M = 2M tokens` trunca agressivamente — se o corpus cresce, a maioria dos tokens novos é descartada silenciosamente. Isso é um **bug de dados**: docs aprovados podem não entrar efetivamente no bin de treino.
- A "janela feast" (`feast_tokens[-20_000_000:]`) é fixa em 20M — sem peso por recência, sem re-sampling ponderado por qualidade.

---

### F5. Pesos de domínio são só display; viés identitário hardcoded no sampling real

**Evidência no código:**
- `scripts/ingest_pipeline.py:158-165` e `:226-228`: `self.domain_weights` é só renderizado em `render_weights()` (linhas 167-174) e lido como `weight = self.domain_weights.get(domain, 0.10)` puramente para print. **Não é aplicado** em nenhum sampling efetivo do `live_path`.
- `scripts/ingest_pipeline.py:320`: `domain = "Math" if "math" in f.name.lower() else "FineWeb"` — inferência de domínio por substring no nome do arquivo.
- `f51_darwin/data.py:99-146` (`document_sample_weight`): **aqui** sim há sampling real, mas com viés hardcoded extremo:
  - `"olavo" in name → 3.0`
  - `"identity" or "f51_identity" → 5.0`
  - `"familia"/"family"/"clan"/"barreto" → 4.0`
  - matemática também em 3.0-5.0
- Esses pesos viram `repeat = max(1, int(round(weight)))` (`data.py:163`) → oversampling literal (repetir o doc 5x).

**Por que é grave:**
- **Dois sistemas paralelos de "mixing"**: o que aparece na UI (cosmético, 6 domínios) e o que de fato roda (`document_sample_weight`, baseado em substring de nome de arquivo). Eles não conversam.
- Oversampling por repetição literal (`repeat=5`) é a estratégia **mais ingênua** possível e sabidamente prejudicial: vicia o modelo, multiplica gradientes, e para tokens curtos vira memorização (ver HuggingFace dedup tophf / BigCode — verbatim memorization scales com重复).
- DoReMi (Xie et al., 2023, NeurIPS) mostra que mistura ótima precisa de **reweighting via proxy model** com DRO, não pesos hardcoded por identidade.
- O viés identitário (family/clan/olavo com peso máximo) é um risco de **alucinação persistente** e de reduzir a generalização matemática — contrário ao objetivo declarado de "matemática = rainha".

---

## Parte 2 — Papers / Sub-Trabalhos Rejeitados ou Subvalorizados com Ideias Aplicáveis

### P1. SemDeDup — Abbas et al., 2023 (Meta)
**URL:** https://arxiv.org/abs/2303.09540 | **Code:** https://github.com/facebookresearch/SemDeDup
**Ideia:** Remover não só duplicatas exatas, mas pares **semanticamente similares** via embeddings de modelo pré-treinado + clustering. Reporta ganhos de eficiência equivalentes a ~dobrar o compute.
**Por que foi "subvalorizado":** Adotado em Pile-T5, mas a comunidade open-source ainda trata MinHash como suficiente. Para o F51 (corpus com muito mirror de Gutenberg/Wikipedia PT), SemDeDup capturaria paráfrases e re-edições que MinHash deixa passar.
**Aplicação F51:** Rodar SemDeDup uma vez offline no corpus approved antes do treino 2.5B.

### P2. MinHash + LSH (CCNet / BigCode / RedPajama)
**URL (ref canônica):** https://blog.nelhage.com/post/fuzzy-dedup/ | **HF:** https://huggingface.co/blog/dedup
**Ideia:** Jaccard aproximado via MinHash + LSH bands. Padrão ouro para near-duplicate dedup em web-scale.
**Por que "abandonado":** Não foi abandonado, mas o F51 não o implementou — provavelmente porque parece "caro". Na verdade, é barato e linear.
**Aplicação F51:** Substituir `content_hash` SHA-256 exato por pipeline MinHash (datasketch em Python puro, sem dependência pesada).

### P3. Dolma perplexity filtering — AllenAI, 2024
**URL:** https://aclanthology.org/2024.acl-long.840.pdf | **Blog:** https://allenai.org/blog/dolma-3-trillion-tokens-open-llm-corpus-9a0ff4b8da64
**Ideia:** KenLM 5-gram treinado em Wikipedia como scorer de "naturalness"; 3 buckets (low/mid/high perplexity). CCNet filtra 84,2% do Common Crawl.
**Insight negativo:** O paper documenta que **"High Threshold" remove menos que o esperado** e que filtering agressivo pode descartar conteúdo valioso (listas longas, tabelas).
**Aplicação F51:** KenLM é trivial de treinar (C++ leve) e dá um sinal muito mais forte que alpha-ratio. Treinar KenLM em ~100MB da Wikipedia PT + arXiv math.

### P4. Perplexity pruning diminishing returns — arXiv 2405.20541
**URL:** https://arxiv.org/html/2405.20541v1
**Ideia:** Mostra que **em regime over-trained**, perplexity pruning dá ganho só ~1,51× em compute-optimal. Ou seja: o filtro ajuda pouco se você já vai treinar muito além do Chinchilla.
**Por que importa para F51:** O F51 2.5B provavelmente **não** está over-trained (objetivo é retomar linhagem, não treinar 10x Chinchilla). Logo, perplexity pruning **ainda vale a pena** para o F51 — mesmo sendo "modinha antiga".
**Aplicação F51:** Não descartar perplexity com desculpa de "só funciona em escala".

### P5. DataComp-LM — Li et al., 2024 (Apple/Stanford)
**URL:** https://arxiv.org/abs/2406.11794 | **Site:** https://www.datacomp.ai/dclm/
**Ideia:** Benchmark que fixa o código de treino e deixa os times competir só em **curadoria de dados**. Achado central: **model-based filtering é o que mais diferencia** — classifier leve (fastText trained on Wikipedia vs random web) escolhendo docs superou heurísticas.
**Insight negativo documentado:** Global dedup across all snapshots "did not work as expected" — reduziu corpus a 4T tokens e deu pouco ganho (referido em https://www.equationblog.com/p/data-is-the-control-surface-for-llms). Ou seja, **dedup tem retorno边际 decrescente** se já passou por MinHash.
**Aplicação F51:** Treinar fastText classifier (positivo = Wikipedia PT + arXiv math; negativo = C4 random) e usá-lo como filtro principal, substituindo alpha-ratio.

### P6. DoReMi — Xie et al., 2023 (Stanford/Google, NeurIPS)
**URL:** https://arxiv.org/abs/2305.10429 | **Code:** https://github.com/sangmichaelxie/doremi
**Ideia:** Treina **small proxy model** com Distributionally Robust Optimization (DRO) para descobrir pesos ótimos de mistura de domínios automaticamente. Reporta +6,5 pp few-shot e 2,6× speedup vs default Pile weights.
**Por que foi "subvalorizado":** Adoção baixa fora de labs grandes porque parece "caro" (treina proxy). Mas o proxy é pequeno (~280M) e dá pesos que transferem para target maior.
**Aplicação F51:** Substituir `document_sample_weight` hardcoded e os `domain_weights` cosméticos por uma rodada DoReMi com proxy 280M. Resolve F5 (pesos falsos) com evidência.

### P7. Boundless BPE — arXiv 2504.00178
**URL:** https://arxiv.org/html/2504.00178v1
**Ideia:** Ablação explícita do papel da **pre-tokenização** (split em whitespace/pontuação antes do BPE). Mostra que cruzar fronteiras de palavra sem pre-tokenização prejudica compressão e aprendizado.
**Por que "subvalorizado":** Todo mundo assume que pre-tokenização regex (GPT-2 style) é necessária e acabou o debate.
**Aplicação F51:** O tokenizer F51 tem pre-tokenização fraca (`_pretokenize` em tokenizer.py:220-224 só separa `\S+|\n`), e o problema se manifesta no `_reconstruct_spacing`. Boundless BPE dá a base teórica para justificar a troca para um tokenizer com `Ġ`.

### P8. "Pre-Training Isn't Bitter Enough" — CMU ML Blog, 2026
**URL:** https://blog.ml.cmu.edu/2026/06/17/pre-training-isnt-bitter-enough/
**Ideia:** Argumenta que pretraining segue o "Bitter Lesson" em **método** de treino, mas **não** em seleção de dados — tentar "ser esperto" com curadoria ainda vence deixar a rede descobrir sozinha.
**Por que "negativo valioso":** É um contraponto ao hype de curadoria. Não invalida filtro, mas alerta: **não passe a vida ajustando mistura** se o treino pode compensar.
**Aplicação F51:** Foco em **aplicar uma vez** bons filtros (perplexity + SemDeDup + MinHash) e parar. Não fica re-tunando `domain_weights` todo ciclo.

### P9. "Can Small Training Runs Reliably Guide Data Curation?" — arXiv 2512.24503
**URL:** https://arxiv.org/html/2512.24503v1
**Ideia:** Testa 23 receitas de dados (composição, filtro, dedup) em 3 escalas e mostra que **pequenos runs não preveem reliably** o que funciona em escala maior. Mistura de dados tem forte interação com scale.
**Por que "negativo valioso":** Cautela contra over-fitting em decisões de dados baseadas em runs pequenos. O F51 faz exatamente isso — ajustar mistura em runs de teste.
**Aplicação F51:** Tratar pesos de mistura atuais como hipóteses, não como verdade. Validar pelo menos uma dimensão (ex: fração de math) em escala média antes de cravar.

### P10. STRR + Occiglot European tokenizer eval
**URLs:** https://arxiv.org/html/2510.09947v1 (STRR) | https://occiglot.eu/posts/eu_tokenizer_perfomance/ (Occiglot)
**Ideia:** STRR (Single-Token Root Recall) complementa fertility medindo **quantas palavras inteiras viram um token único**. Occiglot mostra fertility por linguagem europeia incluindo PT.
**Por que "subvalorizado":** Fertility média virou métrica única, mas esconde que **palavras funcionais** (the, de, que) precisam ser single-token para o modelo aprender sintaxe rápido.
**Aplicação F51:** Medir STRR e fertility-PT antes/depois de retreinar tokenizer. Se STRR para PT < 30%, retreinar é mandatório.

### Bônus — Phi "textbook quality" (Microsoft)
**URL:** https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/
**Ideia-jato:** 1.3B-2.7B treinado em dado **"textbook-quality"** (= sintético + curriculum agressivo) bate modelos 25× maiores. Prova que para SmallLM, **1 doc denso > 100 docs meh**.
**Aplicação F51:** Já meio alinhado com `document_sample_weight` (math=5×), mas a execução errada (repetir literal) destrói o benefício. Phi usa **re-weighting no sampler**, não repetição.

---

## Parte 3 — Recomendações Concretas de Patch (ordenadas por ROI para 2.5B)

### R1. [ALTA] Retreinar tokenizer com espaço-prefix (resolve F1)
- **Ação:** Em `f51_darwin/tokenizer.py:64-116` (`train`), mudar o pre-tokenize para prefixar cada palavra com espaço (`" " + word`) **antes** de gerar byte tokens, igual GPT-2. Isso faz `<b20>` entrar naturalmente em merges → tokens como `" the"`, `" de"` surgem.
- **Validação:** STRR (P10) antes vs depois. Meta: STRR-PT ≥ 40%.
- **Dependência:** Re-tokenizar todo corpus; bump de versão do checkpoint. **Fazer antes de retomar treino 2.5B.**
- **Custo:** ~1 dia de trabalho; re-tokenização offline.
- **Risco:** Pequeno. É o conserto mais barato-com-maior-impacto.

### R2. [ALTA] Substituir alpha-ratio por perplexity (KenLM) + fastText classifier (resolve F2)
- **Ação:**
  1. Treinar KenLM 5-gram em ~200 MB (Wikipedia PT + arXiv math) — binário leve.
  2. Treinar fastText binary classifier (DataComp-LM P5): positivo = docs curados, negativo = random C4.
  3. Em `data_firewall.py:50-56`, trocar `score_text` por `0.5*perplexity_score + 0.3*classifier_score + 0.2*length`.
- **Validação:** Rodar em 1k docs do corpus atual, comparar approvals com ground truth manual.
- **Custo:** ~2 dias. Sem GPU necessária (KenLM e fastText são CPU).

### R3. [ALTA] Implementar MinHash LSH para dedup (resolve F3)
- **Ação:** Em `f51_darwin/provenance.py`, adicionar campo `minhash_signature` ao `DatasetRecord`. Usar `datasketch` (Python puro, sem dep pesada). Threshold Jaccard 0.7 (não 0.85 como ngrama atual).
- **Manter** SHA-256 para dedup exata (barato), **adicionar** MinHash para near-dup.
- **Custo:** ~1 dia. Escala linear, resolve o O(N²) do `_collect_known_texts`.
- **Bonus:** Considerar SemDeDup (P1) como segundo estágio se MinHash não bastar.

### R4. [MÉDIA] Fixar bug de re-tokenização O(N²) em `ingest_pipeline.py` (resolve F4)
- **Ação:** Em `scripts/ingest_pipeline.py:255-280`, cache de tokens aprovados em arquivo `.tokens.bin` por `record.id`. Ingestão nova só tokeniza o **novo** doc e appenda ao bin existente.
- Remover o cap `max_new = 10% de 20M` — é um bug silencioso que descarta docs aprovados.
- **Custo:** ~0,5 dia. Ganho imediato em throughput do run247.

### R5. [MÉDIA] Unificar sistemas de mixing e remover repetição literal (resolve F5)
- **Ação:**
  1. Deletar `domain_weights` cosmético de `ingest_pipeline.py:158-165` (ou conectá-lo a algo real).
  2. Em `data.py:99-146` (`document_sample_weight`), trocar `repeat = int(round(weight))` (que vira 5× cópia literal) por **sample-weight no DataLoader** (uma probabilidade por documento, sem cópia).
  3. Reduzir viés identitário: `family/clan/olavo` em 4-5× sem justificativa empírica é perigoso. Usar 1.5-2× enquanto valida.
- **Custo:** ~1 dia. Reduz risco de alucinação e memorização.
- **Ideal (futuro):** Rodar DoReMi (P6) com proxy 280M para descobrir pesos ótimos sem chute.

### R6. [BAIXA] Adicionar filtro de PII e boilerplate
- **Ação:** Regex simples para email/telefone/CPF no `data_firewall.py`. Flag (não reject) para docs com > 30% de repetição de linha (boilerplate SEO).
- **Custo:** ~0,5 dia.

### R7. [BAIXA] Medir fertility / STRR antes de qualquer retreino
- **Ação:** Rodar Occiglot-style eval (P10) no tokenizer atual sobre 10k docs PT e 10k math. Documentar baseline.
- **Decisão:** Se fertility-PT > 2,5 tokens/palavra, **retreinar tokenizer é bloqueador** para o 2.5B.
- **Custo:** ~0,5 dia de medição.

---

## Sequência Sugerida (1-2 semanas)

1. **Semana 1 (bloqueadores):** R7 (medir) → R1 (tokenizer espaço-prefix) → R3 (MinHash). Sem isso, qualquer treino 2.5B herda os problemas estruturais.
2. **Semana 2 (qualidade):** R2 (perplexity+fastText) → R4 (fix O(N²)) → R5 (unificar mixing).
3. **Antes de retomar 2.5B:** validar pelo menos R1, R3, R4. R2 e R5 podem entrar em paralelo com primeira epoch.

---

## Fontes

- SemDeDup: https://arxiv.org/abs/2303.09540
- MinHash/LSH (Nelhage): https://blog.nelhage.com/post/fuzzy-dedup/
- BigCode dedup (HF): https://huggingface.co/blog/dedup
- Dolma (ACL 2024): https://aclanthology.org/2024.acl-long.840.pdf
- Dolma blog: https://allenai.org/blog/dolma-3-trillion-tokens-open-llm-corpus-9a0ff4b8da64
- Perplexity pruning diminishing returns: https://arxiv.org/html/2405.20541v1
- DataComp-LM: https://arxiv.org/abs/2406.11794
- "Data is the control surface" (Equation blog): https://www.equationblog.com/p/data-is-the-control-surface-for-llms
- DoReMi: https://arxiv.org/abs/2305.10429 | https://github.com/sangmichaelxie/doremi
- Boundless BPE: https://arxiv.org/html/2504.00178v1
- Pre-Training Isn't Bitter Enough (CMU): https://blog.ml.cmu.edu/2026/06/17/pre-training-isnt-bitter-enough/
- Can Small Runs Guide Curation: https://arxiv.org/html/2512.24503v1
- STRR: https://arxiv.org/html/2510.09947v1
- Occiglot EU tokenizer: https://occiglot.eu/posts/eu_tokenizer_perfomance/
- Phi-2 blog: https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/
- Karpathy GPT Tokenizer (motivação space-prefix): https://www.youtube.com/watch?v=zduSFxRajkE
- HF LLM Course (pre-tokenization): https://huggingface.co/learn/llm-course/en/chapter6/4
