# F51 Darwin-X — Auditoria de Arquitetura e Pesquisa de Papers Subvalorizados

**Agente:** pesquisa de arquitetura de modelo
**Data:** 2026-07-15
**Escopo:** `f51_darwin/darwin_x.py`, `f51_darwin/ssm_core.py`, `f51_darwin/ssd_block.py`, `f51_darwin/attention_block.py`, `f51_darwin/rope.py`, `f51_darwin/moe_layer.py`, configs de produção (`darwin_x_1.6b_nitro.yaml`, `darwin_x_600m.yaml`).
**Fonte de verdade da produção:** linhagem `F51-Darwin-X-1.6B-Nitro` (1.764B params, BF16, RTX 5060 Ti + RTX 3060). O `darwin_x.py` descreve o contrato 4B; onde os dois divergem, o fenômeno é apontado.

---

## 1. Resumo executivo

A arquitetura é um híbrido SSD (Mamba-1) + GQA attention em razão 3:1, com MoE estilo DeepSeek (fine + shared experts) e um sistema neuroendócrino custom que regera neurogenesis/pruning/expansion. As ideias são ambiciosas, mas **três gargalos de engenharia** comprometem a viabilidade do treino antes de qualquer questao cientifica:

1. O selective scan e uma implementacao ingenua em PyTorch puro que materializa tensores `O(B * D_inner * d_state * L)` e itera em Python `log2(L)` vezes. E o gargalo de memoria e compute dominante, e forcina fp32 no scan (bloqueia BF16/FP16).
2. O forward do MoE e um loop Python sequencial de 14 experts com gather/scatter por mascara — exatamente o anti-padrao que MegaBlocks agrupado (grouped GEMM) resolve.
3. A config de produção usa `d_state = 16` (Mamba-1 default), contradizendo o docstring do `ssm_core.py` que alega `d_state = 64`. Com `dt` clamped em `[0.001, 0.1]` e 16 canais de estado, 12 das 16 camadas (75%) tem horizonte de memoria curto — e a camada 0 e SSD, entao a primeira representacao que um token recebe ja e memory-starved.

Além disso, o **sistema neuroendócrino** (5 "hormônios", ~640 linhas, sem base na literatura, com dezenas de thresholds magicos e um backward-hook de controle fragil) e a maior superficie de complexidade nao validada do modelo. Ha papers rejeitados/subvalorizados que cobrem quase todos esses gaps com ideias reaproveitaveis e baratas.

---

## 2. Cinco fracas especificas no codigo atual

### F1. Selective scan ingenuo em PyTuro — explosao de memoria O(B·D·S·L) e bloqueio de BF16
**Local:** `f51_darwin/ssm_core.py:32-86` (`selective_scan`), chamado em `ssm_core.py:211`.

A implementacao materializa `a_bar` e `b_bar` com shape `[B, D_inner, d_state, seq_len]` e os recombina num loop Python de `ceil(log2(L))` iteracoes, cada uma fazendo `torch.cat` de tensores almost-tao-grandes. Para a config de producao 1.6B (d_model=1920, ssm_expand=2 -> d_inner=3840, d_state=16, seq=4096, batch=1):

```
a_bar: 1 * 3840 * 16 * 4096 = 251M elementos * 4B (fp32) = ~1.0 GB
b_bar: idem                                                  ~1.0 GB
materializados/copiados ~12x (log2 4096)
```

Isso e **~2 GB por camada SSD so de buffers de scan**, replicados a cada passo do prefix-sum. Com 12 camadas SSD, e catastrofico. Pior: o scan forcina fp32 (`ssm_core.py:211-214` faz `.float()` em u, delta, b, c), o que **invalida o treino BF16/FP16** no mixer — exatamente onde esta o compute. A implementacao de referencia (Mamba oficial) usa um kernel CUDA fusionado (`state-spaces/mamba`, `selective_scan_cuda`) que e `O(B·D·S)` de memoria e `O(L)` de trabalho, rodando em BF16 com acumulador FP32. Este e o unico item que, sozinho, justifica o gap de throughput entre o F51 e um treino Mamba de referencia.

### F2. Forward do MoE e loop Python sequencial por expert, sem grouped GEMM
**Local:** `f51_darwin/darwin_x.py:1873-1885` (e espelhado em `moe_layer.py:202-233`).

```python
for expert_idx, expert in enumerate(self.fine_experts):
    mask = flat_indices == expert_idx
    if not mask.any(): continue
    token_rows, slots = torch.where(mask)
    expert_out = expert(flat_x[token_rows], self.neuroendocrine, expert_idx)
    output[token_rows] += flat_weights[token_rows, slots].unsqueeze(-1) * expert_out
```

Para 14 experts na config 1.6B (32 no contrato 4B), sao ate **14 lancamentos de kernel sequenciais** com gather (`flat_x[token_rows]`) e scatter (`output[token_rows] += ...`) em volta de cada um. MegaBlocks (Gale et al., MLSys 2023) demonstrou que este e o anti-padrao classico: a formulacao correta e um unico grouped GEMM block-sparse que processa todos os experts em um lancamento de kernel, eliminando ainda token-dropping. Alem disso, o roteamento faz `softmax` apenas sobre os top-k logits (`darwin_x.py:1037`) — o esquema Switch/V-MoE — em vez do balanceamento aux-loss-free do DeepSeek-V3, e por isso precisa do patch `aux_loss_adaptive` (linhas 2411-2419) para evitar colapso. O nitro offloading agrava: `nitro_gpu_expert_capacity=5` com 14 experts significa 9 experts vivem na CPU e sao restaurados (`_restore_expert`) durante o treino — thrash de PCIe.

### F3. `d_state = 16` em producao contradiz o docstring; SSM memory-starved
**Local:** `configs/darwin_x_1.6b_nitro.yaml:17` (`ssm_state: 16`) vs `f51_darwin/ssm_core.py:14` (docstring: *"d_state=64 (Mamba-2 default, era 16)"* listado como "correcao vs legado").

O docstring mente sobre o que roda. A config 600M usa `ssm_state: 64`; a **producao 1.6B usa 16**. Combinado com `dt` clamped em `[0.001, 0.1]` (`ssm_core.py:199`) e apenas 16 canais de estado, o horizonte efetivo de retencao do SSM e curto (o proprio smoke-test em `ssm_core.py:278` alerta se <50% dos canais "vivos"). Ocorre que **12 das 16 camadas sao SSD** (75% da mistura), e a **camada 0 e SSD** (atencao ocorre nos indices 3, 7, 11, 15). Logo o primeiro sinal que um token recebe ja e memory-starved. Isto ataca exatamente a capacidade de associative recall — a falha conhecida e documentada de Mamba puro (Trockman 2024; Arora "Zoology" 2023; "When Recalling In-Context, Transformers Are Not SSMs", 2025).

### F4. Atencao GQA com `bias=True`, sem cache de KV, RoPE reconstruido a cada forward
**Local:** `f51_darwin/darwin_x.py:205-211` (projetores QKVO com `bias=True`), `:220` (RoPE rebuilt toda forward), `:226-232` (SDPA sem path varlen/chunked para dual-GPU).

- `bias=True` nos quatro projetores de atencao e nao-padrao (LLaMA, Qwen, DeepSeek, Mistral usam `bias=False`). Adiciona `d_model + 2*kv_dim + d_model` parametros mortos por camada, quebra compatibilidade com kernels fused (FlashAttention varlen espera QKV sem bias) e nao ajuda a loss.
- `build_rope_cache` e chamado dentro de `forward` a **cada passo** (`darwin_x.py:220`); deveria ser cacheado uma vez por `seq_len`/device.
- Existe um modulo `f51_darwin/kv_cache.py`, mas `GQACausalAttention` **nao usa nenhum cache de KV**. A config promete `inference_context_length: 32768` sem atencao em blocos / paged / sliding-window para sustentar isso. Para inferencia longa, isso significa OOM ou recompute integral.
- `_dual_gpu` faz pipeline split de blocos, mas a atencao cruza GPUs sem path de atencao varlen paginada — limitante para contexto longo nas duas GPUs.

### F5. Sistema neuroendocrino: superficie massiva nao validada, com magic numbers e backward-hook fragil
**Local:** `f51_darwin/darwin_x.py:322-961` (NeuroendocrineSystem), `:252-319` (`_make_moe_backward_hook`), `:2612-2641` (`_ghost_loss` re-roda todas as 16 camadas).

- **Zero base na literatura** para a dinamica especifica: 5 "hormonios" (dopamina, NA, cortisol, BDNF, ACh) com τ harcodeados (200/150/300/400/500), thresholds magicos por toda parte (`0.05`, `0.3`, `0.6`, `0.75`, `500 passos`, `1.5x`, etc.). A framing biologica e metaforica, nao matematica.
- `plasticity_decision` (linha 861) usa regras AND-thresholded (`observations >= 20 and reversal > 0.6 and magnitude < 0.5`, etc.) que raramente disparam de forma coerente; o sistema opera quase todo em "shadow mode", o que confirma o proprio STATUS_ATUAL quando diz que o "estado mutacional de gradientes" esta "implementado, nao calibrado".
- O **protected_subspace** (linhas 426-438, 637-685) e um GPM pobre: rank-4 por expert sobre um sketch 64-dim produzido por `adaptive_avg_pool1d` (linha 1204) — pooling destrui informacao direcional. A base e atualizada com uma logica Gram-Schmidt-esque sobre esse sketch, sem SVD/QR real sobre os gradientes.
- O backward hook (`_make_moe_backward_hook`, linhas 252-319) depende de um contador `_autonomic_forward_uses` para "amortizar" a atualizacao — semantica fragil (um forward a mais/menos e o hook atualiza no momento errado). Hooks de backward tambem nao sao boundary seguro para mutacao estrutural; o codigo reconhece isso e adia para `execute_structural_actions`, mas o pipeline de propostas->execução tem muitas condicoes de corrida implicitas (e.g., neurogenesis expansde buffers, mas o optimizer pode ainda ter references antigas).
- **`_ghost_loss` dobra o compute**: re-roda todas as 16 camadas numa forward paralela sobre tokens mascarados (linhas 2624-2641) com peso 0.07. E essencialmente uma segunda forward cheia, sem compartilhamento de grafos/hidden states. Para um treino continuo em 2 GPUs de 16/12 GB, isso e ~50% de compute desperdicado por passo.

Bonus: `residual_scale_multiplier = 1.4` -> `residual_scale = 1.4/√N` aplicado a **ambos** sublayers (`darwin_x.py:1964-1967`) e flhavor de DeepNorm mas **sem** a inicializacao β-escalada companheira (o init usa `normal_(std=min(0.02, 0.30/√d_model))`, linha 2349). Essa mistura de escala DeepNorm com init nao-DeepNorm e uma fonte conhecida de instabilidade (BranchNorm, ACL 2024).

---

## 3. Papers rejeitados/subvalorizados com ideias aplicaveis

Para cada um: por que foi rejeitado/subvalorizado, e o que exatamente usar no F51.

### P1. Mimetic Initialization Helps State Space Models Learn to Recall
- **Autores/ano:** A. Trockman et al., 2024 (arXiv:2410.11135).
- **Por que subvalorizado:** ofuscado pelo consenso "so adicionar camadas de atencao"; papers de inicializacao de SSM raramente entram em venue topo (~12 citacoes). Predecessor "Mimetic Initialization of Self-Attention" (PMLR 2023) e mais conhecido.
- **O que usar:** inicializacao estruturada para B/C/A do SSM que faz a camada mimetizar um induction-head de atencao no init. Resolve a fraqueza **F3** (recall ruim) **sem adicionar atencao** — drop-in barato. Ganho reportado: recall em strings ate 4x mais longas. E o patch de menor risco para a deficiencia de d_state=16.

### P2. Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts
- **Autores/ano:** DeepSeek-AI (Wang et al.), 2024 (arXiv:2408.15664; OpenReview y1iU5czYpE).
- **Por que "subvalorizado/criticado":** a thread OpenReview tem pushback de revisores sobre generalizacao alem da escala DeepSeek-V3 e sensibilidade ao hyperparametro de update de bias. Ainda assim virou o padrao de fato em producao.
- **O que usar:** substituir `load_balance_loss + router_z_loss + aux_loss_adaptive` (`darwin_x.py:1934`, `:2411-2419`) por um bias por expert que acumula `media-movel(fracao_roteada) - uniforme`. Elimina a interferencia do aux-loss no sinal de LM — exatamente o sintoma que o patch `aux_loss_adaptive` trata sintomaticamente. Resolve metade da **F2**.

### P3. MegaBlocks: Efficient Sparse Training with Mixture-of-Experts
- **Autores/ano:** Gale et al., MLSys 2023 (Berkeley/Databricks). Github `databricks/megablocks`.
- **Por que "subvalorizado":** nao e rejeitado, mas sub-adotado fora de grandes labs; em laboratorios pequenos o padrao ainda e o loop por-expert.
- **O que usar:** reformular o loop de `darwin_x.py:1873-1885` como um unico grouped GEMM block-sparse (Triton), todos os 14 experts num lancamento de kernel. Elimina token-dropping e o overhead de gather/scatter. Resolve a outra metade da **F2**. Complementar com o blog PyTorch "Accelerating MoEs with Triton Grouped GEMM" para a implementacao.

### P4. Scattered Mixture-of-Experts Implementation (o "contra-MegaBlocks")
- **Autores/ano:** Liu et al., 2024 (arXiv:2403.08245; OpenReview YDZ7GeFLxq).
- **Por que subvalorizado:** posicao contraria a MegaBlocks, benchmarks mistos, entrou mais como workshop/preprint do que mainstream.
- **O que usar:** e um **caminho de migracao** menos invasivo que reescrever tudo pra MegaBlocks: mantem a estrutura scatter/indice atual do F51 mas roteia os tokens agrupados num grouped GEMM. Ideal como passo intermediario enquanto o time nao adota Triton puro. Reduz risco da **F2**.

### P5. From Sparse to Soft Mixtures of Experts (Soft MoE)
- **Autores/ano:** Puigcerver et al., ICLR 2024 (arXiv:2308.00951; OpenReview jxpsAj7ltE).
- **Por que rejeitado/criticado:** revisores apontaram incompatibilidade com LM autoregressivo/causal (o metodo mistura informacao entre slots de tokens). Aceito no ICLR mas com caveats fortes, e por isso pouco usado em LMs causais.
- **O que usar:** mesmo nao podendo usar Soft MoE diretamente, a **ideia central** — dispatch/combine soft elimina token-dropping e instabilidade — e exatamente o que o hack de `_anti_collapse_jitter` (`darwin_x.py:1010-1024`) tenta simular mal. Substituir o jitter por um soft-top-k causal (sigmoid gating com sparsificacao top-k, estilo "sparse-softmax MoE") remove o colapso de roteamento sem a hack. Apoia **F2/F5**.

### P6. Zoology: Measuring and Improving Recall in Efficient Language Models (+ Based)
- **Autores/ano:** Arora & Eyuboglu, Stanford Hazy Research, 2023-2024 (blog + arXiv; model "Based").
- **Por que subvalorizado:** linhagem "blog + small-scale", nao entrou como paper-marquee; mas e a analise mais limpa do por que SSMs falham em recall.
- **O que usar:** o benchmark **MQAR** e a achado de que SSMs puros falham em recall a menos que recebam um hibrido "sliding-window attention + tiny attention". Valida diretamente: (a) mover uma camada de atencao para o indice 0 (hoje e SSD — **F3**); (b) usar MQAR como gate antes de subir d_state.

### P7. When Recalling In-Context, Transformers Are Not SSMs / Revisiting Associative Recall
- **Autores/ano:** 2025 (arXiv:2508.19029).
- **Por que subvalorizado:** preprint recente, sem venue definida, analisa so Mamba.
- **O que usar:** a scaling-law empirica: recall escala com **depth × width** para SSMs mas so com **width** para atencao. Justifica matematicamente subir `d_state` de 16 para 64-128 (**F3**) e faz auditoria de posicionamento de camadas de atencao.

### P8. Gradient Projection Memory for Continual Learning (+ Rethinking GPC CL)
- **Autores/ano:** Saha et al., ICLR 2021 (GPM); Zhao et al., CVPR 2023 (Rethinking).
- **Por que subvalorizado para o F51:** classicos em CL de visao, quase nunca aplicados a MoE de LLMs.
- **O que usar:** substituir o `protected_subspace` caseiro e o sketch 64-dim (`darwin_x.py:426-438, 637-685, 1198-1206`) por um **GPM real** — SVD/QR sobre os gradientes por-expert de verdade (nao sobre um adaptive-pool sketch que destrui direcao), com a calibracao stability/plasticity do CVPR2023. Isso converte o sistema neuroendocrino de "heuristica" para "metodo com base", atacando **F5**.

### P9. H2O / RocketKV / Quest — compresao de KV cache para contexto longo
- **Autores/ano:** H2O (NeurIPS 2023), Q-Hitter (MLSys 2024), RocketKV (ICML 2025), Quest (2024).
- **Por que subvalorizados:** Quest em particular teve recepcao mista pela complexidade do query-aware chunking; H2O e simples mas pouco adotado em laboratorio pequeno.
- **O que usar:** com `inference_context_length: 32768` e **nenhum gerenciamento de KV cache** (**F4**), pelo menos H2O (recent-tokens + heavy-hitters) previne OOM em inferencia longa. Se houver budget, Quest para selecao query-aware de blocos.

### P10. BranchNorm / "Post-LayerNorm Is Back" (alternativas ao DeepNorm)
- **Autores/ano:** BranchNorm, ACL 2024 Findings; "Post-LN Is Back", 2025.
- **Por que subvalorizados:** alternativas recentes ao DeepNorm sem a co-tunagem α/β.
- **O que usar:** o F51 aplica `residual_scale = 1.4/√N` a ambos sublayers (**F5 bonus**) sem a inicializacao β companheira do DeepNorm — mistura instavel. Ou commita com DeepNorm completo (α-residual + β-init) ou usa BranchNorm que nao precisa de init co-tunada. Remove uma fonte de instabilidade silenciosa.

---

## 4. Recomendacoes concretas de patch (priorizadas)

### P0 — bloqueadores de corretude/throughput

**P0.1 — Substituir o selective scan ingenuo por kernel fusionado.** (`ssm_core.py:32-86`)
- Ideal: adicionar dependencia `mamba-ssm` (kernel CUDA `selective_scan_cuda`) com fallback em BF16/FP16-acumulado.
- Intermediario: ao menos chunkar o scan em blocos de 512-1024 tokens para reduzir o pico de memoria do prefix-sum.
- KPI esperado: reducao de ~10x na memoria do SSD-forward e desbloqueio de BF16 end-to-end.
- Sem paper externo: e a implementacao de referencia do proprio Mamba.

**P0.2 — Forward do MoE em grouped GEMM (MegaBlocks/Triton).** (`darwin_x.py:1873-1885`, `moe_layer.py:202-233`)
- Migrar o loop de 14 experts para um grouped GEMM block-sparse; usar P4 (Scattered MoE) como passo intermediario se a reescrita total for arriscada.
- Combinar com P2 (DeepSeek aux-loss-free) para aposentar `aux_loss_adaptive` e `_anti_collapse_jitter`.
- KPI: ~3-5x throughput no MoE-forward.

**P0.3 — Reconciliar `d_state` e posicionamento de atencao.** (`configs/darwin_x_1.6b_nitro.yaml:17`, `darwin_x.py:158`)
- Decidir: ou subir producao para `ssm_state: 64` (alinha com docstring e com Mamba-1), ou corrigir o docstring para dizer a verdade (16).
- Adotar P1 (Mimetic Init) como mitigacao de baixo risco para a deficiencia de recall enquanto d_state nao sobe.
- Validar qualquer mudanca com o benchmark MQAR (P6) antes de declarar vitoria.
- Mover pelo menos uma camada de atencao para o inicio (ex.: indices 0, 7, 11, 15 em vez de 3, 7, 11, 15) — a camada 0 SSD e o pior caso para recall.

### P1 — corretude arquitetural

**P1.1 — Remover `bias=True` dos projetores QKVO.** (`darwin_x.py:205-208`, `attention_block.py:34`)
- Setar `bias=False` em q_proj/k_proj/v_proj/o_proj (e no qkv do attention_block). Economiza parametros e destrava FlashAttention varlen.

**P1.2 — Cache de RoPE unico e KV cache para inferencia.** (`darwin_x.py:220`)
- Cachear `build_rope_cache` por (seq_len, device); integrar `kv_cache.py` ao `GQACausalAttention`; adicionar H2O (P9) como politica de evacuacao para o contexto de 32k.

**P1.3 — Balanceamento MoE aux-loss-free (DeepSeek-V3).** (`darwin_x.py:1031-1048`, `:1926-1942`, `:2411-2419`)
- Substituir `load_balance_loss`/`router_z_loss`/`aux_loss_adaptive` por bias acumulativo por expert.

### P2 — reducao de complexidade/risco

**P2.1 — Por o sistema neuroendocrino behind ablation gate.** (`darwin_x.py:322-961`)
- Validar BWT/FWT (item em aberto no STATUS_ATUAL) antes de confiar em `plasticity_decision`. Se nao validar em N ciclos, stripar para no-op e manter so observabilidade hormonal.

**P2.2 — Trocar protected_subspace caseiro por GPM real (P8).** (`darwin_x.py:426-438, 637-685, 1198-1206`)
- SVD/QR sobre gradientes por-expert de verdade; abandonar o sketch 64-dim.

**P2.3 — Reescrever `_ghost_loss` para compartilhar o forward.** (`darwin_x.py:2612-2641`)
- Reusar os hidden states do forward principal em vez de re-rodar 16 camadas; ou baixar `ghost_weight` a ~0 ate validar.

**P2.4 — Decidir DeepNorm vs BranchNorm.** (`darwin_x.py:166-167, 2349`)
- Ou DeepNorm completo (α-residual + β-init) ou BranchNorm sem init co-tunada; nao meio-termo.

---

## 5. Notas de verificacao e limites

- Os parametros de producao citados (d_model=1920, n_layers=16, d_state=16, 14 fine experts, nitro_gpu_expert_capacity=5) foram lidos de `configs/darwin_x_1.6b_nitro.yaml` em 2026-07-15. Qualquer mudanca posterior na config invalida os numeros de memoria F1/F2 — recompute com `estimate_darwin_x_parameters`.
- A analise de memoria da F1 assume batch=1; treino real com block size 64 e batch 1 mantem o numero de tokens, mas o pico de memoria do prefix-sum scales com o maior de (batch, d_inner). O gargalo permanece.
- Os papers P1-P10 foram caracterizados a partir de seus abstracts/OpenReview/buscas; antes de implementar, ler o texto completo (especialmente P1, P2, P8) para nao deturpar detalhes.
- Esta pesquisa nao rodou nenhum experimento; as recomendacoes sao hipoteses fortemente suportadas pelo codigo e pela literatura, mas devem ser gated pelo benchmark noturno (5 condicoes x 5 seeds, shadow mode) que o proprio STATUS_ATUAL ja lista como pendencia.

## 6. Fontes (papers e referencias)

- Mimetic Init SSM — https://arxiv.org/abs/2410.11135 ; OpenReview https://openreview.net/forum?id=iVy7aRMb0K
- DeepSeek-V3 / Aux-Loss-Free — https://arxiv.org/abs/2412.19437 ; https://openreview.net/forum?id=y1iU5czYpE ; review https://huggingface.co/blog/NormalUhr/moe-balance
- MegaBlocks — https://people.eecs.berkeley.edu/~matei/papers/2023/mlsys_megablocks.pdf ; https://github.com/databricks/megablocks
- PyTorch Triton Grouped GEMM — https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/
- Scattered MoE — https://arxiv.org/html/2403.08245v2
- Soft MoE — https://arxiv.org/abs/2308.00951 ; https://openreview.net/forum?id=jxpsAj7ltE
- Zoology / Based (Arora & Eyuboglu) — https://hazyresearch.stanford.edu/blog/2023-12-11-zoology1-analysis ; https://github.com/HazyResearch/zoology
- When Recalling In-Context, Transformers Are Not SSMs — https://arxiv.org/html/2508.19029v1
- GPM (Saha et al. ICLR 2021) — https://arxiv.org/abs/2103.09762 ; https://openreview.net/forum?id=3AOj0RCNC2
- Rethinking Gradient Projection CL (CVPR 2023) — https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_Rethinking_Gradient_Projection_Continual_Learning_Stability__Plasticity_Feature_Space_CVPR_2023_paper.pdf
- H2O — https://arxiv.org/abs/2306.14048
- DeepNet/DeepNorm — https://arxiv.org/abs/2203.00555
- BranchNorm (ACL 2024) — https://aclanthology.org/2024.findings-acl.695.pdf
- Mamba (referencia) — https://arxiv.org/abs/2312.00752
- GPU Gems prefix-sum — https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
- Jamba (ratios hibridos) — https://arxiv.org/html/2403.19887v1
