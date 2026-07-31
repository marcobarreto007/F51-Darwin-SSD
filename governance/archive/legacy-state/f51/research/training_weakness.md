# Training Pipeline - Fraquezas e Papers Subvalorizados

**Escopo:** `scripts/darwin_organism.py` (organismo + run247) e `scripts/start_overnight_25b.ps1` (launcher 2.5B).
**Fontes de codigo lidas:** `darwin_organism.py`, `start_overnight_25b.ps1`, `f51_darwin/brainstem.py`, `f51_darwin/data.py`, `f51_darwin/darwin_x_training.py`, `docs/operacao/STATUS_ATUAL.md`.
**Data:** 2026-07-15. **So pesquisa + recomendacoes. Nenhum codigo.**

---

## 1. Fraquezas especificas no pipeline atual

### 1.1 O optimizer e destruido (momento zerado) a cada mudanca de topologia
**Local:** `darwin_organism.py` linhas 768-783 (`run_cycle`, apos `execute_structural_actions`).
```python
self.optimizer = torch.optim.AdamW(
    self.model.parameters(), lr=old_lr,
    weight_decay=self.cfg.weight_decay, fused=True,
)
```
Toda vez que ocorre poda/expansao/neurogenese, o AdamW e reconstruido do zero. Isso descarta **todo** `exp_avg` e `exp_avg_sq` de **todos** os parametros (nao so dos experts afetados). Em escala 2.5B com treino continuo, cada evento estrutural = pico de loss + re-aprendizado de toda a rede. E o dano mais grave para a linhagem 2.5B. Curiosamente o codigo ja tem migracao fina para o caminho v6->v7 (`migrate_adamw_state_with_new_baselines`) — a mesma tecnica de "remap por ordem" poderia preservar os tensores dos parametros inalterados apos neurogenese.

### 1.2 Sem LR schedule, sem warmup, sem estado de scheduler no checkpoint
**Local:** `cfg.learning_rate = 1.5e-4` flat (CLI `--lr` only); `payload["training_state"]` salva apenas `step` e `cycle`.
- LR constante em continual pretraining e subotimo.
- **Critico no resume:** quando o optimizer e reiniciado fresh (Adafactor legacy, falha de load, ou mudanca de topologia), a LR cheia `1.5e-4` incide sobre o modelo sem warmup. Em 2.5B isso e receita classica para divergencia nos primeiros passos.
- Nao existe `scheduler_state` no checkpoint; `start_overnight_25b.ps1` nao passa warmup. O organismo nao sabe distinguir "resume no meio de um ciclo estavel" de "inicio frio".

### 1.3 Sem gradient accumulation — batch efetivo minimo
**Local:** `start_overnight_25b.ps1` `BatchSize=1`, `BlockSize=64`; loop em `_train_cycle` faz `optimizer.step()` a cada micro-batch.
- Batch efetivo = **64 tokens**. Em 2.5B o ruido do gradiente domina completamente o sinal.
- Combinado com `grad_clip=1.0` fixo, o clipping atua em ruido puro: a norma do gradiente e quase inteiramente variancia amostral, nao sinal de descida.
- A *gradient noise scale* (McCandlish 2018) para modelos ~2B tipicamente aponta para batch otimo na casa de milhoes de tokens. Aqui se treina 4 ordens de magnitude abaixo.

### 1.4 Checkpoint totalmente sincrono — bloqueia o treino por minutos
**Local:** `_save_cycle` linhas 1129-1188.
```python
model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}  # copia sincrona GPU->CPU
...
torch.save(payload, temporary_path)  # I/O de disco bloqueante
```
Para o checkpoint 2.5B (~11 GB, conforme `STATUS_ATUAL.md`: `organism_cycle_077.pt 11.10 GB`), isso significa:
- Copia completa do modelo de ambas as GPUs para CPU no thread de treino (sincrono, ~segundos a dezenas de segundos).
- `torch.save` bloqueante gravando ~11 GB (minutos em SSD SATA).
- Sem async save, sem overlap com proximo ciclo. Cada boundary de ciclo = janela de stall grande. Em run247 indefinido, isso e tempo de GPU ocioso acumulado.

### 1.5 Granularidade de checkpoint = boundary de ciclo; crash perde ate 2000 steps
**Local:** `_save_cycle` so roda no fim de `run_cycle`. No `run247` handler de `KeyboardInterrupt` (linhas 1544-1550) o `report` usado e da **ultima** chamada de `run_cycle` — se o sinal chegar no meio de um ciclo, o checkpoint salvo e do ciclo anterior.
- `start_overnight_16b.ps1` usa 2000 steps/ciclo. Um crash as 1999 steps perde quase um ciclo inteiro de treino.
- Nao existe checkpoint de emergencia em nivel de step, nem signal handler para SIGTERM (so KeyboardInterrupt). O processo pode morrer OOM/power fail e o maximo de trabalho perdido e `steps_per_cycle`.

### 1.6 (Secundaria) Brainstem `check_loss` e codigo morto no loop de treino
`brainstem.check_loss` / `full_check` existem e definem `max_loss=10`, `loss_explosion_ratio=3.0`, `min_loss=1e-6` — mas **nenhuma dessas chamadas aparece em `_train_cycle`**. O unico guarda real no loop e `torch.isfinite(loss)` (linha 961). Logo: um pico de loss de 30 contra baseline 5 e tranquilamente backpropagado para os pesos. `min_steps_between_checkpoints`, `max_checkpoints_per_run`, `loss_plateau_threshold` tambem sao checados em nenhum lugar do treino.

### 1.7 (Secundaria) Replay ingenuo + pad com token 0
- `_batch_from_replay` preenche com `ids + [0] * (block_size - len(ids))`. Se o id 0 do `f51_bpe_80k` nao for um pad real, o replay injeta zeros espurios que o modelo aprende como distribuicao.
- O buffer faz replay **naive** (re-emite tokens); sem reservoir sampling garantido, sem distilacao de logits. A literatura de continual learning mostrou que DER++ domina naive replay consistentemente.

### 1.8 (Secundaria) bf16 sem GradScaler e sem monitoramento de underflow
Linha 662: `GradScaler(self.device.type, enabled=False)`. bf16 nao exige loss scaling como fp16, mas em treino ruidoso de micro-batch pequeno, atualizacoes pequenas podem silently underflow no range do bf16 sem ninguem notar.

---

## 2. Papers rejeitados / subvalorizados com ideias aplicaveis

### 2.1 Schedule-Free AdamW — Defazio, Yang, et al. (NeurIPS 2024)
"The Road Less Scheduled" (arXiv:2405.15682). Vencedor do **MLCommons AlgoPerf 2024**, superando schedules cosine pesadamente tunadas. Nao requer schedule de LR nem horizonte definido: faz media de iterates (weight averaging implicito) com custo quase zero. **Aplicacao direta a F51:** o organismo `run247` e treino continuo de horizonte desconhecido por definicao — schedule cosine e a escolha errada. Schedule-Free elimina a fragilidade do resume sem warmup (fraqueza 1.2) porque nao ha schedule para dessincronizar. Em treino longo (>1000 tokens/param) supera SOTA em ~31%.

### 2.2 8-bit Optimizers via Block-wise Quantization — Dettmers et al. (2021)
(arXiv:2110.02861; lib `bitsandbytes`). Comprime estados do AdamW para 8-bit via quantizacao por blocos, isolando outliers. Corta memoria de optimizer de ~8 bytes/param para ~2 bytes/param. **Aplicacao direta:** em 2.5B = economia de ~15 GB de VRAM. No hardware atual (RTX 5060 Ti 16 GB + RTX 3060 12 GB), isso libera espaco para batch maior ou contexto maior — ataca a fraqueza 1.3 indiretamente. Subvalorizado porque "AdamW fp32 e o default seguro", mas validado em escala de bilhoes. Drop-in via `bnb.optim.AdamW8bit`.

### 2.3 Sophia — Second-order Clipped Stochastic Optimization — Hong, Liu, et al. (ICLR 2024)
(arXiv:2305.14342). Usa estimador barato da diagonal da Hessiana (Hutchinson) + clipping. **2x mais rapido que Adam** em passos/compute/wall-clock em GPT-2 125M-770M, mesma perplexidade com 50% dos passos. Foi "abandonado" por inercia de ecossistema (ninguem quer migrar pipelines de bilhoes para um optimizer novo). Em escala 2.5B/continual, a reducao de 2x e real e compensaria o overhead da Hessiana. Ataca diretamente o custo de wall-clock do `run247`.

### 2.4 ZeRO-Offload — Ren et al. (USENIX ATC 2021)
"Democratizing Billion-Scale Model Training". Offload de estado de optimizer + computacao de optimizer para CPU, mantendo uma unica copia no host. Treina 10B+ em uma unica GPU. **Aplicacao direta:** F51 ja faz "model parallel dinamico em um unico processo" em dual GPU; ZeRO-Offload e a versao canonicamente validada dessa ideia. Permitiria subir para 2.5B (e alem) confortavelmente sem OOM, e mover o optimizer state (o culpado pela fraqueza 1.4) para CPU reduziria tambem o stall de checkpoint.

### 2.5 DER++ (Dark Experience Replay) — Buzzega et al. (NeurIPS 2020)
Rehearsal simples e forte para continual learning. Armazena nao so tokens mas **logits** no buffer e aplica uma loss de distilacao entre modelo atual e logits antigos. **Aplicacao direta:** substitui o replay naive do F51 (fraqueza 1.7) por um baseline estabelecido que dominate naive replay em quase todos os benchmarks de forgetting. Custo marginal: armazenar logits + 1 termo de loss.

### 2.6 Muon — Orthogonal Momentum — Jordan et al. (2024)
(kellerjordan.github.io/posts/muon; agora no PyTorch 2.13). Ortogonaliza o momento via iteracao de Newton-Schulz para parametros 2D. ~35% mais rapido que AdamW em NanoGPT speedruns. Novo, adotado rapido em LLM training. **Aplicacao:** pode ser usado para as camadas ocultas (a maioria dos 2.5B params) deixando AdamW no resto. E um "matrix analogue do sign operator" — mesma filosofia do Lion mas mais estavel.

### 2.7 Lion / EvoLved Sign Momentum — Chen et al. (2023)
(arXiv:2302.06675). Descoberto por busca simbolica. Sign-only: acompanha so o momento (um slot, nao dois como Adam). ~metade da memoria de optimizer do AdamW. Foi alvo de ceticismo (sensibilidade a LR/weight decay, resultados piores fora dos benchmarks Google). **Aplicacao:** a economia de memoria e inquestionavel; a sensibilidade pode ser mitigada com LR ~3-10x menor que AdamW e weight decay maior. Como experimento de benchmark no 600M antes de mover para 1.6B/2.5B.

### 2.8 Asynchronous Checkpointing — PyTorch DCP / blog "6x faster Async Checkpointing"
(distributed_async_checkpoint_recipe; pytorch.org/blog/6x-faster-async-checkpointing). Fase 1: copia GPU->CPU em CPU buffers (critico, ~seg). Fase 2: thread/processo em background grava em disco enquanto o treino prossegue. **Aplicacao direta:** elimina a fraqueza 1.4. Implementacao via `torch.distributed.checkpoint.state_dict_saver.async_save` ou `multiprocessing.Process` simples que herda o payload CPU apos o `v.cpu()`. Ganho de wall-clock imediato, sem mudanca de modelo.

### 2.9 "On the Limits of Curriculum Learning for Post-Training LLMs" + Chuang et al.
A literatura recente de curriculum learning em LLMs mostra que curriculum **nao e universalmente util**: quando a metrica de dificuldade (ou peso de amostra) esta mal alinhada com o processo de aprendizado do modelo, o curriculum atrapalha. **Aplicacao reflexiva:** `f51_darwin/data.py::document_sample_weight` atribui pesos manuais (matematica 5x, identidade 5x, familia 4x, olavo 3x). Esse e exatamente o tipo de curriculum "alinhado com a intuicao humana, nao com a incerteza do modelo" que a literatura alerta. Recomendacao: substituir por curriculum dinamico guiado por perplexidade/uncerteza do proprio modelo, ou ao menos validar empiricamente que os pesos 5x ajudam (A/B em 600M).

### 2.10 (Bonus) SWA / "Averaging Weights Leads to Wider Optima" — Izmailov & Podoprikhin (UAI 2018)
Stochastic Weight Averaging com custo quase zero mantem uma media das pesos ao final do treino, encontrando optima mais largos e melhor generalizacao. Trabalhos em RL mostram que SWA melhora estabilidade de treino. **Aplicacao:** F51 nao tem nenhuma forma de weight averaging. Manter `model_ema` com decay 0.999 em boundaries de ciclo e usar para eval/checkpoint estabilizaria o treino continuo e protegeria contra os danos de picos de loss (fraqueza 1.6) — o EMA filtra os pesos corrompidos por um spike.

### 2.11 (Bonus) Gradient Noise Scale — McCandlish, Kaplan, et al. (2018)
"An Empirical Model of Large-Batch Training". Modelo empirico que prediz o batch size otimo a partir da variancia do gradiente. Nao e rejeitado, mas e uma ferramenta subutilizada para escolher accumulation steps. **Aplicacao:** rodar uma sonda de ~50 passos com micro-batch=1 no 2.5B para estimar o `B_noise` e dai derivar `accum_steps`. Substituir o chute atual (batch=1, sem accum) por um valor justificado.

---

## 3. Recomendacoes concretas de patch (sem escrever codigo)

**Prioridade P0 — corrige dano ativo a linhagem 2.5B:**

- **R1. Preservar estado do optimizer em mudancas de topologia.** Hoje `run_cycle` reconstrui o AdamW do zero apos `execute_structural_actions`. Implementar `remap_optimizer_state(old_opt, old_named_params, new_named_params)` que (a) salva `state_dict()` do optimizer antigo, (b) mapeia por nome+forma os tensores de momento dos parametros inalterados, (c) inicializa so os novos params (neurogenese) e descarta os removidos (prune). Reaproveita a logica que ja existe em `migrate_adamw_state_with_new_baselines`. **Ataca fraqueza 1.1.**

- **R2. Adicionar warmup linear no resume/inicio.** Minimo viavel: ~100-200 steps de linear warmup de 0 ate `cfg.learning_rate` sempre que `start_step == 0` OU quando o optimizer for reconstruido. Guardar `warmup_remaining` no `training_state` do checkpoint para resumir fielmente. **Ataca fraqueza 1.2 (parcial).**

**Prioridade P1 — estabilidade e eficiencia:**

- **R3. Adicionar gradient accumulation.** Novo param `--accum-steps` (default derivado de `R11`). Accumular gradiente sobre N micro-batches, dividir loss por N, so chamar `optimizer.step()`/`clip_grad_norm` a cada N. **Ataca fraqueza 1.3.**

- **R4. Checkpoint assincrono.** Trocar o `torch.save(...)` bloqueante por: (1) copia CPU como hoje, (2) spawn de `multiprocessing.Process` (ou `torch.distributed.checkpoint.state_dict_saver.async_save`) que grava o `.tmp` e faz `os.replace`. Aguardar o save anterior antes de iniciar um novo (nao encadear dois async). **Ataca fraqueza 1.4.**

- **R5. Checkpoint de emergencia em nivel de step.** Registrar handler para `SIGTERM`/`SIGINT` que salva estado parcial no `step` corrente (mesmo que descarte o batch em andamento). Reduz max-loss-on-crash de `steps_per_cycle` para ~1 step. **Ataca fraqueza 1.5.**

- **R6. Conectar o brainstem ao loop de treino.** Em `_train_cycle`, apos computar `loss`, chamar `self.brainstem.check_loss(loss_val)`; se `not report.alive`, fazer `optimizer.zero_grad(set_to_none=True)` e `continue` (pular o step) em vez de backpropagar o pico. **Ataca fraqueza 1.6.**

**Prioridade P2 — migracoes de optimizer/replay (validar no 600M primeiro):**

- **R7. Avaliar Schedule-Free AdamW** como substituto do AdamW+flat-LR. Compativel com treino continuo de horizonte aberto; remove a classe inteira de bugs de schedule. **Ataca fraqueza 1.2 (raiz).**

- **R8. Avaliar `bnb.optim.AdamW8bit`** para cortar ~15 GB de VRAM no 2.5B e liberar espaco para batch/contexto maiores. Drop-in. **Ataca fraquezas 1.3/1.4 indiretamente.**

- **R9. Migrar replay para DER++.** Estender `ReplayExample` para incluir logits (ou top-k logits), adicionar termo de distilacao na loss quando o batch vem do replay. **Ataca fraqueza 1.7.**

- **R10. Adicionar EMA/SWA shadow.** Manter `model_ema` com decay 0.999, atualizado apos cada `optimizer.step()`; usar `model_ema` para eval e embutir no checkpoint como `ema_state_dict`. Custo: +1 copia do modelo em VRAM (ou CPU) — combinar com R8 para caber. **Ataca estabilidade geral e protege contra 1.6.**

**Prioridade P3 — diagnostico:**

- **R11. Medir gradient noise scale uma vez.** Probe de ~50 passos micro-batch=1 no 2.5B para estimar `B_noise`; dai escolher `accum_steps = max(1, round(B_noise / (batch*block)))`. Justifica o valor usado em R3 com dados, nao chute.

- **R12. Validar curriculum `document_sample_weight`.** A/B test em 600M: treinar com pesos atuais vs pesos uniformes por 5k steps, medir perplexidade em holdout balanceado. Se pesos manuais nao ajudarem (ou ajudarem), decidir com dados. **Ataca fraqueza 2.9.**

---

## 4. Risco e ordem sugerida

1. **R1 + R2 primeiro** (P0): sao patches cirurgicos no arquivo que ja existe, sem nova dependencia, e param o sangramento da linhagem 2.5B.
2. **R4 + R5 + R6** (P1): ganho de robustez/throughput imediato, baixo risco, sem mudar matematica de treino.
3. **R3 + R11** (P1/P3): ganho de qualidade do gradiente; medir antes de comprometer accum_steps.
4. **R7/R8/R9/R10** (P2): validar cada um no 600M antes de adotar em 1.6B/2.5B; sao mudancas de matematica e merecem benchmark controlado (o proprio projeto ja tem `BENCHMARK_REFERENCE` para isso).

## Fontes

- Schedule-Free: https://arxiv.org/abs/2405.15682 , https://proceedings.neurips.cc/paper_files/paper/2024/file/136b9a13861308c8948cd308ccd02658-Paper-Conference.pdf
- 8-bit Optimizers: https://arxiv.org/abs/2110.02861 , https://github.com/bitsandbytes-foundation/bitsandbytes
- Sophia: https://arxiv.org/abs/2305.14342 , https://openreview.net/forum?id=3xHDeA8Noi
- ZeRO-Offload: https://www.usenix.org/system/files/atc21ren-jie.pdf , https://www.deepspeed.ai/tutorials/zero-offload/
- DER++: https://proceedings.neurips.cc/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf
- Muon: https://kellerjordan.github.io/posts/muon/ , https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html
- Lion: https://arxiv.org/abs/2302.06675
- Async checkpointing: https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html , https://pytorch.org/blog/6x-faster-async-checkpointing/
- Limits of Curriculum Learning for LLMs: https://openreview.net/forum?id=sHn5rq6L0O
- SWA (Izmailov): https://www.semanticscholar.org/paper/Averaging-Weights-Leads-to-Wider-Optima-and-Better-Izmailov-Podoprikhin/b8989afff14fb630ca58b6afa917fb42574228ee
- Gradient Noise Scale: McCandlish, Kaplan, Amodei (2018), "An Empirical Model of Large-Batch Training" — OpenAI arXiv:1812.06162
