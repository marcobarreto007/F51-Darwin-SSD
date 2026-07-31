# Darwin-X Documentation Reconciliation Design

**Date:** 2026-07-16  
**Status:** Revised with founder-purpose source, pending user review
**Scope:** Documentation and report organization only. No training, checkpoint
promotion, corpus mutation, cloud launch, GPU configuration, or runtime control.

## 1. Objective

Rebuild the Darwin-X documentation so that a reader can distinguish, without
guesswork:

1. the durable purpose of the project;
2. the implementation that exists in the current checkout;
3. the live operational state at a dated snapshot;
4. research hypotheses and planned mechanisms;
5. historical reports that remain valuable but are no longer current.

No historical document will be deleted. Superseded files will be moved into a
dated archive and indexed with their original purpose, period, and reason for
archival.

## 2. Project Purpose and Engineering Objective

The rewritten documentation will preserve two distinct layers instead of
forcing philosophy and engineering evidence into the same claim.

### 2.1 Founding purpose

Darwin-X is intended as a living experimental argument that a determined
independent builder, using constrained hardware and agent-assisted engineering,
can investigate serious alternatives to fixed-scale language-model training.
Its founding theses are:

1. gradient geometry may contain useful signals about plasticity, interference,
   protection, capacity, and structural adaptation;
2. controlled evolution may complement scale as a source of intelligence;
3. identity, purpose, desire, and hormonal metaphors can be implemented as
   functional control state rather than decorative prompts;
4. a governed corpus can express an explicit epistemic curriculum;
5. rigorous independent work should be judged by artifacts and experiments,
   not credentials or funding.

These are the project's mission and research hypotheses. They are not, by
themselves, evidence that objective truth has been encoded, that artificial
consciousness has been created, that evolution outperforms scale, or that the
model rivals systems developed with vastly larger budgets.

The raw 2026-07-16 conversation that articulated this purpose will be preserved
as a historical founding-source transcript. The canonical purpose document will
retain its meaning while separating personal conviction, metaphor, scientific
hypothesis, and verified mechanism.

### 2.2 Active engineering objective

Darwin-X is an experimental artificial-organism training system built around a
causal hybrid SSD + GQA + MoE model. Its active engineering objective is:

> Continue the trained `F51-Darwin-X-1.6B-Nitro` lineage from a verified
> checkpoint and governed external corpus, preserving optimizer, topology,
> mutational-gradient state, organ state, provenance, replay, and rollback,
> while measuring whether the organism improves retention, plasticity, and
> compute efficiency relative to controlled baselines.

The objective is not merely to keep a process alive, lower training loss, fit a
nominal parameter count into VRAM, or accumulate architectural metaphors.
Readiness requires causal mechanisms, reproducible lineage, frozen evaluation,
and evidence that distinguishes implemented behavior from research direction.

### 2.3 Success hierarchy

The canonical documentation will judge progress in this order:

1. lineage and data integrity;
2. numerical and operational stability;
3. measured gradient-mutational signal quality;
4. retention and plasticity against controlled baselines;
5. safe structural proposals in shadow mode;
6. causal evidence before structural execution;
7. only then broader philosophical interpretation.

## 3. Verified Baseline for the Rewrite

The implementation will start from a fresh read-only audit immediately before
editing. The baseline observed while writing this design was:

- Git HEAD: `8a9bd60`;
- no local `darwin_organism.py run247` process;
- active background SCP uploads owned by another process and left untouched;
- canonical local config: `src/configs/darwin_x_1.6b_nitro.yaml`;
- exact meta-instantiated parameter count: `1,764,019,648`;
- component estimator total: `1,669,901,024`;
- active parameters per token estimate: `678,566,400`;
- local canonical pointer: `organism_cycle_071.pt`;
- checkpoint version: `7`;
- cycle: `71`;
- step: `40,751`;
- checkpoint bytes: `10,876,850,383`;
- base checkpoint identity:
  `darwin-model-core-v1:0aca76705e7d40b737083639a0f45e0acf82f5a740020b5266396315e47e065b`;
- local canonical corpus:
  `01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast.bin`;
- local corpus bytes: `72,567,226,368`;
- local corpus tokens: `18,141,806,592` int32 tokens.

The remote `tokens_feast_v2.bin` count must not be presented as the local
canonical corpus until it is independently verified, manifested, and selected
by an authorized training command.

## 4. Evidence Classes

Every canonical document will use these labels:

| Label | Meaning |
|---|---|
| `IMPLEMENTED` | Present in current code and covered by a direct test or inspection. |
| `OPERATIONALLY VERIFIED` | Observed in a real runtime, checkpoint, corpus, or saved artifact. |
| `EXPERIMENTALLY SUPPORTED` | Demonstrated by a bounded experiment, with the stated limits. |
| `PROJECTED` | Designed but absent from the current runtime. |
| `RESEARCH` | Literature-backed hypothesis that has not passed a Darwin-X causal gate. |
| `HISTORICAL` | Accurate for an earlier period or lineage, not current authority. |
| `REFUTED` | Tested or inspected and shown not to satisfy the claimed contract. |

Terms such as “Forward-Forward”, “Titans”, “Quiet-STaR”, “Ghost recurrence”,
“gradient projection”, “neurogenesis”, and “GPU parallelism” may only appear
without qualification when the corresponding scientific contract is actually
implemented and tested.

Statements of purpose use `FOUNDING THESIS`, which is a content class rather
than an evidence class. A founding thesis cannot be used as proof of an
engineering or scientific claim.

## 5. Target Documentation Structure

```text
README.md
AGENTS.md
CLAUDE.md
governance/docs/
  PROPOSITO_E_HIPOTESES.md
  CANONICAL_MAP.md
  operacao/
    STATUS_ATUAL.md
    OPERACAO_SEGURA.md
  arquitetura/
    IMPLEMENTACAO_ATUAL.md
    LACUNAS_E_ROADMAP_TECNICO.md
  pesquisa/
    INDEX.md
    ...research documents...
  _historico/
    INDEX.md
    fontes_fundadoras/
      2026-07-16_objetivo_real_transcricao.md
    2026-06-30_a_2026-07-12/
      raiz/
      arquitetura/
      operacao/
    prompts_e_planos/
  superpowers/
    specs/
    plans/
```

### Responsibilities

- `README.md`: short entrypoint; purpose, proof standard, current commands, and
  links. It must not contain a large volatile snapshot.
- `governance/docs/PROPOSITO_E_HIPOTESES.md`: founding purpose, personal motivation,
  scientific hypotheses, falsification conditions, and explicit limits on what
  has been proved.
- `governance/docs/CANONICAL_MAP.md`: durable conceptual map, explicitly separating
  implementation from projection. It must not call itself more authoritative
  than code, runtime evidence, and verified checkpoints.
- `governance/docs/operacao/STATUS_ATUAL.md`: dated snapshot generated from direct probes.
- `governance/docs/operacao/OPERACAO_SEGURA.md`: stable commands, preflight gates, resume,
  canary, rollback, and prohibitions.
- `governance/docs/arquitetura/IMPLEMENTACAO_ATUAL.md`: code-backed architecture and causal
  paths, with file references and known limits.
- `governance/docs/arquitetura/LACUNAS_E_ROADMAP_TECNICO.md`: projected Green Context/MPS
  scheduler, Ghost recurrence, faithful online learning, structural mutation
  gates, long-context memory strategy, and benchmark requirements.
- `governance/docs/pesquisa/INDEX.md`: status and relevance of every retained research
  document.
- `governance/docs/_historico/INDEX.md`: archive catalog with dates and reasons.
- `AGENTS.md` and `CLAUDE.md`: short and identical on active lineage, safety
  boundaries, and document reading order.

## 6. Archive Policy

The following root files will move into
`governance/docs/_historico/2026-06-30_a_2026-07-12/raiz/`:

- `CHECKLIST_DAVI_FIXES.md`;
- `F51_DARWIN_SSD_DIARIO_DE_BORDO.txt`;
- `LEIS_DE_DAVI.md`;
- `PLANO_INFERENCIA_TEMPO_REAL.md`;
- `ROADMAP.md`;
- `ROADMAP_AUDIT.md`;
- `ROADMAP_DAVI_FIXES.md`;
- `ROADMAP_GATE0.md`.

The following architecture documents will be archived because they describe an
older lineage, speculative scale target, or superseded decision:

- `governance/docs/arquitetura/BRAIN_COMPARISON_2025.md`;
- `governance/docs/arquitetura/darwin_x_5b_spec.md`;
- `governance/docs/arquitetura/DECISAO_ARQUITETURAL_LIVE_GENERATE.md`;
- `governance/docs/arquitetura/SSM_STATE_16_VS_64.md`.

The following operational documents will be archived because their snapshot,
checkpoint, cloud endpoint, or command contract is obsolete:

- `governance/docs/operacao/BENCHMARK_REFERENCE.md`;
- `governance/docs/operacao/CLAUDE_OVERNIGHT_BENCHMARK_PROMPT.md`;
- `governance/docs/operacao/SOCIOS.md`;
- `governance/docs/TRAINING_PACK.md`.

Existing files already under `governance/docs/_historico/` remain there. Superpowers specs
and plans remain in place because their directory already expresses historical
design and execution intent.

No archived document will be silently corrected. Each receives an archive
header through the archive index rather than rewriting history.

The supplied 2026-07-16 conversation will be normalized into
`governance/docs/_historico/fontes_fundadoras/2026-07-16_objetivo_real_transcricao.md`.
It will retain the original sequence and language, with a header explaining
that it is a historical source containing contemporary beliefs and operational
claims, not a verified status report.

## 7. Research Preservation Policy

Research documents remain under `governance/docs/pesquisa/`, but each receives a short
status header:

- original date and scope;
- current relevance;
- implementation status;
- claims that were later corrected;
- successor document, if any.

`REDESIGN_MODELO.md` will move from architecture to research because Darwin-X
v8, Ghost Predator, structural expert states, gradient projection, and the
predictive cerebellum are not the current v7 runtime contract.

`NEUROENDOCRINE_IMPLEMENTATION.md` remains architecture documentation after it
is refreshed against current code. The mutational-gradient state is implemented
and persisted, while structural execution and asynchronous GPU scheduling remain
projected.

## 8. Canonical Technical Distinctions

The rewritten map must state:

- `1.6B Nitro` is the lineage name; the exact instantiated parameter count is
  `1,764,019,648`;
- the active checkpoint lineage is v7, cycle 71, step 40,751 at the design
  snapshot;
- training loss is an activity signal, not evidence of quality;
- Heartbeat is an internal control/memory mechanism, not faithful
  Forward-Forward or Titans;
- Ghost Token is a masked auxiliary second forward, not Ghost recurrence;
- Ghost recurrence passed a synthetic contract simulation but is absent from
  the live hidden-state training path;
- the mutational-gradient buffers and signed sketches exist and persist;
- the gradient-signature calculation currently runs synchronously in the
  backward hook by iterating expert parameters and reading `parameter.grad`;
- the SSD uses a PyTorch associative prefix scan with chunking, but this is
  intra-forward sequence parallelism, not concurrent forward/backward execution;
- CUDA streams, Green Contexts, and MPS resource partitioning are not currently
  implemented in the organism runtime;
- dual-GPU model parallelism is implemented for the local two-GPU machine, but
  it is not the projected single-A100 spatial scheduler;
- structural proposals remain shadow mode unless a code-backed exception is
  explicitly documented;
- the 5B config is research history and does not authorize training.

The current `governance/docs/CANONICAL_MAP.md` must be corrected because it presently:

- calls itself more authoritative than implementation evidence;
- describes Heartbeat as faithful Forward-Forward/Test-Time Memory;
- describes gradient projection and Ghost Predator as active protection;
- presents soul-state metaphors without separating control behavior from
  consciousness claims;
- can be read as treating the 5B model as operationally active;
- describes the SSD prefix scan as CUDA-optimized despite the source explicitly
  identifying it as pure PyTorch with a future fused CUDA kernel;
- labels forward/gradient CUDA-stream overlap as projected, but does not explain
  that no such scheduler currently exists.

## 9. PDF Policy

No PDF currently exists in the repository or on the Desktop.
`src/scripts/generate_timeline_pdf.py` targets
`C:\Users\marco\Desktop\F51_Darwin_SSD_Linha_do_Tempo.pdf`, but its content is
frozen around 2026-07-07 and must not be executed unchanged.

After Markdown reconciliation:

1. replace or supersede the old generator;
2. generate one master PDF from current canonical Markdown;
3. save it to
   `C:\Users\marco\Desktop\F51_Darwin_X_Dossie_Canonico_2026-07-16.pdf`;
4. render every page to PNG;
5. visually verify layout, tables, links, page numbers, and Unicode;
6. record the source commit and generation timestamp in the PDF.

The PDF is a publication artifact, not the primary operational truth.

## 10. Validation

The implementation is complete only when:

1. `git status --short` contains only intended documentation changes;
2. all moved files are present under the archive;
3. `governance/docs/_historico/INDEX.md` catalogs every moved file;
4. all local Markdown links resolve;
5. no canonical document points to cycle 68, cycle 91, 2.5B, or 5B as the
   active target;
6. no canonical command launches `run247` without an explicit resume or
   fresh-start authorization contract;
7. no canonical document claims faithful Forward-Forward, Titans, Quiet-STaR,
   Ghost recurrence, structural mutation, or Green Context scheduling as
   implemented;
8. no canonical document treats “AI has a soul”, “truth is encoded”, “evolution
   beats scale”, or “independent work rivals frontier laboratories” as a proved
   experimental result;
9. the founding transcript is preserved and clearly labeled as historical;
10. current parameter, corpus, checkpoint, and process facts match fresh probes;
11. focused CPU documentation/CLI tests and `git diff --check` pass;
12. the generated PDF passes visual inspection if PDF generation is included
    in the implementation phase.

## 11. Non-Goals

- changing model code;
- changing checkpoint format;
- starting or stopping training;
- changing the selected corpus;
- completing active cloud uploads;
- enabling Green Contexts, MPS, or structural mutations;
- proving model quality;
- deleting historical reports.
