# Darwin Active Gradient Engine V1 — Design

**Date:** 2026-07-16  
**Status:** Approved by Marco for implementation  
**Scope:** Darwin-X 1.6B cloud fresh-start on the governed full corpus

## 1. Objective

Train one Darwin-X 1.6B organism in one CUDA process while making the existing
mutational-gradient signals causally affect optimization.  All organs remain
available in the full condition, but every intervention is observable,
checkpointable, bounded, and independently disableable for ablation.

The first production target is the RTX PRO 6000 95 GB instance with the fully
validated corpus:

- path: `/workspace/01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast_v2.bin`;
- tokens: `18,548,972,689` int32 tokens;
- bytes: `74,195,890,756`;
- SHA-256: `9677e9f22f4d78efa7b25c77b2da3cbdb5fb2a499926ac641a75c13b65e25cff`.

## 2. Current Failure Being Corrected

The current backward hook records expert gradient geometry per microbatch and
creates shadow proposals.  The training loop applies those proposals only
after `optimizer.step()`, so they cannot affect the gradients that produced
them.  Nitro can also move an expert back from CPU to CUDA without moving the
optimizer moments for that parameter, producing a CPU/CUDA device mismatch.

Remote configuration keys cannot solve either problem when the model config
does not declare and consume them.

## 3. Architecture

### 3.1 One process and one optimizer boundary

DAE-V1 runs inside the existing organism process.  It does not launch a second
trainer or a second CUDA context.  Backward hooks only collect detached
statistics.  A new optimizer-boundary method consumes the accumulated
observations once, after the last microbatch and before gradient clipping and
`optimizer.step()`.

### 3.2 Active gradient actions

DAE-V1 supports three shape-preserving actions in V1:

- `IGNORE`: set the selected expert gradients to zero for this optimizer step;
- `PROTECT`: remove the component parallel to the expert's protected gradient
  direction, using a bounded blockwise projection derived from the existing
  normalized sketch;
- `UPDATE`: multiply the selected expert gradients by a bounded gain computed
  from magnitude, stability, novelty, residual and hormonal state.

The actions are deterministic for a fixed model state and batch.  Gains and
projection coefficients are clamped.  Non-finite inputs fail closed and leave
the original gradient unchanged while recording an error metric.

`EXPAND`, `PRUNE` and `CREATE` remain topology-boundary operations.  They may
be proposed during training but can execute only between cycles, after a
holdout/replay gate, with optimizer-state remapping and a topology checkpoint.

### 3.3 Accumulation semantics

Microbatch hooks aggregate observations but do not increment the mutational
decision clock independently.  DAE-V1 commits one observation and one action
decision per optimizer step.  This prevents `accum_steps=16` from making the
organism appear to experience sixteen independent evolutionary events.

### 3.4 Optimizer strategy

V1 introduces an optimizer factory and a checkpointed optimizer identity.

- `adamw` remains the compatibility and scientific-control optimizer;
- `dae_hybrid` is the active candidate: spectral normalized updates for matrix
  parameters and reduced-state adaptive updates for embeddings, vectors and
  normalization parameters;
- optimizer selection is explicit in configuration and CLI output;
- resume rejects a mismatched optimizer identity unless `--fresh-start` is
  explicit;
- topology remapping must preserve compatible state by parameter name.

The first canary may use `adamw` with DAE actions to isolate the gradient engine
from the new optimizer.  The full candidate is promoted only after both paths
pass identical numerical and checkpoint tests.

### 3.5 Nitro device invariant

`nitro_enabled` becomes a real `DarwinXConfig` field.  On the 95 GB cloud
profile it is false and every expert remains resident on CUDA.  When Nitro is
enabled elsewhere, restoring an expert must also migrate every tensor in that
parameter's optimizer state to the parameter device before the next step.

The runtime asserts before every optimizer step that each trainable gradient,
parameter and materialized optimizer-state tensor share a device.  A mismatch
aborts before weights are mutated.

### 3.6 Organs and ablations

The full condition keeps Soul, Ghost, replay, JEPA, MTP, Curiosity,
Spider-Sense, Brainstem, neuroendocrine state and structural proposals active.
Each organ receives an explicit condition flag and emits a ledger entry so the
benchmark runner can compare:

1. static AdamW control;
2. shadow-only current behavior;
3. active `IGNORE/PROTECT/UPDATE` with AdamW;
4. hybrid optimizer without DAE actions;
5. hybrid optimizer plus DAE actions;
6. full organism;
7. one-organ-at-a-time ablations.

No condition runs concurrently on the same GPU.

## 4. Observability and Checkpoint Contract

Every optimizer step records JSONL fields for optimizer identity, action counts,
gradient norm before/after, projection magnitude, gain, non-finite rejection,
step latency, tokens/second and peak VRAM.  Checkpoints persist:

- optimizer type, version and state;
- DAE configuration and decision counters;
- mutational buffers and protected directions;
- Nitro placements;
- topology, corpus identity and base-checkpoint identity;
- enabled organ/ablation condition.

Resume must reproduce the next CPU test step within its documented numerical
tolerance.

## 5. Safety and Launch Gates

No cloud training launch is allowed until all gates pass:

1. PowerShell/Python syntax and CPU unit tests;
2. active-action causal tests proving gradients change before the step;
3. fail-closed device-invariant tests;
4. checkpoint/resume equality for both optimizer families;
5. cloud dry inspection confirming one process, full corpus and idle GPU;
6. short fresh-start canary with finite loss, stable VRAM and a saved canary
   checkpoint;
7. heldout/replay comparison against the static control.

The canary never promotes a checkpoint automatically.  A full `run247` starts
only after the canary ledger is complete and one-process exclusivity is
rechecked.

## 6. Success Criteria

DAE-V1 is ready for the long run when:

- exactly one trainer owns the GPU;
- the full corpus and manifest hashes match the approved identities;
- no trainable tensor or optimizer state crosses CPU/CUDA unexpectedly;
- active actions are causally demonstrated, bounded and reversible;
- loss stays finite and checkpoint/resume works;
- the full condition is no worse than the static control on heldout/replay
  beyond the declared tolerance;
- all claims are backed by the experiment ledger, not training loss alone.
