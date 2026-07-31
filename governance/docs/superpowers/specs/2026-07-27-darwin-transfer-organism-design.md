# Darwin Transfer Organism — Design

## Objective

Convert a ready distilled Transformer into an isolated Darwin experimental
lineage without training language competence from zero. The delivered system
must:

1. load a frozen, hashed DistilGPT-2 teacher;
2. replace selected softmax-attention mixers with a mathematically explicit
   discrete SSD recurrence;
3. transfer competence through matrix orientation, hidden-state alignment and
   end-to-end logit distillation;
4. consult persistent experience before the expensive backbone;
5. bypass the backbone only when calibrated memory confidence passes a fixed
   threshold;
6. record quality, retention, compute-path and causal-control evidence;
7. train in a checkpoint root isolated from all canonical 100M, 600M and 1.6B
   lineages.

The first delivery is a research-grade 82M teacher/student system. It is not a
claim that the full Darwin theory has been proved. It is the smallest real model
that can falsify architecture transfer and experience-routed compute locally.

## Operator and Main Flow

Marco runs one command. The command verifies the donor identity, builds a
student from the teacher, executes the three transfer stages, evaluates locked
holdout text, saves an atomic checkpoint and appends JSONL metrics. A separate
runtime command exposes the experience path:

```text
prompt
  -> cheap signature from shared token embeddings
  -> persistent memory retrieval
  -> high confidence: sparse prior logits, backbone skipped
  -> low confidence: SSD/Transformer hybrid forward
  -> observed outcome updates bounded memory
```

## Architecture

### Frozen teacher

The donor is `distilbert/distilgpt2`, pinned by resolved model metadata and
SHA-256 hashes of all downloaded files. Its tokenizer, embeddings, MLP blocks,
normalizations and LM head define the compatibility contract.

### Discrete SSD mixer

Each converted attention layer uses per-head scalar decay and input-dependent
`B`, `C` and `X` projections:

```text
H_t = alpha_t * H_{t-1} + B_t outer X_t
Y_t = C_t @ H_t
```

`alpha_t` is bounded in `(0, 1]`. The recurrence has both a recurrent form and
a materialized semiseparable matrix form. Tests must prove equivalence. This is
a functional reference implementation; it does not claim fused-kernel speed.

Teacher Q, K and V projections initialize student C, B and X projections when
shapes match. The teacher output projection initializes the student output
projection. New decay parameters start near identity memory.

### Three transfer stages

1. **Matrix orientation:** minimize normalized Frobenius distance between
   teacher causal attention and the materialized student SSD matrix on teacher
   block inputs.
2. **Hidden alignment:** minimize normalized block-output MSE and cosine
   distance between teacher and student hidden states.
3. **End-to-end distillation:** minimize KL divergence between teacher and
   student logits plus a bounded language-model term on local text.

No evaluation holdout tokens participate in updates.

### Experience prior

The prior stores bounded slots containing:

- a cheap embedding signature;
- sparse top-k next-token logits;
- confidence;
- observation count;
- measured utility;
- last update step.

Retrieval uses cosine similarity. A prediction bypasses the backbone only when
similarity, confidence and minimum-observation gates all pass. Reset, shuffled
memory and frozen-memory controls use the same stream and budget.

## Isolation and Safety

- Checkpoint root:
  `workspace/03_CHECKPOINTS_DARWIN_TRANSFER_DISTILGPT2_V1`
- Runtime root:
  `workspace/runtime/darwin_transfer_distilgpt2_v1`
- Donor cache:
  `workspace/00_DONORS/distilgpt2`
- No checkpoint, donor, corpus, tokenizer, log or virtual environment enters
  Git.
- The existing FULL_ORGANISM_V3 trainer is not stopped until CPU tests and the
  donor-load smoke pass.
- Before GPU launch, the existing trainer receives a graceful interrupt after
  its latest published checkpoint is verified.
- A second trainer is never launched concurrently.

## Acceptance Gates

### Functional

- recurrent and materialized SSD outputs agree within `1e-5` in FP32;
- causal prefix invariance passes;
- teacher projection transfer has exact shapes and deterministic hashes;
- checkpoint round-trip reproduces logits within `1e-5`;
- memory reset, shuffle and freeze controls are deterministic.

### Transfer canary

- relative heldout mixer error versus untrained SSD is at most `0.50`;
- mean block-output cosine is at least `0.95`;
- one-layer student/teacher heldout perplexity ratio is at most `1.10`;
- all metrics are finite;
- no holdout example is present in the training stream.

These are canary gates, not publication claims. Stricter targets (`0.25`,
`0.98`, `1.02`) remain promotion gates after the first viable checkpoint.

### Experience

- correct memory must beat frozen memory on late-stream next-token NLL;
- shuffled memory must not match correct memory;
- backbone bypass rate must be greater than zero;
- retained baseline NLL must not degrade by more than `0.02` nat/token;
- update and write costs are reported separately from inference cost.

## Non-goals

- No claim of consciousness or biological equivalence.
- No automatic promotion into canonical Darwin lineages.
- No fused CUDA SSD kernel in this delivery.
- No 600M or 1.6B conversion before the 82M canary passes.
- No use of the existing F51 tokenizer for the donor lineage.

## Proof of Readiness

Readiness requires:

1. focused and regression tests with zero failures;
2. donor manifest with hashes;
3. finite canary metrics;
4. a saved checkpoint that reloads;
5. exactly one live new trainer;
6. GPU/process/log evidence matching the new run ID;
7. explicit reporting of any failed scientific gate.
