# Darwin Two-Donor Organ Transplant - Design

## Objective

Create a new Darwin recipient from two immutable donors:

1. **Language donor:** the original local DistilGPT-2 Transformer, which
   supplies embeddings, Transformer blocks, normalizations and language head.
2. **Organ donor:** the F51 `FULL_ORGANISM_V3` checkpoint at cycle 2, step
   1750, which supplies trained organ tensors, persistent organ state,
   topology and identity.
3. **Recipient:** a new isolated model that preserves the language donor
   exactly at initialization and receives the original organs through a
   dimensionally compatible organ bus.

This is a real transplantation experiment. It does not create replacement
organs from random initialization and does not call ordinary fine-tuning an
organ gain.

## Immutable Donor Contract

### Donor A - language competence

- Model: original DistilGPT-2 Transformer.
- Local root: `workspace/00_DONORS/distilgpt2`.
- Model class: `GPT2LMHeadModel`.
- Hidden width: 768.
- Layers: 6.
- Vocabulary: 50,257.
- Identity authority: `donor_manifest.json` and hashes of every donor file.

The donor files and weights are read-only. The recipient receives an exact
copy and records the donor manifest hash in every checkpoint.

### Donor B - Darwin organs

- Lineage: `FULL_ORGANISM_V3`.
- Checkpoint:
  `workspace/03_CHECKPOINTS_100M_FULL_ORGANISM_V3/organism_cycle_002_step_001750.pt`.
- Checkpoint version: 9.
- Training state: cycle 2, step 1750, 7,168,000 tokens seen.
- Hidden width: 512.
- Backbone layers: 12.
- Model state: 1,934 tensor entries.
- Identified organ material: 168 tensors and 10,457,157 parameters.

The transplant bundle includes trained tensors and non-tensor state where
present: JEPA, GABA, Spider Sense, inter-hemispheric processing, TTM residual
gate, neuroendocrine/ghost state, Heartbeat, DAE, replay state, topology and
organ identity.

The original checkpoint remains immutable. The bundle records its checkpoint
hash, base checkpoint identity, topology identity and per-organ tensor hashes.

## Recipient Architecture

### Canonical organ bus

Organs retain their original 512-dimensional interfaces. Receptor hidden
states enter and leave through organ-specific adapters:

```text
GPT hidden 768
    |
    +---------------- original residual -------------------+
    |                                                      |
    +-> input adapter 768->512 -> original organ -> output adapter 512->768
                                                        |
                                                   residual gate
                                                        |
                                                        v
                                              combined GPT hidden 768
```

The original organ tensors are copied bit-for-bit into the recipient. The
adapters are new recipient parameters. A transplant is rejected if any organ
tensor differs from its bundle hash before intentional unfreezing.

### Zero-impact initialization

Every organ residual gate starts at zero. With all gates at zero:

- recipient logits must match the original DistilGPT-2 logits within `1e-5`
  FP32;
- tokenization and generation behavior must remain unchanged;
- no organ state may mutate in baseline mode;
- the recipient is not considered viable if this equivalence fails.

This gives the recipient a safe baseline and prevents an unproven organ from
destroying language competence.

### Global organs

Global 512-dimensional organs attach to explicit receptor taps:

- JEPA consumes and predicts canonical organ-bus representations;
- Spider Sense reads the canonical representation and produces confidence;
- inter-hemispheric processing receives paired canonical streams;
- TTM supplies a gated residual or experience route;
- Heartbeat, Decision Engine, Sleep and Unified Mesh act through declared
  control interfaces rather than silently editing backbone tensors.

Every global organ has an independent adapter, gate, identity and ledger
entry.

### Layer-local organs

The 12 donor GABA modules remain separate and are assigned by normalized
depth to the six DistilGPT-2 blocks:

| Recipient block | Donor GABA modules |
|---|---|
| 0 | 0 and 1 |
| 1 | 2 and 3 |
| 2 | 4 and 5 |
| 3 | 6 and 7 |
| 4 | 8 and 9 |
| 5 | 10 and 11 |

The two modules use separate adapters and gates. Their residuals are combined
with bounded scaling; their weights are never averaged or sliced. This
preserves individual causality and rollback.

## Transplant Lifecycle

Each organ progresses independently through:

```text
candidate -> shadow -> adapter_active -> organ_unfrozen -> active
                                  \-> quarantine
active -> frozen
active -> quarantine
quarantine -> rollback
```

- `candidate`: extracted and hash-verified, gate zero.
- `shadow`: executes without changing receptor output.
- `adapter_active`: only its adapters and gate can learn.
- `organ_unfrozen`: selected organ tensors can learn at a lower learning rate.
- `active`: passed causal, retention, stability and compute gates.
- `frozen`: useful weights retained without further updates.
- `quarantine`: isolated after regression, non-finite state or identity error.
- `rollback`: restores the last accepted per-organ checkpoint.

No transition is inferred from training loss alone.

## Progressive Unfreezing Protocol

### Phase 0 - immutable baselines

Evaluate and freeze:

- original DistilGPT-2;
- a receptor copy with all organ gates zero.

The two must be equivalent before any training.

### Phase 1 - adapters only

- freeze the GPT backbone;
- freeze all transplanted organ tensors and persistent states;
- train one organ's input adapter, output adapter and residual gate;
- execute other organs in disabled or shadow mode;
- reject any undeclared gradient.

### Phase 2 - one organ at a time

- start from the same accepted recipient checkpoint;
- unfreeze exactly one organ at a low learning rate;
- run matched controls with identical tokens, order, seeds and compute budget;
- evaluate future holdout and old-knowledge retention;
- promote, refreeze, quarantine or roll back the organ.

### Phase 3 - interactions

- test accepted organ pairs;
- use randomized organ coalitions to estimate interaction effects and
  approximate Shapley contribution;
- detect organs that are useful alone but harmful together;
- preserve independent rollback boundaries.

### Phase 4 - complete Darwin recipient

- activate only organs whose individual or interaction evidence passed;
- perform selective joint unfreezing;
- run the complete retention, future-context, stability and compute tribunal;
- publish only an immutable recipient checkpoint with both donor identities.

## Four Matched Experimental Arms

Every organ decision uses four arms:

| Arm | Configuration |
|---|---|
| A - Simple GPT | immutable original DistilGPT-2 |
| B - Trained control | GPT receives the same tokens and compute without the organ |
| C - Frozen transplant | GPT plus frozen original organ and trainable adapters |
| D - Living transplant | GPT plus selectively unfrozen organ |

All arms use the same:

- corpus partitions and token order;
- training-token count;
- evaluation-token count;
- seeds;
- optimizer-step budget;
- context lengths;
- precision;
- hardware class;
- checkpoint starting point.

The trained control is mandatory. It separates ordinary adaptation from organ
contribution.

## Organ Gain and Loss

Primary causal gain:

```text
quality_gain_i = NLL(control B) - NLL(living transplant D_i)
```

Utility:

```text
utility_i =
    quality_gain_i
  + future_context_gain_i
  + calibrated_compute_saving_i
  - old_knowledge_forgetting_i
  - added_compute_cost_i
  - instability_penalty_i
```

Training loss is diagnostic only. An organ gains reputation only when it
improves future or retained behavior relative to the matched trained control.

The ledger records per organ:

- donor and tensor hashes;
- recipient checkpoint parent;
- lifecycle state;
- gate value and activation rate;
- train and holdout NLL deltas;
- future-context gain;
- old-knowledge retention delta;
- inference and update compute;
- non-finite or stability events;
- independent and coalition utility;
- confidence interval across seeds;
- promotion, freeze, quarantine and rollback decisions.

## State, Checkpoints and Rollback

The recipient uses a new isolated root:

`workspace/03_CHECKPOINTS_DARWIN_TWO_DONOR_V1`

Every checkpoint contains:

- Donor A manifest hash;
- Donor B checkpoint and organ-bundle hashes;
- receptor config identity;
- parent recipient checkpoint hash;
- per-organ state, adapter, gate and lifecycle status;
- organ ledger snapshot hash;
- exact experimental arm and budget;
- corpus/holdout contract identity;
- optimizer state only for declared trainable parameters.

Rollback is per organ. A failed organ does not require discarding accepted
changes from unrelated organs.

## Failure Handling

The runtime refuses to train or publish when:

- either donor identity differs;
- an organ tensor shape or hash is unexpected;
- zero-gate GPT equivalence fails;
- a gradient reaches an undeclared parameter;
- an organ produces non-finite output or state;
- holdout overlaps training data;
- arm budgets or seeds differ;
- the matched control artifact is missing;
- rollback cannot reproduce its accepted logits;
- a checkpoint lacks parent or organ identity.

Non-finite organs are gated off immediately and quarantined. Donor artifacts
are never overwritten.

## Test Strategy

### Extraction tests

- enumerate the expected organ tensors and persistent states;
- verify exact values, shapes and per-organ hashes;
- reject backbone tensors accidentally included in the organ bundle;
- round-trip the bundle without numerical change.

### Recipient identity tests

- load DistilGPT-2 from the verified donor;
- construct the recipient with all organs and zero gates;
- require logits within `1e-5` FP32 of simple GPT;
- verify no organ state mutation in baseline mode.

### Gradient and unfreezing tests

- adapter-only phase exposes gradients only in one organ's adapters and gate;
- single-organ phase exposes gradients only in that organ plus declared
  adapters;
- frozen organs and GPT remain bit-identical;
- rollback restores logits, tensors and persistent state.

### Causal-arm tests

- A/B/C/D receive identical tokens, starts, seeds and budgets;
- synthetic useful and harmful organs move utility in opposite directions;
- shuffled, frozen, reset and organ-off controls are deterministic;
- ordinary GPT fine-tuning cannot be credited to the organ.

### Integration tests

- transplant JEPA as the first global canary;
- transplant one GABA as the first layer-local canary;
- run individual and paired ablations;
- reload every accepted checkpoint;
- verify ledger reconstruction from immutable artifacts;
- prove no second trainer and no mutation of existing lineage roots.

## Initial Acceptance Gates

- exact donor identities and organ-bundle round-trip;
- zero-gate GPT equivalence within `1e-5`;
- no undeclared gradients;
- finite one-organ forwards and updates;
- A/B/C/D budget identity;
- at least three deterministic seeds for engineering canaries;
- 30 seeds for a scientific promotion claim;
- future holdout is isolated from all update streams;
- old-knowledge NLL regression no greater than `0.02` nat/token;
- positive lower confidence bound for organ utility;
- successful per-organ rollback;
- no publication while any required identity or control gate is red.

## Supported First Slice

The first implementation slice proves the transplant machinery with:

1. immutable donor manifests;
2. exact organ-bundle extraction;
3. zero-impact recipient construction;
4. JEPA global-organ transplant;
5. one GABA local-organ transplant;
6. A/B/C/D causal harness;
7. organ ledger and independent rollback.

Only after this slice passes do the remaining organs enter the transplant
queue.

## Non-Goals

- No random replacement JEPA.
- No direct slicing or padding of organ tensors.
- No claim that lower training loss proves an organ gain.
- No mutation of either donor.
- No immediate joint training of every organ.
- No replacement of the DistilGPT-2 backbone in this experiment.
- No promotion based on a four-batch or training-stream evaluation.
