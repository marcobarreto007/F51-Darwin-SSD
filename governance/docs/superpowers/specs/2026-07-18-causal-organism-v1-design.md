# F51 Causal Organism v1 — Design

## Objective

Transform the F51 organism from a mixture of causal modules, sidecars, and
narrative telemetry into one measurable closed loop:

```text
raw evidence -> typed signal -> bounded intervention -> applied mutation
             -> independent outcome -> persisted causal credit
```

An organ is not considered connected because it was instantiated, called, or
serialized. It is connected only when an ablation can prove that its
intervention changed an allowed training target and that the resulting update
survived an independent heldout/replay gate.

## Current evidence

The live 100M path proves real LM/MoE/replay training, backward, AdamW updates,
checkpoint publication, and structural mutation at cycle boundaries. The
following failures prevent the organism from being a causal whole:

- `aux_loss_scale` is persisted but ignored by the objective;
- Heartbeat/TTM writes and retrieves memory, but training logits do not consume
  retrieved memory;
- JEPA is computed with zero weight in the 100M lineage;
- Spider-Sense is evaluated after logits/loss and has no calibration loss;
- Decision Engine is not instantiated in the 100M forward;
- Sleep replay is not integrated and REM pruning can mutate weights without a
  before/after rollback gate;
- Unified Mesh receives empty signals at the lifecycle boundary and its
  mutation API disagrees with the neuroendocrine decision schema;
- structural authority is duplicated between the model action queue and the
  second death loop;
- Soul commands mostly log intentions instead of issuing bounded, measurable
  actions.

The current pointer targets the 100M cycle 101 while the canonical status
document still describes the older 1.6B cycle 71 state. This design does not
select or rewrite lineage. It builds a new training contract that must be
migrated explicitly.

## Architecture

### 1. Versioned loss composer

`src/f51_darwin/darwin_x_core/losses.py` owns the only legal composition of raw
loss terms.

Inputs:

- LM loss;
- MTP loss and configured scale;
- JEPA loss and configured scale;
- raw MoE auxiliary loss and configured scale;
- Ghost loss and configured scale;
- optional Spider calibration loss and configured scale.

Outputs:

- immutable raw terms;
- immutable effective terms;
- total objective;
- `loss_semantics_version`.

The raw MoE auxiliary loss remains observable. `aux_loss_scale=0.0` produces an
exact zero contribution even when the raw loss is non-finite, avoiding
`0 * NaN`. Adaptive scaling may only reduce a non-zero configured
contribution.

### 2. Synchronous typed causal bus

`src/f51_darwin/organism/causal_bus.py` defines:

- `StepIdentity`;
- `Phase`;
- typed `Signal`;
- allowlisted `Intervention`;
- `Decision`;
- `StepOutcome`;
- `AblationArm` (`CONTROL`, `SHADOW`, `APPLY`).

Organs never receive mutable references to the model or optimizer. They emit
pure proposals. The training loop is the only executor.

Initial intervention operations:

- `SET_LOSS_TERM_SCALE`;
- `SKIP_UPDATE`;
- `SET_GRAD_CLIP`;
- `SCALE_GRADIENT_GROUP`;
- `QUEUE_STRUCTURAL_ACTION`.

Conflicting interventions on the same target reject the decision group. Every
intervention expires after one optimizer window or one cycle boundary.

### 3. Causal ledger

`src/f51_darwin/organism/causal_ledger.py` writes canonical, hash-chained JSONL
under `workspace/runtime/organism/causal_ledger/`.

Every optimizer attempt has two events:

1. `STEP_INTENT`, persisted before mutation;
2. `STEP_OUTCOME`, persisted after mutation.

Outcome evidence includes raw/effective losses, gradient norms before/after,
whether the optimizer advanced, parameter-delta evidence, heldout/replay
snapshots, degraded organs, and exceptions. “No intervention” is also
recorded.

### 4. Single structural boundary

Only the model's deferred structural action queue may mutate topology at
`CYCLE_BOUNDARY`. The legacy second death loop becomes a compatibility
observer and cannot mutate weights or experts.

Sleep respects `sleep_enabled`. The implemented sleep transaction snapshots
weights, optimizer, gradients, RNG, Heartbeat, and Python-side MoE runtime;
runs protected replay consolidation; evaluates fresh, replay, and heldout
losses with paired RNG; and rolls back on regression. Direct unguarded L1
zeroing is forbidden.

### 5. Organ migration

Organs migrate behind adapters in this order:

1. MoE auxiliary loss and brainstem;
2. DAE/neuroendocrine gradient actions;
3. Heartbeat/TTM residual retrieval;
4. JEPA and Spider calibration;
5. structural evolution;
6. sleep consolidation;
7. Soul commands (not migrated in v1).

Each migrated organ requires:

- pure signal adapter;
- negative control;
- intervention-effect test;
- independent primary metric;
- deterministic ablation plan;
- checkpoint roundtrip;
- rollback or bounded expiry.

## Research basis through 2026-07-18

- Switch Transformer and ST-MoE establish explicit, separately measurable
  load-balancing and router-z terms.
- DeepSeek's auxiliary-loss-free routing uses a non-gradient expert bias; it
  is not equivalent to setting an auxiliary coefficient to zero.
- Titans and ATLAS motivate surprise-driven neural memory but do not justify
  unmeasured memory injection.
- 2026 sleep/consolidation work motivates protected replay and distillation,
  not blind magnitude pruning.
- Bilevel/gradient-alignment work motivates treating LM/heldout quality as
  primary and auxiliary organs as subordinate interventions.

These methods are inputs, not authorities. F51 adopts only mechanisms that
survive local causal tests.

## Checkpoint contract

Bus-enabled training is checkpoint v8:

- `causal_contract`;
- `causal_bus_state`;
- `ledger_head`;
- `loss_semantics_version`;
- `training_contract_id`.

`base_checkpoint_id` retains its current meaning. A v7 checkpoint may load in
legacy mode. Migrating v7 to the causal objective requires an explicit
contract-migration event and a new v8 checkpoint. No filename, mtime, pointer,
or cycle number is sufficient lineage proof.

## Verification

Three ablation arms always start from the same checkpoint, optimizer state,
RNG state, and data cursor:

- `CONTROL`: bus bypassed;
- `SHADOW`: identical proposals recorded but not applied;
- `APPLY`: proposals applied.

Required proof:

- control and shadow have identical parameter deltas;
- negative-control interventions have zero effect;
- known interventions produce the predicted gradient/parameter delta;
- improved total loss caused only by reweighting is never labeled model
  improvement;
- heldout and replay gates decide promotion;
- tampering with the ledger is detected;
- crash recovery detects an orphan intent;
- v7 legacy and v8 strict resume are both explicit and tested.

## Non-goals

- No concurrent `run247`;
- no cloud authority;
- no deletion or selection of checkpoints;
- no direct external ingestion;
- no claim that every biological metaphor is useful;
- no structural mutation without one safe boundary and rollback evidence.
