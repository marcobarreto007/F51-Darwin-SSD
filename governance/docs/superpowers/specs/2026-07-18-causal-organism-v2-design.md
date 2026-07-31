# F51 Causal Organism v2 — Two-Timescale Design

## Objective

Build one causally measurable organism from the existing Darwin-X organs:
Soul, Decision Engine, Curiosity, GABA, Ghost Token, Ghost Brain, JEPA,
Spider-Sense, Heartbeat/TTM, MoE neuroendocrine control, structural evolution,
and protected sleep.

The v2 organism is complete only when every enabled organ has:

1. a real input produced by the supported runtime;
2. an immutable typed signal;
3. a bounded and allowlisted causal target;
4. a temporal barrier preventing same-step self-feedback;
5. checkpointed state with strict identity verification;
6. a negative control and an intervention-effect test;
7. a paired `CONTROL` / `SHADOW` / `APPLY` ablation;
8. an independent fresh/replay/heldout promotion gate.

Instantiation, logging, JSON telemetry, a config flag, or an unused loss term
does not count as connection.

## Supported system and operator flow

Darwin-X trains and serves local MoE language models for Marco. The supported
flow remains:

```text
approved feast_v2 corpus
    -> deterministic training attempt
    -> fresh/replay/heldout evaluation
    -> atomic checkpoint
    -> strict resume
    -> optional locally approved promotion
```

This design does not start `run247`, select a checkpoint, merge lineages, alter
the operator's `src/configs/darwin_x_100m.yaml`, or treat cloud state as authority.
The first operational gate remains the no-launch canary documented in
`AGENTS.md`.

## Current-state findings that v2 must correct

### Soul

- Soul receives real training metrics every 50 local steps.
- Its commands for ingestion, exploration, benchmark, emergency protection,
  and checkpointing currently terminate in log output.
- `heldout_loss` is required for the emergency command but is absent from the
  heartbeat status.
- token accounting supplies a cycle-accumulated count to a method that adds it
  again on every heartbeat.
- dopamine and awareness use partial external JSON persistence; their schemas
  do not restore the complete written state.
- the model checkpoint stores a blessing string, not the state that produced
  Soul's decisions.

### Decision Engine

- The module contains confidence factors and action thresholds but is not
  instantiated by the supported organism runtime.
- `decision_engine_enabled` changes model identity while leaving weights,
  logits, and forward behavior unchanged.
- the mesh probes a nonexistent `DarwinXOutput.decision_factors`, producing a
  permanent zero confidence signal.
- the only productive caller is an isolated legacy evolution loop not used by
  the supported entrypoints.
- an internal mathematics threshold self-test disagrees with the implemented
  threshold behavior.

### Curiosity

- Curiosity receives real hidden states and JEPA error, then reduces across
  batch and sequence together.
- its reward is a Python mapping, not a differentiable loss.
- `curiosity_weight` is not consumed by the loss composer.
- lifecycle code attempts `float(mapping)`, catches the exception, and silently
  substitutes `0.5`.
- the only temperature method containing this conversion has no supported
  caller.
- novelty history, pattern bank, budget, and cursor are not checkpointed.

### GABA

- GABA has a real trainable tensor path but is disabled in both operational
  model configs.
- its layer returns a complete `excitatory_input - inhibition` result, after
  which the block adds that result to `moe_out`, approximately doubling the
  excitatory path.
- homeostatic buffers update inside forward, aggregate the whole batch, and
  may update twice when Ghost performs a second forward.
- non-finite homeostatic observations can contaminate buffers and later
  checkpoints.

### Ghost

- integrated Ghost Token has a real loss-to-gradient path, but is operationally
  disabled and performs a second full block pass.
- the mask consumes global RNG and is not identified in the causal ledger.
- the same global weighted Ghost scalar is copied to every block, so the
  implementation cannot support a claim of per-expert specialization.
- Ghost Brain is constructed without a Ghost corpus; its exploration path
  therefore returns no material.
- Ghost Brain lessons, dopamine, counters, and local learning-rate multiplier
  are neither consumed by the optimizer nor persisted in the model checkpoint.
- Ghost Stream/Corpus is a separate data-production surface and must remain
  behind quarantine and explicit approval.

## Design principles

### No same-step self-reward

An observation derived from step `t` may not change the model, optimizer,
sampler, or data used to produce step `t`. It may only propose an intervention
whose earliest validity is step `t+1` or the next cycle boundary.

### Separate observation, policy, and execution

- organs observe and emit immutable signals;
- Soul proposes slow policy goals;
- Decision Engine calibrates, vetoes, defers, or approves proposals;
- the causal bus validates conflicts, expiry, identity, and allowlists;
- the training loop is the sole executor;
- the ledger records intent before mutation and outcome after mutation.

### Biological names do not lower the evidence bar

GABA-like inhibition, curiosity, Soul, and Ghost are engineering mechanisms.
Their names do not establish biological equivalence or usefulness.

### Default identity preservation

Every new tensor intervention starts from an exact-identity state:

- a zero residual gate;
- an effective loss scale of zero;
- no sampler reweighting;
- no queued optimizer action.

Disabled, `CONTROL`, and `SHADOW` must be bitwise equal under paired RNG.

## Architecture

### 1. Temporal barrier

Add an immutable `OrganObservationFrame` with:

- source `StepIdentity`;
- source checkpoint/training contract IDs;
- source phase and completed optimizer step;
- detached finite scalar/vector signals;
- per-sample signal hashes where applicable;
- `valid_from_step`;
- `valid_through_step`;
- frame digest.

The frame is created only after `STEP_OUTCOME`. Adapters consuming a frame at
the same or an earlier step reject it. A frame is single-use for mutation and
may remain multi-use for telemetry.

### 2. Three control planes

#### Tensor plane

Acts during forward using state frozen before the attempt:

- bounded GABA residual delta;
- deterministic Ghost probe;
- existing TTM residual retrieval;
- existing Spider calibration head;
- JEPA prediction.

Tensor-plane modules cannot update Python or buffer state during forward.

#### Optimizer plane

Acts after intent is durable and before the optimizer step:

- exact loss-term scales;
- update veto;
- global gradient clip;
- exact named gradient-group scale;
- bounded fast-adapter budget.

#### Lifecycle plane

Acts only at cycle or protected-sleep boundaries:

- sampler weights proposed by Curiosity;
- replay priority;
- structural proposals;
- consolidation candidates;
- checkpoint or rollback request.

### 3. Wake / Shadow / Apply / Sleep state machine

```text
WAKE(t, theta_t)
    -> forward with frozen organ state
    -> backward and optional optimizer update
    -> append STEP_OUTCOME

SHADOW(t)
    -> compute organ signals from completed outcome
    -> Soul proposes goals
    -> Decision calibrates and emits approve/defer/veto
    -> append next-step proposal without mutation

APPLY(t+1)
    -> validate source identity, TTL, conflicts, finite bounds
    -> append STEP_INTENT
    -> execute allowlisted mutation
    -> append STEP_OUTCOME and causal credit

SLEEP(boundary)
    -> snapshot full protected state
    -> consolidate approved replay/candidates
    -> paired fresh/replay/heldout evaluation
    -> commit or rollback atomically
```

## Organ contracts

### Soul: slow homeostatic policy

Soul no longer executes commands directly. It emits a typed `SoulPolicyFrame`
from lagged windows of:

- fresh LM loss and trend;
- replay LM loss and trend;
- heldout loss and regression;
- optimizer skip/failure rate;
- calibrated Spider error;
- Curiosity coverage;
- router fragility;
- protected-sleep outcomes.

Allowed proposals:

- request fresh-data review;
- request benchmark;
- request protected checkpoint;
- request protected sleep;
- lower or restore exploration budget;
- request update veto on independently measured regression.

External ingestion remains a request only. It cannot bypass quarantine or
explicit approval.

Soul's complete numerical state, windows, counters, last source identity,
pending proposals, and TTLs are stored in the checkpoint contract. Narrative
strings remain telemetry and never participate in identity or policy.

### Decision Engine: calibrated executive veto

Replace the disconnected text-length and static-threshold path in the supported
runtime with `CausalDecisionController`.

Inputs are normalized, finite, lagged evidence:

- empirical Spider correctness/calibration;
- fresh/replay/heldout deltas;
- gradient norm and update success;
- router counterfactual utility;
- novelty coverage;
- intervention history and causal credit.

Outputs:

- `APPROVE`;
- `DEFER`;
- `VETO`;
- `ESCALATE_TO_OPERATOR`.

Decision confidence is calibrated against later observed intervention success,
not against its own verbal report. Until the minimum calibration sample count
is reached, Decision may only operate in `SHADOW` or veto a non-finite/unsafe
proposal.

The existing `decision_engine_enabled` flag becomes effective or is rejected as
unsupported; it may no longer change identity while producing no behavior.

### Curiosity: coverage and information-gain scheduler

Curiosity is not a truth reward. It ranks what the organism should measure or
revisit.

The new `CuriosityMemory`:

- computes novelty per sample, never across unrelated batch members;
- compares against a frozen bank from the previous completed boundary;
- records JEPA surprise, LM surprise, domain coverage, and repeated-noise rate;
- rejects non-finite vectors;
- distinguishes novel-and-learnable from persistently noisy examples;
- updates its bank only after `STEP_OUTCOME`;
- persists bank, cursor, counts, decay state, and digest.

Curiosity may propose bounded next-cycle sampler weights or replay priorities.
It may not directly scale the current step loss.

### GABA: zero-init bounded inhibitory residual

Replace the current full-output composition with:

```text
gaba_delta = -gate * bounded_inhibition
moe_out = moe_out + gaba_delta
```

Requirements:

- `gate` is a trainable scalar or per-channel vector initialized to exact zero;
- `bounded_inhibition` is finite and norm-bounded relative to `moe_out`;
- per-sample statistics are computed before deterministic aggregation;
- homeostatic EMA is updated once in `POST_STEP`;
- eval, Ghost probes, recomputation, and `SHADOW` do not update buffers;
- non-finite input produces a rejected signal and exact identity output;
- state and gate are included in causal checkpoint identity.

### Ghost Token: deterministic predictive probe

Ghost Token becomes a deterministic probe derived from `StepIdentity`.

Requirements:

- mask positions and mask digest are identical across ablation arms;
- raw Ghost loss is computed in all enabled arms;
- `CONTROL` and `SHADOW` use exact effective scale zero;
- `APPLY` uses a declared bounded scale valid for one optimizer window;
- the main hidden representation is reused where semantics permit;
- if a second pass remains necessary, all mutable organs run in observation-off
  mode and no homeostatic state updates twice;
- raw/effective loss, mask digest, compute path, and gradient norm are ledgered.

Per-expert Ghost claims require:

- token-to-expert assignments;
- raw loss attributed to the selected expert;
- named expert-gradient evidence;
- a counterfactual or negative-control routing comparison.

Until those requirements pass, Ghost is a global predictive objective.

### Ghost Brain and Ghost data

Ghost Brain becomes an episodic lesson producer, not an automatic teacher.

- generated lessons enter an append-only quarantine artifact;
- lesson provenance includes model/checkpoint/tokenizer IDs;
- verification and explicit approval precede replay eligibility;
- approved lessons may be proposed to protected sleep;
- rejected or unverified lessons cannot alter weights, memory, or sampler;
- Ghost Brain state is checkpointed only if it influences a supported proposal.

## Bus and executor changes

Extend loss subjects from only `aux` to exact allowlisted subjects:

- `aux`;
- `ghost`;
- `jepa`;
- `spider`.

Every request includes:

- exact source organ;
- exact subject;
- finite value and configured min/max;
- source frame digest;
- source and target step;
- expiry;
- reason code;
- deterministic intervention ID.

Conflicts on the same target reject all conflicting requests. No priority rule
silently chooses a winner. Soul and Decision cannot mutate the model by holding
a model or optimizer reference.

## Causal credit

An intervention receives positive credit only when:

1. its durable intent precedes mutation;
2. the declared target changed as predicted;
3. unrelated targets stayed within the negative-control tolerance;
4. the next independent evaluation contains finite evidence;
5. fresh and heldout do not regress beyond configured bounds;
6. replay improvement is not the sole improvement;
7. the ledger chain and checkpoint identity verify.

Credit is descriptive evidence, not an automatic reward signal. Promotion is a
separate decision.

## Checkpoint contract

Keep checkpoint format v8 and introduce
`causal_cognitive_contract.version = 2`.

The v2 cognitive state covers:

- Soul numerical state and pending policy frames;
- Decision calibration state and sample count;
- Curiosity frozen bank, cursor, decay, and digest;
- GABA gates, EMAs, levels, and update counter;
- Ghost deterministic-mask generator state and pending proposals;
- Heartbeat/TTM state;
- Spider state;
- all temporal-barrier frames and TTLs;
- ledger head and last completed outcome identity.

Strict rules:

- v7 remains legacy and is never reinterpreted;
- v8 cognitive-contract v1 remains loadable only when v2 organs are disabled;
- enabling any v2 organ requires explicit migration;
- a missing, altered, stale, or non-finite organ state rejects strict resume;
- state identity is independent of filenames, mtimes, cycle numbers, and
  narrative text.

## Failure handling

- adapter exception: reject its proposals and record degradation;
- non-finite signal: reject before intent and preserve identity;
- non-finite tensor delta: abort attempt, restore pre-attempt state;
- orphan intent: fail strict continuation until reconciled;
- expired frame: record expiry, perform no mutation;
- heldout unavailable: no promotion; remain `SHADOW`;
- protected-sleep regression: rollback model, optimizer, organs, RNG, replay,
  and ledger outcome;
- external lesson without approval: remain quarantined;
- conflicting organ proposals: reject the conflict set.

## Verification matrix

### Per-organ unit and causal tests

Soul:

- command changes from print-only to typed proposal;
- heldout reaches policy input;
- token counts are not double-counted;
- complete state roundtrip;
- stale proposal rejection.

Decision:

- toggle has real supported behavior or is rejected;
- calibration minimum enforces `SHADOW`;
- approve/defer/veto boundary tests;
- later evidence updates calibration only after outcome;
- no current-output self-authorization.

Curiosity:

- sample-local novelty;
- batch permutation independence;
- frozen-bank one-step delay;
- noisy-TV suppression;
- checkpoint/tamper tests.

GABA:

- gate-zero bitwise identity;
- no doubled excitatory path;
- bounded delta;
- one and only one `POST_STEP` state update;
- eval/Ghost/Shadow state purity;
- NaN fail-closed;
- gradient and checkpoint roundtrip.

Ghost:

- paired deterministic masks;
- raw loss equal across arms;
- effective zero in Control/Shadow;
- predicted gradient delta in Apply;
- no double organ-state update;
- no per-expert claim without attributed evidence.

### Interaction tests

- Soul proposal plus Decision veto;
- Curiosity proposal plus expired frame;
- GABA and Ghost in one optimizer attempt;
- TTM, Spider, JEPA, GABA, and Ghost with batch permutation;
- simultaneous conflicting loss-scale requests;
- crash between intent and outcome;
- checkpoint resume with pending next-step proposal;
- protected sleep with all v2 organ states.

### Paired ablation

For each organ, run from the same tiny deterministic checkpoint:

- disabled;
- `CONTROL`;
- `SHADOW`;
- `APPLY`;
- sign-inverted or permuted negative control.

Required evidence:

- disabled/control/shadow parameter, optimizer, organ state, and RNG equality;
- no divergence at source step `t`;
- exact declared divergence at target step `t+1`;
- parameter-delta attribution;
- fresh/replay/heldout comparison;
- atomic manifest and independent verifier.

After individual organs pass, run the full factorial interaction matrix for the
five v2 organs where practical and a pairwise covering array otherwise. The
artifact must state which combinations were executed rather than implying full
coverage.

## Promotion gates

Source-level completion requires:

- focused v2 tests green;
- complete repository pytest green;
- Python compilation green;
- `git diff --check` green;
- public contract hashes updated;
- independent code review without open P0/P1/P2 findings.

Runtime promotion additionally requires:

- clean commit-bound source;
- no-launch canary green;
- explicitly authorized tiny causal canary;
- verified source checkpoint and corpus;
- paired control/shadow/apply artifacts;
- no fresh or heldout regression beyond declared bounds;
- strict v8 cognitive-contract-v2 resume.

No long 100M or 1.6B run is authorized by this design.

## Research translation

The design adopts mechanisms, not paper headlines:

- test-time refinement supports a separate verifier and revise/defer actions;
- metacognitive activation evidence remains telemetry until independently
  calibrated;
- curiosity-driven exploration motivates experience ranking, not truth;
- surprise-prioritized replay motivates candidate priority, not automatic
  consolidation;
- orthogonal/limited adaptation motivates bounded gradient and fast-adapter
  budgets;
- masked prediction motivates deterministic Ghost probes;
- continual-learning replay motivates a separate protected sleep phase;
- router-counterfactual work motivates per-token route utility instead of
  aggregate load claims.

## Non-goals

- no direct same-step self-reward;
- no uncontrolled online weight updates;
- no external content entering training without approval;
- no concurrent trainer;
- no silent v7-to-v8 migration;
- no GABA biological-equivalence claim;
- no Ghost per-expert claim without attributed gradients;
- no perplexity-improvement claim from reweighted loss alone;
- no checkpoint deletion, merge, or lineage selection in this implementation.

