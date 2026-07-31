# F51 Causal Organism v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a versioned causal training control plane that turns F51 organs into measurable, ablatable interventions instead of sidecar telemetry.

**Architecture:** A pure loss composer owns objective semantics. A synchronous typed bus accepts immutable organ signals and produces allowlisted interventions. The training loop alone applies interventions, while a hash-chained ledger proves intent and outcome. Organs migrate behind adapters and structural mutation is restricted to one cycle boundary.

**Tech Stack:** Python 3.12, PyTorch, dataclasses, JSONL/SHA-256, pytest, PowerShell operational gates.

## Execution status - 2026-07-18

Tasks 1-9 and the source-validation parts of Task 10 are implemented.
Evidence: 589/589 tests, compileall, diff-check, strict cycle-101 inspection,
and a verified atomic paired-ablation artifact. The no-launch canary was run
and stopped at its first gate because the tracked worktree is not clean; it
must be repeated after an explicit commit decision. Soul and Decision Engine
remain explicitly disconnected rather than being mislabeled causal.

## Global Constraints

- Work only in `C:\Users\marco\Desktop\F51-Darwin-SSD`.
- Never launch `run247` or concurrent training.
- Never delete or select checkpoints by filename, mtime, or cycle.
- Preserve the user's existing `src/configs/darwin_x_100m.yaml` worktree change.
- Raw metrics and effective intervention terms must remain distinct.
- No organ is called causal without a negative control and measured effect.
- Structural changes happen only at the cycle boundary.
- V7 legacy behavior and v8 causal migration remain explicit.

---

### Task 1: Versioned loss semantics

**Files:**
- Create: `src/f51_darwin/darwin_x_core/losses.py`
- Modify: `src/f51_darwin/darwin_x_core/config.py`
- Modify: `src/f51_darwin/darwin_x_core/model.py`
- Test: `src/tests/test_loss_semantics.py`

**Interfaces:**
- Produces: `LossPolicy`, `LossTerms`, `ComposedLoss`, `compose_loss(...)`.
- `DarwinXOutput` exposes raw `aux_loss` and effective
  `effective_aux_loss`.

- [ ] Write failing tests for exact zero, fractional scaling, scale one,
  adaptive-only-down, non-finite raw aux at zero, and raw/effective
  observability.
- [ ] Run `python -m pytest src/tests/test_loss_semantics.py -q` and confirm the
  tests fail for the missing composer and ignored scale.
- [ ] Implement finite non-negative config validation and the pure composer.
- [ ] Replace inline objective arithmetic in `DarwinXModel.forward` with the
  composer while keeping all raw loss terms.
- [ ] Run `python -m pytest src/tests/test_loss_semantics.py src/tests/test_darwin_x.py src/tests/test_darwin_x_training.py -q`.

### Task 2: Typed causal bus in control and shadow modes

**Files:**
- Create: `src/f51_darwin/organism/causal_bus.py`
- Modify: `src/f51_darwin/organism/config.py`
- Test: `src/tests/test_organ_causal_bus.py`

**Interfaces:**
- Produces: `StepIdentity`, `Phase`, `Signal`, `Intervention`,
  `Decision`, `StepOutcome`, `AblationArm`, `CausalBus`.
- Consumes: raw/effective `LossTerms` from Task 1.

- [ ] Write failing tests for schema validation, deterministic IDs,
  target allowlist, conflict rejection, intervention expiry, non-finite
  rejection, and control/shadow no-op equivalence.
- [ ] Run `python -m pytest src/tests/test_organ_causal_bus.py -q` and confirm
  failure.
- [ ] Implement immutable contracts and a deterministic reducer with no
  model/optimizer references.
- [ ] Add organism config fields for `disabled`, `shadow`, and `enforce`
  without changing the default legacy path.
- [ ] Run `python -m pytest src/tests/test_organ_causal_bus.py src/tests/test_organism_causal_runtime.py -q`.

### Task 3: Hash-chained causal ledger

**Files:**
- Create: `src/f51_darwin/organism/causal_ledger.py`
- Test: `src/tests/test_causal_ledger.py`

**Interfaces:**
- Produces: `CausalLedger.append_intent`, `append_outcome`,
  `verify_chain`, `recover_head`.
- Consumes: canonical serializable contracts from Task 2.

- [ ] Write failing tests for deterministic serialization, chain
  verification, tamper detection, orphan intent detection, and recovery.
- [ ] Implement append-only JSONL with fsync, sequence numbers, `prev_hash`,
  and `event_hash`.
- [ ] Prove that changing any historical payload invalidates verification.
- [ ] Run `python -m pytest src/tests/test_causal_ledger.py -q`.

### Task 4: Integrate causal phases into optimizer windows

**Files:**
- Modify: `src/f51_darwin/organism/training.py`
- Create: `src/f51_darwin/organism/causal_adapters.py`
- Test: `src/tests/test_causal_training_phases.py`

**Interfaces:**
- Consumes: bus decisions and ledger from Tasks 2–3.
- Produces: ordered execution `PRE_LOSS -> PRE_BACKWARD ->
  PRE_OPTIMIZER -> POST_STEP`.

- [ ] Write an event-order test covering gradient accumulation and skipped
  updates.
- [ ] Add pure adapters for brainstem, aux scaling, grad clip, and DAE.
- [ ] Persist `STEP_INTENT` before mutation and `STEP_OUTCOME` after the
  optimizer decision.
- [ ] Prove control and shadow produce identical parameter deltas on a tiny
  deterministic model.
- [ ] Run `python -m pytest src/tests/test_causal_training_phases.py src/tests/test_organism_causal_runtime.py -q`.

### Task 5: Real signal collection and one structural authority

**Files:**
- Modify: `src/f51_darwin/organism/unified_mesh.py`
- Modify: `src/f51_darwin/organism/lifecycle.py`
- Test: `src/tests/test_causal_cycle_boundary.py`

**Interfaces:**
- Unified Mesh becomes a read-only compatibility facade over typed signals.
- Structural decisions are queued through the bus and executed only once.

- [ ] Characterize the current empty-signal and plasticity-schema failures.
- [ ] Collect actual Spider, Heartbeat, neuroendocrine, router-usage, fresh,
  replay, and heldout signals with source/step identity.
- [ ] Remove mutation authority from the second death loop.
- [ ] Respect `sleep_enabled` and block direct unguarded L1 pruning.
- [ ] Test malformed organ schemas, disabled toggles, exceptions, and a
  single structural commit.
- [ ] Run `python -m pytest src/tests/test_causal_cycle_boundary.py src/tests/test_topology_manifest_v7.py src/tests/test_neuroendocrine_moe.py -q`.

### Task 6: Migrate Heartbeat/TTM, JEPA, and Spider-Sense

**Files:**
- Modify: `src/f51_darwin/heartbeat.py`
- Modify: `src/f51_darwin/spider_sense.py`
- Modify: `src/f51_darwin/darwin_x_core/model.py`
- Modify: `src/f51_darwin/darwin_x_core/config.py`
- Test: `src/tests/test_causal_cognitive_organs.py`

**Interfaces:**
- Produces batch-safe TTM residual retrieval, Spider calibration loss, and
  causal signal adapters.

- [ ] Add batch-safe TTM retrieval and a bounded residual intervention whose
  default is exact no-op.
- [ ] Add token-correctness calibration for Spider with detached labels and
  an explicit loss scale.
- [ ] Keep JEPA raw loss visible and route its surprise to memory/replay
  policy independently of its gradient weight.
- [ ] Test zero-gate identity, non-zero intervention effect, Spider gradient,
  batch isolation, state roundtrip, and negative controls.
- [ ] Run `python -m pytest src/tests/test_causal_cognitive_organs.py src/tests/test_darwin_x_training.py -q`.

### Task 7: Protected sleep and causal evolution

**Files:**
- Modify: `src/f51_darwin/organism/lifecycle.py`
- Modify: `src/f51_darwin/organism/unified_mesh.py`
- Test: `src/tests/test_protected_sleep_evolution.py`

**Interfaces:**
- Produces snapshot/evaluate/rollback sleep consolidation and gated
  structural proposals.

- [ ] Replace direct L1 zeroing with replay consolidation behind an explicit
  snapshot.
- [ ] Evaluate fresh, replay, and heldout metrics before accepting.
- [ ] Restore model, optimizer, RNG, and organ state on regression.
- [ ] Require an accepted bus intervention before structural action.
- [ ] Test successful consolidation, forced regression rollback, disabled
  sleep, and exception recovery.

### Task 8: Checkpoint v8 and explicit v7 migration

**Files:**
- Modify: `src/f51_darwin/organism/checkpoint_mixin.py`
- Modify: `src/f51_darwin/organism/checkpoint.py`
- Modify: `src/f51_darwin/state_identity.py`
- Test: `src/tests/test_causal_checkpoint_contract.py`

**Interfaces:**
- Produces `training_contract_id`, causal state, ledger head, and strict v8
  resume.

- [ ] Write failing v7 legacy, explicit migration, v8 strict roundtrip,
  contract mismatch, and tampered-ledger tests.
- [ ] Persist causal fields without changing `base_checkpoint_id` semantics.
- [ ] Require an explicit migration flag for loss-v2/enforce mode.
- [ ] Run `python -m pytest src/tests/test_causal_checkpoint_contract.py src/tests/test_checkpoint_resume.py src/tests/test_topology_manifest_v7.py -q`.

### Task 9: Paired causal ablation harness

**Files:**
- Create: `src/tools/run_causal_ablation.py`
- Test: `src/tests/test_causal_ablation.py`

**Interfaces:**
- Produces pre-registered CONTROL/SHADOW/APPLY artifacts from identical
  checkpoint, optimizer, RNG, and batches.

- [ ] Implement a tiny deterministic runner and immutable ablation manifest.
- [ ] Prove control/shadow equality and a known applied intervention effect.
- [ ] Reject mismatched checkpoints, data cursors, seeds, or training
  contracts.
- [ ] Persist raw per-arm evidence and a validity verdict.

### Task 10: Full verification and operational truth

**Files:**
- Modify: `governance/docs/operacao/STATUS_ATUAL.md`
- Create: `workspace/runtime/history/agent_bus/2026-07-18-causal-organism-v1.md`

- [ ] Run focused causal suites.
- [ ] Run `python -m pytest -q -p no:cacheprovider`.
- [ ] Run Python compilation for supported entrypoints and modified modules.
- [ ] Run `powershell -ExecutionPolicy Bypass -File src/scripts/start_overnight_16b.ps1 -Canary` without `-Launch`.
- [ ] Run `git diff --check` and inspect `git status --short`.
- [ ] Update status only with evidence from the current source, checkpoint
  contract, ledger, and tests.
- [ ] Do not declare the organism causal until every migrated organ has an
  ablation artifact or is explicitly labeled disconnected.
