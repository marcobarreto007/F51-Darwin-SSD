# Darwin-X 100M Full Organism 65B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the three-scale family contract and start a finite, resumable,
dual-GPU Darwin-X 100M full-organism run with a 65,000,000,000-token budget.

**Architecture:** Preserve the old checkpoint evidence and introduce a new
100M full-v1 root. Add a persisted actual-token counter and a finite
`train-budget` CLI path. A PowerShell launcher owns source, process, GPU,
corpus, disk, canary, manifest, and background-launch gates.

**Tech Stack:** Python 3.11, PyTorch CUDA AMP/SDPA, PowerShell 7/Windows
PowerShell compatibility, pytest, JSON/YAML manifests.

## Global Constraints

- Never invoke `run247`.
- Never overwrite, move, or delete an existing checkpoint.
- The supported family contains exactly 100M, 600M, and 1.6B scales.
- `feast_v2` is the only training corpus.
- The long-run token budget is exactly 65,000,000,000 actual gradient-input
  tokens; evaluation does not count.
- The launch requires two CUDA GPUs and a clean tracked worktree.
- Every canary and launch is bound to the current Git commit.

---

### Task 1: Repair family governance and executable classification

**Files:**
- Modify: `AGENTS.md`
- Modify: `CLAUDE.md`
- Modify: `governance/audit/policy/operational-surface.json`
- Modify: `governance/docs/CANONICAL_MAP.md`
- Modify: `governance/docs/operacao/STATUS_ATUAL.md`
- Modify: `src/tests/test_operational_surface.py`
- Modify: `src/tests/test_architecture_boundaries.py`

**Interfaces:**
- Consumes: executable inventory from `src/scripts/`, `src/tools/`, `research/`, and
  `governance/archive/`.
- Produces: one machine-readable `model_family` contract with three configs and
  complete executable classification.

- [ ] **Step 1: Make tests require 100M/600M/1.6B and identical agent contracts**

  Replace the single-config expectation with a map whose keys are `100m`,
  `600m`, and `1.6b`; assert `AGENTS.md` and `CLAUDE.md` are byte-identical.

- [ ] **Step 2: Run the focused tests and preserve the failing output**

  Run:
  `.\.venv_nitro\Scripts\python.exe -m pytest src\\tests\\test_operational_surface.py src\\tests\\test_architecture_boundaries.py -q`

  Expected before policy repair: failures for the old single 1.6B config and
  unclassified executables.

- [ ] **Step 3: Update policy and documents to the same three-scale truth**

  Add all current executable paths to either `classifications` or
  `archived_executables`. Keep the supported launcher surface explicit. Remove
  unproved claims of a live trainer.

- [ ] **Step 4: Re-run focused governance tests**

  Expected: all focused tests pass.

- [ ] **Step 5: Commit**

  `git commit -m "fix(governance): align Darwin three-scale authority"`

### Task 2: Define the fresh 100M full-organism lineage

**Files:**
- Modify: `src/configs/darwin_x_100m.yaml`
- Create: `src/configs/darwin_x_1.6b_nitro.yaml`
- Move: `src/configs/darwin_x_1.2b.yaml` to
  `src/configs/_archived/darwin_x_1.2b.yaml`
- Modify: `src/f51_darwin/organism/dependencies.py`
- Modify: `src/f51_darwin/serving/runtime.py`
- Modify: `src/scripts/darwin_organism.py`
- Modify: `src/scripts/serve_davi.py`
- Test: `src/tests/test_100m_full_config.py`

**Interfaces:**
- Consumes: `DarwinXConfig.from_yaml()` and checkpoint-root identity policy.
- Produces: a 4K 100M config with full-v8 cognitive state and a separate empty
  root `workspace/03_CHECKPOINTS_100M_FULL_V1`.

- [ ] **Step 1: Add a failing config contract test**

  Assert context 4096, scan chunk 512, every full-organism toggle enabled,
  loss semantics v2, and a unique full-v1 checkpoint root.

- [ ] **Step 2: Run the test and confirm it fails on the current 1K config**

- [ ] **Step 3: Update the 100M config and restore the isolated 1.6B config**

  Restore the archived 1.6B config as an active family member but do not claim
  or fabricate its missing checkpoint.

- [ ] **Step 4: Re-run config, identity, and dual-GPU split tests**

  Run:
  `.\.venv_nitro\Scripts\python.exe -m pytest src\\tests\\test_100m_full_config.py src\\tests\\test_dual_gpu_split.py src\\tests\\test_checkpoint_root_policy.py -q`

- [ ] **Step 5: Commit**

  `git commit -m "feat(config): define full 100m dual-gpu lineage"`

### Task 3: Add a finite persisted training-token budget

**Files:**
- Modify: `src/f51_darwin/organism/bootstrap.py`
- Modify: `src/f51_darwin/organism/training.py`
- Modify: `src/f51_darwin/organism/checkpoint_mixin.py`
- Modify: `src/f51_darwin/organism/cli.py`
- Test: `src/tests/test_training_token_budget.py`

**Interfaces:**
- Produces: `organism.train_tokens_seen: int`.
- Persists: `training_state["train_tokens_seen"]`.
- CLI: `train-budget --max-train-tokens 65000000000`.

- [ ] **Step 1: Write failing tests for fresh counter, resume, exact counting,
  final shortened cycle, and stop-at-budget**

- [ ] **Step 2: Run the focused tests and confirm failures**

- [ ] **Step 3: Increment the counter from `batch.numel()` after a completed
  training step and persist/restore it**

- [ ] **Step 4: Implement `train-budget` as a finite cycle loop**

  It must reject missing/non-positive budgets, never exceed the budget by more
  than an indivisible batch, and join the pending checkpoint writer on exit.

- [ ] **Step 5: Run token-budget plus checkpoint-contract tests**

  Run:
  `.\.venv_nitro\Scripts\python.exe -m pytest src\\tests\\test_training_token_budget.py src\\tests\\test_organism_causal_runtime.py::test_cycle_checkpoint_is_atomic_and_contains_replay_state -q`

- [ ] **Step 6: Commit**

  `git commit -m "feat(training): add resumable finite token budget"`

### Task 4: Add the guarded 100M 65B launcher

**Files:**
- Create: `src/scripts/start_100m_65b.ps1`
- Test: `src/tests/test_100m_65b_launcher.py`
- Modify: `governance/audit/policy/operational-surface.json`
- Modify: `governance/docs/operacao/OPERACAO_SEGURA.md`

**Interfaces:**
- `-Canary`: source/config/corpus/process/GPU/disk gates and finite causal
  canaries.
- `-Launch`: starts one hidden `train-budget` worker with stdout/stderr logs
  and a PID/commit/command manifest.

- [ ] **Step 1: Write static launcher-contract tests**

  Require two GPUs, clean Git, exact config/root/budget, CONTROL/SHADOW/ENFORCE
  canaries, no `run247`, hidden background launch, and a free-space floor.

- [ ] **Step 2: Run and confirm the launcher tests fail**

- [ ] **Step 3: Implement the launcher without destructive filesystem actions**

- [ ] **Step 4: Re-run launcher and operational-surface tests**

- [ ] **Step 5: Commit**

  `git commit -m "feat(launch): add guarded 100m 65b trainer"`

### Task 5: Verify, canary, and launch

**Files:**
- Runtime only under `workspace/runtime/runs/100m-65b/`
- Runtime checkpoints only under
  `workspace/03_CHECKPOINTS_100M_FULL_V1/`

**Interfaces:**
- Consumes: committed clean source.
- Produces: canary manifests, strict checkpoint inspection, long-worker PID and
  live two-GPU evidence.

- [ ] **Step 1: Run compilation, focused tests, source audits, and `git diff --check`**

- [ ] **Step 2: Run `src/scripts/start_100m_65b.ps1 -Canary`**

  Do not proceed if any causal mode, identity check, finite-loss check, or GPU
  usage check fails.

- [ ] **Step 3: Commit any evidence-bound source adjustment and re-run the
  canary from the new clean HEAD**

- [ ] **Step 4: Run `src/scripts/start_100m_65b.ps1 -Launch`**

- [ ] **Step 5: Verify one Python worker, its parent/command, live log growth,
  CUDA 0 and CUDA 1 memory use, exact config/root/budget, and no checkpoint
  collision**

- [ ] **Step 6: Report the measured throughput and recalculated 65B ETA without
  claiming Gold or benchmark quality**
