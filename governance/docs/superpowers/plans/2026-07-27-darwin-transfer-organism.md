# Darwin Transfer Organism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, test and launch an isolated DistilGPT-2-to-SSD Darwin lineage with staged architecture transfer and an experience-routed pre-backbone path.

**Architecture:** Preserve a frozen DistilGPT-2 teacher and its non-mixer weights, replace selected attention modules with a discrete per-head SSD mixer, and train new mixers through orientation, hidden alignment and KL distillation. Add a bounded persistent memory that can emit a high-confidence sparse prior before the backbone and record causal controls.

**Tech Stack:** Python 3.12, PyTorch, Transformers, safetensors, pytest, JSONL metrics, PowerShell launch control.

## Global Constraints

- Work only inside `C:\Users\marco\Desktop\F51-Darwin-SSD`.
- Keep donor, checkpoints, logs and corpora under ignored `workspace/`.
- Never launch a second trainer while FULL_ORGANISM_V3 is alive.
- Preserve all unrelated and untracked user files.
- Keep canonical 100M, 600M and 1.6B roots unchanged.
- Use explicit hashes and manifests; never select checkpoints by mtime.
- Do not claim a gate passed without fresh artifact-backed verification.

---

### Task 1: Dependency and donor contract

**Files:**
- Modify: `pyproject.toml`
- Create: `src/f51_darwin/transfer/donor.py`
- Create: `src/f51_darwin/transfer/__init__.py`
- Test: `src/tests/test_transfer_donor.py`

**Interfaces:**
- Produces: `DonorManifest`, `load_frozen_teacher()`, `write_donor_manifest()`
- Consumes: a local path or the pinned `distilbert/distilgpt2` model ID

- [ ] Write tests requiring frozen parameters, deterministic tokenizer identity and manifest hashes.
- [ ] Run `python -m pytest -q src/tests/test_transfer_donor.py` and observe failure before implementation.
- [ ] Add pinned runtime dependencies and implement local-first donor loading.
- [ ] Download the donor into `workspace/00_DONORS/distilgpt2`.
- [ ] Write and verify `donor_manifest.json`.
- [ ] Re-run the focused test.

### Task 2: Discrete SSD reference mixer

**Files:**
- Create: `src/f51_darwin/transfer/ssd_mixer.py`
- Test: `src/tests/test_transfer_ssd_mixer.py`

**Interfaces:**
- Produces: `DiscreteSSDMixer.forward()`, `materialize_mixer()`, `load_attention_projections()`
- Consumes: `[batch, sequence, d_model]` hidden states

- [ ] Write tests for shape, causality, recurrent/matrix equivalence and projection transfer.
- [ ] Run the tests and verify they fail.
- [ ] Implement per-head scalar decay and `B/C/X` recurrence.
- [ ] Implement the semiseparable matrix materialization used by orientation.
- [ ] Map teacher Q/K/V/output projections without padding or slicing.
- [ ] Re-run focused tests and require zero failures.

### Task 3: Hybrid student and checkpoint identity

**Files:**
- Create: `src/f51_darwin/transfer/student.py`
- Create: `src/f51_darwin/transfer/checkpoint.py`
- Test: `src/tests/test_transfer_student.py`

**Interfaces:**
- Produces: `DarwinTransferStudent`, `TransferCheckpoint`, `save_transfer_checkpoint()`, `load_transfer_checkpoint()`
- Consumes: frozen teacher plus an explicit tuple of replaced layer indices

- [ ] Write tests for one-layer replacement, copied-weight identity, forward logits and checkpoint round-trip.
- [ ] Run tests and verify failure.
- [ ] Implement a GPT-2-compatible attention adapter around `DiscreteSSDMixer`.
- [ ] Copy all compatible teacher weights exactly and replace only requested mixers.
- [ ] Save atomically with donor hash, config identity, stage and step.
- [ ] Re-run focused tests.

### Task 4: Staged trainer

**Files:**
- Create: `src/f51_darwin/transfer/data.py`
- Create: `src/f51_darwin/transfer/trainer.py`
- Create: `src/scripts/train_darwin_transfer.py`
- Test: `src/tests/test_transfer_trainer.py`

**Interfaces:**
- Produces: `TransferTrainer.run_orientation()`, `run_alignment()`, `run_distillation()`
- Consumes: local Markdown/text corpus, frozen teacher and hybrid student

- [ ] Write tests proving stage-specific trainable parameters, finite losses and holdout isolation.
- [ ] Run tests and verify failure.
- [ ] Implement deterministic local-text token chunks and a locked tail holdout.
- [ ] Implement orientation loss on teacher attention versus student mixer matrices.
- [ ] Implement block-output alignment and end-to-end KL/LM loss.
- [ ] Append atomic JSONL metrics and checkpoint at configured intervals.
- [ ] Re-run focused tests.

### Task 5: Experience prior and causal controls

**Files:**
- Create: `src/f51_darwin/transfer/experience.py`
- Create: `src/f51_darwin/transfer/runtime.py`
- Create: `src/scripts/eval_darwin_transfer.py`
- Test: `src/tests/test_transfer_experience.py`

**Interfaces:**
- Produces: `ExperienceMemory`, `DarwinTransferRuntime.predict_next()`, `evaluate_experience_controls()`
- Consumes: shared token embeddings and sparse full-model logits

- [ ] Write tests for bounded writes, retrieval confidence, persistence, reset, shuffle, freeze and actual backbone bypass.
- [ ] Run tests and verify failure.
- [ ] Implement top-k sparse-logit slots and cosine retrieval.
- [ ] Implement confidence and minimum-observation gates.
- [ ] Implement the fallback path and per-path compute counters.
- [ ] Implement paired correct/shuffled/frozen evaluation.
- [ ] Re-run focused tests.

### Task 6: Launcher and operational isolation

**Files:**
- Create: `src/configs/darwin_transfer_distilgpt2_v1.yaml`
- Create: `src/scripts/start_darwin_transfer.ps1`
- Test: `src/tests/test_transfer_launcher.py`

**Interfaces:**
- Produces: `-Canary` validation and the sole supported launch command
- Consumes: verified donor manifest and isolated checkpoint/runtime roots

- [ ] Write launcher contract tests.
- [ ] Run tests and verify failure.
- [ ] Implement source audit, donor hash, root identity, PID and competing-trainer gates.
- [ ] Make `-Canary` strictly non-launching.
- [ ] Implement explicit run IDs and log paths.
- [ ] Re-run focused tests.

### Task 7: Verification, launch and delivery

**Files:**
- Modify: `governance/docs/operacao/STATUS_ATUAL.md`
- Create: `workspace/runtime/history/agent_bus/2026-07-27-darwin-transfer-launch.md`

**Interfaces:**
- Produces: verified live run and operator status
- Consumes: all prior tasks

- [ ] Run all focused transfer tests.
- [ ] Run the full test suite and record unrelated failures separately.
- [ ] Run launcher `-Canary`.
- [ ] Run a CPU donor/student forward and checkpoint round-trip.
- [ ] Run a bounded GPU training canary.
- [ ] Verify the latest FULL_ORGANISM_V3 checkpoint and gracefully stop its process tree.
- [ ] Launch exactly one Darwin transfer trainer.
- [ ] Verify PID, command line, GPU use, advancing metrics and checkpoint root.
- [ ] Update canonical status with actual evidence.
- [ ] Run `git diff --check`, inspect explicit paths and commit with Conventional Commits.
