# Darwin Individual-Organ Causal QA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether each Darwin organ can independently learn and retain corrected QA answers while the Smol language brain remains frozen.

**Architecture:** Extend the paired QA protocol with isolated fresh-checkpoint arms. A reachability layer selects organ parameters and audits language-loss gradients; a causal runner performs one canonical teaching pass followed by ten shuffled retention runs, then emits fail-closed JSON and Markdown evidence.

**Tech Stack:** Python 3.12, PyTorch 2.12 CUDA, Transformers 5.14.1, pytest, JSON/Markdown, two local NVIDIA GPUs.

## Global Constraints

- Work only in `C:\Users\marco\Desktop\F51-Darwin-SSD`.
- Use only `workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt`.
- Never overwrite or publish an adapted arm.
- Reload the immutable checkpoint independently for every arm.
- Never make Smol language parameters trainable.
- Test one organ per arm; no combined-organ arm in this experiment.
- Use the existing 30 questions and grading rules without changing references.
- Teach once in canonical order and evaluate seeds `51..60`.
- Report absent and gradient-disconnected organs honestly.

---

### Task 1: Define isolated organ selection and causal metrics

**Files:**
- Create: `src/f51_darwin/transplant_16b/organ_causal_qa.py`
- Test: `src/tests/test_organ_causal_qa.py`

**Interfaces:**
- Consumes: `DarwinXModel.named_parameters()` and the QA rows from `scripts.benchmark_smol_dense_qa`.
- Produces: `ORGAN_ARMS`, `select_organ_parameters(model, arm)`, `classify_arm(report)`, and `shuffled_orders(seeds)`.

- [ ] **Step 1: Write failing selection tests**

Assert that `gaba`, `ihs`, `spider`, `jepa`, and `mtp` select only their named
parameter prefixes; `ttm` selects the residual gate and separately identifies
Heartbeat TTM parameters; `control` selects nothing. Assert seeds `51..60`
produce ten permutations containing each of the 30 IDs exactly once.

- [ ] **Step 2: Run the tests and confirm the module is absent**

Run:

```powershell
python -m pytest src/tests/test_organ_causal_qa.py -q
```

Expected: collection failure for the missing module.

- [ ] **Step 3: Implement selection and result classification**

Use parameter identity sets, not substring checks at optimizer time. Reject an
unknown arm and reject overlap between selected organ parameters and language
markers `token_embedding`, `attention`, `ffn`, `norm`, and `lm_head`.

Classification order:

```python
if frozen_brain_drift:
    return "invalid_brain_drift"
if nonzero_gradient_parameters == 0 and not state_changed:
    return "no_language_gradient"
if corrected > 0 and forgotten == 0:
    return "improved"
if corrected > 0 or forgotten > 0:
    return "changed_but_regressed"
if state_changed:
    return "state_only"
return "no_effect"
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
python -m pytest src/tests/test_organ_causal_qa.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add -- src/f51_darwin/transplant_16b/organ_causal_qa.py src/tests/test_organ_causal_qa.py
git commit -m "test(transplant): define isolated organ QA arms"
```

### Task 2: Implement supervised teaching and shuffled retention

**Files:**
- Create: `src/scripts/benchmark_darwin_organs_qa.py`
- Modify: `governance/audit/policy/operational-surface.json`
- Test: `src/tests/test_organ_causal_qa.py`

**Interfaces:**
- Consumes: `ORGAN_ARMS`, `select_organ_parameters`, `classify_arm`, the dense checkpoint loader, and the existing QA prompts/graders.
- Produces: one result row per arm and runtime reports `organ-causal-qa.json` and `organ-causal-qa.md`.

- [ ] **Step 1: Add failing tests for answer-only labels**

Build one MCQ and one open-answer training sequence. Assert prompt labels are
all `-100`, answer labels equal their token IDs, and the sequence ends with the
tokenizer EOS ID.

- [ ] **Step 2: Implement the teaching example builder**

Return `(input_ids, labels)` with shape `[1, sequence]`. Preserve the exact chat
template used by the paired benchmark and append only the reference answer plus
EOS.

- [ ] **Step 3: Implement per-arm execution**

For each arm:

- strict-load and dual-GPU place the model;
- freeze all model and Heartbeat parameters;
- enable only the selected organ;
- snapshot frozen parameter `_version` counters;
- run the gradient probe;
- evaluate pre-teaching answers;
- teach 30 answers in canonical order with SGD;
- evaluate ten shuffled orders;
- verify selected parameter changes and zero frozen-brain drift;
- capture Heartbeat/TTM state deltas;
- release CUDA memory before the next arm.

Gate parameters (`residual_gate`, `inter_hemispheric_gate`,
`ttm_residual_gate`) use learning rate `0.05`; other organ parameters use
`0.0001`; clip selected gradients to norm `1.0`.

- [ ] **Step 4: Implement fail-closed reports**

JSON must include the source checkpoint SHA-256, protocol hash, arm selection,
pre/post answers, ten seeds, gradient audit, state deltas, brain drift and
classification. Markdown must show a compact per-arm table and every changed
answer.

- [ ] **Step 5: Classify the executable and run focused tests**

Add `src/scripts/benchmark_darwin_organs_qa.py` as `maintenance` in
`governance/audit/policy/operational-surface.json`.

Run:

```powershell
python -m pytest src/tests/test_organ_causal_qa.py -q
python src/tools/check_operational_surface.py
```

Expected: all tests and the policy checker pass.

- [ ] **Step 6: Commit Task 2**

```powershell
git add -- src/scripts/benchmark_darwin_organs_qa.py governance/audit/policy/operational-surface.json src/tests/test_organ_causal_qa.py
git commit -m "test(transplant): run individual organ QA adaptation"
```

### Task 3: Execute, reconcile status and verify

**Files:**
- Modify: `governance/docs/operacao/STATUS_ATUAL.md`
- Produce locally: `workspace/runtime/darwin_17b_smol_dense_v1/organ-causal-qa.json`
- Produce locally: `workspace/runtime/darwin_17b_smol_dense_v1/organ-causal-qa.md`

**Interfaces:**
- Consumes: the causal benchmark runner and immutable dense candidate.
- Produces: measured per-organ verdicts and canonical status.

- [ ] **Step 1: Verify idle hardware and execute**

Run:

```powershell
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
python src/scripts/benchmark_darwin_organs_qa.py
```

Expected: one final row for every arm and no checkpoint write.

- [ ] **Step 2: Validate the report**

Assert source hash matches the candidate manifest, all ten seeds exist per arm,
all arms have zero brain drift, and every classification is from the enumerated
closed set.

- [ ] **Step 3: Update canonical status**

Record exact per-organ pre/post accuracy, corrected, forgotten, gradient/state
evidence, and whether the original online-adaptation hypothesis passed.

- [ ] **Step 4: Run repository verification**

```powershell
python -m pytest src/tests/test_organ_causal_qa.py src/tests/test_smol_dense_qa_benchmark.py -q
python src/tools/check_operational_surface.py
python src/tools/check_canonical_docs.py --root .
python -m pytest -q
git diff --check
```

Expected: zero failures and no whitespace errors.

- [ ] **Step 5: Commit status and verify clean state**

```powershell
git add -- governance/docs/operacao/STATUS_ATUAL.md
git commit -m "docs(transplant): record individual organ QA evidence"
git status --porcelain=v1
```

Expected: clean output; runtime reports remain ignored under `workspace/`.
