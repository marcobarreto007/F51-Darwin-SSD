# Darwin Individual-Organ Causal QA Design

Date: 2026-07-29

## Objective

Determine whether any Darwin organ, acting alone while the transplanted Smol
language brain remains frozen, can learn explicit corrections to the existing
30-question QA set and retain them when question order changes.

This experiment is not another transplant-equivalence test. Its claim is
strictly causal: a useful arm must change only its assigned organ, improve or
retain QA answers after supervision, and preserve the frozen language brain.

## Candidates

Every arm starts from the immutable checkpoint:

`workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt`

Arms:

1. `control`: no trainable organ and no stateful heartbeat.
2. `gaba`: only floating parameters under `blocks.*.gaba`.
3. `ihs`: only `inter_hemispheric*` parameters.
4. `heartbeat`: no gradient path assumed; run supervised forwards with
   Heartbeat enabled and measure state mutation.
5. `ttm`: `ttm_residual_gate` plus Heartbeat TTM state/parameters.
6. `spider`: only `_spider_sense_module` parameters.
7. `jepa`: only `jepa_predictor` parameters.
8. `mtp`: only `mtp_heads` parameters.

Sleep, Decision Engine and Unified Mesh have no tensors in this checkpoint.
They are reported as orchestration-only and are not falsely presented as
modules that can be unfrozen.

## Protocol

For every arm:

1. Strictly reload the same published checkpoint.
2. Freeze every model parameter.
3. Select only the assigned organ's parameter identities.
4. Run a gradient-reachability probe on one corrected QA example.
5. Evaluate the pre-teaching accuracy.
6. Teach all 30 examples once in canonical order. Each training item contains
   the exact chat prompt followed by its reference answer; labels before the
   answer are masked with `-100`.
7. If the organ has a nonzero language-loss gradient, update only that organ.
   Use SGD with separate gate and body learning rates, gradient clipping, and
   no optimizer state that can spill into another arm.
8. For Heartbeat and TTM, also execute their native state path during teaching
   and record beat, memory-write and memory-slot deltas.
9. Evaluate ten deterministic shuffled orders without corrective feedback.
10. Assert that frozen language parameter version counters did not change.
11. Release the model and empty both CUDA caches before loading the next arm.

The same tokenizer, system prompt, greedy decoding and grading rules from
`src/scripts/benchmark_smol_dense_qa.py` are reused. Shuffle seeds are
`51..60`.

## Metrics

Primary:

- pre-teaching exact-match accuracy;
- post-teaching accuracy for each of ten shuffled orders;
- mean, minimum and maximum post-teaching accuracy;
- number of baseline errors corrected;
- number of baseline-correct answers forgotten.

Causal integrity:

- selected parameter count;
- nonzero-gradient parameter count and norm;
- selected organ parameters changed or unchanged;
- frozen brain version drift;
- Heartbeat beats and TTM slots/writes before and after teaching.

An arm passes the QA utility gate only when it corrects at least one baseline
error, forgets none of the baseline-correct answers, produces valid responses
in every shuffled run, and leaves the frozen brain unchanged.

## Interpretation

- `improved`: the individual organ passes the QA utility gate.
- `changed_but_regressed`: it corrects something but also forgets a previously
  correct answer.
- `no_effect`: answers and accuracy do not change.
- `no_language_gradient`: the organ is present but the current language loss
  cannot train it.
- `state_only`: native Heartbeat/TTM state changes without a QA change.
- `not_applicable`: the named subsystem has no unfreezable checkpoint tensor.

A flat result is a valid and important result: it disproves online QA
adaptation for that organ under the current wiring. It must not be repaired by
silently unfreezing the Smol brain or altering the published checkpoint.

## Safety and Artifacts

All adaptation is in memory. No trained arm overwrites or publishes a model
checkpoint. The source checkpoint SHA-256 and manifest are verified before
every run.

Artifacts:

- `workspace/runtime/darwin_17b_smol_dense_v1/organ-causal-qa.json`
- `workspace/runtime/darwin_17b_smol_dense_v1/organ-causal-qa.md`

The benchmark script and deterministic grading tests are committed; runtime
artifacts remain ignored under `workspace/`.
