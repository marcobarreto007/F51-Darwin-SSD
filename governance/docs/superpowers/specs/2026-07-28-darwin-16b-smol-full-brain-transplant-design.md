# Darwin 1.6B SmolLM Full-Brain Transplant Design

**Date:** 2026-07-28
**Status:** approved design, implementation not started
**Owner:** Fuch F51 Labs / Marco Barreto

## 1. Final objective

Build one native Darwin-X organism whose language weights originate from the
local SmolLM2-1.7B-Instruct donor and whose Darwin organs originate from the
last valid 100M Full Organism V3 checkpoint.

The final runtime must load only one Darwin checkpoint. It must not instantiate,
call, embed, or depend on a Hugging Face language model after the transplant.

The practical product flow is:

1. verify both source specimens by immutable identity;
2. construct a larger native Darwin anatomy;
3. transplant the Smol language system into Darwin matrices;
4. grow the V3 organs into the new hidden dimension;
5. calibrate the transplanted organism against the local donor;
6. publish an isolated candidate only if every engineering and knowledge gate
   passes;
7. serve the candidate through the native Darwin inference path.

The system is for Marco to inspect, demonstrate, continue training, and use as
the experimental basis for weight-level neural surgery.

## 2. Explicit meaning of full-brain transplant

This design does not place SmolLM behind Darwin, wrap SmolLM in organ adapters,
or preserve a second model at inference time.

Full-brain transplant means:

- the Darwin token embedding is donor-derived;
- every Darwin attention matrix is donor-derived;
- every Darwin SSD mixer is donor-derived through system identification;
- every Darwin MoE expert and router is donor-derived;
- the final normalization and language head are donor-derived;
- every donor tensor is classified in an immutable coverage ledger;
- the output is a standard `DarwinXModel` checkpoint;
- the donor directory can become unavailable and native Darwin inference still
  works.

Because 1.711 billion donor parameters cannot be copied bit-for-bit into a
different 1.675 billion-parameter anatomy, "everything" means 100% tensor
accounting and donor-derived initialization for 100% of the Darwin language
core. It does not mean lossless preservation of every donor bit.

## 3. Verified source specimens

### 3.1 Language donor

- Model: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- Snapshot:
  `31b70e2e869a7173562077fd711b654946d38674`
- Architecture: Llama causal language model
- Parameters: approximately 1.711B
- Hidden dimension: 2048
- Layers: 24
- Attention heads: 32
- KV heads: 32
- Head dimension: 64
- Intermediate dimension: 8192
- Vocabulary: 49,152
- Context: 8192
- RoPE theta: 130,000
- Tied token embedding and language head

Source hashes:

| File | SHA-256 |
|---|---|
| `config.json` | `994f50b16abb4ae00880baefe03c10260b5bd608d2bf586f7056ca05a534feea` |
| `model.safetensors` | `f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8` |
| `tokenizer.json` | `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c` |
| `tokenizer_config.json` | `4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df` |
| `special_tokens_map.json` | `2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52` |

The local Llama-3.2-3B cache is not a usable donor: it contains only a ref and
no weight snapshot. SmolLM2-1.7B-Instruct is therefore the largest complete
local donor.

### 3.2 Organ donor

- Model: `F51-Darwin-X-100M` Full Organism V3
- Checkpoint: cycle 2, step 1750
- Hidden dimension: 512
- SHA-256:
  `71c49bc295c0d06d72b3dc640d5b4d846f421ecdcca25820517fffaab6425a30`

This source provides organ weights and runtime state. It is not used as the
language-core donor.

## 4. Target anatomy

Create a new structural lineage:

- Model name: `F51-Darwin-X-1.6B-Smol-Transplant-V1`
- Config: `src/configs/darwin_x_1.6b_smol_transplant.yaml`
- Checkpoint root:
  `workspace/03_CHECKPOINTS_1.6B_SMOL_TRANSPLANT_V1`
- Runtime root:
  `workspace/runtime/darwin_16b_smol_transplant_v1`
- Tokenizer root:
  `workspace/01_TOKENIZER/smol_49152_transplant_v1`

Core anatomy:

| Field | Value | Reason |
|---|---:|---|
| `vocab_size` | 49,152 | exact donor tokenizer contract |
| `d_model` | 1,920 | canonical Darwin 1.6B width; only 6.25% donor reduction |
| `n_layers` | 16 | canonical Darwin 1.6B depth |
| `n_heads` | 30 | preserves the donor head dimension of 64 |
| `n_kv_heads` | 30 | MHA, avoiding destructive 32-to-4 KV collapse |
| `head_dim` | 64 | exact donor head dimension |
| `ssd_attention_ratio` | `3:1` | native Darwin anatomy |
| attention layers | 3, 7, 11, 15 | canonical Darwin cadence |
| `fine_experts` | 14 | native Darwin 1.6B MoE |
| `shared_experts` | 2 | native Darwin 1.6B MoE |
| fine/shared hidden | 896 | native Darwin 1.6B capacity |
| `experts_per_token` | 2 | native sparse execution |
| training context | 4,096 initially | bounded local calibration |
| inference context | 8,192 initially | donor-supported context |
| RoPE train/infer base | 130,000 | donor positional contract |
| weight tying | enabled | donor embedding/head contract |

Estimated language-core parameters:

- total: `1,674,720,224`;
- active per token: `683,385,600`;
- BF16 language-core storage: approximately 3.35 GB before metadata and organs.

The new config uses `init: f51_adapted`. The canonical random-init lineages keep
their current identities and roots unchanged.

## 5. Coherent hidden-space projection

The existing independent SVD prototype is not accepted as a transplant. It
reduces each matrix in an unrelated basis, which breaks residual-stream
coordination.

The transplant derives one global semi-orthogonal projection:

`P: R^2048 -> R^1920`

from the donor embedding and a fixed activation calibration set. The same
projection is used for embeddings, residual streams, attention, MLP interfaces,
normalizations, and the language head.

For a donor residual matrix `W`, the initial target matrix is built from the
coherent transformation:

`W_target = P W P^T`

with orientation adjusted to PyTorch's `[out, in]` convention.

The projection, calibration sample digest, singular values, explained-energy
ratio, implementation version, and output hash are checkpoint identity fields.

## 6. Layer mapping

The 24 donor layers are assigned monotonically to 16 target blocks. No donor
layer can disappear silently.

Each target block owns one contiguous donor group of one or two layers. The
coverage ledger records the exact group. A group endpoint supplies the residual
target for calibration.

### 6.1 Darwin attention blocks

Target blocks 3, 7, 11, and 15 receive donor attention projections from their
assigned donor groups.

- 30 of 32 donor heads are selected by activation contribution.
- Head dimension remains exactly 64.
- The two residual donor heads are folded into the nearest selected heads by a
  least-squares output fit.
- Q, K, V, and O are transformed as a coupled unit.
- Bias remains disabled for the new lineage because the donor has no attention
  bias.

### 6.2 Darwin SSD blocks

The other 12 target blocks remain native Darwin selective-state-space mixers.
They are not replaced with attention.

Their parameters are donor-derived through bounded system identification:

1. capture the assigned donor group's input/output transitions on fixed
   calibration sequences;
2. initialize SSD input/output projections from the coherent donor basis;
3. fit convolution and state parameters to the donor transition;
4. reject the block if it does not outperform random SSD initialization on a
   held-out transition set.

This is the only defensible mapping from attention to SSD because no exact
tensor-shape correspondence exists.

## 7. MLP to MoE transplant

Each donor MLP neuron is treated as one coupled triplet across gate, up, and
down projections. The three pieces may never be separated.

For every target layer:

1. collect activation signatures for the assigned donor MLP group;
2. classify ubiquitous high-contribution neurons into two shared experts;
3. cluster specialized neurons into 14 fine experts;
4. reduce each cluster coherently to the target expert width of 896;
5. initialize each router row from the cluster activation centroid;
6. fit router selection against the donor neuron's activation ownership;
7. record neuron coverage, reduction error, cluster balance, and dead-expert
   checks.

The transplant fails closed if any donor MLP tensor lacks a coupled mapping or
if any target expert remains random.

## 8. Tokenizer, embeddings, and output head

The target uses the exact donor tokenizer files and special-token identities:

- BOS: 1
- EOS: 2
- PAD: 2
- vocabulary: 49,152

The donor embedding is projected from `[49152, 2048]` to `[49152, 1920]` using
the global projection. `lm_head.weight` is tied to the target embedding.

The final checkpoint manifest includes all tokenizer hashes. Loading with an
F51 BPE tokenizer or another Smol snapshot is rejected.

## 9. Growing the V3 organs from 512d to 1920d

The organs are transplanted into native 1920-dimensional modules. Runtime
adapter wrappers are forbidden.

Derive a semi-orthogonal organ embedding:

`Q: R^512 -> R^1920`

For square organ matrices, embed the learned V3 function as:

`W_1920 = Q W_512 Q^T + N`

where `N` is a neutral complement appropriate to the organ. Residual organs use
a zero complement; identity-like state transitions use an identity complement.
Rectangular matrices use the corresponding left/right projections.

Organ rules:

- GABA: expand all learned projections and preserve scalar gates/state counters.
- JEPA: expand predictor interfaces and preserve the V3 predictive subspace.
- Spider: expand confidence interfaces; keep it shadow until recalibrated.
- TTM: expand key/value projections and every persisted memory vector; keep
  memory writes disabled initially.
- Heartbeat: expand feed-forward and thinker matrices; preserve beat, dopamine,
  exploration counters, and memory metadata exactly.
- IHS: expand hemispheric transforms; keep its residual gate at exact zero.
- MTP: expand all future-token heads; keep its contribution disabled initially.
- Ghost, DAE, Sleep, Decision Engine, and Unified Mesh: reconstruct from the new
  target config and import compatible scalar/runtime state only.

A paired subspace test must show that an expanded organ reproduces the original
512d organ when input is restricted to the `Q` subspace. Each organ requires
cosine agreement of at least 0.99 and finite outputs before it can enter shadow.

## 10. Runtime authority at first boot

The first candidate boots with:

- GABA: shadow with exact zero external effect;
- JEPA: train-only;
- Spider: shadow;
- TTM: shadow, writes disabled;
- Heartbeat: observer, mutation disabled;
- IHS: disabled with zero gate;
- MTP and Ghost: loss contribution disabled;
- DAE: shadow;
- Sleep, Decision Engine, and Unified Mesh: control/reporting only.

No organ is allowed to hide a failed language-core transplant. Language
knowledge is measured with every organ disabled first.

## 11. Construction and calibration stages

### Stage A: immutable preflight

- verify source hashes;
- verify disk, RAM, GPU, tokenizer, config, and checkpoint roots;
- reject an existing non-empty target root unless it belongs to the same
  incomplete run and passes resume identity;
- write an atomic transplant plan before allocating model weights.

### Stage B: streaming surgery

- memory-map the donor safetensor;
- build the target on meta/CPU devices;
- transform one tensor family at a time;
- write sharded BF16 target state atomically;
- update the coverage ledger after each completed family;
- support hash-bound resume without overwriting published artifacts.

### Stage C: structural canary

- instantiate native `DarwinXModel`;
- load the transplanted core and grown organs strictly;
- run finite forward/backward at short context;
- prove tied embeddings and checkpoint round-trip;
- prove zero random target tensors in the language core.

### Stage D: layerwise calibration

- keep organs disabled;
- fit the projection, SSD transitions, MoE experts, and routers against fixed
  local donor activations;
- use layerwise CPU/GPU offload so donor and target do not require full optimizer
  state on one GPU;
- keep calibration checkpoints isolated and append-only.

### Stage E: language calibration

- distill logits on approved local `feast_v2` text retokenized with Smol;
- use teacher KL plus next-token cross entropy;
- evaluate on a source-stratified holdout excluded from fitting;
- use at least three calibration seeds before making a knowledge claim.

### Stage F: organ shadow integration

- load grown organ state;
- run paired disabled/shadow checks;
- promote one organ at a time only after causal evidence;
- never use organ activity to mask a language regression.

## 12. Hardware execution

The surgery is streaming and does not require full FP32 donor and target models
on one GPU.

- GPU 0, RTX 5060 Ti 16 GB: target layer/calibration shard.
- GPU 1, RTX 3060 12 GB: donor layer or evaluation shard.
- CPU and disk: optimizer/offload state and atomic target shards.
- Available disk at design time: approximately 176 GB.

No long training starts as part of the converter command. Construction,
structural canary, bounded calibration, and long calibration are separate
operator actions.

## 13. Evidence and acceptance gates

### 13.1 Engineering gates

All must pass:

1. exact source hashes;
2. 100% donor tensor classification;
3. 100% target language tensors donor-derived;
4. strict native checkpoint load;
5. all tensors finite and shape-correct;
6. tokenizer identity exact;
7. embedding/head tying exact;
8. checkpoint save/reload logits stable within declared dtype tolerance;
9. donor-free native inference;
10. original 100M and canonical 1.6B roots unchanged.

### 13.2 Knowledge gates

The candidate is not called intelligent merely because it loads.

On a fixed, source-stratified, byte-accounted holdout:

- next-token bits-per-byte must beat random-init Darwin 1.6B by at least 10%;
- teacher KL must beat random-init Darwin 1.6B by at least 25%;
- next-token top-1 donor agreement must be at least twice the random-init
  agreement;
- all three calibration seeds must improve over their paired random baselines;
- generation must pass the existing repetition/n-gram validity gate;
- no result may rely on prompts seen during calibration.

Failing a knowledge gate leaves the artifact labeled
`engineering_transplant_only`.

### 13.3 Organ gates

- disabled organs must not change core logits;
- shadow organs must not change logits or persistent state;
- Heartbeat observer must not advance;
- TTM shadow must not write memory;
- each grown organ must pass the 512d-subspace reproduction test;
- promotion requires a paired causal canary.

## 14. Donor-free proof

The final smoke runs in an isolated process with:

- only the target checkpoint and copied Smol tokenizer;
- network disabled;
- donor path access denied and audited;
- no `transformers` model construction;
- native Darwin inference entrypoint only.

Success output:

`DARWIN_16B_SMOL_NATIVE_OK donor_loaded=false tokenizer=verified organs=verified`

## 15. Commands to be implemented

Planning and construction:

```powershell
python src/scripts/transplant_smol_to_darwin_1_6b.py plan --config src/configs/darwin_x_1.6b_smol_transplant.yaml
python src/scripts/transplant_smol_to_darwin_1_6b.py build --plan workspace/runtime/darwin_16b_smol_transplant_v1/plan.json
```

Read-only structural canary:

```powershell
powershell -ExecutionPolicy Bypass -File src/scripts/start_smol_darwin_transplant.ps1 -Canary
```

Bounded calibration requires a separate explicit `-Calibrate` operator action.
Publishing requires another explicit `-PublishCandidate` action after all
gates pass.

## 16. Failure handling and rollback

- Never modify or replace the Smol donor.
- Never modify the V3 organ donor.
- Never write into canonical 100M, 600M, or 1.6B checkpoint roots.
- Never select artifacts by filename, modification time, or highest step.
- Every output shard is written to a temporary name, flushed, hashed, and
  atomically promoted.
- Resume requires the same source hashes, config identity, projection identity,
  calibration-data digest, and completed-ledger head.
- A failed candidate remains quarantined with its failure report.
- Rollback means selecting the last manifest-approved candidate in the isolated
  transplant root; it never mutates a source lineage.

## 17. Scope limits

This work can prove weight-level transplantation and measurable transfer of
donor behavior. It cannot claim lossless transfer, consciousness, independent
scientific novelty, or useful inference-time learning without separate causal
evidence.

The earlier GPT-recipient organ stack remains valid engineering evidence but is
not the final architecture described here. The independent-SVD Smol artifact is
retained as historical evidence and is not eligible for publication as the
native Darwin brain.
