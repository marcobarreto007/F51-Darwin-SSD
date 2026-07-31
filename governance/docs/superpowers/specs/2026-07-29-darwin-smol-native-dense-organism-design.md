# Darwin-Smol Native Dense Organism — Design

Date: 2026-07-29
Status: approved by the operator directive to rebuild with a dense model and
deliver autonomously
Scope: `C:\Users\marco\Desktop\F51-Darwin-SSD`

## Objective

Build a new Darwin lineage whose language backbone is structurally dense,
contains the complete SmolLM2-1.7B-Instruct brain, keeps the Darwin organs
connected, and runs without loading the Hugging Face donor.

The existing Exact Brain V3 is function-preserving but does not satisfy this
contract: every block still instantiates `DeepSeekStyleMoE`, a router, expert
containers, routing buffers, and MoE neuroendocrine state. One routed expert
is mathematically equivalent to a dense FFN, but it is not a structurally
dense implementation.

## Final operational result

The new lineage is ready only when all of the following are true:

1. Every one of the 24 blocks uses attention plus a native dense SwiGLU FFN.
2. The instantiated model contains no `DeepSeekStyleMoE`, `FineRouter`, or
   `fine_experts` module and its state dict has no `.moe.` key.
3. All 218 donor tensors are accounted for by an immutable coverage ledger.
4. Donor attention, normalization, embeddings, output head, and all three
   SwiGLU matrices per layer are copied without compression or clustering.
5. Heartbeat is connected. GABA, TTM, and IHS are connected behind exact
   zero gates at first boot. Training-only or shadow organs remain
   non-invasive during the parity gate.
6. A saved checkpoint reloads strictly and generates finite text on the two
   local GPUs with both Hugging Face offline flags set and without loading the
   donor model.
7. The candidate beats three independently initialized dense controls on the
   existing BpB, KL, top-1, and generation-validity gates.
8. A bounded backward/optimizer smoke proves gradients reach the native dense
   gate, up, and down matrices.
9. Publication is fail-closed, hash-bound, isolated from every existing
   checkpoint root, and followed by a green full repository suite.

## Architecture

### Configuration contract

`DarwinXConfig` gains a structural field:

```text
feed_forward_kind: moe | dense_swiglu
```

The default is `moe`, preserving every existing lineage. The dense value
requires exactly one configured fine path and no shared expert solely for
dimension validation and backward config compatibility; runtime construction
does not instantiate any router or expert container.

The new config uses:

- 24 attention layers and zero SSD layers;
- `d_model=2048`, 32 query heads, 32 KV heads;
- dense SwiGLU width 8192;
- Llama split-half RoPE with theta 130000;
- RMSNorm epsilon `1e-5` with FP32 accumulation;
- tied token embedding and language head;
- dropout zero and residual scale 1.0;
- an isolated checkpoint root
  `workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1`.

### Native dense FFN

`DenseSwiGLU` owns exactly three bias-free matrices:

```text
gate_proj: [8192, 2048]
up_proj:   [8192, 2048]
down_proj: [2048, 8192]
```

Its function is:

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

`DarwinXBlock` registers either `moe` or `ffn`, never both. The block preserves
the same residual and GABA placement. Dense forward returns a stable empty
auxiliary record so model-level loss and reporting code can operate without
inventing router statistics.

MoE-only mutation, Nitro expert placement, DAE expert-gradient actions, and
neuroendocrine routing state are explicit no-ops or unavailable on a dense
block. They are not counted as connected dense organs. This avoids preserving
MoE machinery under a new name.

### Organ contract

The language function must be unchanged at publication time.

- Heartbeat: connected and observable; disabled during parity calls.
- GABA: module present in every block, residual gates exactly zero.
- TTM: module present, residual gate and maximum first-boot scale zero.
- IHS: full module present, scalar residual gate exactly zero.
- JEPA, MTP, Spider, Curiosity, Sleep, Decision Engine, Unified Mesh:
  instantiated according to config but their weighted or residual effect is
  zero/shadow during the language-preservation gate.
- DAE and MoE neuroendocrine control: absent for this lineage because their
  current causal target is an expert router, which the dense architecture
  intentionally removes.

An organ may be described as beneficial only after a separate causal
adaptation experiment. Connection is not evidence of improvement.

## Surgery data flow

1. Verify immutable donor snapshot, weight hash, tokenizer identity, V3 organ
   source hash, dense config identity, free checkpoint root, GPUs, and disk.
2. Create a canonical surgery plan and hash it.
3. Instantiate the dense Darwin model on CPU.
4. Copy each Smol tensor to its single exact dense destination:
   embeddings, per-layer norms, Q/K/V/O, gate/up/down, final norm, tied head.
5. Grow only compatible Darwin organ tensors from the V3 organ source.
6. Zero every residual organ gate and verify first-boot neutrality.
7. Assert strict state loading, 218/218 donor coverage, no MoE runtime object,
   no `.moe.` state key, finite tensors, tied weights, and expected shapes.
8. Run teacher parity across fixed prompts on both GPUs.
9. Save checkpoint and immutable manifest atomically.
10. Reload donor-free and run native generation.
11. Evaluate the candidate against dense random controls for seeds 17, 29,
    and 43.
12. Publish `candidate-manifest.json` only after every gate passes.

## Failure handling and rollback

- The builder refuses a non-empty dense root unless it exactly resumes its own
  incomplete plan; it never overwrites a published checkpoint.
- Incomplete outputs use an `.incomplete` suffix and are not candidates.
- Any donor/config/plan/hash mismatch aborts before publication.
- Any MoE module or state key is a hard structural failure.
- Any missing donor tensor, non-finite tensor, strict-load mismatch, parity
  regression, random-control loss, or donor-free failure blocks publication.
- Exact Brain V3 remains immutable and is the rollback candidate until the
  dense lineage is independently approved.

## Verification

Unit and integration tests cover:

- config round-trip and dense validation;
- exact DenseSwiGLU function and gradient flow;
- mutually exclusive `moe`/`ffn` registration;
- state/topology manifests for both legacy and dense models;
- dense dual-GPU placement;
- exact donor mapping and coverage;
- organ zero-gate equivalence;
- strict checkpoint round-trip;
- donor-free native inference;
- three random dense controls;
- fail-closed publication and immutable hashes;
- legacy MoE regression and the full repository suite.

The operator command remains:

```powershell
powershell -ExecutionPolicy Bypass -File src\\scripts\\start_smol_darwin_transplant.ps1 -Canary
```

It will be switched to the dense root only after publication. Before that,
the command continues to target the approved V3 rollback.

## Non-goals

- No layer compression, SVD, clustering, head folding, SSD conversion, or
  dimension projection.
- No external model download; the verified local Smol snapshot is the donor.
- No claim that zero-gated organs already improve intelligence.
- No deletion or mutation of V1, V2, or V3 artifacts.
- No long training run unless exact preservation fails its gates and a
  bounded, explicitly recorded recovery experiment is justified.
