# Darwin-X 100M Full Organism 65B Design

## Objective

Start a new, local-only Darwin-X 100M training lineage on the two installed
NVIDIA GPUs, with a finite budget of 65,000,000,000 optimizer-training tokens.
The family has exactly three supported scales: 100M, 600M, and 1.6B.

## Authority and lineage

- The family contract lists 100M, 600M, and 1.6B independently.
- Each scale has its own config and checkpoint root.
- The old shared checkpoint root remains immutable evidence.
- The new 100M full-organism run starts in a new empty root. It does not resume
  the earlier 1K-context 100M checkpoint because enabling the full cognitive
  state and 4K context changes the model/config identity.
- A scale becomes Gold only after its real checkpoint passes strict identity,
  benchmark, manifest, and hash gates. Gold is not inferred from a filename.

## 100M training contract

- Model scale: Darwin-X 100M.
- Context: 4096 training tokens.
- Token source: verified local `feast_v2`, 18,548,972,689 int32 tokens.
- Training budget: 65,000,000,000 actual batch tokens, including replay
  batches that participate in gradient updates; evaluation tokens do not count.
- Placement: existing pipeline parallel split across CUDA 0 and CUDA 1.
- Precision: runtime-selected BF16 where supported, FP16 fallback only when
  required by the existing AMP policy.
- Flash attention: PyTorch SDPA flash path enabled by the existing attention
  implementation.
- Selective scan: chunk size 512.
- Optimizer: AdamW with warmup and the current causal loss compositor.
- The run is finite. `run247` is not used.

## Full organism definition

The fresh 100M config enables the implemented trainable or transactional
organs: GABA, Heartbeat/TTM residual, Ghost, Curiosity, Spider calibration,
JEPA, MTP, DAE, Inter-Hemispheric module, Sleep, Decision Engine, Unified Mesh,
Nitro expert capacity, and causal checkpoint v8.

The causal sequence is gated:

1. CPU/source tests.
2. One finite CONTROL canary.
3. One finite SHADOW canary.
4. One finite ENFORCE canary.
5. Launch the 65B budget only when all three complete with finite loss,
   strict checkpoint identity, both GPUs used, and no conflicting process.

## Budget and persistence

The organism persists `train_tokens_seen` inside `training_state`. Resume
restores the exact counter. A new `train-budget` command runs finite cycles and
stops before exceeding the requested token budget. The final cycle is shortened
to the remaining number of whole batches.

Each published cycle remains immutable. A free-space guard stops cleanly before
the configured disk floor; it never deletes checkpoints. Reaching 65B therefore
also requires the operator to archive immutable checkpoints or add storage over
the lifetime of the run.

## Launcher

`src/scripts/start_100m_65b.ps1` is the supported local launcher for this run.

- `-Canary` performs gates and finite canaries without a long launch.
- `-Launch` is allowed only after the same gates pass.
- It rejects a dirty tracked worktree, another Darwin trainer, fewer than two
  CUDA GPUs, a non-empty fresh root, a missing corpus manifest, or insufficient
  free disk.
- It writes the command, PID, source commit, config identity, logs, and budget
  under `workspace/runtime/runs/100m-65b/`.

## Governance repair

The machine-readable operational policy and canonical documents are updated to
the three-scale family. `AGENTS.md` and `CLAUDE.md` remain byte-identical.
New maintenance scripts and archived executables are classified explicitly.
Claims such as "training active" require a live process and log evidence.

## Verification

Required before the long launch:

- Python compilation;
- focused budget/resume/dual-GPU/config tests;
- operational-surface and architecture-boundary tests;
- canonical-document and source-audit gates;
- strict inspection of every canary checkpoint;
- live `nvidia-smi` evidence for both GPUs;
- clean Git status bound to the launch manifest.

No benchmark or Gold claim is made merely because training starts.
