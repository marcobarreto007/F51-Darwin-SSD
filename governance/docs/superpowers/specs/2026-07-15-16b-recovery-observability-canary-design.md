# Darwin-X 1.6B Recovery, Observability, and Canary Design

Date: 2026-07-15 EDT

## Objective

Recover the canonical `F51-Darwin-X-1.6B-Nitro` lineage after the abrupt
dual-GPU failure, replace misleading replay-biased training output with
disjoint and durable measurements, and run one bounded 250-step canary before
any new `run247` launch.

The canary must exercise the real 1.6B organism and exact backward. It must not
test the proposed WGrad/gradient-metabolism technology. That work starts only
after the baseline runtime is stable and measurable.

## Proven State

The recoverable base is `organism_cycle_068.pt`, checkpoint v7, cycle 68,
step 36501. Its SHA-256 is
`4416A9AF6E1A3105FCF207616E7A134D2484CB3E0B9EE36281921446230CF6CC`.
The checkpoint contains the canonical 16-layer, d_model 1920 configuration,
Topology Manifest v7, the declared base checkpoint identity, and an AdamW
state with 997 state entries for 997 optimizer parameters.

The process stopped during cycle 69 after logging step 36751. Windows recorded
`nvlddmkm` Event ID 153 on GPUID 400, the RTX 3060 at PCI bus `04:00.0`.
The previous crash recorded the same event on GPUID 100, the RTX 5060 Ti at
PCI bus `01:00.0`. There was no correlated WHEA event.

Both GPUs passed isolated memory and compute tests. They also passed direct
cross-device copies, simultaneous GEMMs, and a 300-second sustained dual-GPU
test. CUDA peer access is unavailable in both directions. The evidence does
not establish a permanently defective GPU; it is more consistent with an
intermittent driver, WDDM, PCIe power-state, or long-running dual-runtime
failure.

## Scope

This design includes:

- a positionally disjoint holdout split for the canonical token stream;
- separate fresh-training, replay-training, and held-out metrics;
- a durable structured metrics artifact;
- an isolated one-cycle 1.6B canary;
- a single reversible PCIe power-management change;
- checkpoint, quality, GPU-health, and promotion gates.

It excludes:

- WGrad offload or approximate gradients;
- changing the NVIDIA driver during the first canary;
- changing TDR registry values;
- rebooting Windows;
- deleting or overwriting the canonical checkpoint;
- starting `run247` automatically;
- treating training loss or generated samples as proof of improvement.

## Approaches Considered

### Logger-only repair

The smallest patch would label every logged batch as `fresh` or `replay` and
maintain separate rolling means. This fixes the cadence collision but still
does not provide out-of-sample evidence. It is insufficient as a promotion
gate.

### Disjoint in-process holdout plus structured metrics

Reserve a fixed tail of the canonical token stream, remove it from the train
loader, evaluate the same frozen blocks throughout the run, and persist
category-specific metrics. This gives cheap and continuous non-inferiority
evidence and is the selected runtime design.

### External benchmark only

Evaluate only through the existing clean checkpoint benchmark. This is the
strongest promotion evidence but too expensive and too sparse to explain an
in-run crash or catch an early quality regression. It remains the final
promotion gate, not the only metric.

## Architecture

### Token stream split

The primary token source retains its existing full-file identity and lineage
metadata. After loading it, the organism creates two non-copying views:

- training view: every token except the last 1,048,576 tokens;
- holdout view: exactly the last 1,048,576 tokens.

The `CausalLMDataLoader` used for optimization receives only the training view.
A dedicated holdout evaluator receives only the holdout view. The canary must
abort before model training if the source is too short, the two views overlap,
or a weighted/multi-source path makes exclusion unprovable.

The split is positionally disjoint, not guaranteed semantically deduplicated.
The final external benchmark remains necessary because repeated text may exist
elsewhere in the corpus.

### Metric channels

The training loop maintains independent rolling windows and cumulative counts:

- `fresh_total_loss` and `fresh_lm_loss` for primary-corpus updates;
- `replay_total_loss` and `replay_lm_loss` for replay updates;
- `heldout_lm_loss` and `heldout_ppl` from side-effect-free evaluation;
- throughput, successful update count, skipped update count, and replay ratio.

Every log record includes `batch_kind`. The displayed rolling values come from
their category windows, never from whichever batch happens to coincide with
the logging cadence. The existing deterministic 20% replay schedule may remain
unchanged.

The evaluator uses 16 fixed batches of the configured block size and batch
size. Batch starts are derived from a fixed evaluation seed and are identical
for baseline and candidate evaluation. It runs under inference mode, restores
the previous train/eval state, and must not mutate heartbeat, neuroendocrine,
router, replay, RNG, or lineage state.

### Durable metrics ledger

Each evaluation appends one JSON object to a canary-specific JSONL file outside
the repository, under the external run artifact root. Each record contains:

- schema version and run ID;
- timestamp, Git commit, command, and process ID;
- checkpoint path, SHA-256, base checkpoint ID, cycle, and step;
- config identity, tokenizer identity, and full token-source identity;
- holdout definition and evaluation seed;
- fresh, replay, heldout, throughput, and skipped-update metrics;
- NVIDIA driver, GPU names, PCI bus IDs, and starting event-log record ID.

Writes use append, flush, and `fsync`. A crash may lose only the object being
written; every preceding newline-delimited record remains parseable. An
incomplete trailing line is ignored and reported during recovery.

### Isolated canary

The existing one-shot `cycle` path is used, not `run247`. The canary command
loads `organism_cycle_068.pt`, performs one real organism cycle of 250 steps,
saves, joins the asynchronous checkpoint writer, and exits.

The canary receives an explicit external checkpoint root such as:

`F51-Dataset-Organizado/03_CHECKPOINTS/canary_recovery_20260715/`

Its checkpoint and `organism_latest.json` are written only inside that root.
The canonical checkpoint directory and canonical pointer are not modified.
The command records the frozen Git commit and refuses to start from a dirty
tracked worktree.

The one-shot `cycle` command does not execute the `run247` Ghost Stream or
Ghost Feeder loop. The organism's normal forward losses and organs remain
active. Topology changes, if the normal cycle authorizes any, are preserved in
the isolated candidate's Topology Manifest and included in the comparison.

## GPU Stability Control

The first canary keeps NVIDIA driver 595.97 and all TDR defaults. The only
environmental variable changed is PCI Express Link State Power Management on
AC power, from `Moderate power savings` to `Off`.

The previous value, active power scheme, exact `powercfg` command, and change
time are written to the launch manifest. The change requires no reboot and is
reversible. The launcher does not install a driver, change clocks, change
power limits, or modify TDR settings.

Immediately before launch, the gate records the latest `nvlddmkm` EventRecordID.
During the canary a read-only monitor samples utilization, memory, temperature,
power, and the process state. The monitor never kills or restarts the process.
After process exit it waits two minutes and checks for new NVIDIA Event IDs 13,
14, 153, or a new LiveKernelEvent 141.

## Canary Data Flow

1. Prove that no `darwin_organism.py run247` or canary process is active.
2. Verify the base checkpoint SHA-256, embedded config, lineage identity,
   topology, and AdamW alignment offline.
3. Verify corpus and tokenizer identities.
4. Create the disjoint train and holdout views.
5. Evaluate the base checkpoint on the 16 frozen holdout batches.
6. Record the current power configuration and disable PCIe link-state power
   management on AC.
7. Record GPU state and the last NVIDIA event record.
8. Run one 250-step exact-backward cycle in the isolated checkpoint root.
9. Join checkpoint writing and require clean process exit.
10. Load and verify the candidate checkpoint offline.
11. Evaluate the candidate on the identical frozen holdout batches.
12. Wait two minutes, collect event-log and GPU-monitor results, then write the
    signed final canary report.
13. Run the existing external checkpoint comparison before any promotion.

## Failure Handling and Rollback

The canary fails closed when any prerequisite or metric is missing. It never
falls back to training without a holdout.

If CUDA or the driver terminates the process, the candidate is marked
`hardware_runtime_failed`; the base checkpoint remains authoritative. If the
candidate checkpoint is incomplete, it remains quarantined and its pointer is
not read as canonical. If JSONL ends with a partial record, the parser preserves
all preceding records and reports the truncated tail.

If quality regresses, the candidate remains in the isolated root with status
`quality_gate_failed`. If the power change needs rollback, the saved AC setting
is restored with `powercfg`; this does not require reboot.

No automated path installs a driver, reboots Windows, deletes a checkpoint, or
publishes the candidate to the canonical pointer.

## Acceptance Gates

The canary is operationally successful only when all conditions hold:

- the process exits with code 0 after exactly one 250-step cycle;
- all scheduled optimizer updates are accounted for;
- fresh, replay, and heldout metrics are finite;
- observed replay fraction is 0.20 within one update of rounding error;
- no new NVIDIA Event ID 13, 14, 153, or LiveKernelEvent 141 appears during
  the canary or the two-minute post-run window;
- peak temperature remains below 85 C on both GPUs;
- the isolated candidate checkpoint passes SHA-256, config, lineage, topology,
  strict model load, and AdamW alignment checks;
- candidate heldout loss on the identical 16 batches is no more than 0.02
  above the base heldout loss;
- the external frozen checkpoint benchmark reports no material regression.

Passing these gates means `stable_canary`, not `model_improved`. Promotion to
the canonical pointer and any new `run247` launch remain explicit operations.

## Tests

CPU tests run with `CUDA_VISIBLE_DEVICES=-1` and cover:

- holdout and training views are disjoint and preserve the full source identity;
- short sources and unprovable weighted-source exclusion fail closed;
- log cadence cannot force fresh metrics to contain replay batches;
- fresh and replay rolling windows/counts are independent;
- baseline and candidate evaluation use identical holdout starts;
- evaluation restores model mode and does not mutate organism state;
- JSONL recovery ignores exactly one incomplete trailing record;
- canary checkpoint root cannot equal or contain the canonical pointer path;
- canary requires a clean tracked worktree and exact base checkpoint identity;
- acceptance gates reject missing metrics, new driver events, excessive
  temperature, invalid checkpoints, and heldout regression.

GPU verification consists of the existing isolated and dual stress probes,
followed by the real 250-step one-shot canary. The full CPU suite runs before
any CUDA allocation. No concurrent training process may exist.

## Implementation Boundaries

The implementation should isolate metric logic from the large organism runtime
in a focused module, with `src/scripts/darwin_organism.py` acting as orchestration.
Expected touched surfaces are:

- a focused training-observability module under `src/f51_darwin/`;
- `src/scripts/darwin_organism.py` for split, orchestration, and CLI wiring;
- the official 1.6B launcher for a non-automatic canary gate;
- targeted unit tests;
- dated `.agent_bus` evidence and external run artifacts.

Unrelated model, dataset-factory, 2.5B, 600M, or WGrad refactors are outside
this implementation.
