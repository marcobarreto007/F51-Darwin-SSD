# Darwin Cognitive Circuit Ablation and Transplant - Design

## Objective

Create a local-only, evidence-gated system that can:

1. identify a candidate cognitive circuit associated with a measured skill;
2. prove or reject its causal contribution through paired ablation;
3. package the circuit with cryptographic provenance;
4. install it into a compatible recipient without changing initial behavior;
5. progressively activate or replace the recipient path; and
6. roll back only that circuit through an immutable checkpoint lineage.

The first supported circuit units are:

- named modules already represented by explicit parameters and buffers, such
  as JEPA or GABA; and
- residual-stream channel groups identified by layer, tap and channel indices.

Arbitrary circuits spanning unrelated tensors and multiple non-contiguous
layers are outside the first version.

## Operator and Main Flow

Marco is the operator. The supported flow is:

```text
discover and confirm a circuit
    -> package it as an immutable .f51circuit artifact
    -> preflight the package against one explicit recipient
    -> install it as candidate with gate zero
    -> run shadow and active-path-zero-gate equivalence
    -> calibrate adapters and gate
    -> activate through an immutable child checkpoint
    -> retain or roll back through another immutable child checkpoint
```

No command may infer donor, recipient, checkpoint, tap or lineage from a
filename, modification time, largest cycle or directory ordering.

## Existing Foundations

The implementation must reuse the existing Darwin transplant foundations:

- tensor and checkpoint SHA-256 identities in
  `src/f51_darwin/transplant/bundle.py`;
- strict organ reconstruction in `src/f51_darwin/transplant/organs.py`;
- zero-impact adapters and gates in `src/f51_darwin/transplant/slots.py`;
- recipient integration in `src/f51_darwin/transplant/recipient.py`;
- per-organ lifecycle and gradient allowlisting in
  `src/f51_darwin/transplant/lifecycle.py`;
- immutable checkpoint writes and organ snapshots in
  `src/f51_darwin/transplant/checkpoint.py`;
- evidence records in `src/f51_darwin/transplant/ledger.py`; and
- paired causal anchors in `src/tools/run_causal_ablation.py`.

The current `run_two_donor_canary.py` remains a first-slice engineering
runner. It must not become the generic circuit implementation.

## Architecture

### Shared Library

Create a focused package under `src/f51_darwin/circuits/`:

- `manifest.py`: canonical circuit, interface, recipient and binding schemas;
- `package.py`: safe deterministic `.f51circuit` read/write and verification;
- `taps.py`: local allowlisted tap providers;
- `ablation.py`: discovery candidates and causal intervention arms;
- `compatibility.py`: static and executable preflight;
- `transaction.py`: installation and lifecycle transaction;
- `replacement.py`: gated inject/replace wrappers;
- `pointer.py`: compare-and-swap publication of the active checkpoint; and
- `ledger.py`: hash-chained circuit events.

The package must not accept dynamic imports or constructors supplied by an
artifact. Constructors and tap providers are selected from a local registry.

### Thin Operator CLIs

Expose three scripts:

1. `src/scripts/circuit_package.py`
   - `package`
   - `verify`
   - `preflight`
   - `install`

2. `src/scripts/circuit_ablate.py`
   - `discover`
   - `run`
   - `verify`

3. `src/scripts/circuit_transplant.py`
   - `inject`
   - `replace`
   - `shadow`
   - `gate-zero`
   - `adapter`
   - `activate`
   - `rollback`
   - `verify`

PowerShell launchers may wrap these commands, but correctness must live in
the Python library and remain testable without launching a worker.

## Circuit Package

### Artifact Format

The `.f51circuit` artifact is deterministic and content-addressed. It
contains:

- `manifest.json`;
- `weights.safetensors`;
- `runtime_state.json`, restricted to an explicit JSON-safe allowlist;
- `checksums.json`; and
- optional immutable evidence under `evidence/`.

Tensor state must never be loaded through arbitrary pickle execution.
Archive entries must reject path traversal, symlinks and unexpected files.

### Manifest Identity

The schema records:

- schema and producer versions;
- producer Git commit and clean-tree status;
- source checkpoint SHA-256, size and checkpoint version;
- source `base_checkpoint_id`;
- source config, structural config, topology and lineage identities;
- circuit kind and local constructor ID;
- every parameter and buffer name, role, shape, dtype and SHA-256;
- input/output rank and layout;
- donor width and minimum sequence length;
- tap semantics and mutation policy;
- adapter and residual-bound policy;
- accepted recipient families;
- mask, cache and cross-attention requirements; and
- explicit default bindings when a mapping is unambiguous.

`circuit_id` is the SHA-256 identity of the canonical semantic core. The
outer artifact also has an independent file SHA-256. The same tensor bytes
installed at a different tap or under a different adapter contract therefore
produce a different installation identity.

### Package Invariants

- The caller supplies the expected outer SHA-256.
- Every internal artifact and tensor is rehashed.
- Declared shape and dtype must equal the loaded tensor.
- Missing, extra, non-finite and backbone tensors fail closed.
- Donor paths are hints, never identities.
- Packaging never edits the donor checkpoint.

## Tap and Recipient Contracts

Tap providers are explicit and versioned:

- Hugging Face GPT-2 Transformer;
- Darwin Transfer discrete SSD; and
- native DarwinX.

Each provider resolves a declared semantic tap to a concrete module and
output selector. A contract includes:

- recipient family and class;
- recipient checkpoint and config identities;
- module path;
- `pre` or `post` position;
- output selector, such as tuple element zero;
- symbolic input and output shapes;
- dtype/device rules;
- mask/cache/cross-attention behavior;
- minimum sequence length; and
- whether the tap is already occupied.

Shape compatibility alone is insufficient. Native DarwinX GABA placement,
for example, is not semantically equivalent to a post-block GPT hook even
when both widths are 512.

Mappings such as twelve donor blocks to six recipient blocks must be
declared and hashed. No implicit averaging, slicing, padding or normalized
depth mapping is permitted.

## Causal Ablation

### Discovery and Confirmation Splits

Candidate discovery and causal confirmation use separate datasets.
Discovery may rank residual channels or named modules using
gradient-times-activation and activation contrast. These scores nominate
candidates but never prove causality.

Confirmation executes paired arms from identical model, data, RNG and
runtime anchors:

- `CLEAN`: untouched reference;
- `ABLATE`: replace the candidate activation with the registered baseline,
  initially a discovery-split mean;
- `RESTORE`: patch the exact clean activation back into the ablated path;
- `SHUFFLE`: apply a deterministic derangement preserving marginal scale;
  and
- `RANDOM_MATCHED`: ablate a size- and activity-matched control circuit.

Every arm records the checkpoint, dataset split, sample order, seed, batch
digests, model digest, intervention and output artifact hashes.

### Default Scientific Gates

Defaults are configurable but must be recorded in the manifest:

- at least five seeds;
- at least 200 confirmation items per evaluated domain;
- target degradation confidence interval strictly above zero;
- Holm-corrected `p < 0.01`;
- target effect at least `0.15` nat/token, unless an approved skill-specific
  metric defines an equivalent threshold;
- target effect at least twice the matched-control effect;
- `RESTORE` recovers at least 95% of the ablation effect;
- restored logits have maximum absolute error at most `1e-6` in FP32;
- `SHUFFLE` reproduces at least half of the ablation direction; and
- unrelated-domain regression stays within its registered retention bound.

Failure to meet the gates labels the candidate `correlated_only` or
`inconclusive`; it is not packageable as a proven skill circuit.

## Installation and Replacement Lifecycle

### Lifecycle

```text
candidate -> shadow -> gate_zero -> adapter_ready -> active
                                      |              |
                                      v              v
                                  quarantine      frozen

active or quarantine -> rollback_child
```

Installation always ends at `candidate`. No install command accepts an
`--active` shortcut.

### Shadow and Gate Zero

`shadow` executes the circuit but returns the original path exactly and may
not mutate circuit buffers, backbone state or RNG progression.

`gate-zero` executes the production wrapper and adapters with the external
gate exactly zero. It must satisfy:

- maximum absolute logit error at most `1e-5` in FP32;
- gate value exactly zero;
- backbone parameters and buffers bit-identical;
- no undeclared state mutation; and
- finite outputs on the registered probe set.

### Inject

Injection adds a bounded residual path:

```text
recipient hidden
    -> input adapter
    -> donor circuit
    -> output adapter
    -> bounded residual
    -> tanh(external gate)
    -> recipient hidden
```

The donor circuit remains byte-identical until an explicit
`organ_unfrozen` or equivalent lifecycle transition.

### Replace

Replacement preserves both original and candidate paths:

```text
original_output = original_module(...)
candidate_output = donor_circuit(...)
output = original_output
       + tanh(gate) * bounded(candidate_output - original_output)
```

Shadow and gate zero return the original output. Direct destructive module
replacement is not allowed in the first version.

If the candidate cannot preserve the original call signature, tuple
contract, mask/cache semantics or output shape, preflight rejects it.

## Checkpoints, Ledger and Rollback

### Hash Chain

Each transaction records:

```text
donor checkpoint
  -> circuit package
  -> tap and binding contract
  -> recipient parent
  -> pre-install snapshot
  -> candidate checkpoint
  -> ablation and calibration evidence
  -> previous ledger record
  -> new ledger record
  -> active pointer
```

The ledger includes `previous_record_sha256` so deletion, truncation and
reordering are detectable.

### Atomic Publication

- Checkpoints are immutable and never overwritten.
- Writes use a temporary file, flush, fsync and atomic replace.
- `active.json` is changed only after fresh-process reload and all gates pass.
- Publication uses compare-and-swap against the expected parent SHA-256.
- A concurrent or stale activation fails without changing the pointer.

### Rollback

Rollback never mutates the active in-memory object and never points silently
back to an old file. It:

1. loads the active checkpoint in a new process;
2. verifies donor, package, tap, recipient, snapshot and ledger identities;
3. restores only the selected circuit;
4. verifies the registered pre-install probes;
5. saves a new immutable rollback child; and
6. atomically advances `active.json` to that child.

Other accepted circuits must remain unchanged.

## CLI Examples

### Package and Preflight

```powershell
$SourceSha = "71c49bc295c0d06d72b3dc640d5b4d846f421ecdcca25820517fffaab6425a30"
$RecipientSha = "59847bdae17017d3a7fe3b82624e475dda9b4f7a20d73115426c7c26ca5afe2e"

.\.venv_nitro\Scripts\python.exe src\\scripts\\circuit_package.py package `
  --spec src\\configs\\circuits\jepa-v3.yaml `
  --source-checkpoint workspace\03_CHECKPOINTS_100M_FULL_ORGANISM_V3\organism_cycle_002_step_001750.pt `
  --expected-source-sha256 $SourceSha `
  --output workspace\04_CIRCUITS\jepa-v3.f51circuit `
  --strict

$PackageSha = (Get-Content workspace\04_CIRCUITS\jepa-v3.f51circuit.sha256).Trim()

.\.venv_nitro\Scripts\python.exe src\\scripts\\circuit_package.py preflight `
  --package workspace\04_CIRCUITS\jepa-v3.f51circuit `
  --expected-package-sha256 $PackageSha `
  --recipient-checkpoint workspace\03_CHECKPOINTS_DARWIN_TRANSFER_DISTILGPT2_V1\generation-06-final.pt `
  --expected-recipient-sha256 $RecipientSha `
  --tap-contract src\\configs\\circuits\taps\darwin-transfer-final-hidden.json `
  --report workspace\runtime\circuit_preflight\jepa-v3.json `
  --device cpu --strict
```

### Ablation

```powershell
$RecipientSha = "59847bdae17017d3a7fe3b82624e475dda9b4f7a20d73115426c7c26ca5afe2e"

.\.venv_nitro\Scripts\python.exe src\\scripts\\circuit_ablate.py run `
  --checkpoint workspace\03_CHECKPOINTS_DARWIN_TRANSFER_DISTILGPT2_V1\generation-06-final.pt `
  --expected-checkpoint-sha256 $RecipientSha `
  --candidate resid:4:120-191 `
  --discovery-contract src\\configs\\circuits\skills\math-discovery.json `
  --confirmation-contract src\\configs\\circuits\skills\math-confirmation.json `
  --output workspace\runtime\circuit_ablation\math-r4-c120-191 `
  --seeds 5 --strict
```

### Install, Activate and Roll Back

```powershell
$PackageSha = (Get-Content workspace\04_CIRCUITS\math-r4-c120-191.f51circuit.sha256).Trim()
$RecipientSha = "59847bdae17017d3a7fe3b82624e475dda9b4f7a20d73115426c7c26ca5afe2e"

.\.venv_nitro\Scripts\python.exe src\\scripts\\circuit_transplant.py inject `
  --package workspace\04_CIRCUITS\math-r4-c120-191.f51circuit `
  --expected-package-sha256 $PackageSha `
  --recipient-checkpoint workspace\03_CHECKPOINTS_DARWIN_TRANSFER_DISTILGPT2_V1\generation-06-final.pt `
  --expected-recipient-sha256 $RecipientSha `
  --tap-contract src\\configs\\circuits\taps\darwin-transfer-r4-post.json `
  --transaction-id math-inject-001 `
  --checkpoint-root workspace\03_CHECKPOINTS_DARWIN_CIRCUITS_V1 `
  --runtime-root workspace\runtime\darwin_circuits_v1 `
  --device cpu

$ParentSha = (Get-Content workspace\runtime\darwin_circuits_v1\transactions\math-inject-001\recipient-parent.sha256).Trim()

.\.venv_nitro\Scripts\python.exe src\\scripts\\circuit_transplant.py activate `
  --transaction-id math-inject-001 `
  --checkpoint-root workspace\03_CHECKPOINTS_DARWIN_CIRCUITS_V1 `
  --expected-active-sha256 $ParentSha `
  --strict

$CircuitId = (Get-Content workspace\runtime\darwin_circuits_v1\transactions\math-inject-001\circuit-id.txt).Trim()
$ActiveSha = (Get-Content workspace\03_CHECKPOINTS_DARWIN_CIRCUITS_V1\active.checkpoint.sha256).Trim()
$SnapshotSha = (Get-Content workspace\runtime\darwin_circuits_v1\transactions\math-inject-001\preinstall-snapshot.sha256).Trim()

.\.venv_nitro\Scripts\python.exe src\\scripts\\circuit_transplant.py rollback `
  --transaction-id math-inject-001 `
  --circuit-id $CircuitId `
  --checkpoint-root workspace\03_CHECKPOINTS_DARWIN_CIRCUITS_V1 `
  --expected-active-sha256 $ActiveSha `
  --to-snapshot-sha256 $SnapshotSha `
  --strict
```

## Failure Handling

- All verify and preflight operations are non-launching.
- Identity, tap, shape, dtype, state or evidence mismatches fail closed.
- Partial staging directories are removed only after validating that they are
  inside the declared transaction root.
- Existing checkpoint or install roots are never overwritten.
- Non-finite tensors or outputs quarantine the candidate.
- A failed shadow, gate-zero, fresh reload or rollback probe cannot publish.
- Runtime and dataset artifacts remain under `workspace/` and do not enter
  Git.

## Test Strategy

### Package and Security

- deterministic package round trip;
- outer and per-artifact tamper detection;
- tensor shape/dtype declaration mismatch;
- missing, extra and forbidden backbone tensors;
- schema and constructor allowlist rejection;
- archive traversal, symlink and decompression-limit rejection; and
- donor bytes unchanged after packaging.

### Ablation

- exact paired anchors across all arms;
- clean/control equivalence;
- known synthetic causal circuit recovered;
- correlated non-causal candidate rejected;
- restore and shuffle controls;
- random matched circuit control;
- corrected statistical gates; and
- immutable, hash-verified artifact publication.

### Compatibility

- GPT-2 Transformer provider;
- Darwin Transfer SSD provider;
- native DarwinX provider;
- wrong provider, tap, mapping, mask/cache or dtype rejection;
- explicit twelve-to-six binding;
- occupied tap rejection; and
- no silent slice, pad, average or inferred depth mapping.

### Transaction and Rollback

- disabled, shadow and active-path gate-zero equivalence;
- shadow state and RNG immutability;
- adapter/gradient change allowlists;
- fresh-process candidate reload;
- atomic pointer compare-and-swap;
- crash injection before and after pointer replacement;
- rollback in a new process;
- rollback preserves other circuits;
- ledger deletion, truncation and reordering detection; and
- launcher never starts a worker while any source gate is red.

## Proof of Completion

The system is ready for its first real circuit experiment only when:

1. the full source audit is green;
2. the package, ablation and transplant CLI tests pass;
3. a synthetic known circuit passes the complete discovery-to-rollback flow;
4. a real JEPA or GABA package passes Transformer and SSD preflight;
5. shadow and gate-zero equivalence pass after fresh reload;
6. activation publishes an immutable child checkpoint;
7. rollback publishes another child and restores registered behavior; and
8. every artifact, ledger record and pointer verifies from disk by SHA-256.

Passing these engineering gates proves a safe transplant mechanism. It does
not prove that a mathematics, philosophy or other semantic circuit has been
found. That claim additionally requires the causal ablation gates on
independent skill and retention datasets.
