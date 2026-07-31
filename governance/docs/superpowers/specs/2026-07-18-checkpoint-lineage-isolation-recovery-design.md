# Checkpoint Lineage Isolation and Gold Recovery Design

**Date:** 2026-07-18  
**Status:** Approved for implementation planning  
**Scope:** `C:\Users\marco\Desktop\F51-Darwin-SSD`

## Objective

Prevent any Darwin model lineage from overwriting checkpoints belonging to
another lineage, preserve the current contaminated checkpoint root as evidence,
isolate future 100M writes, and attempt non-destructive recovery of the missing
1.6B gold checkpoint using its recorded identity.

The system is ready for implementation when:

1. a fresh lineage cannot start in a directory that already contains organism
   checkpoints;
2. a checkpoint save cannot replace an existing `organism_cycle_NNN.pt`;
3. a checkpoint root records and enforces one model-lineage identity;
4. the 100M configuration uses an isolated checkpoint root;
5. the current verified 100M cycle 1 is copied, not moved, into that root with
   its hash and pointer verified;
6. the contaminated canonical root remains untouched as scientific evidence;
7. the missing 1.6B gold is recorded honestly and searched for by its manifest
   identity without treating filename, mtime, cycle, or size alone as proof;
8. focused tests and the source canary gate pass without launching training.

## Confirmed Incident

The recorded 1.6B gold checkpoint was:

- path: `workspace/03_CHECKPOINTS/organism_cycle_071.pt`;
- model: `F51-Darwin-X-1.6B-Nitro`;
- cycle: 71;
- step: 40751;
- bytes: 10,876,850,383;
- SHA-256:
  `239fdcf175ac35d9402d664e2a2b40252ec9c73aea8ee458250d426adde9241b`.

The file currently at that path is a different artifact:

- model: `F51-Darwin-X-100M`;
- cycle: 71;
- step: 35500;
- bytes: 930,327,856;
- SHA-256:
  `cfb7f00946470fa3e097aad389bf51475924e8ac9e8d32c02e6e882ee5c6d911`.

The inspector confirmed incompatible model topology (`d_model=512`, 12 layers
versus `d_model=1920`, 16 layers). The gold manifest and current file therefore
prove replacement of the path, not merely documentation drift.

The immediate root cause is that `run247 --fresh-start` starts cycle numbering
at 1 and `_save_cycle` publishes to `organism_cycle_{cycle:03d}.pt` using
replacement semantics. The 100M YAML pointed to the same root as the 1.6B
lineage. No preflight rejected a non-empty fresh-start root, and no save-time
guard rejected an existing target path.

## Safety Constraints

- Do not delete, move, rename, or overwrite any existing checkpoint.
- Do not rewrite `workspace/03_CHECKPOINTS/organism_latest.json` during
  containment or recovery.
- Do not launch `run247`.
- Do not treat a copied artifact as valid until byte size and SHA-256 match the
  source and the official inspector reports strict resume compatibility.
- Do not promote or recreate the 1.6B gold from a 100M checkpoint.
- Do not weaken the existing atomic save and pointer publication guarantees.
- Keep recovery local-only; cloud state is not operational authority.
- Preserve unrelated worktree changes.

## Options Considered

### Configuration-only isolation

Change `src/configs/darwin_x_100m.yaml` to
`workspace/03_CHECKPOINTS_100M`.

This prevents the default 100M command from writing into the shared root, but a
CLI override or future configuration error can reintroduce the collision.
Rejected as insufficient.

### Save-time no-overwrite guard only

Reject an existing cycle filename immediately before publication.

This prevents replacement but still permits multiple architectures to share a
directory until their cycle numbers collide. It also allows a fresh-start run
to waste training time before failing. Rejected as incomplete.

### Layered isolation and recovery

Combine configuration isolation, preflight validation, a persistent root
identity, a save-time no-overwrite guard, verified copy of the recoverable 100M
candidate, and evidence-backed gold recovery status.

Selected because it fails early, defends against configuration mistakes, and
retains a final guard at the mutation boundary.

## Architecture

### Checkpoint root identity

Each writable checkpoint root owns a small atomic JSON file named
`lineage_root.json`. Its schema contains:

- `schema_version`;
- `model_name`;
- `config_identity`;
- `tokenizer_id`;
- `created_at`;
- `creation_mode` (`fresh_start` or `resume`);
- optional `base_checkpoint_id`;
- optional `source_checkpoint_sha256`.

The identity is created only for an empty, explicitly selected root. Existing
roots without the identity remain readable, but are not writable until they
are classified through the recovery procedure. This prevents automatically
blessing the contaminated canonical directory.

### CLI preflight

Before model construction or GPU allocation:

- resolve the effective checkpoint root to an absolute path;
- for `run247 --fresh-start`, require that the root contain neither organism
  checkpoints nor an incompatible root identity;
- for `run247 --resume`, inspect the resume artifact and require the root
  identity to match its model, configuration identity, tokenizer identity, and
  base checkpoint identity where available;
- reject use of the canonical shared root for a 100M fresh start;
- report the conflicting files and identities without modifying them.

An empty isolated 100M root may be initialized. A root containing the verified
copied cycle 1 must be continued with `--resume`, not `--fresh-start`.

### Save-time collision guard

Immediately before starting the synchronous or asynchronous writer:

- fail if the final `.pt` path already exists;
- fail if the `.pt.tmp` path exists;
- revalidate the root identity against the payload being saved;
- never call replacement publication for an existing final checkpoint.

The existing atomic temporary-file-to-final-file operation remains unchanged
for a new target. Pointer publication happens only after the new checkpoint
passes its existing validation.

### 100M containment

The 100M YAML will use:

`checkpoint_root: "workspace/03_CHECKPOINTS_100M"`

Containment will:

1. create the isolated directory;
2. inspect the current
   `workspace/03_CHECKPOINTS/organism_cycle_001.pt`;
3. copy it to a temporary filename under the isolated root;
4. verify source and copy byte size and SHA-256;
5. inspect the copy for strict resume compatibility with the 100M config;
6. atomically publish the copied file and its isolated pointer;
7. write `lineage_root.json`;
8. leave the source checkpoint and original pointer unchanged.

If any verification fails, the temporary copy is not promoted and the isolated
root is not marked usable.

### Gold recovery evidence

Create `workspace/04_MANIFESTOS/gold_recovery_status.json` recording:

- expected path, size, SHA-256, model identity, cycle, and step;
- current conflicting artifact identity;
- all local search roots examined;
- search results;
- VSS search status;
- recovery status (`missing`, `candidate_found`, or `recovered_verified`);
- timestamp and source commit.

A candidate becomes recovered only after:

1. SHA-256 equals the recorded gold SHA;
2. size equals 10,876,850,383 bytes;
3. the official inspector reports the embedded 1.6B config, expected shapes,
   valid topology, compatible optimizer, cycle 71, and step 40751.

The current non-administrator search found no accessible candidate. A separate
read-only administrator script will enumerate Volume Shadow Copies and test
the historical canonical path. It will not copy or restore anything unless a
candidate first matches size and SHA-256.

## Data Flow

For a new lineage:

`CLI arguments -> resolved root -> empty-root validation -> lineage identity -> bootstrap -> train -> save collision guard -> atomic checkpoint -> atomic pointer`

For resume:

`resume checkpoint -> official inspection -> root identity match -> bootstrap -> train -> new cycle-name guard -> atomic checkpoint -> atomic pointer`

For recovery:

`gold manifest -> local/VSS search -> size filter -> SHA-256 -> official inspection -> verified recovery status`

## Error Handling

All safety failures are fail-closed and occur before training where possible.
Messages must include:

- resolved checkpoint root;
- conflicting checkpoint or identity file;
- requested model name;
- requested mode (`fresh_start` or `resume`);
- safe corrective action.

Save-time collision errors must be surfaced through the existing asynchronous
save join so the process cannot silently continue after a failed writer.

Recovery failures update only the recovery manifest. They never mutate the
candidate or canonical checkpoint directories.

## Tests

Focused tests will prove:

1. fresh start rejects a non-empty checkpoint root;
2. fresh start accepts an empty isolated root and creates the expected
   identity;
3. resume rejects a root identity belonging to another model;
4. resume accepts the verified 100M cycle 1 in the isolated root;
5. save rejects an existing final checkpoint without changing its bytes;
6. save rejects an existing temporary checkpoint;
7. asynchronous save propagates collision failure;
8. 100M configuration no longer references the shared canonical root;
9. recovery requires exact SHA, size, model topology, cycle, and step;
10. containment copy preserves source bytes and produces a verified copy;
11. the original pointer and contaminated root remain unchanged.

After focused tests, run:

```powershell
powershell -ExecutionPolicy Bypass -File src\\scripts\\start_overnight_16b.ps1 -Canary
```

This is the supported no-launch operational gate. No training is started as
part of this design.

## Rollback

Code and configuration changes can be reverted through their explicit Git
paths. The new isolated directory and manifests are additive. Existing
checkpoints remain untouched, so rollback never requires restoring or
reconstructing an overwritten artifact.

The verified 100M copy may remain as evidence even if configuration rollback
is required. It must not be deleted automatically.

