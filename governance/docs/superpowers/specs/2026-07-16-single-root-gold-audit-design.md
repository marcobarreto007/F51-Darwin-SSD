# F51 Darwin-X Single-Root GOLD Audit Design

**Date:** 2026-07-16
**Status:** Approved by Marco on 2026-07-16
**Scope:** Local workspace consolidation, operational recovery, repository
governance, reproducibility, and external-audit readiness. Remote cloud state is
not an operational dependency and is not consulted as authority.

## 1. Final Objective

Turn `C:\Users\marco\Desktop\F51-Darwin-SSD` into the only physical and
operational root for Darwin-X while preserving the trained
`F51-Darwin-X-1.6B-Nitro` lineage and making the checkout independently
understandable, testable, reproducible, and defensible in an external audit.

The finished system must answer these questions without relying on oral history:

1. **What does it do?** It trains, resumes, evaluates, evolves, and locally
   serves the Darwin-X hybrid SSD + Attention + MoE organism.
2. **Who is it for?** Marco/F51 Labs as the local research and demonstration
   operator.
3. **What is the main flow?** Verify workspace and lineage, run a non-launching
   readiness gate, run an isolated canary, compare the candidate with the frozen
   baseline, and require a human decision before continuous training.
4. **What starts it?** `src/scripts/start_overnight_16b.ps1 -Canary` for the gate and
   `src/scripts/start_overnight_16b.ps1 -Canary -Launch` only for an explicitly
   authorized local training launch.
5. **What proves it works?** A clean source commit; verified corpus, tokenizer,
   checkpoint and manifests; a passing CPU suite; a successful non-launching
   readiness report; a bounded two-GPU canary; finite metrics; an atomically
   published candidate; and an external baseline comparison.
6. **What blocks GOLD?** Any unresolved path, missing identity, stale canonical
   document, non-reproducible dependency, unreviewed destructive automation,
   failing test, unclassified artifact, or unsupported readiness claim.

## 2. Approved Architecture

The repository root is also the local system root. Heavy and generated state is
kept inside one ignored `workspace/` subtree; source and audit controls remain
tracked by Git.

```text
F51-Darwin-SSD/
  src/f51_darwin/                 Python package and model/runtime libraries
  src/scripts/                    Reviewed operational entrypoints only
  src/configs/                    Active local configurations
  src/tests/                      Automated regression and contract tests
  governance/docs/                       Canonical, research, and historical documents
  workspace/                  Ignored physical runtime workspace
    00_BRUTOS/                Governed raw inputs
    01_TOKENIZADOS/           Token binaries and their manifests
    02_CORPUS/                Candidate/quarantine/approved corpus lifecycle
    03_CHECKPOINTS/           Checkpoints and canonical latest pointer
    04_MANIFESTOS/            Integrity, readiness, and provenance manifests
    tokenizer/                Tokenizer artifacts bound to the lineage
    runtime/
      runs/                   Per-run reports and PID metadata
      logs/                   Append-only runtime logs
      evaluations/            Benchmark and comparison outputs
  governance/archive/                    Tracked, non-operational historical material
    legacy-scratch/           One-off local/cloud helper scripts
    cloud-local/              Local cloud-era src/scripts/config/docs retained only
                               when they carry unique historical evidence
    historical-governance/docs/          Superseded root reports and roadmaps
  governance/audit/                      Tracked audit policy and generated-report schemas
    provenance/               Source/data/checkpoint lineage declarations
    test-reports/             Commit-bound test report templates and summaries
    sbom/                     Dependency inventory and SBOM artifacts
```

There is no sibling `F51-Dataset-Organizado`, no broken `data/` junction, and no
root-level `checkpoints/` compatibility junction in the final state.

## 3. Workspace Resolution Contract

`f51_darwin.dataset_layout.resolve_dataset_root(project_root, require=False)`
resolves paths in this order:

1. `F51_DATASET_ROOT`, when explicitly set;
2. `<project_root>/workspace`.

The environment override remains available for isolated tests and deliberate
operator overrides. It is never required for the default local flow.

Resolution must be project-root-relative. A process-level environment variable
must not leak into tests that intentionally construct a different temporary
project root unless that test explicitly opts into the override.

Canonical relative locations remain numbered to preserve lineage semantics:

- corpus: `01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast_v2.bin`;
- corpus manifest:
  `01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast_v2.bin.manifest.json`;
- checkpoint root: `03_CHECKPOINTS/`;
- pointer: `03_CHECKPOINTS/organism_latest.json`;
- readiness manifest: `04_MANIFESTOS/overnight_16b_readiness.json`.

All operational code consumes these locations through `dataset_layout.py` or a
single typed path object. New hard-coded absolute Desktop paths are forbidden.

## 4. Lineage Preservation and Canonical Selection

The migration does not promote a checkpoint by filename or modification time.
Before publishing a new pointer, the implementation must verify the current
local candidate `organism_cycle_071.pt` on CPU with:

- checkpoint format version 7;
- embedded config equal to `src/configs/darwin_x_1.6b_nitro.yaml`;
- cycle 71 and step 40751;
- complete model shape compatibility;
- Topology Manifest validity;
- AdamW resume compatibility;
- recalculated integral identity;
- expected `base_checkpoint_id`;
- full-file SHA-256 persisted in an audit manifest.

`organism_cycle_077.pt` remains preserved but non-canonical until it passes the
same verification and an explicit lineage comparison. Its larger cycle number
alone is not promotion evidence.

The `feast_v2` corpus becomes canonical only after full SHA-256, byte count,
int32 divisibility, token count, tokenizer identity, and manifest composition
are verified. The existing manifest declares 74,195,890,756 bytes,
18,548,972,689 tokens, and SHA-256
`9677e9f22f4d78efa7b25c77b2da3cbdb5fb2a499926ac641a75c13b65e25cff`;
the migration must independently reproduce those values.

## 5. Migration Safety

The 223+ GiB workspace is already on the same volume as the target root. The
migration uses same-volume directory renames and never duplicates the complete
workspace.

Before changing paths:

1. stop and retire `_autocommit.ps1` processes;
2. prove no Darwin training, serving, Ghost ingest, or launcher process is using
   the paths;
3. record source HEAD, disk free space, file counts, directory sizes, hashes of
   canonical artifacts, and current reparse points;
4. write a reversible migration manifest with every source and destination;
5. fail closed if any target already contains conflicting content.

After the move, rollback means moving `workspace/` back to its recorded source
name before any new training or ingestion is allowed. No artifact is deleted as
part of the path migration.

Destructive cleanup is a later, evidence-gated operation. Exact duplicates may
be removed only after hashes, provenance, retention policy, and recovery path
are documented. Git object pruning is not part of the functional migration and
must not occur while relevant unreachable objects are the only recovery copy.

## 6. Operational Surface

The supported local entrypoints are:

- `src/scripts/start_overnight_16b.ps1` — readiness, canary, and authorized launch;
- `src/scripts/darwin_organism.py` — organism commands and service runtime;
- `src/scripts/serve_davi.py` — loopback-only local UI/API;
- `src/scripts/inspect_organism_checkpoint.py` — checkpoint verification;
- `src/scripts/darwin_inventory.py` — local inventory;
- `src/scripts/ingest_pipeline.py` — governed ingestion.

Alternative trainers, rental helpers, cloud launchers, one-off patch scripts,
and obsolete model launchers are not operational entrypoints. Unique historical
material moves under `governance/archive/`; exact copies and generated deployment packs are
removed from the tracked working tree after review.

Remote cloud state, instances, endpoints, and remote checkpoints are outside
the definition of local readiness. Canonical documentation may mention cloud
only as historical context or an explicitly future experiment.

## 7. Git and Automation Governance

`src/scripts/_autocommit.ps1` is retired. No background process may run `git add -A`
or `git commit --no-verify`.

Every implementation commit must:

- represent one coherent task;
- use a Conventional Commit message;
- stage explicit paths;
- pass its focused verification before commit;
- avoid `--no-verify`;
- contain no corpus, checkpoint, raw dataset, credential, environment secret,
  generated log, or virtual environment.

The repository must include a CI workflow that runs the CPU-safe suite, syntax
checks, link checks, dependency policy checks, and secret scanning against the
committed source tree. Local verification remains authoritative for hardware
and heavyweight artifact gates that hosted CI cannot reproduce.

## 8. Reproducibility, Legal, and Supply Chain

The audit-ready repository includes:

- a proprietary `LICENSE`/notice identifying F51 Labs and reserving rights;
- a dependency policy with exact direct versions for the supported Python
  runtime;
- a machine-readable environment lock or resolved snapshot;
- a CycloneDX-compatible SBOM or equivalent complete dependency inventory;
- dependency vulnerability and license reports with tool versions and dates;
- installer provenance or removal of opaque installer binaries;
- a `SECURITY.md` with reporting, secret handling, and supported-scope policy;
- a source provenance manifest tied to the audited commit.

No open-source redistribution grant is inferred. Third-party notices remain
separate from the proprietary project license.

## 9. Documentation Contract

Canonical documents are deliberately small and non-overlapping:

- `README.md` — purpose, five-minute orientation, supported commands, and links;
- `AGENTS.md` and `CLAUDE.md` — identical safety and authority rules;
- `governance/docs/CANONICAL_MAP.md` — durable source/component map;
- `governance/docs/operacao/STATUS_ATUAL.md` — dated, generated local snapshot;
- `governance/docs/operacao/OPERACAO_SEGURA.md` — stable gates, launch, rollback, and
  prohibitions;
- `governance/docs/arquitetura/IMPLEMENTACAO_ATUAL.md` — code-backed architecture;
- `governance/docs/_historico/INDEX.md` — catalog of superseded documents;
- `governance/audit/README.md` — audit evidence index and reproduction commands.

Historical documents are preserved without being allowed to masquerade as
current status. Canonical documents must not identify cycle 68, the old
`feast.bin`, 2.5B, 5B, a remote GPU, or cloud runtime as the active local truth.

## 10. Error Handling and Fail-Closed Rules

The system refuses readiness or launch when any of these are true:

- workspace root or canonical artifact is missing;
- a path resolves outside the approved root without an explicit override;
- corpus or checkpoint identity differs from its manifest;
- pointer target is absent or fails strict resume verification;
- source worktree has tracked changes;
- another Darwin process is active;
- disk headroom is below the calculated checkpoint/canary requirement;
- GPU inventory does not satisfy the local config;
- the latest readiness report is for a different source commit or artifact
  identity.

Errors name the failed invariant, inspected path, expected identity, observed
identity, and corrective command. The launcher does not silently fall back to an
older corpus, checkpoint, config, or cloud artifact.

## 11. Testing and Verification Strategy

Testing is layered:

1. **Unit/TDD:** workspace resolution, environment isolation, canonical names,
   pointer parsing, manifest validation, and path-boundary failures.
2. **Focused integration:** ingestion quarantine/approval, latest-checkpoint
   resolution, readiness report, Davi loopback API, and migration dry-run.
3. **Full CPU suite:** `CUDA_VISIBLE_DEVICES=-1`, no cache provider, zero
   collection errors or failures.
4. **Static gates:** `py_compile`, PowerShell AST parse, Markdown link check,
   dependency/secret/license policy, and `git diff --check`.
5. **Artifact verification:** full corpus hash, checkpoint strict resume and
   identity, tokenizer hash, manifests, and pointer consistency.
6. **Operational dry-run:** official launcher without `-Launch`, producing a
   commit-bound readiness manifest inside `workspace/04_MANIFESTOS`.
7. **Hardware canary:** only after all earlier gates, using the two local GPUs,
   isolated checkpoint root, finite metrics, atomic candidate publication, and
   no canonical pointer promotion.
8. **External-audit simulation:** a fresh reviewer follows only tracked docs and
   audit commands, reproduces source-level gates, and reports no P0/P1 finding.

The implementation may be called GOLD only when every layer applicable to the
local machine has fresh evidence tied to the final source commit. Passing unit
tests alone is not GOLD.

## 12. Implementation Sequence

1. freeze baseline and governance;
2. add tests for the approved workspace contract;
3. migrate the physical workspace by reversible same-volume rename;
4. update resolver, launchers, ingestion, and runtime paths;
5. verify and publish canonical corpus/checkpoint manifests and pointer;
6. reduce and archive the operational surface;
7. reconcile canonical documentation;
8. add legal, dependency, SBOM, security, CI, and audit evidence controls;
9. run the full verification ladder and external-audit simulation;
10. publish a dated GOLD report only if all gates pass.

## 13. Non-Goals

- changing the Darwin-X architecture or parameter count;
- launching continuous training without a separate explicit operator command;
- promoting checkpoint 077 by name or date;
- deleting unique corpus, tokenizer, checkpoint, or historical evidence;
- using cloud state as a shortcut for local verification;
- rewriting the repository into a new framework;
- claiming model quality from training loss or process uptime.
