# Internal Gold Hardening Design

**Date:** 2026-07-17  
**Status:** Approved for execution by Marco's blanket authorization  
**Scope:** `C:\Users\marco\Desktop\F51-Darwin-SSD` only; local disk is authoritative

## Final objective

1. **What the system does:** Darwin-X evolves and serves the trained `F51-Darwin-X-1.6B-Nitro` organism, resuming a validated checkpoint and using the organized external corpus.
2. **Who it is for:** Marco/F51 operators, maintainers and independent technical auditors.
3. **Main user flow:** inspect readiness, validate lineage/configuration, launch or resume the organism through the official launcher, observe checkpoints, and serve Davi when required.
4. **Official start command:** `powershell -ExecutionPolicy Bypass -File src/scripts/start_overnight_16b.ps1`; it is a readiness gate unless `-Launch` is explicitly supplied.
5. **Proof it works:** green source gates, a checkpoint whose lineage/config/shapes are validated, a single dual-GPU organism process when intentionally launched, and reproducible package/audit artifacts.
6. **Current blockers to internal gold:** an oversized operational surface, three concentrated modules, root-local runtime residue, and no independently installable Python distribution.

The target is a repository that an internal auditor can understand from the root, install without the dataset, verify without a GPU, and trace from supported entrypoints to package modules without ambiguous duplicate surfaces.

## Non-negotiable constraints

- Preserve corpus, checkpoints, runtime history and model identity.
- Do not launch `run247`, allocate CUDA, rewrite Git history, prune `.git`, or touch cloud state.
- Keep the six supported operational entrypoints stable.
- Preserve public imports, serialized parameter names and `state_dict` compatibility during extraction.
- Move state on the same volume before removing its former root location; delete only caches or reproducible tooling.
- Historical audit evidence remains immutable. Current policies and reproduction instructions may supersede historical paths.
- Security credential history remains visible to its dedicated gate; this hardening does not conceal or waive it.

## Chosen approach: boundary-first hardening

The implementation uses incremental extraction behind compatibility facades. It does not rewrite the organism or alter training behavior. Boundaries are made explicit first, then code is moved behind those boundaries with characterization tests.

Two rejected approaches are:

- **Cosmetic cleanup only:** makes the root look smaller but leaves coupling, packaging and ownership ambiguous.
- **Full architectural rewrite:** creates unacceptable checkpoint and runtime risk for an audit-hardening objective.

## Target repository topology

```text
F51-Darwin-SSD/
|-- src/f51_darwin/              installable production package
|   |-- darwin_x_core/       Darwin-X contracts and model components
|   |-- organism/            checkpoint, runtime and cycle orchestration
|   `-- serving/             Davi application and HTTP adapter
|-- src/scripts/                 exactly six supported operator entrypoints + README
|-- src/tools/                   maintenance, migration, verification and audit utilities
|-- research/                experiments, benchmarks, generators and trainers
|-- src/tests/                   unit, characterization, contract and policy tests
|-- src/configs/                 canonical configuration
|-- governance/docs/                    canonical documentation and audit material
|-- governance/audit/                   immutable evidence and current audit indexes
|-- governance/archive/                 historical material, excluded from distribution
`-- workspace/               ignored runtime, corpus/checkpoint links and local state
```

The supported surface remains:

1. `src/scripts/start_overnight_16b.ps1`
2. `src/scripts/darwin_organism.py`
3. `src/scripts/serve_davi.py`
4. `src/scripts/inspect_organism_checkpoint.py`
5. `src/scripts/darwin_inventory.py`
6. `src/scripts/ingest_pipeline.py`

All other current `src/scripts/` files receive an explicit `src/tools/` or `research/` ownership. The surface checker will reject unclassified files.

## Physical hygiene

Root-local mutable state is consolidated under `workspace/runtime/`, while tracked stale material is moved to the explicit archive:

- mixed legacy `.organism/` -> `workspace/runtime/quarantine/legacy-organism/`; the canonical runtime starts at the separate `workspace/runtime/organism/` path;
- tracked stale `.f51/` evidence -> `governance/archive/legacy-state/f51/`; any future mutable baseline belongs under `workspace/runtime/baselines/`;
- existing `.agent_bus/` notes -> `governance/archive/agent-bus/`; future session notes use `workspace/runtime/history/agent_bus/`;
- `eval_results/` -> `workspace/runtime/evaluations/` when material exists

The old `.organism/` directory is quarantined because it contains mutually incompatible 1.6B, 600M and untyped Ghost state. No file from it is promoted as current lineage merely because its cycle number or filename appears recent. The stale `.f51/agent_memory.json` and baselines are retained as history, not treated as live status.

Active configuration and code defaults are updated before the old locations disappear. The mover rejects reparse points and uses same-volume rename semantics, with a pre/post hash manifest and explicit rollback map. Root `__pycache__`, `.pytest_cache`, empty evaluation directories and the obsolete `.venv_vast` environment are removed after exact-path and live-process checks. `.audit-tools` is reduced to the pinned Gitleaks binary required by the source gate; cached wheels, temporary environments and logs are reproducible and removed.

A machine-readable root allowlist and `check_physical_hygiene.py` enforce:

- no forbidden legacy state directories at root;
- no Python/test caches outside explicitly excluded environments, workspace and archive;
- no unclassified root entries;
- no large local audit cache other than the pinned scanner;
- required local runtime directories remain ignored by Git.

`.git`, `.venv_nitro`, `workspace`, checkpoints and corpora are explicitly preserved.

## Packaging and distribution

`pyproject.toml` becomes a complete PEP 517 definition:

- pinned Setupsrc/tools/Wheel build backend;
- explicit discovery of `f51_darwin*` only;
- exclusion of tests, tools, research, workspace, archive and audit payloads;
- package version exposed without importing Torch;
- a small package CLI with `version` and audit-safe inspection commands;
- lazy top-level exports so a base installation can import the package before optional runtime dependencies are installed.

`uv build` produces both wheel and source distribution from a clean `git archive` of the frozen source commit, not from an untracked working tree. Build requirements are pinned and their resolved artifacts/hashes are recorded; offline reuse is preferred when the pinned cache is available. A distribution verifier must prove:

- both artifacts build from a clean tree;
- wheel and sdist contents match exact allowlists and include any required serving package assets;
- neither artifact includes runtime state, datasets, checkpoints, archive, tests, tools, research, secrets or local paths;
- the wheel installs into a fresh temporary environment;
- `import f51_darwin` and the version CLI succeed without GPU allocation;
- metadata and version agree.

Build output is ephemeral and remains ignored.

## Modular architecture

### Darwin-X model

`src/f51_darwin/darwin_x.py` becomes a compatibility facade. The new namespace is deliberately `darwin_x_core/` because `src/f51_darwin/model.py` is an existing legacy module and Windows cannot host both `model.py` and `model/` as an unambiguous import surface. Public names remain importable from their original location while implementation moves to:

- `src/f51_darwin/darwin_x_core/contracts.py`: configuration and typed contracts;
- `src/f51_darwin/darwin_x_core/blocks.py`: embeddings, attention and reusable blocks;
- `src/f51_darwin/darwin_x_core/neuroendocrine.py`: neuroendocrine state and control;
- `src/f51_darwin/darwin_x_core/experts.py`: expert implementations;
- `src/f51_darwin/darwin_x_core/routing.py`: routing and balancing;
- `src/f51_darwin/darwin_x_core/topology.py`: topology and placement;
- `src/f51_darwin/darwin_x_core/gradient_policy.py`: lifecycle and gradient policy;
- `src/f51_darwin/darwin_x_core/moe.py`: thin MoE composition;
- `src/f51_darwin/darwin_x_core/model.py`: composed `DarwinXModel`.

No module renames parameters after construction. Characterization tests compare public symbols, config round trips, ordered `named_parameters`, optimizer-group order, state keys/shapes/dtypes, tied-weight aliases, topology manifest, strict checkpoint load, deterministic CPU output and backbone identity. The pre-existing `model.py`, `moe_layer.py` and `attention_block.py` receive explicit legacy/compatibility ownership rather than becoming competing authorities.

### Organism runtime

`src/scripts/darwin_organism.py` becomes a thin CLI adapter. Implementation moves to:

- `src/f51_darwin/organism/checkpoint.py`: validation, resume and identity;
- `src/f51_darwin/organism/optimizer_state.py`: optimizer restoration;
- `src/f51_darwin/organism/runtime.py`: construction and dependency wiring;
- `src/f51_darwin/organism/cycle.py`: cycle execution and persistence;
- `src/f51_darwin/organism/cli.py`: argument validation and command orchestration.

The facade re-exports compatibility symbols used by existing tests and tools. Tests must cover bootstrap and CLI paths that previously relied on `__new__`-based mocks. The generic legacy `src/f51_darwin/checkpointing.py` (including its permissive `strict=False` path) is migrated, renamed or retired so the fail-closed v7 checkpoint authority is unique.

The current organism/Davi private callback coupling becomes an explicit typed safe-boundary protocol for pause registration and snapshot requests. State transitions between `model.train()` and `model.eval()` are fail-closed and covered by error-path tests; serving cannot silently continue after a transition failure.

### Davi serving

`src/scripts/serve_davi.py` becomes a thin entrypoint. Static UI, application/controller logic and HTTP transport move into `src/f51_darwin/serving/`. Network behavior and response contracts remain unchanged.

### Dependency direction

Allowed dependency direction is:

```text
scripts -> f51_darwin
tools   -> f51_darwin
research -> f51_darwin
f51_darwin -> Python/declared dependencies
```

Production package modules never import `scripts`, `tools` or `research`. No active Python in the package, scripts, tools or research trees mutates `sys.path`; tools use installed-package execution or an explicit subprocess working directory instead. A static gate enforces both rules and rejects package dependency cycles.

## Measurable gold gates

Internal gold requires all of the following:

- `src/scripts/` contains only the six supported entrypoints and its README;
- no supported operational file exceeds 1,200 lines; entrypoint adapters target at most 250 lines; no class exceeds 500 lines and no function/method exceeds 180 lines without a documented policy exception;
- no active Python file in package, scripts, tools or research mutates `sys.path`;
- package import graph is acyclic and obeys dependency direction;
- canonical local defaults resolve to `src/configs/darwin_x_1.6b_nitro.yaml`; legacy 4B/600M defaults are rejected without changing checkpoint identity;
- physical root policy passes and forbidden legacy state/cache paths are absent;
- wheel and sdist build, pass content inspection and install smoke;
- public symbol and `state_dict` compatibility tests pass;
- Python compilation and PowerShell parsing pass;
- full CPU-safe test suite passes;
- documentation links, canonical-map, duplicate-policy, dependency-policy, operational-surface and supply-chain gates pass;
- final evidence records exact commit, commands, outputs and any remaining external blocker.

No gold claim may rely on a filename, loss value or document alone.

## Implementation sequence

1. Add characterization tests and policy gates while the old layout is intact, including canonical-default drift and legacy-authority checks.
2. Consolidate physical state and remove only verified reproducible residue.
3. Classify and move maintenance/research scripts, updating live references.
4. Complete packaging and distribution verification.
5. Extract the Darwin-X model behind its facade.
6. Extract organism and Davi runtime code behind thin entrypoints and an explicit safe-boundary protocol.
7. Freeze the source commit, regenerate supply-chain indexes/attestations in dependency order, then run the full independent audit and reconcile canonical documentation.

Each stage is committed separately and must leave its focused tests green. Test and compile commands use `-B`, `PYTHONPYCACHEPREFIX` outside the repository and `-p no:cacheprovider`; physical hygiene runs last. If a compatibility test fails, the extraction is corrected rather than weakening the test.

## Rollback and recovery

Git-tracked changes are recoverable commit-by-commit. Mutable-state moves reject reparse points and are recorded with source, destination, file count and hashes before cleanup. No corpus, checkpoint or runtime history is deleted. Any failure after a state move restores the original default/path or completes the migration before proceeding. `workspace/` contains the physical local artifacts governed by this repository; it is not described or treated as a cloud link.
