# Internal Gold Hardening Implementation Plan

> **Execution rule:** complete each task with focused red/green verification and a scoped commit. Preserve model identity and runtime artifacts throughout.

**Goal:** make the local Darwin-X repository independently auditable at internal-gold level for organization, modularity, distribution, documentation and physical hygiene.

**Architecture:** keep the six operator entrypoints stable, extract implementations into an installable `f51_darwin` package, place maintenance and research executables in owned top-level areas, and consolidate mutable state under `workspace/runtime`. Compatibility facades preserve imports and serialization.

**Environment:** Windows PowerShell, Python 3.12, PyTorch runtime in `.venv_nitro`, Setupsrc/tools/Wheel via `uv build`, pytest, Git and Gitleaks.

**Design:** `governance/docs/superpowers/specs/2026-07-17-internal-gold-hardening-design.md`

---

## Task 1: Establish gold policies and characterization baselines

**Files:**

- Create: `governance/audit/policy/physical-root.json`
- Create: `src/tools/check_physical_hygiene.py`
- Create: `src/tools/check_architecture_boundaries.py`
- Create: `src/tools/check_distribution.py`
- Create: `src/tests/test_physical_hygiene.py`
- Create: `src/tests/test_architecture_boundaries.py`
- Create: `src/tests/test_distribution_policy.py`
- Create: `src/tests/test_darwin_x_public_contract.py`
- Modify: `governance/audit/policy/operational-surface.json`
- Modify: `src/tests/test_operational_surface.py`

**Step 1: Write failing policy tests**

Cover the target root allowlist, forbidden legacy state directories, cache exclusions, exact six-entrypoint `src/scripts/` surface, allowed dependency direction, absence of `sys.path` mutation across all active Python, canonical 1.6B default configuration, unique fail-closed checkpoint authority, class/function size budgets, and distribution exclusions.

**Step 2: Capture model compatibility**

Assert the public names imported from `f51_darwin.darwin_x`, configuration round trip, ordered `named_parameters`, optimizer-group order, state keys/shapes/dtypes, tied-weight aliases, topology manifest, strict load, deterministic CPU outputs and backbone identity. Store only deterministic metadata in the test fixture, never a checkpoint.

**Step 3: Prove the tests fail for the known gaps**

Run:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:PYTHONPYCACHEPREFIX="$env:TEMP\f51-gold-pycache"
.\.venv_nitro\Scripts\python.exe -B -m pytest -q -p no:cacheprovider src/tests/test_physical_hygiene.py src/tests/test_architecture_boundaries.py src/tests/test_distribution_policy.py src/tests/test_darwin_x_public_contract.py
Remove-Item Env:PYTHONPYCACHEPREFIX
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

Expected: policy failures identify root state, `sys.path`, oversized entrypoints and missing build metadata; compatibility tests pass.

**Step 4: Implement reusable checkers and make fixture-level tests green**

Checkers accept `--root`, emit stable JSON, avoid importing Torch, and return `0` only on a clean result.

**Step 5: Commit**

```powershell
git add governance/audit/policy tools tests
git commit -m "test: establish internal gold policy gates"
```

## Task 2: Consolidate physical state and remove reproducible residue

**Files:**

- Modify: `src/configs/organism.yaml`
- Modify: active `src/f51_darwin/**/*.py` default state paths
- Modify: active `src/scripts/*.py` default state paths
- Modify: `.gitignore`
- Modify: `src/tools/check_physical_hygiene.py`
- Create: `governance/audit/current/physical-migration-manifest.json`
- Modify: canonical operator documentation that names active state paths

**Step 1: Resolve exact targets and live users**

Record absolute paths, sizes, file counts and hashes for `.organism`, `.f51`, `.agent_bus`, `.audit-tools` and `.venv_vast`. Confirm no live process command line uses `.venv_vast` or the state source paths. Reject reparse points and verify every resolved source/destination remains under the intended repository/workspace root.

**Step 2: Update defaults before moving data**

Use a new `workspace/runtime/organism` for canonical state. Quarantine the mixed old `.organism` tree at `workspace/runtime/quarantine/legacy-organism`; archive tracked `.f51` evidence at `governance/archive/legacy-state/f51`; archive existing `.agent_bus` notes at `governance/archive/agent-bus`; use `workspace/runtime/history/agent_bus` only for future notes and `workspace/runtime/evaluations` for current evaluations. Preserve environment/config overrides.

**Step 3: Move persistent state on the same volume**

Use native PowerShell `Move-Item -LiteralPath` with same-volume rename semantics only after resolving and validating both source and destination under the repository. Record post-move hashes/counts and a rollback map in the manifest.

**Step 4: Remove reproducible residue**

Delete only the verified root caches, empty `eval_results`, obsolete `.venv_vast`, and regenerable `.audit-tools` content other than the pinned Gitleaks binary. Do not touch `.git`, `.venv_nitro`, `workspace`, corpus or checkpoints.

**Step 5: Verify**

```powershell
.\.venv_nitro\Scripts\python.exe src/tools/check_physical_hygiene.py
git status --short --ignored
```

Expected: `status=pass`; persistent state is present at its destination and absent from legacy root paths.

**Step 6: Commit**

```powershell
git add .gitignore configs f51_darwin scripts tools tests docs governance/audit/current/physical-migration-manifest.json
git commit -m "chore: consolidate local runtime state"
```

## Task 3: Make the operational surface exact

**Files:**

- Move: maintenance executables from `src/scripts/` to `src/tools/`
- Move: research executables from `src/scripts/` to `research/`
- Keep: the six supported entrypoints in `src/scripts/`
- Create: `src/scripts/README.md`
- Create: `src/tools/README.md`
- Create: `research/README.md`
- Modify: `governance/audit/policy/operational-surface.json`
- Modify: `.github/workflows/ci.yml`
- Modify: active tests, canonical docs and current audit indexes containing moved paths

**Step 1: Generate an exact move manifest from the existing classifications**

Every executable classified `maintenance` moves to `src/tools/<basename>`; every executable classified `research` moves to `research/<basename>`. Write a complete old-to-new manifest and abort on basename collisions. Preserve Git history with `git mv`.

**Step 2: Update live references mechanically and review them semantically**

Update CI, tests, PowerShell peer calls, current docs, current policies and supply-chain reproduction instructions. Do not rewrite immutable historical evidence merely to modernize paths. Regenerate supply-chain indexes/attestations only after the source commit is frozen, in the documented dependency order, so evidence never hashes itself.

**Step 3: Strengthen the surface checker**

Scan `scripts`, `tools` and `research`; require each executable to match its directory ownership and require the exact supported set in `scripts`.

**Step 4: Verify**

```powershell
.\.venv_nitro\Scripts\python.exe src/tools/check_operational_surface.py
rg -n "src/scripts/(check_|bench|benchmark|generate_|train_|evolve_|auto_|migrate_)" .github docs audit tests tools research
```

Expected: surface checker passes; remaining old-path matches are only explicitly historical.

**Step 5: Commit**

```powershell
git add -A scripts tools research .github tests docs audit
git commit -m "refactor: separate operations maintenance and research"
```

## Task 4: Produce a clean installable distribution

**Files:**

- Modify: `pyproject.toml`
- Modify: `src/f51_darwin/__init__.py`
- Create: `src/f51_darwin/_version.py`
- Create: `src/f51_darwin/cli.py`
- Create: `src/f51_darwin/__main__.py`
- Modify: `.gitignore`
- Modify: `src/tools/check_distribution.py`
- Modify: `src/tests/test_distribution_policy.py`

**Step 1: Make base import lazy**

Expose version without importing Torch. Preserve current top-level names through lazy `__getattr__` exports where needed. Package serving UI assets explicitly and test their presence through `importlib.resources` in an installed wheel.

**Step 2: Add complete PEP 517 metadata**

Pin the build backend, discover only `f51_darwin*`, declare package CLI and exclude non-package trees.

**Step 3: Build and inspect artifacts**

Freeze the source commit and build from its clean `git archive`, not from the working directory. Pin build requirements, record resolved hashes and enforce exact wheel/sdist content allowlists.

```powershell
uv build
.\.venv_nitro\Scripts\python.exe src/tools/check_distribution.py --dist dist
```

**Step 4: Fresh-environment smoke**

Create an explicit temporary directory, install the wheel with `uv pip install --python <temp-python> --no-deps`, and run `import f51_darwin`, `python -m f51_darwin version` and the console script. Remove only that validated temp directory afterwards.

**Step 5: Commit**

```powershell
git add pyproject.toml .gitignore f51_darwin tools tests
git commit -m "build: add verified Python distribution"
```

## Task 5: Extract Darwin-X behind a compatibility facade

**Files:**

- Create: `src/f51_darwin/darwin_x_core/__init__.py`
- Create: `src/f51_darwin/darwin_x_core/contracts.py`
- Create: `src/f51_darwin/darwin_x_core/blocks.py`
- Create: `src/f51_darwin/darwin_x_core/neuroendocrine.py`
- Create: `src/f51_darwin/darwin_x_core/experts.py`
- Create: `src/f51_darwin/darwin_x_core/routing.py`
- Create: `src/f51_darwin/darwin_x_core/topology.py`
- Create: `src/f51_darwin/darwin_x_core/gradient_policy.py`
- Create: `src/f51_darwin/darwin_x_core/moe.py`
- Create: `src/f51_darwin/darwin_x_core/model.py`
- Modify: `src/f51_darwin/darwin_x.py`
- Modify: focused model tests

**Step 1: Lock public and serialization behavior with characterization tests**

Run the contract test before every extraction slice.

**Step 2: Extract contracts and leaf blocks**

Move definitions without changing bodies. Re-export them from the old module. Fix only import direction. Keep `src/f51_darwin/model.py`, `moe_layer.py` and `attention_block.py` explicitly marked as compatibility modules rather than competing current authorities.

**Step 3: Extract neuroendocrine and MoE components**

Keep construction order, attribute names, parameter registration and routing semantics identical. Split experts, routing, topology/placement and gradient policy so `DeepSeekStyleMoE` becomes a thin composition rather than a relocated God object.

**Step 4: Extract the composed model**

Move `DarwinXModel` last. Make `src/f51_darwin/darwin_x.py` a documented compatibility facade.

**Step 5: Verify after each slice**

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:PYTHONPYCACHEPREFIX="$env:TEMP\f51-gold-pycache"
.\.venv_nitro\Scripts\python.exe -B -m pytest -q -p no:cacheprovider src/tests/test_darwin_x_public_contract.py src/tests/test_darwin_x.py src/tests/test_darwin_x_architecture.py
Remove-Item Env:PYTHONPYCACHEPREFIX
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

Expected: identical public symbols, deterministic outputs, ordered parameters/groups, state keys/shapes/dtypes, tied aliases, topology manifest, strict load and backbone identity.

**Step 6: Commit**

```powershell
git add f51_darwin tests
git commit -m "refactor: modularize Darwin-X model internals"
```

## Task 6: Extract organism and Davi applications behind thin adapters

**Files:**

- Create: `src/f51_darwin/organism/{__init__,contracts,checkpoint,optimizer_state,bootstrap,runtime,cycle,cli}.py`
- Create: `src/f51_darwin/serving/{__init__,application,http,static_ui}.py`
- Modify: `src/scripts/darwin_organism.py`
- Modify: `src/scripts/serve_davi.py`
- Create/modify: organism bootstrap, CLI and Davi response contract tests

**Step 1: Add missing characterization tests**

Cover argument validation, canonical 1.6B config selection, checkpoint selection, bootstrap wiring, optimizer-state restoration, safe-boundary pause/snapshot protocol, save delegation, HTTP routes, train/eval transition failures and representative response bodies without allocating CUDA or binding a public interface.

**Step 2: Extract checkpoint and optimizer functions**

Keep compatibility re-exports in the script while active callers migrate to the package. Migrate, rename or retire the permissive generic `src/f51_darwin/checkpointing.py` path so v7 fail-closed loading is the unique current authority.

**Step 3: Extract runtime/cycle orchestration**

Move cohesive methods without changing order or side effects. Replace the private organism/Davi callback injection with a typed safe-boundary protocol. Pass filesystem/process dependencies explicitly where tests need isolation.

**Step 4: Extract Davi UI/application/transport**

Keep the supported script as the command adapter; preserve route and response contracts. A failed `model.eval()`/`model.train()` transition must stop the request path and surface a typed error.

**Step 5: Enforce size and dependency gates**

```powershell
.\.venv_nitro\Scripts\python.exe src/tools/check_architecture_boundaries.py
```

Expected: no supported operational file over 1,200 lines, adapters target 150 lines, no class over 500 lines or function/method over 150 lines without policy exception, no active Python `sys.path` mutation, no reverse dependencies or package cycles.

**Step 6: Commit**

```powershell
git add f51_darwin scripts tests tools
git commit -m "refactor: separate organism and serving applications"
```

## Task 7: Run the independent gold audit and publish evidence

**Files:**

- Create: `governance/audit/current/internal-gold-report.md`
- Create: `governance/audit/current/internal-gold-evidence.json`
- Modify: `README.md`
- Modify: `governance/docs/CANONICAL_MAP.md`
- Modify: `governance/docs/operacao/STATUS_ATUAL.md`
- Modify: `.agent_bus` replacement/current operator note under `workspace/runtime/history/agent_bus/`

**Step 1: Run CPU-safe source gates**

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:PYTHONPYCACHEPREFIX="$env:TEMP\f51-gold-pycache"
.\.venv_nitro\Scripts\python.exe -B -m py_compile src/scripts/darwin_organism.py src/f51_darwin/darwin_x.py
.\.venv_nitro\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
Remove-Item Env:PYTHONPYCACHEPREFIX
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

**Step 2: Run policy and distribution gates**

```powershell
.\.venv_nitro\Scripts\python.exe src/tools/check_docs_links.py
.\.venv_nitro\Scripts\python.exe src/tools/check_canonical_docs.py
.\.venv_nitro\Scripts\python.exe src/tools/check_duplicates.py
.\.venv_nitro\Scripts\python.exe src/tools/check_dependency_policy.py
.\.venv_nitro\Scripts\python.exe src/tools/check_operational_surface.py
.\.venv_nitro\Scripts\python.exe src/tools/check_physical_hygiene.py
.\.venv_nitro\Scripts\python.exe src/tools/check_architecture_boundaries.py
uv build
.\.venv_nitro\Scripts\python.exe src/tools/check_distribution.py --dist dist
.\.venv_nitro\Scripts\python.exe src/tools/check_physical_hygiene.py
```

**Step 3: Parse PowerShell and inspect Git**

Parse all supported PowerShell scripts with the PowerShell parser; run `git diff --check`, `git status --short`, `git ls-files` distribution/root checks, and record the exact commit.

**Step 4: Perform adversarial review**

Have an independent agent inspect only the final commit and evidence against the design's measurable gates. Fix all internal P0/P1 findings and rerun affected gates.

**Step 5: Reconcile canonical documentation and evidence**

The report must distinguish internal gold dimensions from the explicitly separate historical-credential finding. Record every command, exit code, artifact hash and any unexecuted GPU/runtime gate.

**Step 6: Commit and verify clean state**

```powershell
git add README.md docs governance/audit/current
git commit -m "docs: publish internal gold audit evidence"
git status --short
```

Expected: clean worktree; all internal-gold organization, modularity, distribution, documentation and physical-hygiene gates pass on the recorded commit.
