# F51 Darwin-X Single-Root GOLD Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `F51-Darwin-SSD` into the only local physical and operational root, restore the verified 1.6B Nitro lineage, and produce commit-bound evidence suitable for an external GOLD audit.

**Architecture:** Source remains tracked at the repository root while all heavy and generated state moves by same-volume rename into ignored `workspace/`. A typed path contract, strict artifact verifier, reduced operational surface, reproducible dependency controls, canonical documentation, and layered local verification replace sibling paths, junctions, cloud-era drift, and unreviewed automation.

**Tech Stack:** Windows PowerShell 7, CPython 3.12, PyTorch, pytest, Git, JSON/JSON Schema, SHA-256, GitHub Actions, CycloneDX SBOM.

## Global Constraints

- The only physical and operational root is `C:\Users\marco\Desktop\F51-Darwin-SSD`.
- Heavy state lives under ignored `<project_root>/workspace`; no sibling dataset root or compatibility junction remains.
- The local model is `F51-Darwin-X-1.6B-Nitro`; 2.5B, 5B, remote GPUs, and cloud runtime are not local authority.
- Canonical corpus is `workspace/01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast_v2.bin`.
- Canonical checkpoint candidate is `workspace/03_CHECKPOINTS/organism_cycle_071.pt`; cycle 077 remains preserved and non-canonical.
- Never select or promote a checkpoint by filename, cycle number, mtime, or pointer alone.
- Never copy the complete 223+ GiB workspace; use validated same-volume renames.
- Never delete a unique corpus, tokenizer, checkpoint, manifest, or historical artifact.
- Never run `git gc`, `git prune`, history rewriting, `git add -A`, or `git commit --no-verify` during this plan.
- Never start `run247`; the only GPU execution allowed by this plan is the bounded isolated canary after all earlier gates pass.
- All CPU tests set `CUDA_VISIBLE_DEVICES=-1`.
- Every commit stages explicit paths and uses a Conventional Commit message.
- Remote cloud state is ignored as operational evidence.

## Execution Ownership

- **Executor 1 — workspace/runtime/lineage:** Tasks 1–6.
- **Executor 2 — repository/audit controls:** Tasks 7–9.
- **Executor 3 — documentation/verification/GOLD:** Tasks 10–12.

Each executor works after the previous executor's commits are reviewed. The controller independently verifies every report and dispatches a separate review before advancing.

---

### Task 1: Retire unsafe automation and freeze the baseline

**Files:**
- Create: `src/tests/test_repository_governance.py`
- Create: `governance/audit/policy/repository-governance.md`
- Create: `governance/audit/provenance/pre-migration.json`
- Modify: `.gitignore`
- Delete: `src/scripts/_autocommit.ps1`
- Move: `src/scripts/_cleanup_disk.ps1` → `governance/archive/legacy-scratch/src/scripts/_cleanup_disk.ps1`
- Move: `clean_checkpoints.py` → `governance/archive/legacy-scratch/clean_checkpoints.py`

**Interfaces:**
- Consumes: current clean Git HEAD and stopped autocommit PIDs.
- Produces: a tracked baseline manifest and a source tree that cannot silently auto-stage or delete artifacts.

- [ ] **Step 1: Write the governance regression test**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_unsafe_repository_automation_is_not_operational() -> None:
    assert not (ROOT / "src/scripts/_autocommit.ps1").exists()
    operational = [ROOT / "scripts", ROOT / "f51_darwin"]
    forbidden = ("git add -A", "--no-verify", "Remove-Item -Recurse")
    hits = []
    for base in operational:
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".py", ".ps1", ".bat", ".sh"}:
                text = path.read_text(encoding="utf-8", errors="ignore")
                hits.extend((str(path.relative_to(ROOT)), token) for token in forbidden if token in text)
    assert hits == []


def test_heavy_workspace_is_ignored_but_audit_policy_is_tracked() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "workspace/" in ignore
    assert "!checkpoints/organism" not in ignore
    assert (ROOT / "governance/audit/policy/repository-governance.md").is_file()
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.\.venv_nitro\Scripts\python.exe -m pytest -q -p no:cacheprovider src\\tests\\test_repository_governance.py
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

Expected: FAIL because `_autocommit.ps1` exists, `workspace/` is not ignored, and the policy is absent.

- [ ] **Step 3: Record baseline and retire scripts**

`governance/audit/provenance/pre-migration.json` must contain the exact `source_commit`, branch, UTC timestamp, disk free bytes, root file count, workspace source path, workspace bytes, checkpoint/corpus paths, reparse points, and the statement `"destructive_cleanup_executed": false`.

Update `.gitignore` to include:

```gitignore
workspace/
governance/audit/generated/
.audit-src/tools/
```

Remove duplicate `checkpoints_split/` and the exceptions beginning with
`!checkpoints/organism/`. Move the two cleanup scripts with `git mv`; add an
archive README that states they are historical and must not be executed.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the Step 2 command. Expected: all tests in `test_repository_governance.py` pass.

- [ ] **Step 5: Commit explicitly**

```powershell
git add -- .gitignore governance/audit/policy/repository-governance.md governance/audit/provenance/pre-migration.json src/tests/test_repository_governance.py governance/archive/legacy-scratch
git rm -- src/scripts/_autocommit.ps1
git diff --cached --check
git commit -m "chore: retire unsafe repository automation"
```

---

### Task 2: Implement the typed single-root workspace contract

**Files:**
- Modify: `src/f51_darwin/dataset_layout.py`
- Modify: `src/f51_darwin/data_factory.py`
- Modify: `src/tests/test_dataset_layout.py`
- Modify: `src/tests/test_artifact_resolution.py`
- Create: `src/tests/test_workspace_boundary.py`

**Interfaces:**
- Produces: `WorkspacePaths.from_project(project_root, explicit=None, use_environment=False, require=False)`.
- Produces: constants for `feast_v2`, its manifest, the latest pointer, readiness manifest, tokenizer, and runtime directories.
- Consumed by: migration, launcher, organism, Davi, ingestion, inventory, and artifact verification.

- [ ] **Step 1: Replace sibling expectations with workspace expectations**

Add tests equivalent to:

```python
def test_workspace_defaults_inside_project(tmp_path: Path) -> None:
    project = tmp_path / "F51-Darwin-SSD"
    project.mkdir()
    assert resolve_dataset_root(project, use_environment=False) == project / "workspace"


def test_environment_override_requires_opt_in(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    external = tmp_path / "override"
    external.mkdir()
    monkeypatch.setenv(DATASET_ROOT_ENV, str(external))
    assert resolve_dataset_root(project, use_environment=False) == project / "workspace"
    assert resolve_dataset_root(project, use_environment=True, require=True) == external


@pytest.mark.parametrize("unsafe", ["../escape", "C:/escape", "\\\\server\\share"])
def test_dataset_relative_paths_reject_escape(tmp_path: Path, unsafe: str) -> None:
    with pytest.raises(ValueError, match="workspace_boundary"):
        resolve_dataset_path(tmp_path, unsafe)
```

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.\.venv_nitro\Scripts\python.exe -m pytest -q -p no:cacheprovider src\\tests\\test_dataset_layout.py src\\tests\\test_artifact_resolution.py src\\tests\\test_workspace_boundary.py
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

Expected: failures reference the sibling workspace, old feast filename, and missing boundary checks.

- [ ] **Step 3: Implement exact constants and typed paths**

```python
WORKSPACE_DIR_NAME = "workspace"
TOKENIZED_RELATIVE = Path("01_TOKENIZADOS")
FEAST_TOKEN_RELATIVE = TOKENIZED_RELATIVE / "00_CORPUS_PRINCIPAL_tokens_feast_v2.bin"
FEAST_MANIFEST_RELATIVE = Path(f"{FEAST_TOKEN_RELATIVE}.manifest.json")
CHECKPOINTS_RELATIVE = Path("03_CHECKPOINTS")
LATEST_POINTER_RELATIVE = CHECKPOINTS_RELATIVE / "organism_latest.json"
MANIFESTS_RELATIVE = Path("04_MANIFESTOS")
READINESS_MANIFEST_RELATIVE = MANIFESTS_RELATIVE / "overnight_16b_readiness.json"
TOKENIZER_RELATIVE = Path("tokenizer")
RUNTIME_RELATIVE = Path("runtime")
```

Add a frozen `WorkspacePaths` dataclass with `root`, `raw`, `tokenized`, `corpus`, `checkpoints`, `manifests`, `tokenizer`, `runtime`, `runs`, `logs`, `evaluations`, `feast_token_bin`, `feast_manifest`, `latest_pointer`, and `readiness_manifest`.

`resolve_dataset_root` uses `explicit`, then `F51_DATASET_ROOT` only when `use_environment=True`, then `<project_root>/workspace`. `resolve_dataset_path` rejects absolute paths, `..`, and resolved paths outside the workspace.

- [ ] **Step 4: Update DataFactory paths and verify GREEN**

`DataFactoryPaths.from_project()` consumes
`WorkspacePaths.from_project(project_root).corpus`. `from_root()` remains
available only for isolated fixtures.

Run the Step 2 command. Expected: all focused tests pass.

- [ ] **Step 5: Commit**

```powershell
git add -- src/f51_darwin/dataset_layout.py src/f51_darwin/data_factory.py src/tests/test_dataset_layout.py src/tests/test_artifact_resolution.py src/tests/test_workspace_boundary.py
git diff --cached --check
git commit -m "feat: resolve local state under project workspace"
```

---

### Task 3: Add a reversible same-volume migration tool

**Files:**
- Create: `src/f51_darwin/workspace_migration.py`
- Create: `src/scripts/migrate_single_root_workspace.ps1`
- Create: `src/tests/test_single_root_migration.py`
- Create: `governance/audit/schemas/single-root-migration.schema.json`

**Interfaces:**
- Produces: `MigrationPlan`, `build_migration_plan`, `validate_migration_plan`,
  `write_migration_manifest`, `apply_plan`, and `rollback_plan`.
- PowerShell modes: dry-run default, `-Apply`, and `-Rollback -Manifest <path>`.

- [ ] **Step 1: Write migration unit tests**

Tests must prove:

```python
def test_plan_uses_actual_nested_source(tmp_path: Path) -> None:
    project = tmp_path / "F51-Darwin-SSD"
    source = project / "F51-Dataset-Organizado"
    source.mkdir(parents=True)
    plan = build_migration_plan(project, source)
    assert plan.source_root == source.resolve()
    assert plan.target_root == (project / "workspace").resolve()


def test_apply_and_rollback_are_renames_not_copies(tmp_path: Path) -> None:
    project = tmp_path / "F51-Darwin-SSD"
    source = project / "F51-Dataset-Organizado"
    artifact = source / "01_TOKENIZADOS" / "sample.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"lineage-bytes")
    plan = build_migration_plan(project, source)
    manifest = source / "04_MANIFESTOS" / "single_root_migration.json"
    manifest.parent.mkdir(parents=True)
    write_migration_manifest(plan, manifest)

    applied_manifest = apply_plan(plan, manifest)
    assert not source.exists()
    assert (project / "workspace/01_TOKENIZADOS/sample.bin").read_bytes() == b"lineage-bytes"

    rollback_plan(applied_manifest)
    assert not (project / "workspace").exists()
    assert artifact.read_bytes() == b"lineage-bytes"
```

Additional tests cover different volumes, existing target, escaping reparse points, an active-process report, altered manifest, and rollback before runtime use.

- [ ] **Step 2: Run tests and verify RED**

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.\.venv_nitro\Scripts\python.exe -m pytest -q -p no:cacheprovider src\\tests\\test_single_root_migration.py
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

Expected: import failure because the migration module does not exist.

- [ ] **Step 3: Implement fail-closed planning and the PowerShell wrapper**

The actual source is `<project_root>/F51-Dataset-Organizado`, not the absent Desktop sibling. The target is `<project_root>/workspace`.

The tool records HEAD, volume identity, free bytes, file count, total bytes, reparse points, top-level directory sizes, and full SHA-256 for `feast_v2`, its manifest, checkpoints 070/071/077, and the Nitro config. It rejects active `_autocommit`, Darwin, Davi, Ghost, ingest, or launcher processes; a conflicting target; different volumes; or paths outside the project root.

`-Apply` performs directory renames only. It then moves `tokenizer/`, `runs/`, `logs/`, and `eval_results/` into their `workspace/` destinations when non-conflicting. It removes only the validated broken `data` junction after recording its target. It never creates `checkpoints/`.

- [ ] **Step 4: Verify dry-run and rollback on fixtures**

Run the Step 2 command. Then parse the PowerShell file with the PowerShell AST parser and require zero parse errors.

- [ ] **Step 5: Commit**

```powershell
git add -- src/f51_darwin/workspace_migration.py src/scripts/migrate_single_root_workspace.ps1 src/tests/test_single_root_migration.py governance/audit/schemas/single-root-migration.schema.json
git diff --cached --check
git commit -m "feat: add reversible single-root migration"
```

---

### Task 4: Apply the physical single-root migration

**Files:**
- Modify externally/ignored: `F51-Dataset-Organizado/` → `workspace/`
- Create ignored: `workspace/04_MANIFESTOS/single_root_migration.json`
- Remove: broken `data/` junction only

**Interfaces:**
- Consumes: Task 3 tool and a clean worktree.
- Produces: one physical root with unchanged artifact bytes and a tested rollback manifest.

- [ ] **Step 1: Reconfirm the exact targets**

```powershell
git status --short
Get-Item -LiteralPath 'C:\Users\marco\Desktop\F51-Darwin-SSD\F51-Dataset-Organizado'
Get-Item -LiteralPath 'C:\Users\marco\Desktop\F51-Darwin-SSD\data' -Force
Test-Path -LiteralPath 'C:\Users\marco\Desktop\F51-Darwin-SSD\workspace'
```

Expected: clean Git; source is a normal directory inside the project; `data` is the recorded broken junction; target does not exist.

- [ ] **Step 2: Run dry-run**

```powershell
& 'C:\Program Files\PowerShell\7\pwsh.exe' -NoProfile -ExecutionPolicy Bypass `
  -File src\\scripts\\migrate_single_root_workspace.ps1 `
  -ProjectRoot 'C:\Users\marco\Desktop\F51-Darwin-SSD'
```

Expected: same-volume plan, zero active conflicting processes, canonical hashes, exact renames, and rollback command; no filesystem change.

- [ ] **Step 3: Apply and validate byte identity**

Run the same command with `-Apply`. Recompute the recorded artifact hashes and compare every value with the pre-migration manifest. Verify old source and `data/` are absent, `workspace/` is present, and the full data remains ignored by Git.

- [ ] **Step 4: Exercise rollback on a fixture and preserve the real workspace**

Do not roll the real workspace back merely for demonstration. The automated fixture must already prove rollback; validate that the real manifest contains an executable rollback command and has not been altered.

- [ ] **Step 5: Commit only the redacted migration evidence**

Create `governance/audit/provenance/single-root-migration-summary.json` without raw data paths outside the project or large artifact content, stage it explicitly, and commit:

```powershell
git add -- governance/audit/provenance/single-root-migration-summary.json
git commit -m "chore: record single-root migration evidence"
```

---

### Task 5: Verify corpus and checkpoint lineage, then publish the pointer

**Files:**
- Create: `src/f51_darwin/artifact_manifest.py`
- Create: `src/scripts/verify_gold_lineage.py`
- Create: `src/tests/test_gold_lineage.py`
- Create: `governance/audit/schemas/gold-lineage.schema.json`
- Create ignored: `workspace/04_MANIFESTOS/gold_local_lineage.json`
- Create ignored: `workspace/03_CHECKPOINTS/organism_latest.json`

**Interfaces:**
- Produces: streaming `sha256_file`, corpus verifier, checkpoint report validator, and atomic pointer publication.
- Consumes: existing `inspect_organism_checkpoint.py --verify-identity` JSON.

- [ ] **Step 1: Write RED tests for every lineage invariant**

Fixtures cover wrong corpus size/hash/token count/dtype/tokenizer/composition; checkpoint version/config/cycle/step/shapes/topology/AdamW/identity/base ID; cycle 077 non-promotion; and atomic pointer publication.

```python
def test_cycle_number_alone_never_selects_checkpoint(tmp_path: Path) -> None:
    reports = [verified_report(cycle=71, identity_ok=True), verified_report(cycle=77, identity_ok=False)]
    assert select_canonical_checkpoint(reports) == reports[0]
```

- [ ] **Step 2: Run and verify RED**

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.\.venv_nitro\Scripts\python.exe -m pytest -q -p no:cacheprovider src\\tests\\test_gold_lineage.py
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

- [ ] **Step 3: Implement streaming verification**

Require corpus bytes `74195890756`, tokens `18548972689`, int32 little-endian, and SHA-256 `9677e9f22f4d78efa7b25c77b2da3cbdb5fb2a499926ac641a75c13b65e25cff`.

Verify checkpoint 070 first and use its independently verified lineage to corroborate the expected `base_checkpoint_id` for 071. Then verify 071 as v7, cycle 71, step 40751, exact config, shapes, Topology Manifest, AdamW, integral identity, file hash, and base ID. Report 077 separately and never publish it automatically.

- [ ] **Step 4: Publish the pointer atomically only on full success**

Pointer payload includes version, checkpoint version, filename, cycle, step, size, SHA-256, base ID, config SHA-256, source commit, and UTC publication time. Write `.tmp`, fsync where supported, and `os.replace`.

- [ ] **Step 5: Run the real CPU/disk verifier**

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.\.venv_nitro\Scripts\python.exe src\\scripts\\verify_gold_lineage.py --project-root . --publish-pointer
$code=$LASTEXITCODE
Remove-Item Env:CUDA_VISIBLE_DEVICES
exit $code
```

Expected: exit 0; full hashes match; cycle 071 strict resume and identity pass; pointer health is `ok`; 077 remains non-canonical.

- [ ] **Step 6: Commit source and redacted evidence**

```powershell
git add -- src/f51_darwin/artifact_manifest.py src/scripts/verify_gold_lineage.py src/tests/test_gold_lineage.py governance/audit/schemas/gold-lineage.schema.json governance/audit/provenance/gold-lineage-summary.json
git diff --cached --check
git commit -m "feat: verify and publish canonical local lineage"
```

---

### Task 6: Route all supported local operations through workspace

**Files:**
- Modify: `src/scripts/start_overnight_16b.ps1`
- Modify: `src/scripts/darwin_organism.py`
- Modify: `src/scripts/serve_davi.py`
- Modify: `src/scripts/ingest_pipeline.py`
- Modify: `src/scripts/darwin_inventory.py`
- Modify: `src/f51_darwin/artifacts.py`
- Modify: `src/tests/test_canary_launcher_contract.py`
- Modify: `src/tests/test_realtime_ingest_flow.py`
- Modify: `src/tests/test_serve_davi.py`
- Modify: `src/tests/test_artifact_resolution.py`

**Interfaces:**
- Consumes: `WorkspacePaths` and the verified pointer/lineage manifest.
- Produces: default local dry-run/canary, loopback Davi, governed ingestion, and inventory under one root.

- [ ] **Step 1: Write launcher and runtime contract failures**

Tests require: empty `Root` derives from the script parent; empty `DatasetRoot` becomes `$Root\workspace`; corpus is `feast_v2`; readiness is atomic and commit-bound; runtime outputs use `workspace/runtime`; no sibling, feast v1, cloud fallback, or automatic run247 exists; ASPM is restored in a `finally` path after a launched canary.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.\.venv_nitro\Scripts\python.exe -m pytest -q -p no:cacheprovider src\\tests\\test_canary_launcher_contract.py src\\tests\\test_realtime_ingest_flow.py src\\tests\\test_serve_davi.py src\\tests\\test_artifact_resolution.py
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

- [ ] **Step 3: Implement the single-root routing**

Use `WorkspacePaths` in Python. In PowerShell use empty parameters followed by:

```powershell
if (-not $Root) { $Root = Split-Path -Parent $PSScriptRoot }
$Root = (Resolve-Path -LiteralPath $Root).Path
if (-not $DatasetRoot) { $DatasetRoot = Join-Path $Root 'workspace' }
$DatasetRoot = (Resolve-Path -LiteralPath $DatasetRoot).Path
```

Readiness includes exact source commit, clean tracked worktree, config/corpus/tokenizer/checkpoint identities, GPU inventory, disk headroom, and override provenance. JSON writes use `Write-JsonAtomic`. `-Launch` is rejected unless all identities match the same current readiness inputs.

- [ ] **Step 4: Verify GREEN and static parsing**

Run Step 2, Python compilation for modified files, and PowerShell AST parsing. Expected: zero failures/errors.

- [ ] **Step 5: Commit**

```powershell
git add -- src/scripts/start_overnight_16b.ps1 src/scripts/darwin_organism.py src/scripts/serve_davi.py src/scripts/ingest_pipeline.py src/scripts/darwin_inventory.py src/f51_darwin/artifacts.py src/tests/test_canary_launcher_contract.py src/tests/test_realtime_ingest_flow.py src/tests/test_serve_davi.py src/tests/test_artifact_resolution.py
git diff --cached --check
git commit -m "fix: route local operations through single workspace"
```

---

### Task 7: Reduce and classify the operational surface

**Files:**
- Create: `governance/audit/policy/operational-surface.json`
- Create: `src/tests/test_operational_surface.py`
- Create: `governance/archive/README.md`
- Move: `src/scripts/legacy_scratch/` → `governance/archive/legacy-scratch/src/scripts/`
- Move: cloud/rental/sync/pull/deploy scripts and configs → `governance/archive/cloud-local/`
- Move: superseded root roadmaps/reports/launchers → `governance/archive/historical-governance/docs/`
- Remove: tracked installer binaries under `installers/`
- Create: `governance/audit/provenance/retired-installers.json`

**Interfaces:**
- Produces: a machine-readable classification of every tracked executable and root artifact.
- Supported entrypoints: exactly the six listed in the approved design.

- [ ] **Step 1: Write RED classification tests**

The test loads `operational-surface.json`, requires every tracked executable under `src/scripts/` to be classified, requires the six supported entrypoints, rejects `cloud`, `rental`, `sync`, `pull`, or `2.5b/5b` as supported, and rejects executable scratch files in the root.

- [ ] **Step 2: Generate hashes and move with Git history preservation**

Use `git mv` for unique historical material. Remove exact duplicates only after SHA-256 equality is recorded. Remove the two installer binaries and record filename, prior Git blob, SHA-256, signer, version, and official reacquisition instructions. Do not prune history.

- [ ] **Step 3: Prove collection/import health**

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.\.venv_nitro\Scripts\python.exe -m pytest --collect-only -q -p no:cacheprovider
.\.venv_nitro\Scripts\python.exe -m pytest -q -p no:cacheprovider src\\tests\\test_operational_surface.py src\\tests\\test_repository_governance.py
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

Expected: collection succeeds and classification tests pass.

- [ ] **Step 4: Commit**

```powershell
git add -- governance/audit/policy/operational-surface.json governance/audit/provenance/retired-installers.json archive src/tests/test_operational_surface.py
git rm -- installers/Git-2.55.0.2-64-bit.exe installers/PowerShell-7.6.3-win-x64.msi
git diff --cached --check
git commit -m "refactor: reduce Darwin operational surface"
```

---

### Task 8: Add legal, security, dependency, SBOM, and CI controls

**Files:**
- Create: `LICENSE`
- Create: `NOTICE`
- Create: `THIRD_PARTY_NOTICES.md`
- Create: `SECURITY.md`
- Modify: `pyproject.toml`
- Create: `requirements.in`
- Create: `requirements-dev.in`
- Create: `requirements.lock`
- Create: `requirements-dev.lock`
- Create: `governance/audit/policy/dependencies.json`
- Create: `governance/audit/sbom/cyclonedx.json`
- Create: `governance/audit/generated/.gitkeep`
- Create: `src/scripts/check_dependency_policy.py`
- Create: `.github/workflows/ci.yml`
- Create: `src/tests/test_audit_controls.py`

**Interfaces:**
- Produces: proprietary rights notice, support/security policy, hash-locked dependencies, SBOM, vulnerability/license policy, and pinned CI.

- [ ] **Step 1: Write audit control tests**

Tests require all legal/security files; Python 3.12 support; exact `==` direct requirements; `--hash=` entries in locks; CycloneDX schema metadata; CI commands for CPU pytest, py_compile, PowerShell parse, docs, dependency policy, secret scan, and diff checks; and actions pinned to 40-character commit SHAs.

- [ ] **Step 2: Create a clean audit-tools environment and locks**

```powershell
py -3.12 -m venv .audit-tools
.\.audit-src\\tools\\Scripts\python.exe -m pip install --upgrade pip
.\.audit-src\\tools\\Scripts\python.exe -m pip install pip-tools pip-audit cyclonedx-bom pip-licenses
.\.audit-src\\tools\\Scripts\pip-compile.exe --generate-hashes requirements.in --output-file requirements.lock
.\.audit-src\\tools\\Scripts\pip-compile.exe --generate-hashes requirements-dev.in --output-file requirements-dev.lock
```

Build direct inputs from imports and the supported runtime, not from cloud requirements. Validate installability in a clean temporary venv with `--require-hashes` and run `pip check`.

- [ ] **Step 3: Generate and evaluate SBOM/CVE/licenses**

Generate CycloneDX JSON and dated raw reports under ignored `governance/audit/generated/`; commit deterministic summaries under `governance/audit/sbom/` and `THIRD_PARTY_NOTICES.md`. High/Critical vulnerabilities block unless a time-bounded exception records CVE, owner, reason, mitigation, and expiry.

- [ ] **Step 4: Add pinned CI and secret scanning**

Use Windows CI with these verified full-SHA pins:

```yaml
- uses: actions/checkout@08eba0b27e820071cde6df949e0beb9ba4906955 # v4.3.0
- uses: actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405 # v6.2.0
- uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02 # v4
```

Secret scanning covers current tree and Git history; a real secret requires
rotation before any separate history-rewrite decision.

- [ ] **Step 5: Run focused verification and commit**

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.\.venv_nitro\Scripts\python.exe -m pytest -q -p no:cacheprovider src\\tests\\test_audit_controls.py
.\.venv_nitro\Scripts\python.exe src\\scripts\\check_dependency_policy.py
Remove-Item Env:CUDA_VISIBLE_DEVICES
git add -- LICENSE NOTICE THIRD_PARTY_NOTICES.md SECURITY.md pyproject.toml requirements.in requirements-dev.in requirements.lock requirements-dev.lock governance/audit/policy/dependencies.json governance/audit/sbom/cyclonedx.json governance/audit/generated/.gitkeep src/scripts/check_dependency_policy.py .github/workflows/ci.yml src/tests/test_audit_controls.py
git diff --cached --check
git commit -m "chore: add reproducible audit controls"
```

---

### Task 9: Add audit harnesses and local evidence schemas

**Files:**
- Create: `src/scripts/check_canonical_docs.py`
- Create: `src/scripts/check_docs_links.py`
- Create: `src/scripts/check_operational_surface.py`
- Create: `src/scripts/run_gold_source_audit.ps1`
- Create: `governance/audit/schemas/gold-source-report.schema.json`
- Create: `governance/audit/test-reports/README.md`
- Modify: `src/tests/test_audit_controls.py`

**Interfaces:**
- Produces: one source-level audit command with machine-readable JSON output under ignored `workspace/runtime/evaluations/`.

- [ ] **Step 1: Write RED tests for audit harness behavior**

Tests require nonzero exit on broken Markdown links, stale canonical claims, unclassified executable, failed tests, PowerShell parse errors, dirty tracked worktree, or source/report commit mismatch.

- [ ] **Step 2: Implement deterministic checks**

`run_gold_source_audit.ps1` sets CPU-only mode, runs full pytest, Python compilation, PowerShell AST parsing, docs link/canonical checks, dependency policy, operational surface, secret scan, `git diff --check`, and worktree status. It records command, tool versions, start/end UTC, exit codes, counts, and source commit atomically.

- [ ] **Step 3: Verify and commit**

Run focused tests and an expected-failing audit if documentation is still stale. Commit the harness only after its own focused tests pass.

```powershell
git add -- src/scripts/check_canonical_docs.py src/scripts/check_docs_links.py src/scripts/check_operational_surface.py src/scripts/run_gold_source_audit.ps1 governance/audit/schemas/gold-source-report.schema.json governance/audit/test-reports/README.md src/tests/test_audit_controls.py
git commit -m "test: add GOLD source audit harness"
```

---

### Task 10: Reconcile canonical documentation

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `CLAUDE.md`
- Modify: `governance/docs/CANONICAL_MAP.md`
- Modify: `governance/docs/operacao/STATUS_ATUAL.md`
- Create: `governance/docs/operacao/OPERACAO_SEGURA.md`
- Create: `governance/docs/arquitetura/IMPLEMENTACAO_ATUAL.md`
- Create/Modify: `governance/docs/_historico/INDEX.md`
- Create: `governance/audit/README.md`
- Create: `src/tests/test_canonical_docs.py`

**Interfaces:**
- Produces: one consistent local truth and an indexed historical archive.

- [ ] **Step 1: Write canonical document tests**

Tests require `workspace`, `feast_v2`, cycle 071, six supported entrypoints, rollback, proof hierarchy, and local-only readiness. They reject cycle 68, feast v1, sibling dataset, active 2.5B/5B, remote GPU, or cloud runtime in canonical files. `AGENTS.md` and `CLAUDE.md` must be byte-identical.

- [ ] **Step 2: Rewrite documents from fresh evidence**

`STATUS_ATUAL.md` uses the final verified hashes, pointer, process/GPU snapshot, test counts, and current blockers. Historical documents retain original content under `governance/archive/`/`governance/docs/_historico` and are indexed as non-authoritative.

- [ ] **Step 3: Verify links, claims, and commit**

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.\.venv_nitro\Scripts\python.exe -m pytest -q -p no:cacheprovider src\\tests\\test_canonical_docs.py
.\.venv_nitro\Scripts\python.exe src\\scripts\\check_docs_links.py
.\.venv_nitro\Scripts\python.exe src\\scripts\\check_canonical_docs.py
Remove-Item Env:CUDA_VISIBLE_DEVICES
git add -- README.md AGENTS.md CLAUDE.md docs governance/audit/README.md src/tests/test_canonical_docs.py
git diff --cached --check
git commit -m "docs: publish single-root canonical truth"
```

---

### Task 11: Run source gates, readiness dry-run, and bounded hardware canary

**Files:**
- Create ignored: `workspace/runtime/evaluations/<run-id>/`
- Create tracked after source freeze: `governance/audit/test-reports/<source-commit>-source-summary.json`

**Interfaces:**
- Consumes: clean source commit and verified workspace identities.
- Produces: source report, readiness manifest, isolated canary report, candidate checkpoint, and external comparison evidence without pointer promotion.

- [ ] **Step 1: Run the complete source audit from a clean commit**

```powershell
& 'C:\Program Files\PowerShell\7\pwsh.exe' -NoProfile -ExecutionPolicy Bypass -File src\\scripts\\run_gold_source_audit.ps1
```

Expected: exit 0, zero pytest failures, zero static/policy/link/secret findings, clean tracked worktree, report source commit equal to HEAD.

- [ ] **Step 2: Run official readiness without launch**

```powershell
& 'C:\Program Files\PowerShell\7\pwsh.exe' -NoProfile -ExecutionPolicy Bypass -File src\\scripts\\start_overnight_16b.ps1 -Canary
```

Expected: exit 0; no new Darwin process; readiness atomically written with current commit and exact artifact identities; two local GPUs and disk headroom recorded.

- [ ] **Step 3: Run the bounded isolated canary**

The user has approved all in-scope execution. Reconfirm zero conflicting Darwin processes, clean worktree, and matching readiness inputs, then run:

```powershell
& 'C:\Program Files\PowerShell\7\pwsh.exe' -NoProfile -ExecutionPolicy Bypass -File src\\scripts\\start_overnight_16b.ps1 -Canary -Launch
```

Expected: exactly one process, both GPUs used, 250 bounded steps, finite fresh/replay/heldout metrics, no NVIDIA 13/14/153 or LiveKernelEvent 141, isolated candidate, strict resume/identity pass, ASPM restored, canonical pointer untouched.

- [ ] **Step 4: Run frozen baseline/candidate comparison**

Use the existing comparison/finalization flow against the isolated candidate. Require heldout and replay non-regression within the documented `+0.02` bound. Result may be `stable_canary`; it must never claim `model_improved` from this gate.

- [ ] **Step 5: Commit only the source-bound summaries**

Copy redacted deterministic summaries into `governance/audit/test-reports/`, stage explicitly, and commit:

```powershell
git add -- governance/audit/test-reports
git commit -m "test: record single-root GOLD verification"
```

---

### Task 12: Perform an external-audit simulation and publish GOLD status

**Files:**
- Create: `governance/audit/GOLD_REPORT_2026-07-16.md`
- Create: `governance/audit/provenance/source-manifest.json`
- Update: `governance/audit/README.md`
- Update: `governance/docs/operacao/STATUS_ATUAL.md`
- Create: `.agent_bus/2026-07-16-single-root-gold.md` only if the directory policy permits ignored operational notes.

**Interfaces:**
- Consumes: all task commits and Task 11 evidence.
- Produces: requirement-by-requirement audit verdict tied to a frozen source commit and a later evidence commit.

- [ ] **Step 1: Freeze the auditable source commit**

Record HEAD as `gold_source_commit`. Do not make the report claim its own commit hash. The report commit points backward to the frozen source commit.

- [ ] **Step 2: Dispatch a fresh adversarial reviewer**

The reviewer receives the approved design, this plan, the source commit, source report, readiness, lineage manifest, canary/finalization reports, SBOM/CVE/license summaries, and current tree. It must map every design requirement to direct evidence and report P0/P1/P2 findings.

- [ ] **Step 3: Fix every P0/P1 and re-run affected gates**

One fix executor handles the complete finding list, runs covering tests, and receives a re-review. GOLD is forbidden while any P0/P1 remains or evidence is missing/indirect.

- [ ] **Step 4: Write and verify the final report**

The report includes root topology, artifact identities, lineage, test counts, CI/local separation, security/supply chain results, operational dry-run, canary, rollback, known P2 items, exact reproduction commands, and a requirement evidence table.

Run the complete source audit again and validate all JSON against schemas. Verify `git diff --check` and a clean staged scope.

- [ ] **Step 5: Commit the evidence report**

```powershell
git add -- governance/audit/GOLD_REPORT_2026-07-16.md governance/audit/provenance/source-manifest.json governance/audit/README.md governance/docs/operacao/STATUS_ATUAL.md
git diff --cached --check
git commit -m "docs: publish Darwin single-root GOLD audit"
```

After the evidence commit, rerun lightweight source integrity checks and record both `gold_source_commit` and `gold_evidence_commit`. Do not start `run247`.
