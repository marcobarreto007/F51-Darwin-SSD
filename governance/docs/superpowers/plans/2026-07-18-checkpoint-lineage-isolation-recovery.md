# Checkpoint Lineage Isolation and Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent cross-lineage checkpoint replacement, isolate the 100M lineage non-destructively, and record/search for the missing 1.6B gold by its exact identity.

**Architecture:** A focused checkpoint-root policy module validates empty fresh-start roots, persistent lineage identities, resume compatibility, and save targets. The CLI runs the policy before GPU allocation, while the checkpoint writer repeats the collision check at the mutation boundary. A non-destructive containment command copies the verified 100M cycle 1 into its isolated root, and a recovery command records exact gold-search evidence.

**Tech Stack:** Python 3.12, PyTorch checkpoint metadata with `mmap=True`, YAML, atomic JSON publication, pytest, PowerShell operational gate.

## Global Constraints

- Do not delete, move, rename, or overwrite any existing checkpoint.
- Do not rewrite `workspace/03_CHECKPOINTS/organism_latest.json`.
- Do not launch `run247`.
- Preserve the contaminated root as evidence.
- Require exact SHA-256, size, topology, cycle, and step before declaring gold recovered.
- Keep unrelated changes in `src/configs/darwin_x_100m.yaml` and `src/tools/gradient_diagnostics.py` out of scoped commits unless a touched line is required by this plan.

---

### Task 1: Checkpoint-root policy

**Files:**
- Create: `src/f51_darwin/organism/checkpoint_root.py`
- Create: `src/tests/test_checkpoint_root_policy.py`

**Interfaces:**
- Produces: `normalized_model_config(raw: Mapping[str, object]) -> dict[str, object]`
- Produces: `model_config_identity(raw: Mapping[str, object]) -> str`
- Produces: `checkpoint_metadata(path: Path) -> dict[str, object]`
- Produces: `preflight_checkpoint_root(...) -> dict[str, object]`
- Produces: `assert_new_checkpoint_target(root: Path, cycle: int) -> Path`
- Produces: `write_lineage_root_identity(root: Path, identity: Mapping[str, object]) -> Path`

- [ ] **Step 1: Write failing policy tests**

```python
def test_fresh_start_rejects_non_empty_root(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    (root / "organism_cycle_071.pt").write_bytes(b"gold")
    with pytest.raises(ValueError, match="fresh-start checkpoint root must be empty"):
        preflight_checkpoint_root(
            root=root,
            model_raw={"model_name": "F51-Darwin-X-100M"},
            mode="fresh_start",
            resume=None,
        )


def test_existing_cycle_target_is_never_replaced(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    target = root / "organism_cycle_001.pt"
    target.write_bytes(b"scientific evidence")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        assert_new_checkpoint_target(root, 1)
    assert target.read_bytes() == b"scientific evidence"
```

- [ ] **Step 2: Run tests and confirm missing-module failure**

Run:

```powershell
.\.venv_nitro\Scripts\python.exe -m pytest src\\tests\\test_checkpoint_root_policy.py -q
```

Expected: collection fails because `f51_darwin.organism.checkpoint_root` does not exist.

- [ ] **Step 3: Implement deterministic identities and fail-closed guards**

```python
LINEAGE_ROOT_FILENAME = "lineage_root.json"


def model_config_identity(raw):
    normalized = normalized_model_config(raw)
    encoded = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "darwin-config-v1:" + hashlib.sha256(encoded).hexdigest()


def assert_new_checkpoint_target(root: Path, cycle: int) -> Path:
    target = root / f"organism_cycle_{cycle:03d}.pt"
    temporary = target.with_suffix(target.suffix + ".tmp")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {target}")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite incomplete checkpoint: {temporary}")
    return target
```

`preflight_checkpoint_root` must reject all organism checkpoints, temporary
checkpoints, pointers, or lineage identities for `fresh_start`. For `resume`,
it must require `lineage_root.json`, inspect the resume checkpoint with
`torch.load(..., mmap=True, map_location="cpu", weights_only=False)`, and match
`model_name`, normalized config identity, tokenizer ID, and source
`base_checkpoint_id`.

- [ ] **Step 4: Run focused policy tests**

Run:

```powershell
.\.venv_nitro\Scripts\python.exe -m pytest src\\tests\\test_checkpoint_root_policy.py -q
```

Expected: all policy tests pass.

- [ ] **Step 5: Commit policy**

```powershell
git add -- src/f51_darwin/organism/checkpoint_root.py src/tests/test_checkpoint_root_policy.py
git commit -m "fix(checkpoints): enforce lineage root identity"
```

### Task 2: CLI and save-boundary enforcement

**Files:**
- Modify: `src/f51_darwin/organism/cli.py`
- Modify: `src/f51_darwin/organism/checkpoint_mixin.py`
- Modify: `src/tests/test_organism_causal_runtime.py`
- Modify: `src/tests/test_checkpoint_root_policy.py`

**Interfaces:**
- Consumes: Task 1 policy functions.
- Produces: CLI preflight before `DarwinOrganism` construction and save-time no-overwrite behavior.

- [ ] **Step 1: Add failing CLI and asynchronous-save tests**

```python
def test_build_organism_rejects_100m_fresh_start_in_shared_root(tmp_path):
    shared = tmp_path / "workspace" / "03_CHECKPOINTS"
    shared.mkdir(parents=True)
    (shared / "organism_cycle_071.pt").write_bytes(b"gold")
    with pytest.raises(ValueError, match="fresh-start checkpoint root must be empty"):
        preflight_checkpoint_root(
            root=shared,
            model_raw=valid_100m_mapping(),
            mode="fresh_start",
            resume=None,
        )


def test_save_cycle_preserves_existing_target(tmp_path):
    organism = make_tiny_organism(tmp_path)
    target = tmp_path / "checkpoints" / "organism_cycle_001.pt"
    target.parent.mkdir()
    target.write_bytes(b"original")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        organism._save_cycle({"cycle": 1})
    assert target.read_bytes() == b"original"
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```powershell
.\.venv_nitro\Scripts\python.exe -m pytest src\\tests\\test_checkpoint_root_policy.py src\\tests\\test_organism_causal_runtime.py -q
```

Expected: new collision assertions fail before integration.

- [ ] **Step 3: Integrate preflight and save guard**

In `cli.py`, resolve the YAML root, determine `fresh_start` or `resume`, and
call `preflight_checkpoint_root` before constructing `DarwinOrganism`.

In `_save_cycle`, replace:

```python
path = ckpt_dir / f"organism_cycle_{self.cycle:03d}.pt"
```

with:

```python
path = assert_new_checkpoint_target(ckpt_dir, self.cycle)
validate_payload_against_lineage_root(
    ckpt_dir,
    model_config=self.model_config,
    tokenizer_id=getattr(self, "tokenizer_id", ""),
    base_checkpoint_id=base_checkpoint_id,
)
```

The root check must run again inside `_write_checkpoint_async` immediately
before `_write_checkpoint_file` so a concurrent writer cannot create the
target between payload assembly and disk mutation.

- [ ] **Step 4: Run focused tests**

Run:

```powershell
.\.venv_nitro\Scripts\python.exe -m pytest src\\tests\\test_checkpoint_root_policy.py src\\tests\\test_organism_causal_runtime.py -q
```

Expected: all focused tests pass and existing atomic-publication tests remain green.

- [ ] **Step 5: Commit integration**

```powershell
git add -- src/f51_darwin/organism/cli.py src/f51_darwin/organism/checkpoint_mixin.py src/tests/test_checkpoint_root_policy.py src/tests/test_organism_causal_runtime.py
git commit -m "fix(checkpoints): block cross-lineage replacement"
```

### Task 3: Non-destructive 100M containment

**Files:**
- Modify: `src/configs/darwin_x_100m.yaml`
- Create: `src/scripts/isolate_100m_checkpoint.py`
- Create: `src/tests/test_isolate_100m_checkpoint.py`

**Interfaces:**
- Consumes: Task 1 metadata and lineage-root helpers.
- Produces: verified copy, isolated pointer, and `lineage_root.json`.

- [ ] **Step 1: Add containment tests**

```python
def test_isolation_copies_without_mutating_source(tmp_path):
    source = make_checkpoint(tmp_path / "shared" / "organism_cycle_001.pt")
    source_hash = sha256_file(source)
    result = isolate_checkpoint(source, tmp_path / "isolated", config_path)
    assert source.exists()
    assert sha256_file(source) == source_hash
    assert sha256_file(result.checkpoint) == source_hash
    assert result.pointer.parent == tmp_path / "isolated"


def test_isolation_refuses_existing_destination(tmp_path):
    destination = tmp_path / "isolated"
    destination.mkdir()
    (destination / "organism_cycle_001.pt").write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        isolate_checkpoint(source, destination, config_path)
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
.\.venv_nitro\Scripts\python.exe -m pytest src\\tests\\test_isolate_100m_checkpoint.py -q
```

Expected: import failure because the containment script is not implemented.

- [ ] **Step 3: Implement atomic verified copy**

`isolate_checkpoint` must:

1. inspect source metadata and validate against the supplied 100M config;
2. stream-copy to `organism_cycle_001.pt.copying`;
3. compare byte size and SHA-256;
4. atomically rename to the final path only when the destination does not exist;
5. write the isolated pointer atomically;
6. write `lineage_root.json`;
7. leave the source and shared pointer untouched.

Change only the required YAML line:

```yaml
checkpoint_root: "workspace/03_CHECKPOINTS_100M"
```

- [ ] **Step 4: Run containment tests**

Run:

```powershell
.\.venv_nitro\Scripts\python.exe -m pytest src\\tests\\test_isolate_100m_checkpoint.py src\\tests\\test_checkpoint_root_policy.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Execute containment**

Run:

```powershell
$env:PYTHONPATH='.'
.\.venv_nitro\Scripts\python.exe src\\scripts\\isolate_100m_checkpoint.py `
  workspace\03_CHECKPOINTS\organism_cycle_001.pt `
  workspace\03_CHECKPOINTS_100M `
  --config src\\configs\\darwin_x_100m.yaml
```

Expected: source and destination hashes match; isolated inspector status is
strict-resume compatible; shared pointer hash and mtime remain unchanged.

- [ ] **Step 6: Commit containment code and config**

```powershell
git add -- src/configs/darwin_x_100m.yaml src/scripts/isolate_100m_checkpoint.py src/tests/test_isolate_100m_checkpoint.py
git commit -m "fix(checkpoints): isolate the 100m lineage"
```

### Task 4: Gold recovery evidence and administrator VSS probe

**Files:**
- Create: `src/scripts/find_gold_checkpoint.py`
- Create: `src/scripts/find_gold_in_vss.ps1`
- Create: `src/tests/test_find_gold_checkpoint.py`
- Modify: `governance/docs/operacao/STATUS_ATUAL.md`
- Runtime output: `workspace/04_MANIFESTOS/gold_recovery_status.json`

**Interfaces:**
- Produces: exact-identity local search report.
- Produces: read-only elevated VSS candidate enumeration.

- [ ] **Step 1: Add recovery qualification tests**

```python
def test_candidate_requires_exact_size_and_sha(tmp_path):
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"wrong")
    result = qualify_candidate(
        candidate,
        expected_size=10_876_850_383,
        expected_sha256="239fdcf175ac35d9402d664e2a2b40252ec9c73aea8ee458250d426adde9241b",
    )
    assert result.status == "rejected"
    assert "size_mismatch" in result.reasons


def test_missing_gold_is_not_reported_as_recovered(tmp_path):
    report = search_gold([], expected_gold())
    assert report["status"] == "missing"
    assert report["recovered_path"] is None
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
.\.venv_nitro\Scripts\python.exe -m pytest src\\tests\\test_find_gold_checkpoint.py -q
```

Expected: import failure because recovery helpers do not exist.

- [ ] **Step 3: Implement exact qualification and read-only VSS probe**

The Python search filters by exact size, computes SHA-256, then invokes the
official inspector with the archived 1.6B config. It writes the recovery
manifest atomically and never copies candidates automatically.

The PowerShell probe must begin with:

```powershell
#Requires -RunAsAdministrator
$expectedRelative = 'Users\marco\Desktop\F51-Darwin-SSD\workspace\03_CHECKPOINTS\organism_cycle_071.pt'
Get-CimInstance Win32_ShadowCopy | ForEach-Object {
    $candidate = Join-Path $_.DeviceObject $expectedRelative
    if (Test-Path -LiteralPath $candidate) {
        Get-Item -LiteralPath $candidate
    }
}
```

- [ ] **Step 4: Run local recovery search**

Run:

```powershell
$env:PYTHONPATH='.'
.\.venv_nitro\Scripts\python.exe src\\scripts\\find_gold_checkpoint.py `
  --manifest workspace\04_MANIFESTOS\gold_local_lineage.json `
  --output workspace\04_MANIFESTOS\gold_recovery_status.json
```

Expected: `status=missing` unless an exact verified candidate is found.

- [ ] **Step 5: Update current status honestly**

Document the contaminated shared root, isolated 100M copy, absent accessible
gold, exact expected SHA, and administrator-only VSS next step. Remove stale
claims that the current `organism_cycle_071.pt` is the intact 1.6B gold.

- [ ] **Step 6: Commit recovery tooling and status**

```powershell
git add -- src/scripts/find_gold_checkpoint.py src/scripts/find_gold_in_vss.ps1 src/tests/test_find_gold_checkpoint.py governance/docs/operacao/STATUS_ATUAL.md
git commit -m "fix(recovery): record missing gold checkpoint evidence"
```

### Task 5: Verification

**Files:**
- No implementation files unless a test reveals a scoped defect.

**Interfaces:**
- Consumes all prior tasks.
- Produces test, artifact, Git, and no-launch evidence.

- [ ] **Step 1: Run focused suite**

```powershell
.\.venv_nitro\Scripts\python.exe -m pytest `
  src\\tests\\test_checkpoint_root_policy.py `
  src\\tests\\test_isolate_100m_checkpoint.py `
  src\\tests\\test_find_gold_checkpoint.py `
  src\\tests\\test_organism_causal_runtime.py -q
```

- [ ] **Step 2: Inspect copied checkpoint**

```powershell
$env:PYTHONPATH='.'
.\.venv_nitro\Scripts\python.exe src\\scripts\\inspect_organism_checkpoint.py `
  workspace\03_CHECKPOINTS_100M\organism_cycle_001.pt `
  --config src\\configs\\darwin_x_100m.yaml --verify-identity
```

Expected: `strict_resume_compatible=true` and `identity_verified=true`.

- [ ] **Step 3: Verify source preservation**

Compare source and destination SHA-256 and verify the original pointer's bytes
match the pre-containment hash recorded by the containment command.

- [ ] **Step 4: Run supported no-launch canary**

```powershell
powershell -ExecutionPolicy Bypass -File src\\scripts\\start_overnight_16b.ps1 -Canary
```

Expected: source audit and operational preflight complete without launching
`run247`.

- [ ] **Step 5: Verify no training process and scoped Git state**

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'darwin_organism|run247' }
git status --short
git log --oneline -6
```

Expected: no Darwin training worker. Only explicitly preserved unrelated user
changes may remain.

