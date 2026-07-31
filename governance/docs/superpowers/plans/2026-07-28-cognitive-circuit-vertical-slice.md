# Cognitive Circuit Vertical Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver and test one complete causal-circuit flow from ablation through immutable package, gate-zero installation, activation and rollback, then prove that the real Darwin JEPA donor and published SSD recipient pass non-launching preflight.

**Architecture:** Add a focused `f51_darwin.circuits` package and three thin CLIs. The first vertical slice supports named modules and residual channel groups, uses deterministic paired interventions, packages weights with a canonical manifest and SHA-256 chain, and publishes only immutable checkpoints through an atomic compare-and-swap pointer.

**Tech Stack:** Python 3.12, PyTorch, safetensors from `workspace/runtime/transfer_python`, pytest, JSON, SHA-256, PowerShell.

## Global Constraints

- Operate only inside `C:\Users\marco\Desktop\F51-Darwin-SSD`.
- Do not use network services or download models, dependencies or datasets.
- Never launch a concurrent trainer and never invoke `run247`.
- Keep checkpoints, packages, ledgers, probes and runtime reports under `workspace/`.
- Never select a checkpoint by filename ordering, modification time or largest cycle.
- Require explicit lowercase SHA-256 identities for every donor, recipient and parent.
- Do not overwrite published artifacts or checkpoints.
- Keep `disabled`, `shadow` and active-path gate-zero behavior equivalent within `1e-5` FP32.
- Restore-patch equivalence in causal ablation must be within `1e-6` FP32.
- Do not slice, pad, average or depth-map circuit tensors implicitly.
- Preserve and test the pre-existing `extra_organs` change in `src/f51_darwin/transplant/bundle.py`; do not discard it.
- Use the isolated transfer dependency directory by setting `$env:PYTHONPATH` to `workspace/runtime/transfer_python`.
- Every task stages and commits only its explicit paths.

---

## File Map

### Shared circuit library

- `src/f51_darwin/circuits/__init__.py`: public stable imports.
- `src/f51_darwin/circuits/identity.py`: canonical JSON and SHA-256 helpers.
- `src/f51_darwin/circuits/manifest.py`: immutable manifest and tap schemas.
- `src/f51_darwin/circuits/package.py`: deterministic directory package writer and verifier.
- `src/f51_darwin/circuits/taps.py`: allowlisted module-path and output selectors.
- `src/f51_darwin/circuits/compatibility.py`: static and executable recipient preflight.
- `src/f51_darwin/circuits/ablation.py`: paired CLEAN/ABLATE/RESTORE/SHUFFLE/RANDOM_MATCHED execution.
- `src/f51_darwin/circuits/transaction.py`: candidate installation and gate-zero wrapper state.
- `src/f51_darwin/circuits/ledger.py`: append-only hash-chained events.
- `src/f51_darwin/circuits/pointer.py`: atomic active-pointer compare-and-swap.
- `src/f51_darwin/circuits/rollback.py`: immutable rollback child creation.

### Operator surfaces

- `src/scripts/circuit_package.py`: package, verify, preflight and install commands.
- `src/scripts/circuit_ablate.py`: run and verify commands.
- `src/scripts/circuit_transplant.py`: inject, shadow, gate-zero, activate, rollback and verify commands.
- `src/scripts/start_circuit_vertical_slice.ps1`: non-launching source gate and explicit synthetic test launcher.

### Tests

- `src/tests/test_transplant_bundle.py`: preserve `extra_organs` behavior.
- `src/tests/test_circuit_identity.py`
- `src/tests/test_circuit_manifest.py`
- `src/tests/test_circuit_package.py`
- `src/tests/test_circuit_taps.py`
- `src/tests/test_circuit_compatibility.py`
- `src/tests/test_circuit_ablation_runtime.py`
- `src/tests/test_circuit_transaction.py`
- `src/tests/test_circuit_ledger_pointer.py`
- `src/tests/test_circuit_rollback.py`
- `src/tests/test_circuit_cli.py`
- `src/tests/test_circuit_vertical_slice.py`
- `src/tests/test_circuit_real_preflight.py`

---

### Task 1: Reconcile and Lock the Existing Bundle Change

**Files:**
- Modify: `src/tests/test_transplant_bundle.py`
- Modify: `src/f51_darwin/transplant/bundle.py`

**Interfaces:**
- Consumes: `extract_first_slice_bundle(..., extra_organs: tuple[str, ...])`.
- Produces: tested preservation of exact extra tensor prefixes without changing the v1 manifest contract.

- [ ] **Step 1: Add one extra tensor to the synthetic checkpoint**

Add this entry to `_synthetic_checkpoint()`:

```python
"spider_confidence_head.weight": torch.randn(1, 512),
```

- [ ] **Step 2: Write the failing extraction test**

```python
def test_extracts_explicit_extra_organ_prefix_without_backbone(tmp_path: Path) -> None:
    checkpoint = tmp_path / "donor.pt"
    bundle_path = tmp_path / "organs.pt"
    _synthetic_checkpoint(checkpoint)

    extract_first_slice_bundle(
        checkpoint,
        bundle_path,
        source_checkpoint_sha256=_sha256(checkpoint),
        gaba_layers=(0,),
        extra_organs=("spider_confidence_head.",),
    )

    tensors = load_organ_bundle(bundle_path).tensors
    assert "spider_confidence_head.weight" in tensors
    assert "token_embedding.weight" not in tensors
```

- [ ] **Step 3: Run the focused test**

Run:

```powershell
$env:PYTHONPATH = (Resolve-Path workspace\runtime\transfer_python).Path
.\.venv_nitro\Scripts\python.exe -m pytest -q -p no:cacheprovider src\\tests\\test_transplant_bundle.py
```

Expected: all tests pass. If this new test fails, preserve the existing
`extra_organs` behavior and make only the minimum formatting/validation
correction needed for the test.

- [ ] **Step 4: Format `_selected_key` without changing semantics**

Use this exact signature and body:

```python
def _selected_key(
    key: str,
    gaba_layers: tuple[int, ...],
    extra_organs: tuple[str, ...] = (),
) -> bool:
    if key.startswith("jepa_predictor."):
        return True
    if any(key.startswith(f"blocks.{layer}.gaba.") for layer in gaba_layers):
        return True
    return any(key.startswith(prefix) for prefix in extra_organs)
```

- [ ] **Step 5: Commit only the reconciled paths**

```powershell
git add -- src/f51_darwin/transplant/bundle.py src/tests/test_transplant_bundle.py
git commit -m "test(transplant): cover explicit extra organ extraction"
```

---

### Task 2: Canonical Circuit Identity and Manifest

**Files:**
- Create: `src/f51_darwin/circuits/__init__.py`
- Create: `src/f51_darwin/circuits/identity.py`
- Create: `src/f51_darwin/circuits/manifest.py`
- Create: `src/tests/test_circuit_identity.py`
- Create: `src/tests/test_circuit_manifest.py`

**Interfaces:**
- Produces: `canonical_json_bytes(value) -> bytes`.
- Produces: `canonical_sha256(value) -> str`.
- Produces: `TensorIdentity`, `TapContract`, `CircuitManifest`.
- Produces: `CircuitManifest.identity -> str` and strict `from_dict()`.

- [ ] **Step 1: Write canonical identity tests**

```python
from f51_darwin.circuits.identity import canonical_json_bytes, canonical_sha256


def test_canonical_identity_is_order_independent() -> None:
    left = {"b": [2, 3], "a": 1}
    right = {"a": 1, "b": [2, 3]}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_sha256(left) == canonical_sha256(right)


def test_canonical_identity_rejects_nan() -> None:
    import pytest

    with pytest.raises(ValueError):
        canonical_json_bytes({"bad": float("nan")})
```

- [ ] **Step 2: Implement canonical identity**

```python
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
```

- [ ] **Step 3: Write manifest round-trip and validation tests**

Test an actual manifest with:

```python
TensorIdentity(
    key="predictor.weight",
    role="parameter",
    shape=(4, 4),
    dtype="torch.float32",
    sha256="a" * 64,
)
```

and:

```python
TapContract(
    provider="module_path_v1",
    module_path="blocks.1",
    position="post",
    output_selector="tensor",
    rank=3,
    width=4,
    minimum_sequence_length=2,
)
```

Assert that `manifest == CircuitManifest.from_dict(manifest.to_dict())`,
that its identity begins with `f51-circuit-v1:`, and that invalid SHA-256,
empty module paths, negative widths and unknown positions raise `ValueError`.

- [ ] **Step 4: Implement immutable manifest dataclasses**

Implement frozen dataclasses with explicit `__post_init__` validation. The
identity core must exclude path hints and include:

```python
{
    "schema_version": 1,
    "source_checkpoint_sha256": self.source_checkpoint_sha256,
    "source_base_checkpoint_id": self.source_base_checkpoint_id,
    "source_config_identity": self.source_config_identity,
    "source_topology_identity": self.source_topology_identity,
    "circuit_kind": self.circuit_kind,
    "constructor_id": self.constructor_id,
    "tensors": [item.to_dict() for item in self.tensors],
    "tap": self.tap.to_dict(),
    "accepted_recipient_families": list(self.accepted_recipient_families),
}
```

`CircuitManifest.identity` returns:

```python
f"f51-circuit-v1:{canonical_sha256(self.identity_core())}"
```

- [ ] **Step 5: Run tests**

```powershell
.\.venv_nitro\Scripts\python.exe -m pytest -q -p no:cacheprovider src\\tests\\test_circuit_identity.py src\\tests\\test_circuit_manifest.py
```

Expected: pass.

- [ ] **Step 6: Commit**

```powershell
git add -- src/f51_darwin/circuits/__init__.py src/f51_darwin/circuits/identity.py src/f51_darwin/circuits/manifest.py src/tests/test_circuit_identity.py src/tests/test_circuit_manifest.py
git commit -m "feat(circuits): add canonical circuit manifest"
```

---

### Task 3: Safe Deterministic Circuit Package

**Files:**
- Create: `src/f51_darwin/circuits/package.py`
- Create: `src/tests/test_circuit_package.py`
- Modify: `src/f51_darwin/circuits/__init__.py`

**Interfaces:**
- Consumes: `CircuitManifest`.
- Produces: `write_circuit_package(root, manifest, tensors, runtime_state) -> PackageIdentity`.
- Produces: `verify_circuit_package(root, expected_sha256) -> LoadedCircuitPackage`.
- Package layout: `manifest.json`, `weights.safetensors`, `runtime_state.json`, `checksums.json`, `package.sha256`.

- [ ] **Step 1: Write package round-trip and tamper tests**

Use two CPU float32 tensors and assert:

```python
written = write_circuit_package(root, manifest, tensors, {"mode": "candidate"})
loaded = verify_circuit_package(root, written.sha256)
assert loaded.manifest == manifest
assert torch.equal(loaded.tensors["predictor.weight"], tensors["predictor.weight"])
```

Then modify one byte in `runtime_state.json` and assert
`verify_circuit_package()` raises `CircuitPackageError` containing
`"checksum mismatch"`.

Add separate tests for:

- an undeclared extra tensor;
- shape/dtype mismatch against `TensorIdentity`;
- non-finite floating tensor;
- existing non-empty destination;
- unexpected file in the package root; and
- malformed expected SHA-256.

- [ ] **Step 2: Implement atomic package writing**

Write into a sibling staging directory named
`.{destination.name}.staging-{pid}`. Use
`safetensors.torch.save_file()` for tensors. Write canonical UTF-8 JSON with
`sort_keys=True`, `allow_nan=False`, flush and `os.fsync()`.

After all files are present:

```python
checksums = {
    name: sha256_file(staging / name)
    for name in ("manifest.json", "weights.safetensors", "runtime_state.json")
}
```

Write `checksums.json`, calculate the package identity from the canonical
checksums object, write `package.sha256`, verify the staging directory, then
rename it to the absent destination.

- [ ] **Step 3: Implement strict verification**

Require the exact file set:

```python
{
    "manifest.json",
    "weights.safetensors",
    "runtime_state.json",
    "checksums.json",
    "package.sha256",
}
```

Load tensors with `safetensors.torch.load_file(device="cpu")`. Compare exact
keys, shapes, dtypes, finitude and the tensor SHA-256 recorded in the
manifest. Recompute the package identity and compare it with both
`expected_sha256` and `package.sha256`.

- [ ] **Step 4: Run tests in the isolated dependency environment**

```powershell
$env:PYTHONPATH = (Resolve-Path workspace\runtime\transfer_python).Path
.\.venv_nitro\Scripts\python.exe -m pytest -q -p no:cacheprovider src\\tests\\test_circuit_package.py
```

Expected: pass.

- [ ] **Step 5: Commit**

```powershell
git add -- src/f51_darwin/circuits/__init__.py src/f51_darwin/circuits/package.py src/tests/test_circuit_package.py
git commit -m "feat(circuits): add verified circuit packages"
```

---

### Task 4: Tap Providers and Paired Causal Ablation

**Files:**
- Create: `src/f51_darwin/circuits/taps.py`
- Create: `src/f51_darwin/circuits/compatibility.py`
- Create: `src/f51_darwin/circuits/ablation.py`
- Create: `src/tests/test_circuit_taps.py`
- Create: `src/tests/test_circuit_compatibility.py`
- Create: `src/tests/test_circuit_ablation_runtime.py`
- Modify: `src/f51_darwin/circuits/__init__.py`

**Interfaces:**
- Produces: `resolve_module(root, dotted_path) -> nn.Module`.
- Produces: `TapCapture(model, TapContract)`.
- Produces: `preflight_circuit(package, recipient, recipient_identity, probes) -> PreflightReport`.
- Produces: `CircuitSelector(layer_path, channels)`.
- Produces: `discover_residual_channels(model, batches, tap, count) -> CircuitSelector`.
- Produces: `run_paired_ablation(model_factory, batches, selector, seed) -> AblationResult`.
- Produces: `build_causal_verdict(runs, config) -> CausalVerdict`.
- Ablation arms: `CLEAN`, `ABLATE`, `RESTORE`, `SHUFFLE`, `RANDOM_MATCHED`.

- [ ] **Step 1: Write tap contract tests**

Create a toy model with `blocks = nn.ModuleList([nn.Linear(4, 4)])`.
Assert `resolve_module(model, "blocks.0")` returns that layer. Assert missing
paths, private segments, non-numeric `ModuleList` indices and output width
mismatch fail before executing an intervention.

- [ ] **Step 2: Implement strict dotted-path resolution**

Allow only segments matching `[A-Za-z][A-Za-z0-9_]*` or decimal indices.
Reject segments beginning with `_`. Traverse `nn.ModuleList` only with a
decimal index and normal modules only through registered submodules.

- [ ] **Step 3: Write a discovery-split ranking test**

Use a toy model where only hidden channels zero and one determine the correct
class. Pass a discovery batch with labels, rank channels by the accumulated
absolute gradient-times-activation score, and assert:

```python
selector = discover_residual_channels(
    model,
    discovery_batches,
    tap,
    count=2,
)
assert selector.layer_path == "blocks.0"
assert set(selector.channels) == {0, 1}
```

Pass different confirmation batches to `run_paired_ablation()` so the
discovery examples cannot be reused as causal proof.

- [ ] **Step 4: Implement residual-channel discovery**

Register a forward hook that retains the tapped activation gradient. For each
discovery batch, backpropagate the registered task loss and accumulate:

```python
score += (activation.detach() * activation.grad.detach()).abs().mean(
    dim=tuple(range(activation.ndim - 1))
)
```

Return the highest-scoring unique channel indices in ascending order inside
`CircuitSelector`. Reject missing gradients, non-finite scores, duplicate
indices and requests larger than the tap width.

- [ ] **Step 5: Write the known-circuit causal test**

Build a deterministic toy classifier whose first two channels alone decide
the target label. Run 5 arms from independent copies and assert:

```python
assert result.clean.accuracy == 1.0
assert result.ablate.accuracy <= 0.5
assert result.restore.accuracy == result.clean.accuracy
assert result.restore.max_abs_logit_error <= 1e-6
assert result.random_matched.accuracy > result.ablate.accuracy
assert result.anchors_identical is True
```

Also assert the shuffle permutation is deterministic for the same seed and
is a derangement for a batch larger than one.

- [ ] **Step 6: Implement paired arms**

Snapshot the initial model state and RNG once. Before each arm:

```python
model = model_factory()
model.load_state_dict(copy.deepcopy(initial_state), strict=True)
restore_rng(initial_rng)
```

Capture the clean activation for the selected channels. Interventions are:

- `CLEAN`: no replacement;
- `ABLATE`: replace selected channels with discovery baseline;
- `RESTORE`: apply ablation then patch the exact clean activation;
- `SHUFFLE`: replace selected activations with a seeded derangement across
  batch items; and
- `RANDOM_MATCHED`: select the same number of channels from outside the
  candidate using the seed.

Record input digest, model digest, RNG digest and batch order digest before
each arm. Fail unless they are identical.

- [ ] **Step 7: Write statistical gate tests**

Generate five deterministic seeds with 200 paired confirmation items per
seed. The known circuit data must satisfy:

```python
verdict = build_causal_verdict(runs, CausalGateConfig())
assert verdict.causal is True
assert verdict.target_effect_ci_low > 0.0
assert verdict.target_effect_mean >= 0.15
assert verdict.restore_fraction >= 0.95
assert verdict.target_to_control_ratio >= 2.0
assert verdict.holm_rejected is True
```

Create a correlated-only candidate whose ablation effect equals the matched
control and assert `verdict.causal is False` with reason
`"matched_control_not_beaten"`.

- [ ] **Step 8: Implement deterministic statistical gates**

Add:

```python
@dataclass(frozen=True)
class CausalGateConfig:
    minimum_seeds: int = 5
    minimum_items_per_domain: int = 200
    minimum_effect_nats: float = 0.15
    minimum_control_ratio: float = 2.0
    minimum_restore_fraction: float = 0.95
    minimum_shuffle_fraction: float = 0.50
    alpha: float = 0.01
    bootstrap_draws: int = 4096
    signflip_draws: int = 8192
    seed: int = 51
```

Compute the paired effect per item as `ablated_nll - clean_nll`. Use a seeded
paired bootstrap over items for the 95% mean-effect interval. Compute a
seeded two-sided sign-flip p-value for every domain and apply Holm's
step-down correction in ascending p-value order. Fail closed when seed count,
item count, restore, shuffle, retention or matched-control evidence is
missing.

- [ ] **Step 9: Persist and verify the result**

Implement `AblationResult.to_dict()` with arm metrics, logit digests,
candidate selector, anchors and checks. Use `canonical_sha256()` as
`evidence_sha256`.

- [ ] **Step 10: Write compatibility preflight tests**

Use one compatible and one incompatible toy recipient. Assert the compatible
report contains:

```python
assert report.compatible is True
assert report.launch is False
assert report.gate_zero_max_abs_logit_error <= 1e-5
assert report.state_mutations == 0
```

Assert preflight rejects:

- recipient SHA-256 mismatch;
- recipient family not declared by the manifest;
- unresolved or occupied tap;
- width, rank or minimum-sequence mismatch;
- candidate non-finite output;
- mutated backbone parameter or buffer; and
- unsupported mask/cache requirement.

- [ ] **Step 11: Implement static and executable preflight**

`preflight_circuit()` first verifies package and recipient identities, then
resolves the allowlisted provider/tap and compares the static interface.
Next it hashes all recipient parameters and buffers, runs the registered
probe through baseline, shadow and active-path gate-zero, and returns a
frozen `PreflightReport`.

Set:

```python
compatible = (
    max_abs_shadow_error <= 1e-5
    and gate_zero_max_abs_logit_error <= 1e-5
    and state_mutations == 0
    and outputs_finite
)
```

The report always records `launch=False`. Any static incompatibility raises
`CircuitCompatibilityError` before installing a wrapper.

- [ ] **Step 12: Run tests**

```powershell
.\.venv_nitro\Scripts\python.exe -m pytest -q -p no:cacheprovider src\\tests\\test_circuit_taps.py src\\tests\\test_circuit_compatibility.py src\\tests\\test_circuit_ablation_runtime.py
```

Expected: pass.

- [ ] **Step 13: Commit**

```powershell
git add -- src/f51_darwin/circuits/__init__.py src/f51_darwin/circuits/taps.py src/f51_darwin/circuits/compatibility.py src/f51_darwin/circuits/ablation.py src/tests/test_circuit_taps.py src/tests/test_circuit_compatibility.py src/tests/test_circuit_ablation_runtime.py
git commit -m "feat(circuits): prove paired causal ablation"
```

---

### Task 5: Gate-Zero Injection and Replacement Transaction

**Files:**
- Create: `src/f51_darwin/circuits/transaction.py`
- Create: `src/tests/test_circuit_transaction.py`
- Modify: `src/f51_darwin/circuits/__init__.py`

**Interfaces:**
- Produces: `GatedCircuitWrapper(original, candidate, mode, max_residual_ratio)`.
- Produces: `CircuitTransaction.create(...)`.
- Produces: `transaction.verify_shadow(probes, tolerance=1e-5)`.
- Produces: `transaction.verify_gate_zero(probes, tolerance=1e-5)`.
- Modes: `inject` and `replace`.

- [ ] **Step 1: Write zero-impact wrapper tests**

For a deterministic `nn.Linear(4, 4)` original and a different candidate,
assert:

```python
wrapper = GatedCircuitWrapper(original, candidate, mode="replace")
baseline = original(inputs)
shadow = wrapper(inputs, circuit_mode="shadow")
gate_zero = wrapper(inputs, circuit_mode="active")
torch.testing.assert_close(shadow, baseline, atol=0, rtol=0)
torch.testing.assert_close(gate_zero, baseline, atol=1e-7, rtol=0)
assert wrapper.external_gate.item() == 0.0
```

Set the gate to `0.5` and assert output changes but remains finite. Add tests
for tuple output selector zero, incompatible output shapes and residual-bound
enforcement.

- [ ] **Step 2: Implement the gated wrapper**

For `replace`, use:

```python
delta = candidate_output - original_output
combined = original_output + torch.tanh(self.external_gate) * self._bounded(
    original_output, delta
)
```

For `inject`, use the candidate output directly as the bounded residual.
In `shadow`, execute the candidate under `torch.no_grad()` but return the
original output object without reconstructing it.

- [ ] **Step 3: Write transaction identity and state tests**

Create a transaction with:

- package SHA-256;
- recipient parent SHA-256;
- tap contract identity;
- circuit identity;
- pre-install snapshot SHA-256; and
- initial lifecycle `candidate`.

Assert it rejects invalid identities, existing transaction roots, lifecycle
skips and a shadow run that mutates any backbone parameter or buffer.

- [ ] **Step 4: Implement immutable transaction manifests**

Write `transaction.json` exclusively and include:

```python
{
    "schema_version": 1,
    "transaction_id": transaction_id,
    "operation": operation,
    "circuit_id": circuit_id,
    "package_sha256": package_sha256,
    "recipient_parent_sha256": recipient_parent_sha256,
    "tap_contract_sha256": tap_contract_sha256,
    "preinstall_snapshot_sha256": snapshot_sha256,
    "lifecycle": "candidate",
}
```

Every lifecycle update writes a new event artifact; do not rewrite the
original manifest.

- [ ] **Step 5: Run tests**

```powershell
.\.venv_nitro\Scripts\python.exe -m pytest -q -p no:cacheprovider src\\tests\\test_circuit_transaction.py
```

Expected: pass.

- [ ] **Step 6: Commit**

```powershell
git add -- src/f51_darwin/circuits/__init__.py src/f51_darwin/circuits/transaction.py src/tests/test_circuit_transaction.py
git commit -m "feat(circuits): add gate-zero transplant transactions"
```

---

### Task 6: Hash-Chained Ledger, Atomic Pointer and Rollback Child

**Files:**
- Create: `src/f51_darwin/circuits/ledger.py`
- Create: `src/f51_darwin/circuits/pointer.py`
- Create: `src/f51_darwin/circuits/rollback.py`
- Create: `src/tests/test_circuit_ledger_pointer.py`
- Create: `src/tests/test_circuit_rollback.py`
- Modify: `src/f51_darwin/circuits/__init__.py`

**Interfaces:**
- Produces: `CircuitLedger.append(event) -> LedgerRecord`.
- Produces: `CircuitLedger.verify() -> str`.
- Produces: `publish_active_pointer(root, expected_parent_sha256, candidate)`.
- Produces: `create_rollback_child(active_checkpoint, snapshot, circuit_id, output)`.

- [ ] **Step 1: Write ledger tamper tests**

Append three records and verify each record contains the previous record
SHA-256. Assert verification fails after deleting the middle line,
reordering two lines, editing one field or truncating the last JSON object.

- [ ] **Step 2: Implement the hash chain**

The first `previous_record_sha256` is 64 zeros. Each record identity is:

```python
record_sha256 = canonical_sha256(
    {
        "schema_version": 1,
        "sequence": sequence,
        "previous_record_sha256": previous_record_sha256,
        "event": event,
    }
)
```

Append one canonical JSON line using a file opened in append-binary mode,
flush and fsync before returning.

- [ ] **Step 3: Write pointer compare-and-swap tests**

Start with a pointer to checkpoint A. Publish B with expected A and assert
success. Attempt to publish C with expected A and assert `PointerConflict`
while the pointer remains B. Inject an exception before `os.replace` and
assert the original pointer remains valid.

- [ ] **Step 4: Implement atomic pointer publication**

Require pointer payload:

```python
{
    "schema_version": 1,
    "checkpoint": str(candidate.path.resolve()),
    "checkpoint_sha256": candidate.sha256,
    "parent_checkpoint_sha256": expected_parent_sha256,
    "ledger_sha256": ledger_sha256,
}
```

Validate the current pointer and checkpoint hashes, write a sibling temporary
file, flush/fsync, then `os.replace`.

- [ ] **Step 5: Write rollback child tests**

Install two toy circuits, modify both, and roll back only one. Load the new
checkpoint into a fresh model and assert:

```python
for key, expected in preinstall_a.items():
    assert torch.equal(restored.circuit_a.state_dict()[key], expected)
for key, expected in active_b.items():
    assert torch.equal(restored.circuit_b.state_dict()[key], expected)
assert rollback_metadata["parent_checkpoint_sha256"] == active_sha256
assert rollback_metadata["rolled_back_circuit_id"] == circuit_a_id
```

- [ ] **Step 6: Implement rollback as a new immutable checkpoint**

Verify active checkpoint, snapshot, package, tap and ledger identities.
Overlay only keys under the registered circuit state prefix. Save to an
absent output path using the atomic checkpoint helper. Reload into a fresh
model factory and execute the registered probe before pointer publication.

- [ ] **Step 7: Run tests**

```powershell
.\.venv_nitro\Scripts\python.exe -m pytest -q -p no:cacheprovider src\\tests\\test_circuit_ledger_pointer.py src\\tests\\test_circuit_rollback.py
```

Expected: pass.

- [ ] **Step 8: Commit**

```powershell
git add -- src/f51_darwin/circuits/__init__.py src/f51_darwin/circuits/ledger.py src/f51_darwin/circuits/pointer.py src/f51_darwin/circuits/rollback.py src/tests/test_circuit_ledger_pointer.py src/tests/test_circuit_rollback.py
git commit -m "feat(circuits): publish and roll back immutable circuits"
```

---

### Task 7: Three CLIs and Synthetic End-to-End Test

**Files:**
- Create: `src/scripts/circuit_package.py`
- Create: `src/scripts/circuit_ablate.py`
- Create: `src/scripts/circuit_transplant.py`
- Create: `src/scripts/start_circuit_vertical_slice.ps1`
- Create: `src/tests/test_circuit_cli.py`
- Create: `src/tests/test_circuit_vertical_slice.py`

**Interfaces:**
- Consumes: all library interfaces from Tasks 2-6.
- Produces: stable machine-readable output and exit codes.
- Exit 0: verified success.
- Exit 2: contract/evidence incompatibility.
- Exit 3: pointer conflict.
- Exit 4: artifact tamper or identity mismatch.

- [ ] **Step 1: Write CLI parser tests**

Import each script module and assert its parser exposes only the approved
commands:

- package CLI: `package`, `verify`, `preflight`, `install`;
- ablation CLI: `discover`, `run`, `verify`;
- transplant CLI: `inject`, `replace`, `shadow`, `gate-zero`, `adapter`,
  `activate`, `rollback`, `verify`.

Assert mutation commands require explicit package, recipient, expected
hashes, transaction root and checkpoint root. Assert no command has an
`--active` shortcut on package installation.

- [ ] **Step 2: Implement thin CLI entrypoints**

Each script must:

```python
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = dispatch(args)
    except ContractError as exc:
        print(json.dumps({"status": "rejected", "error": str(exc)}))
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0
```

Do not duplicate package, ablation or transaction logic inside scripts.

- [ ] **Step 3: Write the synthetic vertical-slice test**

The test must:

1. construct a deterministic toy donor with a known two-channel causal
   circuit;
2. discover the correct two channels on the discovery split;
3. run five seeds of the five confirmation arms with 200 items per seed;
4. pass the statistical causal gates;
5. package the selected circuit;
6. verify and install it into a different toy recipient;
7. prove shadow and active-path gate-zero equivalence;
8. open the gate and prove the target behavior changes;
9. publish an immutable child checkpoint;
10. roll back only that circuit into another child;
11. reload in a fresh model instance; and
12. verify all package, checkpoint, ledger and pointer SHA-256 values.

The final assertions are:

```python
assert report["ablation"]["causal"] is True
assert report["gate_zero"]["max_abs_logit_error"] <= 1e-5
assert report["activation"]["pointer_swapped"] is True
assert report["rollback"]["restored"] is True
assert report["verification"]["all_hashes_valid"] is True
```

- [ ] **Step 4: Implement the PowerShell launcher**

The launcher accepts only:

```powershell
param(
    [switch]$Canary,
    [string]$OutputRoot = "workspace\runtime\circuit_vertical_slice"
)
```

It sets the isolated `PYTHONPATH`, rejects competing trainer command lines,
runs all `src/tests/test_circuit_*.py`, and:

- with `-Canary`, exits after tests with
  `CIRCUIT_CANARY_OK launch=false`;
- without `-Canary`, runs the deterministic synthetic vertical slice in the
  current process and prints the artifact root and final verification hash.

It must not call `Start-Process`.

- [ ] **Step 5: Run CLI and vertical-slice tests**

```powershell
$env:PYTHONPATH = (Resolve-Path workspace\runtime\transfer_python).Path
.\.venv_nitro\Scripts\python.exe -m pytest -q -p no:cacheprovider src\\tests\\test_circuit_cli.py src\\tests\\test_circuit_vertical_slice.py
powershell -ExecutionPolicy Bypass -File src\\scripts\\start_circuit_vertical_slice.ps1 -Canary
```

Expected:

```text
CIRCUIT_CANARY_OK launch=false
```

- [ ] **Step 6: Commit**

```powershell
git add -- src/scripts/circuit_package.py src/scripts/circuit_ablate.py src/scripts/circuit_transplant.py src/scripts/start_circuit_vertical_slice.ps1 src/tests/test_circuit_cli.py src/tests/test_circuit_vertical_slice.py
git commit -m "feat(circuits): deliver tested circuit transplant CLI"
```

---

### Task 8: Real JEPA-to-SSD Non-Launching Preflight and Final Verification

**Files:**
- Create: `src/tests/test_circuit_real_preflight.py`
- Create: `src/configs/circuits/jepa-v3.json`
- Create: `src/configs/circuits/taps/darwin-transfer-final-hidden.json`
- Modify: `governance/docs/operacao/OPERACAO_SEGURA.md`
- Modify: `governance/docs/operacao/STATUS_ATUAL.md`

**Interfaces:**
- Consumes:
  - V3 organ donor checkpoint SHA-256
    `71c49bc295c0d06d72b3dc640d5b4d846f421ecdcca25820517fffaab6425a30`.
  - Published SSD checkpoint SHA-256
    `59847bdae17017d3a7fe3b82624e475dda9b4f7a20d73115426c7c26ca5afe2e`.
- Produces: immutable CPU preflight report with `launch=false`.

- [ ] **Step 1: Write the real artifact identity test**

Skip only when an explicitly named artifact is absent. When present, assert
its measured SHA-256 equals the fixed identity above. Load the V3 checkpoint
weights-only where possible and confirm the JEPA tensor set is non-empty.
Load the SSD published pointer and confirm:

```python
assert pointer["status"] == "accepted"
assert pointer["attention_layers_remaining"] == 0
assert pointer["checkpoint_sha256"] == SSD_SHA256
```

- [ ] **Step 2: Write the CPU preflight test**

Build the package from the registered JEPA tensor spec, instantiate the SSD
recipient through the existing transfer loader, resolve
`darwin-transfer-final-hidden`, and assert:

```python
assert report["compatible"] is True
assert report["launch"] is False
assert report["recipient_family"] == "darwin-transfer-discrete-ssd"
assert report["gate_zero_max_abs_logit_error"] <= 1e-5
assert report["state_mutations"] == 0
```

The test may create packages and reports only under `tmp_path`.

- [ ] **Step 3: Run all circuit and transplant tests**

```powershell
$env:PYTHONPATH = (Resolve-Path workspace\runtime\transfer_python).Path
.\.venv_nitro\Scripts\python.exe -m pytest -q -p no:cacheprovider src\\tests\\test_circuit_*.py src\\tests\\test_transplant_*.py
```

Expected: pass.

- [ ] **Step 4: Run the integral source audit**

```powershell
$env:PYTHONPATH = (Resolve-Path workspace\runtime\transfer_python).Path
.\.venv_nitro\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Expected: all collected tests pass. Do not run the real preflight launcher
if this command is not fully green.

- [ ] **Step 5: Execute the synthetic vertical slice**

```powershell
powershell -ExecutionPolicy Bypass -File src\\scripts\\start_circuit_vertical_slice.ps1
```

Expected output contains:

```text
CIRCUIT_VERTICAL_SLICE_OK
```

and names an artifact root under
`workspace/runtime/circuit_vertical_slice/`. Verify the report in a fresh
process using `src/scripts/circuit_transplant.py verify`.

- [ ] **Step 6: Execute the real CPU preflight without activation**

Run `src/scripts/circuit_package.py preflight` against the fixed V3 donor and SSD
recipient identities. Expected output:

```json
{"compatible": true, "launch": false, "status": "PREFLIGHT_OK"}
```

Do not open a gate, train an adapter, publish an active pointer or start a
trainer in this task.

- [ ] **Step 7: Update operational documentation with measured evidence**

Document:

- exact Git commit;
- test counts and command;
- synthetic artifact root and verification SHA-256;
- real donor/recipient hashes;
- measured gate-zero error;
- `launch=false`; and
- remaining semantic-circuit research gates.

Do not claim that a mathematics or philosophy circuit was discovered.

- [ ] **Step 8: Final commit**

```powershell
git add -- src/tests/test_circuit_real_preflight.py src/configs/circuits/jepa-v3.json src/configs/circuits/taps/darwin-transfer-final-hidden.json governance/docs/operacao/OPERACAO_SEGURA.md governance/docs/operacao/STATUS_ATUAL.md
git commit -m "test(circuits): verify JEPA to SSD preflight"
```

---

## Completion Gate

The vertical slice is complete only when all of the following are true:

- focused circuit and transplant tests pass;
- the integral source audit is fully green;
- the synthetic known circuit passes ablation, package, gate-zero,
  activation and rollback;
- verification from a fresh process confirms every SHA-256;
- real V3 JEPA and published SSD identities match their fixed hashes;
- real CPU preflight reports `compatible=true` and `launch=false`;
- no trainer was started; and
- Git contains no uncommitted task changes.

This completion proves the transplant mechanism. It does not prove semantic
knowledge localization. A later plan will apply the same mechanism to
independent mathematics and philosophy discovery/confirmation datasets.
