# Two-Donor Organ Transplant First Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first falsifiable two-donor recipient that copies DistilGPT-2 language competence, transplants the original step-1750 JEPA and GABA organs, reproduces simple-GPT logits with organs disabled, and measures matched A/B/C/D gains with per-organ rollback.

**Architecture:** A new `f51_darwin.transplant` package extracts a content-addressed organ bundle from `FULL_ORGANISM_V3`, reconstructs the original 512-dimensional modules, and connects them to an exact DistilGPT-2 copy through 768-to-512 adapters. JEPA remains an auxiliary predictive-loss organ, matching its original role; GABA is a bounded residual organ injected after one GPT block. Lifecycle policies, matched arms, a ledger and recipient checkpoints make every gain, gradient and rollback auditable.

**Tech Stack:** Python 3.12, PyTorch, Transformers, pytest, JSON/JSONL, existing F51 organ identity code, PowerShell.

## Global Constraints

- Work only inside `C:\Users\marco\Desktop\F51-Darwin-SSD`.
- Preserve the untracked Claude files and `src/configs/darwin_x_100m_denso_full_organism.yaml`.
- Donor A is the original verified DistilGPT-2 under `workspace/00_DONORS/distilgpt2`.
- Donor B is `workspace/03_CHECKPOINTS_100M_FULL_ORGANISM_V3/organism_cycle_002_step_001750.pt`.
- Never write into either donor root or any existing checkpoint lineage.
- New ignored root: `workspace/03_CHECKPOINTS_DARWIN_TWO_DONOR_V1`.
- The first slice includes original JEPA and donor GABA layer 0 only.
- Do not pad, slice, average or randomly recreate donor organ tensors.
- All gates start at zero and the disabled recipient must match simple GPT within `1e-5` FP32.
- No organ receives credit over an immutable GPT baseline alone; the matched trained control is mandatory.
- Use locked holdouts and identical tokens, starts, seeds and optimizer-step budgets across arms.
- Checkpoints record both donor identities, parent recipient identity, lifecycle states and ledger hash.
- Use explicit paths in commits, Conventional Commits and never `--no-verify`.

---

## File Structure

- `src/f51_darwin/transplant/__init__.py`: public first-slice interfaces.
- `src/f51_darwin/transplant/bundle.py`: trusted extraction, tensor hashing, manifest and bundle round-trip.
- `src/f51_darwin/transplant/organs.py`: strict reconstruction of original JEPA and GABA modules.
- `src/f51_darwin/transplant/slots.py`: 768-to-512 adapters, zero gates and organ-specific execution.
- `src/f51_darwin/transplant/recipient.py`: exact GPT copy, GABA block injection and JEPA auxiliary output.
- `src/f51_darwin/transplant/lifecycle.py`: declared trainable sets and lifecycle transitions.
- `src/f51_darwin/transplant/ledger.py`: causal-arm measurements, utility and immutable JSONL records.
- `src/f51_darwin/transplant/checkpoint.py`: recipient save/load, parent identity and per-organ rollback.
- `src/f51_darwin/transplant/experiment.py`: deterministic A/B/C/D arm construction and budget validation.
- `src/scripts/extract_darwin_organs.py`: operator extraction command.
- `src/scripts/run_two_donor_canary.py`: non-production first-slice experiment.
- `src/scripts/start_two_donor_transplant.ps1`: source audit and isolated launcher.
- `src/tests/test_transplant_bundle.py`: extraction and round-trip.
- `src/tests/test_transplant_organs.py`: strict original-module reconstruction.
- `src/tests/test_transplant_recipient.py`: zero-gate identity and finite organ paths.
- `src/tests/test_transplant_lifecycle.py`: gradients, transitions and quarantine.
- `src/tests/test_transplant_ledger.py`: matched arms and utility.
- `src/tests/test_transplant_checkpoint.py`: identity and rollback.
- `src/tests/test_transplant_launcher.py`: non-launching canary and isolation.

---

### Task 1: Content-addressed organ bundle

**Files:**
- Create: `src/f51_darwin/transplant/__init__.py`
- Create: `src/f51_darwin/transplant/bundle.py`
- Create: `src/tests/test_transplant_bundle.py`
- Create: `src/scripts/extract_darwin_organs.py`

**Interfaces:**
- Consumes: a trusted local checkpoint path and explicit organ selectors.
- Produces: `TensorRecord`, `OrganBundleManifest`, `OrganBundle`, `extract_first_slice_bundle()` and `load_organ_bundle()`.

- [ ] **Step 1: Write failing manifest and selector tests**

```python
from pathlib import Path

import torch

from f51_darwin.transplant.bundle import (
    extract_first_slice_bundle,
    load_organ_bundle,
)


def synthetic_checkpoint(path: Path) -> None:
    torch.save(
        {
            "version": 9,
            "base_checkpoint_id": "core:test",
            "config": {"d_model": 512, "jepa_weight": 0.05, "gaba_enabled": True},
            "model_state_dict": {
                "token_embedding.weight": torch.randn(11, 512),
                "jepa_predictor.predictor.0.weight": torch.randn(256, 512),
                "jepa_predictor.predictor.0.bias": torch.randn(256),
                "blocks.0.gaba.inhibitory_weight": torch.randn(512, 512),
                "blocks.0.gaba.inhibitory_bias": torch.randn(512),
                "blocks.1.gaba.inhibitory_weight": torch.randn(512, 512),
            },
            "heartbeat_state": {"step": 3},
            "organ_identity_report": {"jepa": "organ:jepa:v1:test"},
        },
        path,
    )


def test_extracts_only_requested_original_organs(tmp_path: Path) -> None:
    checkpoint = tmp_path / "donor.pt"
    bundle_path = tmp_path / "organs.pt"
    synthetic_checkpoint(checkpoint)
    manifest = extract_first_slice_bundle(
        checkpoint,
        bundle_path,
        source_checkpoint_sha256="a" * 64,
        gaba_layers=(0,),
    )
    assert set(manifest.organs) == {"jepa", "gaba.0"}
    bundle = load_organ_bundle(bundle_path)
    assert "jepa_predictor.predictor.0.weight" in bundle.tensors
    assert "blocks.0.gaba.inhibitory_weight" in bundle.tensors
    assert "token_embedding.weight" not in bundle.tensors
    assert "blocks.1.gaba.inhibitory_weight" not in bundle.tensors


def test_bundle_round_trip_preserves_each_tensor_hash(tmp_path: Path) -> None:
    checkpoint = tmp_path / "donor.pt"
    bundle_path = tmp_path / "organs.pt"
    synthetic_checkpoint(checkpoint)
    manifest = extract_first_slice_bundle(
        checkpoint,
        bundle_path,
        source_checkpoint_sha256="b" * 64,
        gaba_layers=(0,),
    )
    restored = load_organ_bundle(bundle_path)
    assert restored.manifest == manifest
    assert restored.verify_tensor_hashes() == []
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
.\.venv_nitro\Scripts\python.exe -m pytest -q src\\tests\\test_transplant_bundle.py
```

Expected: collection fails because `f51_darwin.transplant.bundle` does not exist.

- [ ] **Step 3: Implement exact selectors and deterministic tensor hashing**

```python
@dataclass(frozen=True)
class TensorRecord:
    key: str
    shape: tuple[int, ...]
    dtype: str
    sha256: str


@dataclass(frozen=True)
class OrganBundleManifest:
    schema_version: int
    source_checkpoint: str
    source_checkpoint_sha256: str
    base_checkpoint_id: str
    organs: tuple[str, ...]
    tensors: tuple[TensorRecord, ...]


@dataclass
class OrganBundle:
    manifest: OrganBundleManifest
    tensors: dict[str, torch.Tensor]
    donor_config: dict[str, Any]
    runtime_state: dict[str, Any]

    def verify_tensor_hashes(self) -> list[str]:
        expected = {record.key: record.sha256 for record in self.manifest.tensors}
        return [
            key
            for key, tensor in self.tensors.items()
            if tensor_sha256(tensor) != expected.get(key)
        ]


def selected_key(key: str, gaba_layers: tuple[int, ...]) -> bool:
    normalized = key.replace("_orig_mod.", "")
    if normalized.startswith("jepa_predictor."):
        return True
    return any(
        normalized.startswith(f"blocks.{layer}.gaba.")
        for layer in gaba_layers
    )
```

Use `torch.load(..., weights_only=False, mmap=True)` only for the explicitly trusted local Darwin checkpoint. Clone selected tensors to CPU before writing the new ignored bundle. Serialize a plain dictionary containing the dataclass fields, tensors, filtered config and runtime state so `load_organ_bundle()` can use `weights_only=True`.

- [ ] **Step 4: Add the operator extraction command**

`src/scripts/extract_darwin_organs.py` must require:

```text
--checkpoint
--output
--checkpoint-sha256
--gaba-layers 0
```

It prints JSON with the selected organs, tensor count, parameter count, source hash and output path. It exits non-zero on a source-hash mismatch or bundle verification error.

- [ ] **Step 5: Run GREEN and bundle CLI smoke**

Run:

```powershell
.\.venv_nitro\Scripts\python.exe -m pytest -q src\\tests\\test_transplant_bundle.py
.\.venv_nitro\Scripts\python.exe -m py_compile src\\scripts\\extract_darwin_organs.py
git diff --check
```

Expected: tests pass, compilation exits 0 and diff check is clean.

- [ ] **Step 6: Commit**

```powershell
git add -- src/f51_darwin/transplant/__init__.py src/f51_darwin/transplant/bundle.py src/tests/test_transplant_bundle.py src/scripts/extract_darwin_organs.py
git commit -m "feat(transplant): extract original organ bundle"
```

---

### Task 2: Strict reconstruction of original JEPA and GABA

**Files:**
- Create: `src/f51_darwin/transplant/organs.py`
- Create: `src/tests/test_transplant_organs.py`

**Interfaces:**
- Consumes: `OrganBundle`.
- Produces: `OriginalJEPATransplant`, `OriginalGABATransplant`, `build_original_jepa()` and `build_original_gaba()`.

- [ ] **Step 1: Write failing strict-load tests**

```python
def test_build_original_jepa_loads_every_donor_tensor(bundle) -> None:
    transplant = build_original_jepa(bundle)
    assert isinstance(transplant.module, JEPAHeadV2)
    assert transplant.module.d_model == 512
    assert transplant.source_keys == tuple(
        key for key in bundle.tensors if key.startswith("jepa_predictor.")
    )
    assert transplant.missing_keys == ()
    assert transplant.unexpected_keys == ()


def test_build_original_gaba_preserves_parameters_and_buffers(bundle) -> None:
    transplant = build_original_gaba(bundle, donor_layer=0)
    donor = bundle.tensors["blocks.0.gaba.inhibitory_weight"]
    assert torch.equal(transplant.module.inhibitory_weight, donor)
    assert transplant.module.config.d_model == 512
    assert transplant.missing_keys == ()
    assert transplant.unexpected_keys == ()
```

Add a rejection test that removes one expected JEPA key and requires
`ValueError("incomplete jepa organ")`.

- [ ] **Step 2: Run RED**

Run:

```powershell
.\.venv_nitro\Scripts\python.exe -m pytest -q src\\tests\\test_transplant_organs.py
```

Expected: import failure for `f51_darwin.transplant.organs`.

- [ ] **Step 3: Implement original module factories**

Construct the exact checkpoint architecture:

```python
jepa = JEPAHeadV2(
    d_model=512,
    hidden_dim=768,
    bottleneck_dim=256,
    dropout=0.0,
)
gaba = GABAergicLayer(GABAConfig(d_model=512))
```

Strip `jepa_predictor.` and `blocks.{layer}.gaba.` prefixes, call
`load_state_dict(..., strict=True)`, freeze every parameter and return a
`ReconstructedOrgan` dataclass containing module, source keys and source
tensor hashes. Never fill missing keys with random initialization.

- [ ] **Step 4: Run GREEN and existing organ regression tests**

Run:

```powershell
.\.venv_nitro\Scripts\python.exe -m pytest -q src\\tests\\test_transplant_organs.py src\\tests\\test_gaba_causal.py
```

Expected: zero failures.

- [ ] **Step 5: Commit**

```powershell
git add -- src/f51_darwin/transplant/organs.py src/tests/test_transplant_organs.py
git commit -m "feat(transplant): reconstruct donor organs strictly"
```

---

### Task 3: Organ adapters and zero-impact slots

**Files:**
- Create: `src/f51_darwin/transplant/slots.py`
- Create: `src/tests/test_transplant_recipient.py`

**Interfaces:**
- Consumes: reconstructed 512-dimensional JEPA/GABA modules.
- Produces: `OrganAdapter`, `JEPAAuxiliarySlot`, `GABAResidualSlot`, `OrganSlotOutput`.

- [ ] **Step 1: Write failing slot tests**

```python
def test_gaba_slot_is_exact_noop_at_zero_external_gate(gaba) -> None:
    slot = GABAResidualSlot(gaba)
    hidden = torch.randn(2, 5, 768)
    output = slot(hidden, enabled=True, mutate_state=False)
    assert torch.equal(output.hidden, hidden)
    assert output.observation is not None


def test_jepa_slot_preserves_original_predictor_and_returns_finite_loss(jepa) -> None:
    slot = JEPAAuxiliarySlot(jepa)
    hidden = torch.randn(2, 6, 768)
    output = slot(hidden, enabled=True)
    assert output.hidden.shape == hidden.shape
    assert output.auxiliary_loss.ndim == 0
    assert torch.isfinite(output.auxiliary_loss)
    assert torch.equal(output.hidden, hidden)
```

Also assert all donor-organ parameters remain frozen and only adapter/gate
parameters initially require gradients.

- [ ] **Step 2: Run RED**

Run:

```powershell
.\.venv_nitro\Scripts\python.exe -m pytest -q src\\tests\\test_transplant_recipient.py
```

Expected: import failure for `f51_darwin.transplant.slots`.

- [ ] **Step 3: Implement bounded adapters and organ semantics**

```python
class OrganAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(768)
        self.to_organ = nn.Linear(768, 512)
        self.from_organ = nn.Linear(512, 768)


class GABAResidualSlot(nn.Module):
    def __init__(self, organ: GABAergicLayer) -> None:
        super().__init__()
        self.organ = organ
        self.adapter = OrganAdapter()
        self.external_gate = nn.Parameter(torch.zeros(()))

    def forward(self, hidden, *, enabled, mutate_state=False):
        canonical = self.adapter.to_organ(self.adapter.input_norm(hidden))
        delta, observation = self.organ(canonical, mutate_state=mutate_state)
        projected = self.adapter.from_organ(delta)
        gate = torch.tanh(self.external_gate)
        combined = hidden + (gate * projected if enabled else projected.detach() * 0)
        return OrganSlotOutput(combined, None, observation)
```

`JEPAAuxiliarySlot` maps hidden to 512, calls the original JEPA, computes
`1 - cosine_similarity(predicted, target.detach())`, and returns the input
hidden unchanged. In the control arm it still executes the predictor but
multiplies a detached loss by zero, preserving compute without providing an
organ gradient.

- [ ] **Step 4: Run GREEN**

Run:

```powershell
.\.venv_nitro\Scripts\python.exe -m pytest -q src\\tests\\test_transplant_recipient.py
git diff --check
```

Expected: zero failures and clean diff.

- [ ] **Step 5: Commit**

```powershell
git add -- src/f51_darwin/transplant/slots.py src/tests/test_transplant_recipient.py
git commit -m "feat(transplant): add zero-impact organ slots"
```

---

### Task 4: Exact DistilGPT-2 recipient and organ injection

**Files:**
- Create: `src/f51_darwin/transplant/recipient.py`
- Modify: `src/tests/test_transplant_recipient.py`

**Interfaces:**
- Consumes: verified `GPT2LMHeadModel`, `JEPAAuxiliarySlot`, `GABAResidualSlot`.
- Produces: `TwoDonorRecipient`, `RecipientOutput`, `from_language_donor()`.

- [ ] **Step 1: Add failing zero-gate identity and state-mutation tests**

```python
def test_zero_gate_recipient_matches_simple_gpt(tiny_gpt, slots) -> None:
    recipient = TwoDonorRecipient.from_language_donor(
        tiny_gpt,
        jepa_slot=slots.jepa,
        gaba_slots={0: (slots.gaba,)},
    )
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        baseline = tiny_gpt(input_ids=ids).logits
        actual = recipient(input_ids=ids, organ_mode="disabled").logits
    torch.testing.assert_close(actual, baseline, atol=1e-5, rtol=0)


def test_disabled_recipient_does_not_mutate_gaba_state(tiny_gpt, slots) -> None:
    before = slots.gaba.organ.state_update_count.clone()
    recipient = TwoDonorRecipient.from_language_donor(
        tiny_gpt, jepa_slot=slots.jepa, gaba_slots={0: (slots.gaba,)}
    )
    recipient(input_ids=torch.tensor([[1, 2, 3]]), organ_mode="disabled")
    assert torch.equal(slots.gaba.organ.state_update_count, before)
```

- [ ] **Step 2: Run RED**

Run:

```powershell
.\.venv_nitro\Scripts\python.exe -m pytest -q src\\tests\\test_transplant_recipient.py
```

Expected: `TwoDonorRecipient` import failure.

- [ ] **Step 3: Implement the recipient without changing GPT module code**

Deep-copy the verified language donor. For GABA injection, execute the copied
GPT-2 transformer explicitly using the same embeddings, dropout, block calls
and final layer norm as `GPT2Model.forward`, injecting each GABA slot after
its mapped block. Request hidden states for the JEPA sidecar and apply the
unchanged copied `lm_head`.

The output contract is:

```python
@dataclass
class RecipientOutput:
    logits: torch.Tensor
    jepa_loss: torch.Tensor | None
    organ_observations: dict[str, Any]
    hidden_states: tuple[torch.Tensor, ...] | None
```

Supported modes are `disabled`, `shadow`, `adapter_active` and
`organ_unfrozen`. `disabled` must use the unmodified GPT forward directly,
which is the identity authority. `shadow` executes organs but returns the
unmodified GPT logits.

- [ ] **Step 4: Verify exact GPT compatibility on multiple inputs**

Run:

```powershell
.\.venv_nitro\Scripts\python.exe -m pytest -q src\\tests\\test_transplant_recipient.py
```

Expected: zero failures, including `1e-5` identity on at least three sequence
lengths.

- [ ] **Step 5: Commit**

```powershell
git add -- src/f51_darwin/transplant/recipient.py src/tests/test_transplant_recipient.py
git commit -m "feat(transplant): build exact two-donor recipient"
```

---

### Task 5: Lifecycle and declared gradient boundaries

**Files:**
- Create: `src/f51_darwin/transplant/lifecycle.py`
- Create: `src/tests/test_transplant_lifecycle.py`

**Interfaces:**
- Consumes: `TwoDonorRecipient`.
- Produces: `OrganState`, `configure_trainable_state()`, `audit_gradients()`, `transition_organ()`.

- [ ] **Step 1: Write failing gradient-boundary tests**

```python
def test_adapter_active_exposes_only_one_organs_adapters(recipient) -> None:
    declared = configure_trainable_state(
        recipient,
        organ_name="gaba.0",
        state=OrganState.ADAPTER_ACTIVE,
    )
    assert declared
    assert all(
        parameter.requires_grad == (name in declared)
        for name, parameter in recipient.named_parameters()
    )


def test_organ_unfrozen_does_not_unfreeze_gpt_or_other_organs(recipient) -> None:
    declared = configure_trainable_state(
        recipient,
        organ_name="jepa",
        state=OrganState.ORGAN_UNFROZEN,
    )
    assert any(name.startswith("jepa_slot.organ.") for name in declared)
    assert not any(name.startswith("language_model.") for name in declared)
    assert not any("gaba_slots" in name for name in declared)
```

Add a test that injects a gradient into an undeclared parameter and requires
`audit_gradients()` to raise `RuntimeError("undeclared gradient")`.

- [ ] **Step 2: Run RED**

Run:

```powershell
.\.venv_nitro\Scripts\python.exe -m pytest -q src\\tests\\test_transplant_lifecycle.py
```

Expected: import failure.

- [ ] **Step 3: Implement the finite-state lifecycle**

```python
class OrganState(str, Enum):
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    ADAPTER_ACTIVE = "adapter_active"
    ORGAN_UNFROZEN = "organ_unfrozen"
    ACTIVE = "active"
    FROZEN = "frozen"
    QUARANTINE = "quarantine"
```

Define an explicit transition table. `configure_trainable_state()` first sets
every recipient parameter to `requires_grad=False`, then enables exact names
for the selected organ. It returns a frozen set of declared names.
`audit_gradients()` rejects finite violations and any non-`None` gradient
outside that set.

- [ ] **Step 4: Run GREEN**

Run:

```powershell
.\.venv_nitro\Scripts\python.exe -m pytest -q src\\tests\\test_transplant_lifecycle.py src\\tests\\test_transplant_recipient.py
```

Expected: zero failures.

- [ ] **Step 5: Commit**

```powershell
git add -- src/f51_darwin/transplant/lifecycle.py src/tests/test_transplant_lifecycle.py
git commit -m "feat(transplant): enforce per-organ lifecycle"
```

---

### Task 6: Matched A/B/C/D harness and organ ledger

**Files:**
- Create: `src/f51_darwin/transplant/ledger.py`
- Create: `src/f51_darwin/transplant/experiment.py`
- Create: `src/tests/test_transplant_ledger.py`

**Interfaces:**
- Consumes: recipient factories, locked train/holdout batches and `OrganState`.
- Produces: `ArmBudget`, `ArmResult`, `OrganEvidence`, `OrganLedger`, `validate_matched_arms()` and `organ_utility()`.

- [ ] **Step 1: Write failing budget and utility tests**

```python
def test_matched_arms_reject_different_tokens_or_seeds() -> None:
    arms = {
        "A": ArmBudget(tokens=4096, steps=8, seed=7, starts=(0, 128)),
        "B": ArmBudget(tokens=4096, steps=8, seed=7, starts=(0, 128)),
        "C": ArmBudget(tokens=4096, steps=8, seed=8, starts=(0, 128)),
        "D": ArmBudget(tokens=4096, steps=8, seed=7, starts=(0, 128)),
    }
    with pytest.raises(ValueError, match="matched-arm budget mismatch"):
        validate_matched_arms(arms)


def test_useful_and_harmful_organs_receive_opposite_utility() -> None:
    useful = OrganEvidence(
        control_nll=3.0,
        living_nll=2.8,
        future_gain=0.1,
        forgetting=0.01,
        compute_cost=0.02,
        instability=0.0,
    )
    harmful = replace(useful, living_nll=3.3, future_gain=0.0)
    assert organ_utility(useful) > 0
    assert organ_utility(harmful) < 0
```

Add a test that reconstructs a ledger from JSONL and gets identical evidence
hashes.

- [ ] **Step 2: Run RED**

Run:

```powershell
.\.venv_nitro\Scripts\python.exe -m pytest -q src\\tests\\test_transplant_ledger.py
```

Expected: imports fail.

- [ ] **Step 3: Implement immutable evidence and formula**

Use the approved formula:

```python
def organ_utility(evidence: OrganEvidence) -> float:
    quality_gain = evidence.control_nll - evidence.living_nll
    return (
        quality_gain
        + evidence.future_gain
        + evidence.compute_saving
        - evidence.forgetting
        - evidence.compute_cost
        - evidence.instability
    )
```

Each JSONL record includes organ identity, both donor identities, recipient
parent, arm, budget identity, raw metrics, utility, lifecycle decision and a
SHA-256 evidence ID. Use atomic append with flush and `os.fsync`.

- [ ] **Step 4: Implement matched-arm construction**

`build_first_slice_arms()` returns independent recipients:

- A: immutable GPT, no optimizer;
- B: trained GPT control, organ compute executed with effects and gradients blocked;
- C: frozen donor organ plus trainable adapters;
- D: selectively unfrozen donor organ.

Each arm starts from the same language-donor weights and uses the same batch
indices. Reject shared parameter storage between arms.

- [ ] **Step 5: Run GREEN**

Run:

```powershell
.\.venv_nitro\Scripts\python.exe -m pytest -q src\\tests\\test_transplant_ledger.py
```

Expected: zero failures.

- [ ] **Step 6: Commit**

```powershell
git add -- src/f51_darwin/transplant/ledger.py src/f51_darwin/transplant/experiment.py src/tests/test_transplant_ledger.py
git commit -m "feat(transplant): measure matched organ utility"
```

---

### Task 7: Recipient checkpoint identity and per-organ rollback

**Files:**
- Create: `src/f51_darwin/transplant/checkpoint.py`
- Create: `src/tests/test_transplant_checkpoint.py`

**Interfaces:**
- Consumes: `TwoDonorRecipient`, donor identities, ledger hash and lifecycle states.
- Produces: `save_recipient_checkpoint()`, `load_recipient_checkpoint()`, `save_organ_snapshot()` and `rollback_organ()`.

- [ ] **Step 1: Write failing identity and rollback tests**

```python
def test_checkpoint_rejects_wrong_language_or_organ_donor(tmp_path, recipient) -> None:
    path = tmp_path / "recipient.pt"
    save_recipient_checkpoint(
        recipient,
        path,
        language_donor_sha256="a" * 64,
        organ_donor_sha256="b" * 64,
        parent_checkpoint_sha256=None,
        ledger_sha256="c" * 64,
        organ_states={"jepa": "candidate", "gaba.0": "candidate"},
    )
    with pytest.raises(ValueError, match="language donor identity mismatch"):
        load_recipient_checkpoint(
            recipient,
            path,
            expected_language_donor_sha256="d" * 64,
            expected_organ_donor_sha256="b" * 64,
        )


def test_per_organ_rollback_restores_logits_and_state(tmp_path, recipient, ids) -> None:
    snapshot = save_organ_snapshot(recipient, "gaba.0", tmp_path / "gaba.pt")
    with torch.no_grad():
        recipient.organ_slot("gaba.0").external_gate.fill_(0.7)
    rollback_organ(recipient, snapshot)
    assert recipient.organ_slot("gaba.0").external_gate.item() == 0.0
```

- [ ] **Step 2: Run RED**

Run:

```powershell
.\.venv_nitro\Scripts\python.exe -m pytest -q src\\tests\\test_transplant_checkpoint.py
```

Expected: import failure.

- [ ] **Step 3: Implement atomic recipient checkpoints**

Save a plain `weights_only=True`-compatible payload to a PID-suffixed temporary
path, flush it, then `os.replace`. Metadata must include schema version, both
donor hashes, recipient config identity, parent hash, ledger hash, organ states
and declared optimizer parameter names.

Organ snapshots include only one organ's module, adapters, gate and persistent
buffers. `rollback_organ()` must strict-load them and verify the snapshot hash
before mutation. `TwoDonorRecipient.organ_slot(logical_name)` maps logical
names such as `gaba.0` to PyTorch-safe module keys such as `gaba_0`; dots are
never used directly as `ModuleDict` keys.

- [ ] **Step 4: Run GREEN**

Run:

```powershell
.\.venv_nitro\Scripts\python.exe -m pytest -q src\\tests\\test_transplant_checkpoint.py src\\tests\\test_transplant_lifecycle.py
```

Expected: zero failures.

- [ ] **Step 5: Commit**

```powershell
git add -- src/f51_darwin/transplant/checkpoint.py src/tests/test_transplant_checkpoint.py
git commit -m "feat(transplant): checkpoint and rollback each organ"
```

---

### Task 8: Real-donor canary, launcher and first measurements

**Files:**
- Create: `src/scripts/run_two_donor_canary.py`
- Create: `src/scripts/start_two_donor_transplant.ps1`
- Create: `src/tests/test_transplant_launcher.py`
- Modify: `governance/docs/operacao/STATUS_ATUAL.md`
- Create: `workspace/runtime/history/agent_bus/2026-07-28-two-donor-transplant-canary.md`

**Interfaces:**
- Consumes: all first-slice modules, verified donors and local corpus.
- Produces: ignored organ bundle, isolated recipient checkpoints, A/B/C/D metrics and a non-launching `-Canary` gate.

- [ ] **Step 1: Write failing launcher contract test**

```python
def test_two_donor_launcher_isolated_and_canary_does_not_launch() -> None:
    text = Path("src/scripts/start_two_donor_transplant.ps1").read_text("utf-8")
    assert "[switch]$Canary" in text
    assert "03_CHECKPOINTS_DARWIN_TWO_DONOR_V1" in text
    assert "organism_cycle_002_step_001750.pt" in text
    assert "donor_manifest.json" in text
    assert "Get-CimInstance Win32_Process" in text
    assert "if ($Canary)" in text
    assert "Start-Process" in text
```

- [ ] **Step 2: Run RED**

Run:

```powershell
.\.venv_nitro\Scripts\python.exe -m pytest -q src\\tests\\test_transplant_launcher.py
```

Expected: missing launcher failure.

- [ ] **Step 3: Implement the canary runner**

`src/scripts/run_two_donor_canary.py` must:

1. verify Donor A manifest and Donor B SHA-256;
2. load or extract the first-slice bundle;
3. reconstruct original JEPA and GABA 0 strictly;
4. build the recipient with zero gates;
5. compare simple-GPT and disabled-recipient logits;
6. run shadow forwards and reject non-finite state;
7. run one adapter-only step and audit gradients;
8. run one single-organ step and audit gradients;
9. execute three deterministic engineering seeds for A/B/C/D;
10. save JSON metrics and an organ ledger without publishing.

- [ ] **Step 4: Implement the guarded PowerShell launcher**

`-Canary` runs donor verification, all `src/tests/test_transplant_*.py`, a CPU
bundle smoke and exits before `Start-Process`. The launch path rejects any
active `train_darwin_transfer.py`, `darwin_organism.py` or
`run_two_donor_canary.py` worker and writes PID/stdout/stderr under
`workspace/runtime/darwin_two_donor_v1`.

- [ ] **Step 5: Run the full first-slice source audit**

Run:

```powershell
$env:PYTHONPATH = (Resolve-Path workspace\runtime\transfer_python)
.\.venv_nitro\Scripts\python.exe -m pytest -q -p no:cacheprovider (Get-ChildItem src\\tests\\test_transplant_*.py | Select-Object -ExpandProperty FullName)
powershell -ExecutionPolicy Bypass -File src\\scripts\\start_two_donor_transplant.ps1 -Canary
git diff --check
```

Expected: zero test failures, `CANARY_OK launch=false`, clean diff.

- [ ] **Step 6: Run the real extraction and bounded GPU canary**

Resolve and record the full Donor B SHA-256, then run:

```powershell
powershell -ExecutionPolicy Bypass -File src\\scripts\\start_two_donor_transplant.ps1
```

Verify directly:

- exactly one worker;
- GPU allocation belongs to its PID;
- bundle tensor hashes pass;
- zero-gate GPT equivalence passes;
- one JEPA and one GABA experiment produce finite A/B/C/D records;
- no existing checkpoint root changes.

- [ ] **Step 7: Record the honest result**

Update `governance/docs/operacao/STATUS_ATUAL.md` and the ignored history note with donor
hashes, run ID, PID, bundle counts, equivalence error, gradient audit,
A/B/C/D metrics and every failed gate. Do not call a negative or inconclusive
utility an organ gain.

- [ ] **Step 8: Commit the supported first slice**

```powershell
git add -- src/scripts/run_two_donor_canary.py src/scripts/start_two_donor_transplant.ps1 src/tests/test_transplant_launcher.py governance/docs/operacao/STATUS_ATUAL.md
git commit -m "feat(transplant): validate first two-donor recipient"
```

---

## Final Verification

- [ ] Run all `src/tests/test_transplant_*.py` with zero failures.
- [ ] Run existing transfer, GABA, JEPA and checkpoint regression tests.
- [ ] Run the non-launching PowerShell canary.
- [ ] Recompute both donor identities.
- [ ] Reload the recipient from its checkpoint in a fresh process.
- [ ] Confirm disabled-recipient logits match simple GPT within `1e-5`.
- [ ] Confirm no undeclared gradients in adapter-only and single-organ phases.
- [ ] Confirm organ rollback reproduces pre-update logits and buffers.
- [ ] Confirm A/B/C/D budgets and starts are identical.
- [ ] Confirm the ledger reconstructs from immutable JSONL.
- [ ] Inspect Git status and preserve all unrelated untracked files.
- [ ] Report pre-existing full-suite failures separately rather than masking them.
