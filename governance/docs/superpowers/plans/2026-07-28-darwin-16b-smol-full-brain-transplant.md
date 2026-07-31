# Darwin 1.6B Smol Full-Brain Transplant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and prove one donor-free native `DarwinXModel` checkpoint whose complete language core is derived from SmolLM2-1.7B-Instruct and whose 1920d organs are grown from the valid Darwin 100M V3 specimen.

**Architecture:** A hash-bound, resumable surgery pipeline reads the immutable Smol safetensor and V3 organ checkpoint, derives one coherent residual projection, maps all donor tensor families into the native Darwin attention/SSD/MoE anatomy, grows each organ without runtime adapters, and writes an isolated v9 checkpoint. Construction, structural proof, bounded calibration, knowledge evaluation, and publication are separate fail-closed actions.

**Tech Stack:** Python 3.12, PyTorch 2.13, NumPy 2.5.1, PyYAML 6.0.3, safetensors 0.8.0, tokenizers 0.22.2, Transformers 5.14.1 for donor-side calibration only, pytest 9.1.1, PowerShell 7/Windows.

## Global Constraints

- Model name: `F51-Darwin-X-1.6B-Smol-Transplant-V1`.
- Language donor snapshot: `31b70e2e869a7173562077fd711b654946d38674`.
- Language donor model hash: `f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8`.
- Organ donor hash: `71c49bc295c0d06d72b3dc640d5b4d846f421ecdcca25820517fffaab6425a30`.
- Target anatomy is exactly 49,152 vocabulary, 1,920 hidden width, 16 layers, 30 attention heads, 30 KV heads, 64 head dimension, attention at blocks 3/7/11/15, 14 fine experts, 2 shared experts, 896 expert width, and top-2 routing.
- Target roots are `workspace/03_CHECKPOINTS_1.6B_SMOL_TRANSPLANT_V1`, `workspace/runtime/darwin_16b_smol_transplant_v1`, and `workspace/01_TOKENIZER/smol_49152_transplant_v1`.
- The canonical 100M, 600M, and 1.6B roots are read-only inputs and must be unchanged after every action.
- `research/compress_smol_to_darwin.py` and `workspace/00_DONORS/compressed_smol_to_darwin_100m.pt` remain historical evidence and are never used as publication inputs.
- Every donor tensor is classified and every target language tensor is donor-derived; unclassified, silently dropped, or random language tensors fail construction.
- The same semi-orthogonal `P: R^2048 -> R^1920` is used across the full language residual space.
- Organ growth uses one semi-orthogonal `Q: R^512 -> R^1920`; runtime dimension adapters are forbidden.
- The exact Smol tokenizer contract is BOS 1, EOS 2, PAD 2, vocabulary 49,152.
- No long training starts from `plan`, `build`, or `-Canary`.
- Calibration requires explicit `-Calibrate`; publication requires explicit `-PublishCandidate`.
- First boot keeps GABA/Spider/TTM/DAE in shadow, JEPA train-only, Heartbeat observer-only, IHS disabled with zero gate, MTP/Ghost loss disabled, and Sleep/Decision Engine/Unified Mesh control-only.
- Native proof must run without network, donor path access, or construction of a Transformers model and print `DARWIN_16B_SMOL_NATIVE_OK donor_loaded=false tokenizer=verified organs=verified`.
- Failing a knowledge gate labels the artifact `engineering_transplant_only`; it is not published as a knowledge transplant.
- Every output is temporary, flushed, hashed, then atomically promoted; published artifacts are never overwritten.

---

## Final objective and readiness contract

1. **What it does:** converts two immutable source specimens into one native Darwin 1.6B organism.
2. **Who it is for:** Marco/Fuch F51 Labs for inspection, demonstration, further training, and neural-surgery research.
3. **Main flow:** `plan` -> `build` -> structural canary -> explicit calibration -> three-seed evaluation -> explicit publication -> native inference.
4. **Start command:** `powershell -ExecutionPolicy Bypass -File src/scripts/start_smol_darwin_transplant.ps1 -Canary`.
5. **Proof:** strict checkpoint load, finite forward/backward, complete ledgers, knowledge-gate report, and the exact donor-free success line.
6. **Current blockers:** adapted-init is rejected, bootstrap hardcodes `F51BPETokenizer`, no coherent surgery package exists, organs are still 512d, and no isolated checkpoint has passed the engineering or knowledge gates.

## File and responsibility map

| Path | Responsibility |
|---|---|
| `src/f51_darwin/transplant_16b/contracts.py` | Immutable identities, plan schema, statuses, canonical JSON hashes |
| `src/f51_darwin/transplant_16b/sources.py` | Source hash verification, safetensor inventory, immutable donor access |
| `src/f51_darwin/transplant_16b/tokenizer.py` | Exact Smol tokenizer adapter and identity |
| `src/f51_darwin/transplant_16b/ledger.py` | Append-only tensor coverage and atomic resume head |
| `src/f51_darwin/transplant_16b/projection.py` | Global `P`, coupled matrix projection, norm projection |
| `src/f51_darwin/transplant_16b/layers.py` | 24-to-16 grouping and 32-to-30 attention-head fit |
| `src/f51_darwin/transplant_16b/moe.py` | Coupled MLP-neuron clustering, experts, routers |
| `src/f51_darwin/transplant_16b/ssd_fit.py` | Donor-transition capture contract and bounded SSD fitting |
| `src/f51_darwin/transplant_16b/organs.py` | `Q` derivation, organ tensor/runtime-state growth, subspace proof |
| `src/f51_darwin/transplant_16b/checkpoint.py` | Streaming target state, v9 payload, shard hashes, atomic publication |
| `src/f51_darwin/transplant_16b/calibration.py` | Layerwise and logit calibration with explicit seed manifests |
| `src/f51_darwin/transplant_16b/verification.py` | Engineering, knowledge, organ, source-preservation gates |
| `src/f51_darwin/transplant_16b/cli.py` | `plan`, `build`, `calibrate`, `verify`, `publish` orchestration |
| `src/scripts/transplant_smol_to_darwin_1_6b.py` | Thin Python entrypoint |
| `src/scripts/start_smol_darwin_transplant.ps1` | Fail-closed operator launcher |

No new transplant logic goes into `research/compress_smol_to_darwin.py`,
`src/scripts/darwin_organism.py`, or the existing checkpoint roots.

### Task 1: Establish the adapted lineage and dependency contract

**Files:**
- Create: `src/configs/darwin_x_1.6b_smol_transplant.yaml`
- Modify: `src/f51_darwin/darwin_x_core/config.py:15-145`
- Modify: `pyproject.toml:15-31`
- Test: `src/tests/test_transplant_16b_config.py`

**Interfaces:**
- Produces: `DarwinXConfig(init="f51_adapted")` and the exact target anatomy.
- Produces: optional dependency group `transplant`.
- Consumes: no transplant package code.

- [ ] **Step 1: Write the failing config tests**

```python
from pathlib import Path
import yaml
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.estimation import estimate_darwin_x_parameters

ROOT = Path(__file__).resolve().parents[1]

def test_adapted_init_is_explicitly_allowed() -> None:
    assert DarwinXConfig(init="f51_adapted").init == "f51_adapted"

def test_unknown_init_is_rejected() -> None:
    try:
        DarwinXConfig(init="smol")
    except ValueError as exc:
        assert "random or f51_adapted" in str(exc)
    else:
        raise AssertionError("unknown init accepted")

def test_smol_target_anatomy_is_exact() -> None:
    raw = yaml.safe_load(
        (ROOT / "src/configs/darwin_x_1.6b_smol_transplant.yaml").read_text("utf-8")
    )
    cfg = DarwinXConfig.from_mapping(raw)
    assert (cfg.vocab_size, cfg.d_model, cfg.n_layers) == (49_152, 1_920, 16)
    assert (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim) == (30, 30, 64)
    assert cfg.attention_layer_indices == (3, 7, 11, 15)
    assert (cfg.fine_experts, cfg.shared_experts, cfg.fine_expert_hidden_dim) == (14, 2, 896)
    assert cfg.qkv_bias is False
    assert cfg.rope_base_train == cfg.rope_base_infer == 130_000.0
    assert estimate_darwin_x_parameters(cfg)["total"] == 1_674_720_224
```

- [ ] **Step 2: Run the tests and verify the intended failures**

Run: `python -m pytest src/tests/test_transplant_16b_config.py -q`

Expected: FAIL because `f51_adapted` is rejected and the YAML does not exist.

- [ ] **Step 3: Add the exact config and narrow init validation**

Change the guard to:

```python
if self.init not in {"random", "f51_adapted"}:
    raise ValueError("Darwin-X init must be random or f51_adapted.")
```

Create the YAML from the approved target table, including:

```yaml
model_name: F51-Darwin-X-1.6B-Smol-Transplant-V1
init: f51_adapted
tokenizer: smol_49152_transplant_v1
vocab_size: 49152
context_length: 4096
inference_context_length: 8192
d_model: 1920
n_layers: 16
n_heads: 30
n_kv_heads: 30
ssd_attention_ratio: "3:1"
qkv_bias: false
rope_base_train: 130000.0
rope_base_infer: 130000.0
fine_experts: 14
shared_experts: 2
experts_per_token: 2
fine_expert_hidden_dim: 896
shared_expert_hidden_dim: 896
loss_semantics_version: 2
gaba_enabled: true
heartbeat_enabled: true
spider_sense_enabled: true
ttm_residual_enabled: true
ttm_residual_max_scale: 0.0
spider_calibration_enabled: true
spider_calibration_weight: 0.0
jepa_weight: 0.0
mtp_weight: 0.0
ghost_weight: 0.0
dae_enabled: true
dae_shadow_mode: true
inter_hemispheric_enabled: true
sleep_enabled: true
decision_engine_enabled: true
unified_mesh_enabled: true
checkpoint_root: workspace/03_CHECKPOINTS_1.6B_SMOL_TRANSPLANT_V1
```

Add this exact optional group:

```toml
transplant = [
  "safetensors==0.8.0",
  "tokenizers==0.22.2",
  "transformers==5.14.1",
]
```

- [ ] **Step 4: Run focused and regression tests**

Run: `python -m pytest src/tests/test_transplant_16b_config.py src/tests/test_darwin_x_public_contract.py src/tests/test_checkpoint_root_policy.py -q`

Expected: PASS; existing random-init identities remain unchanged.

- [ ] **Step 5: Commit**

```powershell
git add src/configs/darwin_x_1.6b_smol_transplant.yaml src/f51_darwin/darwin_x_core/config.py pyproject.toml src/tests/test_transplant_16b_config.py
git commit -m "feat(transplant): define isolated adapted 1.6b lineage"
```

### Task 2: Add immutable source and plan contracts

**Files:**
- Create: `src/f51_darwin/transplant_16b/__init__.py`
- Create: `src/f51_darwin/transplant_16b/contracts.py`
- Create: `src/f51_darwin/transplant_16b/sources.py`
- Test: `src/tests/test_transplant_16b_sources.py`

**Interfaces:**
- Produces: `SourceIdentity`, `TargetAnatomy`, `TransplantPlan`, `verify_sources()`, `inventory_safetensor()`.
- Consumes: target config from Task 1.

- [ ] **Step 1: Write failing contract and tamper tests**

```python
import json
from pathlib import Path
import pytest
from f51_darwin.transplant_16b.contracts import SourceFile, SourceIdentity, TransplantPlan
from f51_darwin.transplant_16b.sources import sha256_file, verify_source_file

def test_source_file_fails_closed_after_tamper(tmp_path: Path) -> None:
    path = tmp_path / "donor.bin"
    path.write_bytes(b"valid")
    source = SourceFile(path=str(path), sha256=sha256_file(path), size_bytes=5)
    verify_source_file(source)
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="source identity mismatch"):
        verify_source_file(source)

def test_plan_hash_is_canonical(tmp_path: Path) -> None:
    plan = TransplantPlan.testing(tmp_path)
    first = plan.identity()
    payload = json.loads(plan.to_json())
    assert TransplantPlan.from_mapping(payload).identity() == first
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest src/tests/test_transplant_16b_sources.py -q`

Expected: FAIL with `ModuleNotFoundError: f51_darwin.transplant_16b`.

- [ ] **Step 3: Implement frozen schemas and hash verification**

Use frozen dataclasses with these exact public fields:

```python
@dataclass(frozen=True)
class SourceFile:
    path: str
    sha256: str
    size_bytes: int

@dataclass(frozen=True)
class SourceIdentity:
    smol_snapshot: str
    smol_config: SourceFile
    smol_weights: SourceFile
    tokenizer_json: SourceFile
    tokenizer_config: SourceFile
    special_tokens: SourceFile
    organ_checkpoint: SourceFile

@dataclass(frozen=True)
class TargetAnatomy:
    model_name: str
    config_identity: str
    checkpoint_root: str
    runtime_root: str
    tokenizer_root: str

@dataclass(frozen=True)
class TransplantPlan:
    schema: str
    implementation_version: str
    sources: SourceIdentity
    target: TargetAnatomy
    calibration_digest: str
    projection_seed: int
    layer_groups: tuple[tuple[int, ...], ...]
    attention_blocks: tuple[int, ...]
```

`identity()` hashes canonical UTF-8 JSON with sorted keys and compact
separators. `verify_source_file()` compares resolved path, exact size, and a
streamed SHA-256; `inventory_safetensor()` uses `safe_open(..., device="cpu")`
and returns sorted `(name, shape, dtype, numel)` records without materializing
the complete donor.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest src/tests/test_transplant_16b_sources.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/f51_darwin/transplant_16b src/tests/test_transplant_16b_sources.py
git commit -m "feat(transplant): bind surgery to immutable source identities"
```

### Task 3: Add exact Smol tokenizer loading and organism bootstrap routing

**Files:**
- Create: `src/f51_darwin/transplant_16b/tokenizer.py`
- Modify: `src/f51_darwin/state_identity.py:98-127`
- Modify: `src/f51_darwin/organism/bootstrap.py:207-214`
- Test: `src/tests/test_transplant_16b_tokenizer.py`

**Interfaces:**
- Produces: `SmolTokenizerAdapter.load(path, expected_hashes)`.
- Produces: generic `tokenizer_identity()` dispatch through `identity_payload()`.
- Consumes: `SourceIdentity` from Task 2.

- [ ] **Step 1: Write the failing adapter tests**

```python
from pathlib import Path
import pytest
from f51_darwin.state_identity import tokenizer_identity
from f51_darwin.transplant_16b.tokenizer import SmolTokenizerAdapter

def test_exact_smol_special_ids_and_round_trip(smol_tokenizer_dir: Path) -> None:
    tok = SmolTokenizerAdapter.load(smol_tokenizer_dir)
    assert (tok.vocab_size, tok.bos_id, tok.eos_id, tok.pad_id) == (49_152, 1, 2, 2)
    ids = tok.encode("Darwin aprende.", add_bos=True, add_eos=True)
    assert ids[0] == 1 and ids[-1] == 2
    assert "Darwin" in tok.decode(ids)
    assert tokenizer_identity(tok).startswith("smol-tokenizer-contract-v1:")

def test_f51_tokenizer_is_rejected_for_smol_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Smol tokenizer"):
        SmolTokenizerAdapter.load(tmp_path)
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest src/tests/test_transplant_16b_tokenizer.py -q`

Expected: FAIL because the adapter is absent.

- [ ] **Step 3: Implement the protocol and routing**

The adapter wraps `tokenizers.Tokenizer.from_file(tokenizer.json)`, exposes
`vocab_size`, `bos_id`, `eos_id`, `pad_id`, `unk_id`, `encode()`, `decode()`,
and returns this identity payload:

```python
def identity_payload(self) -> dict[str, object]:
    return {
        "schema": "smol-tokenizer-contract-v1",
        "tokenizer_json_sha256": self.tokenizer_json_sha256,
        "tokenizer_config_sha256": self.tokenizer_config_sha256,
        "special_tokens_sha256": self.special_tokens_sha256,
        "vocab_size": self.vocab_size,
        "special_ids": {"bos": 1, "eos": 2, "pad": 2},
    }
```

Keep the existing F51 hash byte-identical by dispatching only when
`identity_payload` exists. In bootstrap, load `SmolTokenizerAdapter` only when
`model_config.tokenizer == "smol_49152_transplant_v1"`; otherwise retain
`F51BPETokenizer.load()`.

- [ ] **Step 4: Run tokenizer and identity regressions**

Run: `python -m pytest src/tests/test_transplant_16b_tokenizer.py src/tests/test_state_identity.py -q`

Expected: PASS, including the historical F51 BPE identity test.

- [ ] **Step 5: Commit**

```powershell
git add src/f51_darwin/transplant_16b/tokenizer.py src/f51_darwin/state_identity.py src/f51_darwin/organism/bootstrap.py src/tests/test_transplant_16b_tokenizer.py
git commit -m "feat(transplant): load exact Smol tokenizer in native runtime"
```

### Task 4: Make planning and coverage append-only and resumable

**Files:**
- Create: `src/f51_darwin/transplant_16b/ledger.py`
- Test: `src/tests/test_transplant_16b_ledger.py`

**Interfaces:**
- Produces: `CoverageRecord`, `CoverageLedger.open()`, `.append()`, `.head`, `.assert_complete()`.
- Consumes: plan identity from Task 2.

- [ ] **Step 1: Write failing ledger tests**

```python
from pathlib import Path
import pytest
from f51_darwin.transplant_16b.ledger import CoverageLedger, CoverageRecord

def test_ledger_resume_requires_same_plan_and_head(tmp_path: Path) -> None:
    ledger = CoverageLedger.open(tmp_path / "coverage.jsonl", plan_id="a" * 64)
    ledger.append(CoverageRecord("donor.x", ("target.x",), "projection", "complete", {}))
    resumed = CoverageLedger.open(tmp_path / "coverage.jsonl", plan_id="a" * 64)
    assert resumed.head == ledger.head
    with pytest.raises(ValueError, match="plan identity"):
        CoverageLedger.open(tmp_path / "coverage.jsonl", plan_id="b" * 64)

def test_complete_rejects_unclassified_and_random_targets(tmp_path: Path) -> None:
    ledger = CoverageLedger.open(tmp_path / "coverage.jsonl", plan_id="a" * 64)
    with pytest.raises(ValueError, match="donor coverage"):
        ledger.assert_complete({"donor.a"}, {"target.a"})
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest src/tests/test_transplant_16b_ledger.py -q`

Expected: FAIL because `ledger.py` is absent.

- [ ] **Step 3: Implement hash-chained JSONL**

Each line contains `sequence`, `plan_id`, `previous`, `record`, and `digest`.
Open validates the chain from byte zero. Append serializes one canonical line,
flushes, calls `os.fsync()`, and only then updates the in-memory head.
`assert_complete()` requires exact donor-name equality and exact target-name
equality; allowed statuses are `complete`, `folded`, and `classified_nonweight`.

- [ ] **Step 4: Run ledger tests**

Run: `python -m pytest src/tests/test_transplant_16b_ledger.py -q`

Expected: PASS, including rejection after changing one byte in the first line.

- [ ] **Step 5: Commit**

```powershell
git add src/f51_darwin/transplant_16b/ledger.py src/tests/test_transplant_16b_ledger.py
git commit -m "feat(transplant): add hash chained tensor coverage ledger"
```

### Task 5: Implement coherent residual projection and layer/head mapping

**Files:**
- Create: `src/f51_darwin/transplant_16b/projection.py`
- Create: `src/f51_darwin/transplant_16b/layers.py`
- Test: `src/tests/test_transplant_16b_projection.py`
- Test: `src/tests/test_transplant_16b_layers.py`

**Interfaces:**
- Produces: `HiddenProjection`, `derive_hidden_projection()`, `project_linear()`, `project_norm()`.
- Produces: `monotonic_layer_groups()`, `select_and_fold_heads()`.
- Consumes: donor activation samples shaped `[samples, 2048]`.

- [ ] **Step 1: Write failing mathematical tests**

```python
import torch
from f51_darwin.transplant_16b.projection import derive_hidden_projection, project_linear
from f51_darwin.transplant_16b.layers import monotonic_layer_groups

def test_projection_is_semi_orthogonal_and_coherent() -> None:
    torch.manual_seed(7)
    embedding = torch.randn(128, 8)
    activations = torch.randn(256, 8)
    projection = derive_hidden_projection(embedding, activations, target_dim=6)
    assert torch.allclose(
        projection.matrix @ projection.matrix.T, torch.eye(6), atol=1e-5
    )
    weight = torch.randn(8, 8)
    expected = projection.matrix @ weight @ projection.matrix.T
    assert torch.allclose(project_linear(weight, projection, projection), expected)

def test_layer_groups_cover_24_monotonically() -> None:
    groups = monotonic_layer_groups(24, 16)
    assert len(groups) == 16
    assert tuple(index for group in groups for index in group) == tuple(range(24))
    assert all(len(group) in {1, 2} for group in groups)
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest src/tests/test_transplant_16b_projection.py src/tests/test_transplant_16b_layers.py -q`

Expected: FAIL because both modules are absent.

- [ ] **Step 3: Implement deterministic projection and attention fitting**

Center the concatenated embedding/activation calibration matrix, run
`torch.linalg.svd(..., full_matrices=False)`, take the first 1920 right-singular
vectors as rows of `P`, and fix every row sign by making its largest-magnitude
element positive. Store singular values, explained energy, calibration digest,
seed, algorithm version, and matrix hash.

For `[out, in]` weights implement:

```python
def project_linear(
    weight: torch.Tensor,
    output_space: HiddenProjection,
    input_space: HiddenProjection,
) -> torch.Tensor:
    return output_space.matrix @ weight.float() @ input_space.matrix.T
```

`select_and_fold_heads()` ranks 32 heads by held-out output ablation energy,
keeps 30, assigns each rejected head to its maximum-cosine kept head, and solves
`torch.linalg.lstsq(selected_outputs, full_output).solution` for the O-projection
fold. Q/K/V use the same selected-head order; all returned head widths are 64.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest src/tests/test_transplant_16b_projection.py src/tests/test_transplant_16b_layers.py -q`

Expected: PASS with deterministic hashes across two identical runs.

- [ ] **Step 5: Commit**

```powershell
git add src/f51_darwin/transplant_16b/projection.py src/f51_darwin/transplant_16b/layers.py src/tests/test_transplant_16b_projection.py src/tests/test_transplant_16b_layers.py
git commit -m "feat(transplant): map Smol residual space and attention coherently"
```

### Task 6: Transplant coupled MLP neurons into native MoE

**Files:**
- Create: `src/f51_darwin/transplant_16b/moe.py`
- Test: `src/tests/test_transplant_16b_moe.py`

**Interfaces:**
- Produces: `CoupledMLP`, `ExpertAssignment`, `cluster_coupled_neurons()`, `build_expert_state()`, `fit_router()`.
- Consumes: layer groups and `HiddenProjection` from Task 5.

- [ ] **Step 1: Write failing coupling and coverage tests**

```python
import torch
from f51_darwin.transplant_16b.moe import CoupledMLP, cluster_coupled_neurons

def test_triplets_never_split_and_every_neuron_is_covered() -> None:
    mlp = CoupledMLP(
        gate=torch.randn(32, 8),
        up=torch.randn(32, 8),
        down=torch.randn(8, 32),
    )
    signatures = torch.randn(64, 32)
    assignment = cluster_coupled_neurons(
        mlp, signatures, shared_experts=2, fine_experts=4, expert_width=6, seed=11
    )
    flattened = [index for group in assignment.source_neurons for index in group]
    assert sorted(flattened) == list(range(32))
    assert len(set(flattened)) == 32
    assert all(count > 0 for count in assignment.target_widths)
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest src/tests/test_transplant_16b_moe.py -q`

Expected: FAIL because `moe.py` is absent.

- [ ] **Step 3: Implement deterministic coupled clustering**

Compute per-neuron signatures from `silu(gate(x)) * up(x)`, mark the top
ubiquity quantiles for exactly two shared experts, then run seeded cosine
k-means for 14 fine experts. Reduce each cluster to width 896 with one SVD basis
applied jointly to gate rows, up rows, and the matching down columns. Empty
clusters fail. Router rows are normalized activation centroids and are refined
with cross-entropy against ownership labels. Emit source-neuron indices,
reconstruction MSE, cluster counts, and dead-expert rate to the coverage ledger.

- [ ] **Step 4: Run the MoE tests**

Run: `python -m pytest src/tests/test_transplant_16b_moe.py -q`

Expected: PASS; deliberately permuting one down column makes the coupling test fail.

- [ ] **Step 5: Commit**

```powershell
git add src/f51_darwin/transplant_16b/moe.py src/tests/test_transplant_16b_moe.py
git commit -m "feat(transplant): convert coupled Smol MLP neurons into Darwin MoE"
```

### Task 7: Fit native SSD blocks from donor transitions

**Files:**
- Create: `src/f51_darwin/transplant_16b/ssd_fit.py`
- Test: `src/tests/test_transplant_16b_ssd_fit.py`

**Interfaces:**
- Produces: `TransitionBatch`, `SSDFitConfig`, `SSDFitReport`, `fit_ssd_block()`.
- Consumes: grouped donor input/output residuals projected to 1920d.

- [ ] **Step 1: Write the failing superiority test**

```python
import torch
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.layers import SSDMixerOnly
from f51_darwin.transplant_16b.ssd_fit import TransitionBatch, SSDFitConfig, fit_ssd_block

def test_fit_beats_paired_random_initialization() -> None:
    torch.manual_seed(19)
    config = DarwinXConfig(
        d_model=16, n_layers=4, n_heads=4, n_kv_heads=4,
        vocab_size=64, fine_experts=2, shared_experts=1,
        experts_per_token=1, fine_expert_hidden_dim=8,
        shared_expert_hidden_dim=8, ssm_state=4, scan_chunk_size=8,
    )
    teacher = SSDMixerOnly(config)
    inputs = torch.randn(8, 12, 16)
    with torch.no_grad():
        targets = teacher(inputs)
    candidate = SSDMixerOnly(config)
    report = fit_ssd_block(
        candidate,
        TransitionBatch(inputs[:6], targets[:6], inputs[6:], targets[6:]),
        SSDFitConfig(steps=40, learning_rate=1e-2, seed=19),
    )
    assert report.holdout_mse < report.random_holdout_mse
    assert report.finite
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest src/tests/test_transplant_16b_ssd_fit.py -q`

Expected: FAIL because `ssd_fit.py` is absent.

- [ ] **Step 3: Implement bounded system identification**

Initialize SSD input/output projections with `P`-projected donor residual maps,
then optimize only that block's convolution/state/projection parameters using
AdamW in FP32 for the configured step budget. Record the paired random seed,
initial MSE, train MSE, holdout MSE, gradient finiteness, and parameter hash.
Abort and do not write the block when holdout MSE is not strictly lower than
the paired random baseline.

- [ ] **Step 4: Run SSD and scan regressions**

Run: `python -m pytest src/tests/test_transplant_16b_ssd_fit.py src/tests/test_transfer_ssd_mixer.py -q`

Expected: PASS with no change to existing selective-scan behavior.

- [ ] **Step 5: Commit**

```powershell
git add src/f51_darwin/transplant_16b/ssd_fit.py src/tests/test_transplant_16b_ssd_fit.py
git commit -m "feat(transplant): identify native SSD blocks from donor transitions"
```

### Task 8: Grow V3 organs into the 1920d organism

**Files:**
- Create: `src/f51_darwin/transplant_16b/organs.py`
- Modify: `src/f51_darwin/darwin_x_core/model.py:1372-1460`
- Test: `src/tests/test_transplant_16b_organs.py`

**Interfaces:**
- Produces: `OrganProjection`, `expand_square()`, `expand_linear()`, `grow_organ_state()`, `verify_organ_subspace()`.
- Consumes: verified V3 checkpoint and target model state names.

- [ ] **Step 1: Write failing subspace and neutral-effect tests**

```python
import torch
from f51_darwin.transplant_16b.organs import (
    derive_organ_projection, expand_square, verify_organ_subspace
)

def test_square_growth_reproduces_source_subspace() -> None:
    torch.manual_seed(23)
    source = torch.randn(8, 8)
    q = derive_organ_projection(source_dim=8, target_dim=16, seed=23)
    grown = expand_square(source, q, complement="zero")
    assert verify_organ_subspace(source, grown, q) >= 0.99

def test_identity_complement_is_identity_off_source_subspace() -> None:
    source = torch.eye(8)
    q = derive_organ_projection(source_dim=8, target_dim=16, seed=29)
    grown = expand_square(source, q, complement="identity")
    residual = torch.eye(16) - q.matrix @ q.matrix.T
    assert torch.allclose(grown @ residual, residual, atol=1e-5)
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest src/tests/test_transplant_16b_organs.py -q`

Expected: FAIL because `organs.py` is absent.

- [ ] **Step 3: Implement one `Q` and explicit organ policies**

Use deterministic QR to derive `Q`; fix column signs. For square matrices:

```python
projector = q.matrix @ q.matrix.T
grown = q.matrix @ source.float() @ q.matrix.T
if complement == "identity":
    grown = grown + torch.eye(q.target_dim) - projector
return grown
```

Build a declarative policy keyed by exact model/checkpoint prefixes for GABA,
JEPA, Spider, TTM, Heartbeat, IHS, and MTP. Copy scalars/counters exactly;
expand every TTM memory vector; reconstruct Ghost/DAE/Sleep/Decision/Unified
Mesh from target config and import only named compatible scalar state. Set
external-effect gates to exact zero and return a per-organ report containing
source keys, target keys, cosine, finiteness, runtime-state imports, and mode.
Make Heartbeat construction deterministic so state loading does not advance it.

- [ ] **Step 4: Run organ and topology tests**

Run: `python -m pytest src/tests/test_transplant_16b_organs.py src/tests/test_causal_cognitive_organs.py src/tests/test_topology_manifest_v7.py -q`

Expected: PASS; every grown learned organ reports cosine at least 0.99.

- [ ] **Step 5: Commit**

```powershell
git add src/f51_darwin/transplant_16b/organs.py src/f51_darwin/darwin_x_core/model.py src/tests/test_transplant_16b_organs.py
git commit -m "feat(transplant): grow V3 organs into native 1920d modules"
```

### Task 9: Write a strict atomic v9 transplant checkpoint

**Files:**
- Create: `src/f51_darwin/transplant_16b/checkpoint.py`
- Modify: `src/f51_darwin/organism/checkpoint_root.py:207-232`
- Modify: `src/scripts/inspect_organism_checkpoint.py:122-181`
- Test: `src/tests/test_transplant_16b_checkpoint.py`

**Interfaces:**
- Produces: `TransplantCheckpointWriter`, `build_v9_payload()`, `verify_shard_manifest()`.
- Consumes: complete language/organ states and coverage head.

- [ ] **Step 1: Write failing atomicity and strict-load tests**

```python
from pathlib import Path
import torch
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.transplant_16b.checkpoint import TransplantCheckpointWriter

def test_writer_round_trip_is_strict_and_tied(tiny_transplant: tuple) -> None:
    model, config, metadata, root = tiny_transplant
    result = TransplantCheckpointWriter(root).write(model, config, metadata)
    payload = torch.load(result.checkpoint, map_location="cpu", weights_only=False)
    restored = DarwinXModel(config)
    restored.load_state_dict(payload["model_state_dict"], strict=True)
    assert restored.lm_head.weight is restored.token_embedding.weight
    assert payload["version"] == 9
    assert payload["transplant"]["coverage_complete"] is True
    assert not list(Path(root).glob("*.tmp"))
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest src/tests/test_transplant_16b_checkpoint.py -q`

Expected: FAIL because `TransplantCheckpointWriter` is absent.

- [ ] **Step 3: Implement v9 payload and lineage identity**

The payload must contain the existing required v9 keys plus:

```python
"transplant": {
    "schema": "darwin-smol-full-brain-v1",
    "plan_id": metadata.plan_id,
    "source_identity": metadata.source_identity,
    "projection_identity": metadata.projection_identity,
    "organ_projection_identity": metadata.organ_projection_identity,
    "coverage_head": metadata.coverage_head,
    "coverage_complete": True,
    "calibration_digest": metadata.calibration_digest,
    "status": "engineering_transplant_only",
}
```

Write `organism_cycle_000.pt.incomplete`, flush and fsync, compute its SHA-256,
write and fsync the manifest, then use `os.replace()` for the checkpoint and
manifest. Set lineage `creation_mode` to `transplant`, include both source
hashes and `plan_id`, and reject any existing published cycle. Extend the
inspector so a transplanted v9 checkpoint is valid only when its transplant
manifest and coverage head are present and hash-consistent.

- [ ] **Step 4: Run checkpoint policy regressions**

Run: `python -m pytest src/tests/test_transplant_16b_checkpoint.py src/tests/test_checkpoint_root_policy.py src/tests/test_canary_launcher_contract.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/f51_darwin/transplant_16b/checkpoint.py src/f51_darwin/organism/checkpoint_root.py src/scripts/inspect_organism_checkpoint.py src/tests/test_transplant_16b_checkpoint.py
git commit -m "feat(transplant): write strict atomic v9 surgery checkpoints"
```

### Task 10: Assemble the streaming `plan` and `build` CLI

**Files:**
- Create: `src/f51_darwin/transplant_16b/cli.py`
- Create: `src/scripts/transplant_smol_to_darwin_1_6b.py`
- Modify: `governance/audit/policy/operational-surface.json`
- Modify: `src/tests/test_operational_surface.py:10-24`
- Test: `src/tests/test_transplant_16b_cli.py`

**Interfaces:**
- Produces: `main(argv: list[str] | None = None) -> int`.
- Consumes: Tasks 2-9.

- [ ] **Step 1: Write failing dry-plan and no-long-training tests**

```python
import json
import sys
from pathlib import Path
from f51_darwin.transplant_16b.cli import main

def test_plan_writes_hash_bound_atomic_plan(fake_sources: dict, tmp_path: Path) -> None:
    rc = main(["plan", "--config", fake_sources["config"], "--runtime-root", str(tmp_path)])
    assert rc == 0
    payload = json.loads((tmp_path / "plan.json").read_text("utf-8"))
    assert payload["schema"] == "darwin-smol-transplant-plan-v1"
    assert payload["target"]["model_name"] == "F51-Darwin-X-1.6B-Smol-Transplant-V1"
    assert "plan_id" in payload

def test_build_does_not_import_calibration(tiny_plan: Path) -> None:
    sys.modules.pop("f51_darwin.transplant_16b.calibration", None)
    assert main(["build", "--plan", str(tiny_plan), "--tiny-test-mode"]) == 0
    assert "f51_darwin.transplant_16b.calibration" not in sys.modules
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest src/tests/test_transplant_16b_cli.py -q`

Expected: FAIL because the CLI is absent.

- [ ] **Step 3: Implement fail-closed commands**

`plan` verifies source/config/tokenizer hashes, hardware/disk, target-root
isolation, fixed calibration digest, layer grouping, and writes `plan.json`
atomically. `build` re-verifies the plan, memory-maps one donor family at a
time, invokes projection/attention/MoE/SSD/organ builders, appends coverage,
writes the strict checkpoint, and never imports `calibration`. Both commands
support `--resume`; resume requires the same plan identity and ledger head.
`--tiny-test-mode` is accepted only when `PYTEST_CURRENT_TEST` is present.

The entrypoint contains only:

```python
from f51_darwin.transplant_16b.cli import main
if __name__ == "__main__":
    raise SystemExit(main())
```

Classify the converter as `maintenance`, add it to
`script_allowed_files`, and keep it out of `supported_entrypoints` because it
constructs artifacts rather than serving the product.

- [ ] **Step 4: Run CLI and operational-surface tests**

Run: `python -m pytest src/tests/test_transplant_16b_cli.py src/tests/test_operational_surface.py -q`

Expected: PASS with every executable classified.

- [ ] **Step 5: Commit**

```powershell
git add src/f51_darwin/transplant_16b/cli.py src/scripts/transplant_smol_to_darwin_1_6b.py governance/audit/policy/operational-surface.json src/tests/test_transplant_16b_cli.py src/tests/test_operational_surface.py
git commit -m "feat(transplant): add resumable streaming surgery commands"
```

### Task 11: Add explicit calibration and three-seed knowledge gates

**Files:**
- Create: `src/f51_darwin/transplant_16b/calibration.py`
- Create: `src/f51_darwin/transplant_16b/verification.py`
- Create: `src/scripts/evaluate_smol_darwin_transplant.py`
- Modify: `governance/audit/policy/operational-surface.json`
- Test: `src/tests/test_transplant_16b_verification.py`

**Interfaces:**
- Produces: `CalibrationManifest`, `calibrate()`, `KnowledgeMetrics`, `evaluate_knowledge_gates()`.
- Consumes: immutable checkpoint, fixed source-stratified holdout, seeds `(17, 29, 43)`.

- [ ] **Step 1: Write failing threshold tests**

```python
from f51_darwin.transplant_16b.verification import KnowledgeMetrics, evaluate_knowledge_gates

def test_all_three_paired_seeds_must_pass() -> None:
    good = KnowledgeMetrics(seed=17, bpb=3.0, random_bpb=3.5, kl=1.0, random_kl=1.5, top1=0.20, random_top1=0.08, generation_valid=True)
    weak = KnowledgeMetrics(seed=29, bpb=3.2, random_bpb=3.5, kl=1.2, random_kl=1.5, top1=0.14, random_top1=0.08, generation_valid=True)
    third = KnowledgeMetrics(seed=43, bpb=3.0, random_bpb=3.5, kl=1.0, random_kl=1.5, top1=0.20, random_top1=0.08, generation_valid=True)
    report = evaluate_knowledge_gates([good, weak, third])
    assert report.passed is False
    assert report.status == "engineering_transplant_only"
    assert "seed=29" in report.failures

def test_exact_thresholds_publishable() -> None:
    rows = [
        KnowledgeMetrics(seed=s, bpb=3.15, random_bpb=3.5, kl=1.125, random_kl=1.5, top1=0.16, random_top1=0.08, generation_valid=True)
        for s in (17, 29, 43)
    ]
    assert evaluate_knowledge_gates(rows).passed is True
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest src/tests/test_transplant_16b_verification.py -q`

Expected: FAIL because verification types are absent.

- [ ] **Step 3: Implement bounded calibration and evaluation**

`calibrate()` requires an explicit command, loads donor and target only for the
current layer/logit shard, disables all organs, trains with teacher KL plus
next-token cross-entropy, and writes append-only seed-specific checkpoints.
The holdout manifest records source strata, byte counts, SHA-256, excluded
training records, tokenizer identity, and prompts. Evaluate paired target and
random-init baselines for seeds 17, 29, and 43. Enforce exactly:

```python
bpb_pass = candidate.bpb <= 0.90 * candidate.random_bpb
kl_pass = candidate.kl <= 0.75 * candidate.random_kl
top1_pass = candidate.top1 >= 2.0 * candidate.random_top1
seed_pass = candidate.bpb < candidate.random_bpb and candidate.kl < candidate.random_kl
```

Require all per-seed conditions and the existing repetition/n-gram generation
gate. The donor-side loader may construct a Transformers model only inside
calibration/evaluation; the module must never be imported by native serving.
Classify the evaluator as `maintenance`.

- [ ] **Step 4: Run verification tests**

Run: `python -m pytest src/tests/test_transplant_16b_verification.py src/tests/test_operational_surface.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/f51_darwin/transplant_16b/calibration.py src/f51_darwin/transplant_16b/verification.py src/scripts/evaluate_smol_darwin_transplant.py governance/audit/policy/operational-surface.json src/tests/test_transplant_16b_verification.py
git commit -m "feat(transplant): gate calibration claims across three paired seeds"
```

### Task 12: Prove donor-free native inference and safe operator actions

**Files:**
- Create: `src/f51_darwin/operations/start_smol_darwin_transplant.ps1`
- Create: `src/scripts/start_smol_darwin_transplant.ps1`
- Create: `src/scripts/native_smol_darwin_smoke.py`
- Modify: `governance/audit/policy/operational-surface.json`
- Modify: `src/tests/test_operational_surface.py:10-24`
- Test: `src/tests/test_smol_transplant_launcher.py`

**Interfaces:**
- Produces: `-Canary`, `-Calibrate`, `-PublishCandidate`.
- Produces exact native success line.
- Consumes: CLI and gate reports from Tasks 10-11.

- [ ] **Step 1: Write failing launcher contract tests**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_launcher_separates_operator_authorities() -> None:
    text = (ROOT / "src/f51_darwin/operations/start_smol_darwin_transplant.ps1").read_text("utf-8")
    assert "[switch]$Canary" in text
    assert "[switch]$Calibrate" in text
    assert "[switch]$PublishCandidate" in text
    assert "Cannot combine" in text
    assert "DARWIN_16B_SMOL_NATIVE_OK" in text

def test_native_smoke_forbids_donor_and_transformers_model() -> None:
    text = (ROOT / "src/scripts/native_smol_darwin_smoke.py").read_text("utf-8")
    assert "TRANSFORMERS_OFFLINE" in text
    assert "HF_HUB_OFFLINE" in text
    assert "donor_loaded=false" in text
    assert "AutoModel" not in text
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest src/tests/test_smol_transplant_launcher.py -q`

Expected: FAIL because the launchers are absent.

- [ ] **Step 3: Implement separated authority and isolation**

The PowerShell implementation accepts exactly one action, resolves and verifies
the checkpoint by manifest rather than filename/mtime/cycle, fingerprints the
three canonical roots before and after, and:

- `-Canary`: runs source audit, strict inspector, finite forward/backward,
  round-trip logits, organ no-effect checks, then donor-free smoke;
- `-Calibrate`: invokes the explicit calibration CLI and never publishes;
- `-PublishCandidate`: requires a signed local gate report with every
  engineering, knowledge, and organ gate true, then atomically updates the
  isolated transplant candidate manifest.

The donor-free smoke starts a child process with `TRANSFORMERS_OFFLINE=1`,
`HF_HUB_OFFLINE=1`, an empty cache root, and only target checkpoint/tokenizer
paths. Patch `builtins.open`/`Path.open` in that process to abort on the
recorded donor roots, import only native Darwin modules, generate at least
eight tokens, verify organ modes/state, and print the exact success line.

Classify `src/scripts/start_smol_darwin_transplant.ps1` as supported,
`src/scripts/native_smol_darwin_smoke.py` as maintenance, add the launcher to
`script_allowed_files` and the 1.6B transplant config to `model_family`.

- [ ] **Step 4: Run launcher and source audits**

Run: `python -m pytest src/tests/test_smol_transplant_launcher.py src/tests/test_operational_surface.py src/tests/test_canary_launcher_contract.py -q`

Expected: PASS.

Run: `powershell -ExecutionPolicy Bypass -File src/scripts/start_smol_darwin_transplant.ps1 -Canary`

Expected before a real build: fail closed with `No manifest-approved transplant candidate`; no trainer process starts.

- [ ] **Step 5: Commit**

```powershell
git add src/f51_darwin/operations/start_smol_darwin_transplant.ps1 src/scripts/start_smol_darwin_transplant.ps1 src/scripts/native_smol_darwin_smoke.py governance/audit/policy/operational-surface.json src/tests/test_operational_surface.py src/tests/test_smol_transplant_launcher.py
git commit -m "feat(transplant): add donor free native proof and guarded launcher"
```

### Task 13: Execute the first real surgery as a bounded vertical slice

**Files:**
- Create: `workspace/runtime/darwin_16b_smol_transplant_v1/plan.json` (ignored)
- Create: `workspace/runtime/darwin_16b_smol_transplant_v1/coverage.jsonl` (ignored)
- Create: `workspace/runtime/darwin_16b_smol_transplant_v1/engineering-gates.json` (ignored)
- Create: `workspace/runtime/history/agent_bus/2026-07-28-darwin-16b-smol-transplant-build.md`
- Modify: `governance/docs/operacao/STATUS_ATUAL.md`

**Interfaces:**
- Produces: one isolated `engineering_transplant_only` candidate or an explicit quarantined failure.
- Consumes: all previous tasks.

- [ ] **Step 1: Capture immutable preflight**

Run:

```powershell
python src/scripts/transplant_smol_to_darwin_1_6b.py plan --config src/configs/darwin_x_1.6b_smol_transplant.yaml
```

Expected: `PLAN_OK` with plan ID, all seven source hashes, exact anatomy,
calibration digest, free disk, GPU inventory, and unchanged canonical-root
fingerprints. Stop if any identity differs.

- [ ] **Step 2: Build without calibration**

Run:

```powershell
python src/scripts/transplant_smol_to_darwin_1_6b.py build --plan workspace/runtime/darwin_16b_smol_transplant_v1/plan.json
```

Expected: `BUILD_OK status=engineering_transplant_only`, 100% donor coverage,
100% donor-derived target language tensors, 12 SSD blocks better than paired
random baselines, 16 populated MoE experts per layer, and all organ subspace
cosines at least 0.99. No optimizer/trainer remains running.

- [ ] **Step 3: Run the structural canary**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File src/scripts/start_smol_darwin_transplant.ps1 -Canary
```

Expected: strict v9 load, finite forward/backward, exact weight tying, stable
round-trip logits, unchanged organ shadow state, unchanged source roots, and:

```text
DARWIN_16B_SMOL_NATIVE_OK donor_loaded=false tokenizer=verified organs=verified
```

- [ ] **Step 4: Run source tests and record current truth**

Run:

```powershell
python -m pytest src/tests/test_transplant_16b_config.py src/tests/test_transplant_16b_sources.py src/tests/test_transplant_16b_tokenizer.py src/tests/test_transplant_16b_ledger.py src/tests/test_transplant_16b_projection.py src/tests/test_transplant_16b_layers.py src/tests/test_transplant_16b_moe.py src/tests/test_transplant_16b_ssd_fit.py src/tests/test_transplant_16b_organs.py src/tests/test_transplant_16b_checkpoint.py src/tests/test_transplant_16b_cli.py src/tests/test_transplant_16b_verification.py src/tests/test_smol_transplant_launcher.py src/tests/test_operational_surface.py -q
```

Expected: PASS. Write the exact commands, hashes, durations, gate values, failed
gates, artifact paths, and canonical-root before/after fingerprints into the
session report. Update `STATUS_ATUAL.md` to say either
`engineering_transplant_only` or `quarantined`; do not claim transferred
knowledge yet.

- [ ] **Step 5: Commit only source and documentation**

```powershell
git add governance/docs/operacao/STATUS_ATUAL.md workspace/runtime/history/agent_bus/2026-07-28-darwin-16b-smol-transplant-build.md
git commit -m "docs(transplant): record first native 1.6b surgery evidence"
```

Do not add checkpoints, tokenizer copies, plans, ledgers, calibration data, or
gate JSON to Git.

### Task 14: Calibrate, evaluate, and publish only after explicit authorization

**Files:**
- Create: `workspace/runtime/darwin_16b_smol_transplant_v1/calibration/` (ignored)
- Create: `workspace/runtime/darwin_16b_smol_transplant_v1/knowledge-gates.json` (ignored)
- Create: `workspace/runtime/history/agent_bus/2026-07-28-darwin-16b-smol-transplant-calibration.md`
- Modify: `governance/docs/operacao/STATUS_ATUAL.md`

**Interfaces:**
- Produces: a manifest-approved candidate only if every gate passes.
- Consumes: explicit operator authorization and the engineering candidate from Task 13.

- [ ] **Step 1: Await explicit calibration authority**

Do not infer this authority from approval of the design or build. The operator
must explicitly request calibration before running the next command.

- [ ] **Step 2: Run bounded calibration**

Run only after authorization:

```powershell
powershell -ExecutionPolicy Bypass -File src/scripts/start_smol_darwin_transplant.ps1 -Calibrate
```

Expected: three isolated seed lineages `(17, 29, 43)`, no organ effect, no
canonical-root changes, and a knowledge report containing paired BPC, KL,
top-1 agreement, repetition, n-gram, source-stratum, and holdout-leakage fields.

- [ ] **Step 3: Review the gate report without publishing**

Run:

```powershell
python src/scripts/evaluate_smol_darwin_transplant.py --report workspace/runtime/darwin_16b_smol_transplant_v1/knowledge-gates.json
```

Expected: `KNOWLEDGE_GATES_PASS` or
`KNOWLEDGE_GATES_FAIL status=engineering_transplant_only` with every failed
seed/metric named.

- [ ] **Step 4: Await separate publication authority and publish**

Only if all gates passed and Marco explicitly authorizes publication, run:

```powershell
powershell -ExecutionPolicy Bypass -File src/scripts/start_smol_darwin_transplant.ps1 -PublishCandidate
```

Expected: one atomic candidate-manifest update inside the isolated transplant
root, followed by the donor-free smoke success line. No source artifact changes.

- [ ] **Step 5: Record and commit the evidence**

Update the session report and `STATUS_ATUAL.md` with exact measured values and
the honest status. Then run:

```powershell
git add governance/docs/operacao/STATUS_ATUAL.md workspace/runtime/history/agent_bus/2026-07-28-darwin-16b-smol-transplant-calibration.md
git commit -m "docs(transplant): record calibrated knowledge gate evidence"
```

## Completion definition

Implementation is complete only when Tasks 1-13 pass and Task 13 produces the
exact donor-free proof. A scientifically validated knowledge transplant
requires Task 14 as a separately authorized phase. If Task 13 passes and Task
14 fails, the native organism remains a valid engineering transplant labeled
`engineering_transplant_only`.
