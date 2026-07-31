# Darwin-Smol Native Dense Organism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publish a structurally dense Darwin organism that preserves the complete local SmolLM2-1.7B-Instruct language function and keeps compatible Darwin organs connected behind neutral first-boot gates.

**Architecture:** Add an opt-in `dense_swiglu` feed-forward kind while preserving `moe` as the default for every existing lineage. Build a new isolated checkpoint by copying all 218 Smol tensors directly into attention and native dense FFNs, grow compatible organ state, prove the absence of MoE/router objects, then run parity, donor-free dual-GPU inference, dense gradient, random-control, publication, and repository gates.

**Tech Stack:** Python 3.12, PyTorch 2.12 nightly CUDA 12.8, safetensors, Transformers 5.14.1, pytest, PowerShell, JSON/YAML manifests, SHA-256.

## Global Constraints

- Work only in `C:\Users\marco\Desktop\F51-Darwin-SSD`.
- Do not download a model or contact an external service; use the verified local Smol snapshot.
- Do not modify or overwrite V1, V2, or Exact Brain V3 checkpoints.
- Keep `feed_forward_kind="moe"` as the default and preserve strict loading for existing lineages.
- The dense root is `workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1`.
- The dense runtime root is `workspace/runtime/darwin_17b_smol_dense_v1`.
- The dense model must contain no `DeepSeekStyleMoE`, `FineRouter`, `fine_experts`, or `.moe.` state key.
- All 218 donor tensors must be covered without SVD, clustering, folding, projection, or compression.
- Heartbeat remains connected; GABA, TTM, and IHS start behind exact zero gates.
- DAE and MoE neuroendocrine control are disabled for the dense lineage rather than relabeled.
- Publish only after strict reload, parity, donor-free dual-GPU generation, three dense random controls, and a dense optimizer smoke pass.
- Use explicit paths in commits, Conventional Commits, and never `--no-verify`.

---

### Task 1: Add a native dense SwiGLU block contract

**Files:**
- Modify: `src/f51_darwin/darwin_x_core/config.py`
- Modify: `src/f51_darwin/darwin_x_core/layers.py`
- Modify: `src/f51_darwin/darwin_x_core/block.py`
- Modify: `src/f51_darwin/darwin_x_core/estimation.py`
- Test: `src/tests/test_dense_swiglu.py`

**Interfaces:**
- Consumes: existing `DarwinXConfig`, `ExpertFFN`, `DeepSeekStyleMoE`, and `DarwinXBlock`.
- Produces: `DarwinXConfig.feed_forward_kind: str`, `DenseSwiGLU`, `DarwinXBlock.ffn`, and mutually exclusive `block.moe`/`block.ffn`.

- [ ] **Step 1: Write failing config and function tests**

```python
def test_dense_config_is_opt_in_and_validated() -> None:
    assert DarwinXConfig().feed_forward_kind == "moe"
    dense = replace(
        tiny_config(),
        feed_forward_kind="dense_swiglu",
        fine_experts=1,
        shared_experts=0,
        experts_per_token=1,
    )
    assert dense.feed_forward_kind == "dense_swiglu"
    with pytest.raises(ValueError, match="feed_forward_kind"):
        replace(dense, feed_forward_kind="unknown")


def test_dense_swiglu_matches_the_explicit_equation() -> None:
    torch.manual_seed(51)
    layer = DenseSwiGLU(8, 16, dropout=0.0).eval()
    x = torch.randn(2, 3, 8)
    expected = layer.down_proj(
        F.silu(layer.gate_proj(x)) * layer.up_proj(x)
    )
    torch.testing.assert_close(layer(x), expected, atol=0.0, rtol=0.0)
```

- [ ] **Step 2: Run the focused tests and confirm the missing interface**

Run:

```powershell
python -m pytest src/tests/test_dense_swiglu.py -q
```

Expected: collection or assertion failure because `DenseSwiGLU` and
`feed_forward_kind` do not exist.

- [ ] **Step 3: Implement the native layer and mutually exclusive block**

Add to `config.py`:

```python
feed_forward_kind: str = "moe"
```

Validate:

```python
if self.feed_forward_kind not in {"moe", "dense_swiglu"}:
    raise ValueError("feed_forward_kind must be 'moe' or 'dense_swiglu'.")
if self.feed_forward_kind == "dense_swiglu" and (
    self.fine_experts != 1
    or self.shared_experts != 0
    or self.experts_per_token != 1
):
    raise ValueError(
        "dense_swiglu requires fine_experts=1, shared_experts=0, "
        "experts_per_token=1."
    )
```

Add to `layers.py`:

```python
class DenseSwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(hidden))
```

In `DarwinXBlock.__init__`, register only one feed-forward implementation:

```python
self.moe = None
self.ffn = None
if config.feed_forward_kind == "dense_swiglu":
    self.ffn = DenseSwiGLU(
        config.d_model,
        config.fine_expert_hidden_dim,
        config.dropout,
    )
else:
    self.moe = DeepSeekStyleMoE(config)
```

In `DarwinXBlock.forward`:

```python
normalized = self.norm2(x)
if self.ffn is not None:
    feed_forward = self.ffn(normalized)
    aux: dict[str, Any] = {}
else:
    assert self.moe is not None
    feed_forward, aux = self.moe(normalized)
```

Apply GABA and the residual to `feed_forward`. Branch the parameter estimator
so dense active parameters contain exactly the three SwiGLU matrices and no
router.

- [ ] **Step 4: Prove structure, function, gradients, and legacy default**

Add assertions:

```python
assert dense_block.moe is None
assert dense_block.ffn is not None
assert not any(".moe." in key for key in dense_model.state_dict())
loss = dense_block.ffn(torch.randn(2, 3, 8)).square().mean()
loss.backward()
assert all(
    parameter.grad is not None
    for parameter in dense_block.ffn.parameters()
)
assert legacy_block.moe is not None
assert legacy_block.ffn is None
```

Run:

```powershell
python -m pytest src/tests/test_dense_swiglu.py src/tests/test_darwin_x.py src/tests/test_darwin_x_public_contract.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add -- src/f51_darwin/darwin_x_core/config.py src/f51_darwin/darwin_x_core/layers.py src/f51_darwin/darwin_x_core/block.py src/f51_darwin/darwin_x_core/estimation.py src/tests/test_dense_swiglu.py
git commit -m "feat(core): add native dense SwiGLU blocks"
```

---

### Task 2: Make model lifecycle code architecture-aware

**Files:**
- Modify: `src/f51_darwin/darwin_x_core/model.py`
- Modify: `src/f51_darwin/darwin_x.py`
- Test: `src/tests/test_dense_darwin_model.py`
- Test: `src/tests/test_dual_gpu_split.py`

**Interfaces:**
- Consumes: `DarwinXBlock.moe: DeepSeekStyleMoE | None` and `DarwinXBlock.ffn: DenseSwiGLU | None`.
- Produces: dense-safe forward, topology, placement, generation, optimizer-boundary methods, and exported `DenseSwiGLU`.

- [ ] **Step 1: Write failing dense lifecycle tests**

```python
def test_dense_model_forward_and_topology_have_no_moe() -> None:
    model = DarwinXModel(dense_tiny_config()).eval()
    output = model(torch.tensor([[1, 2, 3]]), heartbeat=False)
    assert torch.isfinite(output.logits).all()
    assert output.moe_stats == []
    manifest = model.topology_manifest()
    assert manifest["base_config"]["feed_forward_kind"] == "dense_swiglu"
    assert manifest["neuroendocrine_state"] == []
    assert model.apply_pending_autonomic_actions() == []
    assert model.apply_active_gradient_actions() == []
    assert model.execute_structural_actions() == []
```

- [ ] **Step 2: Run and observe the first unconditional `.moe` failure**

Run:

```powershell
python -m pytest src/tests/test_dense_darwin_model.py -q
```

Expected: failure at an unconditional `block.moe` access.

- [ ] **Step 3: Guard every MoE-only lifecycle path**

Use:

```python
def _moe_blocks(self):
    return tuple(
        block.moe for block in self.blocks if block.moe is not None
    )
```

Return one result per actual MoE block from optimizer-boundary methods.
Serialize `feed_forward_kind` in `base_config`. For dense topology, serialize
the dense hidden width and no expert/neuroendocrine entries. Reject attempts
to apply MoE structural restoration to a dense manifest.

Guard vertical routing:

```python
left = self.blocks[i].moe
right = self.blocks[i + 1].moe
if left is not None and right is not None:
    right._vertical_bias = left._emit_vertical_bias()
```

Guard Ghost-to-MoE signals the same way. Preserve `moe_stats=[]` for a dense
model instead of fabricating zero-valued router metrics.

- [ ] **Step 4: Verify dense dual-GPU placement and legacy topology**

Test `recommended_dual_gpu_split` with a meta dense model and the measured
16 GB / 12 GB capacities. Run:

```powershell
python -m pytest src/tests/test_dense_darwin_model.py src/tests/test_dual_gpu_split.py src/tests/test_darwin_x_public_contract.py src/tests/test_darwin_x_training.py -q
```

Expected: all pass; the legacy public contract remains unchanged.

- [ ] **Step 5: Commit Task 2**

```powershell
git add -- src/f51_darwin/darwin_x_core/model.py src/f51_darwin/darwin_x.py src/tests/test_dense_darwin_model.py src/tests/test_dual_gpu_split.py
git commit -m "fix(core): make Darwin lifecycle dense-aware"
```

---

### Task 3: Define the isolated 1.7B dense lineage and surgery

**Files:**
- Create: `src/configs/darwin_x_1.7b_smol_dense.yaml`
- Create: `src/f51_darwin/transplant_16b/dense_assembly.py`
- Create: `src/scripts/transplant_smol_dense_brain.py`
- Test: `src/tests/test_smol_dense_brain_config.py`
- Test: `src/tests/test_smol_dense_assembly.py`

**Interfaces:**
- Consumes: local Smol safetensors, `DenseSwiGLU`, V3 organ source, checkpoint helpers, tokenizer adapter, and exact assembly verification primitives.
- Produces: `write_dense_plan()`, `build_dense_brain()`, `publish_dense_candidate()`, and a v9 checkpoint under the isolated dense root.

- [ ] **Step 1: Write the dense config contract**

Use the Exact Brain V3 dimensions but set:

```yaml
model_name: F51-Darwin-X-1.7B-Smol-Native-Dense-V1
feed_forward_kind: dense_swiglu
fine_experts: 1
shared_experts: 0
experts_per_token: 1
fine_expert_hidden_dim: 8192
dae_enabled: false
dae_shadow_mode: true
nitro_enabled: false
checkpoint_root: workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1
```

Keep all 24 attention indices, Llama RoPE, FP32 RMSNorm, Heartbeat, GABA,
TTM, IHS, Spider, JEPA, MTP, Sleep, Decision Engine, and Unified Mesh settings
from V3, with all language-affecting auxiliary weights or residual scales at
zero.

- [ ] **Step 2: Write failing mapping and structural tests**

```python
def test_dense_config_has_no_runtime_moe() -> None:
    model = DarwinXModel(load_dense_config())
    assert all(block.ffn is not None for block in model.blocks)
    assert all(block.moe is None for block in model.blocks)
    assert not any(".moe." in key for key in model.state_dict())


def test_dense_language_map_targets_native_ffn() -> None:
    mapping = dense_layer_mapping(0)
    assert mapping["model.layers.0.mlp.gate_proj.weight"] == (
        "blocks.0.ffn.gate_proj.weight",
    )
```

- [ ] **Step 3: Implement the fail-closed dense assembly**

The CLI is:

```python
parser.add_argument("action", choices=("plan", "build", "publish"))
```

The exact per-layer mapping is:

```python
{
    f"{source}.input_layernorm.weight": (f"{target}.norm1.weight",),
    f"{source}.post_attention_layernorm.weight": (f"{target}.norm2.weight",),
    f"{source}.self_attn.q_proj.weight": (f"{target}.attention.q_proj.weight",),
    f"{source}.self_attn.k_proj.weight": (f"{target}.attention.k_proj.weight",),
    f"{source}.self_attn.v_proj.weight": (f"{target}.attention.v_proj.weight",),
    f"{source}.self_attn.o_proj.weight": (f"{target}.attention.o_proj.weight",),
    f"{source}.mlp.gate_proj.weight": (f"{target}.ffn.gate_proj.weight",),
    f"{source}.mlp.up_proj.weight": (f"{target}.ffn.up_proj.weight",),
    f"{source}.mlp.down_proj.weight": (f"{target}.ffn.down_proj.weight",),
}
```

Before save, enforce:

```python
if any(block.moe is not None for block in model.blocks):
    raise ValueError("dense candidate contains a MoE block")
if any(".moe." in key for key in model.state_dict()):
    raise ValueError("dense candidate state contains a MoE key")
ledger.assert_complete(donor_names, language_targets(model.state_dict()))
```

Write checkpoint, manifest, plan, coverage ledger, and build report atomically.
Record `feed_forward_kind="dense_swiglu"`, donor count, parameter count,
organ tensor count, parity, and structural assertions.

- [ ] **Step 4: Run unit tests and plan-only preflight**

```powershell
python -m pytest src/tests/test_dense_swiglu.py src/tests/test_dense_darwin_model.py src/tests/test_smol_dense_brain_config.py src/tests/test_smol_dense_assembly.py -q
python src/scripts/transplant_smol_dense_brain.py plan
```

Expected: tests pass and output contains `DENSE_PLAN_OK plan_id=`.

- [ ] **Step 5: Commit Task 3**

```powershell
git add -- src/configs/darwin_x_1.7b_smol_dense.yaml src/f51_darwin/transplant_16b/dense_assembly.py src/scripts/transplant_smol_dense_brain.py src/tests/test_smol_dense_brain_config.py src/tests/test_smol_dense_assembly.py
git commit -m "feat(transplant): define native dense Smol surgery"
```

---

### Task 4: Build and prove the dense checkpoint

**Files:**
- Create: `src/scripts/evaluate_smol_dense_candidate.py`
- Modify: `src/scripts/native_smol_darwin_smoke.py`
- Modify: `src/f51_darwin/transplant_16b/calibration.py`
- Test: `src/tests/test_smol_dense_runtime.py`
- Produce: `workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt`
- Produce: `workspace/runtime/darwin_17b_smol_dense_v1/build-report.json`
- Produce: `workspace/runtime/darwin_17b_smol_dense_v1/knowledge-gates.json`

**Interfaces:**
- Consumes: the dense surgery plan, existing teacher evaluator, required seeds `(17, 29, 43)`, and native smoke loader.
- Produces: strict dense checkpoint, donor-free proof, dense optimizer proof, and three paired random-control rows.

- [ ] **Step 1: Add dense runtime assertions**

Extend native smoke to emit:

```python
"feed_forward_kind": config.feed_forward_kind,
"moe_modules": sum(
    int(block.moe is not None) for block in model.blocks
),
"dense_ffn_modules": sum(
    int(block.ffn is not None) for block in model.blocks
),
```

When `feed_forward_kind == "dense_swiglu"`, require 0 MoE modules, 24 dense
FFNs, and no `.moe.` state key.

- [ ] **Step 2: Add a bounded dense optimizer smoke**

Load the saved checkpoint, freeze all parameters except
`blocks.0.ffn.{gate_proj,up_proj,down_proj}.weight`, run one cross-entropy
backward and SGD step, and record:

```python
{
    "finite_loss": True,
    "gate_grad_nonzero": True,
    "up_grad_nonzero": True,
    "down_grad_nonzero": True,
    "weights_changed": True,
}
```

The optimizer smoke operates on a temporary in-memory model and never writes
over the candidate.

- [ ] **Step 3: Build on CPU and run parity on both GPUs**

Verify no Python trainer is running, both GPUs are free, and at least 20 GB of
disk remains. Run:

```powershell
python src/scripts/transplant_smol_dense_brain.py build
```

Expected: `DENSE_BUILD_OK`, 218/218 coverage, 0 MoE modules, 24 dense FFNs,
finite parity metrics, and an immutable checkpoint manifest.

- [ ] **Step 4: Run donor-free native inference**

```powershell
$env:TRANSFORMERS_OFFLINE="1"
$env:HF_HUB_OFFLINE="1"
python src/scripts/native_smol_darwin_smoke.py `
  --checkpoint workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt `
  --manifest workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.manifest.json `
  --tokenizer workspace/01_TOKENIZER/smol_49152_transplant_v1 `
  --tokens 12
```

Expected: `donor_loaded=false`, `dual_gpu=true`, `moe_modules=0`,
`dense_ffn_modules=24`, finite non-empty generation, and exact repeated logits.

- [ ] **Step 5: Run the three dense random controls**

```powershell
python src/scripts/evaluate_smol_dense_candidate.py
```

Expected: seeds exactly 17, 29, and 43; candidate BpB and KL lower than every
random dense control; top-1 and generation gates pass.

- [ ] **Step 6: Commit Task 4 source**

```powershell
git add -- src/scripts/evaluate_smol_dense_candidate.py src/scripts/native_smol_darwin_smoke.py src/f51_darwin/transplant_16b/calibration.py src/tests/test_smol_dense_runtime.py
git commit -m "test(transplant): prove dense brain runtime and controls"
```

---

### Task 5: Publish and switch the supported operator surface

**Files:**
- Modify: `src/f51_darwin/operations/start_smol_darwin_transplant.ps1`
- Modify: `governance/audit/policy/operational-surface.json`
- Modify: `src/tools/check_operational_surface.py`
- Modify: `src/tests/test_smol_transplant_launcher.py`
- Modify: `src/tests/test_operational_surface.py`
- Produce: `workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/candidate-manifest.json`

**Interfaces:**
- Consumes: approved dense build report, checkpoint manifest, native log,
  dense optimizer report, and knowledge gates.
- Produces: fail-closed `approved_candidate` and supported `-Canary`,
  `-Calibrate`, and `-PublishCandidate` actions targeting the dense root.

- [ ] **Step 1: Write the fail-closed publication tests**

Assert that publication requires:

```python
assert build["feed_forward_kind"] == "dense_swiglu"
assert build["moe_modules"] == 0
assert build["dense_ffn_modules"] == 24
assert gates["passed"] is True
assert optimizer_smoke["weights_changed"] is True
assert "donor_loaded=false" in native_log
```

- [ ] **Step 2: Publish the immutable candidate**

```powershell
python src/scripts/transplant_smol_dense_brain.py publish
```

Expected: `DENSE_CANDIDATE_PUBLISHED checkpoint_sha256=<64 hex>`.

- [ ] **Step 3: Switch the launcher only after publication**

Point `$root` to `03_CHECKPOINTS_1.7B_SMOL_DENSE_V1`, route `-Calibrate` to
`evaluate_smol_dense_candidate.py`, and route `-PublishCandidate` to
`transplant_smol_dense_brain.py publish`. Add
`"1.7b-smol-dense": "src/configs/darwin_x_1.7b_smol_dense.yaml"` to the operational
model family.

- [ ] **Step 4: Prove all supported actions**

```powershell
powershell -ExecutionPolicy Bypass -File src/scripts/start_smol_darwin_transplant.ps1 -Canary
powershell -ExecutionPolicy Bypass -File src/scripts/start_smol_darwin_transplant.ps1 -Calibrate
powershell -ExecutionPolicy Bypass -File src/scripts/start_smol_darwin_transplant.ps1 -PublishCandidate
python src/tools/check_operational_surface.py
```

Expected: all exit 0 and refer to the dense checkpoint hash.

- [ ] **Step 5: Commit Task 5**

```powershell
git add -- src/f51_darwin/operations/start_smol_darwin_transplant.ps1 governance/audit/policy/operational-surface.json src/tools/check_operational_surface.py src/tests/test_smol_transplant_launcher.py src/tests/test_operational_surface.py
git commit -m "fix(operations): promote native dense Smol candidate"
```

---

### Task 6: Reconcile status, run completion audit, and deliver

**Files:**
- Modify: `governance/docs/operacao/STATUS_ATUAL.md`
- Create locally: `workspace/runtime/history/agent_bus/2026-07-29-darwin-smol-native-dense-v1.md`

**Interfaces:**
- Consumes: every manifest, report, test result, command output, hash, and commit from Tasks 1–5.
- Produces: canonical operational status, local detailed evidence note, clean Git state, and completion proof.

- [ ] **Step 1: Update canonical status without overstating organs**

Record exact checkpoint path, bytes, SHA-256, parameter count, 218/218 donor
coverage, 0 MoE modules, 24 dense FFNs, parity, three controls, optimizer
smoke, donor-free output, gates, and rollback. State explicitly that
connection is not measured organ benefit.

- [ ] **Step 2: Run focused and full source gates**

```powershell
python -m pytest src/tests/test_dense_swiglu.py src/tests/test_dense_darwin_model.py src/tests/test_smol_dense_brain_config.py src/tests/test_smol_dense_assembly.py src/tests/test_smol_dense_runtime.py src/tests/test_smol_transplant_launcher.py src/tests/test_operational_surface.py -q
python src/tools/check_operational_surface.py
python src/tools/check_dependency_policy.py --root .
python src/tools/check_canonical_docs.py --root .
python src/tools/check_duplicates.py --root .
python -m pytest -q
git diff --check
```

Expected: focused tests pass, every checker returns `status=pass`, the full
suite has zero failures, and `git diff --check` has no output.

- [ ] **Step 3: Verify artifacts and idle hardware**

```powershell
Get-FileHash workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt -Algorithm SHA256
Get-Content workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/candidate-manifest.json -Raw
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
git status --short
```

Expected: checkpoint hash equals candidate and checkpoint manifests; status is
`approved_candidate`; GPUs have no model process; worktree contains only the
status documentation change before its commit.

- [ ] **Step 4: Commit documentation**

```powershell
git add -- governance/docs/operacao/STATUS_ATUAL.md
git commit -m "docs(transplant): publish native dense organism evidence"
```

Keep the detailed runtime note ignored under `workspace/`; do not force-add it
because source archives must exclude `workspace/`.

- [ ] **Step 5: Final post-commit audit**

```powershell
git status --porcelain=v1
git log --oneline -12
python -m pytest -q
```

Expected: empty status, visible task commits, and zero test failures.
