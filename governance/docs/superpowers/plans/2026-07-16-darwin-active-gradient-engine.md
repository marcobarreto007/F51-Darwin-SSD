# Darwin Active Gradient Engine V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Darwin's mutational-gradient decisions causally modify gradients, add a non-AdamW hybrid optimizer, and launch one verified fresh-start canary on the complete 18.55B-token corpus.

**Architecture:** Backward hooks collect local MoE evidence; the organism commits one decision at the optimizer boundary and applies bounded expert-gradient actions before clipping. A standalone hybrid optimizer uses orthogonalized matrix updates and tensor-scalar adaptive updates, while a runtime invariant prevents parameter/gradient/optimizer-state device divergence.

**Tech Stack:** Python 3.11, PyTorch 2.7+, pytest, YAML, JSONL, PowerShell 7, CUDA BF16.

## Global Constraints

- One trainer process and one CUDA context.
- Full corpus identity: 74,195,890,756 bytes and SHA-256 `9677e9f22f4d78efa7b25c77b2da3cbdb5fb2a499926ac641a75c13b65e25cff`.
- Active actions execute only at an optimizer boundary; structural actions execute only between cycles.
- Non-finite actions fail closed and device mismatches abort before weight mutation.
- AdamW remains only as a compatibility/control condition; `dae_hybrid` is the full candidate.
- No automatic checkpoint promotion.

---

### Task 1: Active Expert-Gradient Actions

**Files:**
- Modify: `src/f51_darwin/darwin_x.py`
- Test: `src/tests/test_gradient_mutational_state.py`

**Interfaces:**
- Consumes: existing `DeepSeekStyleMoE._pending_autonomic_actions` and protected 64-D bases.
- Produces: `DeepSeekStyleMoE.apply_active_gradient_actions() -> dict[str, Any]` and `DarwinXModel.apply_active_gradient_actions() -> list[dict[str, Any]]`.

- [ ] **Step 1: Write failing causal tests**

```python
def test_active_ignore_zeros_expert_gradient(moe):
    for parameter in moe.fine_experts[0].parameters():
        parameter.grad = torch.ones_like(parameter)
    moe._pending_autonomic_actions = {
        "plasticity": {"ignore": [0], "protect": [], "update": []},
        "shadow_mode": False,
    }
    report = moe.apply_active_gradient_actions()
    assert report["ignored"] == [0]
    assert all(torch.count_nonzero(p.grad) == 0 for p in moe.fine_experts[0].parameters())

def test_active_shadow_never_changes_gradient(moe):
    parameter = next(moe.fine_experts[0].parameters())
    parameter.grad = torch.ones_like(parameter)
    before = parameter.grad.clone()
    moe._pending_autonomic_actions = {"plasticity": {"ignore": [0]}, "shadow_mode": True}
    moe.apply_active_gradient_actions()
    assert torch.equal(parameter.grad, before)
```

- [ ] **Step 2: Run tests and confirm `AttributeError`**

Run: `python -m pytest -q src/tests/test_gradient_mutational_state.py -k "active_ignore or active_shadow"`  
Expected: FAIL because `apply_active_gradient_actions` does not exist.

- [ ] **Step 3: Add config and bounded actions**

```python
@dataclass(frozen=True)
class DarwinXConfig:
    dae_enabled: bool = False
    dae_shadow_mode: bool = True
    dae_update_gain_min: float = 0.75
    dae_update_gain_max: float = 1.25

@torch.no_grad()
def apply_active_gradient_actions(self) -> dict[str, Any]:
    actions = self._pending_autonomic_actions or self._propose_autonomic_actions()
    if not self.config.dae_enabled or actions.get("shadow_mode", True):
        return {"shadow_mode": True, "ignored": [], "protected": [], "updated": []}
    decisions = actions.get("plasticity", {})
    report = {"shadow_mode": False, "ignored": [], "protected": [], "updated": []}
    for expert_idx in decisions.get("ignore", []):
        for parameter in self.fine_experts[expert_idx].parameters():
            if parameter.grad is not None:
                parameter.grad.zero_()
        report["ignored"].append(expert_idx)
    for expert_idx in decisions.get("update", []):
        gain = float((1.0 + 0.1 * self.neuroendocrine.dopamine[expert_idx]).clamp(
            self.config.dae_update_gain_min, self.config.dae_update_gain_max
        ).item())
        for parameter in self.fine_experts[expert_idx].parameters():
            if parameter.grad is not None and torch.isfinite(parameter.grad).all():
                parameter.grad.mul_(gain)
        report["updated"].append({"expert": expert_idx, "gain": gain})
    for expert_idx in decisions.get("protect", []):
        changed = self._project_expert_gradient_from_protected_sketch(expert_idx)
        if changed:
            report["protected"].append(expert_idx)
    return report
```

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest -q src/tests/test_gradient_mutational_state.py src/tests/test_neuroendocrine_moe.py`  
Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/f51_darwin/darwin_x.py src/tests/test_gradient_mutational_state.py
git commit -m "feat: activate bounded Darwin gradient actions"
```

### Task 2: Nitro and Optimizer Device Invariants

**Files:**
- Modify: `src/f51_darwin/darwin_x.py`
- Modify: `src/scripts/darwin_organism.py`
- Test: `src/tests/test_darwin_x_training.py`
- Test: `src/tests/test_organism_causal_runtime.py`

**Interfaces:**
- Produces: `assert_optimizer_device_invariants(model, optimizer) -> None` and `move_optimizer_state_for_parameters(optimizer, parameters) -> int`.

- [ ] **Step 1: Write failing device tests**

```python
def test_optimizer_device_invariant_rejects_mismatch():
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    parameter = next(model.parameters())
    optimizer.state[parameter]["exp_avg"] = torch.zeros_like(parameter, device="meta")
    with pytest.raises(RuntimeError, match="optimizer device invariant"):
        assert_optimizer_device_invariants(model, optimizer)

def test_nitro_enabled_is_real_config_field(tiny_config):
    config = replace(tiny_config, nitro_enabled=False)
    moe = DeepSeekStyleMoE(config)
    assert moe.nitro_enabled is False
```

- [ ] **Step 2: Confirm focused failure**

Run: `python -m pytest -q src/tests/test_darwin_x_training.py src/tests/test_organism_causal_runtime.py -k "device_invariant or nitro_enabled"`  
Expected: FAIL because the interfaces do not exist.

- [ ] **Step 3: Implement the invariant and explicit Nitro flag**

```python
def assert_optimizer_device_invariants(model, optimizer):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and parameter.grad.device != parameter.device:
            raise RuntimeError(f"optimizer device invariant: {name} grad")
        for key, value in optimizer.state.get(parameter, {}).items():
            if torch.is_tensor(value) and value.numel() > 1 and value.device != parameter.device:
                raise RuntimeError(f"optimizer device invariant: {name}.{key}")
```

Add `nitro_enabled: bool = True` to `DarwinXConfig`, consume it in
`DeepSeekStyleMoE`, and call the invariant immediately before every optimizer
step. Scalar counters are allowed to remain on CPU.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest -q src/tests/test_darwin_x_training.py src/tests/test_organism_causal_runtime.py src/tests/test_training_observability.py`  
Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/f51_darwin/darwin_x.py src/scripts/darwin_organism.py tests
git commit -m "fix: enforce Nitro optimizer device invariants"
```

### Task 3: DAE Hybrid Optimizer

**Files:**
- Create: `src/f51_darwin/dae_optimizer.py`
- Modify: `src/scripts/darwin_organism.py`
- Create: `src/tests/test_dae_optimizer.py`

**Interfaces:**
- Produces: `DAEHybridOptimizer`, `build_optimizer(name, named_parameters, lr, weight_decay)`.

- [ ] **Step 1: Write optimizer tests**

```python
def test_dae_hybrid_updates_matrix_and_vector():
    matrix = nn.Parameter(torch.eye(4))
    vector = nn.Parameter(torch.ones(4))
    optimizer = DAEHybridOptimizer([matrix, vector], lr=1e-2)
    (matrix.sum() + vector.sum()).backward()
    optimizer.step()
    assert not torch.equal(matrix, torch.eye(4))
    assert not torch.equal(vector, torch.ones(4))

def test_dae_hybrid_state_roundtrip():
    left = nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4))
    first = DAEHybridOptimizer([left], lr=1e-2)
    left.sum().backward(); first.step(); first.zero_grad(set_to_none=True)
    saved_parameter = left.detach().clone()
    saved_state = copy.deepcopy(first.state_dict())
    right = nn.Parameter(saved_parameter.clone())
    second = DAEHybridOptimizer([right], lr=1e-2)
    second.load_state_dict(saved_state)
    left.sum().backward(); right.sum().backward()
    first.step(); second.step()
    torch.testing.assert_close(left, right)
```

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest -q src/tests/test_dae_optimizer.py`  
Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement spectral and scalar-adaptive paths**

```python
class DAEHybridOptimizer(torch.optim.Optimizer):
    """Orthogonalized momentum for matrices; tensor-scalar RMS for vectors."""

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.float()
                state = self.state[parameter]
                momentum = state.setdefault("momentum", torch.zeros_like(parameter))
                momentum.mul_(group["momentum"]).add_(gradient, alpha=1-group["momentum"])
                if parameter.ndim >= 2 and min(parameter.shape) >= 2:
                    update = zeropower_via_newton_schulz(momentum.reshape(parameter.shape[0], -1))
                    update = update.reshape_as(parameter)
                else:
                    rms = state.setdefault("rms", torch.zeros((), device=parameter.device))
                    rms.mul_(group["beta2"]).add_(gradient.square().mean(), alpha=1-group["beta2"])
                    update = momentum / rms.sqrt().clamp_min(group["eps"])
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])
```

- [ ] **Step 4: Run optimizer and checkpoint tests**

Run: `python -m pytest -q src/tests/test_dae_optimizer.py src/tests/test_topology_manifest_v7.py`  
Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/f51_darwin/dae_optimizer.py src/scripts/darwin_organism.py src/tests/test_dae_optimizer.py
git commit -m "feat: add Darwin spectral hybrid optimizer"
```

### Task 4: Optimizer-Boundary Integration and Ledger

**Files:**
- Modify: `src/scripts/darwin_organism.py`
- Modify: `src/f51_darwin/training_observability.py`
- Test: `src/tests/test_organism_causal_runtime.py`
- Test: `src/tests/test_training_observability.py`

**Interfaces:**
- Consumes: `model.apply_active_gradient_actions()` and `assert_optimizer_device_invariants`.
- Produces: checkpointed `optimizer_identity`, `dae_report`, and one decision per optimizer step.

- [ ] **Step 1: Write an order test**

```python
def test_dae_runs_before_clip_and_optimizer_step(monkeypatch):
    events = []
    model.apply_active_gradient_actions = lambda: events.append("dae") or []
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *a, **k: events.append("clip"))
    optimizer.step = lambda: events.append("step")
    organism.train_cycle(steps=1)
    assert events[:3] == ["dae", "clip", "step"]
```

- [ ] **Step 2: Confirm the test fails with current order**

Run: `python -m pytest -q src/tests/test_organism_causal_runtime.py -k dae_runs_before`  
Expected: FAIL; current autonomic method runs after `step()`.

- [ ] **Step 3: Change the boundary order and persist identity**

```python
if optimizer_due:
    dae_report = model.apply_active_gradient_actions()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    assert_optimizer_device_invariants(model, self.optimizer)
    self.optimizer.step()
    self.optimizer.zero_grad(set_to_none=True)
    model.apply_pending_autonomic_actions()
```

Checkpoint fields must include `optimizer_identity`, DAE config, action counts,
gradient norms before/after and the enabled condition.

```python
payload["optimizer_identity"] = {
    "name": self.cfg.optimizer,
    "class": type(self.optimizer).__name__,
    "version": 1,
}
payload["dae"] = {
    "enabled": self.model_config.dae_enabled,
    "shadow_mode": self.model_config.dae_shadow_mode,
    "last_report": getattr(self, "_last_dae_report", []),
}
```

- [ ] **Step 4: Run causal runtime and observability tests**

Run: `python -m pytest -q src/tests/test_organism_causal_runtime.py src/tests/test_training_observability.py src/tests/test_checkpoint_eval.py`  
Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/scripts/darwin_organism.py src/f51_darwin/training_observability.py tests
git commit -m "feat: execute DAE at the optimizer boundary"
```

### Task 5: Cloud Profile, Full Verification and Canary

**Files:**
- Create: `src/configs/darwin_x_1.6b_dae_cloud.yaml`
- Modify: `src/scripts/experiment_runner_v2.py`
- Test: `src/tests/test_canary_launcher_contract.py`

**Interfaces:**
- Produces: a cloud-only profile with `nitro_enabled: false`, `dae_enabled: true`, `dae_shadow_mode: false`, and `optimizer: dae_hybrid`.

- [ ] **Step 1: Add config-contract tests**

```python
def test_cloud_dae_profile_is_active_and_resident():
    config = load_config("src/configs/darwin_x_1.6b_dae_cloud.yaml")
    assert config["nitro_enabled"] is False
    assert config["dae_enabled"] is True
    assert config["dae_shadow_mode"] is False
    assert config["optimizer"] == "dae_hybrid"
```

- [ ] **Step 2: Create the profile and make runner locking cross-platform**

Use `msvcrt` on Windows and `fcntl` on Linux.  Conditions run sequentially and
write isolated JSONL/checkpoint roots.

```python
if os.name == "nt":
    import msvcrt
    lock = lambda handle: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
else:
    import fcntl
    lock = lambda handle: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
```

- [ ] **Step 3: Run all CPU gates**

Run: `$env:CUDA_VISIBLE_DEVICES='-1'; .\.venv_nitro\Scripts\python.exe -m py_compile src\\f51_darwin\\darwin_x.py src\\f51_darwin\\dae_optimizer.py src\\scripts\\darwin_organism.py src\\scripts\\experiment_runner_v2.py; .\.venv_nitro\Scripts\python.exe -m pytest -q -p no:cacheprovider; Remove-Item Env:CUDA_VISIBLE_DEVICES`  
Expected: all tests pass with no CUDA allocation.

- [ ] **Step 4: Synchronize verified files and run a short cloud canary**

Run the repository's isolated `cycle --canary` contract with exactly 250 steps,
`--checkpoint-root /workspace/03_CHECKPOINTS/canary_dae_v1`, and
`--metrics-jsonl /workspace/runs/dae_v1/canary.jsonl`.  Before launch, require
`pgrep -af 'darwin_organism.py.*run247'` to be empty, `nvidia-smi` to list no
compute process, `stat` to return 74,195,890,756 bytes, and `sha256sum` to return
the approved corpus hash.  Abort on non-finite loss, device mismatch, OOM,
duplicate PID or missing organ ledger.

- [ ] **Step 5: Launch one full run only if canary passes**

Use the exact validated corpus path, `--fresh-start`, one process and the cloud
DAE profile.  Immediately verify PID uniqueness, first finite steps, VRAM,
tokens/second and checkpoint root.

- [ ] **Step 6: Commit local launch artifacts and status, never secrets/log bulk**

```powershell
git add src/configs/darwin_x_1.6b_dae_cloud.yaml src/scripts/experiment_runner_v2.py tests governance/docs/operacao/STATUS_ATUAL.md
git commit -m "feat: gate Darwin DAE cloud canary"
```
