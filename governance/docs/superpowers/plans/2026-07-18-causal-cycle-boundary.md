# Causal Cycle Boundary Implementation Plan

> **For agentic workers:** Execute inline with TDD; do not delegate or commit.

**Goal:** Make Unified Mesh a real-signal observer/proposer while preserving one structural mutation authority at the cycle boundary.

**Architecture:** `UnifiedControlMesh` collects current model/output signals without mutating state. Lifecycle gates mesh and sleep behavior with model-config toggles; REM and the legacy evolution/death hook return proposals/reports only. `model.execute_structural_actions()` remains the sole topology mutation call.

**Tech Stack:** Python, PyTorch, pytest.

## Execution status - 2026-07-18

Implemented and covered by the repository-wide 589-test pass. The structural
boundary now performs exact proposal/preflight/pending/executed validation and
protected sleep has paired evaluation plus complete rollback coverage.

## Global Constraints

- Read/write only inside `C:\Users\marco\Desktop\F51-Darwin-SSD`.
- Do not start training and do not commit.
- Preserve the user's YAML.
- Do not modify `model.py`, `config.py`, `losses.py`, `causal_bus.py`, or `causal_ledger.py`.

---

### Task 1: Lock the causal boundary with failing tests

**Files:**
- Create: `src/tests/test_causal_cycle_boundary.py`

**Interfaces:**
- Consumes: `UnifiedControlMesh.collect_signals`, `run_sleep_cycle`, `execute_evolution_actions`, `_DarwinLifecycleMixin.run_cycle`.
- Produces: regression coverage for non-empty signals, toggle gates, proposal-only legacy APIs, and one structural execution.

- [ ] Add focused tests using tiny fake models and monkeypatches.
- [ ] Run `python -m pytest src/tests/test_causal_cycle_boundary.py -q` and confirm failures identify current mutation/toggle behavior.

### Task 2: Make mesh APIs observational

**Files:**
- Modify: `src/f51_darwin/organism/unified_mesh.py`

**Interfaces:**
- Produces: schema-tolerant real signal collection and proposal-only sleep/evolution reports.

- [ ] Normalize scalar/list neuroendocrine fields when collecting signals.
- [ ] Change REM to report proposed pruning without writing parameters.
- [ ] Change legacy evolution actions to report proposed decisions without calling pruning or changing topology.

### Task 3: Gate lifecycle and retain one authority

**Files:**
- Modify: `src/f51_darwin/organism/lifecycle.py`

**Interfaces:**
- Consumes: model config toggles and proposal-only Unified Mesh APIs.
- Produces: gated reports with `execute_structural_actions()` as the only mutating topology boundary.

- [ ] Respect `unified_mesh_enabled` and `sleep_enabled`.
- [ ] Collect real signals from the latest training output/metric state.
- [ ] Keep `_evolution_death_loop` observer/proposer-only.
- [ ] Run the focused test to green.
- [ ] Run adjacent organism/topology regressions and inspect `git diff --check`.
