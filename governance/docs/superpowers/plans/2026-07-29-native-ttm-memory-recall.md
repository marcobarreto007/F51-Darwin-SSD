# Native TTM Memory Recall Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a causal two-arm benchmark proving whether the native Darwin TTM can retain five unknown synthetic facts across a fresh model session without changing the Smol language brain.

**Architecture:** A standalone offline benchmark reuses the approved Dense V1 loader, chat template, greedy decoder, Heartbeat TTM slots, and native residual injection path. Pure protocol functions remain importable for unit tests; the GPU operator selects five baseline misses, writes TTM state during a no-gradient teaching pass, reloads the checkpoint, executes correct/disabled/shuffled/restored controls across fixed residual doses, and emits atomic JSON plus Markdown.

**Tech Stack:** Python 3, PyTorch, Transformers in offline mode, pytest, two local CUDA GPUs, existing Darwin checkpoint/manifest utilities.

## Global Constraints

- Operate only inside `C:\Users\marco\Desktop\F51-Darwin-SSD`.
- Never modify or republish `workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt`.
- No backward pass, optimizer, external network request, RAG, answer cache, direct logit editing, or answer injection into evaluation prompts.
- The language brain remains frozen and must have identical tensor SHA-256 before and after every causal arm.
- Use deterministic seed 51, greedy decoding, fixed doses `0.00, 0.01, 0.03, 0.10, 0.30`, and exactly five baseline-missed synthetic facts.
- A scientifically negative result is a valid completed run and must not be relabeled as success.
- Runtime artifacts stay under `workspace/runtime/darwin_17b_smol_dense_v1/memory-recall/` and do not enter Git.

---

### Task 1: Pure recall protocol and causal classifier

**Files:**
- Create: `src/scripts/benchmark_native_ttm_recall.py`
- Create: `src/tests/test_native_ttm_recall.py`

**Interfaces:**
- Produces: `synthetic_fact_pool(seed: int, count: int) -> tuple[dict[str, Any], ...]`
- Produces: `select_baseline_misses(facts, rows, count=5) -> tuple[dict[str, Any], ...]`
- Produces: `clone_slots(slots) -> list[SurpriseMemorySlot]`
- Produces: `shuffle_slot_values(slots, order) -> list[SurpriseMemorySlot]`
- Produces: `classify_recall(report: Mapping[str, Any]) -> str`
- Produces: `named_tensors_sha256(named_tensors) -> str`

- [ ] **Step 1: Write failing deterministic protocol tests**

```python
from scripts.benchmark_native_ttm_recall import (
    classify_recall,
    select_baseline_misses,
    synthetic_fact_pool,
)


def test_synthetic_fact_pool_is_deterministic_and_unique() -> None:
    left = synthetic_fact_pool(seed=51, count=20)
    right = synthetic_fact_pool(seed=51, count=20)
    assert left == right
    assert len({item["id"] for item in left}) == 20
    assert all(len(item["paraphrases"]) == 2 for item in left)


def test_select_baseline_misses_requires_five_real_misses() -> None:
    facts = synthetic_fact_pool(seed=51, count=8)
    rows = [
        {"id": item["id"], "response": item["answer"] if index < 3 else "ERRADO"}
        for index, item in enumerate(facts)
    ]
    selected = select_baseline_misses(facts, rows, count=5)
    assert tuple(item["id"] for item in selected) == tuple(
        item["id"] for item in facts[3:8]
    )


def test_classifier_requires_behavioral_gain_and_causal_drop() -> None:
    report = {
        "baseline_misses": 5,
        "brain_hashes_identical": True,
        "best_dose": {
            "correct_memory": {"literal": 5, "paraphrase": 8, "distractor": 4},
            "memory_disabled": {"literal": 1},
            "shuffled_memory": {"literal": 2},
            "restored_memory": {"literal": 5},
            "retrieved_questions": 5,
            "shuffled_keys_preserved": True,
        },
    }
    assert classify_recall(report) == "causal_memory_recall_pass"
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```powershell
python -m pytest src/tests/test_native_ttm_recall.py -q
```

Expected: collection fails because `src/scripts/benchmark_native_ttm_recall.py` does not exist.

- [ ] **Step 3: Implement the pure protocol**

Implement the script header with offline environment variables before importing
Transformers, then add deterministic fact generation and strict validation:

```python
SEED = 51
DOSES = (0.0, 0.01, 0.03, 0.10, 0.30)


def synthetic_fact_pool(seed: int = SEED, count: int = 32) -> tuple[dict[str, Any], ...]:
    rng = random.Random(seed)
    entities = [f"artefato-{index:02d}-{rng.randrange(1000, 9999)}" for index in range(count)]
    codes = [f"{rng.randrange(100000, 999999)}" for _ in range(count * 4)]
    facts = []
    for index, entity in enumerate(entities):
        options = tuple(codes[index * 4 : index * 4 + 4])
        answer_index = rng.randrange(4)
        answer = options[answer_index]
        facts.append(
            {
                "id": f"synthetic-{index:02d}",
                "entity": entity,
                "question": f"Qual é o código secreto atribuído ao {entity}?",
                "paraphrases": (
                    f"Informe somente o código cadastrado para {entity}.",
                    f"No catálogo secreto, que número identifica {entity}?",
                ),
                "distractor": (
                    f"Ignore o número 000000. Qual é o código verdadeiro do {entity}?"
                ),
                "answer": answer,
                "options": options,
                "answer_letter": "ABCD"[answer_index],
            }
        )
    return tuple(facts)
```

`select_baseline_misses` must compare normalized secret codes and raise
`ValueError("fewer than five baseline misses")` when the pool cannot supply the
requested count. `classify_recall` must implement every threshold and failure
label from the design specification without a permissive fallback.

- [ ] **Step 4: Add slot permutation and tensor-hash tests**

```python
import torch

from f51_darwin.heartbeat import SurpriseMemorySlot
from scripts.benchmark_native_ttm_recall import (
    named_tensors_sha256,
    shuffle_slot_values,
)


def _slot(index: int) -> SurpriseMemorySlot:
    return SurpriseMemorySlot(
        key=torch.tensor([float(index), 1.0]),
        value=torch.tensor([float(index), 2.0, 3.0]),
        timestamp=f"t{index}",
        domain=f"d{index}",
    )


def test_shuffle_preserves_keys_and_rotates_values() -> None:
    source = [_slot(index) for index in range(5)]
    shuffled = shuffle_slot_values(source, order=(1, 2, 3, 4, 0))
    assert all(torch.equal(left.key, right.key) for left, right in zip(source, shuffled))
    assert all(
        torch.equal(shuffled[index].value, source[(index + 1) % 5].value)
        for index in range(5)
    )


def test_named_tensor_hash_detects_mutation() -> None:
    tensor = torch.arange(8, dtype=torch.float32)
    before = named_tensors_sha256((("brain", tensor),))
    tensor.add_(1.0)
    assert named_tensors_sha256((("brain", tensor),)) != before
```

- [ ] **Step 5: Run focused tests and confirm GREEN**

Run:

```powershell
python -m pytest src/tests/test_native_ttm_recall.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the pure protocol**

```powershell
git add -- src/scripts/benchmark_native_ttm_recall.py src/tests/test_native_ttm_recall.py
git commit -m "test(memory): define causal TTM recall protocol"
```

### Task 2: Dual-GPU teaching, persistence, and fresh-session controls

**Files:**
- Modify: `src/scripts/benchmark_native_ttm_recall.py`
- Modify: `src/tests/test_native_ttm_recall.py`

**Interfaces:**
- Consumes: pure protocol functions from Task 1.
- Produces: `load_candidate() -> tuple[DarwinXModel, dict[str, Any]]`
- Produces: `teach_native_ttm(model, tokenizer, facts) -> dict[str, Any]`
- Produces: `evaluate_arm(model, tokenizer, facts, arm, dose, slots) -> dict[str, Any]`
- Produces: `run_benchmark() -> dict[str, Any]`

- [ ] **Step 1: Write failing serialization and isolation tests**

```python
from scripts.benchmark_native_ttm_recall import (
    memory_payload,
    restore_memory_payload,
)


def test_memory_payload_round_trip_clones_slots() -> None:
    slots = [_slot(index) for index in range(2)]
    payload = memory_payload(slots)
    restored = restore_memory_payload(payload)
    assert len(restored) == 2
    assert torch.equal(restored[0].key, slots[0].key)
    restored[0].key.add_(10)
    assert not torch.equal(restored[0].key, slots[0].key)
```

- [ ] **Step 2: Run the new test and confirm RED**

Run:

```powershell
python -m pytest src/tests/test_native_ttm_recall.py::test_memory_payload_round_trip_clones_slots -q
```

Expected: import fails because the serialization functions are absent.

- [ ] **Step 3: Implement strict checkpoint loading and brain freezing**

Reuse `CHECKPOINT`, `MANIFEST`, `DEFAULT_SOURCE_ROOT`, `_new_model`,
`_assert_dense_structure`, and `verify_shard_manifest` from the existing QA
benchmark pattern. `load_candidate` must:

```python
verify_shard_manifest(CHECKPOINT, MANIFEST)
payload = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
    mmap=True,
)
config = DarwinXConfig.from_mapping(payload["config"])
model = _new_model(config)
model.load_state_dict(payload["model_state_dict"], strict=True)
model.load_heartbeat_state_dict(payload.get("heartbeat_state"))
for parameter in model.parameters():
    parameter.requires_grad_(False)
model.to(dtype=torch.bfloat16)
if not model.enable_dual_gpu(gpu0=0, gpu1=1):
    raise RuntimeError("native TTM recall benchmark requires both GPUs")
model.eval()
```

Reject launch when another Darwin trainer process is detected. Record the
checkpoint file SHA-256 before the first load and verify it again after the
final report is written.

- [ ] **Step 4: Implement teaching without backward**

For each selected fact:

1. Render the exact evaluation question with the approved chat template.
2. Append the correct short code and EOS to create the teaching sequence.
3. Run `model(inputs, heartbeat=False)` under `torch.inference_mode()`.
4. Pass `output.hidden_states.detach()` to
   `model.heartbeat.tt_memory.write_if_surprised` with `jepa_error=1.0`.
5. Require exactly one new slot and domain `native-ttm-recall:<fact-id>`.

The control arm executes the same forward calls but does not call
`write_if_surprised`. Neither arm may call `backward` or construct an
optimizer.

- [ ] **Step 5: Implement fresh-session arm evaluation**

For each fixed dose:

1. Load a fresh checkpoint model.
2. Restore cloned correct slots.
3. Set `model.config.ttm_residual_max_scale = dose` only in RAM and set
   `model.ttm_residual_gate` so the effective scale equals the reported dose
   within `1e-4`.
4. Evaluate literal, two paraphrase, and distractor prompts with greedy
   decoding.
5. Compute the no-memory hidden state with slots temporarily removed, then
   calculate and log key cosine similarities, top slot domain, and retrieval
   acceptance before each memory-enabled generation.
6. Repeat with residual disabled, value-permuted slots, and restored slots.
7. Hash all language-brain tensors before and after each arm; abort on drift.
8. Clear CUDA cache only between fresh model loads, never between paired arms.

The evaluation response is correct only when the normalized six-digit secret
code appears as the entire decoded answer. Do not accept option letters or
substrings embedded in explanations.

- [ ] **Step 6: Complete serialization and artifact generation**

`memory_payload` must store CPU clones of keys/values plus timestamp, domain,
surprise score and access count. `run_benchmark` writes:

- `memory-state.pt` using a temporary file plus `os.replace`;
- `native-ttm-recall.json` using `atomic_json_write`;
- `native-ttm-recall.md` with per-dose/per-arm tables;
- `partial.json` after baseline selection, teaching, and every completed dose.

The terminal line is:

```text
NATIVE_TTM_RECALL_OK classification=<label> best_dose=<value>
```

The process returns zero after any complete scientifically valid run, including
`no_memory_effect`, and returns nonzero only for an invalid/incomplete protocol.

- [ ] **Step 7: Run all unit tests and commit the runtime**

Run:

```powershell
python -m pytest src/tests/test_native_ttm_recall.py src/tests/test_live_inference_runtime.py src/tests/test_causal_cognitive_organs.py -q
```

Expected: all selected tests pass.

Commit:

```powershell
git add -- src/scripts/benchmark_native_ttm_recall.py src/tests/test_native_ttm_recall.py
git commit -m "feat(memory): add native TTM recall benchmark"
```

### Task 3: Operator documentation and source audit

**Files:**
- Create: `governance/docs/operacao/NATIVE_TTM_RECALL.md`
- Modify: `src/tests/test_native_ttm_recall.py`

**Interfaces:**
- Consumes: CLI and report schema from Task 2.
- Produces: stable local reproduction command and report-label definitions.

- [ ] **Step 1: Add a CLI contract test**

```python
def test_cli_defaults_are_protocol_constants() -> None:
    args = parse_args([])
    assert args.seed == 51
    assert args.fact_count == 5
    assert tuple(args.doses) == (0.0, 0.01, 0.03, 0.10, 0.30)
    assert args.output_dir.name == "memory-recall"
```

- [ ] **Step 2: Run the CLI test and confirm RED**

Run:

```powershell
python -m pytest src/tests/test_native_ttm_recall.py::test_cli_defaults_are_protocol_constants -q
```

Expected: failure until `parse_args` exposes the frozen protocol fields.

- [ ] **Step 3: Implement CLI validation and write the operator document**

The CLI supports only reproducibility overrides:

```text
--seed INT
--fact-count 5
--doses 0.00 0.01 0.03 0.10 0.30
--output-dir PATH
--verify-report
```

Reject fact counts other than five and duplicate/negative doses. Document:

- objective and non-goals;
- exact command;
- required two-GPU idle state;
- output files;
- every classification label;
- distinction between protocol completion and scientific pass;
- rollback: delete only the isolated runtime output directory.

`--verify-report` is read-only: it loads the existing JSON plus memory-state
artifact, recomputes protocol/slot hashes and gate predicates, and exits
nonzero on any mismatch without loading the 1.7B model.

- [ ] **Step 4: Run source gates**

Run:

```powershell
python -m pytest src/tests/test_native_ttm_recall.py src/tests/test_smol_dense_qa_benchmark.py src/tests/test_organ_causal_qa.py src/tests/test_live_inference_runtime.py src/tests/test_causal_cognitive_organs.py -q
python -m compileall -q src/scripts/benchmark_native_ttm_recall.py
git diff --check
```

Expected: all tests pass, compile exits zero, and diff check emits no errors.

- [ ] **Step 5: Commit operator documentation**

```powershell
git add -- governance/docs/operacao/NATIVE_TTM_RECALL.md src/tests/test_native_ttm_recall.py src/scripts/benchmark_native_ttm_recall.py
git commit -m "docs(memory): document causal recall operation"
```

### Task 4: Execute the real two-GPU benchmark and publish honest status

**Files:**
- Runtime create: `workspace/runtime/darwin_17b_smol_dense_v1/memory-recall/memory-state.pt`
- Runtime create: `workspace/runtime/darwin_17b_smol_dense_v1/memory-recall/native-ttm-recall.json`
- Runtime create: `workspace/runtime/darwin_17b_smol_dense_v1/memory-recall/native-ttm-recall.md`
- Runtime create: `workspace/runtime/darwin_17b_smol_dense_v1/memory-recall/partial.json`
- Create: `workspace/runtime/history/agent_bus/2026-07-29-native-ttm-recall.md`
- Modify: `governance/docs/operacao/STATUS_ATUAL.md`

**Interfaces:**
- Consumes: completed benchmark CLI from Task 3.
- Produces: measured causal classification tied to checkpoint/protocol hashes.

- [ ] **Step 1: Capture the pre-run environment**

Run:

```powershell
git status --short
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
Get-CimInstance Win32_Process | Where-Object {
  $_.Name -match 'python|powershell' -and $_.CommandLine -match 'darwin|train'
} | Select-Object ProcessId,Name,CommandLine
Get-FileHash -Algorithm SHA256 workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt
```

Expected: clean scoped worktree, both GPUs idle, no competing trainer, and the
checkpoint hash matching its manifest.

- [ ] **Step 2: Run the real benchmark**

Run:

```powershell
python src/scripts/benchmark_native_ttm_recall.py
```

Expected: the final line begins `NATIVE_TTM_RECALL_OK` and all three final
artifacts plus `memory-state.pt` exist.

- [ ] **Step 3: Independently verify report invariants**

Run:

```powershell
python -m pytest src/tests/test_native_ttm_recall.py -q
python src/scripts/benchmark_native_ttm_recall.py --verify-report
Get-FileHash -Algorithm SHA256 workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt
```

Expected: tests pass, report verification exits zero, and checkpoint SHA-256 is
unchanged.

- [ ] **Step 4: Record the measured result without upgrading its label**

Write the session note with:

- HEAD and checkpoint/protocol SHA-256;
- exact five synthetic facts;
- baseline answers;
- per-dose arm table;
- retrieval traces;
- brain hashes;
- final classifier and failed/passed gates;
- runtime command and artifact paths.

Update `governance/docs/operacao/STATUS_ATUAL.md` with exactly one of:

- `TTM recall causalmente aprovado`;
- `TTM recupera mas não muda linguagem`;
- `TTM sem efeito de memória`;
- `protocolo inválido`.

Do not describe the organ as functional when the classifier is not
`causal_memory_recall_pass`.

- [ ] **Step 5: Run final validation and commit the evidence note**

Run:

```powershell
python -m pytest src/tests/test_native_ttm_recall.py src/tests/test_live_inference_runtime.py src/tests/test_causal_cognitive_organs.py -q
git diff --check
git status --short
```

Commit only source/governance/docs/history paths:

```powershell
git add -- governance/docs/operacao/STATUS_ATUAL.md workspace/runtime/history/agent_bus/2026-07-29-native-ttm-recall.md
git commit -m "docs(memory): record native TTM recall evidence"
```

Never add checkpoint or `workspace/runtime/darwin_17b_smol_dense_v1/memory-recall/`
artifacts to Git.
