#!/usr/bin/env python3
"""Causal fresh-session recall benchmark for the native Darwin TTM.

The module keeps its protocol helpers importable without loading the 1.7B model.
The GPU runtime is implemented below those helpers and remains fully offline.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch

from f51_darwin.hashing import tensor_sha256
from f51_darwin.hashing import atomic_json_write, sha256_file
from f51_darwin.heartbeat import SurpriseMemorySlot


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "workspace/runtime/darwin_17b_smol_dense_v1/memory-recall"
)
DEFAULT_DOSES = (0.0, 0.01, 0.03, 0.10, 0.30)
DEFAULT_SEED = 51
DEFAULT_FACT_COUNT = 5
PROTOCOL_SCHEMA = "darwin-native-ttm-recall-v1"
_SECRET_PATTERN = re.compile(r"^\s*(\d{6})\s*$")
CHECKPOINT = (
    ROOT
    / "workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt"
)
MANIFEST = CHECKPOINT.with_suffix(".manifest.json")
SYSTEM_PROMPT = (
    "Você é um assistente preciso. Responda somente com o código de seis "
    "dígitos solicitado, sem explicação."
)


def _unique_codes(rng: random.Random, count: int) -> list[str]:
    codes: list[str] = []
    used: set[str] = set()
    while len(codes) < count:
        candidate = f"{rng.randrange(100000, 1_000_000):06d}"
        if candidate not in used:
            used.add(candidate)
            codes.append(candidate)
    return codes


def synthetic_fact_pool(
    seed: int = DEFAULT_SEED,
    count: int = 32,
) -> tuple[dict[str, Any], ...]:
    """Create deterministic, non-public six-digit entity associations."""
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    entity_suffixes = _unique_codes(rng, count)
    answers = _unique_codes(rng, count)
    facts: list[dict[str, Any]] = []
    for index, (suffix, answer) in enumerate(zip(entity_suffixes, answers, strict=True)):
        entity = f"artefato-{index:02d}-{suffix}"
        options = [
            answer,
            answers[(index + 1) % count],
            answers[(index + 2) % count],
            answers[(index + 3) % count],
        ]
        rng.shuffle(options)
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
                    "Ignore o código falso 000000. "
                    f"Qual é o código verdadeiro do {entity}?"
                ),
                "answer": answer,
                "options": tuple(options),
            }
        )
    return tuple(facts)


def normalize_secret(value: object) -> str | None:
    match = _SECRET_PATTERN.fullmatch(str(value))
    return match.group(1) if match else None


def select_baseline_misses(
    facts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    count: int = DEFAULT_FACT_COUNT,
) -> tuple[dict[str, Any], ...]:
    """Select facts whose baseline response is not the exact secret."""
    rows_by_id = {str(row["id"]): row for row in rows}
    if len(rows_by_id) != len(rows):
        raise ValueError("baseline rows contain duplicate ids")
    selected: list[dict[str, Any]] = []
    for fact in facts:
        fact_id = str(fact["id"])
        if fact_id not in rows_by_id:
            raise ValueError(f"missing baseline row for {fact_id}")
        response = normalize_secret(rows_by_id[fact_id].get("response", ""))
        if response != str(fact["answer"]):
            selected.append(dict(fact))
            if len(selected) == count:
                return tuple(selected)
    raise ValueError("fewer than five baseline misses")


def clone_slots(
    slots: Iterable[SurpriseMemorySlot],
) -> list[SurpriseMemorySlot]:
    return [
        SurpriseMemorySlot(
            key=slot.key.detach().cpu().clone(),
            value=slot.value.detach().cpu().clone(),
            timestamp=str(slot.timestamp),
            domain=str(slot.domain),
            surprise_score=float(slot.surprise_score),
            access_count=int(slot.access_count),
        )
        for slot in slots
    ]


def shuffle_slot_values(
    slots: Sequence[SurpriseMemorySlot],
    *,
    order: Sequence[int],
) -> list[SurpriseMemorySlot]:
    """Preserve retrieval keys/domains while permuting stored values."""
    if len(order) != len(slots) or sorted(order) != list(range(len(slots))):
        raise ValueError("order must be a permutation of slot indices")
    if len(slots) > 1 and all(index == source for index, source in enumerate(order)):
        raise ValueError("order must not be the identity permutation")
    result = clone_slots(slots)
    source = clone_slots(slots)
    for target_index, source_index in enumerate(order):
        result[target_index].value = source[source_index].value.clone()
    return result


def memory_payload(
    slots: Sequence[SurpriseMemorySlot],
) -> dict[str, Any]:
    cloned = clone_slots(slots)
    return {
        "schema": "darwin-native-ttm-memory-state-v1",
        "slot_count": len(cloned),
        "key_shape": list(cloned[0].key.shape) if cloned else None,
        "value_shape": list(cloned[0].value.shape) if cloned else None,
        "slots": [
            {
                "key": slot.key,
                "value": slot.value,
                "timestamp": slot.timestamp,
                "domain": slot.domain,
                "surprise_score": slot.surprise_score,
                "access_count": slot.access_count,
            }
            for slot in cloned
        ],
    }


def restore_memory_payload(
    payload: Mapping[str, Any],
) -> list[SurpriseMemorySlot]:
    if payload.get("schema") != "darwin-native-ttm-memory-state-v1":
        raise ValueError("unsupported memory payload schema")
    raw_slots = payload.get("slots")
    if not isinstance(raw_slots, list):
        raise ValueError("memory payload slots must be a list")
    if int(payload.get("slot_count", -1)) != len(raw_slots):
        raise ValueError("memory payload slot count mismatch")
    restored: list[SurpriseMemorySlot] = []
    for raw in raw_slots:
        if not isinstance(raw, Mapping):
            raise ValueError("memory slot must be a mapping")
        key = raw.get("key")
        value = raw.get("value")
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise ValueError("memory slot key/value must be tensors")
        restored.append(
            SurpriseMemorySlot(
                key=key.detach().cpu().clone(),
                value=value.detach().cpu().clone(),
                timestamp=str(raw.get("timestamp", "")),
                domain=str(raw.get("domain", "")),
                surprise_score=float(raw.get("surprise_score", 0.0)),
                access_count=int(raw.get("access_count", 0)),
            )
        )
    if restored:
        expected_key_shape = tuple(payload.get("key_shape") or ())
        expected_value_shape = tuple(payload.get("value_shape") or ())
        key_shapes = {tuple(slot.key.shape) for slot in restored}
        value_shapes = {tuple(slot.value.shape) for slot in restored}
        if len(key_shapes) != 1 or key_shapes != {expected_key_shape}:
            raise ValueError("inconsistent key shapes")
        if len(value_shapes) != 1 or value_shapes != {expected_value_shape}:
            raise ValueError("inconsistent value shapes")
    return restored


def named_tensors_sha256(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
) -> str:
    digest = hashlib.sha256()
    count = 0
    for name, tensor in named_tensors:
        count += 1
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor_sha256(tensor).encode("ascii"))
        digest.update(b"\0")
    if count == 0:
        raise ValueError("cannot hash an empty tensor collection")
    return digest.hexdigest()


def _correct_retrieval_count(arm: Mapping[str, Any]) -> int:
    explicit = arm.get("correct_retrieved_questions")
    if explicit is not None:
        return int(explicit)
    rows = arm.get("rows")
    if not isinstance(rows, Sequence):
        return 0
    return sum(
        1
        for row in rows
        if isinstance(row, Mapping)
        and row.get("variant") == "literal"
        and row.get("retrieval", {}).get("top_domain")
        == f"native-ttm-recall:{row.get('id')}"
    )


def _dose_passes(dose: Mapping[str, Any]) -> bool:
    correct = dose["correct_memory"]
    disabled = dose["memory_disabled"]
    shuffled = dose["shuffled_memory"]
    restored = dose["restored_memory"]
    return (
        int(correct["literal"]) >= 4
        and int(correct["paraphrase"]) >= 6
        and int(correct["distractor"]) >= 3
        and int(correct["literal"]) - int(disabled["literal"]) >= 2
        and int(correct["literal"]) - int(shuffled["literal"]) >= 2
        and int(restored["literal"]) >= int(correct["literal"]) - 1
        and _correct_retrieval_count(correct) >= 4
        and bool(shuffled.get("keys_preserved", False))
    )


def classify_recall(report: Mapping[str, Any]) -> str:
    """Classify scientific outcome without upgrading partial evidence."""
    native_diagnostic = report.get("native_diagnostic")
    if isinstance(native_diagnostic, Mapping):
        diagnostic_verdict = str(
            native_diagnostic.get("overall_verdict", "")
        )
        if diagnostic_verdict == "inconclusive_decoder_floor":
            return "invalid_protocol"
        if diagnostic_verdict in {
            "native_retrieval_absent",
            "native_injection_misses_decoder",
        }:
            return "no_memory_effect"
        if (
            diagnostic_verdict == "native_channel_reaches_decoder"
            and report.get("failure_mode")
            == "value_not_behaviorally_decodable"
        ):
            return "no_memory_effect"
    if (
        not bool(report.get("protocol_valid", False))
        or int(report.get("baseline_misses", 0)) != DEFAULT_FACT_COUNT
        or not bool(report.get("brain_hashes_identical", False))
    ):
        return "invalid_protocol"
    doses = report.get("doses")
    if not isinstance(doses, Sequence) or not doses:
        return "invalid_protocol"
    if any(_dose_passes(dose) for dose in doses):
        return "causal_memory_recall_pass"
    best_literal = max(
        int(dose["correct_memory"].get("literal", 0)) for dose in doses
    )
    if best_literal >= 4:
        semantic = any(
            int(dose["correct_memory"].get("paraphrase", 0)) >= 6
            and int(dose["correct_memory"].get("distractor", 0)) >= 3
            for dose in doses
            if int(dose["correct_memory"].get("literal", 0)) >= 4
        )
        if not semantic:
            return "literal_only"
        return "noncausal_gain"
    if any(
        _correct_retrieval_count(dose["correct_memory"]) >= 4
        for dose in doses
    ):
        return "retrieval_only"
    return "no_memory_effect"


def protocol_sha256(
    *,
    seed: int,
    fact_count: int,
    doses: Sequence[float],
    facts: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    body = {
        "schema": PROTOCOL_SCHEMA,
        "seed": int(seed),
        "fact_count": int(fact_count),
        "doses": [float(value) for value in doses],
        "facts": [dict(item) for item in facts] if facts is not None else None,
    }
    return hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _variant_question(fact: Mapping[str, Any], variant: str) -> str:
    if variant == "literal":
        question = str(fact["question"])
    elif variant.startswith("paraphrase-"):
        index = int(variant.rsplit("-", 1)[1])
        question = str(fact["paraphrases"][index])
    elif variant == "distractor":
        question = str(fact["distractor"])
    else:
        raise ValueError(f"unknown prompt variant: {variant}")
    options = "\n".join(
        f"- {option}" for option in fact["options"]
    )
    return (
        f"{question}\nCódigos possíveis:\n{options}\n"
        "Responda somente com os seis dígitos do código correto."
    )


def _prompt_ids(
    tokenizer: Any,
    fact: Mapping[str, Any],
    variant: str,
) -> torch.Tensor:
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _variant_question(fact, variant),
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids


def _teaching_ids(tokenizer: Any, fact: Mapping[str, Any]) -> torch.Tensor:
    prompt = _prompt_ids(tokenizer, fact, "literal")
    answer_ids = tokenizer.encode(
        str(fact["answer"]),
        add_special_tokens=False,
    )
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("tokenizer has no EOS token")
    suffix = torch.tensor(
        [answer_ids + [int(eos_token_id)]],
        dtype=prompt.dtype,
    )
    return torch.cat((prompt, suffix), dim=1)


def association_sources(
    prompt_hidden: torch.Tensor,
    teaching_hidden: torch.Tensor,
    *,
    prompt_length: int,
    answer_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract an aligned prompt key and teacher-forced answer value."""
    if prompt_hidden.ndim != 3 or teaching_hidden.ndim != 3:
        raise ValueError("hidden states must have shape [batch, sequence, d_model]")
    if prompt_hidden.size(0) != 1 or teaching_hidden.size(0) != 1:
        raise ValueError("association extraction requires batch size 1")
    if prompt_hidden.size(2) != teaching_hidden.size(2):
        raise ValueError("prompt and teaching hidden sizes must match")
    if prompt_length < 1 or prompt_length != prompt_hidden.size(1):
        raise ValueError("prompt_length must match the prompt hidden sequence")
    if answer_length < 1:
        raise ValueError("answer_length must be positive")
    answer_end = prompt_length + answer_length
    if answer_end > teaching_hidden.size(1):
        raise ValueError("answer span exceeds the teaching hidden sequence")
    if not bool(torch.isfinite(prompt_hidden).all()) or not bool(
        torch.isfinite(teaching_hidden).all()
    ):
        raise ValueError("association hidden states must be finite")
    key = prompt_hidden[:, prompt_length - 1, :]
    value = teaching_hidden[:, prompt_length:answer_end, :].mean(dim=1)
    return key, value


@torch.inference_mode()
def _forward_raw_hidden(model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    """Capture the normalized hidden before Spider and TTM modify readout."""
    captured: list[torch.Tensor] = []

    def capture_norm(
        _module: Any,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured.append(output.detach().clone())

    handle = model.norm.register_forward_hook(capture_norm)
    try:
        model(input_ids.to("cuda:0"), heartbeat=False)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError(
            f"expected one raw hidden capture, observed {len(captured)}"
        )
    return captured[0]


@torch.inference_mode()
def _greedy_response(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 8,
) -> tuple[str, float]:
    generated = input_ids.to("cuda:0")
    new_tokens: list[int] = []
    started = time.perf_counter()
    for _ in range(max_new_tokens):
        output = model(generated, heartbeat=False)
        next_token = int(output.logits[0, -1].float().argmax().item())
        if tokenizer.eos_token_id is not None and next_token == tokenizer.eos_token_id:
            break
        new_tokens.append(next_token)
        generated = torch.cat(
            (
                generated,
                torch.tensor(
                    [[next_token]],
                    dtype=torch.long,
                    device=generated.device,
                ),
            ),
            dim=1,
        )
    elapsed = time.perf_counter() - started
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), elapsed


def _environment() -> dict[str, Any]:
    try:
        import psutil

        ram_bytes = int(psutil.virtual_memory().total)
    except Exception:
        ram_bytes = 0
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": ram_bytes,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ],
    }


def _competing_trainers() -> list[dict[str, Any]]:
    try:
        import psutil
    except Exception:
        return []
    markers = (
        "darwin_organism.py",
        "start_100m",
        "start_overnight",
        "run247",
        "organism.training",
    )
    current_pid = os.getpid()
    found: list[dict[str, Any]] = []
    for process in psutil.process_iter(("pid", "name", "cmdline")):
        try:
            if int(process.info["pid"]) == current_pid:
                continue
            command = " ".join(process.info.get("cmdline") or [])
            lowered = command.lower()
            if any(marker in lowered for marker in markers):
                found.append(
                    {
                        "pid": int(process.info["pid"]),
                        "name": str(process.info.get("name") or ""),
                        "command": command,
                    }
                )
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return found


def load_candidate() -> tuple[Any, dict[str, Any]]:
    from f51_darwin.darwin_x_core.config import DarwinXConfig
    from f51_darwin.transplant_16b.checkpoint import verify_shard_manifest
    from f51_darwin.transplant_16b.dense_assembly import _assert_dense_structure
    from f51_darwin.transplant_16b.exact_assembly import _new_model

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
    _assert_dense_structure(model, expected_layers=config.n_layers)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    model.to(dtype=torch.bfloat16)
    if not model.enable_dual_gpu(gpu0=0, gpu1=1):
        raise RuntimeError("native TTM recall benchmark requires both GPUs")
    model.eval()
    return model, payload


def _release_model(model: Any, payload: dict[str, Any]) -> None:
    del model, payload
    gc.collect()
    torch.cuda.empty_cache()


def _language_brain(model: Any) -> tuple[tuple[str, torch.Tensor], ...]:
    from f51_darwin.transplant_16b.organ_causal_qa import (
        language_brain_named_parameters,
    )

    return tuple(language_brain_named_parameters(model))


def _brain_hash(model: Any) -> str:
    return named_tensors_sha256(_language_brain(model))


def _brain_versions(model: Any) -> dict[str, int]:
    return {
        name: int(parameter._version)
        for name, parameter in _language_brain(model)
    }


def _set_runtime_ttm_max_scale(model: Any, value: float) -> None:
    """Override a frozen config field in RAM without touching checkpoint state."""

    object.__setattr__(
        model.config,
        "ttm_residual_max_scale",
        float(value),
    )


def _changed_versions(
    expected: Mapping[str, int],
    model: Any,
) -> list[str]:
    current = _brain_versions(model)
    return [
        name
        for name, version in expected.items()
        if current.get(name) != version
    ]


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _memory_state_sha256(payload: Mapping[str, Any]) -> str:
    slots = restore_memory_payload(payload)
    digest = hashlib.sha256()
    digest.update(str(payload["schema"]).encode("utf-8"))
    for slot in slots:
        for value in (
            slot.domain,
            slot.timestamp,
            f"{slot.surprise_score:.17g}",
            str(slot.access_count),
            tensor_sha256(slot.key),
            tensor_sha256(slot.value),
        ):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _evaluate_baseline_pool(
    model: Any,
    tokenizer: Any,
    facts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if model.heartbeat is None:
        raise ValueError("candidate has no Heartbeat/TTM")
    original_slots = clone_slots(model.heartbeat.tt_memory.slots)
    original_max_scale = float(model.config.ttm_residual_max_scale)
    model.heartbeat.tt_memory.slots = []
    _set_runtime_ttm_max_scale(model, 0.0)
    try:
        for fact in facts:
            response, elapsed = _greedy_response(
                model,
                tokenizer,
                _prompt_ids(tokenizer, fact, "literal"),
            )
            rows.append(
                {
                    "id": fact["id"],
                    "response": response,
                    "parsed": normalize_secret(response),
                    "reference": fact["answer"],
                    "correct": normalize_secret(response) == fact["answer"],
                    "elapsed_seconds": elapsed,
                }
            )
            print(
                "NATIVE_TTM_BASELINE "
                f"id={fact['id']} response={response!r} "
                f"correct={rows[-1]['correct']}",
                flush=True,
            )
    finally:
        model.heartbeat.tt_memory.slots = original_slots
        _set_runtime_ttm_max_scale(model, original_max_scale)
    return rows


@torch.inference_mode()
def _process_teaching_sequences(
    model: Any,
    tokenizer: Any,
    facts: Sequence[Mapping[str, Any]],
    *,
    write_memory: bool,
) -> dict[str, Any]:
    if model.heartbeat is None:
        raise ValueError("candidate has no Heartbeat/TTM")
    memory = model.heartbeat.tt_memory
    written: list[dict[str, Any]] = []
    for fact in facts:
        existing = clone_slots(memory.slots)
        memory.slots = []
        prompt = _prompt_ids(tokenizer, fact, "literal").to("cuda:0")
        teaching = _teaching_ids(tokenizer, fact).to("cuda:0")
        answer_length = int(teaching.size(1) - prompt.size(1) - 1)
        try:
            prompt_hidden = _forward_raw_hidden(model, prompt)
            teaching_hidden = _forward_raw_hidden(model, teaching)
            key_source, value_source = association_sources(
                prompt_hidden,
                teaching_hidden,
                prompt_length=int(prompt.size(1)),
                answer_length=answer_length,
            )
        finally:
            memory.slots = existing
        before = len(memory.slots)
        if write_memory:
            wrote = memory.write_association(
                key_source,
                value_source,
                jepa_error=1.0,
                domain=f"native-ttm-recall:{fact['id']}",
            )
            if not wrote or len(memory.slots) != before + 1:
                raise ValueError(f"failed to write one slot for {fact['id']}")
            written.append(
                {
                    "id": fact["id"],
                    "domain": memory.slots[-1].domain,
                    "key_sha256": tensor_sha256(memory.slots[-1].key),
                    "value_sha256": tensor_sha256(memory.slots[-1].value),
                    "prompt_length": int(prompt.size(1)),
                    "answer_length": answer_length,
                }
            )
        elif len(memory.slots) != before:
            raise ValueError("control teaching mutated TTM slots")
    return {
        "write_memory": write_memory,
        "teaching_examples": len(facts),
        "slots_written": len(written),
        "written": written,
    }


def _set_effective_dose(model: Any, dose: float) -> float:
    if model.ttm_residual_gate is None:
        raise ValueError("candidate has no TTM residual gate")
    _set_runtime_ttm_max_scale(model, dose)
    with torch.no_grad():
        model.ttm_residual_gate.fill_(8.0)
    effective = float(model.ttm_residual_scale().detach().float().cpu())
    # The gate is BF16 in the transplanted organism. Its quantization error is
    # roughly 0.4% around the tested doses, so a 1% bound is strict but honest.
    tolerance = max(1e-7, abs(float(dose)) * 0.01)
    if not math.isclose(effective, float(dose), rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(
            f"effective TTM dose mismatch: requested={dose} actual={effective}"
        )
    return effective


@torch.inference_mode()
def _retrieval_trace(
    model: Any,
    input_ids: torch.Tensor,
    slots: Sequence[SurpriseMemorySlot],
) -> dict[str, Any]:
    if model.heartbeat is None:
        raise ValueError("candidate has no Heartbeat/TTM")
    memory = model.heartbeat.tt_memory
    live_slots = clone_slots(memory.slots)
    live_scale = float(model.config.ttm_residual_max_scale)
    memory.slots = []
    _set_runtime_ttm_max_scale(model, 0.0)
    try:
        raw_hidden = _forward_raw_hidden(model, input_ids)
        query = raw_hidden[:, -1, :]
    finally:
        memory.slots = live_slots
        _set_runtime_ttm_max_scale(model, live_scale)
    if not slots:
        return {
            "accepted": False,
            "top_domain": None,
            "top_similarity": None,
            "top": [],
        }
    projected = memory.proj_key(query)
    keys = torch.stack(
        [
            slot.key.to(device=projected.device, dtype=projected.dtype)
            for slot in slots
        ]
    )
    similarities = torch.nn.functional.cosine_similarity(
        projected.unsqueeze(1),
        keys.unsqueeze(0),
        dim=-1,
    )[0]
    count = 1
    values, indices = similarities.topk(count)
    top = [
        {
            "slot_index": int(index),
            "domain": slots[int(index)].domain,
            "similarity": float(similarity.float().cpu()),
        }
        for similarity, index in zip(values, indices, strict=True)
    ]
    accepted = bool(top and top[0]["similarity"] >= 0.5)
    return {
        "accepted": accepted,
        "top_domain": top[0]["domain"] if top else None,
        "top_similarity": top[0]["similarity"] if top else None,
        "top": top,
    }


def _arm_slots(
    arm: str,
    correct_slots: Sequence[SurpriseMemorySlot],
    shuffled_slots: Sequence[SurpriseMemorySlot],
) -> list[SurpriseMemorySlot]:
    if arm == "no_memory":
        return []
    if arm == "shuffled_memory":
        return clone_slots(shuffled_slots)
    if arm in {"correct_memory", "memory_disabled", "restored_memory"}:
        return clone_slots(correct_slots)
    raise ValueError(f"unknown memory arm: {arm}")


def _prompt_variants(
    facts: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], str]]:
    result: list[tuple[Mapping[str, Any], str]] = []
    for fact in facts:
        result.append((fact, "literal"))
        result.append((fact, "paraphrase-0"))
        result.append((fact, "paraphrase-1"))
        result.append((fact, "distractor"))
    return result


def evaluate_arm(
    model: Any,
    tokenizer: Any,
    facts: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    dose: float,
    correct_slots: Sequence[SurpriseMemorySlot],
    shuffled_slots: Sequence[SurpriseMemorySlot],
) -> dict[str, Any]:
    if model.heartbeat is None:
        raise ValueError("candidate has no Heartbeat/TTM")
    slots = _arm_slots(arm, correct_slots, shuffled_slots)
    model.heartbeat.tt_memory.slots = slots
    requested_dose = 0.0 if arm in {"no_memory", "memory_disabled"} else float(dose)
    effective_dose = _set_effective_dose(model, requested_dose)
    brain_hash_before = _brain_hash(model)
    versions_before = _brain_versions(model)
    rows: list[dict[str, Any]] = []
    for fact, variant in _prompt_variants(facts):
        prompt = _prompt_ids(tokenizer, fact, variant)
        trace = _retrieval_trace(model, prompt, slots)
        response, elapsed = _greedy_response(model, tokenizer, prompt)
        parsed = normalize_secret(response)
        rows.append(
            {
                "id": fact["id"],
                "variant": variant,
                "response": response,
                "parsed": parsed,
                "reference": fact["answer"],
                "correct": parsed == fact["answer"],
                "elapsed_seconds": elapsed,
                "retrieval": trace,
            }
        )
        print(
            "NATIVE_TTM_ARM "
            f"arm={arm} dose={dose:.2f} id={fact['id']} "
            f"variant={variant} response={response!r} "
            f"correct={rows[-1]['correct']} "
            f"retrieved={trace['accepted']}",
            flush=True,
        )
    changed_versions = _changed_versions(versions_before, model)
    brain_hash_after = _brain_hash(model)
    if changed_versions or brain_hash_after != brain_hash_before:
        raise ValueError(
            f"language brain drift in arm {arm}: {changed_versions[:5]}"
        )
    literal_rows = [row for row in rows if row["variant"] == "literal"]
    paraphrase_rows = [
        row for row in rows if row["variant"].startswith("paraphrase-")
    ]
    distractor_rows = [row for row in rows if row["variant"] == "distractor"]
    latencies = [float(row["elapsed_seconds"]) for row in rows]
    ordered_latencies = sorted(latencies)
    p95_index = max(0, math.ceil(0.95 * len(ordered_latencies)) - 1)
    return {
        "arm": arm,
        "requested_dose": float(dose),
        "effective_dose": effective_dose,
        "literal": sum(bool(row["correct"]) for row in literal_rows),
        "literal_total": len(literal_rows),
        "paraphrase": sum(bool(row["correct"]) for row in paraphrase_rows),
        "paraphrase_total": len(paraphrase_rows),
        "distractor": sum(bool(row["correct"]) for row in distractor_rows),
        "distractor_total": len(distractor_rows),
        "retrieved_questions": sum(
            bool(row["retrieval"]["accepted"]) for row in literal_rows
        ),
        "correct_retrieved_questions": _correct_retrieval_count(
            {"rows": literal_rows}
        ),
        "slot_count": len(slots),
        "brain_hash_before": brain_hash_before,
        "brain_hash_after": brain_hash_after,
        "brain_hash_identical": brain_hash_before == brain_hash_after,
        "changed_brain_versions": changed_versions,
        "latency_seconds": {
            "median": statistics.median(latencies),
            "p95": ordered_latencies[p95_index],
            "total": sum(latencies),
        },
        "rows": rows,
    }


def _slots_keys_preserved(
    correct: Sequence[SurpriseMemorySlot],
    shuffled: Sequence[SurpriseMemorySlot],
) -> bool:
    return len(correct) == len(shuffled) and all(
        tensor_sha256(left.key) == tensor_sha256(right.key)
        and left.domain == right.domain
        for left, right in zip(correct, shuffled, strict=True)
    )


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Native TTM Memory Recall",
        "",
        f"Classificação: `{report['classification']}`",
        "",
        f"- Checkpoint SHA-256: `{report['checkpoint_sha256']}`",
        f"- Protocolo SHA-256: `{report['protocol_sha256']}`",
        f"- Memória SHA-256: `{report['memory_state_sha256']}`",
        f"- Fatos selecionados: {report['baseline_misses']}",
        f"- Hash cerebral invariável: {report['brain_hashes_identical']}",
        "",
        "## Fatos e baseline",
        "",
        "| ID | Entidade | Resposta secreta | Resposta baseline |",
        "|---|---|---:|---|",
    ]
    baseline_by_id = {
        row["id"]: row for row in report["baseline"]["rows"]
    }
    for fact in report["facts"]:
        row = baseline_by_id[fact["id"]]
        lines.append(
            f"| {fact['id']} | {fact['entity']} | {fact['answer']} | "
            f"`{row['response']}` |"
        )
    lines.extend(
        [
            "",
            "## Controles por dose",
            "",
            "| Dose | Correta L/P/D | Desligada L | Embaralhada L | "
            "Restaurada L | Recuperadas |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dose in report["doses"]:
        correct = dose["correct_memory"]
        disabled = dose["memory_disabled"]
        shuffled = dose["shuffled_memory"]
        restored = dose["restored_memory"]
        lines.append(
            f"| {dose['dose']:.2f} | "
            f"{correct['literal']}/{correct['paraphrase']}/"
            f"{correct['distractor']} | {disabled['literal']} | "
            f"{shuffled['literal']} | {restored['literal']} | "
            f"{correct['retrieved_questions']}/5 |"
        )
    no_memory = report["no_memory"]
    lines.extend(
        [
            "",
            "## Controle A sem memória",
            "",
            f"- Literal: {no_memory['literal']}/{no_memory['literal_total']}",
            f"- Paráfrases: {no_memory['paraphrase']}/"
            f"{no_memory['paraphrase_total']}",
            f"- Distrações: {no_memory['distractor']}/"
            f"{no_memory['distractor_total']}",
            "",
            "## Leitura",
            "",
            "- `causal_memory_recall_pass`: lembrança comportamental causal.",
            (
                "- `retrieval_only`: o domínio correto foi recuperado, mas a "
                "linguagem não melhorou."
            ),
            (
                "- Causa `native_retrieval_absent`: o caminho real da geração "
                "não aceitou nenhum slot."
            ),
            (
                "- Causa `native_injection_misses_decoder`: houve recuperação, "
                "mas ela não alterou o logit da última posição."
            ),
            (
                "- Causa `inconclusive_decoder_floor`: o modelo não executou "
                "nem o controle com o fato explícito no contexto."
            ),
            "- `literal_only`: repetiu perguntas literais sem generalizar.",
            "- `noncausal_gain`: ganho não desapareceu nos controles.",
            "- `no_memory_effect`: nenhum ganho e nenhuma recuperação útil.",
            "- `invalid_protocol`: isolamento, hashes ou artefatos falharam.",
            "",
            "Reprodução:",
            "",
            "```powershell",
            "python research/benchmark_native_ttm_recall.py",
            "python research/benchmark_native_ttm_recall.py --verify-report",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _best_dose(report: Mapping[str, Any]) -> dict[str, Any]:
    doses = list(report["doses"])
    passing = [dose for dose in doses if _dose_passes(dose)]
    candidates = passing or doses
    return max(
        candidates,
        key=lambda dose: (
            int(dose["correct_memory"]["literal"]),
            int(dose["correct_memory"]["paraphrase"]),
            int(dose["correct_memory"]["distractor"]),
            -float(dose["dose"]),
        ),
    )


def _partial_write(
    output_dir: Path,
    *,
    stage: str,
    body: Mapping[str, Any],
) -> None:
    payload = {
        "schema": "darwin-native-ttm-recall-partial-v1",
        "stage": stage,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **dict(body),
    }
    atomic_json_write(payload, output_dir / "partial.json")


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("native TTM recall benchmark requires both local GPUs")
    competing = _competing_trainers()
    if competing:
        raise RuntimeError(f"competing Darwin trainer detected: {competing}")
    from transformers import AutoTokenizer

    from f51_darwin.transplant_16b.cli import DEFAULT_SOURCE_ROOT

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_sha_before = sha256_file(CHECKPOINT)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected_checkpoint_sha = str(manifest["checkpoint_sha256"])
    if checkpoint_sha_before != expected_checkpoint_sha:
        raise ValueError("checkpoint SHA-256 does not match manifest")
    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_SOURCE_ROOT,
        local_files_only=True,
    )
    pool = synthetic_fact_pool(
        seed=int(args.seed),
        count=max(16, int(args.fact_count) * 3),
    )

    print("NATIVE_TTM_STAGE baseline", flush=True)
    baseline_model, baseline_payload = load_candidate()
    baseline_brain_hash = _brain_hash(baseline_model)
    baseline_rows = _evaluate_baseline_pool(
        baseline_model,
        tokenizer,
        pool,
    )
    facts = select_baseline_misses(
        pool,
        baseline_rows,
        count=int(args.fact_count),
    )
    selected_ids = {fact["id"] for fact in facts}
    selected_baseline = [
        row for row in baseline_rows if row["id"] in selected_ids
    ]
    control_versions = _brain_versions(baseline_model)
    control_teaching = _process_teaching_sequences(
        baseline_model,
        tokenizer,
        facts,
        write_memory=False,
    )
    control_changed = _changed_versions(control_versions, baseline_model)
    control_brain_after = _brain_hash(baseline_model)
    if control_changed or control_brain_after != baseline_brain_hash:
        raise ValueError("control teaching mutated the language brain")
    _partial_write(
        output_dir,
        stage="baseline_selected",
        body={
            "checkpoint_sha256": checkpoint_sha_before,
            "facts": list(facts),
            "baseline": selected_baseline,
            "control_teaching": control_teaching,
        },
    )
    _release_model(baseline_model, baseline_payload)
    baseline_model = None
    baseline_payload = None

    print("NATIVE_TTM_STAGE teaching", flush=True)
    teacher_model, teacher_payload = load_candidate()
    teacher_brain_before = _brain_hash(teacher_model)
    teacher_versions = _brain_versions(teacher_model)
    teaching = _process_teaching_sequences(
        teacher_model,
        tokenizer,
        facts,
        write_memory=True,
    )
    if teacher_model.heartbeat is None:
        raise ValueError("teacher has no Heartbeat/TTM")
    correct_slots = clone_slots(teacher_model.heartbeat.tt_memory.slots)
    if len(correct_slots) != int(args.fact_count):
        raise ValueError("teaching did not produce exactly five TTM slots")
    teacher_changed = _changed_versions(teacher_versions, teacher_model)
    teacher_brain_after = _brain_hash(teacher_model)
    if teacher_changed or teacher_brain_after != teacher_brain_before:
        raise ValueError("memory teaching mutated the language brain")
    persisted_memory = memory_payload(correct_slots)
    persisted_memory["checkpoint_sha256"] = checkpoint_sha_before
    persisted_memory["facts"] = [dict(fact) for fact in facts]
    memory_state_sha = _memory_state_sha256(persisted_memory)
    persisted_memory["memory_state_sha256"] = memory_state_sha
    memory_path = output_dir / "memory-state.pt"
    _atomic_torch_save(persisted_memory, memory_path)
    _partial_write(
        output_dir,
        stage="memory_written",
        body={
            "checkpoint_sha256": checkpoint_sha_before,
            "facts": list(facts),
            "teaching": teaching,
            "memory_state_sha256": memory_state_sha,
        },
    )
    _release_model(teacher_model, teacher_payload)
    teacher_model = None
    teacher_payload = None

    restored_payload = torch.load(
        memory_path,
        map_location="cpu",
        weights_only=False,
    )
    restored_slots = restore_memory_payload(restored_payload)
    shuffle_order = tuple(range(1, len(restored_slots))) + (0,)
    shuffled_slots = shuffle_slot_values(
        restored_slots,
        order=shuffle_order,
    )
    keys_preserved = _slots_keys_preserved(restored_slots, shuffled_slots)
    if not keys_preserved:
        raise ValueError("shuffled control changed TTM retrieval keys")

    no_memory: dict[str, Any] | None = None
    dose_reports: list[dict[str, Any]] = []
    all_arm_hashes_identical = True
    for dose_index, dose in enumerate(args.doses):
        print(f"NATIVE_TTM_STAGE dose={dose:.2f}", flush=True)
        model, payload = load_candidate()
        session_hash = _brain_hash(model)
        if dose_index == 0:
            no_memory = evaluate_arm(
                model,
                tokenizer,
                facts,
                arm="no_memory",
                dose=0.0,
                correct_slots=restored_slots,
                shuffled_slots=shuffled_slots,
            )
            all_arm_hashes_identical = (
                all_arm_hashes_identical
                and bool(no_memory["brain_hash_identical"])
            )
        arms: dict[str, Any] = {}
        for arm in (
            "correct_memory",
            "memory_disabled",
            "shuffled_memory",
            "restored_memory",
        ):
            arm_report = evaluate_arm(
                model,
                tokenizer,
                facts,
                arm=arm,
                dose=float(dose),
                correct_slots=restored_slots,
                shuffled_slots=shuffled_slots,
            )
            if arm == "shuffled_memory":
                arm_report["keys_preserved"] = keys_preserved
                arm_report["value_permutation"] = list(shuffle_order)
            arms[arm] = arm_report
            all_arm_hashes_identical = (
                all_arm_hashes_identical
                and bool(arm_report["brain_hash_identical"])
                and arm_report["brain_hash_before"] == session_hash
            )
        dose_report = {"dose": float(dose), **arms}
        dose_reports.append(dose_report)
        _partial_write(
            output_dir,
            stage=f"dose_{dose:.2f}_complete",
            body={
                "checkpoint_sha256": checkpoint_sha_before,
                "facts": list(facts),
                "doses": dose_reports,
                "no_memory": no_memory,
            },
        )
        _release_model(model, payload)
        model = None
        payload = None

    if no_memory is None:
        raise ValueError("no-memory control was not executed")
    selected_protocol_sha = protocol_sha256(
        seed=int(args.seed),
        fact_count=int(args.fact_count),
        doses=tuple(float(value) for value in args.doses),
        facts=facts,
    )
    checkpoint_sha_after = sha256_file(CHECKPOINT)
    report: dict[str, Any] = {
        "schema": PROTOCOL_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": checkpoint_sha_before,
        "checkpoint_sha256_after": checkpoint_sha_after,
        "checkpoint_unchanged": checkpoint_sha_after == checkpoint_sha_before,
        "protocol_sha256": selected_protocol_sha,
        "seed": int(args.seed),
        "configured_doses": [float(value) for value in args.doses],
        "baseline_misses": len(facts),
        "facts": [dict(fact) for fact in facts],
        "baseline": {
            "rows": selected_baseline,
            "brain_hash": baseline_brain_hash,
        },
        "control_teaching": {
            **control_teaching,
            "brain_hash_after": control_brain_after,
            "changed_brain_versions": control_changed,
        },
        "memory_teaching": {
            **teaching,
            "brain_hash_before": teacher_brain_before,
            "brain_hash_after": teacher_brain_after,
            "changed_brain_versions": teacher_changed,
        },
        "memory_state_path": str(memory_path),
        "memory_state_sha256": memory_state_sha,
        "memory_slot_count": len(restored_slots),
        "no_memory": no_memory,
        "doses": dose_reports,
        "brain_hashes_identical": all_arm_hashes_identical,
        "protocol_valid": (
            checkpoint_sha_after == checkpoint_sha_before
            and len(facts) == DEFAULT_FACT_COUNT
            and len(restored_slots) == DEFAULT_FACT_COUNT
            and keys_preserved
            and all_arm_hashes_identical
        ),
        "competing_trainers": competing,
        "environment": _environment(),
    }
    report["classification"] = classify_recall(report)
    report["strict_gate_passed"] = (
        report["classification"] == "causal_memory_recall_pass"
    )
    best = _best_dose(report)
    report["best_dose"] = float(best["dose"])
    json_path = output_dir / "native-ttm-recall.json"
    markdown_path = output_dir / "native-ttm-recall.md"
    atomic_json_write(report, json_path)
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    print(
        "NATIVE_TTM_RECALL_OK "
        f"classification={report['classification']} "
        f"best_dose={report['best_dose']:.2f}",
        flush=True,
    )
    return report


def verify_report(output_dir: Path) -> dict[str, Any]:
    report_path = output_dir / "native-ttm-recall.json"
    memory_path = output_dir / "memory-state.pt"
    if not report_path.is_file() or not memory_path.is_file():
        raise ValueError("recall report or memory state is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported recall report schema")
    recomputed_protocol = protocol_sha256(
        seed=int(report["seed"]),
        fact_count=int(report["baseline_misses"]),
        doses=tuple(float(value) for value in report["configured_doses"]),
        facts=report["facts"],
    )
    if recomputed_protocol != report["protocol_sha256"]:
        raise ValueError("protocol SHA-256 mismatch")
    state = torch.load(memory_path, map_location="cpu", weights_only=False)
    restore_memory_payload(state)
    recomputed_memory = _memory_state_sha256(state)
    if recomputed_memory != report["memory_state_sha256"]:
        raise ValueError("memory state SHA-256 mismatch")
    native_diagnostic = report.get("native_diagnostic")
    if isinstance(native_diagnostic, Mapping):
        diagnostic_path = Path(str(native_diagnostic["path"]))
        if not diagnostic_path.is_file():
            raise ValueError("native TTM diagnostic artifact is missing")
        diagnostic_sha = sha256_file(diagnostic_path)
        if diagnostic_sha != native_diagnostic["sha256"]:
            raise ValueError("native TTM diagnostic SHA-256 mismatch")
        diagnostic = json.loads(
            diagnostic_path.read_text(encoding="utf-8")
        )
        if diagnostic.get("schema") != "darwin-native-ttm-diagnostic-v1":
            raise ValueError("unsupported native TTM diagnostic schema")
        if (
            diagnostic.get("overall_verdict")
            != native_diagnostic.get("overall_verdict")
            or diagnostic.get("checkpoint_sha256")
            != report.get("checkpoint_sha256")
            or not bool(diagnostic.get("brain_unchanged", False))
        ):
            raise ValueError("native TTM diagnostic does not match recall report")
    recomputed_classification = classify_recall(report)
    if recomputed_classification != report["classification"]:
        raise ValueError("recall classification mismatch")
    checkpoint_sha = sha256_file(Path(report["checkpoint"]))
    if checkpoint_sha != report["checkpoint_sha256"]:
        raise ValueError("checkpoint changed after recall benchmark")
    return {
        "classification": recomputed_classification,
        "strict_gate_passed": (
            recomputed_classification == "causal_memory_recall_pass"
        ),
        "protocol_sha256": recomputed_protocol,
        "memory_state_sha256": recomputed_memory,
        "checkpoint_sha256": checkpoint_sha,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prove or refute native Darwin TTM recall causally."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fact-count", type=int, default=DEFAULT_FACT_COUNT)
    parser.add_argument(
        "--doses",
        type=float,
        nargs="+",
        default=list(DEFAULT_DOSES),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--verify-report", action="store_true")
    args = parser.parse_args(argv)
    if args.fact_count != DEFAULT_FACT_COUNT:
        raise ValueError("fact-count must be exactly 5")
    if len(set(args.doses)) != len(args.doses):
        raise ValueError("doses must be unique")
    if any(value < 0.0 for value in args.doses):
        raise ValueError("doses must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify_report:
        verified = verify_report(args.output_dir)
        print(
            json.dumps(
                {
                    "status": "verified",
                    "report": str(args.output_dir / "native-ttm-recall.json"),
                    "classification": verified["classification"],
                    "strict_gate_passed": verified["strict_gate_passed"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    report = run_benchmark(args)
    print(
        json.dumps(
            {
                "status": "complete",
                "report": str(args.output_dir / "native-ttm-recall.json"),
                "classification": report["classification"],
                "strict_gate_passed": report["strict_gate_passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
