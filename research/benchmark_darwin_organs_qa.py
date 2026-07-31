#!/usr/bin/env python3
"""Causal QA adaptation benchmark with one Darwin organ per arm."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Sequence

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer

from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.transplant_16b.checkpoint import verify_shard_manifest
from f51_darwin.transplant_16b.cli import DEFAULT_SOURCE_ROOT
from f51_darwin.transplant_16b.dense_assembly import _assert_dense_structure
from f51_darwin.transplant_16b.exact_assembly import _new_model
from f51_darwin.transplant_16b.organ_causal_qa import (
    ORGAN_ARMS,
    OrganSelection,
    build_answer_only_example,
    classify_arm,
    language_brain_named_parameters,
    select_organ_parameters,
    shuffled_orders,
)
if __package__:
    from research.benchmark_smol_dense_qa import (
        CHECKPOINT,
        MANIFEST,
        MCQ,
        OPEN_QA,
        RUNTIME_ROOT,
        _atomic_json,
        _environment,
        _greedy,
        _prompt,
        accepted_answer,
        extract_choice,
        normalize_answer,
    )
else:
    from benchmark_smol_dense_qa import (
        CHECKPOINT,
        MANIFEST,
        MCQ,
        OPEN_QA,
        RUNTIME_ROOT,
        _atomic_json,
        _environment,
        _greedy,
        _prompt,
        accepted_answer,
        extract_choice,
        normalize_answer,
    )


JSON_REPORT = RUNTIME_ROOT / "organ-causal-qa.json"
MARKDOWN_REPORT = RUNTIME_ROOT / "organ-causal-qa.md"
PARTIAL_REPORT = RUNTIME_ROOT / "organ-causal-qa.partial.json"
SHUFFLE_SEEDS = tuple(range(51, 61))


def _items() -> tuple[dict[str, Any], ...]:
    return tuple(
        ({**item, "kind": "mcq"} for item in MCQ)
    ) + tuple(
        ({**item, "kind": "open"} for item in OPEN_QA)
    )


def _answer_text(item: dict[str, Any]) -> str:
    if item["kind"] == "mcq":
        return str(item["answer"])
    return str(item["accepted"][0])


def _teaching_example(
    tokenizer: Any,
    item: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt = _prompt(tokenizer, item, item["kind"])
    answer_ids = tokenizer.encode(
        _answer_text(item),
        add_special_tokens=False,
    )
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("QA tokenizer has no EOS token")
    return build_answer_only_example(
        prompt,
        answer_ids=answer_ids,
        eos_token_id=eos_token_id,
    )


def _load_model() -> tuple[Any, dict[str, Any]]:
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
    model.to(dtype=torch.bfloat16)
    if not model.enable_dual_gpu(gpu0=0, gpu1=1):
        raise RuntimeError("organ QA benchmark requires both GPUs")
    return model, payload


def _all_parameters(model: Any) -> tuple[torch.nn.Parameter, ...]:
    parameters = list(model.parameters())
    heartbeat = model.heartbeat
    if heartbeat is not None:
        for module in (
            heartbeat.ff_stack,
            heartbeat.tt_memory,
            heartbeat.thinker,
        ):
            parameters.extend(module.parameters())
    unique: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for parameter in parameters:
        if id(parameter) not in seen:
            unique.append(parameter)
            seen.add(id(parameter))
    return tuple(unique)


def _state_snapshot(model: Any) -> dict[str, Any]:
    heartbeat = model.heartbeat
    if heartbeat is None:
        return {
            "beats": 0,
            "memory_writes": 0,
            "memory_slots": 0,
            "thoughts": 0,
        }
    return {
        "beats": int(heartbeat.total_beats),
        "memory_writes": int(heartbeat.memory_writes),
        "memory_slots": len(heartbeat.tt_memory.slots),
        "thoughts": int(heartbeat.thoughts_generated),
    }


def _state_delta(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, int]:
    return {
        key: int(after[key]) - int(before[key])
        for key in before
    }


def _grade(
    item: dict[str, Any],
    response: str,
) -> tuple[str | None, bool, bool]:
    if item["kind"] == "mcq":
        parsed = extract_choice(response)
        return parsed, parsed == item["answer"], parsed is not None
    parsed = normalize_answer(response)
    return parsed, accepted_answer(response, item["accepted"]), bool(response.strip())


def _evaluate_order(
    model: Any,
    tokenizer: Any,
    ordered: Sequence[dict[str, Any]],
    *,
    heartbeat: bool,
) -> list[dict[str, Any]]:
    model.eval()
    rows = []
    for item in ordered:
        prompt = _prompt(tokenizer, item, item["kind"])
        tokens, elapsed = _greedy(
            lambda ids: model(ids, heartbeat=heartbeat).logits,
            prompt,
            device=torch.device("cuda:0"),
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=1 if item["kind"] == "mcq" else 8,
        )
        response = tokenizer.decode(
            tokens,
            skip_special_tokens=True,
        ).strip()
        parsed, correct, valid = _grade(item, response)
        rows.append(
            {
                "id": item["id"],
                "kind": item["kind"],
                "category": item["category"],
                "response": response,
                "parsed": parsed,
                "correct": correct,
                "valid": valid,
                "elapsed_seconds": elapsed,
            }
        )
    return rows


def _optimizer(selection: OrganSelection) -> torch.optim.Optimizer | None:
    gates = [
        parameter
        for name, parameter in selection.named
        if "gate" in name
    ]
    bodies = [
        parameter
        for name, parameter in selection.named
        if "gate" not in name
    ]
    groups = []
    if gates:
        groups.append({"params": gates, "lr": 0.05})
    if bodies:
        groups.append({"params": bodies, "lr": 0.0001})
    return torch.optim.SGD(groups) if groups else None


def _clip(parameters: Sequence[torch.nn.Parameter]) -> None:
    by_device: dict[torch.device, list[torch.nn.Parameter]] = {}
    for parameter in parameters:
        if parameter.grad is not None:
            by_device.setdefault(parameter.device, []).append(parameter)
    for group in by_device.values():
        torch.nn.utils.clip_grad_norm_(group, max_norm=1.0)


def _teach(
    model: Any,
    tokenizer: Any,
    selection: OrganSelection,
) -> dict[str, Any]:
    selected = [parameter for _, parameter in selection.named]
    optimizer = _optimizer(selection)
    nonzero_names: set[str] = set()
    max_gradient_norm = 0.0
    optimizer_steps = 0
    losses: list[float] = []
    stateful = selection.arm in {"heartbeat", "ttm"}
    for item in _items():
        inputs, labels = _teaching_example(tokenizer, item)
        inputs = inputs.to("cuda:0")
        labels = labels.to("cuda:0")
        model.train()
        for parameter in selected:
            parameter.grad = None
        output = model(
            inputs,
            labels=labels,
            heartbeat=(selection.arm == "heartbeat"),
            domain=f"organ-qa:{item['id']}",
        )
        if output.loss is None or not torch.isfinite(output.loss):
            raise ValueError(
                f"{selection.arm} produced non-finite teaching loss"
            )
        losses.append(float(output.loss.detach().float().item()))

        if selection.arm == "ttm":
            if model.heartbeat is None or output.hidden_states is None:
                raise ValueError("TTM arm has no Heartbeat hidden state")
            wrote = model.heartbeat.tt_memory.write_if_surprised(
                output.hidden_states.detach(),
                jepa_error=1.0,
                domain=f"organ-qa:{item['id']}",
            )
            if wrote:
                model.heartbeat.memory_writes += 1

        if optimizer is not None and output.loss.requires_grad:
            output.loss.backward()
            norms = []
            for name, parameter in selection.named:
                gradient = parameter.grad
                if (
                    gradient is not None
                    and torch.isfinite(gradient).all()
                    and bool(torch.count_nonzero(gradient).item())
                ):
                    nonzero_names.add(name)
                    norms.append(float(gradient.float().norm().item()))
            if norms:
                max_gradient_norm = max(max_gradient_norm, max(norms))
                _clip(selected)
                optimizer.step()
                optimizer_steps += 1
                if selection.arm == "gaba":
                    model.commit_gaba_observations(output)
        if stateful and selection.arm == "ttm":
            # TTM writes above are the native state update for this arm.
            pass
    del optimizer
    return {
        "nonzero_gradient_parameters": len(nonzero_names),
        "nonzero_gradient_names": sorted(nonzero_names),
        "max_gradient_norm": max_gradient_norm,
        "optimizer_steps": optimizer_steps,
        "loss_start": losses[0],
        "loss_end": losses[-1],
    }


def _changed_versions(
    before: dict[str, int],
    named: Sequence[tuple[str, torch.nn.Parameter]],
) -> list[str]:
    return [
        name
        for name, parameter in named
        if int(parameter._version) != int(before[name])
    ]


def _run_arm(
    arm: str,
    tokenizer: Any,
) -> dict[str, Any]:
    print(f"ORGAN_ARM_START arm={arm}", flush=True)
    torch.manual_seed(51)
    torch.cuda.manual_seed_all(51)
    model, payload = _load_model()
    for parameter in _all_parameters(model):
        parameter.requires_grad_(False)
        parameter.grad = None
    selection = select_organ_parameters(model, arm)
    for _, parameter in selection.named:
        parameter.requires_grad_(True)

    brain = language_brain_named_parameters(model)
    brain_before = {name: int(parameter._version) for name, parameter in brain}
    selected_before = {
        name: int(parameter._version) for name, parameter in selection.named
    }
    state_before = _state_snapshot(model)
    canonical = _items()
    pre_rows = _evaluate_order(
        model,
        tokenizer,
        canonical,
        heartbeat=False,
    )
    teaching = _teach(model, tokenizer, selection)
    state_after_teaching = _state_snapshot(model)

    items_by_id = {item["id"]: item for item in canonical}
    item_ids = tuple(items_by_id)
    orders = shuffled_orders(item_ids, seeds=SHUFFLE_SEEDS)
    stateful = arm in {"heartbeat", "ttm"}
    runs = []
    for seed, order in zip(SHUFFLE_SEEDS, orders, strict=True):
        rows = _evaluate_order(
            model,
            tokenizer,
            [items_by_id[item_id] for item_id in order],
            heartbeat=stateful,
        )
        runs.append(
            {
                "seed": seed,
                "accuracy": sum(row["correct"] for row in rows),
                "valid": sum(row["valid"] for row in rows),
                "rows": rows,
            }
        )
        print(
            f"ORGAN_ARM_SHUFFLE arm={arm} seed={seed} "
            f"accuracy={runs[-1]['accuracy']}/30",
            flush=True,
        )

    pre_by_id = {row["id"]: row for row in pre_rows}
    post_by_id = {
        item_id: [
            next(row for row in run["rows"] if row["id"] == item_id)
            for run in runs
        ]
        for item_id in item_ids
    }
    corrected_ids = [
        item_id
        for item_id in item_ids
        if not pre_by_id[item_id]["correct"]
        and all(row["correct"] for row in post_by_id[item_id])
    ]
    forgotten_ids = [
        item_id
        for item_id in item_ids
        if pre_by_id[item_id]["correct"]
        and any(not row["correct"] for row in post_by_id[item_id])
    ]
    changed_answer_ids = [
        item_id
        for item_id in item_ids
        if any(
            row["parsed"] != pre_by_id[item_id]["parsed"]
            for row in post_by_id[item_id]
        )
    ]
    brain_drift = _changed_versions(brain_before, brain)
    selected_changed = _changed_versions(
        selected_before,
        selection.named,
    )
    state_after = _state_snapshot(model)
    state_delta = _state_delta(state_before, state_after)
    state_changed = any(value != 0 for value in state_delta.values())
    accuracies = [int(run["accuracy"]) for run in runs]
    report = {
        "arm": arm,
        "selected_tensor_count": len(selection.named),
        "selected_parameter_count": selection.parameter_count,
        "selected_changed_count": len(selected_changed),
        "selected_changed_names": selected_changed,
        "frozen_brain_drift": len(brain_drift),
        "frozen_brain_drift_names": brain_drift,
        "pre_accuracy": sum(row["correct"] for row in pre_rows),
        "pre_rows": pre_rows,
        "post_accuracies": accuracies,
        "post_accuracy_mean": statistics.mean(accuracies),
        "post_accuracy_min": min(accuracies),
        "post_accuracy_max": max(accuracies),
        "valid_responses_min": min(int(run["valid"]) for run in runs),
        "corrected": len(corrected_ids),
        "corrected_ids": corrected_ids,
        "forgotten": len(forgotten_ids),
        "forgotten_ids": forgotten_ids,
        "changed_answer_ids": changed_answer_ids,
        "state_before": state_before,
        "state_after_teaching": state_after_teaching,
        "state_after": state_after,
        "state_delta": state_delta,
        "state_changed": state_changed,
        "runs": runs,
        **teaching,
    }
    report["classification"] = classify_arm(report)
    print(
        f"ORGAN_ARM_DONE arm={arm} class={report['classification']} "
        f"pre={report['pre_accuracy']}/30 "
        f"post_mean={report['post_accuracy_mean']:.2f}/30 "
        f"corrected={report['corrected']} forgotten={report['forgotten']} "
        f"grad={report['nonzero_gradient_parameters']} "
        f"brain_drift={report['frozen_brain_drift']}",
        flush=True,
    )
    del model, payload
    gc.collect()
    torch.cuda.empty_cache()
    return report


def _protocol_hash() -> str:
    payload = {
        "arms": ORGAN_ARMS,
        "items": _items(),
        "shuffle_seeds": SHUFFLE_SEEDS,
        "gate_lr": 0.05,
        "body_lr": 0.0001,
        "teaching_passes": 1,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# QA causal por órgão Darwin",
        "",
        f"Data: {payload['created_at']}",
        "",
        "| Órgão | Pré | Pós média (min–max) | Corrigiu | Esqueceu | Gradientes | Estado | Veredito |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm in payload["arms"]:
        delta = arm["state_delta"]
        state = (
            f"beats {delta['beats']}, slots {delta['memory_slots']}, "
            f"writes {delta['memory_writes']}"
        )
        lines.append(
            f"| {arm['arm']} | {arm['pre_accuracy']}/30 "
            f"| {arm['post_accuracy_mean']:.2f} "
            f"({arm['post_accuracy_min']}–{arm['post_accuracy_max']})/30 "
            f"| {arm['corrected']} | {arm['forgotten']} "
            f"| {arm['nonzero_gradient_parameters']} | {state} "
            f"| `{arm['classification']}` |"
        )
    lines.extend(
        [
            "",
            "## Contrato",
            "",
            "- Um checkpoint fresco por braço.",
            "- Cérebro Smol congelado.",
            "- Uma passagem de ensino na ordem canônica.",
            "- Dez avaliações embaralhadas, seeds 51–60, sem correção.",
            "- Nenhum braço é salvo como checkpoint.",
            "",
            "Reprodução:",
            "",
            "```powershell",
            "python research/benchmark_darwin_organs_qa.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teach and test one Darwin organ at a time."
    )
    parser.add_argument(
        "--arms",
        default=",".join(ORGAN_ARMS),
        help="Comma-separated subset of organ arms.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    arms = tuple(part.strip() for part in args.arms.split(",") if part.strip())
    if not arms or any(arm not in ORGAN_ARMS for arm in arms):
        raise ValueError(f"--arms must be a subset of {ORGAN_ARMS}")
    if len(set(arms)) != len(arms):
        raise ValueError("--arms contains duplicates")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("organ QA benchmark requires both local GPUs")
    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_SOURCE_ROOT,
        local_files_only=True,
    )
    checkpoint_sha = json.loads(MANIFEST.read_text(encoding="utf-8"))[
        "checkpoint_sha256"
    ]
    completed = []
    for arm in arms:
        completed.append(_run_arm(arm, tokenizer))
        partial = {
            "schema": "darwin-organ-causal-qa-partial-v1",
            "checkpoint_sha256": checkpoint_sha,
            "protocol_sha256": _protocol_hash(),
            "requested_arms": list(arms),
            "arms": completed,
        }
        _atomic_json(PARTIAL_REPORT, partial)
    payload = {
        "schema": "darwin-organ-causal-qa-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": checkpoint_sha,
        "protocol_sha256": _protocol_hash(),
        "shuffle_seeds": list(SHUFFLE_SEEDS),
        "teaching_passes": 1,
        "environment": _environment(),
        "arms": completed,
    }
    _atomic_json(JSON_REPORT, payload)
    MARKDOWN_REPORT.write_text(_markdown(payload), encoding="utf-8")
    if PARTIAL_REPORT.exists():
        PARTIAL_REPORT.unlink()
    invalid = [
        arm["arm"]
        for arm in completed
        if arm["classification"] == "invalid_brain_drift"
    ]
    if invalid:
        print(
            "ORGAN_CAUSAL_QA_INVALID brain_drift=" + ",".join(invalid),
            flush=True,
        )
        return 2
    print(
        "ORGAN_CAUSAL_QA_OK "
        + " ".join(
            f"{arm['arm']}={arm['classification']}:"
            f"{arm['post_accuracy_mean']:.2f}/30"
            for arm in completed
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
