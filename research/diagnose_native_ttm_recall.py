#!/usr/bin/env python3
"""Measure the decoder floor and the real native TTM injection path."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch

from f51_darwin.hashing import atomic_json_write, sha256_file
from f51_darwin.heartbeat import SurpriseMemorySlot, bound_memory_residual
try:
    from research.benchmark_native_ttm_recall import (
        CHECKPOINT,
        DEFAULT_OUTPUT_DIR,
        SYSTEM_PROMPT,
        _brain_versions,
        _changed_versions,
        _greedy_response,
        _markdown_report,
        _prompt_ids,
        _set_effective_dose,
        _variant_question,
        clone_slots,
        load_candidate,
        normalize_secret,
        restore_memory_payload,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from benchmark_native_ttm_recall import (  # type: ignore[no-redef]
        CHECKPOINT,
        DEFAULT_OUTPUT_DIR,
        SYSTEM_PROMPT,
        _brain_versions,
        _changed_versions,
        _greedy_response,
        _markdown_report,
        _prompt_ids,
        _set_effective_dose,
        _variant_question,
        clone_slots,
        load_candidate,
        normalize_secret,
        restore_memory_payload,
    )


DIAGNOSTIC_SCHEMA = "darwin-native-ttm-diagnostic-v1"
DEFAULT_DOSE = 0.30


def ceiling_user_text(fact: Mapping[str, Any]) -> str:
    return (
        f"Fato fornecido: o código secreto do {fact['entity']} é "
        f"{fact['answer']}.\n\n{_variant_question(fact, 'literal')}"
    )


def ceiling_prompt_ids(tokenizer: Any, fact: Mapping[str, Any]) -> torch.Tensor:
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ceiling_user_text(fact)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids


def channel_verdict(rows: Sequence[Mapping[str, Any]]) -> str:
    if not any(bool(row["native_memory_retrieved"]) for row in rows):
        return "native_retrieval_absent"
    if all(float(row["last_logits_max_abs_diff"]) == 0.0 for row in rows):
        return "native_injection_misses_decoder"
    return "native_channel_reaches_decoder"


def overall_verdict(
    *,
    ceiling_correct: int,
    ceiling_total: int,
    channel: str,
) -> str:
    required = max(1, math.ceil(0.8 * ceiling_total))
    if ceiling_correct < required:
        return "inconclusive_decoder_floor"
    return channel


def scientific_classification(diagnostic_verdict: str) -> str:
    if diagnostic_verdict == "inconclusive_decoder_floor":
        return "invalid_protocol"
    if diagnostic_verdict in {
        "native_retrieval_absent",
        "native_injection_misses_decoder",
    }:
        return "no_memory_effect"
    return "unresolved_native_channel"


def corrected_recall_payload(
    recall: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    *,
    diagnostic_path: Path,
    diagnostic_sha256: str,
) -> dict[str, Any]:
    updated = dict(recall)
    updated.setdefault(
        "classification_before_native_diagnostic",
        recall.get("classification"),
    )
    updated["native_diagnostic"] = {
        "schema": diagnostic["schema"],
        "path": str(diagnostic_path),
        "sha256": diagnostic_sha256,
        "decoder_floor_correct": int(
            diagnostic["decoder_floor"]["correct"]
        ),
        "decoder_floor_total": int(
            diagnostic["decoder_floor"]["total"]
        ),
        "channel_verdict": diagnostic["channel_verdict"],
        "overall_verdict": diagnostic["overall_verdict"],
        "brain_unchanged": bool(diagnostic["brain_unchanged"]),
    }
    failure_mode = str(diagnostic["overall_verdict"])
    source_classification = str(recall.get("classification", "invalid_protocol"))
    if failure_mode == "native_channel_reaches_decoder":
        if source_classification in {"retrieval_only", "no_memory_effect"}:
            updated["classification"] = "no_memory_effect"
            updated["failure_mode"] = "value_not_behaviorally_decodable"
        else:
            updated["classification"] = source_classification
            updated["failure_mode"] = failure_mode
    else:
        updated["classification"] = scientific_classification(failure_mode)
        updated["failure_mode"] = failure_mode
    updated["strict_gate_passed"] = (
        updated["classification"] == "causal_memory_recall_pass"
    )
    return updated


def reconcile_existing(output_dir: Path) -> dict[str, Any]:
    recall_path = output_dir / "native-ttm-recall.json"
    diagnostic_path = output_dir / "native-ttm-diagnostic.json"
    recall = json.loads(recall_path.read_text(encoding="utf-8"))
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if diagnostic.get("schema") != DIAGNOSTIC_SCHEMA:
        raise ValueError("unsupported native TTM diagnostic schema")
    if diagnostic.get("checkpoint_sha256") != recall.get("checkpoint_sha256"):
        raise ValueError("diagnostic and recall checkpoint SHA-256 differ")
    if not bool(diagnostic.get("brain_unchanged", False)):
        raise ValueError("diagnostic observed language-brain drift")
    updated = corrected_recall_payload(
        recall,
        diagnostic,
        diagnostic_path=diagnostic_path.resolve(),
        diagnostic_sha256=sha256_file(diagnostic_path),
    )
    atomic_json_write(updated, recall_path)
    (output_dir / "native-ttm-recall.md").write_text(
        _markdown_report(updated),
        encoding="utf-8",
    )
    return updated


@torch.inference_mode()
def evaluate_decoder_floor(
    model: Any,
    tokenizer: Any,
    facts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if model.heartbeat is None:
        raise ValueError("candidate has no Heartbeat/TTM")
    model.heartbeat.tt_memory.slots = []
    _set_effective_dose(model, 0.0)
    rows: list[dict[str, Any]] = []
    for fact in facts:
        response, elapsed = _greedy_response(
            model,
            tokenizer,
            ceiling_prompt_ids(tokenizer, fact),
        )
        parsed = normalize_secret(response)
        rows.append(
            {
                "id": fact["id"],
                "entity": fact["entity"],
                "reference": fact["answer"],
                "response": response,
                "parsed": parsed,
                "correct": parsed == fact["answer"],
                "elapsed_seconds": elapsed,
            }
        )
        print(
            "NATIVE_TTM_CEILING "
            f"id={fact['id']} response={response!r} "
            f"correct={rows[-1]['correct']}",
            flush=True,
        )
    correct = sum(bool(row["correct"]) for row in rows)
    return {
        "correct": correct,
        "total": len(rows),
        "floor_passed": correct >= math.ceil(0.8 * len(rows)),
        "rows": rows,
    }


def _similarity_trace(
    memory: Any,
    query: torch.Tensor,
    slots: Sequence[SurpriseMemorySlot],
    *,
    already_pooled: bool,
) -> list[dict[str, Any]]:
    pooled = query if already_pooled else query.mean(dim=1)
    projected = memory.proj_key(pooled)
    keys = torch.stack(
        [
            slot.key.to(
                device=projected.device,
                dtype=projected.dtype,
            )
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
    return [
        {
            "slot_index": int(index),
            "domain": slots[int(index)].domain,
            "similarity": float(value.float().cpu()),
        }
        for value, index in zip(values, indices, strict=True)
    ]


@torch.inference_mode()
def probe_native_channel(
    model: Any,
    tokenizer: Any,
    fact: Mapping[str, Any],
    slots: Sequence[SurpriseMemorySlot],
    *,
    dose: float,
) -> dict[str, Any]:
    if model.heartbeat is None:
        raise ValueError("candidate lacks native TTM channel")
    prompt = _prompt_ids(tokenizer, fact, "literal").to("cuda:0")
    memory = model.heartbeat.tt_memory

    memory.slots = []
    _set_effective_dose(model, 0.0)
    control = model(prompt, heartbeat=False)

    memory.slots = clone_slots(slots)
    effective_dose = _set_effective_dose(model, dose)
    retrieval_calls: list[dict[str, Any]] = []
    original_retrieve = memory.retrieve

    def traced_retrieve(
        query: torch.Tensor,
        top_k: int = 4,
        min_similarity: float = 0.5,
        already_pooled: bool = False,
    ) -> torch.Tensor | None:
        top = _similarity_trace(
            memory,
            query,
            memory.slots,
            already_pooled=already_pooled,
        )
        result = original_retrieve(
            query,
            top_k=top_k,
            min_similarity=min_similarity,
            already_pooled=already_pooled,
        )
        bounded = (
            bound_memory_residual(result, query)
            if result is not None
            else None
        )
        retrieval_calls.append(
            {
                "already_pooled": already_pooled,
                "query_shape": list(query.shape),
                "top": top,
                "top_domain": top[0]["domain"] if top else None,
                "top_similarity": top[0]["similarity"] if top else None,
                "returned": result is not None,
                "returned_norm": (
                    float(result.float().norm().cpu())
                    if result is not None
                    else 0.0
                ),
                "bounded_returned_norm": (
                    float(bounded.float().norm().cpu())
                    if bounded is not None
                    else 0.0
                ),
                "query_norm": float(query.float().norm().cpu()),
            }
        )
        return result

    object.__setattr__(memory, "retrieve", traced_retrieve)
    try:
        active = model(prompt, heartbeat=False)
    finally:
        object.__delattr__(memory, "retrieve")

    if len(retrieval_calls) != 1:
        raise ValueError(
            f"native T-1 readout must retrieve once, got {len(retrieval_calls)}"
        )
    last_position = int(prompt.size(1) - 1)
    for trace in retrieval_calls:
        trace["position"] = last_position
        trace["is_last_position"] = True
        trace["correct_domain"] = (
            trace["top_domain"] == f"native-ttm-recall:{fact['id']}"
        )

    injection_norm = float(retrieval_calls[0]["bounded_returned_norm"])
    scaled_injection_norm = abs(effective_dose) * injection_norm
    hidden_norm = float(retrieval_calls[0]["query_norm"])
    last_control = control.logits[0, -1].float()
    last_active = active.logits[0, -1].float()
    all_diff = (active.logits.float() - control.logits.float()).abs()
    last_diff = (last_active - last_control).abs()
    return {
        "id": fact["id"],
        "prompt_tokens": int(prompt.size(1)),
        "last_position": last_position,
        "entity_positions": [last_position],
        "last_position_selected": True,
        "injection_position_contract": "T-1",
        "native_memory_retrieved": bool(active.ttm_memory_retrieved),
        # DarwinXOutput exposes retrieval, while the applied flag remains in
        # heartbeat_stats only when heartbeat telemetry is requested. This
        # probe runs heartbeat=False, so the model contract makes application
        # equivalent to a successful retrieval under a non-zero effective dose.
        "native_residual_applied": bool(
            active.ttm_memory_retrieved and effective_dose != 0.0
        ),
        "effective_dose": effective_dose,
        "injection_norm": injection_norm,
        "scaled_injection_norm": scaled_injection_norm,
        "hidden_norm": hidden_norm,
        "scaled_injection_to_hidden_ratio": (
            scaled_injection_norm / hidden_norm if hidden_norm else None
        ),
        "correct_domain_top_count": sum(
            bool(trace["correct_domain"]) for trace in retrieval_calls
        ),
        "retrieval_calls": retrieval_calls,
        "control_argmax_token": int(last_control.argmax().item()),
        "active_argmax_token": int(last_active.argmax().item()),
        "last_logits_max_abs_diff": float(last_diff.max().cpu()),
        "last_logits_l2_diff": float(last_diff.norm().cpu()),
        "all_logits_max_abs_diff": float(all_diff.max().cpu()),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Diagnóstico do TTM nativo",
        "",
        f"Veredito geral: `{report['overall_verdict']}`",
        f"Canal: `{report['channel_verdict']}`",
        "",
        "## Piso do decoder",
        "",
        (
            f"- Com o fato explícito no prompt: "
            f"{report['decoder_floor']['correct']}/"
            f"{report['decoder_floor']['total']}"
        ),
        "",
        "| ID | Esperado | Resposta | Acerto |",
        "|---|---:|---:|---|",
    ]
    for row in report["decoder_floor"]["rows"]:
        lines.append(
            f"| {row['id']} | {row['reference']} | `{row['response']}` | "
            f"{row['correct']} |"
        )
    lines.extend(
        [
            "",
            "## Canal nativo",
            "",
            (
                "| ID | T-1 selecionado | Retrieve | Top correto | "
                "Δlogit T-1 | Razão resíduo/hidden |"
            ),
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in report["channel"]["rows"]:
        lines.append(
            f"| {row['id']} | {row['last_position_selected']} | "
            f"{row['native_memory_retrieved']} | "
            f"{row['correct_domain_top_count']}/"
            f"{len(row['retrieval_calls'])} | "
            f"{row['last_logits_max_abs_diff']:.8g} | "
            f"{row['scaled_injection_to_hidden_ratio']:.8g} |"
        )
    return "\n".join(lines) + "\n"


def run(output_dir: Path, dose: float) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from f51_darwin.transplant_16b.cli import DEFAULT_SOURCE_ROOT

    recall_path = output_dir / "native-ttm-recall.json"
    memory_path = output_dir / "memory-state.pt"
    recall = json.loads(recall_path.read_text(encoding="utf-8"))
    memory_state = torch.load(memory_path, map_location="cpu", weights_only=False)
    slots = restore_memory_payload(memory_state)
    facts = tuple(recall["facts"])
    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_SOURCE_ROOT,
        local_files_only=True,
    )

    model, payload = load_candidate()
    versions_before = _brain_versions(model)
    try:
        decoder_floor = evaluate_decoder_floor(model, tokenizer, facts)
        channel_rows = [
            probe_native_channel(
                model,
                tokenizer,
                fact,
                slots,
                dose=dose,
            )
            for fact in facts
        ]
        changed_versions = _changed_versions(versions_before, model)
        if changed_versions:
            raise ValueError(
                f"language brain drift during diagnostic: {changed_versions[:5]}"
            )
        channel = channel_verdict(channel_rows)
        report = {
            "schema": DIAGNOSTIC_SCHEMA,
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "source_recall_report": str(recall_path),
            "source_classification": recall["classification"],
            "dose": float(dose),
            "decoder_floor": decoder_floor,
            "channel": {"verdict": channel, "rows": channel_rows},
            "channel_verdict": channel,
            "overall_verdict": overall_verdict(
                ceiling_correct=int(decoder_floor["correct"]),
                ceiling_total=int(decoder_floor["total"]),
                channel=channel,
            ),
            "changed_brain_versions": changed_versions,
            "brain_unchanged": not changed_versions,
        }
        atomic_json_write(report, output_dir / "native-ttm-diagnostic.json")
        (output_dir / "native-ttm-diagnostic.md").write_text(
            _markdown(report),
            encoding="utf-8",
        )
        reconcile_existing(output_dir)
        return report
    finally:
        del model, payload
        gc.collect()
        torch.cuda.empty_cache()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose decoder floor and native TTM channel."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dose", type=float, default=DEFAULT_DOSE)
    parser.add_argument("--reconcile-existing", action="store_true")
    args = parser.parse_args(argv)
    if args.dose <= 0.0:
        raise ValueError("diagnostic dose must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.reconcile_existing:
        report = reconcile_existing(args.output_dir)
        print(
            json.dumps(
                {
                    "status": "reconciled",
                    "classification": report["classification"],
                    "artifact": str(
                        args.output_dir / "native-ttm-recall.json"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    report = run(args.output_dir, float(args.dose))
    print(
        json.dumps(
            {
                "status": "complete",
                "decoder_floor": (
                    f"{report['decoder_floor']['correct']}/"
                    f"{report['decoder_floor']['total']}"
                ),
                "channel_verdict": report["channel_verdict"],
                "overall_verdict": report["overall_verdict"],
                "artifact": str(
                    args.output_dir / "native-ttm-diagnostic.json"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
