#!/usr/bin/env python3
"""Hash-bound three-organ foundation shadow probe.

Loads the published Darwin-Smol Native Dense V1 checkpoint, builds a
shadow-only cognitive runtime alongside it, and proves that the three
organ adapters execute with zero gate and produce exactly equal logits.
This script has no checkpoint writer, optimizer, backward call, or
training launcher.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch

from f51_darwin.cognition import (
    CANONICAL_ORGAN_IDS,
    CognitiveForwardMetadata,
)
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.hashing import (
    atomic_json_write,
    sha256_file,
    tensor_sha256,
)
from f51_darwin.state_identity import backbone_identity
from f51_darwin.transplant_16b.checkpoint import verify_shard_manifest


REPORT_SCHEMA = "darwin-three-organ-foundation-shadow-v1"


def parse_token_ids(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(","))
    except ValueError as error:
        raise ValueError("token ids must be comma-separated integers") from error
    if len(values) < 2:
        raise ValueError("at least two token ids are required")
    if any(value < 0 for value in values):
        raise ValueError("token ids must be non-negative")
    return values


def validate_missing_keys(
    missing: list[str],
    unexpected: list[str],
) -> None:
    if unexpected:
        raise ValueError(f"unexpected checkpoint keys: {unexpected}")
    if not missing:
        raise ValueError("cognitive shadow load exposed no new state")
    foreign = [
        key
        for key in missing
        if not key.startswith("cognitive_runtime.")
    ]
    if foreign:
        raise ValueError(f"non-cognition missing keys: {foreign}")


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("wrong shadow report schema")
    if report["checkpoint_sha256"] != report["checkpoint_sha256_after"]:
        raise ValueError("checkpoint changed during shadow probe")
    if report["brain_identity_before"] != report["brain_identity_after"]:
        raise ValueError("brain identity changed during shadow probe")
    if float(report["max_abs_logit_error"]) != 0.0:
        raise ValueError("shadow logit error is not exactly zero")
    if int(report["pulse_events"]) != 1:
        raise ValueError("shadow must emit exactly one pulse event")
    if int(report["canonical_organ_count"]) != 3:
        raise ValueError("shadow manifest must contain exactly three organs")
    if report["all_gates_zero"] is not True:
        raise ValueError("a cognitive gate is not exactly zero")


def build_bfloat16_model(config: DarwinXConfig) -> DarwinXModel:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return DarwinXModel(config)
    finally:
        torch.set_default_dtype(previous)


def place_for_inference(
    model: DarwinXModel,
    mode: str,
) -> tuple[DarwinXModel, torch.device]:
    if mode == "dual":
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("dual mode requires two CUDA devices")
        model.to(dtype=torch.bfloat16)
        if not model.enable_dual_gpu(gpu0=0, gpu1=1):
            raise RuntimeError("DarwinXModel refused dual-GPU placement")
        return model, torch.device("cuda:0")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda mode requires CUDA")
        device = torch.device("cuda:0")
    elif mode == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"unsupported device mode: {mode}")
    model.to(device=device, dtype=torch.bfloat16)
    return model, device


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hash-bound three-organ foundation shadow probe."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--token-ids", required=True)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "dual"),
        default="dual",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    checkpoint = Path(args.checkpoint).resolve()
    manifest = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()
    token_ids = parse_token_ids(args.token_ids)
    expected_sha256 = str(args.expected_sha256).lower()
    if len(expected_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_sha256
    ):
        raise ValueError("expected SHA-256 must be 64 lowercase hex chars")

    checkpoint_sha_before = sha256_file(checkpoint)
    if checkpoint_sha_before != expected_sha256:
        raise ValueError("checkpoint SHA-256 mismatch")
    verify_shard_manifest(checkpoint, manifest)
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if payload.get("version") != 9:
        raise ValueError("foundation shadow probe requires checkpoint v9")
    base_config = DarwinXConfig.from_mapping(payload["config"])
    if any(token_id >= base_config.vocab_size for token_id in token_ids):
        raise ValueError("token id exceeds checkpoint vocabulary")
    input_cpu = torch.tensor([token_ids], dtype=torch.long)

    base_model = build_bfloat16_model(base_config)
    base_model.load_state_dict(payload["model_state_dict"], strict=True)
    base_model.eval()
    brain_before = backbone_identity(base_model.state_dict(), base_config)
    base_model, base_device = place_for_inference(base_model, args.device)
    with torch.inference_mode():
        base_logits = base_model(
            input_cpu.to(base_device),
            heartbeat=False,
        ).logits.cpu()
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cognitive_config = replace(
        base_config,
        cognitive_architecture_version="three_organs_v1",
        cognitive_shadow_enabled=True,
        cognitive_pulse_enabled=True,
        cognitive_organ_width=512,
    )
    cognitive_model = build_bfloat16_model(cognitive_config)
    missing, unexpected = cognitive_model.load_state_dict(
        payload["model_state_dict"],
        strict=False,
    )
    validate_missing_keys(list(missing), list(unexpected))
    cognitive_model.eval()
    brain_after = backbone_identity(
        cognitive_model.state_dict(),
        cognitive_config,
    )
    cognitive_model, cognitive_device = place_for_inference(
        cognitive_model,
        args.device,
    )
    metadata = CognitiveForwardMetadata(
        step_id=0,
        checkpoint_id="sha256:" + checkpoint_sha_before,
        context_digest="sha256:" + tensor_sha256(input_cpu),
    )
    with torch.inference_mode():
        output = cognitive_model(
            input_cpu.to(cognitive_device),
            heartbeat=False,
            cognitive_metadata=metadata,
        )
    cognitive_logits = output.logits.cpu()
    max_abs_error = float(
        (cognitive_logits.float() - base_logits.float()).abs().max()
    )
    runtime = cognitive_model.cognitive_runtime
    if runtime is None:
        raise RuntimeError("cognitive runtime was not constructed")
    report = {
        "schema": REPORT_SCHEMA,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha_before,
        "checkpoint_sha256_after": sha256_file(checkpoint),
        "brain_identity_before": brain_before,
        "brain_identity_after": brain_after,
        "max_abs_logit_error": max_abs_error,
        "pulse_events": len(output.cognitive_pulse_events),
        "pulse_head_sha256": runtime.pulse_state_dict()["head_sha256"],
        "canonical_organ_count": len(
            runtime.manifest()["organ_ids"]
        ),
        "canonical_organ_ids": list(CANONICAL_ORGAN_IDS),
        "all_gates_zero": bool(
            runtime.memory_adapter.gate.item() == 0.0
            and runtime.world_model_adapter.gate.item() == 0.0
        ),
        "missing_cognitive_keys": sorted(missing),
        "device_mode": args.device,
    }
    validate_report(report)
    atomic_json_write(report, output_path)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    print("THREE_ORGAN_FOUNDATION_SHADOW_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
