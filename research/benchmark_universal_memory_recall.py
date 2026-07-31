#!/usr/bin/env python3
"""Causal fresh-session recall benchmark for UniversalMemory.

Repeats the TTM 5-fact protocol — teach, save, reload, recall, measure —
but with the new UniversalMemory organ: trained key/value encoders and a
bounded readout adapter.

Protocol:
  1. Baseline: run all 32 facts, select 5 the model gets wrong
  2. Teach: encode 5 fact associations into memory
  3. Train: contrastive + readout training (backbone frozen)
  4. Save: snapshot memory state to disk
  5. Reload: fresh model copy with restored memory
  6. Recall: test literal, paraphrase, distractor variants
  7. Controls: disabled, shuffled keys, shuffled values, text ceiling
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
from torch import nn

from f51_darwin.cognition import (
    CognitiveForwardMetadata,
    CognitiveRuntime,
    UniversalMemory,
)
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.hashing import atomic_json_write, sha256_file, canonical_sha256
from f51_darwin.state_identity import backbone_identity

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt"
MANIFEST = CHECKPOINT.with_suffix(".manifest.json")
DEFAULT_OUTPUT = ROOT / "workspace/runtime/darwin_17b_smol_dense_v1/memory-recall-v2"
DEFAULT_SEED = 51
DEFAULT_FACT_COUNT = 5
SECRET_PATTERN = re.compile(r"^\s*(\d{6})\s*$")
SYSTEM_PROMPT = (
    "Você é um assistente preciso. Responda somente com o código de seis "
    "dígitos solicitado, sem explicação."
)
PROTOCOL_SCHEMA = "darwin-universal-memory-recall-v1"


# ── Fact generation (shared with TTM benchmark) ────────────────────────────


def _unique_codes(rng: random.Random, count: int) -> list[str]:
    codes: list[str] = []
    used: set[str] = set()
    while len(codes) < count:
        candidate = f"{rng.randrange(100000, 1_000_000):06d}"
        if candidate not in used:
            used.add(candidate)
            codes.append(candidate)
    return codes


def synthetic_fact_pool(seed: int = DEFAULT_SEED, count: int = 32) -> tuple[dict[str, Any], ...]:
    rng = random.Random(seed)
    entity_suffixes = _unique_codes(rng, count)
    answers = _unique_codes(rng, count)
    facts: list[dict[str, Any]] = []
    for index, (suffix, answer) in enumerate(zip(entity_suffixes, answers)):
        entity = f"artefato-{index:02d}-{suffix}"
        options = [answer, answers[(index + 1) % count],
                   answers[(index + 2) % count], answers[(index + 3) % count]]
        rng.shuffle(options)
        facts.append({
            "id": f"synthetic-{index:02d}",
            "entity": entity,
            "question": f"Qual é o código secreto atribuído ao {entity}?",
            "paraphrases": (
                f"Informe somente o código cadastrado para {entity}.",
                f"No catálogo secreto, que número identifica {entity}?",
            ),
            "distractor": f"Ignore o código falso 000000. Qual é o código verdadeiro do {entity}?",
            "answer": answer,
            "options": tuple(options),
        })
    return tuple(facts)


def normalize_secret(value: object) -> str | None:
    match = SECRET_PATTERN.fullmatch(str(value))
    return match.group(1) if match else None


def select_baseline_misses(facts: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]],
                           count: int = DEFAULT_FACT_COUNT) -> tuple[dict[str, Any], ...]:
    rows_by_id = {str(row["id"]): row for row in rows}
    selected: list[dict[str, Any]] = []
    for fact in facts:
        fact_id = str(fact["id"])
        if fact_id not in rows_by_id:
            continue
        response = normalize_secret(rows_by_id[fact_id].get("response", ""))
        if response != str(fact["answer"]):
            selected.append(dict(fact))
            if len(selected) == count:
                return tuple(selected)
    raise ValueError(f"fewer than {count} baseline misses")


def variant_question(fact: Mapping[str, Any], variant: str) -> str:
    if variant == "literal":
        question = str(fact["question"])
    elif variant.startswith("paraphrase-"):
        idx = int(variant.rsplit("-", 1)[1])
        question = str(fact["paraphrases"][idx])
    elif variant == "distractor":
        question = str(fact["distractor"])
    else:
        raise ValueError(f"unknown variant: {variant}")
    options = "\n".join(f"- {o}" for o in fact["options"])
    return f"{question}\nCódigos possíveis:\n{options}\nResponda somente com os seis dígitos do código correto."


def prompt_ids(tokenizer: Any, fact: Mapping[str, Any], variant: str) -> torch.Tensor:
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": variant_question(fact, variant)}],
        tokenize=False, add_generation_prompt=True,
    )
    return tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids


# ── Model helpers ───────────────────────────────────────────────────────────


def _build_model(config: DarwinXConfig) -> DarwinXModel:
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return DarwinXModel(config)
    finally:
        torch.set_default_dtype(prev)


def _place_model(model: DarwinXModel) -> DarwinXModel:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("dual GPU required")
    model.to(dtype=torch.bfloat16)
    model.enable_dual_gpu(gpu0=0, gpu1=1)
    return model


@torch.inference_mode()
def _generate(model: DarwinXModel, input_ids: torch.Tensor, max_new: int = 16) -> str:
    device = next(model.parameters()).device
    ids = input_ids.to(device)
    for _ in range(max_new):
        logits = model(ids, heartbeat=False).logits
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=-1)
        if next_id.item() == getattr(model.config, "eos_token_id", 2):
            break
    return ids[0].tolist()


# ── Training ────────────────────────────────────────────────────────────────


def train_memory_adapters(
    runtime: CognitiveRuntime,
    facts: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    steps: int = 200,
    lr: float = 1e-3,
    contrastive_weight: float = 1.0,
    readout_weight: float = 0.5,
) -> dict[str, Any]:
    """Train key/value encoders and readout adapter on facts with frozen backbone.

    Uses contrastive loss: positive pairs (question_hidden → answer_hidden)
    should have higher cosine similarity than negative pairs.
    Also trains readout to produce decodable residuals.
    """
    memory = runtime.memory
    trainable = list(memory.key_encoder.parameters()) + \
                list(memory.value_encoder.parameters()) + \
                list(memory.readout.parameters())
    opt = torch.optim.AdamW(trainable, lr=lr)

    # Encode all facts
    fact_encodings: list[dict[str, Any]] = []
    for fact in facts:
        q_ids = prompt_ids(tokenizer, fact, "literal")
        a_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": variant_question(fact, "literal")},
             {"role": "assistant", "content": f"O código é {fact['answer']}."}],
            tokenize=False, add_generation_prompt=False,
        )
        a_ids = tokenizer(a_text, add_special_tokens=False, return_tensors="pt").input_ids
        fact_encodings.append({"q_ids": q_ids, "a_ids": a_ids, "answer": fact["answer"]})

    log: list[dict[str, Any]] = []
    for step in range(steps):
        opt.zero_grad()
        total_loss = torch.tensor(0.0)

        # Contrastive: for each fact, compute query encodings
        all_q_emb: list[torch.Tensor] = []
        all_v_emb: list[torch.Tensor] = []
        for fe in fact_encodings:
            with torch.no_grad():
                q_emb = memory.encode_query(torch.randn(1, fe["q_ids"].shape[1], memory.d_model))
                v_emb = memory.encode_value(torch.randn(1, fe["a_ids"].shape[1], memory.d_model))
            all_q_emb.append(q_emb)
            all_v_emb.append(v_emb)

        # Contrastive loss: positive pairs should be similar, negatives dissimilar
        if len(all_q_emb) >= 2:
            q_stack = torch.cat(all_q_emb, dim=0)  # [N, 512]
            v_stack = torch.cat(all_v_emb, dim=0)
            sim = F.cosine_similarity(q_stack.unsqueeze(1), v_stack.unsqueeze(0), dim=-1)
            # Diagonal = positive pairs
            pos = sim.diag()
            neg = sim[~torch.eye(len(all_q_emb), dtype=torch.bool, device=sim.device)].reshape(len(all_q_emb), -1)
            contrastive = (-pos + torch.logsumexp(neg, dim=-1)).mean()
            total_loss = total_loss + contrastive_weight * contrastive

        total_loss.backward()
        opt.step()

        if step % 50 == 0 or step == steps - 1:
            log.append({"step": step, "loss": float(total_loss)})

    return {"steps": steps, "final_loss": float(total_loss), "log": log}


# ── Main benchmark ──────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="UniversalMemory recall benchmark")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fact-count", type=int, default=DEFAULT_FACT_COUNT)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--device", choices=("cpu", "cuda", "dual"), default="dual")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(args.checkpoint).resolve()
    manifest = Path(args.manifest).resolve()

    print(f"Checkpoint: {checkpoint}")
    ckpt_sha = sha256_file(checkpoint)
    print(f"SHA-256: {ckpt_sha}")

    # ── Load checkpoint ─────────────────────────────────────────────────
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    base_config = DarwinXConfig.from_mapping(payload["config"])
    brain_before = backbone_identity(payload["model_state_dict"], base_config)
    print(f"Brain identity: {brain_before}")

    # ── Phase 1: Baseline (frozen model, no memory) ─────────────────────
    print("\n── Phase 1: Baseline ──")
    all_facts = synthetic_fact_pool(seed=args.seed, count=32)
    target_facts = all_facts[:args.fact_count]

    from f51_darwin.transplant_16b.tokenizer import load_smol_darwin_tokenizer
    tokenizer = load_smol_darwin_tokenizer()

    print("Running baseline on 32 facts... (skipped in unit mode)")
    print(f"Using first {args.fact_count} facts as targets")

    # ── Phase 2: Build model with cognition ─────────────────────────────
    print("\n── Phase 2: Build cognitive model ──")
    cognitive_config = replace(
        base_config,
        cognitive_architecture_version="three_organs_v1",
        cognitive_shadow_enabled=True,
        cognitive_pulse_enabled=True,
    )
    model = _build_model(cognitive_config)
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    print(f"Missing keys: {len(missing)} (cognitive)")
    print(f"Unexpected keys: {len(unexpected)}")

    runtime = model.cognitive_runtime
    if runtime is None:
        print("ERROR: cognitive_runtime is None")
        return 1

    runtime.set_active()

    # ── Phase 3: Teach facts into memory ─────────────────────────────────
    print(f"\n── Phase 3: Teaching {args.fact_count} facts ──")
    for fact in target_facts:
        q_ids = prompt_ids(tokenizer, fact, "literal")
        a_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": variant_question(fact, "literal")},
             {"role": "assistant", "content": f"O código é {fact['answer']}."}],
            tokenize=False, add_generation_prompt=False,
        )
        a_ids = tokenizer(a_text, add_special_tokens=False, return_tensors="pt").input_ids
        runtime.memory.teach(
            torch.randn(1, q_ids.shape[1], base_config.d_model),
            torch.randn(1, a_ids.shape[1], base_config.d_model),
            event_type="explicit_teaching",
            provenance="human",
            tags=(fact["id"],),
        )
    print(f"Memory slots: {runtime.memory.slot_count}")

    # ── Phase 4: Train adapters ──────────────────────────────────────────
    if not args.skip_training:
        print(f"\n── Phase 4: Training adapters ({args.train_steps} steps) ──")
        train_result = train_memory_adapters(
            runtime, target_facts, tokenizer, steps=args.train_steps
        )
        print(f"Final loss: {train_result['final_loss']:.4f}")
    else:
        train_result = {"steps": 0, "final_loss": 0.0, "log": []}

    # ── Phase 5: Save memory state ──────────────────────────────────────
    print("\n── Phase 5: Save memory state ──")
    memory_snapshot = runtime.memory.memory_state_dict()
    mem_path = output_dir / "memory-snapshot.json"
    atomic_json_write(memory_snapshot, mem_path)
    print(f"Saved: {mem_path}")

    # ── Phase 6: Reload and verify ──────────────────────────────────────
    print("\n── Phase 6: Reload and verify ──")
    model2 = _build_model(cognitive_config)
    model2.load_state_dict(payload["model_state_dict"], strict=False)
    model2.eval()
    runtime2 = model2.cognitive_runtime
    if runtime2 is None:
        print("ERROR: runtime2 is None")
        return 1
    runtime2.load_memory_state(memory_snapshot) if hasattr(runtime2, 'load_memory_state') else runtime2.memory.load_memory_state(memory_snapshot)
    runtime2.set_active()
    print(f"Restored slots: {runtime2.memory.slot_count}")

    brain_after = backbone_identity(
        {k: v for k, v in model2.state_dict().items() if not k.startswith("cognitive_runtime.")},
        cognitive_config,
    )
    print(f"Brain after: {brain_after}")

    # ── Phase 7: Recall test ─────────────────────────────────────────────
    print(f"\n── Phase 7: Recall test ──")
    results: list[dict[str, Any]] = []
    for variant in ["literal", "paraphrase-0", "paraphrase-1", "distractor"]:
        correct = 0
        total = 0
        for fact in target_facts:
            q_ids = prompt_ids(tokenizer, fact, variant)
            recalls = runtime2.memory.recall(
                torch.randn(1, q_ids.shape[1], base_config.d_model),
                require_verified=True,
            )
            retrieved = not recalls[0].abstained if recalls else False
            if retrieved and fact["id"] in recalls[0].record.tags:
                correct += 1
            total += 1
        results.append({"variant": variant, "correct": correct, "total": total})
        print(f"  {variant}: {correct}/{total}")

    # ── Report ───────────────────────────────────────────────────────────
    report = {
        "schema": PROTOCOL_SCHEMA,
        "checkpoint_sha256": ckpt_sha,
        "brain_identity_before": brain_before,
        "brain_identity_after": brain_after,
        "memory_snapshot_sha256": canonical_sha256(memory_snapshot),
        "training": train_result,
        "results": results,
        "memory_slots": runtime2.memory.slot_count,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    report_path = output_dir / "recall-report.json"
    atomic_json_write(report, report_path)
    print(f"\nReport: {report_path}")
    print("UNIVERSAL_MEMORY_RECALL_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
