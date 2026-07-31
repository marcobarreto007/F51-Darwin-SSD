#!/usr/bin/env python3
"""Full-cycle validation of UniversalMemory (v2).

The v1 version of this script proved less than it claimed:

  * both recall phases queried with `f["q_hidden"]` -- the very tensor whose
    projection had been stored as the key -- so `cosine(k, k) == 1.0` was an
    identity, not evidence, and would have passed with untrained adapters;
  * the "fresh session" ran in the same Python process, holding the original
    tensors in RAM, so nothing crossed a session boundary;
  * the only negative control was `torch.randn`, which is far out of
    distribution for real hidden states (norm ~200 vs ~315) and is rejected
    trivially, while the control that matters -- a question about an entity
    that was never taught -- was never run, and fails;
  * every metric was slot-id retrieval, yet the result was reported as the
    model "answering correctly".

This version re-encodes every query through a forward pass, runs the recall
phase in a separate process, tests held-out phrasings, and treats
never-taught entities as the primary negative control.  It measures
retrieval only, and says so.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch

from f51_darwin.cognition import CognitiveForwardMetadata
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt"
ARTIFACTS = ROOT / "workspace/runtime/memory-training"
ADAPTERS = ARTIFACTS / "trained-adapters.pt"
SNAPSHOT = ARTIFACTS / "memory-snapshot.json"
FACTS_FILE = ARTIFACTS / "validation-facts.json"
REPORT = ARTIFACTS / "full-cycle-report.json"

SYSTEM = "Voce e um assistente preciso. Responda com o codigo de seis digitos."

# Entity names use prefixes absent from the trainer's pool, so every validation
# entity is one the adapters have never seen.
VAL_ENTITIES = [
    "arquivo-baleia", "consorcio-tapir", "arquivo-jacaranda", "consorcio-peroba",
    "arquivo-mutum", "consorcio-arara", "arquivo-sucuri", "consorcio-jaguar",
    "arquivo-tucano", "consorcio-cambui",
]
UNSEEN_ENTITIES = [
    "arquivo-quiriri", "consorcio-tamandua", "arquivo-pirarucu",
    "consorcio-bugio", "arquivo-caravela", "consorcio-mandacaru",
]

TEACH_TEMPLATE = "Qual e o codigo atribuido a {e}?"
# Phrasings the adapters were never trained on.
PARAPHRASE_TEMPLATES = [
    "Preciso do codigo da {e}.",
    "Nao consigo lembrar o codigo de {e}, voce sabe?",
]


def build_model() -> tuple[DarwinXModel, DarwinXConfig, Any, dict[str, Any]]:
    payload = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    config = DarwinXConfig.from_mapping(payload["config"])
    cog_config = replace(
        config,
        cognitive_architecture_version="three_organs_v1",
        cognitive_shadow_enabled=True,
        cognitive_pulse_enabled=True,
    )
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = DarwinXModel(cog_config)
    torch.set_default_dtype(prev)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    model.to(dtype=torch.bfloat16)
    model.enable_dual_gpu(gpu0=0, gpu1=1)

    trained = torch.load(ADAPTERS, map_location="cpu", weights_only=False)
    memory = model.cognitive_runtime.memory
    memory.key_encoder.load_state_dict(trained["key_encoder"])
    memory.value_encoder.load_state_dict(trained["value_encoder"])
    memory.readout.load_state_dict(trained["readout"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    memory.key_encoder.to(device=device, dtype=torch.float32)
    memory.value_encoder.to(device=device, dtype=torch.float32)
    memory.readout.to(device=device, dtype=torch.float32)
    memory.recall_threshold = float(trained.get("recall_threshold", 0.6))
    memory.margin_threshold = float(trained.get("margin_threshold", 0.0))
    memory.high_confidence_threshold = float(
        trained.get("high_confidence_threshold", float("inf"))
    )
    model.cognitive_runtime.set_active()

    from transformers import AutoTokenizer
    from f51_darwin.transplant_16b.cli import DEFAULT_SOURCE_ROOT
    tokenizer = AutoTokenizer.from_pretrained(str(DEFAULT_SOURCE_ROOT), local_files_only=True)
    return model, config, tokenizer, trained


class Encoder:
    def __init__(self, model: DarwinXModel, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._step = 0

    def __call__(self, text: str, role: str) -> torch.Tensor:
        self._step += 1
        chat = [
            {"role": "system", "content": SYSTEM},
            {"role": role, "content": text},
        ]
        formatted = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=(role == "user"),
        )
        ids = self.tokenizer(
            formatted, add_special_tokens=False, return_tensors="pt",
        ).input_ids.to(self.model.token_embedding.weight.device)
        with torch.inference_mode():
            out = self.model(ids, heartbeat=False, cognitive_metadata=CognitiveForwardMetadata(
                step_id=self._step,
                checkpoint_id="sha256:" + "a" * 64,
                context_digest="sha256:" + "b" * 64,
            ))
        if out.hidden_states is None:
            raise RuntimeError("hidden_states is None")
        return out.hidden_states


def phase_teach() -> int:
    print("=" * 66)
    print("PHASE 1/2  teach  (process A)")
    print("=" * 66)
    model, _, tokenizer, _ = build_model()
    encode = Encoder(model, tokenizer)
    memory = model.cognitive_runtime.memory

    facts = []
    for i, entity in enumerate(VAL_ENTITIES):
        code = f"{410000 + i * 7919:06d}"
        q = encode(TEACH_TEMPLATE.format(e=entity), "user")
        a = encode(f"O codigo de {entity} e {code}.", "assistant")
        rid = memory.teach(
            q, a, event_type="explicit_teaching",
            provenance="human", tags=(entity,),
        )
        facts.append({"entity": entity, "code": code, "memory_id": rid})

    snapshot = memory.memory_state_dict()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(json.dumps(snapshot, indent=2))
    FACTS_FILE.write_text(json.dumps(facts, indent=2))
    print(f"  taught {memory.slot_count} facts")
    print(f"  snapshot sha256={snapshot['snapshot_sha256'][:16]}...")
    print(f"  wrote {SNAPSHOT.name}, {FACTS_FILE.name}")
    print("  process A exiting; no tensor survives to phase 2\n")
    return 0


def phase_recall() -> int:
    print("=" * 66)
    print("PHASE 2/2  recall  (process B -- fresh interpreter)")
    print("=" * 66)
    model, config, tokenizer, trained = build_model()
    encode = Encoder(model, tokenizer)
    memory = model.cognitive_runtime.memory

    facts = json.loads(FACTS_FILE.read_text())
    snapshot = json.loads(SNAPSHOT.read_text())
    memory.load_memory_state(snapshot)
    print(f"  restored {memory.slot_count} slots "
          f"({memory.verified_count} verified) from disk")
    print(f"  recall_threshold={memory.recall_threshold:.4f}  "
          f"margin_threshold={memory.margin_threshold:.4f}  "
          f"high_confidence={memory.high_confidence_threshold:.4f}\n")

    results: dict[str, Any] = {}

    def run_block(name: str, queries: list[tuple[str, str | None]]) -> dict[str, Any]:
        """queries: (text, expected_memory_id or None if it must abstain)."""
        ok = 0
        argmax_ok = 0
        rows = []
        for text, expected in queries:
            r = memory.recall(encode(text, "user"), top_k=2, require_verified=True)[0]
            # Whether the store ranked the right slot first, independent of
            # whether the abstention gate then let it through. Retrieval
            # quality and gate calibration are separate failures with separate
            # fixes, and collapsing them hides which one is actually broken.
            retrieved = expected is not None and r.record.memory_id == expected
            argmax_ok += retrieved
            if expected is None:
                success = r.abstained
                verdict = "ABSTAIN" if r.abstained else "FALSE-ACCEPT"
            else:
                success = (not r.abstained) and retrieved
                verdict = "HIT" if success else (
                    "GATED" if r.abstained and retrieved
                    else "ABSTAIN" if r.abstained else "WRONG"
                )
            ok += success
            rows.append({
                "query": text, "verdict": verdict,
                "score": round(r.score, 4), "margin": round(r.margin, 4),
                "retrieved_correct": bool(retrieved),
                "reason": r.abstention_reason,
            })
        n = max(len(queries), 1)
        rate = ok / n
        argmax_rate = argmax_ok / n
        head = f"--- {name}: {ok}/{len(queries)} accepted ({rate:.1%})"
        if queries and queries[0][1] is not None:
            head += f"  |  top-1 correct {argmax_ok}/{len(queries)} ({argmax_rate:.1%})"
        print(head)
        for row in rows:
            print(f"    {row['verdict']:12s} score={row['score']:+.4f} "
                  f"margin={row['margin']:+.4f}  {row['query'][:46]}")
        print()
        return {
            "passed": ok, "total": len(queries), "rate": rate,
            "argmax_correct": argmax_ok, "argmax_rate": argmax_rate,
            "rows": rows,
        }

    results["literal"] = run_block(
        "A. literal phrasing (as taught, re-encoded)",
        [(TEACH_TEMPLATE.format(e=f["entity"]), f["memory_id"]) for f in facts],
    )
    results["paraphrase"] = run_block(
        "B. held-out paraphrase (wording never trained)",
        [(t.format(e=f["entity"]), f["memory_id"])
         for f in facts for t in PARAPHRASE_TEMPLATES],
    )
    results["unseen_entity"] = run_block(
        "C. never-taught entity (must abstain)",
        [(t.format(e=e), None)
         for e in UNSEEN_ENTITIES
         for t in [TEACH_TEMPLATE] + PARAPHRASE_TEMPLATES],
    )
    results["off_topic"] = run_block(
        "D. off-topic query (must abstain)",
        [(q, None) for q in [
            "Qual e a capital da Franca?",
            "Quanto e 17 vezes 3?",
            "Me explique o que e entropia.",
        ]],
    )

    print("--- E. empty store (must abstain)")
    empty = type(memory)(config.d_model)
    r_empty = empty.recall(encode("Qual e o codigo atribuido a X?", "user"), top_k=1)[0]
    print(f"    abstained={r_empty.abstained}  reason={r_empty.abstention_reason}\n")
    results["empty_store"] = {
        "abstained": bool(r_empty.abstained),
        "reason": r_empty.abstention_reason,
    }

    gates = {
        "literal >= 0.90": results["literal"]["rate"] >= 0.90,
        "paraphrase >= 0.70": results["paraphrase"]["rate"] >= 0.70,
        "unseen_entity_abstain >= 0.70": results["unseen_entity"]["rate"] >= 0.70,
        "off_topic_abstain == 1.00": results["off_topic"]["rate"] >= 1.0,
        "empty_store_abstains": results["empty_store"]["abstained"],
    }
    print("=" * 66)
    for name, passed in gates.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    all_ok = all(gates.values())
    print("=" * 66)
    print(f"FULL_CYCLE_{'OK' if all_ok else 'PARTIAL'}  "
          f"literal={results['literal']['passed']}/{results['literal']['total']} "
          f"paraphrase={results['paraphrase']['passed']}/{results['paraphrase']['total']} "
          f"abstain={results['unseen_entity']['passed']}/{results['unseen_entity']['total']}")
    print("NOTE: every metric above is slot retrieval. Whether the backbone can")
    print("      decode a recalled value into text is NOT tested here.")
    print("=" * 66)

    REPORT.write_text(json.dumps({
        "schema": "memory-full-cycle-report-v2",
        "metric": "slot_retrieval_only",
        "recall_threshold": memory.recall_threshold,
        "margin_threshold": memory.margin_threshold,
        "high_confidence_threshold": memory.high_confidence_threshold,
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "gates": gates,
        "results": results,
    }, indent=2))
    print(f"report: {REPORT}")
    return 0 if all_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["teach", "recall"], default=None)
    args = parser.parse_args()

    if args.phase == "teach":
        return phase_teach()
    if args.phase == "recall":
        return phase_recall()

    # Driver: run each phase in its own interpreter so the recall phase cannot
    # inherit a single tensor, module or cached value from the teach phase.
    for phase in ("teach", "recall"):
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--phase", phase],
            cwd=str(ROOT),
        )
        if proc.returncode != 0 and phase == "teach":
            return proc.returncode
        rc = proc.returncode
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
