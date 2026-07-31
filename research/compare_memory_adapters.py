#!/usr/bin/env python3
"""Head-to-head between memory adapter sets, on ground neither one trained on.

Two adapter sets exist in workspace/runtime/memory-training/, trained by
different sessions against different evaluation protocols, and their headline
numbers are not comparable:

  trained-adapters.pt     reported paraphrase 9/20 (45%) -- 10 slots, entities
                          and phrasings unseen, calibrated score+margin gate,
                          plus never-taught and off-topic queries that had to
                          abstain.
  trained-adapters-v2.pt  reported paraphrase 7/9 (78%) -- 3 slots, and the
                          three entities and four phrasings of that evaluation
                          all appear in its own training set, with the default
                          0.6 threshold and no negative control.

The second measures training-set accuracy, the first measures generalisation
under a calibrated gate. Ranking them by those numbers would pick the more
permissive trainer.

This harness fixes that: fresh entity names (prefixes and nouns absent from
both training sets), fresh phrasings (absent from both template lists), the
same store size, and the same queries for both. It separates retrieval from
gating, because those are different failures: whether the store ranked the
right slot first is a property of the encoder, whether it was then accepted is
a property of the threshold.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

from f51_darwin.cognition import CognitiveForwardMetadata
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt"
ARTIFACTS = ROOT / "workspace/runtime/memory-training"

CANDIDATES = {
    "v2-query-space": ARTIFACTS / "trained-adapters.pt",
    "v2-paraphrase": ARTIFACTS / "trained-adapters-v2.pt",
}

SYSTEM = "Voce e um assistente preciso. Responda com o codigo de seis digitos."

# Prefixes and nouns absent from both trainers' pools.
_PREFIX = ["dossie", "protocolo", "ficha", "lacre"]
_NOUN = ["andorinha", "capivara", "onca", "tatu", "quati",
         "surucucu", "irara", "jaburu", "sabia", "pacu",
         "aracari", "peixe-boi", "cutia", "maritaca", "jacutinga",
         "muriqui", "tamandua-bandeira", "gralha", "bicudo", "sagui"]

# Present in both training sets, so teaching is neutral between them.
TEACH_TEMPLATE = "Qual e o codigo atribuido a {e}?"
# Extra phrasings offered at teach time when multi-key is enabled. These are
# a property of how the fact is written, not of the test: the probe phrasings
# below stay unseen either way.
TEACH_ALT_TEMPLATES = [
    "Me diga qual codigo pertence a {e}.",
    "O {e} esta registrado sob qual codigo?",
    "Informe o codigo cadastrado para {e}.",
    "Que numero identifica {e}?",
]
# Absent from both template lists.
PROBE_TEMPLATES = [
    "Voce guardou algum codigo para {e}?",
    "Sob que numeracao {e} aparece no arquivo?",
    "Qual sequencia numerica corresponde a {e}?",
    "Localize {e} e informe o valor de seis digitos.",
]


def build_entities(count: int, offset: int = 0) -> list[str]:
    out = []
    for i in range(offset, offset + count):
        out.append(f"{_PREFIX[i % len(_PREFIX)]}-{_NOUN[i % len(_NOUN)]}")
    return out


class Harness:
    def __init__(self, adapters: Path, device: torch.device) -> None:
        payload = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
        config = DarwinXConfig.from_mapping(payload["config"])
        cog = replace(
            config,
            cognitive_architecture_version="three_organs_v1",
            cognitive_shadow_enabled=True,
            cognitive_pulse_enabled=True,
        )
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        model = DarwinXModel(cog)
        torch.set_default_dtype(prev)
        model.load_state_dict(payload["model_state_dict"], strict=False)
        model.eval()
        model.to(dtype=torch.bfloat16)
        model.enable_dual_gpu(gpu0=0, gpu1=1)

        blob = torch.load(adapters, map_location="cpu", weights_only=False)
        memory = model.cognitive_runtime.memory
        memory.key_encoder.load_state_dict(blob["key_encoder"])
        memory.value_encoder.load_state_dict(blob["value_encoder"])
        memory.readout.load_state_dict(blob["readout"])
        # Both sets run in float32 regardless of how they were stored: bf16
        # cosine carries ~8e-3 of noise, which would otherwise handicap
        # whichever set was serialised in bf16.
        memory.key_encoder.to(device=device, dtype=torch.float32)
        memory.value_encoder.to(device=device, dtype=torch.float32)
        memory.readout.to(device=device, dtype=torch.float32)

        self.own_threshold = float(blob.get("recall_threshold", 0.6))
        self.own_margin = float(blob.get("margin_threshold", 0.0))
        self.own_high = float(blob.get("high_confidence_threshold", float("inf")))
        model.cognitive_runtime.set_active()

        from transformers import AutoTokenizer
        from f51_darwin.transplant_16b.cli import DEFAULT_SOURCE_ROOT

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(DEFAULT_SOURCE_ROOT), local_files_only=True
        )
        self.model = model
        self.memory = memory
        self.device = device
        self._step = 0

    def encode(self, text: str, role: str) -> torch.Tensor:
        self._step += 1
        chat = [{"role": "system", "content": SYSTEM}, {"role": role, "content": text}]
        fmt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=(role == "user")
        )
        ids = self.tokenizer(
            fmt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.model.token_embedding.weight.device)
        with torch.inference_mode():
            out = self.model(
                ids, heartbeat=False,
                cognitive_metadata=CognitiveForwardMetadata(
                    step_id=self._step,
                    checkpoint_id="sha256:" + "a" * 64,
                    context_digest="sha256:" + "b" * 64,
                ),
            )
        if out.hidden_states is None:
            raise RuntimeError("hidden_states is None")
        return out.hidden_states

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()


def best_operating_point(
    positives: list[dict], negatives: list[dict],
) -> dict:
    """Best balanced accuracy this adapter set can reach on this data.

    Each set arrived with its own gate, calibrated (or not) against its own
    distribution, so comparing them at their shipped thresholds measures
    calibration luck rather than encoder quality. Sweeping both to their own
    optimum puts the comparison on the encoder.
    """
    grid_s = sorted({round(p["score"], 2) for p in positives + negatives})
    grid_m = [0.0] + sorted(
        {round(p["margin"], 3) for p in positives + negatives if p["margin"] > 0}
    )
    best = {"balanced_accuracy": -1.0}
    for t in grid_s:
        for m in grid_m:
            tp = sum(1 for p in positives
                     if p["correct"] and p["score"] >= t and p["margin"] >= m)
            tn = sum(1 for n in negatives
                     if not (n["score"] >= t and n["margin"] >= m))
            bal = 0.5 * (tp / max(len(positives), 1) + tn / max(len(negatives), 1))
            if bal > best["balanced_accuracy"]:
                best = {
                    "threshold": t, "margin": m, "balanced_accuracy": bal,
                    "accepted": tp, "abstained": tn,
                }
    return best


def evaluate(
    name: str, adapters: Path, n_facts: int, device: torch.device,
    multi_key: bool,
) -> dict:
    label = f"{name}{'  [multi-key]' if multi_key else ''}"
    print(f"\n{'=' * 68}\n{label}  ({adapters.name})\n{'=' * 68}")
    h = Harness(adapters, device)
    print(f"  own gate: score>={h.own_threshold:.3f} margin>={h.own_margin:.3f} "
          f"high={h.own_high:.3f}")

    taught = build_entities(n_facts)
    unseen = build_entities(n_facts, offset=n_facts)

    rids = {}
    for i, entity in enumerate(taught):
        q = h.encode(TEACH_TEMPLATE.format(e=entity), "user")
        a = h.encode(f"O codigo de {entity} e {410000 + i * 7919:06d}.", "assistant")
        alts = (
            [h.encode(t.format(e=entity), "user") for t in TEACH_ALT_TEMPLATES]
            if multi_key else []
        )
        rids[entity] = h.memory.teach(
            q, a, event_type="explicit_teaching", provenance="human",
            tags=(entity,), alt_key_hiddens=alts,
        )
    keys_per_slot = 1 + (len(TEACH_ALT_TEMPLATES) if multi_key else 0)
    print(f"  taught {h.memory.slot_count} slots x {keys_per_slot} keys")

    # Gate off: pure retrieval quality.
    h.memory.recall_threshold = -1.0
    h.memory.margin_threshold = 0.0

    top1 = 0
    probes = []
    for entity in taught:
        for template in PROBE_TEMPLATES:
            r = h.memory.recall(
                h.encode(template.format(e=entity), "user"),
                top_k=2, require_verified=True,
            )[0]
            correct = r.record.memory_id == rids[entity]
            top1 += correct
            probes.append({"correct": correct, "score": r.score, "margin": r.margin})
    total = len(probes)

    negatives = []
    for entity in unseen:
        for template in PROBE_TEMPLATES[:2]:
            r = h.memory.recall(
                h.encode(template.format(e=entity), "user"),
                top_k=2, require_verified=True,
            )[0]
            negatives.append({"score": r.score, "margin": r.margin})

    def accepts(row: dict) -> bool:
        return row["score"] >= h.own_threshold and (
            row["margin"] >= h.own_margin or row["score"] >= h.own_high
        )

    accepted = sum(1 for p in probes if p["correct"] and accepts(p))
    abstained = sum(1 for n in negatives if not accepts(n))
    best = best_operating_point(probes, negatives)

    print(f"  top-1 correct (gate off) : {top1}/{total}  ({top1/total:.1%})")
    print(f"  accepted with own gate   : {accepted}/{total}  ({accepted/total:.1%})")
    print(f"  abstained on never-taught: {abstained}/{len(negatives)}  "
          f"({abstained/len(negatives):.1%})")
    print(f"  best reachable gate      : t={best['threshold']:.2f} "
          f"m={best['margin']:.3f} -> accepted {best['accepted']}/{total}, "
          f"abstained {best['abstained']}/{len(negatives)}, "
          f"bal={best['balanced_accuracy']:.3f}")
    print(f"  score  mean pos/neg      : "
          f"{sum(p['score'] for p in probes)/total:.4f} / "
          f"{sum(n['score'] for n in negatives)/len(negatives):.4f}")

    h.close()
    return {
        "adapters": adapters.name,
        "multi_key": multi_key,
        "keys_per_slot": keys_per_slot,
        "slots": n_facts,
        "top1_correct": top1,
        "total_probes": total,
        "top1_rate": top1 / total,
        "accepted": accepted,
        "accepted_rate": accepted / total,
        "abstained": abstained,
        "negatives": len(negatives),
        "abstain_rate": abstained / len(negatives),
        "best_operating_point": best,
        "gate": {
            "threshold": h.own_threshold,
            "margin": h.own_margin,
            "high_confidence": h.own_high,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", type=int, default=10)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Fresh entities and phrasings, unseen by BOTH trainers.")
    print(f"store={args.facts} slots  probes={len(PROBE_TEMPLATES)} phrasings/fact")

    results = {}
    for name, path in CANDIDATES.items():
        if not path.exists():
            print(f"\n[skip] {name}: {path} not found")
            continue
        for multi_key in (False, True):
            key = f"{name}{'+multikey' if multi_key else ''}"
            results[key] = evaluate(name, path, args.facts, device, multi_key)

    print(f"\n{'=' * 68}\nSUMMARY (same ground for every row)\n{'=' * 68}")
    print(f"{'adapter set':<26} {'top-1':>8} {'own gate':>9} {'abstain':>8} "
          f"{'best bal':>9}")
    for name, r in results.items():
        print(f"{name:<26} {r['top1_rate']:>7.1%} {r['accepted_rate']:>9.1%} "
              f"{r['abstain_rate']:>8.1%} "
              f"{r['best_operating_point']['balanced_accuracy']:>9.3f}")

    out = ARTIFACTS / "adapter-comparison.json"
    out.write_text(json.dumps(
        {"schema": "memory-adapter-comparison-v1", "results": results}, indent=2
    ))
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
