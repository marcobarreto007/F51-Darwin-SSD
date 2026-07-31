#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from f51_darwin.transplant_16b.calibration import evaluate_exact_candidate
from f51_darwin.transplant_16b.cli import DEFAULT_SOURCE_ROOT
from f51_darwin.transplant_16b.verification import evaluate_knowledge_gates


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = ROOT / "workspace/03_CHECKPOINTS_1.7B_SMOL_EXACT_V3"
RUNTIME_ROOT = ROOT / "workspace/runtime/darwin_17b_smol_exact_v3"
TOKENIZER_ROOT = ROOT / "workspace/01_TOKENIZER/smol_49152_transplant_v1"


def main() -> int:
    result = evaluate_exact_candidate(
        checkpoint=CHECKPOINT_ROOT / "organism_cycle_000.pt",
        manifest=CHECKPOINT_ROOT / "organism_cycle_000.manifest.json",
        tokenizer_root=TOKENIZER_ROOT,
        donor_root=DEFAULT_SOURCE_ROOT,
    )
    gate = evaluate_knowledge_gates(result.metrics)
    report = {
        "schema": "darwin-smol-exact-knowledge-gates-v3",
        "passed": gate.passed,
        "status": gate.status,
        "failures": list(gate.failures),
        "metrics": [asdict(row) for row in result.metrics],
        "checkpoint": str(
            CHECKPOINT_ROOT / "organism_cycle_000.pt"
        ),
        "manifest": str(
            CHECKPOINT_ROOT / "organism_cycle_000.manifest.json"
        ),
        "training_steps": 0,
        "reason": "exact function preservation made calibration unnecessary",
    }
    destination = RUNTIME_ROOT / "knowledge-gates.json"
    temporary = destination.with_suffix(".json.incomplete")
    temporary.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    if gate.passed:
        print(
            "EXACT_KNOWLEDGE_GATES_PASS "
            f"bpb={result.candidate_bpb:.6f} "
            f"kl={result.candidate_kl:.6f} "
            f"top1={result.candidate_top1:.6f}"
        )
        return 0
    print(
        "EXACT_KNOWLEDGE_GATES_FAIL "
        + " failures="
        + ";".join(gate.failures)
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
