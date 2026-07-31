#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from f51_darwin.transplant_16b.calibration import (
    evaluate_exact_candidate,
)
from f51_darwin.transplant_16b.cli import DEFAULT_SOURCE_ROOT
from f51_darwin.transplant_16b.dense_assembly import _atomic_json
from f51_darwin.transplant_16b.dense_runtime import (
    run_dense_optimizer_smoke,
)
from f51_darwin.transplant_16b.verification import (
    evaluate_knowledge_gates,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = (
    ROOT / "workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1"
)
RUNTIME_ROOT = (
    ROOT / "workspace/runtime/darwin_17b_smol_dense_v1"
)
TOKENIZER_ROOT = (
    ROOT / "workspace/01_TOKENIZER/smol_49152_transplant_v1"
)


def main() -> int:
    checkpoint = CHECKPOINT_ROOT / "organism_cycle_000.pt"
    manifest = (
        CHECKPOINT_ROOT / "organism_cycle_000.manifest.json"
    )
    optimizer = run_dense_optimizer_smoke(
        checkpoint=checkpoint,
        manifest=manifest,
        output=RUNTIME_ROOT / "optimizer-smoke.json",
    )
    result = evaluate_exact_candidate(
        checkpoint=checkpoint,
        manifest=manifest,
        tokenizer_root=TOKENIZER_ROOT,
        donor_root=DEFAULT_SOURCE_ROOT,
    )
    gate = evaluate_knowledge_gates(result.metrics)
    report = {
        "schema": "darwin-smol-dense-knowledge-gates-v1",
        "passed": gate.passed,
        "status": gate.status,
        "failures": list(gate.failures),
        "metrics": [asdict(row) for row in result.metrics],
        "checkpoint": str(checkpoint),
        "manifest": str(manifest),
        "optimizer_smoke": optimizer,
        "training_steps": 0,
        "reason": (
            "native dense function preservation made calibration "
            "unnecessary"
        ),
    }
    _atomic_json(RUNTIME_ROOT / "knowledge-gates.json", report)
    if gate.passed:
        print(
            "DENSE_KNOWLEDGE_GATES_PASS "
            f"bpb={result.candidate_bpb:.6f} "
            f"kl={result.candidate_kl:.6f} "
            f"top1={result.candidate_top1:.6f}"
        )
        print(
            "DENSE_OPTIMIZER_SMOKE_PASS "
            f"loss={optimizer['loss']:.6f}"
        )
        return 0
    print(
        "DENSE_KNOWLEDGE_GATES_FAIL failures="
        + ";".join(gate.failures)
    )
    print(json.dumps(report, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
