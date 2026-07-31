#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from f51_darwin.transplant_16b.calibration import (
    assert_explicit_seeds,
    calibrate,
)
from f51_darwin.transplant_16b.verification import evaluate_knowledge_gates


ROOT = Path(__file__).resolve().parents[1]
DONOR = (
    ROOT
    / "workspace"
    / "00_DONORS"
    / "models--HuggingFaceTB--SmolLM2-1.7B-Instruct"
    / "snapshots"
    / "31b70e2e869a7173562077fd711b654946d38674"
)
RUNTIME = ROOT / "workspace/runtime/darwin_16b_smol_transplant_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate Darwin-Smol on two GPUs and gate three seeds."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--donor", default=str(DONOR))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--seeds",
        default="17,29,43",
        help="Must remain exactly 17,29,43 for a publishable run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    seeds = assert_explicit_seeds(
        int(value) for value in args.seeds.split(",")
    )
    calibration_root = RUNTIME / "calibration"
    results = []
    for seed in seeds:
        print(f"CALIBRATION_SEED_START seed={seed} steps={args.steps}", flush=True)
        result = calibrate(
            checkpoint=args.checkpoint,
            manifest=args.manifest,
            tokenizer_root=args.tokenizer,
            donor_root=args.donor,
            seed=seed,
            steps=args.steps,
            output_root=calibration_root,
        )
        results.append(result)
        print(
            "CALIBRATION_SEED_DONE "
            f"seed={seed} loss={result.mean_train_loss:.6f} "
            f"bpb={result.metrics.bpb:.6f} "
            f"random_bpb={result.metrics.random_bpb:.6f} "
            f"kl={result.metrics.kl:.6f} "
            f"random_kl={result.metrics.random_kl:.6f} "
            f"top1={result.metrics.top1:.6f} "
            f"random_top1={result.metrics.random_top1:.6f}",
            flush=True,
        )
    gate = evaluate_knowledge_gates(result.metrics for result in results)
    best = min(
        results,
        key=lambda row: (
            row.metrics.kl / max(row.metrics.random_kl, 1e-12)
            + row.metrics.bpb / max(row.metrics.random_bpb, 1e-12)
        ),
    )
    report = {
        "schema": "darwin-smol-knowledge-gates-v1",
        "passed": gate.passed,
        "status": gate.status,
        "failures": list(gate.failures),
        "metrics": [asdict(row.metrics) for row in results],
        "runs": [asdict(row) for row in results],
        "best_seed": best.seed,
        "best_checkpoint": best.checkpoint,
        "best_manifest": best.manifest,
    }
    destination = RUNTIME / "knowledge-gates.json"
    temporary = destination.with_suffix(".json.incomplete")
    temporary.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    if gate.passed:
        print(
            f"KNOWLEDGE_GATES_PASS best_seed={best.seed} report={destination}"
        )
        return 0
    print(
        "KNOWLEDGE_GATES_FAIL status=engineering_transplant_only "
        + " failures="
        + ";".join(gate.failures)
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
