#!/usr/bin/env python3
"""Fail-closed runtime gate for an isolated DAE cloud canary.

This gate authorizes continuation only.  It never promotes a checkpoint and it
does not claim model quality beyond the fixed heldout evidence in the ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from f51_darwin.training_observability import recover_jsonl


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_canary(
    *,
    metrics_path: Path,
    pointer_path: Path,
    run_id: str,
    expected_microbatches: int,
    accum_steps: int,
    expected_git_commit: str,
    expected_corpus_sha256: str,
    max_heldout_regression: float,
) -> dict[str, Any]:
    failures: list[str] = []
    records, truncated = recover_jsonl(metrics_path)
    if truncated:
        failures.append("metrics_truncated_tail")
    baselines = [record for record in records if record.get("phase") == "baseline"]
    candidates = [record for record in records if record.get("phase") == "candidate"]
    if len(baselines) != 1:
        failures.append("baseline_record_count")
    if len(candidates) != 1:
        failures.append("candidate_record_count")

    baseline = baselines[0] if len(baselines) == 1 else {}
    candidate = candidates[0] if len(candidates) == 1 else {}
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    heldout_before = (baseline.get("heldout") or {}).get("lm_loss")
    heldout_after = (candidate.get("heldout") or {}).get("lm_loss")
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value))
               for value in (heldout_before, heldout_after)):
        failures.append("heldout_missing_or_nonfinite")
    elif float(heldout_after) > float(heldout_before) + max_heldout_regression:
        failures.append("heldout_regression")

    successful = int(metrics.get("successful_updates", -1))
    skipped = int(metrics.get("skipped_updates", -1))
    optimizer_steps = int(metrics.get("optimizer_steps", -1))
    expected_optimizer_steps = math.ceil(expected_microbatches / accum_steps)
    if successful + skipped != expected_microbatches:
        failures.append("microbatch_accounting")
    if skipped != 0:
        failures.append("skipped_microbatches")
    if optimizer_steps != expected_optimizer_steps:
        failures.append("optimizer_step_accounting")

    optimizer = candidate.get("optimizer_identity") or {}
    if optimizer.get("name") != "dae_hybrid":
        failures.append("optimizer_identity")
    dae = candidate.get("dae") or {}
    reports = dae.get("last_report") if isinstance(dae, dict) else None
    if not dae.get("enabled") or dae.get("shadow_mode") is not False:
        failures.append("dae_not_active")
    if not isinstance(reports, list) or not reports:
        failures.append("dae_report_missing")
        reports = []
    if sum(int(report.get("nonfinite_rejected", 0)) for report in reports):
        failures.append("dae_nonfinite_rejected")

    if candidate.get("git_commit") != expected_git_commit:
        failures.append("git_commit_identity")
    token_source = candidate.get("token_source") or {}
    if token_source.get("manifest_sha256") != expected_corpus_sha256:
        failures.append("corpus_identity")

    checkpoint_path: Path | None = None
    checkpoint_sha256: str | None = None
    pointer: dict[str, Any] = {}
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        name = pointer["path"]
        if Path(name).name != name:
            raise ValueError("checkpoint pointer must contain a basename")
        checkpoint_path = pointer_path.parent / name
        size = checkpoint_path.stat().st_size
        if size != int(pointer.get("size_bytes", -1)):
            failures.append("checkpoint_size_identity")
        checkpoint_sha256 = _sha256(checkpoint_path)
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        failures.append("checkpoint_pointer_invalid")

    passed = not failures
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": "runtime_canary_passed" if passed else "quality_gate_failed",
        "passed": passed,
        "failures": failures,
        "git_commit": expected_git_commit,
        "corpus_sha256": expected_corpus_sha256,
        "expected_microbatches": expected_microbatches,
        "successful_microbatches": successful,
        "skipped_microbatches": skipped,
        "expected_optimizer_steps": expected_optimizer_steps,
        "optimizer_steps": optimizer_steps,
        "baseline_heldout_lm_loss": heldout_before,
        "candidate_heldout_lm_loss": heldout_after,
        "max_heldout_regression": max_heldout_regression,
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_pointer": pointer,
        "automatic_promotion": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--pointer", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-microbatches", type=int, required=True)
    parser.add_argument("--accum-steps", type=int, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--expected-corpus-sha256", required=True)
    parser.add_argument("--max-heldout-regression", type=float, default=0.02)
    args = parser.parse_args()
    decision = evaluate_canary(
        metrics_path=args.metrics,
        pointer_path=args.pointer,
        run_id=args.run_id,
        expected_microbatches=args.expected_microbatches,
        accum_steps=args.accum_steps,
        expected_git_commit=args.expected_git_commit,
        expected_corpus_sha256=args.expected_corpus_sha256,
        max_heldout_regression=args.max_heldout_regression,
    )
    _atomic_json(args.decision, decision)
    print(json.dumps(decision, sort_keys=True))
    return 0 if decision["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
