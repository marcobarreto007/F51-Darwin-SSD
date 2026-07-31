#!/usr/bin/env python3
"""Verify local corpus/checkpoint lineage and publish cycle 071 only on full success."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]

from f51_darwin.artifact_manifest import (
    publish_pointer_atomic,
    select_canonical_checkpoint,
    sha256_file,
    validate_checkpoint_report,
    verify_corpus_manifest,
    write_json_atomic,
)
from f51_darwin.dataset_layout import WorkspacePaths
from f51_darwin.state_identity import tokenizer_identity
from f51_darwin.tokenizer import F51BPETokenizer


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def source_commit(project: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(project), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def require_clean_tracked_worktree(project: Path) -> None:
    result = subprocess.run(
        ["git", "-C", str(project), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    )
    if result.stdout.strip():
        raise RuntimeError(f"tracked_worktree_dirty: {result.stdout.strip()}")


def inspect_checkpoint(project: Path, checkpoint: Path, config: Path) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    result = subprocess.run(
        [
            sys.executable,
            str(project / "src/scripts/inspect_organism_checkpoint.py"),
            str(checkpoint),
            "--config",
            str(config),
            "--verify-identity",
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"checkpoint_inspector_invalid_json: {checkpoint}: {result.stderr.strip()}"
        ) from exc
    report["inspector_exit_code"] = result.returncode
    report["file_sha256"] = sha256_file(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    report["tokenizer_id"] = payload.get("tokenizer_id")
    del payload
    return report


def identity_contract_issue(report: dict[str, Any]) -> str | None:
    """Explain the current v7 save/resume identity contract without relabeling it.

    `darwin_organism.py` computes `base_checkpoint_id` from the model state saved
    in the same payload and resume rejects a mismatch. It is therefore the
    declared integral identity of that checkpoint under the current v7 contract,
    not a parent-checkpoint pointer.
    """

    identity = report.get("identity") or {}
    declared = identity.get("declared")
    recomputed = identity.get("with_raw_embedded_config")
    if declared == recomputed and report.get("identity_verified") is True:
        return None
    return f"declared_integral_identity_mismatch: declared={declared}, recomputed={recomputed}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--publish-pointer", action="store_true")
    args = parser.parse_args()

    project = args.project_root.resolve()
    require_clean_tracked_worktree(project)
    paths = WorkspacePaths.from_project(project, require=True)
    config = project / "src/configs/darwin_x_100m.yaml"  # 100M FULL V1 active; 1.6B-Nitro gold not locally accessible
    if not config.is_file():
        raise FileNotFoundError(f"config_missing: {config}")
    commit = source_commit(project)
    blockers: list[str] = []

    try:
        corpus = verify_corpus_manifest(paths.feast_token_bin, paths.feast_manifest)
    except (FileNotFoundError, ValueError) as exc:
        corpus = {"status": "blocked", "error": str(exc)}
        blockers.append(f"corpus:{exc}")

    tokenizer_path = paths.tokenizer / "f51_bpe_80k"
    try:
        tokenizer = F51BPETokenizer.load(tokenizer_path)
        tokenizer_id = tokenizer_identity(tokenizer)
        tokenizer_report: dict[str, Any] = {
            "status": "ok",
            "path": str(tokenizer_path),
            "identity": tokenizer_id,
            "files": {
                path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in sorted(tokenizer_path.iterdir())
                if path.is_file()
            },
        }
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        tokenizer_id = ""
        tokenizer_report = {"status": "blocked", "error": str(exc)}
        blockers.append(f"tokenizer:{exc}")

    reports: dict[str, dict[str, Any]] = {}
    for cycle in (70, 71, 77):
        checkpoint = paths.checkpoints / f"organism_cycle_{cycle:03d}.pt"
        try:
            reports[str(cycle)] = inspect_checkpoint(project, checkpoint, config)
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            reports[str(cycle)] = {
                "checkpoint": str(checkpoint),
                "gold_valid": False,
                "error": str(exc),
            }
            blockers.append(f"checkpoint_{cycle}:{exc}")

    expected = {70: (70, 38751), 71: (71, 40751)}
    for cycle, (expected_cycle, expected_step) in expected.items():
        report = reports[str(cycle)]
        issues = validate_checkpoint_report(
            report,
            expected_cycle=expected_cycle,
            expected_step=expected_step,
        )
        contract_issue = identity_contract_issue(report) if "error" not in report else None
        if contract_issue:
            issues.append("integral_checkpoint_identity")
            report["identity_contract_error"] = contract_issue
        if tokenizer_id and report.get("tokenizer_id") != tokenizer_id:
            issues.append("tokenizer_identity")
        report["gold_issues"] = sorted(set(issues))
        report["gold_valid"] = not issues
        blockers.extend(f"checkpoint_{cycle}:{issue}" for issue in sorted(set(issues)))

    report_077 = reports["77"]
    report_077["gold_valid"] = False
    report_077["canonical_eligible"] = False
    report_077["selection_reason"] = "explicit_lineage_comparison_required"
    if identity_contract_issue(report_077):
        report_077["identity_contract_error"] = identity_contract_issue(report_077)

    # Schema v7 stores the same-checkpoint integral identity. It validates 070
    # and 071 independently but cannot prove a parent-child edge between them.
    lineage_link = {
        "checkpoint_070_independently_valid": reports["70"].get("gold_valid") is True,
        "checkpoint_071_independently_valid": reports["71"].get("gold_valid") is True,
        "parent_id_field_present": False,
        "corroborated": False,
        "note": (
            "base_checkpoint_id is the same-checkpoint integral identity under the current "
            "save/resume contract; no separate parent id is persisted"
        ),
    }

    pointer_report: dict[str, Any]
    if blockers:
        pointer_report = {
            "status": "not_published_fail_closed",
            "path": str(paths.latest_pointer),
            "preexisting": paths.latest_pointer.exists(),
        }
    else:
        selected = select_canonical_checkpoint(reports.values())
        pointer_report = {
            "status": "verified_not_requested",
            "path": str(paths.latest_pointer),
        }
        if args.publish_pointer:
            checkpoint = Path(selected["checkpoint"])
            pointer_report = publish_pointer_atomic(
                paths.latest_pointer,
                checkpoint=checkpoint,
                report=selected,
                config=config,
                source_commit=commit,
                base_checkpoint_id=str(selected["base_checkpoint_id"]),
                checkpoint_sha256=str(selected["file_sha256"]),
            )
            pointer_report["status"] = "ok"

    full_report = {
        "schema_version": 1,
        "status": "blocked" if blockers else "ok",
        "generated_at_utc": utc_now(),
        "source_commit": commit,
        "tracked_worktree_clean": True,
        "corpus": corpus,
        "tokenizer": tokenizer_report,
        "config": {
            "path": str(config),
            "sha256": sha256_file(config),
        },
        "checkpoints": reports,
        "lineage_070_to_071": lineage_link,
        "pointer": pointer_report,
        "blockers": sorted(set(blockers)),
    }
    full_path = paths.manifests / "gold_local_lineage.json"
    write_json_atomic(full_path, full_report)

    print(json.dumps(full_report, indent=2, sort_keys=True))
    return 2 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
