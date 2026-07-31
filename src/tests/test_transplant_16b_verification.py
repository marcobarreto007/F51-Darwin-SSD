from __future__ import annotations

import json

import pytest

from f51_darwin.transplant_16b.calibration import (
    CalibrationManifest,
    assert_explicit_seeds,
    write_calibration_manifest,
)
from f51_darwin.transplant_16b.verification import (
    KnowledgeMetrics,
    evaluate_knowledge_gates,
)


def _row(seed: int, **overrides: object) -> KnowledgeMetrics:
    values = {
        "seed": seed,
        "bpb": 3.15,
        "random_bpb": 3.5,
        "kl": 1.125,
        "random_kl": 1.5,
        "top1": 0.16,
        "random_top1": 0.08,
        "generation_valid": True,
    }
    values.update(overrides)
    return KnowledgeMetrics(**values)


def test_all_three_paired_seeds_must_pass() -> None:
    rows = [_row(17), _row(29, top1=0.14), _row(43)]
    report = evaluate_knowledge_gates(rows)
    assert report.passed is False
    assert report.status == "engineering_transplant_only"
    assert "seed=29" in "\n".join(report.failures)


def test_exact_thresholds_are_accepted() -> None:
    assert evaluate_knowledge_gates(
        [_row(seed) for seed in (17, 29, 43)]
    ).passed


def test_missing_seed_fails_closed() -> None:
    report = evaluate_knowledge_gates([_row(17), _row(43)])
    assert not report.passed
    assert "seeds must be exactly" in report.failures[0]


def test_calibration_lineages_are_immutable(tmp_path) -> None:
    manifest = CalibrationManifest(
        seed=17,
        plan_id="a" * 64,
        base_checkpoint_id="b" * 64,
        holdout_sha256="c" * 64,
        tokenizer_id="smol:" + "d" * 64,
        trained_steps=8,
    )
    path = write_calibration_manifest(manifest, tmp_path)
    assert json.loads(path.read_text("utf-8"))["seed"] == 17
    assert write_calibration_manifest(manifest, tmp_path) == path
    changed = CalibrationManifest(
        **{**manifest.__dict__, "trained_steps": 9}
    )
    with pytest.raises(FileExistsError):
        write_calibration_manifest(changed, tmp_path)


def test_calibration_requires_three_exact_seeds() -> None:
    assert assert_explicit_seeds((17, 29, 43)) == (17, 29, 43)
    with pytest.raises(ValueError):
        assert_explicit_seeds((17,))
