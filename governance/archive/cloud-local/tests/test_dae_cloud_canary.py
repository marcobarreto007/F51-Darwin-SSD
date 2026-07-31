import hashlib
import json
from pathlib import Path

from scripts.evaluate_dae_cloud_canary import evaluate_canary


def _write_case(tmp_path: Path, *, candidate_loss: float = 1.01) -> tuple[Path, Path]:
    checkpoint = tmp_path / "organism_cycle_001.pt"
    checkpoint.write_bytes(b"checkpoint")
    pointer = tmp_path / "organism_latest.json"
    pointer.write_text(json.dumps({
        "path": checkpoint.name,
        "size_bytes": checkpoint.stat().st_size,
        "cycle": 1,
        "step": 250,
    }), encoding="utf-8")
    common = {
        "git_commit": "a" * 40,
        "optimizer_identity": {"name": "dae_hybrid"},
        "token_source": {"manifest_sha256": "b" * 64},
    }
    metrics = tmp_path / "metrics.jsonl"
    records = [
        {**common, "phase": "baseline", "heldout": {"lm_loss": 1.0}},
        {**common, "phase": "candidate", "heldout": {"lm_loss": candidate_loss},
         "metrics": {"successful_updates": 250, "skipped_updates": 0,
                     "optimizer_steps": 16},
         "dae": {"enabled": True, "shadow_mode": False,
                 "last_report": [{"nonfinite_rejected": 0}]}}
    ]
    metrics.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return metrics, pointer


def _evaluate(tmp_path: Path, **kwargs):
    metrics, pointer = _write_case(tmp_path, **kwargs)
    return evaluate_canary(
        metrics_path=metrics,
        pointer_path=pointer,
        run_id="test-run",
        expected_microbatches=250,
        accum_steps=16,
        expected_git_commit="a" * 40,
        expected_corpus_sha256="b" * 64,
        max_heldout_regression=0.02,
    )


def test_runtime_gate_passes_complete_stable_canary(tmp_path: Path) -> None:
    decision = _evaluate(tmp_path)
    assert decision["status"] == "runtime_canary_passed"
    assert decision["passed"] is True
    assert decision["optimizer_steps"] == 16
    assert decision["checkpoint_sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
    assert decision["automatic_promotion"] is False


def test_runtime_gate_blocks_heldout_regression(tmp_path: Path) -> None:
    decision = _evaluate(tmp_path, candidate_loss=1.021)
    assert decision["status"] == "quality_gate_failed"
    assert decision["passed"] is False
    assert "heldout_regression" in decision["failures"]


def test_runtime_gate_blocks_missing_optimizer_steps(tmp_path: Path) -> None:
    metrics, pointer = _write_case(tmp_path)
    records = [json.loads(line) for line in metrics.read_text().splitlines()]
    del records[-1]["metrics"]["optimizer_steps"]
    metrics.write_text("".join(json.dumps(record) + "\n" for record in records))
    decision = evaluate_canary(
        metrics_path=metrics,
        pointer_path=pointer,
        run_id="test-run",
        expected_microbatches=250,
        accum_steps=16,
        expected_git_commit="a" * 40,
        expected_corpus_sha256="b" * 64,
        max_heldout_regression=0.02,
    )
    assert "optimizer_step_accounting" in decision["failures"]
