from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import run_causal_ablation as ablation_module
from tools.run_causal_ablation import (
    REGISTERED_ARMS,
    AblationContractError,
    run_causal_ablation,
    verify_artifacts,
)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_checkpoint_id", "tiny:other"),
        ("data_cursor", "tiny-batch-1"),
        ("seed", 52),
        ("training_contract_id", "tiny-contract-other"),
    ],
)
def test_rejects_any_cross_arm_anchor_mismatch(
    tmp_path: Path, field: str, value: str | int
) -> None:
    with pytest.raises(AblationContractError, match=field):
        run_causal_ablation(
            tmp_path / field,
            registration_overrides={"APPLY": {field: value}},
        )

    assert not (tmp_path / field).exists()


def test_paired_run_restores_every_anchor_and_proves_known_effect(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "ablation"

    result = run_causal_ablation(output_dir, seed=51)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    verdict = json.loads((output_dir / "verdict.json").read_text(encoding="utf-8"))
    evidence = {
        arm: json.loads(
            (output_dir / "arms" / f"{arm.lower()}.json").read_text(
                encoding="utf-8"
            )
        )
        for arm in ("DISABLED", *REGISTERED_ARMS)
    }

    assert result["valid"] is True
    assert verdict["valid"] is True
    assert tuple(manifest["registered_arms"]) == REGISTERED_ARMS
    assert manifest["reference_arm"] == "DISABLED"
    assert manifest["manifest_id"].startswith("causal-ablation-v1:")
    assert set(manifest["artifacts"]) == {
        "arms/disabled.json",
        "arms/control.json",
        "arms/shadow.json",
        "arms/apply.json",
        "verdict.json",
    }
    assert all(
        len(entry["sha256"]) == 64
        for entry in manifest["artifacts"].values()
    )
    assert verify_artifacts(output_dir)["valid"] is True

    expected = manifest["anchors"]
    for arm in evidence.values():
        assert arm["pre_anchors"] == expected
        assert arm["batch_digest"] == expected["batch_digest"]

    reference = evidence["DISABLED"]
    for arm in ("CONTROL", "SHADOW"):
        assert evidence[arm]["post_model_digest"] == reference["post_model_digest"]
        assert (
            evidence[arm]["post_optimizer_digest"]
            == reference["post_optimizer_digest"]
        )
        assert evidence[arm]["post_rng_digest"] == reference["post_rng_digest"]
        assert evidence[arm]["parameter_digests"] == reference["parameter_digests"]

    apply = evidence["APPLY"]
    assert apply["post_model_digest"] != reference["post_model_digest"]
    assert apply["effective_intervention_ids"] == ["zero-head-gradient"]
    assert apply["parameter_digests"]["head.weight"] == apply[
        "pre_parameter_digests"
    ]["head.weight"]
    assert reference["parameter_digests"]["head.weight"] != reference[
        "pre_parameter_digests"
    ]["head.weight"]

    assert verdict["checks"] == {
        "anchors_restored": True,
        "apply_effect_observed": True,
        "disabled_control_shadow_bitwise": True,
        "rng_progression_paired": True,
    }


def test_artifact_directory_is_immutable_and_cannot_be_overwritten(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "immutable"
    run_causal_ablation(output_dir)

    before = (output_dir / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_causal_ablation(output_dir)
    assert (output_dir / "manifest.json").read_bytes() == before


def test_failed_publication_cleans_staging_and_can_be_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "retryable"
    original_write = ablation_module._write_json_exclusive
    writes = 0

    def fail_after_two_files(path, payload):
        nonlocal writes
        writes += 1
        if writes == 3:
            raise RuntimeError("injected partial publication failure")
        return original_write(path, payload)

    monkeypatch.setattr(
        ablation_module,
        "_write_json_exclusive",
        fail_after_two_files,
    )
    with pytest.raises(RuntimeError, match="partial publication failure"):
        run_causal_ablation(output_dir)

    assert not output_dir.exists()
    assert list(tmp_path.glob(".retryable.staging-*")) == []

    monkeypatch.setattr(
        ablation_module,
        "_write_json_exclusive",
        original_write,
    )
    assert run_causal_ablation(output_dir)["valid"] is True
    assert verify_artifacts(output_dir)["valid"] is True


def test_verify_artifacts_detects_post_publication_tampering(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "tampered"
    run_causal_ablation(output_dir)
    arm_path = output_dir / "arms" / "shadow.json"
    payload = json.loads(arm_path.read_text(encoding="utf-8"))
    payload["loss"] += 1.0
    arm_path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(AblationContractError, match="SHA-256 mismatch"):
        verify_artifacts(output_dir)


def test_cli_runs_only_the_tiny_local_experiment(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.run_causal_ablation",
            "--output",
            str(output_dir),
            "--seed",
            "51",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["valid"] is True
    assert (output_dir / "verdict.json").exists()
