from pathlib import Path

from dataclasses import asdict
import json
import subprocess
import sys

import torch
import yaml

from f51_darwin.darwin_x import DarwinXConfig, DarwinXModel
from f51_darwin.state_identity import backbone_identity
from scripts.darwin_organism import validate_v7_checkpoint_identity


def test_cloud_dae_profile_is_archived_and_non_operational() -> None:
    assert not Path("src/configs/darwin_x_1.6b_dae_cloud.yaml").exists()
    # O arquivo morto preserva o layout que foi arquivado. A migracao moveu
    # archive/ para governance/archive/, mas o find/replace de "configs/" para
    # "src/configs/" tambem atingiu este caminho, onde nao se aplica.
    archived = Path("governance/archive/cloud-local/configs/darwin_x_1.6b_dae_cloud.yaml")
    assert archived.is_file()
    config = yaml.safe_load(archived.read_text(encoding="utf-8"))
    assert config["dae_enabled"] is True


def test_cloud_deploy_is_archived_and_non_operational() -> None:
    assert not Path("src/scripts/deploy_dae_cloud.ps1").exists()
    archived = Path("governance/archive/cloud-local/scripts/deploy_dae_cloud.ps1")
    assert archived.is_file()
    assert "darwin_organism.py run247" in archived.read_text(encoding="utf-8")


def test_official_launcher_has_bounded_isolated_canary_contract() -> None:
    adapter = Path("src/scripts/start_overnight_16b.ps1").read_text(encoding="utf-8")
    implementation = Path(
        "src/f51_darwin/operations/start_overnight_16b.ps1"
    ).read_text(encoding="utf-8")
    text = adapter + "\n" + implementation
    assert '[string]$Root = ""' in text
    assert '[string]$DatasetRoot = ""' in text
    assert 'if (-not $Root) { $Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot) }' in text
    assert "if (-not $DatasetRoot) { $DatasetRoot = Join-Path $Root 'workspace' }" in text
    assert "00_CORPUS_PRINCIPAL_tokens_feast_v2.bin" in text
    assert "C:\\Users\\marco\\Desktop\\F51-Dataset-Organizado" not in text
    assert "C:\\Users\\" not in text
    assert "git executable is required on PATH" in text
    assert "cloud_future_lineage" not in text
    assert 'Join-Path $DatasetRoot "runtime\\runs\\overnight_16b"' in text
    assert "Write-JsonAtomic -Path $readinessPath" in text
    assert "-m tools.check_nitro_runtime --root $Root --require-hardware" in text
    assert "source_commit = $sourceCommit" in text
    assert "tracked_worktree_clean = $true" in text
    assert "override_provenance" in text
    assert "finally" in text
    assert "Restore-PcieAspmOnAc" in text
    assert "[switch]$Canary" in text
    assert "[int]$CanarySteps = 250" in text
    assert '"cycle"' in text
    assert '"--checkpoint-root"' in text
    assert '"--metrics-jsonl"' in text
    assert '"--holdout-tokens"' in text
    assert '"--base-checkpoint-sha256"' in text
    assert "WaitForExit" in text
    assert "Start-Sleep -Seconds $PostRunObservationSeconds" in text
    assert "13, 14, 153" in text
    assert "LiveKernelEvent" in text
    assert "SUB_PCIEXPRESS" in text
    assert "ASPM" in text
    assert "NoMatchingEventsFound" in text
    assert "-ErrorAction Stop" in text
    assert "$monitorErrors" in text
    assert "must be a child of the canonical checkpoint root" in text
    assert "escaped the isolated canary root" in text
    assert "$metricsTruncatedTail" in text
    assert "metrics_truncated_tail" in text
    assert "candidate_checkpoint_sha256" in text
    assert "topology_manifest_valid" in text
    assert "optimizer_resume_compatible" in text
    assert "$checkpointPath --config $configPath --verify-identity" in text
    assert 'if (-not $Canary) { $inspectArgs += "--verify-identity" }' not in text
    assert '"-m", "scripts.darwin_organism"' in text
    assert '"src/scripts/darwin_organism.py"' not in text
    assert "-m tools.inspect_darwin_x" in text
    assert "-m tools.check_nitro_runtime" in text


def test_checkpoint_inspector_exposes_strict_resume_contract() -> None:
    text = Path("src/scripts/inspect_organism_checkpoint.py").read_text(encoding="utf-8")
    assert '"topology_manifest_valid"' in text
    assert '"optimizer_resume_compatible"' in text
    assert '"identity_verified"' in text
    assert '"strict_resume_compatible"' in text


def test_checkpoint_inspector_fails_closed_without_topology_manifest(
    tmp_path: Path,
) -> None:
    config = DarwinXConfig(
        model_name="F51-Darwin-X-Inspector-Test",
        vocab_size=64,
        context_length=8,
        inference_context_length=16,
        d_model=16,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        fine_experts=4,
        shared_experts=1,
        experts_per_token=2,
        fine_expert_hidden_dim=8,
        shared_expert_hidden_dim=8,
        mtp_depth=1,
        mtp_weight=0.0,
        jepa_weight=0.0,
        ghost_weight=0.0,
        curiosity_weight=0.0,
        heartbeat_enabled=False,
        nitro_gpu_expert_capacity=4,
    )
    model = DarwinXModel(config)
    model_state = model.state_dict()
    optimizer = torch.optim.AdamW(model.parameters())
    payload = {
        "version": 7,
        "model_state_dict": model_state,
        "config": asdict(config),
        "topology_manifest": model.topology_manifest(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_type": "AdamW",
        "base_checkpoint_id": backbone_identity(model_state, config),
        "training_state": {"cycle": 1, "step": 1},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(asdict(config)), encoding="utf-8")

    def inspect(checkpoint: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.inspect_organism_checkpoint",
                str(checkpoint),
                "--config",
                str(config_path),
                "--verify-identity",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    valid_path = tmp_path / "valid.pt"
    torch.save(payload, valid_path)
    valid = inspect(valid_path)
    assert valid.returncode == 0, valid.stderr or valid.stdout
    valid_report = json.loads(valid.stdout)
    assert valid_report["strict_resume_compatible"] is True
    assert valid_report["topology_manifest_valid"] is True
    assert valid_report["optimizer_resume_compatible"] is True
    assert valid_report["identity_verified"] is True

    invalid_path = tmp_path / "missing_topology.pt"
    torch.save(
        {key: value for key, value in payload.items() if key != "topology_manifest"},
        invalid_path,
    )
    invalid = inspect(invalid_path)
    assert invalid.returncode == 2
    invalid_report = json.loads(invalid.stdout)
    assert invalid_report["strict_resume_compatible"] is False
    assert invalid_report["topology_manifest_valid"] is False


def test_historical_checkpoint_identity_uses_raw_embedded_config(
    tmp_path: Path,
) -> None:
    config = DarwinXConfig(
        model_name="F51-Darwin-X-Historical-Config-Test",
        vocab_size=64,
        context_length=8,
        inference_context_length=16,
        d_model=16,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        fine_experts=4,
        shared_experts=1,
        experts_per_token=2,
        fine_expert_hidden_dim=8,
        shared_expert_hidden_dim=8,
        mtp_depth=1,
        mtp_weight=0.0,
        jepa_weight=0.0,
        ghost_weight=0.0,
        curiosity_weight=0.0,
        heartbeat_enabled=False,
        nitro_gpu_expert_capacity=4,
    )
    model = DarwinXModel(config)
    model_state = model.state_dict()
    optimizer = torch.optim.AdamW(model.parameters())
    historical_raw = asdict(config)
    added_later = {
        "nitro_enabled",
        "dae_enabled",
        "dae_shadow_mode",
        "dae_update_gain_min",
        "dae_update_gain_max",
    }
    for key in added_later:
        historical_raw.pop(key)
    declared = backbone_identity(model_state, historical_raw)
    payload = {
        "version": 7,
        "model_state_dict": model_state,
        "config": historical_raw,
        "topology_manifest": model.topology_manifest(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_type": "AdamW",
        "base_checkpoint_id": declared,
        "training_state": {"cycle": 71, "step": 40751},
    }
    checkpoint = tmp_path / "historical.pt"
    torch.save(payload, checkpoint)
    config_path = tmp_path / "current.yaml"
    config_path.write_text(yaml.safe_dump(asdict(config)), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.inspect_organism_checkpoint",
            str(checkpoint),
            "--config",
            str(config_path),
            "--verify-identity",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    report = json.loads(result.stdout)
    assert report["identity_verified"] is True
    assert report["identity"]["with_raw_embedded_config"] == declared
    assert report["identity"]["with_current_file_config"] != declared
    assert report["embedded_config_matches_file"] is True
    assert report["strict_resume_compatible"] is True
    assert validate_v7_checkpoint_identity(payload, model_state) == declared
