from __future__ import annotations

import json
from pathlib import Path

import yaml

from f51_darwin.darwin_x import DarwinXConfig


ROOT = Path(__file__).resolve().parents[2]


def _raw_config(name: str) -> dict:
    value = yaml.safe_load((ROOT / "src" / "configs" / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_supported_family_has_exactly_three_active_scales() -> None:
    policy = json.loads(
        (ROOT / "governance/audit/policy/operational-surface.json").read_text("utf-8")
    )
    family = policy["gold_boundaries"]["model_family"]
    active = {
        Path(family[key]).name
        for key in ("100m", "600m", "1.6b")
    }
    assert active == {
        "darwin_x_100m.yaml",
        "darwin_x_600m.yaml",
        "darwin_x_1.6b_nitro.yaml",
    }
    assert all((ROOT / "src" / "configs" / name).is_file() for name in active)


def test_100m_full_v1_enables_complete_causal_organism() -> None:
    raw = _raw_config("darwin_x_100m.yaml")
    config = DarwinXConfig.from_mapping(raw)

    assert config.model_name == "F51-Darwin-X-100M"
    assert config.context_length == 5120
    assert config.scan_chunk_size == 128
    assert config.gradient_checkpointing is True
    assert config.loss_semantics_version == 2
    assert config.mtp_weight > 0.0
    assert config.jepa_weight > 0.0
    # ghost temporarily disabled — hangs at step ~1100 with dual GPU + causal shadow
    assert config.ghost_enabled is False
    assert config.ghost_weight > 0.0
    assert config.curiosity_weight > 0.0
    assert config.spider_sense_enabled
    assert config.spider_calibration_enabled
    assert config.spider_calibration_weight > 0.0
    assert config.heartbeat_enabled
    assert config.ttm_residual_enabled
    assert config.gaba_enabled
    assert config.inter_hemispheric_enabled
    assert config.sleep_enabled
    assert config.decision_engine_enabled
    assert config.unified_mesh_enabled
    assert config.dae_enabled and not config.dae_shadow_mode
    assert config.nitro_enabled
    assert raw["checkpoint_root"] == "workspace/03_CHECKPOINTS_100M_FULL_V9"


def test_600m_and_16b_checkpoint_roots_are_isolated() -> None:
    configs = {
        "600m": _raw_config("darwin_x_600m.yaml"),
        "1.6b": _raw_config("darwin_x_1.6b_nitro.yaml"),
    }
    assert configs["600m"]["checkpoint_root"] == "workspace/03_CHECKPOINTS_600M"
    assert configs["1.6b"]["checkpoint_root"] == "workspace/03_CHECKPOINTS_1.6B"
    assert configs["1.6b"]["model_name"] == "F51-Darwin-X-1.6B-Nitro"
