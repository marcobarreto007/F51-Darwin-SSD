import json
import os

from f51_darwin.legacy_layers import LegacyLayers, LayerTier


def test_update_expert_tier_is_idempotent(tmp_path):
    legacy = LegacyLayers(tmp_path / "legacy")

    legacy.update_expert_tier("L0_E0", None, LayerTier.GPU)
    legacy.update_expert_tier("L0_E0", None, LayerTier.GPU)

    assert legacy.layers[LayerTier.GPU].experts == {"L0_E0"}
    assert legacy.layers[LayerTier.GPU].count == 1


def test_load_repairs_duplicate_membership_and_inflated_counts(tmp_path):
    save_dir = tmp_path / "legacy"
    save_dir.mkdir()
    (save_dir / "legacy_state.json").write_text(
        json.dumps(
            {
                "cycle": 3,
                "layers": {
                    "GPU": {"count": 434, "experts": ["L0_E0", "L0_E1"]},
                    "RAM": {"count": 144, "experts": ["L0_E0"]},
                },
                "graves": {},
                "inhibitors": {},
                "dna": {
                    "seed_architecture": {"model_name": "test"},
                    "immutable_params": {},
                    "tokenizer_config": {"vocab_size": 32},
                    "birth": "2026-07-10T00:00:00+00:00",
                },
            }
        ),
        encoding="utf-8",
    )

    legacy = LegacyLayers(save_dir)

    assert legacy.layers[LayerTier.GPU].experts == {"L0_E0", "L0_E1"}
    assert legacy.layers[LayerTier.GPU].count == 2
    assert legacy.layers[LayerTier.RAM].experts == set()
    assert legacy.layers[LayerTier.RAM].count == 0
    assert legacy.layers[LayerTier.DNA].count == 1
    assert legacy.dna is not None
    assert legacy.dna.birth_timestamp == "2026-07-10T00:00:00+00:00"


def test_save_retries_transient_windows_replace_lock(tmp_path, monkeypatch):
    legacy = LegacyLayers(tmp_path / "legacy")
    real_replace = os.replace
    attempts = 0

    def flaky_replace(source, target):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("transient scanner lock")
        return real_replace(source, target)

    monkeypatch.setattr("f51_darwin.legacy_layers.os.replace", flaky_replace)

    legacy.update_expert_tier("L0_E0", None, LayerTier.GPU)

    assert attempts == 2
    assert (tmp_path / "legacy" / "legacy_state.json").exists()
    assert not list((tmp_path / "legacy").glob(".*.tmp"))
