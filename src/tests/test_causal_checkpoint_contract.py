from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from f51_darwin.organism import cli as organism_cli
from f51_darwin.organism.checkpoint import (
    EMPTY_CAUSAL_LEDGER_HEAD,
    build_v8_causal_contract,
    checkpoint_version_for_runtime,
    validate_v8_causal_contract,
)
from f51_darwin.organism.causal_adapters import training_step_identity
from f51_darwin.organism.cli import _build_parser
from f51_darwin.organism.config import DarwinOrganismConfig
from f51_darwin.organism.control import DarwinOrganism
from f51_darwin.darwin_x import DarwinXConfig, DarwinXModel
from f51_darwin.state_identity import (
    backbone_identity,
    training_contract_identity,
)


def test_v7_legacy_contract_remains_v7_without_causal_fields() -> None:
    assert DarwinOrganismConfig().causal_v8_migration is False
    assert (
        checkpoint_version_for_runtime(
            loss_semantics_version=1,
            causal_mode="disabled",
            causal_v8_migration=False,
        )
        == 7
    )
    assert (
        checkpoint_version_for_runtime(
            loss_semantics_version=1,
            causal_mode="disabled",
            causal_v8_migration=True,
        )
        == 7
    )


def test_cli_requires_explicit_operator_migration_flag(
    monkeypatch,
    tmp_path,
) -> None:
    parser = _build_parser(None, "config.yaml")

    legacy = parser.parse_args(["cycle"])
    causal = parser.parse_args(
        [
            "cycle",
            "--causal-mode",
            "shadow",
            "--causal-v8-migration",
        ]
    )

    assert legacy.causal_mode == "disabled"
    assert legacy.causal_v8_migration is False
    assert causal.causal_mode == "shadow"
    assert causal.causal_v8_migration is True

    captured = {}

    def capture_config(config, model_config_path):
        captured["config"] = config
        captured["model_config_path"] = model_config_path
        return SimpleNamespace()

    monkeypatch.setattr(organism_cli, "DarwinOrganism", capture_config)
    workspace = SimpleNamespace(
        corpus=tmp_path / "corpus",
        checkpoints=tmp_path / "checkpoints",
        runs=tmp_path / "runs",
    )
    model_config = tmp_path / "config.yaml"
    model_config.write_text("{}\n", encoding="utf-8")
    causal.config = str(model_config)
    organism_cli._build_organism(causal, parser, workspace, None)

    assert captured["config"].causal_mode == "shadow"
    assert captured["config"].causal_v8_migration is True


def test_config_rejects_non_boolean_migration_authorization() -> None:
    with pytest.raises(TypeError, match="explicit boolean"):
        DarwinOrganismConfig(causal_v8_migration="true")  # type: ignore[arg-type]


def test_causal_step_uses_the_persisted_training_contract_identity() -> None:
    organism = SimpleNamespace(
        cfg=SimpleNamespace(
            canary_run_id="run-test",
            organism_name="test",
            causal_mode="shadow",
        ),
        cycle=1,
        total_steps=2,
        base_checkpoint_id="darwin-model-core-v1:" + "a" * 64,
    )

    identity = training_step_identity(
        organism,
        torch.tensor([[1, 2]], dtype=torch.long),
        accumulation_window=0,
    )

    assert identity.training_contract_id == training_contract_identity()


@pytest.mark.parametrize(
    ("loss_semantics_version", "causal_mode"),
    [
        (2, "disabled"),
        (1, "control"),
        (1, "shadow"),
        (1, "enforce"),
    ],
)
def test_causal_runtime_rejects_implicit_v8_migration(
    loss_semantics_version: int,
    causal_mode: str,
) -> None:
    with pytest.raises(ValueError, match="causal_v8_migration"):
        checkpoint_version_for_runtime(
            loss_semantics_version=loss_semantics_version,
            causal_mode=causal_mode,
            causal_v8_migration=False,
        )


def _v8_payload(*, ledger_head: str = "a" * 64) -> dict:
    loss_semantics_version = 2
    causal_mode = "shadow"
    return {
        "version": 8,
        "training_contract_id": training_contract_identity(),
        "loss_semantics_version": loss_semantics_version,
        "causal_contract": build_v8_causal_contract(
            loss_semantics_version=loss_semantics_version,
            causal_mode=causal_mode,
            ledger_head=ledger_head,
        ),
    }


def test_v8_causal_contract_roundtrip() -> None:
    payload = _v8_payload()

    validated = validate_v8_causal_contract(
        payload,
        loss_semantics_version=2,
        causal_mode="shadow",
        causal_v8_migration=True,
        ledger_head="a" * 64,
    )

    assert validated == payload["causal_contract"]
    assert validated["state"] == {
        "bus_enabled": True,
        "ablation_arm": "SHADOW",
    }


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("training_contract_id", "wrong", "training_contract_id"),
        ("loss_semantics_version", 1, "loss_semantics_version"),
    ],
)
def test_v8_causal_contract_rejects_top_level_mismatch(
    field: str,
    replacement: object,
    message: str,
) -> None:
    payload = _v8_payload()
    payload[field] = replacement

    with pytest.raises(ValueError, match=message):
        validate_v8_causal_contract(
            payload,
            loss_semantics_version=2,
            causal_mode="shadow",
            causal_v8_migration=True,
            ledger_head="a" * 64,
        )


def test_v8_causal_contract_rejects_mode_and_state_mismatch() -> None:
    payload = _v8_payload()
    payload["causal_contract"]["mode"] = "enforce"

    with pytest.raises(ValueError, match="causal mode"):
        validate_v8_causal_contract(
            payload,
            loss_semantics_version=2,
            causal_mode="shadow",
            causal_v8_migration=True,
            ledger_head="a" * 64,
        )

    payload = _v8_payload()
    payload["causal_contract"]["state"]["ablation_arm"] = "APPLY"
    with pytest.raises(ValueError, match="causal state"):
        validate_v8_causal_contract(
            payload,
            loss_semantics_version=2,
            causal_mode="shadow",
            causal_v8_migration=True,
            ledger_head="a" * 64,
        )


def test_v8_causal_contract_rejects_tampered_ledger_head() -> None:
    payload = _v8_payload(ledger_head="b" * 64)

    with pytest.raises(ValueError, match="ledger_head"):
        validate_v8_causal_contract(
            payload,
            loss_semantics_version=2,
            causal_mode="shadow",
            causal_v8_migration=True,
            ledger_head="a" * 64,
        )


def test_loss_v2_without_bus_uses_explicit_empty_ledger_head() -> None:
    contract = build_v8_causal_contract(
        loss_semantics_version=2,
        causal_mode="disabled",
        ledger_head=EMPTY_CAUSAL_LEDGER_HEAD,
    )

    assert contract["ledger_head"] == EMPTY_CAUSAL_LEDGER_HEAD
    assert contract["state"] == {
        "bus_enabled": False,
        "ablation_arm": None,
    }


def _minimal_checkpoint_organism(
    tmp_path,
    *,
    loss_semantics_version: int,
    causal_mode: str,
    causal_v8_migration: bool,
) -> DarwinOrganism:
    organism = DarwinOrganism.__new__(DarwinOrganism)
    organism.root = tmp_path
    organism.cfg = SimpleNamespace(
        checkpoint_root="checkpoints",
        runs_dir="runs",
        optimizer_name="sgd",
        async_checkpoint=False,
        causal_mode=causal_mode,
        causal_v8_migration=causal_v8_migration,
    )
    organism.cycle = 1
    organism.total_steps = 10
    organism.model = nn.Linear(2, 2)
    organism.model.topology_manifest = lambda: {
        "version": 7,
        "base_config": {},
        "topology": [],
        "neuroendocrine_state": [],
        "organism_memory": {},
    }
    organism.model_config = SimpleNamespace(
        model_name="tiny",
        loss_semantics_version=loss_semantics_version,
    )
    organism.optimizer = torch.optim.SGD(
        organism.model.parameters(), lr=0.01
    )
    organism.tokenizer_id = "tokenizer-test"
    organism.replay = SimpleNamespace(state_dict=lambda: {"records": []})
    organism._ashes_streak = {}
    organism.expert_pool = SimpleNamespace(metadata=lambda: {})
    organism.legacy = SimpleNamespace(
        status=lambda: {},
        save=lambda: None,
        cycle=0,
    )
    organism.lineage = SimpleNamespace(
        population_stats=lambda: {},
        timeline=lambda count: [],
        save_report=lambda: None,
        _total_births=0,
        _total_deaths=0,
    )
    organism.soul = SimpleNamespace(
        blessing=lambda: "test",
        family=SimpleNamespace(mantra=lambda: "test"),
    )
    organism.resume_checkpoint_path = None
    organism.resume_migration = None
    organism.causal_ledger = None
    organism._join_pending_save = lambda: None
    return organism


def test_save_fails_before_writing_when_v8_migration_is_implicit(
    tmp_path,
) -> None:
    organism = _minimal_checkpoint_organism(
        tmp_path,
        loss_semantics_version=2,
        causal_mode="disabled",
        causal_v8_migration=False,
    )

    with pytest.raises(ValueError, match="causal_v8_migration"):
        organism._save_cycle({})

    assert not (tmp_path / "checkpoints").exists()


def test_loss_v2_checkpoint_save_persists_v8_contract(tmp_path) -> None:
    organism = _minimal_checkpoint_organism(
        tmp_path,
        loss_semantics_version=2,
        causal_mode="disabled",
        causal_v8_migration=True,
    )

    organism._save_cycle({"cycle": 1})

    checkpoint = tmp_path / "checkpoints" / "organism_cycle_001.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["version"] == 9
    assert payload["training_contract_id"] == training_contract_identity()
    assert payload["loss_semantics_version"] == 2
    assert payload["causal_contract"] == build_v8_causal_contract(
        loss_semantics_version=2,
        causal_mode="disabled",
        ledger_head=EMPTY_CAUSAL_LEDGER_HEAD,
    )
    # ── v9 blockchain fields ──
    assert "checkpoint_version" in payload
    assert payload["checkpoint_version"] == 9
    assert "block_hash" in payload
    assert "block_number" in payload
    assert "checkpoint_sha256" in payload
    assert "ledger_snapshot_hash" in payload
    pointer = json.loads(
        (tmp_path / "checkpoints" / "organism_latest.json").read_text()
    )
    assert pointer["checkpoint_version"] == 9


def test_strict_resume_rejects_tampered_v8_ledger_head(tmp_path) -> None:
    config = DarwinXConfig(
        model_name="tiny-v8",
        vocab_size=32,
        context_length=32,
        inference_context_length=32,
        d_model=16,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        fine_experts=2,
        shared_experts=1,
        experts_per_token=1,
        fine_expert_hidden_dim=16,
        shared_expert_hidden_dim=16,
        mtp_depth=1,
        heartbeat_enabled=False,
        loss_semantics_version=2,
    )
    model = DarwinXModel(config)
    state = model.state_dict()
    payload = {
        "version": 8,
        "base_checkpoint_id": backbone_identity(state, config),
        "tokenizer_id": "tokenizer-test",
        "model_state_dict": state,
        "topology_manifest": model.topology_manifest(),
        "config": dict(config.__dict__),
        "training_contract_id": training_contract_identity(),
        "loss_semantics_version": 2,
        "causal_contract": build_v8_causal_contract(
            loss_semantics_version=2,
            causal_mode="shadow",
            ledger_head="b" * 64,
        ),
    }
    checkpoint = tmp_path / "tampered-head.pt"
    torch.save(payload, checkpoint)

    organism = DarwinOrganism.__new__(DarwinOrganism)
    organism.model = model
    organism.model_config = config
    organism.tokenizer_id = "tokenizer-test"
    organism.cfg = SimpleNamespace(
        causal_mode="shadow",
        causal_v8_migration=True,
    )
    organism.checkpoint_version = 8
    organism.causal_ledger = SimpleNamespace(
        head_hash="a" * 64,
        verify=lambda: SimpleNamespace(
            valid=True,
            head_hash="a" * 64,
            errors=(),
        ),
        truncate_to_head=lambda _hash: None,
        recover_head=lambda: "a" * 64,
    )

    with pytest.raises(ValueError, match="ledger_head"):
        organism._restore_resume(str(checkpoint))
