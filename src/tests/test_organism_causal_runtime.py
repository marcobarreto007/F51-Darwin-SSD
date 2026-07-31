from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from f51_darwin.expert_pool import ExpertModule, ExpertPool, ModuleState
from f51_darwin.legacy_layers import LayerTier, LegacyLayers
from f51_darwin.lineage_tracker import LineageTracker
from f51_darwin.replay_buffer import ReplayBuffer
from f51_darwin.state_identity import backbone_identity
from scripts.darwin_organism import (
    DarwinOrganism,
    assert_optimizer_device_invariants,
    advance_ashes_streak,
    metric_channel_lm_loss,
    replay_step_due,
    retired_expert_ids,
    token_source_state,
    canary_token_source_identity,
    validate_canary_lineage,
    validate_canary_paths,
    validate_run247_lineage,
    validate_legacy_v6_resume,
    migrate_adamw_state_with_new_baselines,
    remap_optimizer_state,
)


def test_optimizer_device_invariant_rejects_mismatch() -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    parameter = next(model.parameters())
    optimizer.state[parameter]["exp_avg"] = torch.zeros_like(
        parameter, device="meta"
    )

    with pytest.raises(RuntimeError, match="optimizer device invariant"):
        assert_optimizer_device_invariants(model, optimizer)


def test_metric_channel_lm_loss_returns_numeric_snapshot_value() -> None:
    channel = SimpleNamespace(
        snapshot=lambda: {"count": 3, "total_loss": 2.5, "lm_loss": 1.25}
    )

    assert metric_channel_lm_loss(channel, default=None) == pytest.approx(1.25)


def test_metric_channel_lm_loss_uses_default_for_empty_window() -> None:
    channel = SimpleNamespace(
        snapshot=lambda: {"count": 0, "total_loss": None, "lm_loss": None}
    )

    assert metric_channel_lm_loss(channel, default=4.5) == pytest.approx(4.5)
    assert metric_channel_lm_loss(channel, default=None) is None


def test_canary_checkpoint_child_root_is_isolated(tmp_path: Path) -> None:
    canonical = tmp_path / "03_CHECKPOINTS"
    canonical.mkdir()

    validate_canary_paths(
        canonical_root=canonical,
        canary_root=canonical / "canary_recovery_20260715",
    )
    with pytest.raises(ValueError, match="isolated"):
        validate_canary_paths(canonical_root=canonical, canary_root=canonical)
    with pytest.raises(ValueError, match="isolated"):
        validate_canary_paths(canonical_root=canonical, canary_root=tmp_path)
    with pytest.raises(ValueError, match="isolated"):
        validate_canary_paths(
            canonical_root=canonical,
            canary_root=canonical.parent / "canary_sibling",
        )


def test_run247_requires_explicit_lineage_choice() -> None:
    validate_run247_lineage("run247", "organism_cycle_071.pt", False)
    validate_run247_lineage("run247", None, True)
    validate_run247_lineage("cycle", None, False)

    with pytest.raises(ValueError, match="requires --resume"):
        validate_run247_lineage("run247", None, False)
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_run247_lineage(
            "run247", "organism_cycle_071.pt", True
        )


def test_canary_accepts_exactly_one_explicit_lineage_source() -> None:
    validate_canary_lineage(
        resume=None, fresh_start=True, base_checkpoint_sha256=None
    )
    validate_canary_lineage(
        resume="checkpoint.pt",
        fresh_start=False,
        base_checkpoint_sha256="a" * 64,
    )

    with pytest.raises(ValueError, match="exactly one"):
        validate_canary_lineage(
            resume=None, fresh_start=False, base_checkpoint_sha256=None
        )
    with pytest.raises(ValueError, match="exactly one"):
        validate_canary_lineage(
            resume="checkpoint.pt",
            fresh_start=True,
            base_checkpoint_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="requires --base-checkpoint"):
        validate_canary_lineage(
            resume="checkpoint.pt",
            fresh_start=False,
            base_checkpoint_sha256=None,
        )


def test_v7_resume_identity_drift_fails_closed() -> None:
    source = Path("src/f51_darwin/organism/checkpoint.py").read_text(encoding="utf-8")

    assert 'raise ValueError("checkpoint backbone identity does not match its tensors")' in source
    assert "backbone identity hash drifted" not in source


def test_run247_has_no_research_backdoor_or_direct_corpus_promotion() -> None:
    source = Path("src/f51_darwin/organism/cli.py").read_text(encoding="utf-8")

    assert "research.ghost_stream" not in source
    assert "research.ghost_feeder" not in source
    assert "scan_for_new_data" not in source
    assert "disc_path.write_text" not in source
    assert "register_candidate" in source
    assert "SourceType.EDITED" in source


class _TinyCausalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.head = nn.Linear(8, 32)
        self.batch_sizes: list[int] = []
        self.batches: list[torch.Tensor] = []

    def forward(self, input_ids, labels=None, domain=None, heartbeat=None):
        self.batch_sizes.append(input_ids.shape[0])
        self.batches.append(input_ids.detach().cpu().clone())
        logits = self.head(self.embedding(input_ids))
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return SimpleNamespace(
            loss=loss,
            lm_loss=loss,
            aux_loss=torch.zeros((), device=loss.device),
            heartbeat_stats={},
        )


def test_replay_schedule_uses_exact_fraction_without_batch_spike() -> None:
    replay_steps = [step for step in range(1, 101) if replay_step_due(step, 0.20)]

    assert len(replay_steps) == 20
    assert replay_steps[:3] == [5, 10, 15]


def test_token_source_state_records_mutable_corpus_identity(tmp_path) -> None:
    token_path = tmp_path / "tokens_live.bin"
    token_path.write_bytes(b"\x00" * 16)

    state = token_source_state(token_path, token_count=4)

    assert state["path"] == str(token_path.resolve())
    assert state["size_bytes"] == 16
    assert state["token_count"] == 4
    assert state["int32_aligned"] is True


def test_canary_token_identity_requires_matching_manifest(tmp_path: Path) -> None:
    token_path = tmp_path / "tokens.bin"
    token_path.write_bytes(b"\x00" * 16)
    manifest_path = Path(str(token_path) + ".manifest.json")
    manifest_path.write_text(json.dumps({
        "tokens": 4,
        "bytes": 16,
        "dtype": "int32",
        "little_endian": True,
        "sha256": "a" * 64,
    }), encoding="utf-8")

    identity = canary_token_source_identity(token_path, 4)

    assert identity["manifest_sha256"] == "a" * 64
    assert identity["manifest_path"] == str(manifest_path.resolve())


def test_ashes_streak_requires_two_distinct_cycles() -> None:
    first_cycle = advance_ashes_streak(0, entering_ashes=True)
    second_cycle = advance_ashes_streak(first_cycle, entering_ashes=False)

    assert first_cycle == 1
    assert second_cycle == 2


def test_train_cycle_uses_twenty_percent_replay_without_larger_batch() -> None:
    organism = DarwinOrganism.__new__(DarwinOrganism)
    organism.cfg = SimpleNamespace(
        block_size=4,
        batch_size=1,
        seed=51,
        replay_warmup_examples=4,
        replay_ratio=0.20,
        replay_add_every=20,
        replay_sample_size=2,
        eval_every=10,
        grad_clip=1.0,
        accum_steps=1,
        holdout_tokens=0,
    )
    organism.cycle = 1
    organism.total_steps = 0
    organism.device = torch.device("cpu")
    organism.amp_dtype = None
    organism.token_ids = np.arange(256, dtype=np.int64) % 32
    organism.holdout_starts = ()
    organism.token_count = len(organism.token_ids)
    organism.replay = ReplayBuffer(capacity=16, seed=51)
    for offset in range(4):
        organism.replay.add([offset, offset + 1, offset + 2, offset + 3])
    organism.model = _TinyCausalModel()
    organism.optimizer = torch.optim.SGD(organism.model.parameters(), lr=0.01)
    organism.soul = SimpleNamespace(heartbeat=lambda status: None)
    report = {"forgetting": 0.0}

    organism._train_cycle(10, report)

    assert report["replay_train_count"] == 2
    assert report["fresh_train_count"] == 8
    assert report["replay_fraction"] == pytest.approx(0.20)
    assert report["metrics"]["fresh"]["count"] == 8
    assert report["metrics"]["replay"]["count"] == 2
    assert report["replay_loss"] > 0
    assert organism.model.batch_sizes == [1] * 12


def test_train_cycle_excludes_tail_holdout_from_optimizer_batches() -> None:
    organism = DarwinOrganism.__new__(DarwinOrganism)
    organism.cfg = SimpleNamespace(
        block_size=4,
        batch_size=1,
        seed=51,
        replay_warmup_examples=100,
        replay_ratio=0.20,
        replay_add_every=20,
        replay_sample_size=2,
        eval_every=2,
        grad_clip=1.0,
        accum_steps=1,
        holdout_tokens=32,
        holdout_batches=2,
        metrics_jsonl=None,
    )
    organism.cycle = 1
    organism.total_steps = 0
    organism.device = torch.device("cpu")
    organism.amp_dtype = None
    organism.token_ids = np.concatenate((
        np.zeros(96, dtype=np.int64),
        np.ones(32, dtype=np.int64),
    ))
    organism.train_token_ids = organism.token_ids[:96]
    organism.holdout_token_ids = organism.token_ids[96:]
    organism.holdout_starts = ((0,), (8,))
    organism.token_count = len(organism.token_ids)
    organism.replay = ReplayBuffer(capacity=16, seed=51)
    organism.model = _TinyCausalModel()
    organism.optimizer = torch.optim.SGD(organism.model.parameters(), lr=0.01)
    organism.soul = SimpleNamespace(heartbeat=lambda status: None)
    organism._train_cycle(2, {"forgetting": 0.0})

    assert all(torch.count_nonzero(batch) == 0 for batch in organism.model.batches[:2])
    assert all(torch.all(batch == 1) for batch in organism.model.batches[-2:])


def test_dae_runs_before_clip_and_optimizer_step(monkeypatch) -> None:
    organism = DarwinOrganism.__new__(DarwinOrganism)
    organism.cfg = SimpleNamespace(
        block_size=4,
        batch_size=1,
        seed=51,
        replay_warmup_examples=100,
        replay_ratio=0.0,
        replay_add_every=20,
        replay_sample_size=1,
        eval_every=10,
        grad_clip=1.0,
        accum_steps=1,
        holdout_tokens=0,
    )
    organism.cycle = 1
    organism.total_steps = 0
    organism.device = torch.device("cpu")
    organism.amp_dtype = None
    organism.token_ids = np.arange(64, dtype=np.int64) % 32
    organism.token_count = len(organism.token_ids)
    organism.holdout_starts = ()
    organism.replay = ReplayBuffer(capacity=4, seed=51)
    organism.model = _TinyCausalModel()
    organism.optimizer = torch.optim.SGD(organism.model.parameters(), lr=0.01)
    organism.soul = SimpleNamespace(heartbeat=lambda status: None)
    events: list[str] = []
    organism.model.apply_active_gradient_actions = (
        lambda: events.append("dae") or []
    )
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda *args, **kwargs: events.append("clip"),
    )
    monkeypatch.setattr(
        organism.optimizer,
        "step",
        lambda *args, **kwargs: events.append("step"),
    )

    organism._train_cycle(1, {"forgetting": 0.0})

    assert events[:3] == ["dae", "clip", "step"]


def test_completed_cycle_always_publishes_checkpoint() -> None:
    organism = DarwinOrganism.__new__(DarwinOrganism)
    organism.cycle = 4
    organism.total_steps = 32500
    organism.cfg = SimpleNamespace(max_steps_per_cycle=1, weight_decay=0.0)
    organism.model = SimpleNamespace(execute_structural_actions=lambda: [])
    saved: list[dict] = []
    organism._data_lifecycle = lambda report: None

    def train(steps, report):
        organism.total_steps += steps
        report["steps"] = steps

    organism._train_cycle = train
    organism._evolution_cycle = lambda report: None
    organism._save_cycle = lambda report: saved.append(dict(report))

    report = organism.run_cycle(steps=1)

    assert organism.cycle == 5
    assert report["steps"] == 1
    assert saved == [report]


def test_replay_evaluation_uses_configured_autocast() -> None:
    organism = DarwinOrganism.__new__(DarwinOrganism)
    organism.cfg = SimpleNamespace(batch_size=1, block_size=4)
    organism.device = torch.device("cpu")
    organism.amp_dtype = torch.bfloat16
    organism.model = _TinyCausalModel().to(dtype=torch.bfloat16)
    samples = [
        SimpleNamespace(input_ids=(1, 2, 3, 4)),
        SimpleNamespace(input_ids=(4, 3, 2, 1)),
    ]

    loss = organism._evaluate_replay_loss(organism.model, samples)

    assert loss is not None
    assert loss > 0
    assert organism.model.batch_sizes == [1, 1]


def test_replay_state_roundtrip_preserves_samples_and_rng() -> None:
    source = ReplayBuffer(capacity=4, seed=51)
    for index in range(20):
        source.add([index, index + 1], label=f"sample-{index}")

    restored = ReplayBuffer(capacity=4, seed=999)
    restored.load_state_dict(source.state_dict())

    assert restored.seen_count == 20
    assert restored.to_records() == source.to_records()
    assert restored.sample(3) == source.sample(3)


def test_expert_pool_routes_through_bound_real_expert_without_shadow_module() -> None:
    pool = ExpertPool()
    real_expert = ExpertModule(4)
    pool.add_expert("L0_E0", None, created_at_cycle=0, state=ModuleState.ACTIVE)
    pool.bind_real_expert("L0_E0", 0, 0, real_expert)
    inputs = torch.randn(1, 2, 4)

    routed = pool.route_active(inputs)

    assert routed.shape == inputs.shape
    assert "L0_E0" not in pool.modules_by_id


def test_legacy_ashes_and_resurrection_change_real_expert(tmp_path) -> None:
    pool = ExpertPool()
    real_expert = ExpertModule(4)
    pool.add_expert("L0_E0", None, created_at_cycle=0, state=ModuleState.ACTIVE)
    pool.bind_real_expert("L0_E0", 0, 0, real_expert)
    router = SimpleNamespace(external_bias=torch.zeros(1))
    model = SimpleNamespace(blocks=[SimpleNamespace(moe=SimpleNamespace(fine_router=router))])
    legacy = LegacyLayers(tmp_path / "legacy")
    legacy.set_causal_refs(pool, model)
    legacy.update_expert_tier("L0_E0", None, LayerTier.GPU)

    legacy.bury_expert("L0_E0", "low_temperature", -1.0, [])

    assert all(not parameter.requires_grad for parameter in real_expert.parameters())
    assert router.external_bias[0].item() == pytest.approx(-5.0)
    assert pool.records["L0_E0"].state == ModuleState.ACTIVE

    legacy.resurrect_expert("L0_E0")

    assert all(parameter.requires_grad for parameter in real_expert.parameters())
    assert router.external_bias[0].item() == pytest.approx(0.0)
    assert "L0_E0" not in legacy.graves
    assert legacy.expert_tier("L0_E0") == LayerTier.GPU


def test_retired_experts_are_derived_from_persistent_legacy_state(tmp_path) -> None:
    pool = ExpertPool()
    real_expert = ExpertModule(4)
    pool.add_expert("L0_E0", None, created_at_cycle=0, state=ModuleState.DEAD)
    pool.bind_real_expert("L0_E0", 0, 0, real_expert)
    legacy = LegacyLayers(tmp_path / "legacy")
    legacy.update_expert_tier("L0_E0", None, LayerTier.ASHES)

    assert retired_expert_ids(pool, legacy) == ["L0_E0"]


def test_legacy_grave_roundtrip_restores_enum(tmp_path) -> None:
    save_dir = tmp_path / "legacy"
    legacy = LegacyLayers(save_dir)
    legacy.update_expert_tier("L0_E0", None, LayerTier.GPU)
    legacy.bury_expert("L0_E0", "low_temperature", -1.0, [])

    restored = LegacyLayers(save_dir)
    restored.save()

    assert restored.graves["L0_E0"].original_layer == LayerTier.GPU


def test_legacy_rejects_corrupt_state_instead_of_silently_resetting(tmp_path) -> None:
    save_dir = tmp_path / "legacy"
    save_dir.mkdir()
    (save_dir / "legacy_state.json").write_text("{broken", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid LegacyLayers state"):
        LegacyLayers(save_dir)


def test_lineage_reloads_existing_history_without_duplicate_birth(tmp_path) -> None:
    tracker = LineageTracker(tmp_path / "lineage")
    tracker.record_birth("L0_E0", cycle=0)
    tracker.record_death("L0_E0", cycle=3)

    restored = LineageTracker(tmp_path / "lineage")

    assert restored.has_module("L0_E0")
    assert restored.population_stats()["total_born"] == 1
    assert restored.population_stats()["total_died"] == 1
    assert restored.population_stats()["alive"] == 0

    restored.record_promotion(
        "L0_E0", cycle=4, from_state="dead", to_state="active"
    )
    assert restored.population_stats()["alive"] == 1
    assert LineageTracker(tmp_path / "lineage").population_stats()["alive"] == 1


def test_cycle_checkpoint_is_atomic_and_contains_replay_state(tmp_path) -> None:
    organism = DarwinOrganism.__new__(DarwinOrganism)
    organism.root = tmp_path
    organism.cfg = SimpleNamespace(checkpoint_root="checkpoints")
    organism.cycle = 1
    organism.total_steps = 10
    organism.train_tokens_seen = 40960
    organism.model = nn.Linear(2, 2)
    organism.model.topology_manifest = lambda: {
        "version": 7,
        "base_config": {"model_name": "tiny", "d_model": 2, "n_layers": 0},
        "topology": [],
        "neuroendocrine_state": [],
        "organism_memory": {},
    }
    organism.model_config = SimpleNamespace(model_name="tiny")
    organism.tokenizer_id = "tokenizer-test-id"
    organism.optimizer = torch.optim.SGD(organism.model.parameters(), lr=0.01)
    organism.replay = ReplayBuffer(capacity=4, seed=51)
    organism.replay.add([1, 2], label="ancestral")
    organism._ashes_streak = {"L0_E0": 2}
    organism.expert_pool = ExpertPool()
    organism.legacy = SimpleNamespace(
        status=lambda: {"total_experts": 0},
        save=lambda: None,
        cycle=0,
    )
    organism.lineage = SimpleNamespace(
        population_stats=lambda: {"alive": 0},
        timeline=lambda count: [],
        save_report=lambda: None,
        _total_births=0,
        _total_deaths=0,
    )
    organism.soul = SimpleNamespace(
        blessing=lambda: "test",
        family=SimpleNamespace(mantra=lambda: "test"),
    )
    organism.resume_migration = {"schema": "test-migration"}
    old_checkpoint = tmp_path / "checkpoints" / "organism_cycle_000.pt"
    old_checkpoint.parent.mkdir(parents=True)
    old_checkpoint.write_bytes(b"scientific evidence")

    expected_base_id = backbone_identity(
        organism.model.state_dict(), organism.model_config
    )
    online_path = tmp_path / "runs" / "online_learning" / "adapter_state.pt"
    online_path.parent.mkdir(parents=True)
    torch.save(
        {
            "version": 1,
            "adapter_rank": 1,
            "d_model": 2,
            "base_checkpoint_id": expected_base_id,
            "tokenizer_id": organism.tokenizer_id,
            "adapter_state": {"up.weight": torch.ones(1)},
            "optimizer_state": {},
            "counters": {"interactions": 1},
        },
        online_path,
    )

    organism._save_cycle({"cycle": 1})

    checkpoint = tmp_path / "checkpoints" / "organism_cycle_001.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["version"] == 7
    assert payload["topology_manifest"]["version"] == 7
    assert payload["replay_buffer"]["records"][0]["input_ids"] == [1, 2]
    assert payload["organism"]["ashes_streak"] == {"L0_E0": 2}
    assert "rng_state" in payload
    assert "heartbeat_state" in payload
    assert payload["base_checkpoint_id"] == expected_base_id
    assert payload["tokenizer_id"] == organism.tokenizer_id
    assert payload["online_learning_state"]["adapter_state"]["up.weight"].item() == 1
    assert payload["online_learning_artifact"]["embedded"] is True
    assert payload["checkpoint_migration"] == {"schema": "test-migration"}
    assert payload["optimizer_identity"]["class"] == "SGD"
    assert payload["training_state"]["train_tokens_seen"] == 40960
    assert payload["dae"]["enabled"] is False
    assert payload["dae"]["shadow_mode"] is True
    assert old_checkpoint.read_bytes() == b"scientific evidence"
    pointer = json.loads((tmp_path / "checkpoints" / "organism_latest.json").read_text())
    assert pointer["path"] == "organism_cycle_001.pt"
    assert pointer["checkpoint_version"] == 7
    assert not checkpoint.with_suffix(".pt.tmp").exists()


def test_v6_migration_accepts_only_known_new_buffers() -> None:
    from f51_darwin.darwin_x import DarwinXConfig, DarwinXModel

    config = DarwinXConfig(
        model_name="tiny-v6",
        vocab_size=32,
        context_length=32,
        inference_context_length=32,
        d_model=16,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        fine_experts=2,
        shared_experts=1,
        experts_per_token=1,
        fine_expert_hidden_dim=16,
        shared_expert_hidden_dim=16,
        mtp_depth=1,
        heartbeat_enabled=True,
        heartbeat_memory_capacity=8,
    )
    model = DarwinXModel(config)
    legacy_state = {
        key: value
        for key, value in model.state_dict().items()
        if ".moe.neuroendocrine." not in key
        and not key.endswith(".moe._expert_usage_buffer")
    }
    declared = backbone_identity(legacy_state, config)
    payload = {
        "version": 6,
        "config": dict(config.__dict__),
        "base_checkpoint_id": declared,
        "training_state": {"cycle": 1, "step": 10},
    }

    report = validate_legacy_v6_resume(payload, legacy_state, model, config)

    assert report["legacy_identity_recomputed"] is True
    assert report["missing_initialized_buffers"]

    broken_state = dict(legacy_state)
    broken_state.pop("token_embedding.weight")
    with pytest.raises(ValueError, match="anatomy is incompatible"):
        validate_legacy_v6_resume(payload, broken_state, model, config)


def test_v6_adamw_migration_preserves_old_momenta_and_adds_only_baseline() -> None:
    class LegacyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Parameter(torch.ones(2))
            self.second = nn.Parameter(torch.ones(3))

    class CurrentModel(LegacyModel):
        def __init__(self) -> None:
            super().__init__()
            self.block = nn.Module()
            self.block.moe = nn.Module()
            self.block.moe.neuroendocrine = nn.Module()
            self.block.moe.neuroendocrine.baseline_dopamine = nn.Parameter(
                torch.zeros(1)
            )

    legacy = LegacyModel()
    legacy_optimizer = torch.optim.AdamW(legacy.parameters(), lr=1e-3)
    sum(parameter.sum() for parameter in legacy.parameters()).backward()
    legacy_optimizer.step()
    saved = legacy_optimizer.state_dict()

    current = CurrentModel()
    current_optimizer = torch.optim.AdamW(current.parameters(), lr=1e-3)
    inserted = migrate_adamw_state_with_new_baselines(
        saved, current, current_optimizer
    )

    assert inserted == ["block.moe.neuroendocrine.baseline_dopamine"]
    assert len(current_optimizer.state) == 2
    baseline = current.block.moe.neuroendocrine.baseline_dopamine
    assert baseline not in current_optimizer.state


def test_remap_optimizer_state_preserves_unchanged_drops_pruned_inits_new() -> None:
    """R1: topology change keeps AdamW moments for surviving parameters,
    drops pruned ones, and leaves newborn ones for lazy initialisation."""

    class OldModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.keep = nn.Linear(2, 2)
            self.prune = nn.Linear(2, 2)

    class NewModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.keep = nn.Linear(2, 2)
            self.born = nn.Linear(2, 2)

    old_model = OldModel()
    old_opt = torch.optim.AdamW(old_model.parameters(), lr=1e-3)
    # Populate AdamW moments with a real step.
    (old_model.keep(torch.randn(1, 2)).sum() + old_model.prune(torch.randn(1, 2)).sum()).backward()
    old_opt.step()
    old_named = dict(old_model.named_parameters())
    keep_weight_old = old_model.keep.weight
    preserved_moment = old_opt.state[keep_weight_old]["exp_avg"].clone()

    new_model = NewModel()
    new_model.keep.load_state_dict(old_model.keep.state_dict())  # same name + shape
    new_opt = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    new_named = dict(new_model.named_parameters())

    stats = remap_optimizer_state(old_opt, old_named, new_opt, new_named)

    assert stats["preserved"] == 2          # keep.weight + keep.bias
    assert stats["dropped"] == 2            # prune.weight + prune.bias
    assert stats["initialised"] == 2        # born.weight + born.bias

    new_keep_weight = new_model.keep.weight
    assert new_keep_weight in new_opt.state
    assert torch.equal(new_opt.state[new_keep_weight]["exp_avg"], preserved_moment)
    # Newborn parameter is left for lazy init on the first step.
    assert new_model.born.weight not in new_opt.state


def test_apply_lr_schedule_linear_warmup_then_plateau() -> None:
    """R2: LR ramps linearly to cfg.learning_rate over warmup_steps, then flat."""
    org = DarwinOrganism.__new__(DarwinOrganism)
    org.cfg = SimpleNamespace(learning_rate=1e-3, warmup_steps=10)
    org.optimizer = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=1e-3)

    org._apply_lr_schedule(0)
    assert org.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3 * 1 / 10)
    org._apply_lr_schedule(9)
    assert org.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    org._apply_lr_schedule(500)
    assert org.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_apply_lr_schedule_noop_when_warmup_disabled() -> None:
    """R2: with warmup_steps <= 0 the schedule must not touch the LR."""
    org = DarwinOrganism.__new__(DarwinOrganism)
    org.cfg = SimpleNamespace(learning_rate=1e-3, warmup_steps=0)
    org.optimizer = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=7e-4)

    org._apply_lr_schedule(0)
    assert org.optimizer.param_groups[0]["lr"] == pytest.approx(7e-4)
