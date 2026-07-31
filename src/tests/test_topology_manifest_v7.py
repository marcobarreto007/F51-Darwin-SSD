from __future__ import annotations

import torch

from f51_darwin.darwin_x import DarwinXConfig, DarwinXModel


def small_config() -> DarwinXConfig:
    return DarwinXConfig(
        model_name="F51-Darwin-X-DNA-Test",
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


def train_step(model: DarwinXModel, optimizer: torch.optim.Optimizer) -> float:
    batch = torch.randint(0, model.config.vocab_size, (1, 6))
    output = model(batch, labels=batch)
    optimizer.zero_grad(set_to_none=True)
    assert output.loss is not None
    output.loss.backward()
    optimizer.step()
    return float(output.loss.detach())


def test_c01_birth_has_stable_new_identity_and_parent() -> None:
    model = DarwinXModel(small_config())
    moe = model.blocks[0].moe
    before = list(moe._expert_ids)
    new_index = moe._create_expert()

    assert new_index == len(before)
    assert moe._expert_ids[: len(before)] == before
    assert moe._expert_ids[new_index] not in before
    assert moe._expert_parent_ids[new_index] in before


def test_c02_growth_preserves_module_identity_uuid_and_weights() -> None:
    model = DarwinXModel(small_config())
    moe = model.blocks[0].moe
    expert = moe.fine_experts[0]
    expert_id = moe._expert_ids[0]
    old_hidden = expert.gate_proj.out_features
    old_weight = expert.gate_proj.weight.detach().clone()

    moe._expand_expert(0)

    assert moe.fine_experts[0] is expert
    assert moe._expert_ids[0] == expert_id
    assert expert.gate_proj.out_features > old_hidden
    torch.testing.assert_close(expert.gate_proj.weight[:old_hidden], old_weight)


def test_c03_apoptosis_compacts_state_without_renaming_survivors() -> None:
    model = DarwinXModel(small_config())
    moe = model.blocks[0].moe
    before = list(moe._expert_ids)

    assert moe._apoptosis(1) is True

    assert moe._expert_ids == [before[0], before[2], before[3]]
    assert moe.neuroendocrine.num_experts == 3
    assert moe.fine_router.router.out_features == 3
    assert moe.fine_router.expert_miss_count.numel() == 3
    assert moe._expert_usage_buffer.numel() == 3


def test_c04_manifest_v7_records_router_order_and_lineage() -> None:
    model = DarwinXModel(small_config())
    moe = model.blocks[0].moe
    new_index = moe._create_expert()
    manifest = model.topology_manifest()
    layer = manifest["topology"][0]

    assert manifest["version"] == 7
    assert layer["router_order"] == moe._expert_ids
    assert layer["experts"][new_index]["uuid"] == moe._expert_ids[new_index]
    assert layer["experts"][new_index]["parent_uuid"] in moe._expert_ids
    assert len(set(layer["router_order"])) == layer["num_experts"]


def test_c05_restore_rebuilds_anatomy_ids_and_hormones() -> None:
    source = DarwinXModel(small_config())
    source_moe = source.blocks[0].moe
    source_moe._create_expert()
    source_moe._expand_expert(0)
    source_moe.neuroendocrine.dopamine.copy_(
        torch.linspace(0.1, 0.5, len(source_moe.fine_experts))
    )
    source_moe.neuroendocrine.cortisol.fill_(0.25)
    manifest = source.topology_manifest()

    restored = DarwinXModel(small_config())
    repairs = DarwinXModel.restore_topology(manifest, restored)
    restored_moe = restored.blocks[0].moe

    assert repairs == 2
    assert restored_moe._expert_ids == source_moe._expert_ids
    assert [e.gate_proj.out_features for e in restored_moe.fine_experts] == [
        e.gate_proj.out_features for e in source_moe.fine_experts
    ]
    torch.testing.assert_close(
        restored_moe.neuroendocrine.dopamine,
        source_moe.neuroendocrine.dopamine,
    )
    assert restored_moe.neuroendocrine.cortisol.item() == 0.25


def test_c06_ghost_predator_rewards_retention_and_punishes_forgetting() -> None:
    model = DarwinXModel(small_config())
    system = model.blocks[0].moe.neuroendocrine
    count = system.num_experts
    gradients = torch.full((count,), 0.5)
    usage = torch.full((count,), 1.0 / count)
    dopamine_before = system.dopamine.clone()
    cortisol_before = float(system.cortisol)

    system.update_from_local_signals(gradients, usage, 1.0, 0.3, ghost_loss=3.0)
    dopamine_after_retention = system.dopamine.clone()
    system.update_from_local_signals(gradients, usage, 1.0, 0.3, ghost_loss=20.0)

    assert torch.any(dopamine_after_retention > dopamine_before)
    assert float(system.cortisol) > cortisol_before


def test_c07_checkpoint_roundtrip_is_functionally_equivalent(tmp_path) -> None:
    torch.manual_seed(51)
    source = DarwinXModel(small_config()).eval()
    source.blocks[0].moe._create_expert()
    source.blocks[0].moe._expand_expert(0)
    batch = torch.randint(0, source.config.vocab_size, (1, 6))
    expected = source(batch).logits.detach()
    path = tmp_path / "v7.pt"
    torch.save(
        {
            "version": 7,
            "topology_manifest": source.topology_manifest(),
            "model_state_dict": source.state_dict(),
        },
        path,
    )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    restored = DarwinXModel(small_config()).eval()
    DarwinXModel.restore_topology(payload["topology_manifest"], restored)
    restored.load_state_dict(payload["model_state_dict"], strict=True)
    actual = restored(batch).logits.detach()

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_c08_autonomic_actions_are_deferred_then_executed() -> None:
    model = DarwinXModel(small_config())
    moe = model.blocks[0].moe
    before = len(moe.fine_experts)
    moe._pending_autonomic_actions = {
        "neurogenesis": True,
        "prune_experts": [],
        "expand_experts": [],
        "sleep_phase": False,
    }

    proposal = moe.apply_pending_autonomic_actions()
    assert proposal["structural"]["neurogenesis"] is True
    assert len(moe.fine_experts) == before
    executed = moe.execute_structural_actions()

    assert executed["neurogenesis"] == 1
    assert len(moe.fine_experts) == before + 1


def test_c09_capability_gap_threshold_has_positive_and_negative_controls() -> None:
    model = DarwinXModel(small_config())
    system = model.blocks[0].moe.neuroendocrine

    system.bdnf.zero_()
    system.norepinephrine.zero_()
    assert system.should_expand_capacity() == []
    assert system.should_trigger_neurogenesis() is False

    system.bdnf[2] = 0.75
    system.norepinephrine.fill_(0.2)
    assert system.should_expand_capacity() == [2]
    assert system.should_trigger_neurogenesis() is True


def test_c10_optimizer_resumes_after_structural_checkpoint(tmp_path) -> None:
    torch.manual_seed(51)
    source = DarwinXModel(small_config())
    source.blocks[0].moe._create_expert()
    source.blocks[0].moe._expand_expert(0)
    optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
    first_loss = train_step(source, optimizer)
    path = tmp_path / "optimizer_v7.pt"
    torch.save(
        {
            "version": 7,
            "topology_manifest": source.topology_manifest(),
            "model_state_dict": source.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    restored = DarwinXModel(small_config())
    DarwinXModel.restore_topology(payload["topology_manifest"], restored)
    restored.load_state_dict(payload["model_state_dict"], strict=True)
    resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(payload["optimizer_state_dict"])
    loaded_state_count = len(resumed_optimizer.state)
    steps_before = [
        int(state["step"].item())
        for state in resumed_optimizer.state.values()
        if "step" in state
    ]
    second_loss = train_step(restored, resumed_optimizer)
    steps_after = [
        int(state["step"].item())
        for state in resumed_optimizer.state.values()
        if "step" in state
    ]

    assert torch.isfinite(torch.tensor(first_loss))
    assert torch.isfinite(torch.tensor(second_loss))
    assert steps_before and min(steps_before) == 1
    assert loaded_state_count == len(optimizer.state)
    assert max(steps_after) == 2
