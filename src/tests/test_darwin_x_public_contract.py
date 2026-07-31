from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import uuid

import torch

import f51_darwin.darwin_x as darwin_x
from f51_darwin.state_identity import backbone_identity


PUBLIC_NAMES = (
    "DarwinXBlock",
    "DarwinXConfig",
    "DarwinXModel",
    "DarwinXOutput",
    "DeepSeekStyleMoE",
    "ExpertFFN",
    "FineRouter",
    "GQACausalAttention",
    "NeuroendocrineSystem",
    "SSDMixerOnly",
    "estimate_darwin_x_parameters",
    "migrate_mutational_state_for_load",
)
EXPECTED_PARAMETER_CONTRACT = "c79ba66bb38937ffa815bc812789aa47efea7b99eab07a14aeb6621c9f5f52bc"
EXPECTED_STATE_CONTRACT = "6d1e6cc7eee5d9ed75b590fda13a86d2f0daf76a837519b6f1e3ca66b9a5e349"
# Updated for 6097176, which made topology_manifest() always emit a
# "cognition" entry (null when no cognitive runtime is attached). The manifest
# is otherwise byte-identical to the one this constant was minted against in
# b41a999, verified by diffing both manifests directly.
EXPECTED_TOPOLOGY_CONTRACT = "3301a28d09c1c055326846332afb23c8e16781e2214d8fa4612d558aadf7a9e8"


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _config() -> darwin_x.DarwinXConfig:
    return darwin_x.DarwinXConfig(
        model_name="F51-Darwin-X-1.6B-Nitro",
        vocab_size=32,
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
        nitro_enabled=False,
        nitro_gpu_expert_capacity=4,
    )


def _model(monkeypatch, *, seed: int = 51) -> darwin_x.DarwinXModel:
    sequence = itertools.count(1)
    monkeypatch.setattr(darwin_x.uuid, "uuid4", lambda: uuid.UUID(int=next(sequence)))
    torch.manual_seed(seed)
    return darwin_x.DarwinXModel(_config()).cpu().eval()


def test_public_import_and_config_round_trip_contract() -> None:
    assert tuple(name for name in PUBLIC_NAMES if hasattr(darwin_x, name)) == PUBLIC_NAMES
    config = _config()
    assert darwin_x.DarwinXConfig.from_mapping(dataclasses.asdict(config)) == config


def test_parameter_state_topology_and_optimizer_order_contract(monkeypatch) -> None:
    model = _model(monkeypatch)
    parameters = [
        (name, list(parameter.shape), str(parameter.dtype), parameter.requires_grad)
        for name, parameter in model.named_parameters()
    ]
    state = [
        (name, list(tensor.shape), str(tensor.dtype))
        for name, tensor in model.state_dict().items()
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer_order = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert optimizer_order == [id(parameter) for parameter in model.parameters()]
    assert model.lm_head.weight is model.token_embedding.weight
    assert _digest(parameters) == EXPECTED_PARAMETER_CONTRACT
    assert _digest(state) == EXPECTED_STATE_CONTRACT
    assert _digest(model.topology_manifest()) == EXPECTED_TOPOLOGY_CONTRACT


def test_strict_load_backbone_identity_and_deterministic_cpu_output(monkeypatch) -> None:
    source = _model(monkeypatch)
    state = {name: tensor.detach().clone() for name, tensor in source.state_dict().items()}
    expected_identity = backbone_identity(state, dataclasses.asdict(source.config))
    restored = _model(monkeypatch, seed=99)
    incompatible = restored.load_state_dict(state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert backbone_identity(restored.state_dict(), dataclasses.asdict(restored.config)) == expected_identity
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.inference_mode():
        source_output = source(input_ids).logits.detach().cpu()
        restored_output = restored(input_ids).logits.detach().cpu()
        repeated_output = restored(input_ids).logits.detach().cpu()
    assert restored_output.shape == (1, 4, source.config.vocab_size)
    assert torch.isfinite(restored_output).all()
    torch.testing.assert_close(restored_output, source_output, rtol=0.0, atol=0.0)
    torch.testing.assert_close(repeated_output, restored_output, rtol=0.0, atol=0.0)
