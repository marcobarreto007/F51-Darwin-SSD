from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn

from f51_darwin.state_identity import backbone_identity, tokenizer_identity


class _Core(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(8, 4)
        self.blocks = nn.ModuleList([nn.Linear(4, 4)])
        self.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.heartbeat = nn.Linear(4, 4)


def _tokenizer() -> SimpleNamespace:
    return SimpleNamespace(
        vocab={"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "a": 4},
        merges=[("a", "b")],
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )


def test_backbone_identity_ignores_heartbeat_but_tracks_core_weights() -> None:
    torch.manual_seed(51)
    model = _Core()
    config = {"d_model": 4, "n_layers": 1}
    original = model.state_dict()
    identity = backbone_identity(original, config)

    heartbeat_changed = deepcopy(original)
    heartbeat_changed["heartbeat.weight"].add_(1)
    assert backbone_identity(heartbeat_changed, config) == identity

    core_changed = deepcopy(original)
    core_changed["blocks.0.weight"].add_(1)
    assert backbone_identity(core_changed, config) != identity


def test_backbone_identity_supports_scalar_bfloat16_core_buffers() -> None:
    state = {
        "blocks.0.scalar_state": torch.tensor(0.25, dtype=torch.bfloat16),
        "blocks.0.weight": torch.ones(2, 2, dtype=torch.bfloat16),
    }

    identity = backbone_identity(state, {"d_model": 2, "n_layers": 1})

    assert identity.startswith("darwin-model-core-v1:")


def test_backbone_identity_ignores_training_only_loss_semantics_version() -> None:
    torch.manual_seed(51)
    state = _Core().state_dict()
    base_config = {"d_model": 4, "n_layers": 1}

    without_version = backbone_identity(state, base_config)
    legacy_v1 = backbone_identity(
        state, {**base_config, "loss_semantics_version": 1}
    )
    causal_v2 = backbone_identity(
        state, {**base_config, "loss_semantics_version": 2}
    )

    assert legacy_v1 == without_version
    assert causal_v2 == without_version


def test_tokenizer_identity_tracks_vocab_and_ordered_merges() -> None:
    tokenizer = _tokenizer()
    identity = tokenizer_identity(tokenizer)

    tokenizer.merges.append(("ab", "c"))
    assert tokenizer_identity(tokenizer) != identity

    tokenizer = _tokenizer()
    tokenizer.vocab["a"] = 5
    assert tokenizer_identity(tokenizer) != identity
