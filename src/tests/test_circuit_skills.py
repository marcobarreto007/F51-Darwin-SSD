from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from f51_darwin.circuits.skills import (
    SkillContract,
    SkillContractError,
    build_skill_batches,
    registered_builders,
    registered_token_pools,
    token_pool_identity,
)

CONFIGS = Path("src/configs/circuits/skills")


def _letters(index: int) -> str:
    """Nome puramente alfabético de 4 letras — `str.isalpha()` precisa passar."""
    value = index
    out = []
    for _ in range(4):
        out.append(chr(ord("a") + value % 26))
        value //= 26
    return "".join(reversed(out))


def _vocab(size: int = 64) -> dict[str, int]:
    """Vocabulário sintético: índices pares entram no pool, ímpares não."""
    vocab: dict[str, int] = {}
    for index in range(size):
        if index % 2 == 0:
            vocab[f"Ġ{_letters(index)}"] = index
        else:
            vocab[f"<sym{index}>"] = index
    return vocab


def _contract(**overrides) -> SkillContract:
    payload = {
        "schema_version": 1,
        "skill_id": "induction-repeat-v1",
        "builder": "induction_repeat_v1",
        "split": "discovery",
        "domain": "induction",
        "seed": 11,
        "items": 8,
        "batch_size": 4,
        "copy_length": 6,
        "token_pool": "gpt2_alpha_space_v1",
    }
    payload.update(overrides)
    return SkillContract.from_dict(payload)


def test_registries_are_closed() -> None:
    assert registered_builders() == (
        "induction_repeat_offset_v1",
        "induction_repeat_v1",
    )
    assert registered_token_pools() == ("gpt2_alpha_space_v1",)


def _offset_contract(**overrides) -> SkillContract:
    payload = _contract().to_dict()
    payload.update(
        {"builder": "induction_repeat_offset_v1", "skill_id": "induction-offset"}
    )
    payload.update(overrides)
    return SkillContract.from_dict(payload)


def test_offset_builder_repeats_a_block_at_a_varying_boundary() -> None:
    contract = _offset_contract(items=16, batch_size=16)
    batch = build_skill_batches(contract, _vocab())[0]
    ids, labels = batch["input_ids"], batch["labels"]
    length = contract.copy_length
    assert ids.shape == (16, 3 * length)

    boundaries = set()
    for row in range(ids.shape[0]):
        scored = labels[row].ne(-100).nonzero().flatten()
        assert scored.numel() == length - 1
        start = int(scored[0])
        offset = start - length
        boundaries.add(offset)
        assert 0 <= offset <= length
        # A cópia é exata nas duas metades.
        torch.testing.assert_close(
            ids[row, offset : offset + length],
            ids[row, offset + length : offset + 2 * length],
        )
        # labels[t] continua sendo o token seguinte.
        torch.testing.assert_close(
            labels[row, scored], ids[row, scored + 1]
        )
    # O ponto do experimento: a fronteira precisa variar entre itens, senão
    # o arm SHUFFLE (que permuta entre itens) vira intervenção vazia.
    assert len(boundaries) > 1


def test_offset_builder_is_deterministic() -> None:
    first = build_skill_batches(_offset_contract(), _vocab())
    second = build_skill_batches(_offset_contract(), _vocab())
    for left, right in zip(first, second):
        torch.testing.assert_close(left["input_ids"], right["input_ids"])
        torch.testing.assert_close(left["labels"], right["labels"])


def test_offset_splits_are_disjoint() -> None:
    discovery = build_skill_batches(_offset_contract(split="discovery"), _vocab())
    confirmation = build_skill_batches(
        _offset_contract(split="confirmation"), _vocab()
    )
    left = {tuple(r.tolist()) for b in discovery for r in b["input_ids"]}
    right = {tuple(r.tolist()) for b in confirmation for r in b["input_ids"]}
    assert left.isdisjoint(right)


def test_contract_rejects_unregistered_constructors() -> None:
    with pytest.raises(SkillContractError, match="builder"):
        _contract(builder="exec_arbitrary_code")
    with pytest.raises(SkillContractError, match="token pool"):
        _contract(token_pool="anything")


def test_contract_rejects_unknown_and_missing_fields() -> None:
    payload = _contract().to_dict()
    payload["surprise"] = 1
    with pytest.raises(SkillContractError, match="unknown fields"):
        SkillContract.from_dict(payload)
    payload.pop("surprise")
    payload.pop("seed")
    with pytest.raises(SkillContractError, match="missing fields"):
        SkillContract.from_dict(payload)


def test_contract_rejects_degenerate_values() -> None:
    with pytest.raises(SkillContractError):
        _contract(items=0)
    with pytest.raises(SkillContractError):
        _contract(copy_length=1)
    with pytest.raises(SkillContractError):
        _contract(split="train")
    with pytest.raises(SkillContractError):
        _contract(seed=True)


def test_identity_is_field_sensitive_and_order_independent() -> None:
    base = _contract()
    assert base.identity.startswith("f51-skill-v1:")
    assert base.identity == SkillContract.from_dict(base.to_dict()).identity
    assert base.identity != _contract(split="confirmation").identity
    assert base.identity != _contract(seed=12).identity


def test_batches_carry_the_shapes_the_ablation_runtime_expects() -> None:
    contract = _contract()
    batches = build_skill_batches(contract, _vocab())
    assert len(batches) == 2
    for batch in batches:
        ids, labels = batch["input_ids"], batch["labels"]
        assert ids.shape == (4, 12)
        assert labels.shape == ids.shape
        assert batch["domains"] == "induction"
        # A segunda cópia repete a primeira.
        assert torch.equal(ids[:, :6], ids[:, 6:])


def test_scored_positions_exclude_the_unpredictable_first_token() -> None:
    contract = _contract()
    batch = build_skill_batches(contract, _vocab())[0]
    ids, labels = batch["input_ids"], batch["labels"]
    scored = labels.ne(-100)
    # copy_length=6 -> pontua t = 6..10, ou seja 5 posições por item.
    assert scored.sum(dim=1).tolist() == [5] * 4
    assert scored[:, :6].sum() == 0
    assert scored[:, 11:].sum() == 0
    # labels[:, t] deve ser o token seguinte de input_ids.
    torch.testing.assert_close(labels[:, 6:11], ids[:, 7:12])


def test_discovery_and_confirmation_never_share_an_item() -> None:
    """run_paired_ablation recusa reuso; aqui a separação é estrutural."""
    discovery = build_skill_batches(_contract(split="discovery"), _vocab())
    confirmation = build_skill_batches(_contract(split="confirmation"), _vocab())
    left = {tuple(row.tolist()) for batch in discovery for row in batch["input_ids"]}
    right = {
        tuple(row.tolist()) for batch in confirmation for row in batch["input_ids"]
    }
    assert left and right
    assert left.isdisjoint(right)


def test_builder_is_deterministic() -> None:
    first = build_skill_batches(_contract(), _vocab())
    second = build_skill_batches(_contract(), _vocab())
    for left, right in zip(first, second):
        torch.testing.assert_close(left["input_ids"], right["input_ids"])


def test_pool_excludes_non_word_tokens() -> None:
    contract = _contract()
    batches = build_skill_batches(contract, _vocab())
    used = {int(value) for batch in batches for value in batch["input_ids"].flatten()}
    assert used
    assert all(value % 2 == 0 for value in used)


def test_ragged_final_batch_is_allowed() -> None:
    batches = build_skill_batches(_contract(items=10, batch_size=4), _vocab())
    assert [batch["input_ids"].shape[0] for batch in batches] == [4, 4, 2]


def test_pool_identity_tracks_the_vocabulary() -> None:
    left = token_pool_identity("gpt2_alpha_space_v1", _vocab())
    assert left == token_pool_identity("gpt2_alpha_space_v1", _vocab())
    assert left != token_pool_identity("gpt2_alpha_space_v1", _vocab(size=32))
    with pytest.raises(SkillContractError):
        token_pool_identity("nope", _vocab())


def test_build_rejects_an_empty_vocabulary() -> None:
    with pytest.raises(SkillContractError):
        build_skill_batches(_contract(), {})


@pytest.mark.parametrize(
    "name", ["induction-discovery.json", "induction-confirmation.json"]
)
def test_shipped_contracts_load_and_satisfy_the_causal_gate_minimum(name: str) -> None:
    payload = json.loads((CONFIGS / name).read_text(encoding="utf-8"))
    contract = SkillContract.from_dict(payload)
    assert contract.skill_id == "induction-repeat-v1"
    # CausalGateConfig.minimum_items_per_domain = 200
    assert contract.items >= 200


def test_shipped_splits_are_disjoint() -> None:
    discovery = SkillContract.from_dict(
        json.loads((CONFIGS / "induction-discovery.json").read_text(encoding="utf-8"))
    )
    confirmation = SkillContract.from_dict(
        json.loads(
            (CONFIGS / "induction-confirmation.json").read_text(encoding="utf-8")
        )
    )
    assert discovery.identity != confirmation.identity
    assert discovery.stream_seed() != confirmation.stream_seed()
