from pathlib import Path

import numpy as np
import pytest
import torch

from f51_darwin.config import DarwinConfig
from f51_darwin.data import (
    CausalLMDataLoader,
    discover_corpus_files,
    load_text_documents,
    resolve_corpus_dir,
    tokenize_documents,
)
from f51_darwin.tokenizer import F51BPETokenizer, MIN_BPE_VOCAB_SIZE


def test_resolve_fixture_corpus() -> None:
    root = Path(__file__).resolve().parents[2]
    corpus_dir = resolve_corpus_dir(root, "src/tests/fixtures/corpus")
    files = discover_corpus_files(corpus_dir)
    assert files
    documents = load_text_documents(corpus_dir)
    assert documents


def test_causal_loader_batch_shape() -> None:
    root = Path(__file__).resolve().parents[2]
    corpus_dir = root / "src" / "tests" / "fixtures" / "corpus"
    tokenizer = F51BPETokenizer.train(load_text_documents(corpus_dir), vocab_size=MIN_BPE_VOCAB_SIZE)
    token_ids = tokenize_documents(load_text_documents(corpus_dir), tokenizer)
    loader = CausalLMDataLoader(token_ids, block_size=16, batch_size=2, seed=51)
    batch = loader.next_batch()
    assert batch.shape == (2, 16)
    assert batch.dtype == torch.long


def test_causal_loader_keeps_read_only_memmap_on_disk(tmp_path: Path) -> None:
    token_path = tmp_path / "tokens.bin"
    writable = np.memmap(token_path, dtype=np.int32, mode="w+", shape=(256,))
    writable[:] = np.arange(256, dtype=np.int32)
    writable.flush()
    del writable

    token_ids = np.memmap(token_path, dtype=np.int32, mode="r")
    loader = CausalLMDataLoader(token_ids, block_size=16, batch_size=4, seed=51)

    assert isinstance(loader.token_ids, np.memmap)
    assert loader.token_ids.mode == "r"

    batch = loader.next_batch()
    eval_batch = loader.take_eval_batch()

    assert batch.shape == (4, 16)
    assert eval_batch.shape == (4, 16)
    assert batch.dtype == torch.long
    assert eval_batch.dtype == torch.long


def test_causal_loader_default_preserves_sequential_v1_positions() -> None:
    token_ids = np.arange(256, dtype=np.int32)
    loader = CausalLMDataLoader(token_ids, block_size=16, batch_size=4, seed=51)

    first = loader.next_batch()
    second = loader.next_batch()

    assert first[:, 0].tolist() == [51, 74, 97, 120]
    assert second[:, 0].tolist() == [67, 90, 113, 136]
    assert loader.sampler_identity == {
        "name": "f51_causal_lm_sampler",
        "mode": "sequential",
        "version": 1,
        "seed": 51,
        "block_size": 16,
        "batch_size": 4,
        "token_count": 256,
        "block_count": 16,
    }


def test_permuted_blocks_v1_is_bijective_before_repeating() -> None:
    block_size = 8
    block_count = 22
    token_ids = np.arange(block_size * block_count + 3, dtype=np.int32)
    loader = CausalLMDataLoader(
        token_ids,
        block_size=block_size,
        batch_size=2,
        seed=51,
        sampler_mode="permuted_blocks",
        sampler_version=1,
    )

    starts: list[int] = []
    for _ in range(block_count // loader.batch_size):
        starts.extend(loader.next_batch()[:, 0].tolist())

    expected_blocks = [index * block_size for index in range(block_count)]
    assert sorted(starts) == expected_blocks
    assert starts != expected_blocks
    assert starts != list(reversed(expected_blocks))


def test_permuted_blocks_v1_resumes_from_step_without_rng_state() -> None:
    token_ids = np.arange(1024, dtype=np.int32)
    direct = CausalLMDataLoader(
        token_ids,
        block_size=16,
        batch_size=4,
        seed=123,
        sampler_mode="permuted_blocks",
    )
    replayed = CausalLMDataLoader(
        token_ids,
        block_size=16,
        batch_size=4,
        seed=123,
        sampler_mode="permuted_blocks",
    )
    torch_rng_before = torch.random.get_rng_state()

    direct.set_step(7)
    direct_batch = direct.next_batch()
    for _ in range(7):
        replayed.next_batch()
    replayed_batch = replayed.next_batch()

    assert torch.equal(direct_batch, replayed_batch)
    assert torch.equal(torch_rng_before, torch.random.get_rng_state())
    assert direct.step == replayed.step == 8


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sampler_mode": "random"}, "unsupported sampler_mode"),
        ({"sampler_version": 2}, "unsupported sampler_version"),
        ({"seed": "51"}, "seed must be an integer"),
    ],
)
def test_causal_loader_rejects_invalid_sampler_parameters(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CausalLMDataLoader(
            np.arange(256, dtype=np.int32),
            block_size=16,
            batch_size=4,
            **kwargs,
        )


def test_permuted_blocks_rejects_unavoidable_in_batch_collisions() -> None:
    with pytest.raises(ValueError, match="at least batch_size full blocks"):
        CausalLMDataLoader(
            np.arange(48, dtype=np.int32),
            block_size=16,
            batch_size=4,
            sampler_mode="permuted_blocks",
        )
