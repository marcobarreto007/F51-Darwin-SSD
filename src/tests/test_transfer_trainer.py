from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from f51_darwin.transfer.student import DarwinTransferStudent
from f51_darwin.transfer.trainer import TransferTrainer


def models():
    torch.manual_seed(4)
    teacher = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=101,
            n_positions=32,
            n_ctx=32,
            n_embd=32,
            n_layer=2,
            n_head=4,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    ).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher, DarwinTransferStudent.from_teacher(
        teacher, replaced_layers=(1,)
    )


def test_all_three_training_stages_are_finite(tmp_path: Path) -> None:
    teacher, student = models()
    trainer = TransferTrainer(
        teacher,
        student,
        learning_rate=1e-3,
        metrics_path=tmp_path / "metrics.jsonl",
    )
    batch = torch.randint(0, 101, (2, 12))
    orientation = trainer.train_step("orientation", batch)
    alignment = trainer.train_step("alignment", batch)
    distillation = trainer.train_step("distillation", batch)
    assert all(
        torch.isfinite(torch.tensor(value))
        for value in (orientation, alignment, distillation)
    )
    lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3


def test_holdout_is_locked_tail() -> None:
    from f51_darwin.transfer.data import split_locked_holdout

    chunks = [torch.tensor([index]) for index in range(10)]
    train, holdout = split_locked_holdout(chunks, holdout_chunks=2)
    assert [int(item) for chunk in train for item in chunk] == list(range(8))
    assert [int(item) for chunk in holdout for item in chunk] == [8, 9]


def test_promotion_holdout_is_disjoint_and_partitioned() -> None:
    from f51_darwin.transfer.data import (
        evaluation_partitions,
        split_transfer_holdouts,
    )

    chunks = [torch.arange(index, index + 5) for index in range(20)]
    train, monitor, promotion = split_transfer_holdouts(
        chunks,
        sequence_length=4,
        monitor_chunks=3,
        promotion_tokens=32,
    )
    assert len(train) == 9
    assert len(monitor) == 3
    assert len(promotion) == 8
    partitions = evaluation_partitions(promotion, batches=4)
    assert [len(partition) for partition in partitions] == [2, 2, 2, 2]
    assert {id(chunk) for chunk in train}.isdisjoint(
        {id(chunk) for chunk in monitor + promotion}
    )


def test_token_cache_builds_and_reuses_identity(tmp_path: Path) -> None:
    from f51_darwin.transfer.data import load_or_build_token_cache

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("abcdef" * 20, encoding="utf-8")

    class Tokenizer:
        def __call__(self, text, *, add_special_tokens):
            assert not add_special_tokens
            return {"input_ids": [ord(char) % 17 for char in text]}

    cache = tmp_path / "chunks.pt"
    first, first_hit = load_or_build_token_cache(
        Tokenizer(),
        corpus,
        max_bytes=100,
        sequence_length=4,
        cache_path=cache,
        tokenizer_identity="tokenizer-v1",
    )
    second, second_hit = load_or_build_token_cache(
        Tokenizer(),
        corpus,
        max_bytes=100,
        sequence_length=4,
        cache_path=cache,
        tokenizer_identity="tokenizer-v1",
    )
    assert not first_hit
    assert second_hit
    assert first.dtype == torch.int32
    assert torch.equal(first, second)


def test_optimizer_state_persists_between_steps(tmp_path: Path) -> None:
    teacher, student = models()
    trainer = TransferTrainer(
        teacher,
        student,
        learning_rate=1e-3,
        metrics_path=tmp_path / "metrics.jsonl",
    )
    batch = torch.randint(0, 101, (2, 12))
    trainer.train_step("orientation", batch)
    optimizer_id = id(trainer.optimizer)
    trainer.train_step("orientation", batch)
    assert id(trainer.optimizer) == optimizer_id
    assert trainer.optimizer is not None
    assert trainer.optimizer.state
