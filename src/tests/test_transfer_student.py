from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from f51_darwin.transfer.checkpoint import (
    inherit_matching_weights,
    load_transfer_checkpoint,
    save_transfer_checkpoint,
)
from f51_darwin.transfer.student import DarwinTransferStudent, SSDGPT2Attention


def tiny_teacher() -> GPT2LMHeadModel:
    torch.manual_seed(7)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=97,
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


def test_replaces_only_requested_attention_and_runs_forward() -> None:
    teacher = tiny_teacher()
    student = DarwinTransferStudent.from_teacher(teacher, replaced_layers=(1,))
    assert not isinstance(student.model.transformer.h[0].attn, SSDGPT2Attention)
    assert isinstance(student.model.transformer.h[1].attn, SSDGPT2Attention)
    assert torch.equal(
        teacher.transformer.wte.weight,
        student.model.transformer.wte.weight,
    )
    output = student(input_ids=torch.tensor([[1, 2, 3, 4]]))
    assert output.logits.shape == (1, 4, 97)
    assert torch.isfinite(output.logits).all()


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    teacher = tiny_teacher()
    student = DarwinTransferStudent.from_teacher(teacher, replaced_layers=(1,))
    path = tmp_path / "transfer.pt"
    metadata = save_transfer_checkpoint(
        student,
        path,
        donor_manifest_sha256="abc123",
        stage="orientation",
        step=11,
    )
    restored = DarwinTransferStudent.from_teacher(teacher, replaced_layers=(1,))
    loaded = load_transfer_checkpoint(
        restored,
        path,
        expected_donor_manifest_sha256="abc123",
    )
    assert loaded == metadata
    for left, right in zip(student.parameters(), restored.parameters(), strict=True):
        assert torch.equal(left, right)


def test_progressive_inheritance_reaches_zero_attention_layers(tmp_path: Path) -> None:
    teacher = tiny_teacher()
    first = DarwinTransferStudent.from_teacher(teacher, replaced_layers=(1,))
    with torch.no_grad():
        first.ssd_attention(1).mixer.decay_proj.bias.fill_(2.5)
    path = tmp_path / "first.pt"
    save_transfer_checkpoint(
        first,
        path,
        donor_manifest_sha256="donor",
        stage="distillation",
        step=3,
        generation=1,
        stage_step=1,
    )
    complete = DarwinTransferStudent.from_teacher(
        teacher, replaced_layers=(0, 1)
    )
    inherited = inherit_matching_weights(
        complete,
        path,
        expected_donor_manifest_sha256="donor",
    )
    assert inherited > 0
    assert complete.attention_layer_count == 0
    assert torch.equal(
        complete.ssd_attention(1).mixer.decay_proj.bias,
        torch.full((4,), 2.5),
    )
