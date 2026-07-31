from __future__ import annotations

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from f51_darwin.gaba_inhibition import GABAConfig, GABAergicLayer
from f51_darwin.jepa_v2 import JEPAHeadV2
from f51_darwin.transplant.checkpoint import (
    load_recipient_checkpoint,
    rollback_organ,
    save_organ_snapshot,
    save_recipient_checkpoint,
)
from f51_darwin.transplant.recipient import TwoDonorRecipient
from f51_darwin.transplant.slots import GABAResidualSlot, JEPAAuxiliarySlot


def _recipient() -> TwoDonorRecipient:
    torch.manual_seed(71)
    language_model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=31,
            n_positions=16,
            n_ctx=16,
            n_embd=768,
            n_layer=1,
            n_head=12,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
    ).eval()
    jepa = JEPAHeadV2(
        d_model=512,
        hidden_dim=768,
        dropout=0.0,
        bottleneck_dim=256,
    ).eval()
    gaba = GABAergicLayer(GABAConfig(d_model=512)).eval()
    return TwoDonorRecipient.from_language_donor(
        language_model,
        jepa_slot=JEPAAuxiliarySlot(jepa),
        gaba_slots={0: (GABAResidualSlot(gaba),)},
    )


def test_checkpoint_rejects_wrong_language_or_organ_donor(tmp_path) -> None:
    recipient = _recipient()
    path = tmp_path / "recipient.pt"
    save_recipient_checkpoint(
        recipient,
        path,
        language_donor_sha256="a" * 64,
        organ_donor_sha256="b" * 64,
        parent_checkpoint_sha256=None,
        ledger_sha256="c" * 64,
        organ_states={"jepa": "candidate", "gaba.0": "candidate"},
    )

    with pytest.raises(ValueError, match="language donor identity mismatch"):
        load_recipient_checkpoint(
            recipient,
            path,
            expected_language_donor_sha256="d" * 64,
            expected_organ_donor_sha256="b" * 64,
        )
    with pytest.raises(ValueError, match="organ donor identity mismatch"):
        load_recipient_checkpoint(
            recipient,
            path,
            expected_language_donor_sha256="a" * 64,
            expected_organ_donor_sha256="d" * 64,
        )


def test_checkpoint_roundtrip_restores_recipient(tmp_path) -> None:
    recipient = _recipient()
    path = tmp_path / "recipient.pt"
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        recipient.organ_slot("gaba.0").external_gate.fill_(0.3)
        expected = recipient(
            input_ids=ids,
            organ_mode="adapter_active",
        ).logits.clone()
    saved = save_recipient_checkpoint(
        recipient,
        path,
        language_donor_sha256="a" * 64,
        organ_donor_sha256="b" * 64,
        parent_checkpoint_sha256=None,
        ledger_sha256="c" * 64,
        organ_states={"jepa": "frozen", "gaba.0": "active"},
        declared_optimizer_parameters={"gaba_slots.gaba_0.external_gate"},
    )
    with torch.no_grad():
        recipient.organ_slot("gaba.0").external_gate.zero_()

    metadata = load_recipient_checkpoint(
        recipient,
        saved.path,
        expected_language_donor_sha256="a" * 64,
        expected_organ_donor_sha256="b" * 64,
    )
    with torch.no_grad():
        actual = recipient(
            input_ids=ids,
            organ_mode="adapter_active",
        ).logits

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert metadata["organ_states"]["gaba.0"] == "active"
    assert metadata["declared_optimizer_parameters"] == [
        "gaba_slots.gaba_0.external_gate"
    ]


def test_per_organ_rollback_restores_logits_and_state(tmp_path) -> None:
    recipient = _recipient()
    ids = torch.tensor([[1, 2, 3, 4]])
    snapshot = save_organ_snapshot(
        recipient,
        "gaba.0",
        tmp_path / "gaba.pt",
    )
    with torch.no_grad():
        expected = recipient(
            input_ids=ids,
            organ_mode="adapter_active",
        ).logits.clone()
        recipient.organ_slot("gaba.0").external_gate.fill_(0.7)
    rollback_organ(recipient, snapshot)
    with torch.no_grad():
        actual = recipient(
            input_ids=ids,
            organ_mode="adapter_active",
        ).logits

    assert recipient.organ_slot("gaba.0").external_gate.item() == 0.0
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_tampered_snapshot_is_rejected_before_mutation(tmp_path) -> None:
    recipient = _recipient()
    snapshot = save_organ_snapshot(
        recipient,
        "gaba.0",
        tmp_path / "gaba.pt",
    )
    with snapshot.path.open("ab") as handle:
        handle.write(b"tamper")
    with torch.no_grad():
        recipient.organ_slot("gaba.0").external_gate.fill_(0.4)

    with pytest.raises(ValueError, match="snapshot hash mismatch"):
        rollback_organ(recipient, snapshot)

    assert recipient.organ_slot("gaba.0").external_gate.item() == pytest.approx(0.4)

