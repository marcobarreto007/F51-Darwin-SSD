from __future__ import annotations

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from f51_darwin.gaba_inhibition import GABAConfig, GABAergicLayer
from f51_darwin.jepa_v2 import JEPAHeadV2
from f51_darwin.transplant.lifecycle import (
    OrganState,
    audit_gradients,
    configure_trainable_state,
    configure_trainable_states,
    transition_organ,
)
from f51_darwin.transplant.recipient import TwoDonorRecipient
from f51_darwin.transplant.slots import GABAResidualSlot, JEPAAuxiliarySlot


def _recipient() -> TwoDonorRecipient:
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


def test_adapter_active_exposes_only_one_organs_adapters() -> None:
    recipient = _recipient()

    declared = configure_trainable_state(
        recipient,
        organ_name="gaba.0",
        state=OrganState.ADAPTER_ACTIVE,
    )

    assert declared
    assert all(
        parameter.requires_grad == (name in declared)
        for name, parameter in recipient.named_parameters()
    )
    assert all(
        name.startswith("gaba_slots.gaba_0.adapter.")
        or name == "gaba_slots.gaba_0.external_gate"
        for name in declared
    )


def test_organ_unfrozen_does_not_unfreeze_gpt_or_other_organs() -> None:
    recipient = _recipient()

    declared = configure_trainable_state(
        recipient,
        organ_name="jepa",
        state=OrganState.ORGAN_UNFROZEN,
    )

    assert any(name.startswith("jepa_slot.organ.") for name in declared)
    assert not any(name.startswith("language_model.") for name in declared)
    assert not any("gaba_slots" in name for name in declared)


def test_multiple_organs_can_train_adapters_without_unfreezing_donors() -> None:
    recipient = _recipient()

    declared = configure_trainable_states(
        recipient,
        organ_states={
            "gaba.0": OrganState.ADAPTER_ACTIVE,
            "jepa": OrganState.ADAPTER_ACTIVE,
        },
    )

    assert any(name.startswith("gaba_slots.gaba_0.adapter.") for name in declared)
    assert "gaba_slots.gaba_0.external_gate" in declared
    assert any(name.startswith("jepa_slot.adapter.") for name in declared)
    assert not any(name.startswith("language_model.") for name in declared)
    assert not any(".organ." in name for name in declared)


def test_nontraining_states_freeze_every_parameter() -> None:
    for state in (
        OrganState.CANDIDATE,
        OrganState.SHADOW,
        OrganState.ACTIVE,
        OrganState.FROZEN,
        OrganState.QUARANTINE,
    ):
        recipient = _recipient()
        declared = configure_trainable_state(
            recipient,
            organ_name="jepa",
            state=state,
        )
        assert declared == frozenset()
        assert not any(
            parameter.requires_grad for parameter in recipient.parameters()
        )


def test_gradient_audit_rejects_undeclared_gradient() -> None:
    recipient = _recipient()
    declared = configure_trainable_state(
        recipient,
        organ_name="gaba.0",
        state=OrganState.ADAPTER_ACTIVE,
    )
    _, forbidden = next(
        (name, parameter)
        for name, parameter in recipient.named_parameters()
        if name.startswith("language_model.")
    )
    forbidden.grad = torch.ones_like(forbidden)

    with pytest.raises(RuntimeError, match="undeclared gradient"):
        audit_gradients(recipient, declared)


def test_gradient_audit_rejects_nonfinite_declared_gradient() -> None:
    recipient = _recipient()
    declared = configure_trainable_state(
        recipient,
        organ_name="gaba.0",
        state=OrganState.ADAPTER_ACTIVE,
    )
    name, parameter = next(
        (name, parameter)
        for name, parameter in recipient.named_parameters()
        if name in declared
    )
    parameter.grad = torch.full_like(parameter, float("nan"))

    with pytest.raises(FloatingPointError, match=name.replace(".", r"\.")):
        audit_gradients(recipient, declared)


def test_lifecycle_rejects_skipping_shadow_gate() -> None:
    assert (
        transition_organ(OrganState.CANDIDATE, OrganState.SHADOW)
        is OrganState.SHADOW
    )
    with pytest.raises(ValueError, match="invalid organ transition"):
        transition_organ(OrganState.CANDIDATE, OrganState.ORGAN_UNFROZEN)
