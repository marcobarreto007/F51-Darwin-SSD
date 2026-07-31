from __future__ import annotations

import torch
import pytest
from transformers import GPT2Config, GPT2LMHeadModel

from f51_darwin.gaba_inhibition import GABAConfig, GABAergicLayer
from f51_darwin.heartbeat import Heartbeat, HeartbeatConfig
from f51_darwin.inter_hemispheric import HemisphereConfig, InterHemisphericSystem
from f51_darwin.jepa_v2 import JEPAHeadV2
from f51_darwin.spider_sense import SpiderSense
from f51_darwin.transplant.organs import HeartbeatOrgan
from f51_darwin.transplant.recipient import (
    GUARDED_TRANSPLANT_POLICY,
    TwoDonorRecipient,
)
from f51_darwin.transplant.slots import (
    GABAResidualSlot,
    HeartbeatSlot,
    IHSResidualSlot,
    JEPAAuxiliarySlot,
    SpiderSenseSlot,
    TTMResidualSlot,
)


def _frozen_jepa() -> JEPAHeadV2:
    torch.manual_seed(53)
    module = JEPAHeadV2(
        d_model=512,
        hidden_dim=768,
        dropout=0.0,
        bottleneck_dim=256,
    )
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module.eval()


def _frozen_gaba() -> GABAergicLayer:
    torch.manual_seed(54)
    module = GABAergicLayer(GABAConfig(d_model=512))
    with torch.no_grad():
        module.residual_gate.fill_(0.5)
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module.eval()


def test_gaba_slot_is_exact_noop_at_zero_external_gate() -> None:
    slot = GABAResidualSlot(_frozen_gaba())
    hidden = torch.randn(2, 5, 768)

    output = slot(hidden, enabled=True, mutate_state=False)

    assert torch.equal(output.hidden, hidden)
    assert output.observation is not None
    assert output.auxiliary_loss is None
    assert slot.external_gate.item() == 0.0


def test_gaba_slot_nonzero_gate_produces_finite_bounded_residual() -> None:
    slot = GABAResidualSlot(_frozen_gaba())
    with torch.no_grad():
        slot.external_gate.fill_(0.5)
    hidden = torch.randn(2, 5, 768)

    output = slot(hidden, enabled=True, mutate_state=False)

    assert output.hidden.shape == hidden.shape
    assert torch.isfinite(output.hidden).all()
    assert not torch.equal(output.hidden, hidden)


def test_jepa_slot_preserves_input_and_returns_finite_loss() -> None:
    slot = JEPAAuxiliarySlot(_frozen_jepa())
    hidden = torch.randn(2, 6, 768)

    output = slot(hidden, enabled=True)

    assert torch.equal(output.hidden, hidden)
    assert output.auxiliary_loss is not None
    assert output.auxiliary_loss.ndim == 0
    assert torch.isfinite(output.auxiliary_loss)
    assert output.observation is None


def test_disabled_jepa_executes_but_contributes_zero_loss() -> None:
    slot = JEPAAuxiliarySlot(_frozen_jepa())
    hidden = torch.randn(1, 4, 768)

    output = slot(hidden, enabled=False)

    assert torch.equal(output.hidden, hidden)
    assert output.auxiliary_loss is not None
    assert output.auxiliary_loss.item() == 0.0


def test_initially_only_adapters_and_external_gates_are_trainable() -> None:
    slots = {
        "jepa": JEPAAuxiliarySlot(_frozen_jepa()),
        "gaba": GABAResidualSlot(_frozen_gaba()),
    }

    for slot in slots.values():
        trainable = {
            name for name, parameter in slot.named_parameters()
            if parameter.requires_grad
        }
        assert trainable
        assert all(
            name.startswith("adapter.") or name == "external_gate"
            for name in trainable
        )
        assert all(
            not parameter.requires_grad
            for parameter in slot.organ.parameters()
        )


def _tiny_gpt() -> GPT2LMHeadModel:
    torch.manual_seed(55)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=97,
            n_positions=32,
            n_ctx=32,
            n_embd=768,
            n_layer=2,
            n_head=12,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
    ).eval()


def _recipient(teacher: GPT2LMHeadModel | None = None) -> TwoDonorRecipient:
    return TwoDonorRecipient.from_language_donor(
        teacher or _tiny_gpt(),
        jepa_slot=JEPAAuxiliarySlot(_frozen_jepa()),
        gaba_slots={0: (GABAResidualSlot(_frozen_gaba()),)},
    )


@pytest.mark.parametrize("sequence_length", (2, 5, 11))
def test_zero_gate_recipient_matches_simple_gpt(sequence_length: int) -> None:
    teacher = _tiny_gpt()
    recipient = _recipient(teacher)
    ids = torch.arange(1, sequence_length + 1).unsqueeze(0)

    with torch.no_grad():
        baseline = teacher(input_ids=ids).logits
        actual = recipient(input_ids=ids, organ_mode="disabled").logits

    torch.testing.assert_close(actual, baseline, atol=1e-5, rtol=0)
    assert not any(
        left.data_ptr() == right.data_ptr()
        for left, right in zip(
            teacher.parameters(),
            recipient.language_model.parameters(),
            strict=True,
        )
    )


def test_shadow_recipient_executes_organs_without_changing_logits() -> None:
    recipient = _recipient()
    ids = torch.tensor([[1, 2, 3, 4]])

    with torch.no_grad():
        disabled = recipient(input_ids=ids, organ_mode="disabled")
        shadow = recipient(input_ids=ids, organ_mode="shadow")

    torch.testing.assert_close(shadow.logits, disabled.logits, atol=1e-5, rtol=0)
    assert "gaba.0" in shadow.organ_observations
    assert shadow.jepa_loss is not None
    assert shadow.jepa_loss.item() == 0.0


def test_active_gaba_changes_logits_only_after_external_gate_opens() -> None:
    recipient = _recipient()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        baseline = recipient(input_ids=ids, organ_mode="disabled").logits
        recipient.organ_slot("gaba.0").external_gate.fill_(0.5)
        active = recipient(input_ids=ids, organ_mode="adapter_active").logits

    assert active.shape == baseline.shape
    assert torch.isfinite(active).all()
    assert not torch.equal(active, baseline)


def test_disabled_recipient_does_not_mutate_gaba_state() -> None:
    recipient = _recipient()
    slot = recipient.organ_slot("gaba.0")
    before = slot.organ.state_update_count.clone()

    recipient(input_ids=torch.tensor([[1, 2, 3]]), organ_mode="disabled")

    assert torch.equal(slot.organ.state_update_count, before)


def test_recipient_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unsupported organ mode"):
        _recipient()(
            input_ids=torch.tensor([[1, 2]]),
            organ_mode="mystery",
        )


def test_ihs_zero_gate_and_heartbeat_shadow_preserve_logits_and_state() -> None:
    heartbeat = HeartbeatOrgan(
        Heartbeat(512, HeartbeatConfig(ff_layers=2, memory_capacity=8))
    )
    recipient = TwoDonorRecipient.from_language_donor(
        _tiny_gpt(),
        jepa_slot=JEPAAuxiliarySlot(_frozen_jepa()),
        gaba_slots={0: (GABAResidualSlot(_frozen_gaba()),)},
        heartbeat_slot=HeartbeatSlot(heartbeat),
        ihs_slot=IHSResidualSlot(
            InterHemisphericSystem(
                HemisphereConfig(d_model=512, n_heads=8, dropout=0.0)
            )
        ),
    )
    ids = torch.tensor([[1, 2, 3, 4]])
    before = heartbeat.export_state()["beat"]

    with torch.no_grad():
        disabled = recipient(input_ids=ids, organ_mode="disabled")
        shadow = recipient(input_ids=ids, organ_mode="shadow")

    torch.testing.assert_close(shadow.logits, disabled.logits, atol=1e-5, rtol=0)
    assert heartbeat.export_state()["beat"] == before
    assert set(("ihs", "heartbeat")).issubset(shadow.organ_observations)
    assert shadow.heartbeat_loss is not None
    assert shadow.heartbeat_loss.item() == 0.0


def test_active_ihs_changes_logits_and_heartbeat_advances() -> None:
    heartbeat = HeartbeatOrgan(
        Heartbeat(512, HeartbeatConfig(ff_layers=2, memory_capacity=8))
    )
    recipient = TwoDonorRecipient.from_language_donor(
        _tiny_gpt(),
        jepa_slot=JEPAAuxiliarySlot(_frozen_jepa()),
        gaba_slots={0: (GABAResidualSlot(_frozen_gaba()),)},
        heartbeat_slot=HeartbeatSlot(heartbeat),
        ihs_slot=IHSResidualSlot(
            InterHemisphericSystem(
                HemisphereConfig(d_model=512, n_heads=8, dropout=0.0)
            )
        ),
    )
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        baseline = recipient(input_ids=ids, organ_mode="disabled").logits
        recipient.organ_slot("ihs").external_gate.fill_(0.5)
        active = recipient(input_ids=ids, organ_mode="adapter_active")

    assert not torch.equal(active.logits, baseline)
    assert heartbeat.export_state()["beat"] == 1
    assert active.organ_observations["heartbeat"]["beat"] == 1


def test_guarded_policy_runs_all_selected_organs_with_separate_authority() -> None:
    heartbeat = HeartbeatOrgan(
        Heartbeat(512, HeartbeatConfig(ff_layers=2, memory_capacity=8))
    )
    recipient = TwoDonorRecipient.from_language_donor(
        _tiny_gpt(),
        jepa_slot=JEPAAuxiliarySlot(_frozen_jepa()),
        gaba_slots={0: (GABAResidualSlot(_frozen_gaba()),)},
        spider_slot=SpiderSenseSlot(SpiderSense(d_model=512)),
        ttm_slot=TTMResidualSlot(
            torch.nn.Linear(512, 128),
            torch.nn.Linear(512, 512),
        ),
        heartbeat_slot=HeartbeatSlot(heartbeat),
        ihs_slot=IHSResidualSlot(
            InterHemisphericSystem(
                HemisphereConfig(d_model=512, n_heads=8, dropout=0.0)
            )
        ),
    )
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        recipient.organ_slot("gaba.0").external_gate.fill_(0.5)
        disabled = recipient(input_ids=ids, organ_mode="disabled")
        before_beat = heartbeat.export_state()["beat"]
        recipient.train()
        training = recipient(
            input_ids=ids,
            organ_policy=GUARDED_TRANSPLANT_POLICY,
        )
        recipient.eval()
        evaluation = recipient(
            input_ids=ids,
            organ_policy=GUARDED_TRANSPLANT_POLICY,
        )

    assert not torch.equal(training.logits, disabled.logits)
    assert training.jepa_loss is not None
    assert training.jepa_loss.item() > 0.0
    assert evaluation.jepa_loss is not None
    assert evaluation.jepa_loss.item() == 0.0
    assert set(("gaba.0", "jepa", "ttm", "spider", "heartbeat")).issubset(
        training.organ_observations
    )
    assert "ihs" not in training.organ_observations
    assert "mtp" not in training.organ_observations
    assert training.organ_observations["ttm"]["memory_size"] == 0
    assert heartbeat.export_state()["beat"] == before_beat
    assert training.organ_observations["heartbeat"]["shadow"] is True


def test_execution_policy_rejects_unknown_organ_or_mode() -> None:
    recipient = _recipient()
    ids = torch.tensor([[1, 2]])

    with pytest.raises(ValueError, match="unknown organs"):
        recipient(
            input_ids=ids,
            organ_policy={"imaginary": "active"},
        )
    with pytest.raises(ValueError, match="unsupported organ execution modes"):
        recipient(
            input_ids=ids,
            organ_policy={"jepa": "reckless"},
        )
