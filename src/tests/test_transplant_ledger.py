from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from f51_darwin.gaba_inhibition import GABAConfig, GABAergicLayer
from f51_darwin.jepa_v2 import JEPAHeadV2
from f51_darwin.transplant.experiment import (
    ArmBudget,
    build_first_slice_arms,
    validate_matched_arms,
)
from f51_darwin.transplant.ledger import (
    OrganEvidence,
    OrganLedger,
    organ_utility,
)
from f51_darwin.transplant.recipient import TwoDonorRecipient
from f51_darwin.transplant.slots import GABAResidualSlot, JEPAAuxiliarySlot


def test_matched_arms_reject_different_tokens_or_seeds() -> None:
    arms = {
        "A": ArmBudget(tokens=4096, steps=8, seed=7, starts=(0, 128)),
        "B": ArmBudget(tokens=4096, steps=8, seed=7, starts=(0, 128)),
        "C": ArmBudget(tokens=4096, steps=8, seed=8, starts=(0, 128)),
        "D": ArmBudget(tokens=4096, steps=8, seed=7, starts=(0, 128)),
    }
    with pytest.raises(ValueError, match="matched-arm budget mismatch"):
        validate_matched_arms(arms)


def test_useful_and_harmful_organs_receive_opposite_utility() -> None:
    useful = OrganEvidence(
        control_nll=3.0,
        living_nll=2.8,
        future_gain=0.1,
        forgetting=0.01,
        compute_cost=0.02,
        instability=0.0,
    )
    harmful = replace(useful, living_nll=3.3, future_gain=0.0)
    assert organ_utility(useful) > 0
    assert organ_utility(harmful) < 0


def test_jsonl_roundtrip_preserves_evidence_hashes(tmp_path) -> None:
    path = tmp_path / "organ_ledger.jsonl"
    budget = ArmBudget(tokens=1024, steps=2, seed=17, starts=(0, 64))
    evidence = OrganEvidence(
        control_nll=3.2,
        living_nll=3.0,
        future_gain=0.04,
        forgetting=0.01,
        compute_cost=0.03,
        instability=0.0,
        compute_saving=0.02,
    )
    ledger = OrganLedger(path)
    first = ledger.append(
        organ_name="gaba.0",
        language_donor_sha256="a" * 64,
        organ_donor_sha256="b" * 64,
        recipient_parent_sha256="c" * 64,
        arm="D",
        budget=budget,
        evidence=evidence,
        lifecycle_decision="active",
    )

    reconstructed = OrganLedger.from_jsonl(path)

    assert reconstructed.records == (first,)
    assert reconstructed.records[0].evidence_id == first.evidence_id
    assert reconstructed.sha256 == ledger.sha256


def _recipient_factory() -> TwoDonorRecipient:
    torch.manual_seed(61)
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


def test_first_slice_arms_are_independent_and_have_declared_boundaries() -> None:
    budget = ArmBudget(tokens=128, steps=1, seed=23, starts=(0,))

    arms = build_first_slice_arms(
        _recipient_factory,
        organ_name="gaba.0",
        budgets={name: budget for name in "ABCD"},
    )

    assert tuple(arms) == ("A", "B", "C", "D")
    assert arms["A"].organ_mode == "disabled"
    assert arms["A"].declared_parameters == frozenset()
    assert arms["B"].organ_mode == "shadow"
    assert arms["B"].declared_parameters
    assert all(
        name.startswith("language_model.")
        for name in arms["B"].declared_parameters
    )
    assert arms["C"].organ_mode == "adapter_active"
    assert all("gaba_slots.gaba_0" in name for name in arms["C"].declared_parameters)
    assert arms["D"].organ_mode == "organ_unfrozen"
    assert any(
        ".organ." in name for name in arms["D"].declared_parameters
    )

    parameters = {
        name: tuple(parameter.data_ptr() for parameter in arm.recipient.parameters())
        for name, arm in arms.items()
    }
    for left_index, left in enumerate("ABCD"):
        for right in "ABCD"[left_index + 1 :]:
            assert set(parameters[left]).isdisjoint(parameters[right])

