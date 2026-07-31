from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.losses import CAUSAL_LOSS_SEMANTICS_VERSION
from f51_darwin.darwin_x_core.model import (
    DarwinXModel,
    deterministic_ghost_mask,
)
from f51_darwin.organism.causal_adapters import (
    CausalExecutionError,
    CausalTrainingExecutor,
)
from f51_darwin.organism.causal_bus import (
    AblationArm,
    BusDecision,
    Intervention,
    InterventionOperation,
    InterventionTarget,
    Phase,
    StepIdentity,
)


def _small_config(**overrides: object) -> DarwinXConfig:
    values = {
        "vocab_size": 32,
        "context_length": 8,
        "inference_context_length": 16,
        "d_model": 16,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "fine_experts": 2,
        "shared_experts": 1,
        "experts_per_token": 1,
        "fine_expert_hidden_dim": 8,
        "shared_expert_hidden_dim": 8,
        "mtp_depth": 1,
        "aux_loss_adaptive": False,
        "aux_loss_scale": 0.0,
        "ghost_enabled": True,
        "ghost_weight": 0.25,
        "ghost_mask_ratio": 0.5,
        "spider_sense_enabled": False,
        "heartbeat_enabled": False,
        "loss_semantics_version": CAUSAL_LOSS_SEMANTICS_VERSION,
    }
    values.update(overrides)
    return DarwinXConfig(**values)


def _identity() -> StepIdentity:
    return StepIdentity(
        run_id="ghost-run",
        cycle=1,
        optimizer_step=7,
        accumulation_window=0,
        batch_digest="batch:ghost",
        rng_digest="rng:ghost",
        base_checkpoint_id="base:ghost",
        training_contract_id="contract:ghost",
        ablation_plan_id="ghost-arms",
        attempt_id="attempt-ghost",
    )


def _ghost_decision(
    arm: AblationArm,
    scale: float,
    *,
    subject: str = "ghost",
) -> BusDecision:
    identity = _identity()
    request = Intervention(
        intervention_id=f"{subject}-scale",
        source="ghost:v2",
        phase=Phase.PRE_LOSS,
        target=InterventionTarget.LOSS_TERM,
        operation=InterventionOperation.SET_SCALE,
        subject=subject,
        value=scale,
        valid_from_step=identity.optimizer_step,
        valid_through_step=identity.optimizer_step,
    )
    return BusDecision(
        identity=identity,
        phase=Phase.PRE_LOSS,
        arm=arm,
        accepted=(request,),
    )


def test_ghost_mask_is_digest_deterministic_and_rng_pure() -> None:
    ids = torch.arange(24).reshape(2, 12)
    before = torch.random.get_rng_state().clone()

    first = deterministic_ghost_mask(ids, "step-a", 0.25)
    second = deterministic_ghost_mask(ids, "step-a", 0.25)
    after = torch.random.get_rng_state()

    assert first.dtype is torch.bool
    assert torch.equal(first, second)
    assert torch.equal(before, after)
    assert not torch.equal(
        first,
        deterministic_ghost_mask(ids, "step-b", 0.25),
    )


def test_ghost_mask_is_sample_local() -> None:
    a = torch.tensor([[4, 5, 6, 7], [9, 8, 7, 6]])
    b = torch.tensor([[4, 5, 6, 7], [0, 0, 0, 0]])

    assert torch.equal(
        deterministic_ghost_mask(a, "step", 0.5)[0],
        deterministic_ghost_mask(b, "step", 0.5)[0],
    )


def test_ghost_mask_excludes_specials_and_guarantees_an_eligible_token() -> None:
    ids = torch.tensor([[0, 1, 2, 3, 4, 5]])

    full = deterministic_ghost_mask(ids, "step", 1.0)
    sparse = deterministic_ghost_mask(ids, "step", 1e-12)

    assert full.tolist() == [[False, False, False, False, True, True]]
    assert sparse[:, :4].sum().item() == 0
    assert sparse.sum().item() == 1
    assert not deterministic_ghost_mask(ids, "step", 0.0).any()
    assert not deterministic_ghost_mask(
        torch.tensor([[0, 1, 2, 3]]),
        "step",
        0.5,
    ).any()
    assert deterministic_ghost_mask(
        torch.tensor([[4, 5, 6]]),
        "step",
        1.0,
    ).tolist() == [[False, True, True]]


def test_ghost_mask_fallback_is_per_sample_and_permutation_local() -> None:
    rows = torch.tensor(
        [
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
        ]
    )
    ratio = 1e-12

    alone = deterministic_ghost_mask(rows[:1], "step", ratio)
    batched = deterministic_ghost_mask(rows, "step", ratio)
    permutation = torch.tensor([2, 0, 1])
    permuted = deterministic_ghost_mask(
        rows[permutation],
        "step",
        ratio,
    )

    assert batched.sum(dim=1).tolist() == [1, 1, 1]
    assert torch.equal(alone[0], batched[0])
    assert torch.equal(permuted, batched[permutation])


def test_ghost_mask_uses_no_tensor_item_or_python_bool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("device tensor synchronized into Python")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", forbidden)
        patch.setattr(torch.Tensor, "__bool__", forbidden)
        mask = deterministic_ghost_mask(
            torch.tensor([[4, 5, 6], [7, 8, 9]]),
            "step",
            1e-12,
        )

    assert mask.sum(dim=1).tolist() == [1, 1]


@pytest.mark.parametrize("ratio", [-0.1, 1.1, float("nan")])
def test_ghost_mask_rejects_invalid_ratio(ratio: float) -> None:
    with pytest.raises(ValueError, match="ratio"):
        deterministic_ghost_mask(torch.tensor([[4, 5]]), "step", ratio)


def test_causal_ghost_probe_traverses_each_block_once() -> None:
    torch.manual_seed(51)
    model = DarwinXModel(_small_config())
    ids = torch.tensor([[4, 5, 6, 7, 8, 9]])
    calls = [0 for _ in model.blocks]
    hooks = [
        block.register_forward_hook(
            lambda _module, _args, _result, index=index: calls.__setitem__(
                index, calls[index] + 1
            )
        )
        for index, block in enumerate(model.blocks)
    ]
    try:
        output = model(
            ids,
            labels=ids,
            heartbeat=False,
            step_digest="step-7",
        )
    finally:
        for hook in hooks:
            hook.remove()

    assert calls == [1 for _ in model.blocks]
    assert output.raw_ghost_loss is not None
    assert output.raw_ghost_loss.item() > 0.0
    assert output.ghost_loss is not None
    # Direct forward (no causal executor): ghost_loss carries the effective
    # composed value (raw * ghost_weight).  CONTROL/SHADOW zeroing is
    # handled by CausalTrainingExecutor.set_loss_scales.
    assert output.ghost_loss.item() > 0.0
    assert output.ghost_mask_digest


def test_causal_ghost_predictor_does_not_see_the_target_token() -> None:
    torch.manual_seed(51)
    config = _small_config(ghost_mask_ratio=1.0)
    initial = copy.deepcopy(DarwinXModel(config).state_dict())
    paired_rng = torch.random.get_rng_state().clone()
    original = torch.tensor([[4, 5, 6, 7, 8, 9]])
    changed = original.clone()
    changed[0, 3] = 19

    def ghost_representation(ids: torch.Tensor) -> torch.Tensor:
        model = DarwinXModel(config)
        model.load_state_dict(initial, strict=True)
        torch.random.set_rng_state(paired_rng)
        captured: list[torch.Tensor] = []

        def capture(_module, args):
            representation = args[0]
            if representation.ndim == 3 and representation.size(1) == ids.size(1) - 1:
                captured.append(representation.detach().clone())

        hook = model.lm_head.register_forward_pre_hook(capture)
        try:
            model(
                ids,
                labels=ids,
                heartbeat=False,
                step_digest="step-7",
            )
        finally:
            hook.remove()
        assert len(captured) == 1
        return captured[0]

    original_prediction = ghost_representation(original)
    changed_prediction = ghost_representation(changed)

    torch.testing.assert_close(
        original_prediction[:, 2],
        changed_prediction[:, 2],
        rtol=0.0,
        atol=0.0,
    )


def test_causal_ghost_digest_never_transfers_mask_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(51)
    model = DarwinXModel(_small_config())

    def forbidden_numpy(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("mask digest transferred a tensor to CPU")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "numpy", forbidden_numpy)
        output = model(
            torch.tensor([[4, 5, 6, 7, 8, 9]]),
            labels=torch.tensor([[4, 5, 6, 7, 8, 9]]),
            heartbeat=False,
            step_digest="step-7",
        )

    assert output.ghost_mask_digest


def test_causal_ghost_stabilizes_nonfinite_and_extreme_logits_before_ce() -> None:
    class HostileHead(torch.nn.Module):
        def __init__(self, logits: torch.Tensor) -> None:
            super().__init__()
            self.logits = torch.nn.Parameter(logits)

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return self.logits.expand(hidden.size(0), -1, -1)

    torch.manual_seed(51)
    config = _small_config(ghost_mask_ratio=1.0)
    model = DarwinXModel(config)
    ids = torch.tensor([[4, 5, 6, 7]])
    labels = ids.clone()
    hidden = torch.randn(1, ids.size(1), config.d_model)
    hostile = torch.full(
        (1, ids.size(1) - 1, config.vocab_size),
        1e30,
    )
    hostile[0, 0, 0] = float("nan")
    hostile[0, 1, 1] = float("inf")
    hostile[0, 2, 2] = -float("inf")
    head = HostileHead(hostile)
    model.lm_head = head

    raw, _digest = model._causal_ghost_loss(
        ids,
        labels,
        hidden,
        step_digest="step-hostile",
    )
    safe_logits = torch.nan_to_num(
        head.logits.float(),
        nan=0.0,
        posinf=30.0,
        neginf=-30.0,
    ).clamp(min=-30.0, max=30.0)
    expected = torch.nn.functional.cross_entropy(
        safe_logits.reshape(-1, safe_logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).mean()

    assert torch.isfinite(raw)
    torch.testing.assert_close(raw, expected, rtol=0.0, atol=0.0)
    raw.backward()
    assert head.logits.grad is not None
    assert torch.isfinite(head.logits.grad).all()


def test_control_shadow_apply_share_raw_probe_but_only_apply_changes_gradient() -> None:
    torch.manual_seed(51)
    config = _small_config()
    initial = copy.deepcopy(DarwinXModel(config).state_dict())
    paired_rng = torch.random.get_rng_state().clone()
    ids = torch.tensor([[4, 5, 6, 7, 8, 9]])
    results: dict[AblationArm, tuple[object, torch.Tensor]] = {}

    for arm in (
        AblationArm.CONTROL,
        AblationArm.SHADOW,
        AblationArm.APPLY,
    ):
        model = DarwinXModel(config)
        model.load_state_dict(initial, strict=True)
        torch.random.set_rng_state(paired_rng)
        output = model(
            ids,
            labels=ids,
            heartbeat=False,
            step_digest="step-7",
        )
        adjusted = CausalTrainingExecutor(model).set_loss_scales(
            _ghost_decision(arm, config.ghost_weight),
            output,
        )
        assert adjusted.loss is not None
        adjusted.loss.backward()
        results[arm] = (
            adjusted,
            model.lm_head.weight.grad.detach().clone(),
        )

    control, control_grad = results[AblationArm.CONTROL]
    shadow, shadow_grad = results[AblationArm.SHADOW]
    apply, apply_grad = results[AblationArm.APPLY]

    assert control.ghost_mask_digest == shadow.ghost_mask_digest
    assert shadow.ghost_mask_digest == apply.ghost_mask_digest
    torch.testing.assert_close(
        control.raw_ghost_loss,
        shadow.raw_ghost_loss,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        shadow.raw_ghost_loss,
        apply.raw_ghost_loss,
        rtol=0.0,
        atol=0.0,
    )
    assert control.ghost_loss.item() == 0.0
    assert shadow.ghost_loss.item() == 0.0
    torch.testing.assert_close(
        apply.ghost_loss,
        apply.raw_ghost_loss * config.ghost_weight,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(control.loss, shadow.loss, rtol=0.0, atol=0.0)
    torch.testing.assert_close(control_grad, shadow_grad, rtol=0.0, atol=0.0)
    assert not torch.equal(apply_grad, shadow_grad)


@pytest.mark.parametrize(
    ("subject", "field"),
    [
        ("aux", "aux_loss"),
        ("ghost", "raw_ghost_loss"),
        ("jepa", "jepa_loss"),
        ("spider", "spider_loss"),
    ],
)
def test_loss_scale_requires_explicit_raw_requested_term(
    subject: str,
    field: str,
) -> None:
    output = SimpleNamespace(
        loss=torch.tensor(10.0),
        lm_loss=torch.tensor(2.0),
        mtp_loss=torch.tensor(3.0),
        jepa_loss=torch.tensor(4.0),
        aux_loss=torch.tensor(5.0),
        raw_ghost_loss=torch.tensor(6.0),
        spider_loss=torch.tensor(7.0),
        ghost_mask_digest="digest",
    )
    delattr(output, field)
    model = SimpleNamespace(
        config=SimpleNamespace(
            loss_semantics_version=CAUSAL_LOSS_SEMANTICS_VERSION,
            mtp_weight=0.15,
            jepa_weight=0.05,
            aux_loss_scale=1.0,
            ghost_weight=0.25,
            spider_calibration_weight=0.1,
            aux_loss_adaptive=False,
        )
    )

    with pytest.raises(
        CausalExecutionError,
        match=f"{subject} SET_SCALE requires explicit",
    ):
        CausalTrainingExecutor(model).set_loss_scales(
            _ghost_decision(
                AblationArm.APPLY,
                0.5,
                subject=subject,
            ),
            output,
        )
