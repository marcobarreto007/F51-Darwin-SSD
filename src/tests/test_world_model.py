from __future__ import annotations

import pytest
import torch

from f51_darwin.cognition.world_model import (
    HORIZON_LONG,
    HORIZON_MEDIUM,
    HORIZON_SHORT,
    NUM_PROPOSAL_SLOTS,
    HierarchicalWorldModel,
    PlanAdapter,
    TargetEncoder,
    TrajectoryCandidate,
    TrajectoryProposer,
)


# ── helpers ─────────────────────────────────────────────────────────────────


def _sq(batch: int = 2, seq: int = 8, d: int = 64) -> torch.Tensor:
    return torch.randn(batch, seq, d)


def _make_organ(d_model: int = 64, **kwargs: object) -> HierarchicalWorldModel:
    return HierarchicalWorldModel(d_model, **kwargs)


# ── Target Encoder ──────────────────────────────────────────────────────────


def test_target_encoder_produces_512_dim_output() -> None:
    encoder = TargetEncoder(d_model=64)
    hidden = _sq()
    out = encoder(hidden)
    assert out.shape == (2, 512)
    target = encoder.encode_target(hidden)
    assert target.shape == (2, 512)


def test_target_encoder_ema_diverges_from_online() -> None:
    encoder = TargetEncoder(d_model=64, ema_decay=0.5)
    hidden = _sq()
    online = encoder(hidden)
    target = encoder.encode_target(hidden)
    # EMA is initialized as copy, so first call they match
    assert torch.allclose(online, target, atol=1e-5)

    # Modify online weights via gradient, then update EMA — they should diverge
    loss = online.sum()
    loss.backward()
    with torch.no_grad():
        for p in encoder.encoder.parameters():
            if p.grad is not None:
                p.add_(p.grad, alpha=-0.1)
    encoder.update_ema()

    hidden2 = _sq()
    online2 = encoder(hidden2)
    target2 = encoder.encode_target(hidden2)
    # EMA lags behind updated online weights
    assert not torch.equal(online2, target2)


# ── Trajectory Proposer ─────────────────────────────────────────────────────


def test_proposer_output_shapes() -> None:
    proposer = TrajectoryProposer()
    context = torch.randn(2, 10, 512)
    zs, zm, zl = proposer(context)
    assert zs.shape == (2, NUM_PROPOSAL_SLOTS, 512)
    assert zm.shape == (2, NUM_PROPOSAL_SLOTS, 512)
    assert zl.shape == (2, NUM_PROPOSAL_SLOTS, 512)


def test_proposals_are_not_collapsed() -> None:
    torch.manual_seed(51)
    proposer = TrajectoryProposer()
    context = torch.randn(2, 10, 512)
    zs, _, _ = proposer(context)
    # Variance across proposal slots should be non-zero
    var = zs.var(dim=1).mean()
    assert var > 0.0


# ── Plan Adapter ────────────────────────────────────────────────────────────


def test_plan_adapter_gate_zero_no_effect() -> None:
    torch.manual_seed(51)
    adapter = PlanAdapter(d_model=16, max_scale=0.15)
    hidden = torch.randn(2, 4, 16)
    zs = torch.randn(2, 512)
    zm = torch.randn(2, 512)
    zl = torch.randn(2, 512)
    pos = torch.tensor([[2], [1]], dtype=torch.long)
    result = adapter.condition(hidden, zs, zm, zl, pos)
    assert torch.equal(result, hidden)
    assert adapter.gate.item() == 0.0


def test_plan_adapter_with_gate_applies_bounded_residual() -> None:
    torch.manual_seed(51)
    adapter = PlanAdapter(d_model=16, max_scale=0.15)
    hidden = torch.randn(2, 4, 16)
    zs = torch.randn(2, 512)
    zm = torch.randn(2, 512)
    zl = torch.randn(2, 512)
    pos = torch.tensor([[2], [1]], dtype=torch.long)
    with torch.no_grad():
        adapter.gate.fill_(2.0)
    result = adapter.condition(hidden, zs, zm, zl, pos)
    assert not torch.equal(result, hidden)


# ── HierarchicalWorldModel ──────────────────────────────────────────────────


def test_encode_context_returns_correct_shape() -> None:
    organ = _make_organ(d_model=64)
    hidden = _sq(batch=1)
    ctx = organ.encode_context(hidden)
    assert ctx.shape[0] == 1
    assert ctx.shape[-1] == 512


def test_encode_context_with_goal_and_memory() -> None:
    organ = _make_organ(d_model=64)
    hidden = _sq(batch=1)
    goal = torch.randn(1, 512)
    memories = [torch.randn(512), torch.randn(512)]
    ctx = organ.encode_context(hidden, goal, memories)
    # T_ctx = seq + 1 (goal) + 2 (memories)
    assert ctx.shape[1] == hidden.shape[1] + 3


def test_propose_trajectories_returns_k_candidates() -> None:
    organ = _make_organ(d_model=64)
    hidden = _sq(batch=1)
    candidates = organ.propose_trajectories(hidden)
    assert len(candidates) == NUM_PROPOSAL_SLOTS
    for c in candidates:
        assert isinstance(c, TrajectoryCandidate)
        assert c.z_short.shape == (512,)
        assert c.z_medium.shape == (512,)
        assert c.z_long.shape == (512,)
        assert 0.0 <= c.diversity_score <= 2.0


def test_compute_losses_returns_all_components() -> None:
    organ = _make_organ(d_model=64)
    hidden = _sq(batch=1)
    future_s = torch.randn(1, HORIZON_SHORT, 64)
    future_m = torch.randn(1, HORIZON_MEDIUM, 64)
    future_l = torch.randn(1, HORIZON_LONG, 64)
    losses = organ.compute_losses(hidden, future_s, future_m, future_l)
    assert "prediction_error" in losses
    assert "diversity" in losses
    assert "anti_collapse" in losses
    assert torch.isfinite(losses["prediction_error"])
    assert torch.isfinite(losses["diversity"])


def test_compute_losses_without_futures_is_finite() -> None:
    organ = _make_organ(d_model=64)
    hidden = _sq(batch=1)
    losses = organ.compute_losses(hidden, None, None, None)
    # Prediction error should be 0 when no futures provided
    assert losses["prediction_error"].item() == 0.0
    assert losses["diversity"].isfinite()
    assert losses["anti_collapse"].isfinite()


def test_update_target_encoder() -> None:
    organ = _make_organ(d_model=64)
    # Should not raise
    organ.update_target_encoder()
