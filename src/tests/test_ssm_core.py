import torch

from f51_darwin.config import DarwinConfig
from f51_darwin.ssm_core import SelectiveSSM, selective_scan_sequential


def test_selective_scan_preserves_shape() -> None:
    batch, dim, seq_len, state = 2, 8, 12, 4
    u = torch.randn(batch, dim, seq_len)
    delta = torch.rand(batch, dim, seq_len).abs() + 0.01
    a = -torch.rand(dim, state).abs()
    b = torch.randn(batch, state, seq_len)
    c = torch.randn(batch, state, seq_len)
    d = torch.ones(dim)
    out = selective_scan_sequential(u, delta, a, b, c, d)
    assert out.shape == (batch, dim, seq_len)
    assert torch.isfinite(out).all()


def test_selective_ssm_block_runs() -> None:
    torch.manual_seed(51)
    module = SelectiveSSM(d_model=32, expand=2, d_state=8)
    x = torch.randn(2, 16, 32)
    y = module(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_selective_ssm_gradient_checkpointing_preserves_gradients() -> None:
    torch.manual_seed(51)
    baseline = SelectiveSSM(
        8,
        expand=1,
        d_state=4,
        scan_chunk_size=4,
        gradient_checkpointing=False,
    ).train()
    checkpointed = SelectiveSSM(
        8,
        expand=1,
        d_state=4,
        scan_chunk_size=4,
        gradient_checkpointing=True,
    ).train()
    checkpointed.load_state_dict(baseline.state_dict(), strict=True)
    baseline_input = torch.randn(1, 8, 8, requires_grad=True)
    checkpointed_input = baseline_input.detach().clone().requires_grad_(True)

    baseline(baseline_input).sum().backward()
    checkpointed(checkpointed_input).sum().backward()

    torch.testing.assert_close(
        checkpointed_input.grad,
        baseline_input.grad,
        rtol=1e-5,
        atol=1e-6,
    )


def test_ssm_block_inside_model() -> None:
    from f51_darwin.model import F51DarwinModel

    config = DarwinConfig(vocab_size=64, context_length=16, d_model=32, n_layers=4, n_heads=4)
    model = F51DarwinModel(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    out = model(input_ids, labels=input_ids)
    assert out.loss is not None
    assert torch.isfinite(out.loss)
