import math

import torch

from f51_darwin.config import DarwinConfig
from f51_darwin.moe_layer import MoERouter
from f51_darwin.model import F51DarwinModel


def test_darwin_forward_logits_and_loss_shape() -> None:
    torch.manual_seed(51)
    config = DarwinConfig(
        model_name="F51-Darwin-SSD-Test",
        vocab_size=128,
        context_length=16,
        d_model=32,
        n_layers=4,
        n_heads=4,
        mlp_ratio=2,
    )
    model = F51DarwinModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    out = model(input_ids, labels=input_ids)
    assert out.logits.shape == (2, 16, config.vocab_size)
    assert out.loss is not None
    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)


def test_darwin_rejects_too_long_context() -> None:
    config = DarwinConfig(vocab_size=64, context_length=8, d_model=16, n_layers=4, n_heads=4)
    model = F51DarwinModel(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 9))
    try:
        model(input_ids)
    except ValueError as exc:
        assert "context_length" in str(exc)
    else:
        raise AssertionError("Expected ValueError for long context.")


def test_large_moe_random_init_stays_near_entropy() -> None:
    torch.manual_seed(51)
    config = DarwinConfig(
        model_name="F51-Darwin-MoE-Init-Smoke",
        vocab_size=4096,
        context_length=32,
        d_model=768,
        n_layers=4,
        n_heads=8,
    )
    model = F51DarwinModel(
        config,
        experts_enabled=True,
        moe_config={
            "num_experts": 12,
            "experts_per_token": 2,
            "expert_hidden_mult": 3,
            "use_nitro_tiering": False,
        },
    )
    input_ids = torch.randint(0, config.vocab_size, (1, 16))

    out = model(input_ids, labels=input_ids)
    assert out.loss is not None
    assert torch.isfinite(out.loss)
    assert out.logits.detach().std().item() < 0.45
    assert out.loss.item() < math.log(config.vocab_size) + 0.75


def test_moe_router_keeps_logits_float32_under_low_precision() -> None:
    router = MoERouter(d_model=32, num_experts=4, top_k=2).to(dtype=torch.bfloat16)
    x = torch.randn(2, 8, 32, dtype=torch.bfloat16)

    routing_weights, expert_indices, router_logits = router(x)

    assert router_logits.dtype == torch.float32
    assert routing_weights.dtype == torch.bfloat16
    assert expert_indices.dtype == torch.long
    assert torch.isfinite(router_logits).all()
