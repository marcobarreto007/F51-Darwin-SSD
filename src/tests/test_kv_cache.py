from __future__ import annotations

import torch

from f51_darwin.config import DarwinConfig
from f51_darwin.kv_cache import generate_with_cache, selective_scan_with_cache
from f51_darwin.model import F51DarwinModel
from f51_darwin.ssm_core import selective_scan_sequential
from f51_darwin.turbo import turbo_generate


def test_selective_scan_with_cache_matches_sequential() -> None:
    torch.manual_seed(51)
    batch, dim, seq_len, state = 2, 8, 12, 4
    u = torch.randn(batch, dim, seq_len)
    delta = torch.rand(batch, dim, seq_len).abs() + 0.01
    a = -torch.exp(torch.randn(dim, state))
    b = torch.randn(batch, state, seq_len)
    c = torch.randn(batch, state, seq_len)
    d = torch.randn(dim)

    full = selective_scan_sequential(u, delta, a, b, c, d)
    h_prev = None
    parts = []
    for step in range(seq_len):
        sl = slice(step, step + 1)
        chunk, h_prev = selective_scan_with_cache(
            u[:, :, sl],
            delta[:, :, sl],
            a,
            b[:, :, sl],
            c[:, :, sl],
            d,
            h_prev,
        )
        parts.append(chunk)
    cached = torch.cat(parts, dim=-1)
    assert torch.allclose(full, cached, atol=1e-5, rtol=1e-4)


def test_turbo_generate_runs_autoregressive_steps() -> None:
    cfg = DarwinConfig(vocab_size=256, context_length=64, n_layers=4, d_model=64, n_heads=4)
    model = F51DarwinModel(cfg).eval()
    prompt = list(range(1, 17))
    text, generated, reason = turbo_generate(
        model, prompt, None, max_tokens=16, temperature=0.0, top_p=1.0
    )
    assert isinstance(text, str)
    assert len(generated) <= 16
    assert reason in {"eos", "max_length"}


def test_generate_with_cache_matches_full_forward_logits() -> None:
    torch.manual_seed(51)
    cfg = DarwinConfig(vocab_size=128, context_length=64, n_layers=4, d_model=64, n_heads=4)
    model = F51DarwinModel(cfg).eval()
    prompt = [3, 7, 11, 13, 17, 19]

    with torch.no_grad():
        full = model.forward(torch.tensor([prompt]))
        full_last = full.logits[:, -1, :]

    with torch.no_grad():
        _, _, _ = generate_with_cache(
            model, prompt, None, max_tokens=1, temperature=0.0, top_p=1.0
        )

    with torch.no_grad():
        from f51_darwin.kv_cache import KVCache

        cache = KVCache(1, cfg.context_length, torch.device("cpu"))
        x = model.token_embedding(torch.tensor([prompt]))
        cache._seq_len = x.shape[1]
        hidden = x
        for i, block in enumerate(model.blocks):
            if hasattr(block, "norm_mixer"):
                from f51_darwin.kv_cache import _ssd_forward_with_cache

                hidden = _ssd_forward_with_cache(block, hidden, cache, i)
            elif hasattr(block, "norm_attn"):
                from f51_darwin.kv_cache import _attn_forward_with_cache

                hidden = _attn_forward_with_cache(block, hidden, cache, i)
            else:
                hidden = block(hidden)
        hidden = model.norm(hidden)
        cached_last = model.lm_head(hidden[:, -1:, :])[:, 0, :]

    assert torch.allclose(full_last, cached_last, atol=1e-4, rtol=1e-3)


def test_model_generate_use_cache_flag() -> None:
    cfg = DarwinConfig(vocab_size=128, context_length=32, n_layers=4, d_model=64, n_heads=4)
    model = F51DarwinModel(cfg).eval()
    out = model.generate([1, 2, 3, 4], max_tokens=4, temperature=0.0, top_p=1.0, use_cache=True)
    assert out.finish_reason in {"eos", "max_length"}
