from __future__ import annotations

import torch

from f51_darwin.rope import apply_rope_llama, build_rope_cache


def test_llama_rope_matches_split_half_definition() -> None:
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(2, 3, 5, 8, generator=generator)
    cos, sin = build_rope_cache(5, 8, base=130000.0)
    x = x.to(torch.bfloat16)
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    expected = (
        x * torch.cat((cos, cos), dim=-1)[None, None]
        + rotated * torch.cat((sin, sin), dim=-1)[None, None]
    )
    torch.testing.assert_close(
        apply_rope_llama(x, cos, sin),
        expected.to(torch.bfloat16),
    )
