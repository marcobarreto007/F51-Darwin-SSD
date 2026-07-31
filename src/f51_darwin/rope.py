from __future__ import annotations

import torch


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    *,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE.")
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(positions, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to queries or keys shaped [batch, heads, seq, head_dim]."""
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    out = torch.empty_like(x)
    out[..., ::2] = rotated_even
    out[..., 1::2] = rotated_odd
    return out


def apply_rope_llama(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply the split-half RoPE convention used by Llama/SmolLM."""
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    cos_full = torch.cat((cos, cos), dim=-1).unsqueeze(0).unsqueeze(0)
    sin_full = torch.cat((sin, sin), dim=-1).unsqueeze(0).unsqueeze(0)
    return (x * cos_full + rotated * sin_full).to(dtype=x.dtype)


class RoPECache:
    """Cache single-entry para tabelas cos/sin do RoPE (P1.2).

    As tabelas depend apenas de (seq_len, head_dim, base, device). Reconstruilas
    a cada forward e puro desperdicio. Mantemos a entrada mais recente (caso
    comum: seq_len fixo num run de treino), em O(1) memoria.

    Nao e buffer registrado: as tabelas sao deterministicas (nao aprendidas),
    entao nao pertencem ao state_dict. Keying por device garante que um
    model.to(device) provoque rebuild automatico.
    """

    __slots__ = ("_key", "_cos", "_sin")

    def __init__(self) -> None:
        self._key: tuple | None = None
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None

    def get(
        self,
        seq_len: int,
        head_dim: int,
        *,
        base: float = 10000.0,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dev = torch.device(device) if device is not None else None
        key = (seq_len, head_dim, base, dev)
        if self._key == key and self._cos is not None:
            assert self._sin is not None
            return self._cos, self._sin
        cos, sin = build_rope_cache(seq_len, head_dim, base=base, device=dev)
        self._key = key
        self._cos = cos
        self._sin = sin
        return cos, sin

    def clear(self) -> None:
        """Invalida o cache (ex.: apos mudanca explicita de dispositivo)."""
        self._key = None
        self._cos = None
        self._sin = None
