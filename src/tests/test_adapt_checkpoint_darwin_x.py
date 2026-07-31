from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_adapt_module():
    root = Path(__file__).resolve().parents[2]
    script = root / "src" / "tools" / "adapt_checkpoint_darwin_x.py"
    spec = importlib.util.spec_from_file_location("adapt_checkpoint_darwin_x", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adapted_tensor_slices_larger_source() -> None:
    adapt = _load_adapt_module()
    source_state = {"src": torch.ones(4, 3)}
    target = torch.zeros(2, 3)

    out = adapt.adapted_tensor(target, source_state, {"mode": "slice", "source": "src"})

    assert out.shape == target.shape
    assert torch.equal(out, torch.ones(2, 3))


def test_adapted_tensor_pads_smaller_source() -> None:
    adapt = _load_adapt_module()
    source_state = {"src": torch.ones(2, 3)}
    target = torch.zeros(4, 3)

    out = adapt.adapted_tensor(target, source_state, {"mode": "pad_from_source", "source": "src"})

    assert torch.equal(out[:2], torch.ones(2, 3))
    assert torch.equal(out[2:], torch.zeros(2, 3))


def test_adapted_tensor_splits_qkv_prefix_for_gqa() -> None:
    adapt = _load_adapt_module()
    qkv = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    source_state = {"src": qkv}
    target = torch.zeros(1, 3)

    out = adapt.adapted_tensor(
        target,
        source_state,
        {"mode": "split_qkv", "source": "src", "part": "k"},
    )

    assert torch.equal(out, qkv[2:3])
