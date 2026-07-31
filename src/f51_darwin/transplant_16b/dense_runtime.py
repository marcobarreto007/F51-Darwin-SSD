from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch

from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel

from .checkpoint import verify_shard_manifest
from .dense_assembly import _assert_dense_structure
from .exact_assembly import _atomic_json, _new_model


def _nonzero_gradient(parameter: torch.nn.Parameter) -> bool:
    return (
        parameter.grad is not None
        and bool(torch.count_nonzero(parameter.grad).item())
    )


def run_dense_optimizer_smoke(
    *,
    checkpoint: str | Path,
    manifest: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    verify_shard_manifest(checkpoint, manifest)
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    config = DarwinXConfig.from_mapping(payload["config"])
    if config.feed_forward_kind != "dense_swiglu":
        raise ValueError("optimizer smoke requires dense_swiglu")
    model = _new_model(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    _assert_dense_structure(model, expected_layers=config.n_layers)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    ffn = model.blocks[0].ffn
    if ffn is None:
        raise ValueError("dense optimizer smoke has no layer-0 FFN")
    trainable = (
        ffn.gate_proj.weight,
        ffn.up_proj.weight,
        ffn.down_proj.weight,
    )
    for parameter in trainable:
        parameter.requires_grad_(True)

    model.to(dtype=torch.bfloat16)
    if not model.enable_dual_gpu(gpu0=0, gpu1=1):
        raise RuntimeError("dense optimizer smoke requires both GPUs")
    model.train()
    optimizer = torch.optim.SGD(trainable, lr=0.05)
    before = [parameter.detach().clone() for parameter in trainable]
    tokens = torch.tensor(
        [[1, 451, 673, 1042, 224, 982]],
        dtype=torch.long,
        device="cuda:0",
    )
    optimizer.zero_grad(set_to_none=True)
    result = model(
        tokens,
        labels=tokens,
        heartbeat=False,
    )
    if result.loss is None or not torch.isfinite(result.loss):
        raise ValueError("dense optimizer smoke produced non-finite loss")
    result.loss.backward()
    gate_grad_nonzero = _nonzero_gradient(ffn.gate_proj.weight)
    up_grad_nonzero = _nonzero_gradient(ffn.up_proj.weight)
    down_grad_nonzero = _nonzero_gradient(ffn.down_proj.weight)
    grad_norms = {
        "gate_proj": float(ffn.gate_proj.weight.grad.float().norm().item()),
        "up_proj": float(ffn.up_proj.weight.grad.float().norm().item()),
        "down_proj": float(ffn.down_proj.weight.grad.float().norm().item()),
    }
    optimizer.step()
    weights_changed = all(
        bool(torch.not_equal(previous, current).any().item())
        for previous, current in zip(before, trainable, strict=True)
    )
    report = {
        "schema": "darwin-smol-dense-optimizer-smoke-v1",
        "checkpoint": str(Path(checkpoint).resolve()),
        "feed_forward_kind": config.feed_forward_kind,
        "finite_loss": True,
        "loss": float(result.loss.detach().float().item()),
        "gate_grad_nonzero": gate_grad_nonzero,
        "up_grad_nonzero": up_grad_nonzero,
        "down_grad_nonzero": down_grad_nonzero,
        "grad_norms": grad_norms,
        "weights_changed": weights_changed,
        "dual_gpu": True,
        "split_layer": int(model._split_layer),
    }
    required = (
        gate_grad_nonzero,
        up_grad_nonzero,
        down_grad_nonzero,
        weights_changed,
    )
    if not all(required):
        raise ValueError(f"dense optimizer smoke failed: {report}")
    _atomic_json(Path(output), report)
    del optimizer, model, payload
    gc.collect()
    torch.cuda.empty_cache()
    return report
