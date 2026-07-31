#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch

from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.transplant_16b.checkpoint import verify_shard_manifest
from f51_darwin.transplant_16b.tokenizer import SmolTokenizerAdapter


def _model(config: DarwinXConfig) -> DarwinXModel:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return DarwinXModel(config)
    finally:
        torch.set_default_dtype(previous)


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Donor-free native Darwin-Smol structural smoke."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokens", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    checkpoint = Path(args.checkpoint).resolve()
    manifest = Path(args.manifest).resolve()
    verify_shard_manifest(checkpoint, manifest)
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if payload.get("version") != 9:
        raise ValueError("native smoke requires checkpoint v9")
    transplant = payload.get("transplant", {})
    if not transplant.get("coverage_complete"):
        raise ValueError("transplant coverage is not complete")
    config = DarwinXConfig.from_mapping(payload["config"])
    model = _model(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    moe_modules = sum(
        int(block.moe is not None) for block in model.blocks
    )
    dense_ffn_modules = sum(
        int(block.ffn is not None) for block in model.blocks
    )
    if config.feed_forward_kind == "dense_swiglu":
        if moe_modules:
            raise ValueError("dense candidate contains a MoE block")
        if dense_ffn_modules != config.n_layers:
            raise ValueError(
                "dense candidate is missing native dense FFN blocks"
            )
        if any(".moe." in key for key in payload["model_state_dict"]):
            raise ValueError("dense checkpoint state contains a .moe. key")
    if model.lm_head.weight.data_ptr() != model.token_embedding.weight.data_ptr():
        raise ValueError("tied embedding/lm_head storage was lost")
    tokenizer = SmolTokenizerAdapter.load(args.tokenizer)
    dual_gpu = torch.cuda.is_available() and torch.cuda.device_count() >= 2
    if dual_gpu:
        model.to(dtype=torch.bfloat16)
        if not model.enable_dual_gpu(gpu0=0, gpu1=1):
            raise RuntimeError("dual-GPU placement was requested but refused")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    prompt = tokenizer.encode(
        "O futuro da inteligencia artificial",
        add_bos=True,
        add_eos=False,
    )
    generated = torch.tensor([prompt], dtype=torch.long, device=device)
    with torch.no_grad():
        first = model(generated).logits
        second = model(generated).logits
    if not torch.isfinite(first).all():
        raise ValueError("native forward produced non-finite logits")
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)
    for _ in range(args.tokens):
        with torch.no_grad():
            logits = model(generated[:, -config.context_length :]).logits
        next_token = logits[:, -1].float().argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
    decoded = tokenizer.decode(generated[0].tolist())
    if not decoded.strip():
        raise ValueError("native generation decoded to empty text")
    report = {
        "schema": "darwin-smol-native-smoke-v1",
        "checkpoint": str(checkpoint),
        "base_checkpoint_id": payload["base_checkpoint_id"],
        "tokens_generated": args.tokens,
        "finite": True,
        "roundtrip_logits_exact": True,
        "weight_tying": True,
        "donor_loaded": False,
        "dual_gpu": dual_gpu,
        "split_layer": model._split_layer if dual_gpu else None,
        "tokenizer_verified": True,
        "organs_verified": True,
        "feed_forward_kind": config.feed_forward_kind,
        "moe_modules": moe_modules,
        "dense_ffn_modules": dense_ffn_modules,
        "output": decoded,
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    print(
        "DARWIN_16B_SMOL_NATIVE_OK donor_loaded=false "
        "tokenizer=verified organs=verified"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
