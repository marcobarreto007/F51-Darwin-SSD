#!/usr/bin/env python3
"""Safe Darwin-X training entrypoint for cloud runs.

The script is intentionally fail-fast: no optimizer step is performed after a
non-finite loss/gradient, and checkpoints are saved only after a finite step.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

from f51_darwin.darwin_x import DarwinXConfig, DarwinXModel
from f51_darwin.darwin_x_training import (
    amp_settings,
    load_token_ids,
    train_step,
    validate_token_source,
)
from f51_darwin.data import CausalLMDataLoader
from f51_darwin.tokenizer import F51BPETokenizer
from f51_darwin.dataset_layout import resolve_darwin_x_checkpoints_root


def load_config(path: Path) -> DarwinXConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return DarwinXConfig.from_mapping(raw)


def small_smoke_config(vocab_size: int) -> DarwinXConfig:
    return DarwinXConfig(
        model_name="F51-Darwin-X-Smoke",
        vocab_size=vocab_size,
        context_length=32,
        inference_context_length=128,
        d_model=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        fine_experts=4,
        shared_experts=1,
        experts_per_token=2,
        fine_expert_hidden_dim=32,
        shared_expert_hidden_dim=32,
        mtp_depth=2,
    )


def save_darwin_x_checkpoint(
    path: Path,
    model: DarwinXModel,
    config: DarwinXConfig,
    *,
    step: int,
    metrics: dict[str, float],
    token_source: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "metrics": metrics,
        "training_state": {
            "step": step,
            "run_id": path.parent.name,
            "token_source": str(token_source),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "heartbeat_state": model.heartbeat_state_dict(),
        "lineage": "F51 Darwin-X random/adapted F51-owned lineage",
    }
    torch.save(payload, path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="F51 Darwin-X safe trainer")
    parser.add_argument("--config", default=str(ROOT / "src" / "configs" / "darwin_x_4b.yaml"))
    parser.add_argument("--token-bin", default=None)
    parser.add_argument("--token-index", default=None)
    parser.add_argument("--tokenizer", default=str(ROOT / "tokenizer" / "f51_bpe_80k"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument(
        "--checkpoint-dir",
        default=str(resolve_darwin_x_checkpoints_root(ROOT)),
        help="External checkpoint directory (defaults to F51-Dataset-Organizado/03_CHECKPOINTS/darwin_x).",
    )
    parser.add_argument("--resume", default=None, help="Darwin-X checkpoint only; not legacy F51DarwinModel.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/token source and exit.")
    parser.add_argument("--smoke-small", action="store_true", help="Run a tiny CPU/GPU architecture smoke.")
    parser.add_argument("--strict-tokenizer-vocab", action="store_true")
    args = parser.parse_args()

    tokenizer = F51BPETokenizer.load(Path(args.tokenizer))
    config = small_smoke_config(max(512, tokenizer.vocab_size)) if args.smoke_small else load_config(Path(args.config))
    block_size = args.block_size or min(256, config.context_length)
    if block_size > config.context_length:
        raise ValueError(f"block_size={block_size} exceeds context_length={config.context_length}")

    token_ids, token_path = load_token_ids(ROOT, token_bin=args.token_bin, token_index=args.token_index)
    token_health = validate_token_source(
        token_ids,
        token_path,
        config,
        tokenizer_vocab_size=tokenizer.vocab_size,
        strict_tokenizer_vocab=args.strict_tokenizer_vocab,
    )
    if len(token_ids) < block_size * max(args.batch_size, 1) + 1:
        raise ValueError("Token source is too small for the requested batch/block size.")

    report = {
        "config": asdict(config),
        "block_size": block_size,
        "batch_size": args.batch_size,
        "token_source": token_health.to_dict(),
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.dry_run:
        return 0

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

    amp_dtype, amp_enabled = amp_settings(device, args.precision)
    model = DarwinXModel(config)
    if amp_dtype is not None:
        model = model.to(dtype=amp_dtype)
    model = model.to(device)

    # 🔥 ATIVAR ORGANISMO COMPLETO — coração, curiosidade, ghost brain
    if config.heartbeat_enabled:
        from f51_darwin.curiosity import CuriosityDrive
        from f51_darwin.ghost_brain import GhostBrain
        curiosity = CuriosityDrive()
        ghost = GhostBrain(jepa=model.jepa_predictor)
        model.activate_organism(
            curiosity=curiosity,
            ghost_brain=ghost,
            jepa=model.jepa_predictor,
        )
        print(f"🧬 ORGANISMO ATIVO: coração + curiosidade + ghost brain + JEPA", flush=True)

    start_step = 0
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        if payload.get("config", {}).get("model_name") != config.model_name:
            raise ValueError("Resume checkpoint does not match Darwin-X config/model_name.")
        model.load_state_dict(payload["model_state_dict"], strict=False)
        model.load_heartbeat_state_dict(payload.get("heartbeat_state"))
        start_step = int(payload.get("training_state", {}).get("step", 0))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=(amp_enabled and amp_dtype == torch.float16))
    loader = CausalLMDataLoader(
        token_ids,
        block_size=block_size,
        batch_size=args.batch_size,
        seed=51,
        device=device,
    )
    loader.set_step(start_step)

    run_id = f"darwin_x_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = Path(args.checkpoint_dir) / run_id
    started = time.time()
    metrics: dict[str, float] = {}
    for local_step in range(1, args.steps + 1):
        batch = loader.next_batch()
        try:
            metrics = train_step(
                model,
                batch,
                optimizer,
                grad_clip=args.grad_clip,
                amp_dtype=amp_dtype,
                scaler=scaler,
            )
        except FloatingPointError:
            optimizer.zero_grad(set_to_none=True)
            raise
        step = start_step + local_step
        if step % 10 == 0 or local_step == 1:
            elapsed = max(time.time() - started, 1e-6)
            tok_s = local_step * args.batch_size * block_size / elapsed
            dead = metrics.get("moe_dead", "?")
            lb = metrics.get("moelb", "?")
            print(
                f"step={step} loss={metrics['loss']:.4f} "
                f"lm={metrics.get('lm_loss', 0.0):.4f} "
                f"mtp={metrics.get('mtp_loss', 0.0):.4f} "
                f"jepa={metrics.get('jepa_loss', 0.0):.4f} "
                f"aux={metrics.get('aux_loss', 0.0):.4f} "
                f"moelb={lb if isinstance(lb, str) else f'{lb:.2f}'} "
                f"dead={dead} "
                f"tk/s={tok_s:.0f}",
                flush=True,
            )
        if step % args.save_every == 0 or local_step == args.steps:
            path = checkpoint_dir / f"step_{step:07d}.pt"
            save_darwin_x_checkpoint(path, model, config, step=step, metrics=metrics, token_source=token_path)
            print(f"saved={path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
