from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import torch
import yaml

from f51_darwin.artifacts import resolve_latest_checkpoint
from f51_darwin.config import DarwinConfig
from f51_darwin.data import (
    CausalLMDataLoader,
    corpus_stats,
    load_text_documents,
    resolve_corpus_dir,
    tokenize_documents,
)
from f51_darwin.model import F51DarwinModel
from f51_darwin.tokenizer import F51BPETokenizer
from f51_darwin.training import BaseTrainer, BaseTrainingConfig


DEFAULT_BASE_TRAINING_CONFIG = {
    "corpus_dir": "data/corpus",
    "tokenizer_dir": "tokenizer/f51_bpe_80k",
    "checkpoint_dir": "checkpoints/base",
    "training": {
        "batch_size": 4,
        "block_size": 1024,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "max_steps": 5000,
        "eval_every": 500,
        "save_every": 500,
        "grad_clip": 1.0,
        "seed": 51,
    },
    "replay": {
        "capacity": 512,
        "sample_size": 16,
        "seed_every": 1,
    },
    "smoke": {
        "batch_size": 2,
        "block_size": 64,
        "max_steps": 20,
        "eval_every": 10,
        "save_every": 10,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train F51 Darwin-SSD from scratch on a local corpus.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional training runtime YAML. Inline defaults are used when omitted.",
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="Model seed config YAML. Defaults to model_config in base config.",
    )
    parser.add_argument("--corpus", default=None, help="Override corpus directory.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer directory to load.")
    parser.add_argument("--resume", default=None, help="Checkpoint path or 'latest'.")
    parser.add_argument("--run-id", default=None, help="Run identifier for versioned checkpoints.")
    parser.add_argument("--smoke", action="store_true", help="Short smoke run using smoke settings.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps.")
    parser.add_argument("--device", default=None, help="cpu or cuda. Defaults to auto.")
    parser.add_argument(
        "--allow-debug-candidates",
        action="store_true",
        help="DEBUG ONLY: allow training from data/generated/candidates/.",
    )
    return parser


def load_training_config(path: str | None) -> dict:
    if path is None:
        return dict(DEFAULT_BASE_TRAINING_CONFIG)
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Training config must be a mapping: {config_path}")
    return raw


def resolve_resume_path(resume: str, checkpoint_dir: Path) -> Path:
    if resume == "latest":
        checkpoint = resolve_latest_checkpoint(
            ROOT,
            pointers=[checkpoint_dir / "latest.json"],
            search_roots=[checkpoint_dir],
        )
        if checkpoint is None:
            raise FileNotFoundError("No usable latest checkpoint found.")
        return checkpoint
    candidate = Path(resume)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if not candidate.exists():
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")
    return candidate


def apply_smoke_overrides(raw: dict) -> dict:
    merged = dict(raw)
    smoke = dict(raw.get("smoke", {}))
    training = dict(raw.get("training", {}))
    training.update(
        {
            "batch_size": smoke.get("batch_size", 2),
            "block_size": smoke.get("block_size", 64),
            "max_steps": smoke.get("max_steps", 20),
            "eval_every": smoke.get("eval_every", 10),
            "save_every": smoke.get("save_every", 10),
        }
    )
    merged["training"] = training
    return merged


def main() -> int:
    args = build_parser().parse_args()
    raw_config = load_training_config(args.config)
    if args.smoke:
        raw_config = apply_smoke_overrides(raw_config)

    training_config = BaseTrainingConfig.from_mapping(raw_config)
    if args.max_steps is not None:
        training_config = replace(training_config, max_steps=args.max_steps)
    if args.run_id:
        training_config = replace(training_config, run_id=args.run_id)
    elif not training_config.run_id:
        training_config = replace(
            training_config,
            run_id=datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S"),
        )

    if args.model_config or raw_config.get("model_config"):
        model_config_path = Path(args.model_config or raw_config["model_config"])
        if not model_config_path.is_absolute():
            model_config_path = ROOT / model_config_path
        base_model_config = DarwinConfig.from_yaml(model_config_path)
    else:
        base_model_config = DarwinConfig()

    corpus_dir = resolve_corpus_dir(
        ROOT,
        args.corpus or raw_config.get("corpus_dir"),
        debug_candidates=args.allow_debug_candidates,
    )
    tokenizer_dir = Path(args.tokenizer or raw_config.get("tokenizer_dir", "tokenizer/f51_bpe_80k"))
    if not tokenizer_dir.is_absolute():
        tokenizer_dir = ROOT / tokenizer_dir
    if not tokenizer_dir.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_dir}. Run: python research/train_tokenizer.py"
        )

    tokenizer = F51BPETokenizer.load(tokenizer_dir)
    documents = load_text_documents(corpus_dir)
    token_ids = tokenize_documents(documents, tokenizer)
    stats = corpus_stats(corpus_dir, token_ids)

    model_config = replace(
        base_model_config,
        vocab_size=tokenizer.vocab_size,
        context_length=max(
            training_config.block_size,
            min(base_model_config.context_length, training_config.block_size)
            if args.smoke
            else base_model_config.context_length,
        ),
    )

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(training_config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training_config.seed)

    data_loader = CausalLMDataLoader(
        token_ids,
        block_size=training_config.block_size,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
        device=device,
    )
    model = F51DarwinModel(model_config).to(device)
    checkpoint_dir = ROOT / training_config.checkpoint_dir

    if args.resume:
        checkpoint_path = resolve_resume_path(args.resume, checkpoint_dir)
        trainer = BaseTrainer.resume_from_checkpoint(
            checkpoint_path,
            project_root=ROOT,
            model=model,
            model_config=model_config,
            training_config=training_config,
            data_loader=data_loader,
            device=device,
            tokenizer_path=tokenizer_dir,
        )
    else:
        trainer = BaseTrainer(
            model=model,
            model_config=model_config,
            training_config=training_config,
            data_loader=data_loader,
            device=device,
            tokenizer_path=tokenizer_dir,
            project_root=ROOT,
        )

    summary = trainer.run()
    payload = {
        "mode": "smoke" if args.smoke else "base",
        "run_id": trainer.metrics.run_id,
        "corpus_dir": str(corpus_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "corpus_stats": stats.__dict__,
        "model_config": model_config.model_name,
        "params": sum(p.numel() for p in model.parameters()),
        "resume": args.resume,
        "summary": summary,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
