#!/usr/bin/env python3
"""Controlled C0/C1/F51 continual-learning experiment.

One process owns both GPUs. A Windows file lock prevents duplicate launches,
and every run writes line-buffered logs plus atomic partial/final results.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import msvcrt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

from f51_darwin.darwin_x import DarwinXConfig, DarwinXModel
from f51_darwin.dataset_layout import TOKENIZED_RELATIVE, resolve_dataset_path
from f51_darwin.tokenizer import F51BPETokenizer


BASE_YAML = ROOT / "src" / "configs" / "darwin_x_600m_ckpt239.yaml"
MATH_BIN = "MATEMATICA/tokens_openwebmath2.bin"
LIT_BIN = "CLASSICOS/tokens_classical.bin"
DEFAULT_BLOCK = 32


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


class Tee:
    def __init__(self, terminal: TextIO, log: TextIO) -> None:
        self.terminal = terminal
        self.log = log

    def write(self, data: str) -> int:
        self.terminal.write(data)
        self.terminal.flush()
        self.log.write(data)
        self.log.flush()
        return len(data)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def isatty(self) -> bool:
        return False


class ExperimentAlreadyRunning(RuntimeError):
    pass


class SingleRunLock(AbstractContextManager["SingleRunLock"]):
    """OS-owned lock: it is released automatically if the process crashes."""

    def __init__(self, path: Path, metadata: dict[str, Any]) -> None:
        self.path = path
        self.metadata = metadata
        self.handle: Any = None

    def __enter__(self) -> "SingleRunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        self.handle.seek(0)
        if self.path.stat().st_size == 0:
            self.handle.write(b"\0")
            self.handle.flush()
            self.handle.seek(0)
        try:
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            self.handle.close()
            raise ExperimentAlreadyRunning(
                f"another experiment_runner is active; lock={self.path}"
            ) from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps(self.metadata).encode("utf-8"))
        self.handle.flush()
        self.handle.seek(0)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            self.handle.seek(0)
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            self.handle.close()


@torch.no_grad()
def perplexity(model: DarwinXModel, tokens: np.ndarray, block: int, samples: int) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    device = torch.device("cuda:0")
    for _ in range(samples):
        start = torch.randint(0, max(1, len(tokens) - block - 1), (1,)).item()
        batch = torch.as_tensor(
            np.array(tokens[start : start + block], copy=True),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        output = model(batch, labels=batch)
        if output.lm_loss is not None and torch.isfinite(output.lm_loss):
            total_loss += output.lm_loss.item() * block
            total_tokens += block
    model.train()
    return math.exp(total_loss / total_tokens) if total_tokens else float("inf")


def train_phase(
    model: DarwinXModel,
    tokens: np.ndarray,
    *,
    steps: int,
    block: int,
    replay: bool,
    ghost_tokens: np.ndarray | None = None,
) -> tuple[list[float], list[torch.Tensor]]:
    device = torch.device("cuda:0")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=0.01, fused=True
    )
    losses: list[float] = []
    replay_buffer: list[torch.Tensor] = []
    progress_every = max(1, steps // 10)

    for step in range(steps):
        start = torch.randint(0, max(1, len(tokens) - block - 1), (1,)).item()
        fresh_batch = torch.as_tensor(
            np.array(tokens[start : start + block], copy=True),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        batch = fresh_batch
        if replay and len(replay_buffer) > 4 and step % 5 == 0:
            replay_index = torch.randint(0, len(replay_buffer), (1,)).item()
            batch = torch.cat([fresh_batch, replay_buffer[replay_index].to(device)], dim=0)

        if ghost_tokens is not None and step % 50 == 0 and len(ghost_tokens) > block:
            ghost_start = torch.randint(0, len(ghost_tokens) - block - 1, (1,)).item()
            ghost_batch = torch.as_tensor(
                np.array(ghost_tokens[ghost_start : ghost_start + block], copy=True),
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            with torch.no_grad():
                ghost_output = model(ghost_batch, labels=ghost_batch)
                ghost_loss = (
                    ghost_output.lm_loss.item()
                    if ghost_output.lm_loss is not None
                    else 10.0
                )
            for layer in model.blocks:
                layer.moe._ghost_loss = ghost_loss

        output = model(batch, labels=batch)
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(output.loss.item())

        # Store only the fresh sample. Storing an already replay-mixed batch makes
        # future batches grow recursively and eventually exhausts RAM/VRAM.
        if replay and step % 10 == 0:
            replay_buffer.append(fresh_batch.detach().cpu())
            if len(replay_buffer) > 128:
                replay_buffer.pop(0)

        if (step + 1) % progress_every == 0 or step + 1 == steps:
            recent = float(np.mean(losses[-progress_every:]))
            print(
                f"    step {step + 1}/{steps} loss={recent:.4f} "
                f"replay={len(replay_buffer)}",
                flush=True,
            )
    return losses, replay_buffer


def run_experiment(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    print("=" * 60)
    print("  DARWIN-X CONTROLLED EXPERIMENT: C0 / C1 / F51")
    print("=" * 60)
    print(f"  run_id={args.run_id} pid={os.getpid()}")
    print(f"  config={BASE_YAML}")
    print(f"  steps_per_phase={args.steps} seeds={args.seeds} block={args.block}")

    tokenizer = F51BPETokenizer.load(ROOT / "tokenizer" / "f51_bpe_80k")
    dataset_root = resolve_dataset_path(ROOT, TOKENIZED_RELATIVE, require=True)
    math_tokens = np.memmap(dataset_root / MATH_BIN, dtype=np.int32, mode="r")
    lit_tokens = np.memmap(dataset_root / LIT_BIN, dtype=np.int32, mode="r")
    validation_math = math_tokens[-500_000:]
    validation_lit = lit_tokens[-500_000:]
    print(f"  tokenizer_vocab={tokenizer.vocab_size} dataset={dataset_root}")

    results: dict[str, Any] = {
        "run_id": args.run_id,
        "pid": os.getpid(),
        "started_at": datetime.now().astimezone().isoformat(),
        "config": str(BASE_YAML),
        "steps_per_phase": args.steps,
        "block": args.block,
        "seeds": args.seeds,
        "conditions": {},
    }

    for name, replay in (("C0_Static", False), ("C1_Replay", True), ("F51_Full", True)):
        print(f"\n--- {name} ---", flush=True)
        condition_results: list[dict[str, Any]] = []
        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            raw = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
            if name == "C0_Static":
                raw.update(
                    heartbeat_enabled=False,
                    ghost_mask_ratio=0.0,
                    mtp_weight=0.0,
                    jepa_weight=0.0,
                    ghost_weight=0.0,
                )
            config = DarwinXConfig.from_mapping(raw)
            model = DarwinXModel(config).to(dtype=torch.bfloat16)
            if torch.cuda.device_count() < 2:
                raise RuntimeError("this experiment requires both GPUs")
            split = model.recommended_dual_gpu_split(gpu0=0, gpu1=1)
            if not model.enable_dual_gpu(gpu0=0, gpu1=1, split_layer=split):
                raise RuntimeError("dual-GPU model placement failed")
            print(f"  seed={seed} dual_gpu_split={split}/{config.n_layers - split}")
            if name == "C0_Static":
                for layer in model.blocks:
                    layer.moe.nitro_enabled = False

            started = time.time()
            train_phase(
                model,
                math_tokens,
                steps=args.steps,
                block=args.block,
                replay=replay,
            )
            if name == "F51_Full":
                model.apply_pending_autonomic_actions()
                model.execute_structural_actions()
            math_before = perplexity(
                model, validation_math, args.block, args.eval_samples
            )
            ghost_sample = math_tokens[:100_000] if name == "F51_Full" else None
            train_phase(
                model,
                lit_tokens,
                steps=args.steps,
                block=args.block,
                replay=replay,
                ghost_tokens=ghost_sample,
            )
            if name == "F51_Full":
                model.apply_pending_autonomic_actions()
                model.execute_structural_actions()
            math_after = perplexity(
                model, validation_math, args.block, args.eval_samples
            )
            lit_after = perplexity(
                model, validation_lit, args.block, args.eval_samples
            )
            forgetting_ratio = math_after / max(math_before, 1.0)
            elapsed = time.time() - started
            record = {
                "seed": seed,
                "forgetting_ratio": forgetting_ratio,
                "ppl_math_before": math_before,
                "ppl_math_after": math_after,
                "ppl_lit_after": lit_after,
                "elapsed_seconds": elapsed,
                "dual_gpu_split": [split, config.n_layers - split],
            }
            condition_results.append(record)
            print(
                f"  seed={seed} FR={forgetting_ratio:.3f} "
                f"M:{math_before:.0f}>{math_after:.0f} "
                f"L:{lit_after:.0f} [{elapsed:.0f}s]",
                flush=True,
            )
            del model
            torch.cuda.empty_cache()

        ratios = [item["forgetting_ratio"] for item in condition_results]
        results["conditions"][name] = condition_results
        results[f"{name}_summary"] = {
            "forgetting_ratio_mean": float(np.mean(ratios)),
            "forgetting_ratio_std": float(np.std(ratios)),
            "seed_count": len(ratios),
        }
        atomic_json(run_dir / "partial_results.json", results)
        print(f"  FR={np.mean(ratios):.3f}+/-{np.std(ratios):.3f}")

    results["finished_at"] = datetime.now().astimezone().isoformat()
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--block", type=int, default=DEFAULT_BLOCK)
    parser.add_argument("--eval-samples", type=int, default=30)
    parser.add_argument("--seeds", default="51")
    parser.add_argument("--run-id")
    parsed = parser.parse_args()
    parsed.seeds = [int(value.strip()) for value in parsed.seeds.split(",") if value.strip()]
    if not parsed.seeds:
        parser.error("--seeds must contain at least one integer")
    if parsed.steps < 1 or parsed.block < 2 or parsed.eval_samples < 1:
        parser.error("steps/eval-samples must be positive and block must be >= 2")
    parsed.run_id = parsed.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    return parsed


def main() -> int:
    args = parse_args()
    run_dir = ROOT / "runs" / "experiments" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    log_handle = (run_dir / "experiment.log").open("a", encoding="utf-8", buffering=1)
    terminal_stdout = sys.__stdout__
    terminal_stderr = sys.__stderr__
    sys.stdout = Tee(terminal_stdout, log_handle)
    sys.stderr = Tee(terminal_stderr, log_handle)
    status_path = run_dir / "status.json"
    metadata = {"pid": os.getpid(), "run_id": args.run_id, "run_dir": str(run_dir)}
    atomic_json(status_path, {**metadata, "status": "starting"})

    try:
        with SingleRunLock(ROOT / "runs" / "experiment_runner.lock", metadata):
            atomic_json(status_path, {**metadata, "status": "running"})
            results = run_experiment(args, run_dir)
            atomic_json(run_dir / "results.json", results)
            atomic_json(
                ROOT / "runs" / "experiments" / "latest.json",
                {**metadata, "status": "completed", "results": str(run_dir / "results.json")},
            )
            atomic_json(status_path, {**metadata, "status": "completed"})
            print(f"\nSaved: {run_dir / 'results.json'}")
        return 0
    except Exception as exc:
        failure = {
            **metadata,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_json(status_path, failure)
        if not isinstance(exc, ExperimentAlreadyRunning):
            atomic_json(ROOT / "runs" / "experiments" / "latest.json", failure)
            traceback.print_exc()
        else:
            print(str(exc), file=sys.stderr)
        return 1
    finally:
        sys.stdout = terminal_stdout
        sys.stderr = terminal_stderr
        log_handle.flush()
        log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
