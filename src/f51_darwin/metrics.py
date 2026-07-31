from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from f51_darwin.replay_buffer import forgetting_proxy


@dataclass
class StepMetrics:
    step: int
    train_loss: float
    tokens: int
    tokens_per_sec: float
    vram_gb: float | None
    elapsed_sec: float


@dataclass
class EvalMetrics:
    step: int
    eval_loss: float
    replay_loss: float | None
    forgetting_proxy: float | None


@dataclass
class TrainingMetricsTracker:
    run_id: str
    started_at: float = field(default_factory=time.time)
    steps: list[StepMetrics] = field(default_factory=list)
    evals: list[EvalMetrics] = field(default_factory=list)
    last_replay_loss: float | None = None

    def record_train_step(
        self,
        *,
        step: int,
        train_loss: float,
        tokens: int,
        elapsed_sec: float,
        device: torch.device,
    ) -> StepMetrics:
        tokens_per_sec = float(tokens) / elapsed_sec if elapsed_sec > 0 else 0.0
        vram_gb = None
        if device.type == "cuda" and torch.cuda.is_available():
            vram_gb = round(torch.cuda.max_memory_allocated(device) / 1e9, 4)
        metric = StepMetrics(
            step=step,
            train_loss=float(train_loss),
            tokens=int(tokens),
            tokens_per_sec=round(tokens_per_sec, 2),
            vram_gb=vram_gb,
            elapsed_sec=round(elapsed_sec, 4),
        )
        self.steps.append(metric)
        return metric

    def record_eval(
        self,
        *,
        step: int,
        eval_loss: float,
        replay_loss: float | None,
    ) -> EvalMetrics:
        forgetting = None
        if replay_loss is not None and self.last_replay_loss is not None:
            forgetting = forgetting_proxy(self.last_replay_loss, replay_loss)
        if replay_loss is not None:
            self.last_replay_loss = replay_loss
        metric = EvalMetrics(
            step=step,
            eval_loss=float(eval_loss),
            replay_loss=None if replay_loss is None else float(replay_loss),
            forgetting_proxy=None if forgetting is None else float(forgetting),
        )
        self.evals.append(metric)
        return metric

    def latest_train(self) -> StepMetrics | None:
        return self.steps[-1] if self.steps else None

    def latest_eval(self) -> EvalMetrics | None:
        return self.evals[-1] if self.evals else None

    def summary(self) -> dict[str, Any]:
        latest_train = self.latest_train()
        latest_eval = self.latest_eval()
        return {
            "run_id": self.run_id,
            "elapsed_sec": round(time.time() - self.started_at, 2),
            "train_loss": None if latest_train is None else latest_train.train_loss,
            "eval_loss": None if latest_eval is None else latest_eval.eval_loss,
            "replay_loss": None if latest_eval is None else latest_eval.replay_loss,
            "forgetting_proxy": None if latest_eval is None else latest_eval.forgetting_proxy,
            "tokens_per_sec": None if latest_train is None else latest_train.tokens_per_sec,
            "vram_gb": None if latest_train is None else latest_train.vram_gb,
            "steps_logged": len(self.steps),
            "evals_logged": len(self.evals),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "steps": [asdict(item) for item in self.steps],
            "evals": [asdict(item) for item in self.evals],
            "summary": self.summary(),
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output
