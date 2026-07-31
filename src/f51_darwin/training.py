from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from f51_darwin.checkpointing import load_training_checkpoint, save_training_checkpoint
from f51_darwin.config import DarwinConfig
from f51_darwin.data import CausalLMDataLoader
from f51_darwin.metrics import TrainingMetricsTracker
from f51_darwin.model import F51DarwinModel
from f51_darwin.replay_buffer import ReplayBuffer, ReplayExample


@dataclass(frozen=True)
class BaseTrainingConfig:
    batch_size: int
    block_size: int
    learning_rate: float
    weight_decay: float
    max_steps: int
    eval_every: int
    save_every: int
    grad_clip: float
    seed: int
    replay_capacity: int
    replay_sample_size: int
    replay_seed_every: int
    checkpoint_dir: str
    run_id: str | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "BaseTrainingConfig":
        training = raw.get("training", raw)
        replay = raw.get("replay", {})
        return cls(
            batch_size=int(training["batch_size"]),
            block_size=int(training["block_size"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            max_steps=int(training["max_steps"]),
            eval_every=int(training["eval_every"]),
            save_every=int(training["save_every"]),
            grad_clip=float(training["grad_clip"]),
            seed=int(training["seed"]),
            replay_capacity=int(replay.get("capacity", 512)),
            replay_sample_size=int(replay.get("sample_size", 16)),
            replay_seed_every=int(replay.get("seed_every", 50)),
            checkpoint_dir=str(raw.get("checkpoint_dir", "checkpoints/base")),
            run_id=raw.get("run_id"),
        )


@dataclass
class BaseTrainerState:
    step: int = 0
    run_id: str = ""


class BaseTrainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        model_config: DarwinConfig | Any,
        training_config: BaseTrainingConfig,
        data_loader: CausalLMDataLoader,
        device: torch.device,
        tokenizer_path: str | Path,
        project_root: Path,
        replay_buffer: ReplayBuffer | None = None,
        metrics: TrainingMetricsTracker | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        state: BaseTrainerState | None = None,
    ) -> None:
        self.model = model
        self.model_config = model_config
        self.training_config = training_config
        self.data_loader = data_loader
        self.device = device
        self.tokenizer_path = str(tokenizer_path)
        self.project_root = project_root
        self.replay_buffer = replay_buffer or ReplayBuffer(
            capacity=training_config.replay_capacity,
            seed=training_config.seed,
        )
        self.metrics = metrics or TrainingMetricsTracker(run_id=training_config.run_id or "base_run")
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        self.state = state or BaseTrainerState(
            step=0,
            run_id=self.metrics.run_id,
        )
        self.checkpoint_root = project_root / training_config.checkpoint_dir

    @property
    def step(self) -> int:
        return self.state.step

    def train_step(self, batch: torch.Tensor) -> float:
        import time

        self.model.train()
        start = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        end = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        wall_start = time.perf_counter()
        if start is not None:
            start.record()
        output = self.model(batch, labels=batch)
        assert output.loss is not None
        loss = output.loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.grad_clip)
        self.optimizer.step()
        apply_autonomic = getattr(self.model, "apply_pending_autonomic_actions", None)
        if callable(apply_autonomic):
            apply_autonomic()
        if end is not None and start is not None:
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) / 1000.0
        else:
            elapsed = time.perf_counter() - wall_start

        tokens = batch.numel()
        self.metrics.record_train_step(
            step=self.state.step + 1,
            train_loss=float(loss.detach().cpu()),
            tokens=tokens,
            elapsed_sec=elapsed,
            device=self.device,
        )
        return float(loss.detach().cpu())

    @torch.no_grad()
    def evaluate(self, batch: torch.Tensor) -> float:
        self.model.eval()
        output = self.model(batch, labels=batch)
        assert output.loss is not None
        return float(output.loss.detach().cpu())

    @torch.no_grad()
    def evaluate_replay(self) -> float | None:
        if len(self.replay_buffer) == 0:
            return None
        samples = self.replay_buffer.sample(min(self.training_config.replay_sample_size, len(self.replay_buffer)))
        if not samples:
            return None
        rows = []
        for example in samples:
            token_ids = list(example.input_ids)
            if len(token_ids) < self.training_config.block_size:
                repeats = (self.training_config.block_size // len(token_ids)) + 1
                token_ids = (token_ids * repeats)[: self.training_config.block_size]
            else:
                token_ids = token_ids[: self.training_config.block_size]
            rows.append(token_ids)
        batch = torch.tensor(rows, dtype=torch.long, device=self.device)
        return self.evaluate(batch)

    def seed_replay_from_batch(self, batch: torch.Tensor) -> None:
        first_row = batch[0].detach().cpu().tolist()
        self.replay_buffer.add(first_row, label=f"step_{self.state.step}")

    def maybe_seed_replay(self, batch: torch.Tensor) -> None:
        if self.state.step == 0 or self.state.step % self.training_config.replay_seed_every == 0:
            self.seed_replay_from_batch(batch)

    def run(
        self,
        *,
        max_steps: int | None = None,
        eval_every: int | None = None,
        save_every: int | None = None,
    ) -> dict[str, Any]:
        max_steps = self.training_config.max_steps if max_steps is None else max_steps
        eval_every = self.training_config.eval_every if eval_every is None else eval_every
        save_every = self.training_config.save_every if save_every is None else save_every
        self.data_loader.set_step(self.state.step)

        while self.state.step < max_steps:
            batch = self.data_loader.next_batch()
            self.train_step(batch)
            self.maybe_seed_replay(batch)
            self.state.step += 1

            if self.state.step % eval_every == 0:
                eval_batch = self.data_loader.take_eval_batch()
                eval_loss = self.evaluate(eval_batch)
                replay_loss = self.evaluate_replay()
                self.metrics.record_eval(
                    step=self.state.step,
                    eval_loss=eval_loss,
                    replay_loss=replay_loss,
                )

            if self.state.step % save_every == 0 or self.state.step == max_steps:
                self.save_checkpoint(versioned=True)

        report_path = self.project_root / "runs" / f"{self.metrics.run_id}_metrics.json"
        self.metrics.save(report_path)
        return self.metrics.summary()

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "step": self.state.step,
            "run_id": self.metrics.run_id,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "replay_buffer": self.replay_buffer.to_records(),
            "metrics": self.metrics.to_dict(),
            "training_config": asdict(self.training_config),
            "tokenizer_path": self.tokenizer_path,
            "data_loader_step": self.data_loader.step,
        }

    def save_checkpoint(self, *, versioned: bool = True) -> Path:
        if versioned:
            run_dir = self.checkpoint_root / self.metrics.run_id
            path = run_dir / f"step_{self.state.step:07d}.pt"
        else:
            path = self.checkpoint_root / "latest.pt"
        saved = save_training_checkpoint(
            path,
            self.model,
            self.model_config,
            training_state=self.checkpoint_payload(),
        )
        latest_pointer = self.checkpoint_root / "latest.json"
        latest_pointer.parent.mkdir(parents=True, exist_ok=True)
        latest_pointer.write_text(
            json.dumps({"path": str(saved.relative_to(self.project_root)), "step": self.state.step}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        return saved

    @classmethod
    def resume_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        project_root: Path,
        model: F51DarwinModel,
        model_config: DarwinConfig,
        training_config: BaseTrainingConfig,
        data_loader: CausalLMDataLoader,
        device: torch.device,
        tokenizer_path: str | Path,
    ) -> "BaseTrainer":
        payload = load_training_checkpoint(checkpoint_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        # CRÍTICO: garantir que todos os módulos (incluindo MoE experts) estão no dispositivo correto
        model = model.to(device)
        training_state = payload["training_state"]
        replay_buffer = ReplayBuffer(capacity=training_config.replay_capacity, seed=training_config.seed)
        replay_buffer.load_records(training_state.get("replay_buffer", []))
        metrics_data = training_state.get("metrics", {})
        metrics = TrainingMetricsTracker(run_id=training_state.get("run_id", "base_run"))
        metrics.started_at = metrics_data.get("started_at", metrics.started_at)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
        trainer = cls(
            model=model,
            model_config=model_config,
            training_config=training_config,
            data_loader=data_loader,
            device=device,
            tokenizer_path=tokenizer_path,
            project_root=project_root,
            replay_buffer=replay_buffer,
            metrics=metrics,
            optimizer=optimizer,
            state=BaseTrainerState(
                step=int(training_state.get("step", 0)),
                run_id=training_state.get("run_id", metrics.run_id),
            ),
        )
        trainer.data_loader.set_step(int(training_state.get("data_loader_step", trainer.state.step)))
        return trainer
