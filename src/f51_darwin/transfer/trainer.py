from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .student import DarwinTransferStudent


class TransferTrainer:
    def __init__(
        self,
        teacher: nn.Module,
        student: DarwinTransferStudent,
        *,
        learning_rate: float,
        metrics_path: Path,
        temperature: float = 2.0,
        metric_context: dict | None = None,
        eval_microbatch_size: int = 16,
    ) -> None:
        self.teacher = teacher.eval()
        self.student = student
        self.learning_rate = float(learning_rate)
        self.metrics_path = metrics_path
        self.temperature = float(temperature)
        self.metric_context = dict(metric_context or {})
        self.eval_microbatch_size = int(eval_microbatch_size)
        if self.eval_microbatch_size <= 0:
            raise ValueError("eval_microbatch_size must be positive")
        self.step = 0
        self.stage: str | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

    def _loss(self, stage: str, input_ids: torch.Tensor) -> torch.Tensor:
        needs_hidden_states = stage in {"orientation", "alignment"}
        with torch.no_grad():
            teacher = self.teacher(
                input_ids=input_ids,
                output_hidden_states=needs_hidden_states,
            )
        if stage == "orientation":
            losses = []
            for index in self.student.replaced_layers:
                hidden = self.teacher.transformer.h[index].ln_1(
                    teacher.hidden_states[index]
                )
                attention = self.teacher.transformer.h[index].attn
                query, key, _ = attention.c_attn(hidden).split(
                    attention.split_size, dim=2
                )
                batch, sequence, width = query.shape
                query = query.view(
                    batch, sequence, attention.num_heads, attention.head_dim
                ).transpose(1, 2)
                key = key.view(
                    batch, sequence, attention.num_heads, attention.head_dim
                ).transpose(1, 2)
                target = query @ key.transpose(-1, -2)
                if attention.scale_attn_weights:
                    target = target / math.sqrt(attention.head_dim)
                causal = torch.ones(
                    sequence,
                    sequence,
                    dtype=torch.bool,
                    device=hidden.device,
                ).tril()
                target = torch.softmax(
                    target.masked_fill(~causal, torch.finfo(target.dtype).min),
                    dim=-1,
                )
                matrix = self.student.ssd_attention(index).mixer.materialize_mixer(
                    hidden
                )
                losses.append(
                    F.mse_loss(
                        torch.softmax(
                            matrix.masked_fill(
                                ~causal, torch.finfo(matrix.dtype).min
                            ),
                            dim=-1,
                        ),
                        target,
                    )
                )
            return torch.stack(losses).mean()
        student = self.student(
            input_ids=input_ids,
            output_hidden_states=stage == "alignment",
        )
        if stage == "alignment":
            return torch.stack(
                [
                    1.0
                    - F.cosine_similarity(
                        student.hidden_states[index + 1].flatten(1),
                        teacher.hidden_states[index + 1].flatten(1),
                    ).mean()
                    for index in self.student.replaced_layers
                ]
            ).mean()
        if stage == "distillation":
            temperature = self.temperature
            kl = F.kl_div(
                F.log_softmax(student.logits / temperature, dim=-1),
                F.softmax(teacher.logits / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature * temperature / input_ids.shape[1])
            lm = F.cross_entropy(
                student.logits[:, :-1].reshape(-1, student.logits.shape[-1]),
                input_ids[:, 1:].reshape(-1),
            )
            return 0.8 * kl + 0.2 * lm
        raise ValueError(f"unsupported stage: {stage}")

    def train_step(self, stage: str, input_ids: torch.Tensor) -> float:
        if self.stage != stage or self.optimizer is None:
            self.configure_stage(stage)
        assert self.optimizer is not None
        parameters = [p for p in self.student.parameters() if p.requires_grad]
        self.student.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = self._loss(stage, input_ids)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite {stage} loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        self.optimizer.step()
        self.step += 1
        value = float(loss.detach())
        self._append_metric(
            {
                **self.metric_context,
                "event": "train",
                "step": self.step,
                "stage": stage,
                "loss": value,
            }
        )
        return value

    def configure_stage(self, stage: str) -> None:
        self.student.set_trainable_stage(stage)
        parameters = [p for p in self.student.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError(f"stage has no trainable parameters: {stage}")
        self.optimizer = torch.optim.AdamW(parameters, lr=self.learning_rate)
        self.stage = stage

    def optimizer_state_dict(self) -> dict | None:
        return None if self.optimizer is None else self.optimizer.state_dict()

    @torch.no_grad()
    def measure_loss(self, stage: str, input_ids: torch.Tensor) -> float:
        self.student.eval()
        value = self._loss(stage, input_ids)
        if not torch.isfinite(value):
            raise FloatingPointError(f"non-finite {stage} evaluation loss")
        return float(value)

    @torch.no_grad()
    def evaluate_nll(self, batches: list[torch.Tensor], device: torch.device) -> float:
        self.student.eval()
        losses = []
        for start in range(0, len(batches), self.eval_microbatch_size):
            window = batches[start : start + self.eval_microbatch_size]
            ids = (
                window
                if isinstance(window, torch.Tensor)
                else torch.stack(window)
            ).to(device=device, dtype=torch.long)
            logits = self.student(input_ids=ids).logits
            losses.append(
                F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]),
                    ids[:, 1:].reshape(-1),
                    reduction="none",
                ).view(ids.shape[0], -1).mean(dim=1)
            )
        return float(torch.cat(losses).mean())

    @torch.no_grad()
    def evaluate_teacher_nll(
        self,
        batches: list[torch.Tensor],
        device: torch.device,
    ) -> float:
        self.teacher.eval()
        losses = []
        for start in range(0, len(batches), self.eval_microbatch_size):
            window = batches[start : start + self.eval_microbatch_size]
            ids = (
                window
                if isinstance(window, torch.Tensor)
                else torch.stack(window)
            ).to(device=device, dtype=torch.long)
            logits = self.teacher(input_ids=ids).logits
            losses.append(
                F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]),
                    ids[:, 1:].reshape(-1),
                    reduction="none",
                ).view(ids.shape[0], -1).mean(dim=1)
            )
        return float(torch.cat(losses).mean())

    def _append_metric(self, payload: dict) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def log_metric(self, payload: dict) -> None:
        self._append_metric(payload)
