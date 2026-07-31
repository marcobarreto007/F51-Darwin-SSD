from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F


@dataclass
class ExperienceSlot:
    signature: torch.Tensor
    indices: torch.Tensor
    values: torch.Tensor
    vocab_size: int
    observations: int = 1


class ExperienceMemory:
    def __init__(
        self,
        *,
        capacity: int,
        top_k: int,
        threshold: float,
        min_observations: int,
    ) -> None:
        self.capacity = int(capacity)
        self.top_k = int(top_k)
        self.threshold = float(threshold)
        self.min_observations = int(min_observations)
        self.slots: list[ExperienceSlot] = []
        self.frozen = False

    def __len__(self) -> int:
        return len(self.slots)

    @staticmethod
    def normalize(signature: torch.Tensor) -> torch.Tensor:
        return F.normalize(signature.detach().float().cpu().flatten(), dim=0)

    def write(self, signature: torch.Tensor, logits: torch.Tensor) -> None:
        if self.frozen:
            return
        signature = self.normalize(signature)
        logits = logits.detach().float().cpu().flatten()
        values, indices = torch.topk(logits, min(self.top_k, logits.numel()))
        if self.slots:
            similarities = torch.tensor(
                [torch.dot(slot.signature, signature) for slot in self.slots]
            )
            best = int(similarities.argmax())
            if float(similarities[best]) >= self.threshold:
                slot = self.slots[best]
                count = slot.observations
                slot.values = (slot.values * count + values) / (count + 1)
                slot.indices = indices
                slot.observations += 1
                return
        self.slots.append(
            ExperienceSlot(signature, indices, values, logits.numel())
        )
        if len(self.slots) > self.capacity:
            self.slots.pop(0)

    def write_target(
        self,
        signature: torch.Tensor,
        *,
        target_id: int,
        vocab_size: int,
        reward: float = 12.0,
    ) -> None:
        if target_id < 0 or target_id >= vocab_size:
            raise ValueError("target token is outside the vocabulary")
        logits = torch.full((vocab_size,), -float(reward))
        logits[target_id] = float(reward)
        self.write(signature, logits)

    def retrieve(self, signature: torch.Tensor) -> tuple[torch.Tensor, float] | None:
        if not self.slots:
            return None
        signature = self.normalize(signature)
        similarities = torch.tensor(
            [torch.dot(slot.signature, signature) for slot in self.slots]
        )
        best = int(similarities.argmax())
        confidence = float(similarities[best])
        slot = self.slots[best]
        if confidence < self.threshold or slot.observations < self.min_observations:
            return None
        logits = torch.full((slot.vocab_size,), -torch.inf)
        logits[slot.indices] = slot.values
        return logits, confidence

    def freeze(self) -> None:
        self.frozen = True

    def reset(self) -> None:
        self.slots.clear()

    def shuffle(self, *, seed: int) -> None:
        generator = random.Random(seed)
        values = [(slot.indices, slot.values) for slot in self.slots]
        generator.shuffle(values)
        for slot, (indices, logits) in zip(self.slots, values, strict=True):
            slot.indices = indices
            slot.values = logits

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        torch.save(
            {
                "config": {
                    "capacity": self.capacity,
                    "top_k": self.top_k,
                    "threshold": self.threshold,
                    "min_observations": self.min_observations,
                },
                "frozen": self.frozen,
                "slots": self.slots,
            },
            temporary,
        )
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: Path) -> "ExperienceMemory":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        memory = cls(**payload["config"])
        memory.frozen = bool(payload["frozen"])
        memory.slots = payload["slots"]
        return memory
