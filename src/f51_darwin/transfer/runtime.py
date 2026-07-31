from __future__ import annotations

from dataclasses import dataclass
import copy

import torch

from .experience import ExperienceMemory


@dataclass(frozen=True)
class Prediction:
    logits: torch.Tensor
    path: str
    confidence: float


class DarwinTransferRuntime:
    def __init__(self, backbone, embedding, memory: ExperienceMemory) -> None:
        self.backbone = backbone
        self.embedding = embedding
        self.memory = memory
        self.backbone_calls = 0
        self.experience_hits = 0

    def signature(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.embedding(input_ids).mean(dim=(0, 1))

    def predict_next(self, input_ids: torch.Tensor, *, learn: bool) -> Prediction:
        signature = self.signature(input_ids)
        recalled = self.memory.retrieve(signature)
        if recalled is not None:
            logits, confidence = recalled
            self.experience_hits += 1
            return Prediction(logits.to(input_ids.device), "experience", confidence)
        output = self.backbone(input_ids=input_ids)
        logits = output.logits[0, -1].detach()
        self.backbone_calls += 1
        if learn:
            self.memory.write(signature, logits)
        return Prediction(logits, "backbone", 0.0)

    def observe(
        self,
        input_ids: torch.Tensor,
        *,
        target_id: int,
        vocab_size: int,
    ) -> None:
        self.memory.write_target(
            self.signature(input_ids),
            target_id=target_id,
            vocab_size=vocab_size,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        learn: bool = True,
    ) -> tuple[torch.Tensor, list[str]]:
        generated = input_ids
        paths: list[str] = []
        for _ in range(max_new_tokens):
            prediction = self.predict_next(generated, learn=learn)
            next_id = prediction.logits.argmax().view(1, 1).to(generated.device)
            generated = torch.cat((generated, next_id), dim=1)
            paths.append(prediction.path)
        return generated, paths


def evaluate_experience_controls(
    backbone,
    embedding,
    examples: list[tuple[torch.Tensor, int]],
    *,
    vocab_size: int,
) -> dict[str, dict[str, float | int]]:
    base = ExperienceMemory(
        capacity=max(1, len(examples)),
        top_k=min(64, vocab_size),
        threshold=0.995,
        min_observations=1,
    )
    learner = DarwinTransferRuntime(backbone, embedding, base)
    for input_ids, target_id in examples:
        learner.observe(
            input_ids,
            target_id=target_id,
            vocab_size=vocab_size,
        )

    memories = {
        "correct": copy.deepcopy(base),
        "shuffled": copy.deepcopy(base),
        "frozen": ExperienceMemory(
            capacity=max(1, len(examples)),
            top_k=min(64, vocab_size),
            threshold=0.995,
            min_observations=1,
        ),
    }
    memories["shuffled"].shuffle(seed=51)
    memories["frozen"].freeze()
    results: dict[str, dict[str, float | int]] = {}
    for name, memory in memories.items():
        runtime = DarwinTransferRuntime(backbone, embedding, memory)
        correct = 0
        for input_ids, target_id in examples:
            prediction = runtime.predict_next(input_ids, learn=False)
            correct += int(int(prediction.logits.argmax()) == target_id)
        results[name] = {
            "accuracy": correct / max(1, len(examples)),
            "backbone_calls": runtime.backbone_calls,
            "experience_hits": runtime.experience_hits,
        }
    return results
