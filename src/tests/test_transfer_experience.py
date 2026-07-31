from pathlib import Path

import torch

from f51_darwin.transfer.experience import ExperienceMemory
from f51_darwin.transfer.runtime import (
    DarwinTransferRuntime,
    evaluate_experience_controls,
)


class CountingBackbone:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, input_ids):
        self.calls += 1
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 11)
        logits[..., 3] = 5.0
        return type("Output", (), {"logits": logits})


def test_memory_is_bounded_persistent_and_controllable(tmp_path: Path) -> None:
    memory = ExperienceMemory(capacity=2, top_k=3, threshold=0.9, min_observations=1)
    memory.write(torch.tensor([1.0, 0.0]), torch.arange(5.0))
    memory.write(torch.tensor([0.0, 1.0]), torch.arange(5.0).flip(0))
    memory.write(torch.tensor([0.7, 0.7]), torch.ones(5))
    assert len(memory) == 2
    path = tmp_path / "memory.pt"
    memory.save(path)
    restored = ExperienceMemory.load(path)
    assert len(restored) == 2
    restored.freeze()
    restored.write(torch.tensor([1.0, 0.0]), torch.ones(5))
    assert len(restored) == 2
    restored.shuffle(seed=3)
    restored.reset()
    assert len(restored) == 0


def test_runtime_really_bypasses_backbone() -> None:
    backbone = CountingBackbone()
    embedding = torch.nn.Embedding(11, 2)
    with torch.no_grad():
        embedding.weight.zero_()
        embedding.weight[1] = torch.tensor([1.0, 0.0])
    memory = ExperienceMemory(capacity=4, top_k=2, threshold=0.99, min_observations=1)
    runtime = DarwinTransferRuntime(backbone, embedding, memory)
    ids = torch.tensor([[1, 1]])
    first = runtime.predict_next(ids, learn=True)
    second = runtime.predict_next(ids, learn=True)
    assert first.path == "backbone"
    assert second.path == "experience"
    assert backbone.calls == 1
    assert runtime.backbone_calls == 1
    assert runtime.experience_hits == 1


def test_observed_next_token_changes_future_prediction() -> None:
    backbone = CountingBackbone()
    embedding = torch.nn.Embedding(11, 2)
    with torch.no_grad():
        embedding.weight.zero_()
        embedding.weight[1] = torch.tensor([1.0, 0.0])
    memory = ExperienceMemory(capacity=4, top_k=2, threshold=0.99, min_observations=1)
    runtime = DarwinTransferRuntime(backbone, embedding, memory)
    ids = torch.tensor([[1, 1]])
    assert int(runtime.predict_next(ids, learn=False).logits.argmax()) == 3
    runtime.observe(ids, target_id=7, vocab_size=11)
    learned = runtime.predict_next(ids, learn=False)
    assert learned.path == "experience"
    assert int(learned.logits.argmax()) == 7


def test_paired_controls_separate_learning_from_compute() -> None:
    backbone = CountingBackbone()
    embedding = torch.nn.Embedding(11, 2)
    with torch.no_grad():
        embedding.weight.zero_()
        embedding.weight[1] = torch.tensor([1.0, 0.0])
        embedding.weight[2] = torch.tensor([0.0, 1.0])
    results = evaluate_experience_controls(
        backbone,
        embedding,
        [
            (torch.tensor([[1, 1]]), 7),
            (torch.tensor([[2, 2]]), 8),
        ],
        vocab_size=11,
    )
    assert results["correct"]["accuracy"] == 1.0
    assert results["correct"]["backbone_calls"] == 0
    assert results["frozen"]["backbone_calls"] == 2
    assert results["shuffled"]["accuracy"] < results["correct"]["accuracy"]
