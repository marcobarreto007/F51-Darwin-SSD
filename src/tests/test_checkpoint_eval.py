from __future__ import annotations

import gc
import struct
import weakref
from pathlib import Path

import torch

import f51_darwin.checkpoint_eval as checkpoint_eval
from f51_darwin.checkpoint_eval import CheckpointEvalConfig, compare_checkpoints
from f51_darwin.checkpointing import save_checkpoint
from f51_darwin.config import DarwinConfig
from f51_darwin.model import F51DarwinModel
from tools.compare_checkpoints import write_result


def _write_tokens(path: Path, values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack(f"{len(values)}i", *values))


def _train_model(model: F51DarwinModel, batch: torch.Tensor, *, steps: int) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch, labels=batch)
        assert output.loss is not None
        output.loss.backward()
        optimizer.step()


def test_compare_releases_baseline_before_loading_candidate(
    monkeypatch, tmp_path: Path
) -> None:
    active: list[weakref.ReferenceType[torch.nn.Module]] = []

    def fake_load(path, map_location):
        del path, map_location
        gc.collect()
        assert not [reference for reference in active if reference() is not None]
        model = F51DarwinModel(
            DarwinConfig(
                vocab_size=32,
                context_length=8,
                d_model=16,
                n_layers=4,
                n_heads=4,
            )
        )
        active.append(weakref.ref(model))
        return model, model.config, {}

    monkeypatch.setattr(checkpoint_eval, "load_model_from_checkpoint", fake_load)
    baseline = tmp_path / "baseline.pt"
    candidate = tmp_path / "candidate.pt"
    baseline.touch()
    candidate.touch()
    tokens = tmp_path / "tokens.bin"
    _write_tokens(tokens, [index % 32 for index in range(128)])

    checkpoint_eval.compare_checkpoints(
        baseline,
        candidate,
        project_root=tmp_path,
        token_bin=tokens,
        config=CheckpointEvalConfig(block_size=8, batch_size=1, max_batches=1),
    )


def test_compare_checkpoint_result_can_be_persisted(tmp_path: Path) -> None:
    output = tmp_path / "comparison.json"
    write_result(output, '{"promotion":{"passed":true}}')
    assert output.read_text(encoding="utf-8") == '{"promotion":{"passed":true}}\n'


def test_compare_checkpoints_rejects_without_verified_generation(tmp_path: Path) -> None:
    torch.manual_seed(51)
    config = DarwinConfig(vocab_size=32, context_length=8, d_model=16, n_layers=4, n_heads=4)
    token_values = [(i % config.vocab_size) for i in range(256)]
    tokens = tmp_path / "data" / "tokens.bin"
    _write_tokens(tokens, token_values)

    baseline = F51DarwinModel(config)
    candidate = F51DarwinModel(config)
    candidate.load_state_dict(baseline.state_dict())
    batch = torch.tensor([token_values[0:8], token_values[8:16]], dtype=torch.long)
    _train_model(candidate, batch, steps=5)

    baseline_path = tmp_path / "baseline.pt"
    candidate_path = tmp_path / "candidate.pt"
    save_checkpoint(baseline_path, baseline, config)
    save_checkpoint(candidate_path, candidate, config)

    result = compare_checkpoints(
        baseline_path,
        candidate_path,
        project_root=tmp_path,
        token_bin=tokens,
        config=CheckpointEvalConfig(block_size=8, batch_size=2, max_batches=2, max_tokens=128),
        generated_total=2,
        verified_generated=0,
    )

    assert result.heldout_candidate.loss < result.heldout_baseline.loss
    assert result.promotion["action"] in {"quarantine", "reject"}
    assert not result.promotion["passed"]
    assert not result.promotion["gates"]["verified_generation"]


def test_compare_checkpoints_can_promote_when_all_gates_pass(tmp_path: Path) -> None:
    torch.manual_seed(51)
    config = DarwinConfig(vocab_size=32, context_length=8, d_model=16, n_layers=4, n_heads=4)
    token_values = [(i % config.vocab_size) for i in range(256)]
    tokens = tmp_path / "data" / "tokens.bin"
    _write_tokens(tokens, token_values)

    baseline = F51DarwinModel(config)
    candidate = F51DarwinModel(config)
    candidate.load_state_dict(baseline.state_dict())
    batch = torch.tensor([token_values[0:8], token_values[8:16]], dtype=torch.long)
    _train_model(candidate, batch, steps=5)

    baseline_path = tmp_path / "baseline.pt"
    candidate_path = tmp_path / "candidate.pt"
    save_checkpoint(baseline_path, baseline, config)
    save_checkpoint(candidate_path, candidate, config)

    result = compare_checkpoints(
        baseline_path,
        candidate_path,
        project_root=tmp_path,
        token_bin=tokens,
        config=CheckpointEvalConfig(block_size=8, batch_size=2, max_batches=2, max_tokens=128),
        generated_total=2,
        verified_generated=2,
    )

    assert result.heldout_candidate.loss < result.heldout_baseline.loss
    assert result.promotion["passed"]
    assert result.promotion["action"] == "promote"
