from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from f51_darwin.artifacts import resolve_token_bin
from f51_darwin.checkpointing import load_model_from_checkpoint
from f51_darwin.data import CausalLMDataLoader
from f51_darwin.evolution_gate import EvolutionGateMetrics, EvolutionGateThresholds, decide_promotion


@dataclass(frozen=True)
class CheckpointEvalConfig:
    block_size: int = 64
    batch_size: int = 2
    max_batches: int = 4
    eval_seed: int = 999
    replay_seed: int = 51
    max_tokens: int = 200_000
    device: str = "cpu"


@dataclass(frozen=True)
class LossSummary:
    loss: float
    batches: int
    tokens: int


@dataclass(frozen=True)
class CheckpointComparison:
    baseline: str
    candidate: str
    token_source: str
    heldout_baseline: LossSummary
    heldout_candidate: LossSummary
    replay_baseline: LossSummary
    replay_candidate: LossSummary
    promotion: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def load_token_ids(path: str | Path, *, max_tokens: int | None = None) -> np.ndarray:
    token_path = Path(path)
    if token_path.stat().st_size % 4 != 0:
        raise ValueError(f"token bin is not int32 aligned: {token_path}")
    tokens = np.memmap(token_path, dtype=np.int32, mode="r")
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    return tokens


def evaluate_model_loss(
    model: torch.nn.Module,
    token_ids,
    config: CheckpointEvalConfig,
    *,
    seed: int,
) -> LossSummary:
    model.eval()
    loader = CausalLMDataLoader(
        token_ids,
        block_size=config.block_size,
        batch_size=config.batch_size,
        seed=seed,
        device=config.device,
    )
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(config.max_batches):
            batch = loader.next_batch()
            output = model(batch, labels=batch)
            if output.loss is None:
                raise RuntimeError("model did not return loss")
            losses.append(float(output.loss.detach().cpu()))
    return LossSummary(
        loss=sum(losses) / max(len(losses), 1),
        batches=len(losses),
        tokens=len(losses) * config.batch_size * config.block_size,
    )


def evaluate_checkpoint_losses(
    checkpoint: str | Path,
    token_ids: object,
    config: CheckpointEvalConfig,
) -> tuple[LossSummary, LossSummary]:
    """Load, evaluate, and fully release one checkpoint before the next load."""
    device = torch.device(config.device)
    model, _, _ = load_model_from_checkpoint(checkpoint, map_location=device)
    try:
        model = model.to(device)
        heldout = evaluate_model_loss(model, token_ids, config, seed=config.eval_seed)
        replay = evaluate_model_loss(model, token_ids, config, seed=config.replay_seed)
        return heldout, replay
    finally:
        del model
        gc.collect()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def compare_checkpoints(
    baseline_checkpoint: str | Path,
    candidate_checkpoint: str | Path,
    *,
    project_root: str | Path,
    token_bin: str | Path | None = None,
    config: CheckpointEvalConfig | None = None,
    verified_generated: int = 0,
    generated_total: int = 0,
    gate_thresholds: EvolutionGateThresholds | None = None,
) -> CheckpointComparison:
    config = config or CheckpointEvalConfig()
    if token_bin is None:
        token_path = resolve_token_bin(project_root)
        if token_path is None:
            raise FileNotFoundError("No usable token bin found.")
    else:
        token_path = Path(token_bin)
    token_ids = load_token_ids(token_path, max_tokens=config.max_tokens)

    heldout_baseline, replay_baseline = evaluate_checkpoint_losses(
        baseline_checkpoint, token_ids, config
    )
    heldout_candidate, replay_candidate = evaluate_checkpoint_losses(
        candidate_checkpoint, token_ids, config
    )

    gate = decide_promotion(
        EvolutionGateMetrics(
            baseline_loss=heldout_baseline.loss,
            candidate_loss=heldout_candidate.loss,
            baseline_replay_loss=replay_baseline.loss,
            candidate_replay_loss=replay_candidate.loss,
            generated_total=generated_total,
            verified_generated=verified_generated,
            tokens_seen=heldout_candidate.tokens,
        ),
        gate_thresholds,
    )

    return CheckpointComparison(
        baseline=str(baseline_checkpoint),
        candidate=str(candidate_checkpoint),
        token_source=str(token_path),
        heldout_baseline=heldout_baseline,
        heldout_candidate=heldout_candidate,
        replay_baseline=replay_baseline,
        replay_candidate=replay_candidate,
        promotion=asdict(gate),
    )
