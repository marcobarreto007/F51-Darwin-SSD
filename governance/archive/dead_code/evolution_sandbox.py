from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from f51_darwin.artifacts import resolve_token_bin
from f51_darwin.checkpoint_eval import CheckpointEvalConfig, compare_checkpoints, load_token_ids
from f51_darwin.checkpointing import save_checkpoint
from f51_darwin.config import DarwinConfig
from f51_darwin.data import CausalLMDataLoader
from f51_darwin.evolution_gate import EvolutionGateThresholds
from f51_darwin.model import F51DarwinModel


FORBIDDEN_WORK_DIRS = ("checkpoints", "runs", "data", "tokenizer", ".git")


@dataclass(frozen=True)
class EvolutionSandboxConfig:
    block_size: int = 32
    batch_size: int = 2
    train_steps: int = 8
    max_batches: int = 4
    max_tokens: int = 200_000
    lr: float = 5e-3
    seed: int = 51
    device: str = "cpu"
    min_vocab_size: int = 16_000
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    verified_generated: int = 0
    generated_total: int = 0
    generation_checks: int = 0
    generation_prompt_size: int = 8
    min_loss_delta: float = 0.01
    max_replay_regression: float = 0.02
    min_verified_generated: int = 1
    min_verification_rate: float = 1.0


@dataclass(frozen=True)
class GenerationCheck:
    start: int
    prompt: list[int]
    expected: int
    generated: int | None
    passed: bool


@dataclass(frozen=True)
class EvolutionSandboxResult:
    work_dir: str
    token_source: str
    baseline_checkpoint: str
    candidate_checkpoint: str
    config: dict[str, Any]
    train_losses: list[float] = field(default_factory=list)
    generation_checks: list[dict[str, Any]] = field(default_factory=list)
    comparison: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _prepare_work_dir(project_root: Path, work_dir: str | Path | None) -> Path:
    if work_dir is None:
        return Path(tempfile.mkdtemp(prefix="f51_evolution_sandbox_"))

    target = Path(work_dir)
    if not target.is_absolute():
        target = project_root / target
    target = target.resolve()

    forbidden = [(project_root / item).resolve() for item in FORBIDDEN_WORK_DIRS]
    if any(_is_relative_to(target, item) for item in forbidden):
        raise ValueError(f"work_dir must not be under forbidden runtime dirs: {target}")

    target.mkdir(parents=True, exist_ok=True)
    return target


def _build_tiny_config(token_ids, sandbox: EvolutionSandboxConfig) -> DarwinConfig:
    max_token_id = int(np.max(token_ids)) if len(token_ids) else 0
    vocab_size = max(sandbox.min_vocab_size, max_token_id + 1)
    return DarwinConfig(
        model_name="F51-Darwin-SSD-sandbox",
        vocab_size=vocab_size,
        context_length=sandbox.block_size,
        d_model=sandbox.d_model,
        n_layers=sandbox.n_layers,
        n_heads=sandbox.n_heads,
    )


def verify_next_token_generation(
    model: F51DarwinModel,
    token_ids,
    *,
    prompt_size: int,
    checks: int,
    seed: int,
    device: torch.device,
) -> list[GenerationCheck]:
    if checks <= 0:
        return []
    if prompt_size < 1:
        raise ValueError("prompt_size must be positive")
    if len(token_ids) <= prompt_size + 1:
        raise ValueError("token stream is too small for generation checks")

    max_start = len(token_ids) - prompt_size - 1
    model.eval()
    probes: list[GenerationCheck] = []
    for row in range(checks):
        start = (seed + row * 31) % max_start
        prompt = [int(item) for item in token_ids[start : start + prompt_size]]
        expected = int(token_ids[start + prompt_size])
        output = model.generate(
            prompt,
            tokenizer=None,
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            use_cache=False,
        )
        generated = int(output.token_ids[0]) if output.token_ids else None
        probes.append(
            GenerationCheck(
                start=start,
                prompt=prompt,
                expected=expected,
                generated=generated,
                passed=generated == expected,
            )
        )
    return probes


def run_evolution_sandbox(
    *,
    project_root: str | Path,
    token_bin: str | Path | None = None,
    work_dir: str | Path | None = None,
    sandbox: EvolutionSandboxConfig | None = None,
) -> EvolutionSandboxResult:
    sandbox = sandbox or EvolutionSandboxConfig()
    root = Path(project_root).resolve()
    token_path = Path(token_bin).resolve() if token_bin is not None else resolve_token_bin(root)
    if token_path is None:
        raise FileNotFoundError("No usable token bin found.")

    output_dir = _prepare_work_dir(root, work_dir)
    token_ids = load_token_ids(token_path, max_tokens=sandbox.max_tokens)
    if len(token_ids) <= sandbox.block_size + 1:
        raise ValueError("token stream is too small for sandbox training")

    torch.manual_seed(sandbox.seed)
    device = torch.device(sandbox.device)
    model_config = _build_tiny_config(token_ids, sandbox)

    baseline = F51DarwinModel(model_config).to(device)
    candidate = F51DarwinModel(model_config).to(device)
    candidate.load_state_dict(baseline.state_dict())

    baseline_path = output_dir / "baseline.pt"
    candidate_path = output_dir / "candidate.pt"
    save_checkpoint(
        baseline_path,
        baseline,
        model_config,
        metrics={"sandbox": True, "role": "baseline", "seed": sandbox.seed},
    )

    loader = CausalLMDataLoader(
        token_ids,
        block_size=sandbox.block_size,
        batch_size=sandbox.batch_size,
        seed=sandbox.seed,
        device=device,
    )
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=sandbox.lr)
    train_losses: list[float] = []
    candidate.train()
    for _ in range(sandbox.train_steps):
        batch = loader.next_batch()
        optimizer.zero_grad(set_to_none=True)
        output = candidate(batch, labels=batch)
        if output.loss is None:
            raise RuntimeError("model did not return loss")
        loss = output.loss
        train_losses.append(float(loss.detach().cpu()))
        loss.backward()
        optimizer.step()

    save_checkpoint(
        candidate_path,
        candidate,
        model_config,
        metrics={
            "sandbox": True,
            "role": "candidate",
            "seed": sandbox.seed,
            "train_steps": sandbox.train_steps,
            "train_loss_first": train_losses[0] if train_losses else None,
            "train_loss_last": train_losses[-1] if train_losses else None,
        },
    )

    generation_checks = verify_next_token_generation(
        candidate,
        token_ids,
        prompt_size=min(sandbox.generation_prompt_size, sandbox.block_size - 1),
        checks=sandbox.generation_checks,
        seed=sandbox.seed + 1009,
        device=device,
    )
    verified_generated = sandbox.verified_generated + sum(1 for item in generation_checks if item.passed)
    generated_total = sandbox.generated_total + len(generation_checks)

    comparison = compare_checkpoints(
        baseline_path,
        candidate_path,
        project_root=root,
        token_bin=token_path,
        config=CheckpointEvalConfig(
            block_size=sandbox.block_size,
            batch_size=sandbox.batch_size,
            max_batches=sandbox.max_batches,
            max_tokens=sandbox.max_tokens,
            device=sandbox.device,
        ),
        verified_generated=verified_generated,
        generated_total=generated_total,
        gate_thresholds=EvolutionGateThresholds(
            min_loss_delta=sandbox.min_loss_delta,
            max_replay_regression=sandbox.max_replay_regression,
            min_verified_generated=sandbox.min_verified_generated,
            min_verification_rate=sandbox.min_verification_rate,
        ),
    )

    return EvolutionSandboxResult(
        work_dir=str(output_dir),
        token_source=str(token_path),
        baseline_checkpoint=str(baseline_path),
        candidate_checkpoint=str(candidate_path),
        config=asdict(sandbox),
        train_losses=train_losses,
        generation_checks=[asdict(item) for item in generation_checks],
        comparison=comparison.to_dict(),
    )
