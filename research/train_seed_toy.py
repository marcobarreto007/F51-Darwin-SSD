from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import torch

from f51_darwin.checkpointing import save_checkpoint
from f51_darwin.config import DarwinConfig
from f51_darwin.model import F51DarwinModel
from f51_darwin.tokenizer_plan import SimpleByteTokenizer


CORPUS = [
    "F51 Darwin SSD is born from scratch.",
    "The model grows only when necessary.",
    "Bad modules are quarantined before deletion.",
    "The system must learn today without destroying yesterday.",
]


def build_training_stream(tokenizer: SimpleByteTokenizer) -> torch.Tensor:
    text = "\n".join(CORPUS) + "\n"
    tokens = tokenizer.encode(text) * 32
    return torch.tensor(tokens, dtype=torch.long)


def sample_batch(tokens: torch.Tensor, batch_size: int, block_size: int, step: int) -> torch.Tensor:
    max_start = tokens.numel() - block_size - 1
    starts = [(step * 7 + row * 13) % max_start for row in range(batch_size)]
    return torch.stack([tokens[start : start + block_size] for start in starts])


def main() -> int:
    torch.manual_seed(51)
    tokenizer = SimpleByteTokenizer()
    config = DarwinConfig(
        model_name="F51-Darwin-SSD-Toy",
        vocab_size=tokenizer.vocab_size,
        context_length=64,
        d_model=64,
        n_layers=4,
        n_heads=4,
        mlp_ratio=2,
        dropout=0.0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = F51DarwinModel(config).to(device)
    tokens = build_training_stream(tokenizer).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    losses: list[float] = []
    model.train()
    for step in range(80):
        batch = sample_batch(tokens, batch_size=8, block_size=config.context_length, step=step).to(device)
        out = model(batch, labels=batch)
        assert out.loss is not None
        optimizer.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(out.loss.detach().cpu()))
    checkpoint_path = save_checkpoint(
        ROOT / "checkpoints" / "darwin_seed_toy.pt",
        model,
        config,
        metrics={"initial_loss": losses[0], "final_loss": losses[-1], "steps": len(losses)},
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_decreased": losses[-1] < losses[0],
        "steps": len(losses),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["loss_decreased"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

