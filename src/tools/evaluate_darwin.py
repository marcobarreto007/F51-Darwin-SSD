from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import torch

from f51_darwin.config import DarwinConfig
from f51_darwin.model import F51DarwinModel


def main() -> int:
    torch.manual_seed(51)
    config = DarwinConfig(
        model_name="F51-Darwin-SSD-EvalToy",
        vocab_size=128,
        context_length=16,
        d_model=32,
        n_layers=4,
        n_heads=4,
        mlp_ratio=2,
    )
    model = F51DarwinModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, config.context_length))
    out = model(input_ids, labels=input_ids)
    payload = {
        "model_name": config.model_name,
        "logits_shape": list(out.logits.shape),
        "loss": float(out.loss.detach()) if out.loss is not None else None,
        "ssd_layers": list(config.ssd_layer_indices),
        "attention_layers": list(config.attention_layer_indices),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

