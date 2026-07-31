from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import torch
import yaml

from f51_darwin.darwin_x import DarwinXConfig, DarwinXModel, estimate_darwin_x_parameters
from f51_darwin.tokenizer_plan import TOKENIZER_PLAN


REQUIRED_PATHS = [
    "src/configs/darwin_x_600m.yaml",
    "src/configs/darwin_x_4b.yaml",
    "src/f51_darwin/darwin_x.py",
    "src/f51_darwin/ssm_core.py",
    "research/train_darwin_x.py",
]


def check_paths() -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for relative in REQUIRED_PATHS:
        path = ROOT / relative
        results.append({"path": relative, "exists": path.exists()})
    return results


def main() -> int:
    blockers: list[str] = []
    warnings: list[str] = []

    for item in check_paths():
        if not item["exists"]:
            blockers.append(f"missing required path: {item['path']}")

    warnings.extend(
        [
            "WARNING: data/corpus should contain only approved data.",
            "Synthetic candidates must pass through data_firewall first.",
        ]
    )

    if not (ROOT / "data" / "ledger").exists():
        warnings.append("data provenance ledger not initialized yet")

    raw = yaml.safe_load((ROOT / "src" / "configs" / "darwin_x_600m.yaml").read_text(encoding="utf-8"))
    config = DarwinXConfig.from_mapping(raw)
    estimate = estimate_darwin_x_parameters(config)
    device = torch.device("cpu")
    smoke_config = DarwinXConfig(
        model_name="F51-Darwin-X-Readiness-Smoke",
        vocab_size=512,
        context_length=32,
        inference_context_length=128,
        d_model=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        fine_experts=4,
        shared_experts=1,
        experts_per_token=2,
        fine_expert_hidden_dim=32,
        shared_expert_hidden_dim=32,
        heartbeat_enabled=True,
        heartbeat_memory_capacity=16,
    )
    model = DarwinXModel(smoke_config).to(device)
    params = sum(p.numel() for p in model.parameters())

    input_ids = torch.randint(0, smoke_config.vocab_size, (1, 32), device=device)
    with torch.no_grad():
        out = model(input_ids, labels=input_ids, domain="readiness")
    if out.loss is None or not torch.isfinite(out.loss):
        blockers.append("Darwin-X smoke forward/loss failed")

    report = {
        "ready_for_darwin_x_training": len(blockers) == 0,
        "blockers": blockers,
        "warnings": warnings,
        "tokenizer_plan": TOKENIZER_PLAN,
        "parameter_estimate": {
            **estimate,
            "total_b": round(estimate["total"] / 1e9, 3),
            "active_per_token_b": round(estimate["active_per_token"] / 1e9, 3),
        },
        "smoke_params": params,
        "smoke_params_millions": round(params / 1e6, 3),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "next_owner_steps": [
            "Keep token bins verified through the corpus factory index gate",
            "Run src/tools/inspect_darwin_x.py on src/configs/darwin_x_600m.yaml",
            "Run research/train_darwin_x.py with --token-index and --dry-run",
            "Start a short Darwin-X train run only after dry-run passes",
        ],
        "note": "This script validates Darwin-X infrastructure only. It does not start training.",
    }
    output_path = ROOT / "runs" / "base_training_readiness.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready_for_darwin_x_training"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
