#!/usr/bin/env python3
"""Plan F51-owned checkpoint adaptation into the Darwin-X 4B contract.

Default mode is read-only. It loads source and target tensor metadata on the
meta device, then reports which target tensors can be initialized from existing
F51 weights by exact copy, slicing, or attention qkv splitting.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]

from f51_darwin.darwin_x import DarwinXConfig, DarwinXModel


def load_target_state(config_path: Path) -> tuple[DarwinXConfig, dict[str, torch.Tensor]]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = DarwinXConfig.from_mapping(raw)
    with torch.device("meta"):
        model = DarwinXModel(config)
    return config, model.state_dict()


def compatible_slice(src_shape: tuple[int, ...], dst_shape: tuple[int, ...]) -> bool:
    return len(src_shape) == len(dst_shape) and all(dst <= src for dst, src in zip(dst_shape, src_shape))


def compatible_pad(src_shape: tuple[int, ...], dst_shape: tuple[int, ...]) -> bool:
    return len(src_shape) == len(dst_shape) and all(src <= dst for dst, src in zip(dst_shape, src_shape))


def qkv_source_for(target_key: str) -> tuple[str, str] | None:
    if ".attention.q_proj." in target_key:
        return target_key.replace(".attention.q_proj.", ".attention.qkv."), "q"
    if ".attention.k_proj." in target_key:
        return target_key.replace(".attention.k_proj.", ".attention.qkv."), "k"
    if ".attention.v_proj." in target_key:
        return target_key.replace(".attention.v_proj.", ".attention.qkv."), "v"
    return None


def source_candidates(target_key: str) -> list[str]:
    keys = [target_key]
    if ".moe.fine_router." in target_key:
        keys.append(target_key.replace(".moe.fine_router.", ".moe.router."))
    if ".moe.fine_experts." in target_key:
        keys.append(target_key.replace(".moe.fine_experts.", ".moe.experts."))
    if ".moe.shared_experts." in target_key:
        suffix = target_key.split(".moe.shared_experts.", 1)[1]
        shared_idx = int(suffix.split(".", 1)[0])
        keys.append(target_key.replace(f".moe.shared_experts.{shared_idx}.", f".moe.experts.{shared_idx}."))
    return keys


def plan_adaptation(source_state: dict[str, torch.Tensor], target_state: dict[str, torch.Tensor]) -> dict:
    actions: list[dict] = []
    counts = {"exact": 0, "slice": 0, "pad_from_source": 0, "split_qkv": 0, "random": 0}
    source_used = set()

    for target_key, target_tensor in target_state.items():
        target_shape = tuple(target_tensor.shape)
        action = None
        for source_key in source_candidates(target_key):
            source_tensor = source_state.get(source_key)
            if source_tensor is None:
                continue
            source_shape = tuple(source_tensor.shape)
            source_used.add(source_key)
            if source_shape == target_shape:
                action = {"target": target_key, "source": source_key, "mode": "exact", "shape": target_shape}
                counts["exact"] += 1
                break
            if compatible_slice(source_shape, target_shape):
                action = {
                    "target": target_key,
                    "source": source_key,
                    "mode": "slice",
                    "source_shape": source_shape,
                    "target_shape": target_shape,
                }
                counts["slice"] += 1
                break
            if compatible_pad(source_shape, target_shape):
                action = {
                    "target": target_key,
                    "source": source_key,
                    "mode": "pad_from_source",
                    "source_shape": source_shape,
                    "target_shape": target_shape,
                }
                counts["pad_from_source"] += 1
                break

        if action is None:
            qkv = qkv_source_for(target_key)
            if qkv is not None:
                source_key, part = qkv
                source_tensor = source_state.get(source_key)
                if source_tensor is not None:
                    action = {
                        "target": target_key,
                        "source": source_key,
                        "mode": "split_qkv",
                        "part": part,
                        "source_shape": tuple(source_tensor.shape),
                        "target_shape": target_shape,
                    }
                    source_used.add(source_key)
                    counts["split_qkv"] += 1

        if action is None:
            action = {"target": target_key, "source": None, "mode": "random", "shape": target_shape}
            counts["random"] += 1
        actions.append(action)

    return {
        "counts": counts,
        "target_tensors": len(target_state),
        "source_tensors": len(source_state),
        "source_tensors_used": len(source_used),
        "actions": actions,
        "actions_preview": actions[:80],
        "random_preview": [a for a in actions if a["mode"] == "random"][:80],
    }


def _prefix_slices(shape: tuple[int, ...]) -> tuple[slice, ...]:
    return tuple(slice(0, int(size)) for size in shape)


def _qkv_part(source: torch.Tensor, part: str) -> torch.Tensor:
    chunks = source.chunk(3, dim=0)
    index = {"q": 0, "k": 1, "v": 2}[part]
    return chunks[index]


def adapted_tensor(
    target: torch.Tensor,
    source_state: dict[str, torch.Tensor],
    action: dict,
) -> torch.Tensor:
    mode = action["mode"]
    if mode == "random":
        return target

    source = source_state[action["source"]].detach()
    if mode == "split_qkv":
        source = _qkv_part(source, action["part"])

    source = source.to(dtype=target.dtype)
    if tuple(source.shape) == tuple(target.shape):
        return source.clone()

    output = target.clone()
    if mode == "slice":
        return source[_prefix_slices(tuple(target.shape))].clone()
    if mode in {"pad_from_source", "split_qkv"}:
        source_slices = _prefix_slices(tuple(source.shape))
        target_slices = _prefix_slices(tuple(min(s, t) for s, t in zip(source.shape, target.shape)))
        output[target_slices] = source[source_slices][target_slices]
        return output
    raise ValueError(f"Unsupported adaptation mode: {mode}")


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def materialize_adapted_checkpoint(
    *,
    source_path: Path,
    output_path: Path,
    config: DarwinXConfig,
    target_state: dict[str, torch.Tensor],
    source_checkpoint: dict,
    plan: dict,
    dtype: torch.dtype,
) -> Path:
    source_state = source_checkpoint.get("model_state_dict")
    if not isinstance(source_state, dict):
        raise ValueError("source checkpoint has no model_state_dict")

    model = DarwinXModel(config).to(dtype=dtype)
    current = model.state_dict()
    action_by_target = {action["target"]: action for action in plan["actions"]}
    adapted_state = {}
    for key, tensor in current.items():
        adapted_state[key] = adapted_tensor(tensor, source_state, action_by_target[key])
    model.load_state_dict(adapted_state)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "metrics": {},
        "training_state": {
            "step": 0,
            "source_checkpoint": str(source_path),
            "source_step": source_checkpoint.get("training_state", {}).get("step"),
            "source_run_id": source_checkpoint.get("training_state", {}).get("run_id"),
            "adaptation_counts": plan["counts"],
        },
        "lineage": "F51 Darwin-X adapted only from F51-owned checkpoint tensors",
    }
    torch.save(payload, output_path)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(ROOT / "checkpoints" / "unified" / "step_0007900.pt"))
    parser.add_argument("--config", default=str(ROOT / "src" / "configs" / "darwin_x_4b.yaml"))
    parser.add_argument("--output", default=None, help="If set, writes an adapted Darwin-X checkpoint.")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()

    source_path = Path(args.source)
    config_path = Path(args.config)
    map_location = "cpu" if args.output else "meta"
    checkpoint = torch.load(source_path, map_location=map_location, weights_only=False, mmap=True)
    source_state = checkpoint.get("model_state_dict")
    if not isinstance(source_state, dict):
        raise ValueError("source checkpoint has no model_state_dict")

    config, target_state = load_target_state(config_path)
    plan = plan_adaptation(source_state, target_state)
    plan["source"] = str(source_path)
    plan["source_step"] = checkpoint.get("training_state", {}).get("step")
    plan["source_run_id"] = checkpoint.get("training_state", {}).get("run_id")
    plan["target_model"] = config.model_name
    if args.output:
        output = materialize_adapted_checkpoint(
            source_path=source_path,
            output_path=Path(args.output),
            config=config,
            target_state=target_state,
            source_checkpoint=checkpoint,
            plan=plan,
            dtype=dtype_from_name(args.dtype),
        )
        plan["output"] = str(output)
    plan.pop("actions", None)
    print(json.dumps(plan, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
