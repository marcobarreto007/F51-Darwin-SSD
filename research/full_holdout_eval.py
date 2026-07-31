#!/usr/bin/env python3
"""Standalone, read-only full-coverage holdout evaluator.

The live ``run247`` trainer only evaluates a subset of the reserved holdout
tail (``--holdout-batches`` blocks out of the full tail, see
``src/f51_darwin/organism/bootstrap.py``) because ``holdout_batches`` is baked
into the ``training_data_contract`` persisted in every checkpoint and
validated with exact equality on resume
(``src/f51_darwin/organism/support.py:validate_training_data_contract``).
Changing that value for the live lineage would break resume for every
checkpoint already on disk.

This script evaluates the SAME excluded-from-training holdout tail — same
tokens, same split boundary, same source file — but scores every
non-overlapping block that fits in it (100% coverage) instead of the
partial subset the trainer scores during the run. It:

* never writes to the checkpoint file,
* never writes to ``training_data_contract`` or any pointer/manifest,
* never touches a live training process,
* defaults to CPU so it never competes for VRAM with a live trainer.

Usage:
    python research/full_holdout_eval.py \\
        --checkpoint workspace/03_CHECKPOINTS_100M_FULL_ORGANISM_V1/organism_cycle_008_step_003600.pt \\
        --config src/configs/darwin_x_100m_full_organism.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

from f51_darwin.darwin_x import (  # noqa: E402
    DarwinXConfig,
    DarwinXModel,
    migrate_mutational_state_for_load,
)
from f51_darwin.darwin_x_training import amp_settings, load_token_ids  # noqa: E402
from f51_darwin.training_observability import (  # noqa: E402
    evaluate_fixed_holdout,
    fixed_batch_starts,
    split_tail_holdout,
)


def _load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    # map_location=cpu always for the initial read regardless of --device:
    # the payload is moved to the target device only after reconstruction,
    # matching src/scripts/inspect_organism_checkpoint.py's read pattern.
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)


def _build_model(payload: dict, file_config: DarwinXConfig) -> DarwinXModel:
    checkpoint_state = {
        key.replace("_orig_mod.", ""): value
        for key, value in payload.get("model_state_dict", {}).items()
    }
    checkpoint_version = int(payload.get("version", 0))
    model = DarwinXModel(file_config)
    topology_manifest = payload.get("topology_manifest")
    if isinstance(topology_manifest, dict):
        repaired = DarwinXModel.restore_topology(topology_manifest, model)
        if repaired:
            print(f"   Topology manifest: {repaired} repairs applied before loading tensors")
    elif checkpoint_version >= 7:
        raise ValueError("checkpoint version >= 7 but topology_manifest is missing")
    initialized = migrate_mutational_state_for_load(checkpoint_state, model)
    if initialized:
        print(f"   Mutational state migration: {len(initialized)} buffers initialized")
    model.load_state_dict(checkpoint_state, strict=checkpoint_version >= 7)
    return model


def _resolve_holdout_geometry(payload: dict, args: argparse.Namespace) -> dict:
    """Prefer the exact geometry recorded in the checkpoint's own contract.

    Falls back to explicit CLI overrides only if the checkpoint predates the
    training_data_contract schema (never observed on FULL_ORGANISM_V1, but
    kept so this tool is not useless against older checkpoints).
    """
    contract = payload.get("training_data_contract")
    if isinstance(contract, dict) and isinstance(contract.get("holdout"), dict):
        holdout = contract["holdout"]
        block_size = int(contract["block_size"])
        batch_size = int(contract["batch_size"])
        holdout_token_count = int(holdout["holdout_token_count"])
        seed = int(holdout["seed"])
        source_path = contract.get("source", {}).get("path")
        train_stop = int(holdout["train_stop"])
        holdout_start = int(holdout["holdout_start"])
        source_token_count = int(holdout["source_token_count"])
    else:
        if args.holdout_tokens is None or args.block_size is None or args.batch_size is None:
            raise ValueError(
                "checkpoint has no training_data_contract with a holdout definition; "
                "pass --holdout-tokens/--block-size/--batch-size/--holdout-seed explicitly"
            )
        block_size = args.block_size
        batch_size = args.batch_size
        holdout_token_count = args.holdout_tokens
        seed = args.holdout_seed
        source_path = None
        train_stop = None
        holdout_start = None
        source_token_count = None

    if args.token_bin is not None:
        source_path = str(args.token_bin)
    if source_path is None:
        raise ValueError("no token source path available (checkpoint contract and --token-bin both empty)")

    if holdout_token_count % block_size != 0:
        raise ValueError(
            f"holdout_token_count={holdout_token_count} is not an exact multiple of "
            f"block_size={block_size}; full non-overlapping coverage is impossible"
        )
    num_tiles = holdout_token_count // block_size
    if num_tiles % batch_size != 0:
        raise ValueError(
            f"num_tiles={num_tiles} is not an exact multiple of batch_size={batch_size}; "
            "full non-overlapping coverage is impossible"
        )
    full_batches = num_tiles // batch_size

    return {
        "block_size": block_size,
        "batch_size": batch_size,
        "holdout_token_count": holdout_token_count,
        "seed": seed,
        "source_path": source_path,
        "full_batches": full_batches,
        "num_tiles": num_tiles,
        "train_stop": train_stop,
        "holdout_start": holdout_start,
        "source_token_count": source_token_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a .pt organism checkpoint")
    parser.add_argument("--config", type=Path, required=True, help="Path to the lineage's YAML config")
    parser.add_argument(
        "--device", default="cpu",
        help="Device for the forward pass. Defaults to cpu so this never competes for VRAM "
             "with a live trainer. Pass --device cuda only if you have verified the GPU is free.",
    )
    parser.add_argument(
        "--token-bin", type=Path, default=None,
        help="Override the corpus token bin path. Defaults to the path recorded in the "
             "checkpoint's training_data_contract.",
    )
    parser.add_argument("--holdout-tokens", type=int, default=None,
                         help="Fallback if the checkpoint has no training_data_contract.")
    parser.add_argument("--block-size", type=int, default=None,
                         help="Fallback if the checkpoint has no training_data_contract.")
    parser.add_argument("--batch-size", type=int, default=None,
                         help="Fallback if the checkpoint has no training_data_contract.")
    parser.add_argument("--holdout-seed", type=int, default=999,
                         help="Fallback if the checkpoint has no training_data_contract.")
    parser.add_argument("--amp", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of text")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    config_path = args.config.resolve()
    device = torch.device(args.device)
    if device.type == "cuda":
        print(
            "WARNING: --device cuda requested. This tool does not check for a live "
            "training process on the GPU. Verify nvidia-smi is clear before proceeding.",
            file=sys.stderr,
        )

    payload = _load_checkpoint(checkpoint_path, device)
    file_config = DarwinXConfig.from_mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    embedded_config_raw = payload.get("config")
    if isinstance(embedded_config_raw, dict):
        embedded_config = DarwinXConfig.from_mapping(embedded_config_raw)
        if embedded_config != file_config:
            print(
                "WARNING: --config file does not match the config embedded in the checkpoint. "
                "Proceeding with the explicit --config file (as instructed), but the "
                "reconstructed architecture may not match what produced this checkpoint.",
                file=sys.stderr,
            )

    geometry = _resolve_holdout_geometry(payload, args)

    model = _build_model(payload, file_config)
    model.to(device)
    model.eval()

    amp_dtype, _ = amp_settings(device, args.amp)

    source_path = Path(geometry["source_path"])
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    token_ids, resolved_path = load_token_ids(ROOT, token_bin=source_path)
    total_tokens = len(token_ids)

    split = split_tail_holdout(
        token_ids,
        holdout_tokens=geometry["holdout_token_count"],
        block_size=geometry["block_size"],
    )
    if geometry["source_token_count"] is not None and split.source_token_count != geometry["source_token_count"]:
        raise ValueError(
            "corpus token count on disk does not match the checkpoint's recorded "
            f"source_token_count (disk={split.source_token_count}, "
            f"checkpoint={geometry['source_token_count']}); the corpus file may have changed "
            "since training started"
        )
    if geometry["train_stop"] is not None and split.train_stop != geometry["train_stop"]:
        raise ValueError("computed train/holdout boundary disagrees with the checkpoint's contract")

    starts = fixed_batch_starts(
        token_count=geometry["holdout_token_count"],
        block_size=geometry["block_size"],
        batch_size=geometry["batch_size"],
        batches=geometry["full_batches"],
        seed=geometry["seed"],
    )
    flat_starts = [start for row in starts for start in row]
    covered_tiles = sorted(start // geometry["block_size"] for start in flat_starts)
    full_coverage = (
        len(flat_starts) == geometry["num_tiles"]
        and len(set(flat_starts)) == geometry["num_tiles"]
        and covered_tiles == list(range(geometry["num_tiles"]))
    )

    # Combined metric via the exact production evaluator (single call, one
    # buffer/RNG snapshot-restore cycle) — this is the headline number.
    combined = evaluate_fixed_holdout(
        model, split.holdout_tokens,
        starts=starts, block_size=geometry["block_size"],
        device=device, amp_dtype=amp_dtype,
    )

    # Per-block breakdown: reuse the same production evaluator once per
    # individual block so we get real variance across the 100%-coverage set
    # instead of just the aggregate mean.
    per_block: list[dict[str, object]] = []
    for start in flat_starts:
        block_result = evaluate_fixed_holdout(
            model, split.holdout_tokens,
            starts=((start,),), block_size=geometry["block_size"],
            device=device, amp_dtype=amp_dtype,
        )
        per_block.append({
            "tile": start // geometry["block_size"],
            "token_start": start,
            "lm_loss": block_result.lm_loss,
            "ppl": block_result.ppl,
        })

    losses = [entry["lm_loss"] for entry in per_block]
    mean_loss = sum(losses) / len(losses)
    stdev_loss = statistics.stdev(losses) if len(losses) > 1 else 0.0
    mean_ppl = math.exp(min(mean_loss, 20.0))
    consistency_delta = abs(mean_loss - combined.lm_loss)

    training_state = payload.get("training_state", {})
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_cycle": training_state.get("cycle"),
        "checkpoint_step": training_state.get("step"),
        "base_checkpoint_id": payload.get("base_checkpoint_id"),
        "config": str(config_path),
        "token_source": str(resolved_path),
        "corpus_total_tokens": total_tokens,
        "holdout_token_count": geometry["holdout_token_count"],
        "block_size": geometry["block_size"],
        "batch_size": geometry["batch_size"],
        "seed": geometry["seed"],
        "blocks_evaluated": len(per_block),
        "num_tiles_available": geometry["num_tiles"],
        "full_coverage": full_coverage,
        "coverage_tokens": len(per_block) * geometry["block_size"],
        "combined_lm_loss": combined.lm_loss,
        "combined_ppl": combined.ppl,
        "per_block_mean_lm_loss": mean_loss,
        "per_block_stdev_lm_loss": stdev_loss,
        "per_block_mean_ppl": mean_ppl,
        "per_block_min_lm_loss": min(losses),
        "per_block_max_lm_loss": max(losses),
        "combined_vs_per_block_mean_delta": consistency_delta,
        "per_block": per_block,
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"Checkpoint: {result['checkpoint']}")
        print(f"  cycle={result['checkpoint_cycle']} step={result['checkpoint_step']}")
        print(f"  base_checkpoint_id={result['base_checkpoint_id']}")
        print(f"Token source: {result['token_source']} ({total_tokens} tokens total)")
        print(
            f"Holdout tail: {geometry['holdout_token_count']} tokens, "
            f"block_size={geometry['block_size']}, batch_size={geometry['batch_size']}, seed={geometry['seed']}"
        )
        print(
            f"Coverage: {len(per_block)}/{geometry['num_tiles']} tiles "
            f"({result['coverage_tokens']}/{geometry['holdout_token_count']} tokens) "
            f"full_coverage={full_coverage}"
        )
        print()
        print(f"{'tile':>4}  {'token_start':>12}  {'lm_loss':>8}  {'ppl':>10}")
        for entry in per_block:
            print(f"{entry['tile']:>4}  {entry['token_start']:>12}  {entry['lm_loss']:>8.4f}  {entry['ppl']:>10.2f}")
        print()
        print(f"Combined (production evaluate_fixed_holdout): lm_loss={combined.lm_loss:.4f}  ppl={combined.ppl:.2f}")
        print(
            f"Per-block:    mean_lm_loss={mean_loss:.4f}  stdev_lm_loss={stdev_loss:.4f}  "
            f"mean_ppl={mean_ppl:.2f}  min={min(losses):.4f}  max={max(losses):.4f}"
        )
        print(f"Consistency check (combined vs per-block mean): delta={consistency_delta:.6f}")

    if not full_coverage:
        print("WARNING: full coverage was NOT achieved — result is still partial.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
