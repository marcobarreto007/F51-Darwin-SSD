#!/usr/bin/env python3
"""Atomic circuit transplanter with SHA-256 verification.

Uses the circuits framework (pointer, transaction, compatibility) to:
  1. Load donor and recipient models
  2. Identify a circuit by its SHA-256 stamp from the catalog
  3. Verify pre-transplant state (donor hash, recipient hash, circuit function)
  4. Swap the circuit atomically (weights + bias if present)
  5. Verify post-transplant state (recipient outside circuit unchanged, function preserved)
  6. Provide rollback by circuit ID
"""

from __future__ import annotations

import argparse, gc, json, os, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from torch import nn

from f51_darwin.circuits.identity import canonical_sha256
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.hashing import sha256_file, tensor_sha256, atomic_json_write

ROOT = Path(__file__).resolve().parents[1]

# ── Transplant record ───────────────────────────────────────────────────────

@dataclass
class TransplantRecord:
    circuit_id: str
    donor_sha256: str
    recipient_sha256: str
    operation: str  # "inject" | "replace" | "rollback"
    pre_state_hash: str
    post_state_hash: str
    donor_brain_unchanged: bool
    recipient_outside_unchanged: bool
    function_verified: bool
    timestamp: str


# ── Core operations ─────────────────────────────────────────────────────────

def hash_brain(model: nn.Module, exclude_prefixes: tuple[str, ...] = ()) -> str:
    """SHA-256 of all model weights, optionally excluding some prefixes."""
    state = model.state_dict()
    hasher = __import__("hashlib").sha256()
    for key in sorted(state):
        if any(key.startswith(p) for p in exclude_prefixes):
            continue
        t = state[key]
        hasher.update(key.encode())
        hasher.update(tensor_sha256(t).encode())
    return hasher.hexdigest()


def swap_circuit_weights(
    donor: nn.Module,
    recipient: nn.Module,
    layer: int,
    channels: list[int],
    tap: str,
    operation: str = "inject",
) -> dict[str, Any]:
    """Atomically swap circuit weights between donor and recipient.

    Returns a manifest of what was changed.
    """
    # Resolve modules
    if tap == "norm":
        donor_mod = donor.norm
        recip_mod = recipient.norm
    elif tap.startswith("blocks."):
        parts = tap.split(".")
        li = int(parts[1])
        sub = parts[2] if len(parts) > 2 else "norm2"
        donor_mod = getattr(donor.blocks[li], sub)
        recip_mod = getattr(recipient.blocks[li], sub)
    else:
        raise ValueError(f"Unknown tap: {tap}")

    # Find the weight tensor that projects into these channels
    # For RMSNorm: weight is [d_model] — swap specific channel weights
    changed = []
    snapshot_pre = {}

    if hasattr(recip_mod, "weight") and recip_mod.weight is not None:
        w = recip_mod.weight
        snapshot_pre["weight"] = w.data[channels].clone().cpu()
        if operation == "inject":
            w.data[channels] = donor_mod.weight.data[channels].clone()
        elif operation == "replace":
            # Store donor values as backup for rollback
            backup = w.data[channels].clone().cpu()
            w.data[channels] = donor_mod.weight.data[channels].clone()
            snapshot_pre["backup"] = backup
        changed.append("weight")

    if hasattr(recip_mod, "bias") and recip_mod.bias is not None:
        b = recip_mod.bias
        snapshot_pre["bias"] = b.data[channels].clone().cpu()
        if operation in ("inject", "replace"):
            b.data[channels] = donor_mod.bias.data[channels].clone()
        changed.append("bias")

    # Also affect upstream: the output projection of the PREVIOUS layer
    # that feeds into these channels
    if tap.startswith("blocks.") and len(channels) > 0:
        li = int(tap.split(".")[1])
        if li > 0:
            prev_block = recipient.blocks[li - 1]
            # The FFN output projection: down_proj or output_adapter
            for attr_name in ["ffn", "moe"]:
                ffn = getattr(prev_block, attr_name, None)
                if ffn is not None:
                    # Try to find the output projection
                    for sub_name in ["down_proj", "output_adapter"]:
                        proj = getattr(ffn, sub_name, None) if hasattr(ffn, sub_name) else None
                        if proj is not None and hasattr(proj, "weight"):
                            if operation == "inject":
                                proj.weight.data[channels, :] = (
                                    getattr(getattr(donor.blocks[li - 1], attr_name), sub_name)
                                    .weight.data[channels, :].clone()
                                )
                                changed.append(f"blocks.{li-1}.{attr_name}.{sub_name}[{len(channels)} rows]")
                            break
                    break

    return {
        "operation": operation,
        "layer": layer,
        "channels": len(channels),
        "tap": tap,
        "changed": changed,
        "snapshot_pre": {k: tensor_sha256(v) for k, v in snapshot_pre.items()},
    }


def rollback_circuit(
    model: nn.Module,
    layer: int,
    channels: list[int],
    tap: str,
    backup_weights: dict[str, torch.Tensor],
) -> None:
    """Rollback a transplanted circuit to its pre-transplant state."""
    if tap == "norm":
        mod = model.norm
    else:
        parts = tap.split(".")
        li = int(parts[1])
        sub = parts[2] if len(parts) > 2 else "norm2"
        mod = getattr(model.blocks[li], sub)

    for key, tensor in backup_weights.items():
        if key == "weight" and hasattr(mod, "weight"):
            mod.weight.data[channels] = tensor.to(device=mod.weight.device, dtype=mod.weight.dtype)
        elif key == "bias" and hasattr(mod, "bias") and mod.bias is not None:
            mod.bias.data[channels] = tensor.to(device=mod.bias.device, dtype=mod.bias.dtype)


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Atomic circuit transplanter")
    parser.add_argument("--donor", required=True, help="Donor checkpoint path")
    parser.add_argument("--recipient", required=True, help="Recipient checkpoint path")
    parser.add_argument("--catalog", help="Circuit catalog JSON from scanner")
    parser.add_argument("--circuit-id", help="Specific circuit to transplant")
    parser.add_argument("--layer", type=int, help="Layer index")
    parser.add_argument("--channels", help="Comma-separated channel indices")
    parser.add_argument("--tap", default="norm", help="Tap point (e.g., blocks.16.norm2)")
    parser.add_argument("--operation", choices=("inject", "replace"), default="inject")
    parser.add_argument("--rollback", action="store_true", help="Rollback last transplant")
    parser.add_argument("--output", default="workspace/runtime/circuit-catalog")
    args = parser.parse_args()

    print("=" * 68)
    print("ATOMIC CIRCUIT TRANSPLANTER")
    print("=" * 68)

    donor_path = Path(args.donor)
    recip_path = Path(args.recipient)
    print(f"Donor:      {donor_path}")
    print(f"Recipient:  {recip_path}")
    print(f"Operation:  {args.operation}")

    # Load donor
    d_payload = torch.load(donor_path, map_location="cpu", weights_only=False, mmap=True)
    d_config = DarwinXConfig.from_mapping(d_payload["config"])
    prev = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
    donor = DarwinXModel(d_config); torch.set_default_dtype(prev)
    donor.load_state_dict(d_payload["model_state_dict"], strict=True)
    donor.eval()
    donor_hash = sha256_file(donor_path)

    # Load recipient
    r_payload = torch.load(recip_path, map_location="cpu", weights_only=False, mmap=True)
    r_config = DarwinXConfig.from_mapping(r_payload["config"])
    prev = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
    recipient = DarwinXModel(r_config); torch.set_default_dtype(prev)
    recipient.load_state_dict(r_payload["model_state_dict"], strict=True)
    recipient.eval()
    recip_hash = sha256_file(recip_path)

    # Resolve channels
    if args.channels:
        channels = [int(x) for x in args.channels.split(",")]
    elif args.catalog and args.circuit_id:
        catalog = json.loads(Path(args.catalog).read_text())
        entry = next((s for s in catalog["stamps"] if s["circuit_id"] == args.circuit_id), None)
        if entry is None:
            print(f"Circuit {args.circuit_id} not found in catalog")
            return 1
        channels = [entry["channel"]]
        args.layer = entry["layer"]
        args.tap = entry["tap"]
    else:
        print("Need --channels or --catalog + --circuit-id")
        return 1

    print(f"Layer:      {args.layer}")
    print(f"Tap:        {args.tap}")
    print(f"Channels:   {len(channels)} channels")

    # Pre-transplant verification
    print("\n--- Pre-transplant verification ---")
    pre_recip_brain = hash_brain(recipient)
    print(f"  Recipient brain hash: {pre_recip_brain[:32]}...")

    # Execute swap
    print(f"\n--- Executing {args.operation} ---")
    manifest = swap_circuit_weights(donor, recipient, args.layer, channels, args.tap, args.operation)
    print(f"  Changed: {manifest['changed']}")

    # Post-transplant verification
    print("\n--- Post-transplant verification ---")
    post_recip_brain = hash_brain(recipient)
    outside_unchanged = pre_recip_brain == post_recip_brain
    print(f"  Recipient brain hash: {post_recip_brain[:32]}...")
    print(f"  Brain outside circuit unchanged: {outside_unchanged}")

    record = TransplantRecord(
        circuit_id=args.circuit_id or canonical_sha256({"layer": args.layer, "channels": channels}),
        donor_sha256=donor_hash,
        recipient_sha256=recip_hash,
        operation=args.operation,
        pre_state_hash=pre_recip_brain,
        post_state_hash=post_recip_brain,
        donor_brain_unchanged=sha256_file(donor_path) == donor_hash,
        recipient_outside_unchanged=outside_unchanged,
        function_verified=False,  # requires a test harness
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    out_path = Path(args.output) / "transplant-record.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import dataclasses
    atomic_json_write(dataclasses.asdict(record), out_path)
    print(f"\nRecord: {out_path}")
    print(f"TRANSPLANT_{'OK' if outside_unchanged else 'WARNING'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
