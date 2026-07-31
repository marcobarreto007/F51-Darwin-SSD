"""Inspect a training checkpoint."""
import sys
from pathlib import Path

import torch

checkpoint_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("checkpoints/unified/step_0007900.pt")

print(f"Loading: {checkpoint_path}")
print(f"File size: {checkpoint_path.stat().st_size / 1e9:.2f} GB")

# Load with map_location=cpu to avoid GPU
state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

print("\n" + "=" * 60)
print("CHECKPOINT KEYS")
print("=" * 60)
for key in sorted(state.keys()):
    print(f"  {key}")

print("\n" + "=" * 60)
print("MODEL STATE")
print("=" * 60)
if "model" in state:
    model_state = state["model"]
    print(f"  Parameters: {len(model_state)} tensors")
    total_params = sum(p.numel() for p in model_state.values())
    print(f"  Total parameters: {total_params / 1e9:.2f}B")

print("\n" + "=" * 60)
print("TRAINING STATE")
print("=" * 60)
for key in ["step", "epoch", "cycle", "iterations", "updates"]:
    if key in state:
        print(f"  {key}: {state[key]}")

print("\n" + "=" * 60)
print("OPTIMIZER")
print("=" * 60)
if "optimizer" in state:
    opt = state["optimizer"]
    print(f"  Type: {opt.get('state', {}).get('__class__') if isinstance(opt, dict) else 'dict'}")
    if isinstance(opt, dict) and "state_dict" in opt:
        print(f"  States: {len(opt['state_dict'])}")

print("\n" + "=" * 60)
print("METRICS")
print("=" * 60)
for key in ["train_loss", "eval_loss", "replay_loss", "learning_rate", "gradient_norm"]:
    if key in state:
        val = state[key]
        if isinstance(val, torch.Tensor):
            val = val.item()
        print(f"  {key}: {val}")

print("\n" + "=" * 60)
print("CONFIG")
print("=" * 60)
if "config" in state:
    cfg = state["config"]
    if hasattr(cfg, "__dict__"):
        for k, v in cfg.__dict__.items():
            print(f"  {k}: {v}")
    elif isinstance(cfg, dict):
        for k, v in cfg.items():
            print(f"  {k}: {v}")

print("\n" + "=" * 60)
print("EXPERT POOL")
print("=" * 60)
if "expert_pool" in state:
    pool = state["expert_pool"]
    print(f"  Type: {type(pool)}")
    if hasattr(pool, "records"):
        print(f"  Active: {len([r for r in pool.records.values() if r.state.name == 'ACTIVE'])}")
        print(f"  Total: {len(pool.records)}")

print("\n" + "=" * 60)
print("MODEL ARCHITECTURE (sample keys)")
print("=" * 60)
if "model" in state:
    model_keys = list(state["model"].keys())[:10]
    for k in model_keys:
        shape = state["model"][k].shape
        print(f"  {k}: {shape}")
    if len(state["model"]) > 10:
        print(f"  ... and {len(state['model']) - 10} more")

print("\n" + "=" * 60)
print("TIMESTAMP")
print("=" * 60)
if "timestamp" in state:
    print(f"  {state['timestamp']}")
if "created_at" in state:
    print(f"  {state['created_at']}")

print(f"\nDone. Memory: {sum(sys.getsizeof(v) for v in state.values()) / 1e6:.1f}MB (dict overhead)")
