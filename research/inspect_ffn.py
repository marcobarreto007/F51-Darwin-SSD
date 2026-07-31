import os, torch
os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ["HF_HUB_OFFLINE"]="1"
from pathlib import Path
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel

ckpt = Path("workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt")
payload = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
config = DarwinXConfig.from_mapping(payload["config"])
prev = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
model = DarwinXModel(config); torch.set_default_dtype(prev)
model.load_state_dict(payload["model_state_dict"], strict=True)

print(f"Config: {config.model_name}")
print(f"feed_forward_kind={config.feed_forward_kind}")
print(f"d_model={config.d_model} n_layers={config.n_layers}")
if config.feed_forward_kind == "dense_swiglu":
    block0 = model.blocks[0]
    print(f"Block 0 ffn type: {type(block0.ffn).__name__}")
    for name in ["gate_proj", "up_proj", "down_proj"]:
        w = getattr(block0.ffn, name)
        print(f"  {name}: {list(w.weight.shape)} = {w.weight.numel()/1e6:.1f}M params")
    total = sum(getattr(model.blocks[0].ffn, n).weight.numel() for n in ["gate_proj","up_proj","down_proj"])
    print(f"  Per-layer FFN: {total/1e6:.1f}M params")
    print(f"  24 layers FFN: {total/1e6*24:.1f}M params")
elif config.feed_forward_kind == "moe":
    print("MoE model - experts per layer")
    for li in [0, -1]:
        block = model.blocks[li]
        print(f"  Layer {li}: {len(block.moe.fine_experts)} experts")
        for name in ["gate_proj","up_proj","down_proj"]:
            w = getattr(block.moe.fine_experts[0], name)
            print(f"    {name}: {list(w.weight.shape)}")
