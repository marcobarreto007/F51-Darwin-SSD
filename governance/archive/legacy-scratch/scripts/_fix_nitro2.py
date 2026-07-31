import subprocess, tempfile
bash = """cd /workspace
# Fix: add nitro_enabled to config AND patch the organism to respect it
# First, add the field to the config
echo 'nitro_enabled: false' >> configs/darwin_x_1.6b_nitro.yaml
# Second, patch the organism to disable Nitro after bootstrap
python3 << 'PYEOF'
path = 'scripts/darwin_organism.py'
with open(path) as f: content = f.read()
# After model.to(device), add Nitro disable
marker = "total_params = sum(p.numel() for p in self.model.parameters()) / 1e9"
patch = marker + "\n        # Single-GPU with plenty VRAM: disable expert offloading\n        for block in self.model.blocks:\n            if hasattr(block, 'moe') and hasattr(block.moe, 'nitro_enabled'):\n                block.moe.nitro_enabled = False\n                block.moe.gpu_capacity = 999"
if marker in content and 'block.moe.nitro_enabled = False' not in content:
    content = content.replace(marker, patch)
    with open(path, 'w') as f: f.write(content)
    print('PATCHED')
else:
    print('ALREADY PATCHED or MARKER NOT FOUND')
PYEOF
python3 -m py_compile scripts/darwin_organism.py && echo COMPILE_OK
echo "Launching..."
env F51_DATASET_ROOT=/workspace PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python3 -u scripts/darwin_organism.py run247 --config configs/darwin_x_1.6b_nitro.yaml --device cuda --steps 2000 --block-size 256 --batch-size 1 --accum-steps 16 --lr 0.00015 --eval-every 250 --token-bin /workspace/01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast.bin --fresh-start > /workspace/train5.log 2>&1 & echo LAUNCHED && sleep 120 && tail -5 /workspace/train5.log && nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader"""
script = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False)
with open(script.name, 'w', newline='\n') as f: f.write(bash)
r = subprocess.run(["ssh", "-i", r"C:\Users\marco\.ssh\id_ed25519", "-o", "StrictHostKeyChecking=no", "-p", "32554", "root@ssh5.vast.ai", "bash"], stdin=open(script.name, "rb"), capture_output=True, timeout=300)
print(r.stdout.decode()[-2000:])
script.close()