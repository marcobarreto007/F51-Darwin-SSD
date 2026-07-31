import subprocess, tempfile
bash = """cd /workspace
# Force ALL parameters to CUDA by patching _init_expert_devices to do nothing
python3 << 'PYEOF'
path = 'f51_darwin/darwin_x.py'
with open(path) as f: content = f.read()
# Make _init_expert_devices a no-op
old = 'def _init_expert_devices(self, default_device: torch.device) -> None:'
new = 'def _init_expert_devices(self, default_device: torch.device) -> None:\n        return  # DISABLED: all experts stay on GPU'
content = content.replace(old, new)
with open(path, 'w') as f: f.write(content)
print('PATCHED')
PYEOF
python3 -m py_compile f51_darwin/darwin_x.py && echo COMPILE_OK
env F51_DATASET_ROOT=/workspace PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python3 -u scripts/darwin_organism.py run247 --config configs/darwin_x_1.6b_nitro.yaml --device cuda --steps 2000 --block-size 256 --batch-size 1 --accum-steps 16 --lr 0.00015 --eval-every 250 --token-bin /workspace/01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast.bin --fresh-start > /workspace/train3.log 2>&1 & echo LAUNCHED && sleep 120 && tail -5 /workspace/train3.log && nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader"""
script = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False)
with open(script.name, 'w', newline='\n') as f: f.write(bash)
r = subprocess.run(["ssh", "-i", r"C:\Users\marco\.ssh\id_ed25519", "-o", "StrictHostKeyChecking=no", "-p", "32554", "root@ssh5.vast.ai", "bash"], stdin=open(script.name, "rb"), capture_output=True, timeout=240)
print(r.stdout.decode()[-2000:])
script.close()