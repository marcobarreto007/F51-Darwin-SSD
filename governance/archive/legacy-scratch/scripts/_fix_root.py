import subprocess, tempfile
bash = """cd /workspace
# Fix: set nitro capacity to 99 so all experts stay on GPU
sed -i 's/nitro_gpu_expert_capacity: 0/nitro_gpu_expert_capacity: 99/' configs/darwin_x_1.6b_nitro.yaml
grep nitro configs/darwin_x_1.6b_nitro.yaml
env F51_DATASET_ROOT=/workspace PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python3 -u scripts/darwin_organism.py run247 --config configs/darwin_x_1.6b_nitro.yaml --device cuda --steps 2000 --block-size 256 --batch-size 1 --accum-steps 16 --lr 0.00015 --eval-every 250 --token-bin /workspace/01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast.bin --fresh-start > /workspace/train4.log 2>&1 & echo LAUNCHED && sleep 110 && tail -5 /workspace/train4.log && nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader"""
script = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False)
with open(script.name, 'w', newline='\n') as f: f.write(bash)
r = subprocess.run(["ssh", "-i", r"C:\Users\marco\.ssh\id_ed25519", "-o", "StrictHostKeyChecking=no", "-p", "32554", "root@ssh5.vast.ai", "bash"], stdin=open(script.name, "rb"), capture_output=True, timeout=240)
print(r.stdout.decode()[-2000:])
script.close()