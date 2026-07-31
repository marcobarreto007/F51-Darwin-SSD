import subprocess, tempfile
bash = "cd /workspace && export F51_DATASET_ROOT=/workspace && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && nohup python3 -u scripts/darwin_organism.py run247 --config configs/darwin_x_1.6b_nitro.yaml --device cuda --steps 500 --block-size 512 --batch-size 1 --accum-steps 16 --lr 0.00015 --eval-every 250 --token-bin /workspace/01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast_v2.bin --fresh-start > /workspace/train_512.log 2>&1 & echo LAUNCHED && sleep 90 && tail -5 /workspace/train_512.log && nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader"
script = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False)
with open(script.name, 'w', newline='\n') as f: f.write(bash)
r = subprocess.run(["ssh", "-i", r"C:\Users\marco\.ssh\id_ed25519", "-o", "StrictHostKeyChecking=no", "-p", "16878", "root@ssh5.vast.ai", "bash"], stdin=open(script.name, "rb"), capture_output=True, text=True, timeout=240)
print(r.stdout[-2000:])
script.close()