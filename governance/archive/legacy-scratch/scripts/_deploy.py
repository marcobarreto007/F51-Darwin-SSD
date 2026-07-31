import subprocess, os, tarfile, io

# Create tar in memory
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode='w') as tar:
    tar.add(r"C:\Users\marco\Desktop\F51-Darwin-SSD\scripts\darwin_organism.py", arcname="scripts/darwin_organism.py")
data = buf.getvalue()
print(f"Tar: {len(data)/1e3:.0f} KB")

# Upload and launch in one shot
ssh_cmd = ["ssh", "-i", r"C:\Users\marco\.ssh\id_ed25519", "-o", "StrictHostKeyChecking=no", "-p", "16878", "root@ssh5.vast.ai",
           "cd /workspace && tar xf - && python3 -m py_compile scripts/darwin_organism.py && echo COMPILE_OK && export F51_DATASET_ROOT=/workspace && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && nohup python3 -u scripts/darwin_organism.py run247 --config configs/darwin_x_1.6b_nitro.yaml --device cuda --steps 2000 --block-size 64 --batch-size 1 --accum-steps 4 --lr 0.00015 --eval-every 500 --token-bin /workspace/01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast_v2.bin --fresh-start > /workspace/train_v7.log 2>&1 & echo LAUNCHED && sleep 40 && tail -5 /workspace/train_v7.log && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader"]

r = subprocess.run(ssh_cmd, input=data, capture_output=True, timeout=180)
print(r.stdout.decode())
if r.stderr:
    err = r.stderr.decode()
    if "Welcome" not in err:
        print("STDERR:", err[:200])