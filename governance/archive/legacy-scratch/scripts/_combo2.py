import subprocess, tempfile, tarfile, io, os

# Upload all fixed files
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode='w') as tar:
    tar.add(r"C:\Users\marco\Desktop\F51-Darwin-SSD\f51_darwin\darwin_x.py", arcname="f51_darwin/darwin_x.py")
    tar.add(r"C:\Users\marco\Desktop\F51-Darwin-SSD\scripts\darwin_organism.py", arcname="scripts/darwin_organism.py")
    tar.add(r"C:\Users\marco\Desktop\F51-Darwin-SSD\f51_darwin\soul.py", arcname="f51_darwin/soul.py")
data = buf.getvalue()

r2 = subprocess.run(["ssh", "-i", r"C:\Users\marco\.ssh\id_ed25519", "-o", "StrictHostKeyChecking=no", "-p", "16878", "root@ssh5.vast.ai", "cd /workspace && tar xf - && python3 -m py_compile f51_darwin/darwin_x.py scripts/darwin_organism.py f51_darwin/soul.py && echo ALL_COMPILE_OK"], input=data, capture_output=True, timeout=30)
print("Upload:", r2.stdout.decode().strip())
if r2.stderr:
    err = r2.stderr.decode().strip()
    if "Welcome" not in err: print("ERR:", err[:200])

# Launch combo
if b"ALL_COMPILE_OK" in r2.stdout:
    bash = "cd /workspace && export F51_DATASET_ROOT=/workspace && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && nohup python3 -u scripts/darwin_organism.py run247 --config configs/darwin_x_1.6b_nitro.yaml --device cuda --steps 1000 --block-size 256 --batch-size 1 --accum-steps 16 --lr 0.00015 --eval-every 250 --token-bin /workspace/01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast_v2.bin --fresh-start > /workspace/train_combo.log 2>&1 & echo LAUNCHED && sleep 70 && tail -5 /workspace/train_combo.log && nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader"
    script = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False)
    with open(script.name, 'w', newline='\n') as f: f.write(bash)
    r3 = subprocess.run(["ssh", "-i", r"C:\Users\marco\.ssh\id_ed25519", "-o", "StrictHostKeyChecking=no", "-p", "16878", "root@ssh5.vast.ai", "bash"], stdin=open(script.name, "rb"), capture_output=True, timeout=240)
    print("LAUNCH:", r3.stdout.decode()[-2000:])
    script.close()