import subprocess, base64

# Read local file
with open(r"C:\Users\marco\Desktop\F51-Darwin-SSD\scripts\darwin_organism.py", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

print(f"Base64: {len(b64)/1e3:.0f} KB")

# Step 1: Upload via base64 decode
cmd1 = ["ssh", "-i", r"C:\Users\marco\.ssh\id_ed25519", "-o", "StrictHostKeyChecking=no", "-p", "16878", "root@ssh5.vast.ai",
        f"echo {b64} | base64 -d > /workspace/scripts/darwin_organism.py && python3 -m py_compile /workspace/scripts/darwin_organism.py && echo UPLOAD_COMPILE_OK"]
r1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=60)
print("Step1:", r1.stdout.strip())
if r1.stderr:
    err = r1.stderr.strip()
    if "Welcome" not in err and "Have fun" not in err:
        print("STDERR:", err[:200])

if "UPLOAD_COMPILE_OK" in r1.stdout:
    # Step 2: Launch
    cmd2 = ["ssh", "-i", r"C:\Users\marco\.ssh\id_ed25519", "-o", "StrictHostKeyChecking=no", "-p", "16878", "root@ssh5.vast.ai",
            "cd /workspace && export F51_DATASET_ROOT=/workspace && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && nohup python3 -u scripts/darwin_organism.py run247 --config configs/darwin_x_1.6b_nitro.yaml --device cuda --steps 2000 --block-size 64 --batch-size 1 --accum-steps 4 --lr 0.00015 --eval-every 500 --token-bin /workspace/01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast_v2.bin --fresh-start > /workspace/train_v9.log 2>&1 & echo LAUNCHED && sleep 45 && tail -5 /workspace/train_v9.log && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader"]
    r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=180)
    print("Step2:", r2.stdout[-2000:])