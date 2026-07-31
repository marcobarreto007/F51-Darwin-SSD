import subprocess, tempfile

bash = """cd /workspace
export F51_DATASET_ROOT=/workspace
mkdir -p /workspace/02_CORPUS/approved /workspace/02_CORPUS/generated/candidates /workspace/02_CORPUS/ledger /workspace/03_CHECKPOINTS /workspace/04_MANIFESTOS
echo "Corpus:"
ls -la /workspace/01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast.bin
echo "Launching ctx=256, accum=16..."
nohup python3 -u scripts/darwin_organism.py run247 --config configs/darwin_x_1.6b_nitro.yaml --device cuda --steps 2000 --block-size 256 --batch-size 1 --accum-steps 16 --lr 0.00015 --eval-every 250 --token-bin /workspace/01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast.bin --fresh-start > /workspace/train.log 2>&1 &
PID=$!
echo "PID=$PID"
echo $PID > /workspace/train.pid
sleep 90
tail -5 /workspace/train.log
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader
"""

script = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False)
with open(script.name, 'w', newline='\n') as f: f.write(bash)
r = subprocess.run(["ssh", "-i", r"C:\Users\marco\.ssh\id_ed25519", "-o", "StrictHostKeyChecking=no", "-p", "32554", "root@ssh5.vast.ai", "bash"], stdin=open(script.name, "rb"), capture_output=True, timeout=240)
print(r.stdout.decode()[-2000:])
script.close()