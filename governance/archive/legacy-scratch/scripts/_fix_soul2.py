import subprocess, tempfile

# Fix the broken soul methods directly on the instance
bash = """cd /workspace
python3 << 'PYEOF'
path = 'scripts/darwin_organism.py'
with open(path) as f: content = f.read()

# Fix exploration: replace broken chr(39) line with proper print
old1 = '''print(f"  ALMA: {cmd[chr(39)+chr(39)]}", flush=True)'''
new1 = '''print(f"  ALMA: {cmd['reason']}", flush=True)'''
content = content.replace(old1, new1)

# Fix emergency: same issue  
old2 = '''print(f"  ALMA: {cmd[chr(39)+chr(39)]}", flush=True)'''
content = content.replace(old2, new1)

# Also fix the first occurrence in exploration (there might be another broken pattern)
old3 = "cmd[chr(39)+chr(39)]"
new3 = "cmd['reason']"
content = content.replace(old3, new3)

with open(path, 'w') as f: f.write(content)
print("FIXED")
PYEOF
python3 -m py_compile scripts/darwin_organism.py && echo COMPILE_OK
"""

script = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False)
with open(script.name, 'w', newline='\n') as f: f.write(bash)
r = subprocess.run(["ssh", "-i", r"C:\Users\marco\.ssh\id_ed25519", "-o", "StrictHostKeyChecking=no", "-p", "32554", "root@ssh5.vast.ai", "bash"], stdin=open(script.name, "rb"), capture_output=True, timeout=60)
print(r.stdout.decode().strip())
if r.stderr:
    err = r.stderr.decode().strip()
    if "Welcome" not in err: print("ERR:", err[:200])
script.close()

if "COMPILE_OK" in r.stdout.decode():
    # Relaunch
    bash2 = "cd /workspace && export F51_DATASET_ROOT=/workspace && nohup python3 -u scripts/darwin_organism.py run247 --config configs/darwin_x_1.6b_nitro.yaml --device cuda --steps 2000 --block-size 256 --batch-size 1 --accum-steps 16 --lr 0.00015 --eval-every 250 --token-bin /workspace/01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast.bin --fresh-start > /workspace/train.log 2>&1 & echo LAUNCHED && sleep 90 && tail -5 /workspace/train.log && nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader"
    script2 = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False)
    with open(script2.name, 'w', newline='\n') as f: f.write(bash2)
    r2 = subprocess.run(["ssh", "-i", r"C:\Users\marco\.ssh\id_ed25519", "-o", "StrictHostKeyChecking=no", "-p", "32554", "root@ssh5.vast.ai", "bash"], stdin=open(script2.name, "rb"), capture_output=True, timeout=240)
    print(r2.stdout.decode()[-2000:])
    script2.close()