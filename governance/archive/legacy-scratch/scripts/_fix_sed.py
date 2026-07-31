import subprocess, tempfile

# Use sed to delete the broken soul methods and replace with stubs
bash_script = """cd /workspace
# Delete the broken method bodies (keep def lines, replace body with pass)
python3 << 'ENDPY'
lines = open('scripts/darwin_organism.py').readlines()
out = []
skip = 0
for i, line in enumerate(lines):
    if skip > 0:
        skip -= 1
        continue
    if 'def _soul_increase_exploration' in line:
        out.append(line)
        if i+1 < len(lines) and '"""' in lines[i+1]:
            out.append(lines[i+1])
            skip = 1
        out.append('        print("  ALMA: exploration requested", flush=True)\n')
        # skip until next def
        j = i + (2 if skip == 1 else 1)
        while j < len(lines) and 'def ' not in lines[j]:
            j += 1
        skip = j - i - 1
        continue
    if 'def _soul_emergency_protocol' in line:
        out.append(line)
        if i+1 < len(lines) and '"""' in lines[i+1]:
            out.append(lines[i+1])
            skip = 1
        out.append('        print("  ALMA: emergency protocol", flush=True)\n')
        j = i + (2 if skip == 1 else 1)
        while j < len(lines) and 'def ' not in lines[j]:
            j += 1
        skip = j - i - 1
        continue
    out.append(line)
open('scripts/darwin_organism.py','w').writelines(out)
print('FIXED')
ENDPY
python3 -m py_compile scripts/darwin_organism.py && echo COMPILE_OK
"""
script = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False)
with open(script.name, 'w', newline='\n') as f:
    f.write(bash_script)
r = subprocess.run(["ssh", "-i", r"C:\Users\marco\.ssh\id_ed25519", "-o", "StrictHostKeyChecking=no", "-p", "16878", "root@ssh5.vast.ai", "bash"],
                   stdin=open(script.name, "rb"), capture_output=True, text=True, timeout=60)
print("FIX:", r.stdout.strip())
if r.stderr:
    err = r.stderr.strip()
    if "Welcome" not in err: print("ERR:", err[:200])
script.close()

if "COMPILE_OK" in r.stdout:
    bash2 = """cd /workspace && export F51_DATASET_ROOT=/workspace && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && nohup python3 -u scripts/darwin_organism.py run247 --config configs/darwin_x_1.6b_nitro.yaml --device cuda --steps 2000 --block-size 64 --batch-size 1 --accum-steps 4 --lr 0.00015 --eval-every 500 --token-bin /workspace/01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast_v2.bin --fresh-start > /workspace/train_v9.log 2>&1 & echo LAUNCHED && sleep 50 && tail -5 /workspace/train_v9.log && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader"""
    script2 = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False)
    with open(script2.name, 'w', newline='\n') as f:
        f.write(bash2)
    r2 = subprocess.run(["ssh", "-i", r"C:\Users\marco\.ssh\id_ed25519", "-o", "StrictHostKeyChecking=no", "-p", "16878", "root@ssh5.vast.ai", "bash"],
                        stdin=open(script2.name, "rb"), capture_output=True, text=True, timeout=180)
    print("LAUNCH:", r2.stdout[-2000:])
    script2.close()