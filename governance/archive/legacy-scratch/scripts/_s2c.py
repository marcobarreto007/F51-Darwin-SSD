import subprocess, gzip, os, time

corpus = r"C:\Users\marco\Desktop\F51-Dataset-Organizado\01_TOKENIZADOS\00_CORPUS_PRINCIPAL_tokens_feast.bin"
total = os.path.getsize(corpus)
print(f"Corpus: {total/1e9:.1f} GB — gzip to direct port 198.53.64.194:40717...")
t0 = time.time()
ok = True
try:
    proc = subprocess.Popen(["ssh", "-i", r"C:\Users\marco\.ssh\id_ed25519", "-o", "StrictHostKeyChecking=no", "-p", "40717", "root@198.53.64.194",
        "mkdir -p /workspace/01_TOKENIZADOS && gunzip > /workspace/01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast.bin && ls -la /workspace/01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast.bin"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
except:
    print("Direct failed, falling back to proxy...")
    proc = subprocess.Popen(["ssh", "-i", r"C:\Users\marco\.ssh\id_ed25519", "-o", "StrictHostKeyChecking=no", "-p", "32554", "root@ssh5.vast.ai",
        "mkdir -p /workspace/01_TOKENIZADOS && gunzip > /workspace/01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast.bin && ls -la /workspace/01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast.bin"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ok = False

gz = gzip.GzipFile(fileobj=proc.stdin, mode='wb', compresslevel=1)
sent = 0
with open(corpus, 'rb') as f:
    while True:
        chunk = f.read(32*1024*1024)
        if not chunk: break
        gz.write(chunk)
        sent += len(chunk)
        if sent % (10*1024*1024*1024) == 0:
            elapsed = max(time.time()-t0, 0.1)
            print(f"  {sent/total*100:.0f}% — {sent/1e9:.1f}GB — {sent/1e6/elapsed:.1f} MB/s", flush=True)
gz.close()
out, err = proc.communicate(timeout=7200)
elapsed = time.time() - t0
print(f"DONE: {sent/1e9:.1f}GB in {elapsed:.0f}s ({sent/1e6/elapsed:.1f} MB/s)")
print(out.decode().strip())