import subprocess, tempfile, tarfile, io, gzip, os, time, threading

ssh_key = r"C:\Users\marco\.ssh\id_ed25519"
proxy = "root@ssh5.vast.ai"
port = "32554"

results = {}

def upload_code():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w') as tar:
        tar.add(r"C:\Users\marco\Desktop\F51-Darwin-SSD\f51_darwin", arcname="f51_darwin")
        tar.add(r"C:\Users\marco\Desktop\F51-Darwin-SSD\scripts\darwin_organism.py", arcname="scripts/darwin_organism.py")
        tar.add(r"C:\Users\marco\Desktop\F51-Darwin-SSD\configs", arcname="configs")
        tar.add(r"C:\Users\marco\Desktop\F51-Darwin-SSD\tokenizer", arcname="tokenizer")
    data = buf.getvalue()
    print(f"Code tar: {len(data)/1e6:.1f} MB")
    r = subprocess.run(["ssh", "-i", ssh_key, "-o", "StrictHostKeyChecking=no", "-p", port, proxy, "cd /workspace && tar xf - && echo CODE_OK"], input=data, capture_output=True, timeout=30)
    results['code'] = r.stdout.decode().strip()

def upload_corpus():
    corpus = r"C:\Users\marco\Desktop\F51-Dataset-Organizado\01_TOKENIZADOS\00_CORPUS_PRINCIPAL_tokens_feast.bin"
    total = os.path.getsize(corpus)
    print(f"Corpus: {total/1e9:.1f} GB — streaming...")
    t0 = time.time()
    proc = subprocess.Popen(["ssh", "-i", ssh_key, "-o", "StrictHostKeyChecking=no", "-p", port, proxy,
        "mkdir -p /workspace/01_TOKENIZADOS && gunzip > /workspace/01_TOKENIZADOS/corpus.bin && ls -la /workspace/01_TOKENIZADOS/corpus.bin"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    gz = gzip.GzipFile(fileobj=proc.stdin, mode='wb', compresslevel=1)
    sent = 0
    with open(corpus, 'rb') as f:
        while True:
            chunk = f.read(16*1024*1024)
            if not chunk: break
            gz.write(chunk)
            sent += len(chunk)
    gz.close()
    out, err = proc.communicate(timeout=3600)
    elapsed = time.time() - t0
    speed = sent/1e6/elapsed if elapsed > 0 else 0
    results['corpus'] = f"Sent {sent/1e9:.1f} GB in {elapsed:.0f}s ({speed:.1f} MB/s): {out.decode().strip()}"

# Run sequentially to avoid SSH proxy overload
upload_code()
print("Code:", results.get('code', 'FAIL'))
upload_corpus()
print("Corpus:", results.get('corpus', 'FAIL'))