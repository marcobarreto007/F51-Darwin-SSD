import subprocess, tarfile, io
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode='w') as tar:
    for d in ["f51_darwin", "configs", "tokenizer"]:
        tar.add(r"C:\Users\marco\Desktop\F51-Darwin-SSD\\" + d, arcname=d)
    tar.add(r"C:\Users\marco\Desktop\F51-Darwin-SSD\scripts\darwin_organism.py", arcname="scripts/darwin_organism.py")
data = buf.getvalue()
print(f"Code tar: {len(data)/1e6:.1f} MB — uploading...")
r = subprocess.run(["ssh", "-i", r"C:\Users\marco\.ssh\id_ed25519", "-o", "StrictHostKeyChecking=no", "-p", "32554", "root@ssh5.vast.ai", "cd /workspace && tar xf - && ls f51_darwin/darwin_x.py scripts/darwin_organism.py configs/darwin_x_1.6b_nitro.yaml && echo ALL_CODE_OK"], input=data, capture_output=True, timeout=60)
print(r.stdout.decode().strip())
if r.stderr:
    err = r.stderr.decode().strip()
    if "Welcome" not in err: print("ERR:", err[:200])