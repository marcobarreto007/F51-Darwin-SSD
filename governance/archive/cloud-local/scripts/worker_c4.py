from datasets import load_dataset
from pathlib import Path
OUT = Path("data/real_ingestion")
OUT.mkdir(parents=True, exist_ok=True)
ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
chunk, n = [], 0
for doc in ds:
    t = doc["text"].strip()
    if len(t) > 200:
        chunk.append(t)
        if len(chunk) >= 10000:
            (OUT / f"c4_w1_{n:04d}.txt").write_text("\n\n".join(chunk), encoding="utf-8")
            print(f"C4 chunk {n}: {sum(len(c) for c in chunk)/1e6:.1f} MB")
            n += 1; chunk = []
            if n >= 20: break
if chunk:
    (OUT / f"c4_w1_{n:04d}.txt").write_text("\n\n".join(chunk), encoding="utf-8")
print(f"C4 DONE: {n} chunks")
