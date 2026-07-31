import urllib.request, json, time, re
from pathlib import Path
OUT = Path("data/real_ingestion/gutenberg")
OUT.mkdir(parents=True, exist_ok=True)
queries = ["physics","chemistry","biology","mathematics","medicine","philosophy","economics","history","astronomy","geology","botany","zoology","psychology","sociology","engineering","literature","poetry","biography","essays","letters"]
total = 0
for q in queries:
    try:
        url = f"https://gutendex.com/books?search={q}&languages=en,fr,pt&page=1"
        data = json.loads(urllib.request.urlopen(url, timeout=30).read())
        for book in data.get("results",[]):
            bid = book["id"]
            title = re.sub(r"[^a-zA-Z]", "_", book["title"])[:60]
            txt_url = f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt"
            try:
                text = urllib.request.urlopen(txt_url, timeout=45).read().decode("utf-8", errors="replace")
                if len(text) > 10000:
                    (OUT / f"{bid}_{title}.txt").write_text(text, encoding="utf-8")
                    total += len(text)
                    print(f"  {bid}: {title}")
                time.sleep(0.5)
            except: pass
        time.sleep(1)
    except Exception as e:
        print(f"Gutenberg {q}: {e}")
        time.sleep(2)
print(f"Gutenberg DONE: {total/1e6:.1f} MB")
