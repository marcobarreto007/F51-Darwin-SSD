import urllib.request, json, time
from pathlib import Path
OUT = Path("data/real_ingestion")
OUT.mkdir(parents=True, exist_ok=True)
queries = ["biology","genetics","neuroscience","immunology","physics","mathematics","chemistry","medicine","pharmacology","cardiology","oncology","molecular","epidemiology","physiology","biochemistry"]
all_text = []
for field in queries:
    try:
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={field}&retmax=100&retmode=json"
        data = json.loads(urllib.request.urlopen(url, timeout=30).read())
        ids = data.get("esearchresult",{}).get("idlist",[])
        for i in range(0, len(ids), 100):
            batch = ids[i:i+100]
            url2 = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={','.join(batch)}&rettype=abstract&retmode=text"
            text = urllib.request.urlopen(url2, timeout=60).read().decode("utf-8", errors="replace")
            all_text.append(text)
            time.sleep(0.4)
        time.sleep(0.3)
        print(f"PubMed {field}: OK")
    except Exception as e:
        print(f"PubMed {field}: {e}")
        time.sleep(2)
(OUT / "pubmed_all.txt").write_text("\n\n".join(all_text), encoding="utf-8")
print(f"PubMed DONE: {(OUT/'pubmed_all.txt').stat().st_size/1e6:.1f} MB")
