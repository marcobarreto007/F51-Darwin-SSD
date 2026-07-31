#!/usr/bin/env python3
"""
F51 CORPUS MINER — 20 agentes de mineracao paralela.
Fontes: Gutenberg, arXiv, OpenStax, classicos, ciencia pura.
Output: data/staging/  →  anti-woke filter  →  Hub.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]

# ── Agent definitions ──
# Each agent: (name, source_type, urls_or_ids, max_files, description)

AGENTS = [
    # === PROJECT GUTENBERG — Science & Math ===
    ("gutenberg_physics", "gutenberg", [
        "https://www.gutenberg.org/cache/epub/30155/pg30155.txt", # Einstein - Relativity ✅
        "https://www.gutenberg.org/cache/epub/22764/pg22764.txt", # Darwin - Origin of Species ✅
        "https://www.gutenberg.org/cache/epub/944/pg944.txt",     # Darwin - Voyage of the Beagle ✅
        "https://www.gutenberg.org/cache/epub/20417/pg20417.txt", # Faraday - Chemical History
        "https://www.gutenberg.org/cache/epub/14725/pg14725.txt", # Maxwell - Matter and Motion
        "https://www.gutenberg.org/cache/epub/37768/pg37768.txt", # Galileo - Dialogues
        "https://www.gutenberg.org/cache/epub/37729/pg37729.txt", # Copernicus
        "https://www.gutenberg.org/cache/epub/37134/pg37134.txt", # Kepler
        "https://www.gutenberg.org/cache/epub/33504/pg33504.txt", # Pascal
        "https://www.gutenberg.org/cache/epub/35877/pg35877.txt", # Huygens - Treatise on Light
        "https://www.gutenberg.org/cache/epub/5827/pg5827.txt",   # Curie - Radioactive Substances
        "https://www.gutenberg.org/cache/epub/40857/pg40857.txt", # Rutherford
    ], 12, "Fisica classica — Einstein, Darwin, Faraday, Maxwell, Curie"),

    ("gutenberg_math", "gutenberg", [
        "https://www.gutenberg.org/cache/epub/33283/pg33283.txt", # Whitehead - Introduction to Mathematics
        "https://www.gutenberg.org/cache/epub/38769/pg38769.txt", # Russell - Principles of Mathematics
        "https://www.gutenberg.org/cache/epub/41627/pg41627.txt", # Poincare - Science and Hypothesis
        "https://www.gutenberg.org/cache/epub/38059/pg38059.txt", # Dedekind - Essays on Number Theory
        "https://www.gutenberg.org/cache/epub/21094/pg21094.txt", # Boole - Laws of Thought
        "https://www.gutenberg.org/cache/epub/15114/pg15114.txt", # Peano
        "https://www.gutenberg.org/ebooks/search/?query=mathematics&submit_search=Search",
    ], 6, "Matematica pura — Whitehead, Russell, Poincare, Boole"),

    ("gutenberg_philosophy", "gutenberg", [
        "https://www.gutenberg.org/cache/epub/1497/pg1497.txt",   # Plato - Republic
        "https://www.gutenberg.org/cache/epub/6762/pg6762.txt",   # Aristotle - Nicomachean Ethics
        "https://www.gutenberg.org/cache/epub/8438/pg8438.txt",   # Aristotle - Politics
        "https://www.gutenberg.org/cache/epub/2412/pg2412.txt",   # Aristotle - Poetics
        "https://www.gutenberg.org/cache/epub/3600/pg3600.txt",   # Descartes - Discourse on Method
        "https://www.gutenberg.org/cache/epub/59/pg59.txt",       # Descartes - Meditations
        "https://www.gutenberg.org/cache/epub/4280/pg4280.txt",   # Kant - Critique of Pure Reason ✅
        "https://www.gutenberg.org/cache/epub/5683/pg5683.txt",   # Kant - Critique of Practical Reason ✅
        "https://www.gutenberg.org/cache/epub/38427/pg38427.txt", # Schopenhauer - World as Will ✅
        "https://www.gutenberg.org/cache/epub/4363/pg4363.txt",   # Nietzsche - Beyond Good and Evil ✅
        "https://www.gutenberg.org/cache/epub/17897/pg17897.txt", # Aquinas - Summa Theologica ✅
        "https://www.gutenberg.org/cache/epub/45988/pg45988.txt", # Bacon - Novum Organum ✅
        "https://www.gutenberg.org/cache/epub/10800/pg10800.txt", # Bacon - Novum Organum
    ], 15, "Filosofia — Platao, Aristoteles, Kant, Nietzsche, Aquinas"),

    ("gutenberg_science_biology", "gutenberg", [
        "https://www.gutenberg.org/cache/epub/36216/pg36216.txt", # Mendel
        "https://www.gutenberg.org/cache/epub/38912/pg38912.txt", # Linnaeus
        "https://www.gutenberg.org/cache/epub/35256/pg35256.txt", # Pasteur
        "https://www.gutenberg.org/cache/epub/1754/pg1754.txt",   # Huxley - Evidence as to Man's Place in Nature
        "https://www.gutenberg.org/cache/epub/59355/pg59355.txt", # Wallace
        "https://www.gutenberg.org/cache/epub/35152/pg35152.txt", # Lyell - Principles of Geology
        "https://www.gutenberg.org/cache/epub/37864/pg37864.txt", # Cuvier
        "https://www.gutenberg.org/cache/epub/35980/pg35980.txt", # Lamarck
    ], 8, "Biologia — Mendel, Pasteur, Huxley, Wallace, Cuvier"),

    ("gutenberg_astronomy", "gutenberg", [
        "https://www.gutenberg.org/cache/epub/39027/pg39027.txt", # Herschel
        "https://www.gutenberg.org/cache/epub/37134/pg37134.txt", # Kepler - Harmonies
        "https://www.gutenberg.org/cache/epub/28283/pg28283.txt", # Flammarion
        "https://www.gutenberg.org/cache/epub/25267/pg25267.txt", # Lockyer
    ], 4, "Astronomia — Herschel, Kepler, Flammarion"),

    ("gutenberg_history", "gutenberg", [
        "https://www.gutenberg.org/cache/epub/10927/pg10927.txt", # Gibbon - Decline and Fall (vol 1)
        "https://www.gutenberg.org/cache/epub/25717/pg25717.txt", # Gibbon (vol 2)
        "https://www.gutenberg.org/cache/epub/6145/pg6145.txt",   # Thucydides
        "https://www.gutenberg.org/cache/epub/14033/pg14033.txt", # Herodotus
        "https://www.gutenberg.org/cache/epub/2607/pg2607.txt",   # Plutarch - Lives
        "https://www.gutenberg.org/cache/epub/1232/pg1232.txt",   # Machiavelli - The Prince ✅
        "https://www.gutenberg.org/cache/epub/815/pg815.txt",     # Tocqueville - Democracy in America ✅
    ], 9, "Historia — Gibbon, Thucydides, Plutarch, Tocqueville"),

    ("gutenberg_economics", "gutenberg", [
        "https://www.gutenberg.org/cache/epub/3300/pg3300.txt",   # Adam Smith - Wealth of Nations
        "https://www.gutenberg.org/cache/epub/58559/pg58559.txt", # Mises (if available)
        "https://www.gutenberg.org/cache/epub/3301/pg3301.txt",   # Ricardo
        "https://www.gutenberg.org/cache/epub/11267/pg11267.txt", # Mill - Political Economy
        "https://www.gutenberg.org/cache/epub/15000/pg15000.txt", # Keynes
        "https://www.gutenberg.org/cache/epub/13400/pg13400.txt", # Hayek
    ], 6, "Economia — Smith, Ricardo, Mill, Hayek"),

    ("gutenberg_literature_en", "gutenberg", [
        "https://www.gutenberg.org/cache/epub/11/pg11.txt",       # Carroll - Alice in Wonderland
        "https://www.gutenberg.org/cache/epub/84/pg84.txt",       # Shelley - Frankenstein
        "https://www.gutenberg.org/cache/epub/345/pg345.txt",     # Stoker - Dracula
        "https://www.gutenberg.org/cache/epub/43/pg43.txt",       # Stevenson - Jekyll and Hyde
        "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",   # Doyle - Sherlock Holmes
        "https://www.gutenberg.org/cache/epub/108/pg108.txt",     # Wells - War of the Worlds
        "https://www.gutenberg.org/cache/epub/1184/pg1184.txt",   # Dumas - Monte Cristo
        "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",   # Melville - Moby Dick
        "https://www.gutenberg.org/cache/epub/98/pg98.txt",       # Dickens - Tale of Two Cities
        "https://www.gutenberg.org/cache/epub/1400/pg1400.txt",   # Dickens - Great Expectations
        "https://www.gutenberg.org/cache/epub/174/pg174.txt",     # Wilde - Dorian Gray
        "https://www.gutenberg.org/cache/epub/5200/pg5200.txt",   # Kafka - Metamorphosis
        "https://www.gutenberg.org/cache/epub/4085/pg4085.txt",   # Austen - Pride and Prejudice
        "https://www.gutenberg.org/cache/epub/25344/pg25344.txt", # Tolstoy - Anna Karenina
        "https://www.gutenberg.org/cache/epub/2600/pg2600.txt",   # Tolstoy - War and Peace
    ], 15, "Literatura classica inglesa"),

    ("gutenberg_essays", "gutenberg", [
        "https://www.gutenberg.org/cache/epub/36034/pg36034.txt", # Montaigne - Essays
        "https://www.gutenberg.org/cache/epub/205/pg205.txt",     # Bacon - Essays
        "https://www.gutenberg.org/cache/epub/4495/pg4495.txt",   # Emerson - Essays
        "https://www.gutenberg.org/cache/epub/205/pg205.txt",     # Thoreau - Walden ✅
        "https://www.gutenberg.org/cache/epub/18269/pg18269.txt", # Chesterton
        "https://www.gutenberg.org/cache/epub/13085/pg13085.txt", # Orwell
    ], 6, "Ensaios — Montaigne, Bacon, Emerson, Thoreau"),

    # === OPENSTAX — Free Textbooks ===
    ("openstax_physics", "openstax", [
        "https://openstax.org/books/university-physics-volume-1/pages/1-introduction",
        "https://openstax.org/books/university-physics-volume-2/pages/1-introduction",
        "https://openstax.org/books/university-physics-volume-3/pages/1-introduction",
        "https://openstax.org/books/college-physics/pages/1-introduction",
    ], 4, "Fisica universitaria — OpenStax"),

    ("openstax_math", "openstax", [
        "https://openstax.org/books/calculus-volume-1/pages/1-introduction",
        "https://openstax.org/books/calculus-volume-2/pages/1-introduction",
        "https://openstax.org/books/calculus-volume-3/pages/1-introduction",
        "https://openstax.org/books/precalculus/pages/1-introduction",
        "https://openstax.org/books/college-algebra/pages/1-introduction",
        "https://openstax.org/books/algebra-and-trigonometry/pages/1-introduction",
        "https://openstax.org/books/statistics/pages/1-introduction",
    ], 7, "Matematica — OpenStax textbooks"),

    ("openstax_science", "openstax", [
        "https://openstax.org/books/biology-2e/pages/1-introduction",
        "https://openstax.org/books/chemistry-2e/pages/1-introduction",
        "https://openstax.org/books/anatomy-and-physiology/pages/1-introduction",
        "https://openstax.org/books/microbiology/pages/1-introduction",
        "https://openstax.org/books/astronomy/pages/1-introduction",
    ], 5, "Ciencias — OpenStax textbooks"),

    ("openstax_humanities", "openstax", [
        "https://openstax.org/books/american-government/pages/1-introduction",
        "https://openstax.org/books/us-history/pages/1-introduction",
        "https://openstax.org/books/world-history-volume-1/pages/1-introduction",
        "https://openstax.org/books/psychology-2e/pages/1-introduction",
        "https://openstax.org/books/economics-2e/pages/1-introduction",
    ], 5, "Humanidades — OpenStax textbooks"),

    # === ARXIV — Papers (OAI-PMH API) ===
    ("arxiv_math_cs", "arxiv_api", [
        "http://export.arxiv.org/api/query?search_query=cat:math.*&start=0&max_results=50",
        "http://export.arxiv.org/api/query?search_query=cat:cs.*&start=0&max_results=50",
    ], 100, "arXiv — Matematica e Computacao"),

    ("arxiv_physics", "arxiv_api", [
        "http://export.arxiv.org/api/query?search_query=cat:physics.*&start=0&max_results=50",
        "http://export.arxiv.org/api/query?search_query=cat:astro-ph.*&start=0&max_results=50",
    ], 100, "arXiv — Fisica e Astronomia"),

    ("arxiv_qbio", "arxiv_api", [
        "http://export.arxiv.org/api/query?search_query=cat:q-bio.*&start=0&max_results=50",
        "http://export.arxiv.org/api/query?search_query=cat:stat.*&start=0&max_results=50",
    ], 100, "arXiv — Biologia Quantitativa e Estatistica"),

    # === WIKISOURCE — Public Domain ===
    ("wikisource_science", "wikisource", [
        "https://en.wikisource.org/wiki/Category:Physics",
        "https://en.wikisource.org/wiki/Category:Mathematics",
        "https://en.wikisource.org/wiki/Category:Biology",
        "https://en.wikisource.org/wiki/Category:Astronomy",
    ], 4, "Wikisource — Ciencia"),

    ("wikisource_classics", "wikisource", [
        "https://en.wikisource.org/wiki/Category:Philosophy",
        "https://en.wikisource.org/wiki/Category:History",
        "https://en.wikisource.org/wiki/Category:Economics",
    ], 3, "Wikisource — Humanidades"),

    # === INTERNET ARCHIVE — Public Domain ===
    ("archive_texts", "archive", [
        "https://archive.org/details/pub_encyclopaedia-britannica-11ed",
        "https://archive.org/details/texts?and[]=subject%3A%22Mathematics%22&and[]=mediatype%3A%22texts%22",
        "https://archive.org/details/texts?and[]=subject%3A%22Physics%22&and[]=mediatype%3A%22texts%22",
        "https://archive.org/details/texts?and[]=subject%3A%22Philosophy%22&and[]=mediatype%3A%22texts%22",
    ], 4, "Internet Archive — Enciclopedia e textos"),

    # === TESLA / ENGINEERING ===
    ("gutenberg_engineering", "gutenberg", [
        "https://www.gutenberg.org/cache/epub/13476/pg13476.txt", # Tesla - Experiments
        "https://www.gutenberg.org/cache/epub/39293/pg39293.txt", # Tesla - Inventions
        "https://www.gutenberg.org/cache/epub/47282/pg47282.txt", # Edison
        "https://www.gutenberg.org/cache/epub/56509/pg56509.txt", # Bell
        "https://www.gutenberg.org/cache/epub/23374/pg23374.txt", # Wright Brothers
    ], 5, "Engenharia — Tesla, Edison, Bell, Wright"),
]


@dataclass
class AgentReport:
    name: str
    status: str          # "ok" | "partial" | "failed"
    files_downloaded: int = 0
    bytes_downloaded: int = 0
    files_blocked: int = 0
    errors: list[str] = field(default_factory=list)
    output_paths: list[str] = field(default_factory=list)


def download_url(url: str, dest: Path, timeout: int = 120) -> bool:
    """Download a single URL to dest. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 100:
        return True  # already downloaded

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "F51-Corpus-Miner/1.0 (research; contact@f51.ai)"
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return len(data) > 100
    except Exception as e:
        return False


def mine_gutenberg(agent_name: str, urls: list[str], out_dir: Path) -> AgentReport:
    """Download Gutenberg texts directly."""
    report = AgentReport(name=agent_name, status="ok")
    for url in urls:
        if "/ebooks/search" in url:
            continue  # skip search URLs for now
        fname = urlparse(url).path.rstrip("/").split("/")[-1]
        if not fname.endswith(".txt"):
            fname += ".txt"
        dest = out_dir / agent_name / fname
        ok = download_url(url, dest)
        if ok:
            report.files_downloaded += 1
            report.bytes_downloaded += dest.stat().st_size
            report.output_paths.append(str(dest))
        else:
            report.errors.append(f"Failed: {url}")
    if report.files_downloaded == 0:
        report.status = "failed"
    elif report.files_downloaded < len([u for u in urls if "/ebooks/search" not in u]):
        report.status = "partial"
    return report


def mine_arxiv_api(agent_name: str, urls: list[str], out_dir: Path) -> AgentReport:
    """Fetch arXiv paper metadata via API, download PDFs."""
    report = AgentReport(name=agent_name, status="ok")
    for api_url in urls:
        try:
            req = urllib.request.Request(api_url, headers={
                "User-Agent": "F51-Corpus-Miner/1.0"
            })
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read().decode("utf-8")
            root = ET.fromstring(data)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", ns)
            for entry in entries:
                title = entry.find("atom:title", ns)
                summary = entry.find("atom:summary", ns)
                pdf_link = None
                for link in entry.findall("atom:link", ns):
                    if link.get("title") == "pdf":
                        pdf_link = link.get("href")
                        break
                if not pdf_link:
                    continue
                # Download PDF
                arxiv_id = pdf_link.split("/")[-1].replace(".pdf", "")
                dest_txt = out_dir / agent_name / f"{arxiv_id}.txt"
                if dest_txt.exists():
                    continue
                # Download PDF, convert to text
                pdf_dest = out_dir / agent_name / f"{arxiv_id}.pdf"
                ok = download_url(pdf_link, pdf_dest)
                if ok:
                    # Try pdftotext
                    try:
                        subprocess.run(
                            ["pdftotext", str(pdf_dest), str(dest_txt)],
                            capture_output=True, timeout=30
                        )
                        if dest_txt.exists() and dest_txt.stat().st_size > 100:
                            report.files_downloaded += 1
                            report.bytes_downloaded += dest_txt.stat().st_size
                            report.output_paths.append(str(dest_txt))
                        pdf_dest.unlink()  # remove PDF, keep text
                    except Exception:
                        pass
                if report.files_downloaded >= 100:
                    break
        except Exception as e:
            report.errors.append(f"API error: {e}")
    if report.files_downloaded == 0:
        report.status = "failed"
    elif report.files_downloaded < 10:
        report.status = "partial"
    return report


def mine_openstax(agent_name: str, urls: list[str], out_dir: Path) -> AgentReport:
    """Download OpenStax textbook pages as text."""
    report = AgentReport(name=agent_name, status="ok")
    for base_url in urls:
        book_name = base_url.split("/books/")[1].split("/pages/")[0]
        dest = out_dir / agent_name / f"{book_name}.txt"
        if dest.exists() and dest.stat().st_size > 1000:
            report.files_downloaded += 1
            report.bytes_downloaded += dest.stat().st_size
            report.output_paths.append(str(dest))
            continue
        # OpenStax pages are HTML; try fetching chapter by chapter
        all_text = []
        for ch in range(1, 30):
            ch_url = base_url.replace("/1-introduction", f"/{ch}-introduction") if ch > 1 else base_url
            try:
                req = urllib.request.Request(ch_url, headers={"User-Agent": "F51-Miner/1.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    html = resp.read().decode("utf-8", errors="replace")
                # Extract text from HTML (simple approach)
                text = re.sub(r"<[^>]+>", " ", html)
                text = re.sub(r"\s+", " ", text).strip()
                if len(text) > 500:
                    all_text.append(text)
            except Exception:
                break
        if all_text:
            dest.write_text("\n\n".join(all_text), encoding="utf-8")
            report.files_downloaded += 1
            report.bytes_downloaded += dest.stat().st_size
            report.output_paths.append(str(dest))
    if report.files_downloaded == 0:
        report.status = "partial"
    return report


def mine_wikisource(agent_name: str, urls: list[str], out_dir: Path) -> AgentReport:
    """Fetch Wikisource category pages (placeholder for now)."""
    report = AgentReport(name=agent_name, status="partial")
    for url in urls:
        cat_name = url.rstrip("/").split("/")[-1]
        dest = out_dir / agent_name / f"{cat_name}_index.txt"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "F51-Miner/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 200:
                dest.write_text(text, encoding="utf-8")
                report.files_downloaded += 1
                report.bytes_downloaded += len(text)
        except Exception as e:
            report.errors.append(f"{url}: {e}")
    return report


def mine_archive(agent_name: str, urls: list[str], out_dir: Path) -> AgentReport:
    """Fetch Internet Archive metadata (placeholder)."""
    report = AgentReport(name=agent_name, status="partial")
    for url in urls:
        cat_name = urlparse(url).path.split("/")[-1] or "index"
        dest = out_dir / agent_name / f"{cat_name}.txt"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "F51-Miner/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 200:
                dest.write_text(text, encoding="utf-8")
                report.files_downloaded += 1
                report.bytes_downloaded += len(text)
        except Exception as e:
            report.errors.append(f"{url}: {e}")
    return report


MINERS = {
    "gutenberg": mine_gutenberg,
    "arxiv_api": mine_arxiv_api,
    "openstax": mine_openstax,
    "wikisource": mine_wikisource,
    "archive": mine_archive,
}


def run_agent(agent_def: tuple, out_dir: Path) -> AgentReport:
    """Run a single mining agent."""
    name, source_type, urls, max_files, desc = agent_def
    miner = MINERS.get(source_type)
    if not miner:
        return AgentReport(name=name, status="failed", errors=[f"Unknown source: {source_type}"])
    print(f"  [{name}] {desc} — {len(urls)} URLs", flush=True)
    report = miner(name, urls, out_dir)
    print(f"  [{name}] -> {report.status}: {report.files_downloaded} files, {report.bytes_downloaded/1e6:.1f} MB", flush=True)
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="F51 Corpus Miner — 20 parallel agents")
    parser.add_argument("--out", default=str(ROOT / "data" / "staging" / "mined"))
    parser.add_argument("--agents", type=int, default=20, help="Max parallel agents (default: 20)")
    parser.add_argument("--filter", action="store_true", help="Run anti-woke filter after download")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"F51 CORPUS MINER — {len(AGENTS)} agents, {args.agents} parallel")
    print(f"Output: {out_dir}")
    print()

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.agents) as pool:
        futures = {pool.submit(run_agent, agent, out_dir): agent[0] for agent in AGENTS}
        for future in as_completed(futures):
            name = futures[future]
            try:
                report = future.result()
                results.append(report)
            except Exception as e:
                results.append(AgentReport(name=name, status="failed", errors=[str(e)]))

    elapsed = time.time() - t0

    # Summary
    total_files = sum(r.files_downloaded for r in results)
    total_bytes = sum(r.bytes_downloaded for r in results)
    ok = [r for r in results if r.status == "ok"]
    partial = [r for r in results if r.status == "partial"]
    failed = [r for r in results if r.status == "failed"]

    print(f"\n{'='*60}")
    print(f"  MINER COMPLETE — {elapsed:.0f}s")
    print(f"  {len(ok)} ok | {len(partial)} partial | {len(failed)} failed")
    print(f"  {total_files} files | {total_bytes/1e6:.1f} MB")
    print(f"{'='*60}")

    if failed:
        print("\n  FAILED AGENTS:")
        for r in failed:
            print(f"    {r.name}: {r.errors[:2]}")

    # Save report
    report_path = out_dir / "miner_report.json"
    report_path.write_text(json.dumps({
        "elapsed_s": elapsed,
        "total_files": total_files,
        "total_bytes": total_bytes,
        "ok": len(ok),
        "partial": len(partial),
        "failed": len(failed),
        "agents": [
            {"name": r.name, "status": r.status, "files": r.files_downloaded,
             "bytes": r.bytes_downloaded, "errors": r.errors[:3]}
            for r in sorted(results, key=lambda r: r.name)
        ]
    }, indent=2))
    print(f"\n  Report: {report_path}")

    # Anti-woke filter
    if args.filter:
        print("\n  Running anti-woke filter...")
        from f51_darwin.anti_woke_filter import AntiWokeFilter
        filt = AntiWokeFilter()
        staged = out_dir
        approved_dir = ROOT / "data" / "staging" / "approved"
        approved_dir.mkdir(parents=True, exist_ok=True)
        approved_count = 0
        blocked_count = 0
        for txt_file in staged.rglob("*.txt"):
            try:
                text = txt_file.read_text(encoding="utf-8", errors="replace")
                if not filt.should_block(text, file_path=str(txt_file)):
                    dest = approved_dir / txt_file.relative_to(staged)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(text, encoding="utf-8")
                    approved_count += 1
                else:
                    blocked_count += 1
            except Exception:
                pass
        print(f"  Approved: {approved_count} | Blocked: {blocked_count}")
        print(f"  Approved dir: {approved_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
