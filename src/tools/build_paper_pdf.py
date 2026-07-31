#!/usr/bin/env python3
"""Render a governance paper to PDF without a LaTeX toolchain.

The host has no pandoc, no TeX engine and no package manager, but Edge ships
with Windows and prints HTML to PDF headlessly. Markdown -> styled HTML ->
Edge --headless --print-to-pdf keeps the whole pipeline to one 150 KB Python
dependency instead of a multi-gigabyte TeX distribution.

The stylesheet targets a single-column academic look: serif body, tabular
figures, code and the ASCII diagram in monospace with wrapping disabled so
Figure 1 survives pagination.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[2]
EDGE_CANDIDATES = (
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
)

STYLE = """
@page { size: A4; margin: 20mm 18mm; }
body {
  font-family: "Charter", "Georgia", "Times New Roman", serif;
  font-size: 10.5pt; line-height: 1.5; color: #111;
  max-width: 100%; margin: 0;
}
h1 { font-size: 19pt; line-height: 1.2; margin: 0 0 .2em; font-weight: 700; }
h2 { font-size: 13pt; margin: 1.6em 0 .5em; border-bottom: 1px solid #bbb;
     padding-bottom: .2em; }
h3 { font-size: 11.5pt; margin: 1.2em 0 .4em; }
p, li { text-align: justify; hyphens: auto; }
hr { border: none; border-top: 1px solid #ccc; margin: 1.4em 0; }
strong { font-weight: 650; }
code {
  font-family: "Cascadia Mono", "Consolas", monospace;
  font-size: 9pt; background: #f4f4f4; padding: .08em .3em; border-radius: 2px;
}
pre {
  font-family: "Cascadia Mono", "Consolas", monospace;
  font-size: 8.2pt; line-height: 1.28; background: #f7f7f7;
  border: 1px solid #ddd; border-radius: 3px; padding: .7em .9em;
  white-space: pre; overflow-x: visible; page-break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: inherit; }
table {
  border-collapse: collapse; width: 100%; font-size: 9pt; margin: .9em 0;
  font-variant-numeric: tabular-nums; page-break-inside: avoid;
}
th, td { border: 1px solid #ccc; padding: .35em .5em; text-align: left;
         vertical-align: top; }
th { background: #f0f0f0; font-weight: 650; }
blockquote {
  margin: 1em 0; padding: .1em 0 .1em 1em; border-left: 3px solid #999;
  color: #333; font-style: italic;
}
h2, h3 { page-break-after: avoid; }
"""


def find_edge() -> Path:
    for candidate in EDGE_CANDIDATES:
        if candidate.exists():
            return candidate
    raise SystemExit(
        "Edge not found. Install a Chromium browser or supply --engine."
    )


def render_html(source: Path, destination: Path) -> None:
    text = source.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    destination.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{source.stem}</title><style>{STYLE}</style></head>"
        f"<body>{body}</body></html>",
        encoding="utf-8",
    )


def render_pdf(html: Path, pdf: Path, engine: Path) -> None:
    pdf.unlink(missing_ok=True)
    result = subprocess.run(
        [
            str(engine),
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf}",
            html.resolve().as_uri(),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if not pdf.exists():
        raise SystemExit(
            f"PDF was not produced (exit {result.returncode}).\n{result.stderr[:800]}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Markdown paper to PDF")
    parser.add_argument(
        "source",
        nargs="?",
        default=str(ROOT / "governance/docs/PAPER_REFUSE_NOT_REPORT.md"),
    )
    parser.add_argument("--out-dir", default=str(ROOT / "workspace/runtime/paper"))
    parser.add_argument("--engine", default=None)
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_file():
        raise SystemExit(f"source not found: {source}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html = out_dir / f"{source.stem}.html"
    pdf = out_dir / f"{source.stem}.pdf"
    engine = Path(args.engine) if args.engine else find_edge()

    render_html(source, html)
    render_pdf(html, pdf, engine)

    print(f"source : {source}")
    print(f"html   : {html}")
    print(f"pdf    : {pdf}  ({pdf.stat().st_size / 1024:.0f} KB)")
    print("PAPER_PDF_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
