"""Download Olavo de Carvalho free sources into F51 candidate staging.

Writes raw files + provenance manifest to:
    data/generated/candidates/olavo/

Priority order:
    P0 — Archive.org OCR text (DjVuTXT) for foundational books
    P1 — Official site CV PDF + sample articles (WordPress API)
    P2 — Optional PDF mirrors (skipped by default; use --include-pdf)

Does NOT auto-register DataFactory candidates or touch git/ledger.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "generated" / "candidates" / "olavo"

ARCHIVE_BASE = "https://archive.org/download"
METADATA_URL = "https://archive.org/metadata/{identifier}"

# P0 — foundational books (prefer DjVuTXT OCR)
ARCHIVE_P0 = [
    {
        "id": "olavo_jardim_aflicoes",
        "archive_id": "carvalho-o-jardim-das-aflicoes",
        "title": "O Jardim das Aflições",
        "prefer_format": "DjVuTXT",
        "subdir": "livros",
    },
    {
        "id": "olavo_imbecil_coletivo",
        "archive_id": "OImbecilColetivoOlavoDeCarvalho_201810",
        "title": "O Imbecil Coletivo",
        "prefer_format": "DjVuTXT",
        "subdir": "livros",
    },
    {
        "id": "olavo_aristoteles",
        "archive_id": "aristo-teles-em-nova-perspectiva",
        "title": "Aristóteles em Nova Perspectiva",
        "prefer_format": "DjVuTXT",
        "subdir": "livros",
    },
    {
        "id": "olavo_nova_era",
        "archive_id": "a-nova-era-e-a-revoluca-o-cultural",
        "title": "A Nova Era e a Revolução Cultural",
        "prefer_format": "DjVuTXT",
        "subdir": "livros",
    },
]

OFFICIAL_CV = {
    "id": "olavo_curriculo_pdf",
    "url": "https://olavodecarvalho.org/wp-content/uploads/2018/07/Olavo-de-Carvalho-Curr%C3%ADculo.pdf",
    "filename": "Olavo-de-Carvalho-Curriculo.pdf",
    "subdir": "oficial",
}

USER_AGENT = "F51-Darwin-SSD/1.0 (research corpus; +https://github.com/marcobarreto007/F51-Darwin-SSD)"


@dataclass
class DownloadRecord:
    id: str
    title: str
    source_type: str
    source_url: str
    archive_id: str | None
    local_path: str
    bytes: int
    status: str
    downloaded_at: str
    notes: str = ""


@dataclass
class Manifest:
    version: str = "v1"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    output_root: str = ""
    records: list[DownloadRecord] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "output_root": self.output_root,
            "records": [asdict(r) for r in self.records],
            "failures": self.failures,
            "total_bytes": sum(r.bytes for r in self.records if r.status == "ok"),
            "ok_count": sum(1 for r in self.records if r.status == "ok"),
            "fail_count": len(self.failures),
        }


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _archive_file_url(archive_id: str, filename: str) -> str:
    return f"{ARCHIVE_BASE}/{quote(archive_id, safe='')}/{quote(filename, safe='')}"


def _pick_archive_file(files: list[dict[str, Any]], prefer_format: str) -> dict[str, Any] | None:
    for f in files:
        if f.get("format") == prefer_format and f.get("name"):
            return f
    for f in files:
        name = f.get("name", "")
        if name.endswith("_djvu.txt") or name.endswith(".txt"):
            return f
    for f in files:
        if f.get("format") == "Text PDF":
            return f
    return None


def download_url(
    session: requests.Session,
    url: str,
    dest: Path,
    *,
    timeout: int = 300,
    retries: int = 3,
) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                total = 0
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            fh.write(chunk)
                            total += len(chunk)
            tmp.replace(dest)
            return total
        except Exception as exc:  # noqa: BLE001 — retry loop
            last_err = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"download failed after {retries} attempts: {url}") from last_err


def download_archive_item(
    session: requests.Session,
    spec: dict[str, str],
    out_root: Path,
    *,
    include_pdf: bool,
) -> DownloadRecord:
    archive_id = spec["archive_id"]
    meta_resp = session.get(METADATA_URL.format(identifier=archive_id), timeout=60)
    meta_resp.raise_for_status()
    files = meta_resp.json().get("files", [])
    prefer = "Text PDF" if include_pdf else spec.get("prefer_format", "DjVuTXT")
    chosen = _pick_archive_file(files, prefer)
    if not chosen:
        raise RuntimeError(f"no suitable file for {archive_id}")

    filename = chosen["name"]
    ext = ".pdf" if filename.lower().endswith(".pdf") else ".txt"
    safe_name = re.sub(r"[^\w\-]+", "_", spec["id"]).strip("_") + ext
    dest = out_root / spec["subdir"] / safe_name
    url = _archive_file_url(archive_id, filename)
    size = download_url(session, url, dest)

    return DownloadRecord(
        id=spec["id"],
        title=spec["title"],
        source_type="archive.org",
        source_url=f"https://archive.org/details/{archive_id}",
        archive_id=archive_id,
        local_path=str(dest.relative_to(out_root)),
        bytes=size,
        status="ok",
        downloaded_at=datetime.now(timezone.utc).isoformat(),
        notes=f"file={filename}; format={chosen.get('format', '')}",
    )


def download_official_cv(session: requests.Session, out_root: Path) -> DownloadRecord:
    spec = OFFICIAL_CV
    dest = out_root / spec["subdir"] / spec["filename"]
    size = download_url(session, spec["url"], dest)
    return DownloadRecord(
        id=spec["id"],
        title="Currículo Olavo de Carvalho (PDF)",
        source_type="olavodecarvalho.org",
        source_url=spec["url"],
        archive_id=None,
        local_path=str(dest.relative_to(out_root)),
        bytes=size,
        status="ok",
        downloaded_at=datetime.now(timezone.utc).isoformat(),
    )


def _strip_html(html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<.*?>", " ", text)
    text = unescape(re.sub(r"\s+", " ", text)).strip()
    return text


def download_official_articles(
    session: requests.Session,
    out_root: Path,
    *,
    max_pages: int,
    per_page: int,
) -> list[DownloadRecord]:
    records: list[DownloadRecord] = []
    art_dir = out_root / "artigos"
    art_dir.mkdir(parents=True, exist_ok=True)

    for page in range(1, max_pages + 1):
        resp = session.get(
            "https://olavodecarvalho.org/wp-json/wp/v2/posts",
            params={"per_page": per_page, "page": page, "_fields": "id,slug,link,date,title,content"},
            timeout=60,
        )
        if resp.status_code == 400:
            break
        resp.raise_for_status()
        posts = resp.json()
        if not posts:
            break

        for post in posts:
            slug = post.get("slug") or str(post.get("id"))
            title = _strip_html(post.get("title", {}).get("rendered", slug))
            content_html = post.get("content", {}).get("rendered", "")
            content = _strip_html(content_html)
            if len(content) < 200:
                continue

            header = (
                f"# {title}\n"
                f"# source_url: {post.get('link', '')}\n"
                f"# published: {post.get('date', '')}\n"
                f"# source: olavodecarvalho.org\n\n"
            )
            body = header + content + "\n"
            dest = art_dir / f"{slug}.txt"
            dest.write_text(body, encoding="utf-8")
            records.append(
                DownloadRecord(
                    id=f"olavo_artigo_{post.get('id')}",
                    title=title[:120],
                    source_type="olavodecarvalho.org",
                    source_url=post.get("link", ""),
                    archive_id=None,
                    local_path=str(dest.relative_to(out_root)),
                    bytes=len(body.encode("utf-8")),
                    status="ok",
                    downloaded_at=datetime.now(timezone.utc).isoformat(),
                )
            )
        time.sleep(0.5)

    return records


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download Olavo free sources to candidate staging.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")
    p.add_argument("--include-pdf", action="store_true", help="Download PDF instead of OCR text")
    p.add_argument("--articles-pages", type=int, default=5, help="WP API pages of articles (20/page)")
    p.add_argument("--articles-per-page", type=int, default=20)
    p.add_argument("--skip-articles", action="store_true")
    p.add_argument("--skip-cv", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    out_root: Path = args.out if args.out.is_absolute() else ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = Manifest(output_root=str(out_root))
    session = _session()

    if args.dry_run:
        print(json.dumps({"out": str(out_root), "p0": ARCHIVE_P0}, indent=2))
        return 0

    for spec in ARCHIVE_P0:
        try:
            rec = download_archive_item(session, spec, out_root, include_pdf=args.include_pdf)
            manifest.records.append(rec)
            print(f"OK archive {rec.id}: {rec.bytes:,} bytes -> {rec.local_path}")
        except Exception as exc:  # noqa: BLE001 — collect and continue
            manifest.failures.append({"id": spec["id"], "error": str(exc), "archive_id": spec["archive_id"]})
            print(f"FAIL archive {spec['id']}: {exc}", file=sys.stderr)

    if not args.skip_cv:
        try:
            rec = download_official_cv(session, out_root)
            manifest.records.append(rec)
            print(f"OK cv: {rec.bytes:,} bytes -> {rec.local_path}")
        except Exception as exc:  # noqa: BLE001
            manifest.failures.append({"id": OFFICIAL_CV["id"], "error": str(exc)})
            print(f"FAIL cv: {exc}", file=sys.stderr)

    if not args.skip_articles:
        try:
            arts = download_official_articles(
                session,
                out_root,
                max_pages=args.articles_pages,
                per_page=args.articles_per_page,
            )
            manifest.records.extend(arts)
            print(f"OK articles: {len(arts)} files -> artigos/")
        except Exception as exc:  # noqa: BLE001
            manifest.failures.append({"id": "olavo_artigos", "error": str(exc)})
            print(f"FAIL articles: {exc}", file=sys.stderr)

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = manifest.to_dict()
    print(json.dumps({k: summary[k] for k in ("ok_count", "fail_count", "total_bytes", "output_root")}, indent=2))
    print(f"manifest: {manifest_path}")
    return 0 if not manifest.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
