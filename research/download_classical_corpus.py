"""Download classical liberal / conservative corpus into F51 candidate staging.

Writes raw files + provenance manifest to:
    data/generated/candidates/{author_slug}/

Reuses patterns from research/download_olavo_sources.py.
Does NOT auto-register DataFactory candidates or touch git/ledger.

Usage:
    python research/download_classical_corpus.py --list-authors
    python research/download_classical_corpus.py --author adam_smith --dry-run
    python research/download_classical_corpus.py --author hume --priority P0
    python research/download_classical_corpus.py --phase 1 --dry-run

See governance/docs/CORPUS_CLASSICAL_LIBERAL_CORPUS.md for full source research.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = ROOT / "data" / "generated" / "candidates"

GUTENBERG_BASE = "https://www.gutenberg.org/files"
ARCHIVE_BASE = "https://archive.org/download"
METADATA_URL = "https://archive.org/metadata/{identifier}"

USER_AGENT = "F51-Darwin-SSD/1.0 (research corpus; +https://github.com/marcobarreto007/F51-Darwin-SSD)"


# ═══════════════════════════════════════════════════════════
# AUTHOR REGISTRY — extend per governance/docs/CORPUS_CLASSICAL_LIBERAL_CORPUS.md
# ═══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SourceSpec:
    id: str
    title: str
    priority: str  # P0, P1, P2, P3
    source_type: str  # gutenberg, archive.org, url, mises, liberty_fund
    url: str
    gutenberg_id: int | None = None
    archive_id: str | None = None
    prefer_format: str = "txt"
    subdir: str = "works"
    lang: str = "en"
    tier: str = "T1"
    est_lines: int = 0
    notes: str = ""


@dataclass(frozen=True)
class AuthorSpec:
    slug: str
    name_en: str
    name_pt: str
    moe_domain: str
    sources: tuple[SourceSpec, ...]


AUTHOR_REGISTRY: dict[str, AuthorSpec] = {
    "adam_smith": AuthorSpec(
        slug="adam_smith",
        name_en="Adam Smith",
        name_pt="Adam Smith",
        moe_domain="financas",
        sources=(
            SourceSpec(
                id="smith_wealth_of_nations",
                title="An Inquiry into the Nature and Causes of the Wealth of Nations",
                priority="P0",
                source_type="gutenberg",
                url="https://www.gutenberg.org/ebooks/3300",
                gutenberg_id=3300,
                lang="en",
                tier="T1",
                est_lines=15400,
            ),
            SourceSpec(
                id="smith_theory_moral_sentiments",
                title="The Theory of Moral Sentiments",
                priority="P0",
                source_type="gutenberg",
                url="https://www.gutenberg.org/ebooks/5850",
                gutenberg_id=5850,
                lang="en",
                tier="T1",
                est_lines=12000,
            ),
        ),
    ),
    "hume": AuthorSpec(
        slug="hume",
        name_en="David Hume",
        name_pt="David Hume",
        moe_domain="tecnico_en",
        sources=(
            SourceSpec(
                id="hume_treatise",
                title="A Treatise of Human Nature",
                priority="P0",
                source_type="gutenberg",
                url="https://www.gutenberg.org/ebooks/4705",
                gutenberg_id=4705,
                lang="en",
                tier="T1",
                est_lines=80000,
            ),
            SourceSpec(
                id="hume_enquiry_understanding",
                title="An Enquiry Concerning Human Understanding",
                priority="P0",
                source_type="gutenberg",
                url="https://www.gutenberg.org/ebooks/9662",
                gutenberg_id=9662,
                lang="en",
                tier="T1",
                est_lines=15000,
            ),
            SourceSpec(
                id="hume_enquiry_morals",
                title="An Enquiry Concerning the Principles of Morals",
                priority="P0",
                source_type="gutenberg",
                url="https://www.gutenberg.org/ebooks/4320",
                gutenberg_id=4320,
                lang="en",
                tier="T1",
                est_lines=12000,
            ),
        ),
    ),
    "burke": AuthorSpec(
        slug="burke",
        name_en="Edmund Burke",
        name_pt="Edmund Burke",
        moe_domain="identidade_f51",
        sources=(
            SourceSpec(
                id="burke_reflections",
                title="Reflections on the Revolution in France",
                priority="P0",
                source_type="gutenberg",
                url="https://www.gutenberg.org/ebooks/15679",
                gutenberg_id=15679,
                lang="en",
                tier="T1",
                est_lines=25000,
            ),
            SourceSpec(
                id="burke_conciliation_america",
                title="Speech on Conciliation with America",
                priority="P0",
                source_type="gutenberg",
                url="https://www.gutenberg.org/ebooks/21708",
                gutenberg_id=21708,
                lang="en",
                tier="T1",
                est_lines=5000,
            ),
        ),
    ),
    "tocqueville": AuthorSpec(
        slug="tocqueville",
        name_en="Alexis de Tocqueville",
        name_pt="Alexis de Tocqueville",
        moe_domain="tecnico_en",
        sources=(
            SourceSpec(
                id="tocqueville_democracy_america",
                title="Democracy in America",
                priority="P0",
                source_type="gutenberg",
                url="https://www.gutenberg.org/ebooks/815",
                gutenberg_id=815,
                lang="en",
                tier="T1",
                est_lines=60000,
            ),
            SourceSpec(
                id="tocqueville_old_regime",
                title="The Old Regime and the Revolution",
                priority="P1",
                source_type="gutenberg",
                url="https://www.gutenberg.org/ebooks/1309",
                gutenberg_id=1309,
                lang="en",
                tier="T1",
                est_lines=20000,
            ),
        ),
    ),
    "aristotle": AuthorSpec(
        slug="aristotle",
        name_en="Aristotle",
        name_pt="Aristóteles",
        moe_domain="reserva",
        sources=(
            SourceSpec(
                id="aristotle_nicomachean_ethics",
                title="Nicomachean Ethics",
                priority="P0",
                source_type="gutenberg",
                url="https://www.gutenberg.org/ebooks/8438",
                gutenberg_id=8438,
                lang="en",
                tier="T1",
                est_lines=25000,
            ),
            SourceSpec(
                id="aristotle_politics",
                title="Politics",
                priority="P0",
                source_type="gutenberg",
                url="https://www.gutenberg.org/ebooks/6762",
                gutenberg_id=6762,
                lang="en",
                tier="T1",
                est_lines=20000,
            ),
            SourceSpec(
                id="aristotle_poetics",
                title="Poetics",
                priority="P0",
                source_type="gutenberg",
                url="https://www.gutenberg.org/ebooks/1974",
                gutenberg_id=1974,
                lang="en",
                tier="T1",
                est_lines=3000,
            ),
        ),
    ),
    "gomez_davila": AuthorSpec(
        slug="gomez_davila",
        name_en="Nicolás Gómez Dávila",
        name_pt="Nicolás Gómez Dávila",
        moe_domain="literatura_pt",
        sources=(
            SourceSpec(
                id="gomez_davila_escolios",
                title="Escolios a un Texto Implícito",
                priority="P0",
                source_type="url",
                url="https://es.wikisource.org/wiki/Escolios_a_un_texto_impl%C3%ADcito",
                lang="es",
                tier="T1",
                est_lines=8000,
                notes="Wikisource — requires HTML→text extractor (not implemented in skeleton)",
            ),
        ),
    ),
    "olavo": AuthorSpec(
        slug="olavo",
        name_en="Olavo de Carvalho",
        name_pt="Olavo de Carvalho",
        moe_domain="identidade_f51",
        sources=(),  # use research/download_olavo_sources.py
    ),
}

# Phase → author slugs (from CORPUS doc)
DOWNLOAD_PHASES: dict[int, list[str]] = {
    1: ["mises", "cicero", "aquinas", "aristotle", "hume"],
    2: ["adam_smith", "hayek", "buchanan", "burke", "tocqueville", "friedman"],
    3: ["olavo", "roberto_campos", "gomez_davila", "george_grant"],
    4: ["solzhenitsyn", "kirk", "sowell", "oakeshott", "scruton"],
}


@dataclass
class DownloadRecord:
    id: str
    title: str
    author_slug: str
    source_type: str
    source_url: str
    local_path: str
    bytes: int
    status: str
    downloaded_at: str
    priority: str = "P0"
    tier: str = "T1"
    est_lines: int = 0
    notes: str = ""


@dataclass
class Manifest:
    version: str = "v1"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    author_slug: str = ""
    output_root: str = ""
    records: list[DownloadRecord] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "author_slug": self.author_slug,
            "output_root": self.output_root,
            "records": [asdict(r) for r in self.records],
            "failures": self.failures,
            "total_bytes": sum(r.bytes for r in self.records if r.status == "ok"),
            "ok_count": sum(1 for r in self.records if r.status == "ok"),
            "fail_count": len(self.failures),
            "est_lines_total": sum(r.est_lines for r in self.records if r.status == "ok"),
        }


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _gutenberg_txt_urls(gutenberg_id: int) -> list[str]:
    gid = str(gutenberg_id)
    return [
        f"{GUTENBERG_BASE}/{gid}/{gid}-0.txt",
        f"{GUTENBERG_BASE}/{gid}/{gid}.txt",
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
    ]


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


def download_gutenberg(
    session: requests.Session,
    spec: SourceSpec,
    author: AuthorSpec,
    out_root: Path,
) -> DownloadRecord:
    if spec.gutenberg_id is None:
        raise ValueError(f"missing gutenberg_id for {spec.id}")
    safe_name = re.sub(r"[^\w\-]+", "_", spec.id).strip("_") + ".txt"
    dest = out_root / spec.subdir / safe_name
    last_err: Exception | None = None
    size = 0
    url = ""
    for candidate_url in _gutenberg_txt_urls(spec.gutenberg_id):
        try:
            url = candidate_url
            size = download_url(session, url, dest)
            break
        except Exception as exc:  # noqa: BLE001 — try next mirror
            last_err = exc
            if dest.exists():
                dest.unlink(missing_ok=True)
    else:
        raise RuntimeError(f"all Gutenberg URLs failed for {spec.gutenberg_id}") from last_err

    header = (
        f"# {spec.title}\n"
        f"# author: {author.name_en}\n"
        f"# source_url: {spec.url}\n"
        f"# source_type: gutenberg\n"
        f"# gutenberg_id: {spec.gutenberg_id}\n"
        f"# tier: {spec.tier}\n"
        f"# priority: {spec.priority}\n\n"
    )
    body = dest.read_text(encoding="utf-8", errors="replace")
    if not body.startswith("# "):
        dest.write_text(header + body, encoding="utf-8")
        size = dest.stat().st_size

    return DownloadRecord(
        id=spec.id,
        title=spec.title,
        author_slug=author.slug,
        source_type="gutenberg",
        source_url=spec.url,
        local_path=str(dest.relative_to(out_root)),
        bytes=size,
        status="ok",
        downloaded_at=datetime.now(timezone.utc).isoformat(),
        priority=spec.priority,
        tier=spec.tier,
        est_lines=spec.est_lines,
        notes=spec.notes or f"gutenberg_id={spec.gutenberg_id}",
    )


def download_author(
    session: requests.Session,
    author_slug: str,
    out_base: Path,
    *,
    priority_filter: str | None = None,
) -> Manifest:
    if author_slug == "olavo":
        raise RuntimeError("Use research/download_olavo_sources.py for Olavo corpus.")

    author = AUTHOR_REGISTRY.get(author_slug)
    if not author:
        raise KeyError(
            f"Unknown author '{author_slug}'. "
            f"Known: {', '.join(sorted(AUTHOR_REGISTRY))}. "
            "See governance/docs/CORPUS_CLASSICAL_LIBERAL_CORPUS.md for unimplemented authors."
        )

    out_root = out_base / author_slug
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(author_slug=author_slug, output_root=str(out_root))

    for spec in author.sources:
        if priority_filter and spec.priority != priority_filter:
            continue
        if spec.source_type == "gutenberg":
            try:
                rec = download_gutenberg(session, spec, author, out_root)
                manifest.records.append(rec)
                print(f"OK {author_slug}/{rec.id}: {rec.bytes:,} bytes -> {rec.local_path}")
            except Exception as exc:  # noqa: BLE001
                manifest.failures.append({"id": spec.id, "error": str(exc), "url": spec.url})
                print(f"FAIL {author_slug}/{spec.id}: {exc}", file=sys.stderr)
        else:
            manifest.failures.append({
                "id": spec.id,
                "error": f"source_type '{spec.source_type}' not implemented in skeleton",
                "url": spec.url,
            })
            print(f"SKIP {author_slug}/{spec.id}: {spec.source_type} not implemented", file=sys.stderr)

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download classical liberal corpus to candidate staging.")
    p.add_argument("--out", type=Path, default=DEFAULT_CANDIDATES, help="Candidates base directory")
    p.add_argument("--author", default=None, help="Author slug (e.g. adam_smith, hume)")
    p.add_argument("--phase", type=int, default=None, choices=[1, 2, 3, 4], help="Download phase (see doc)")
    p.add_argument("--priority", default=None, help="Filter by priority P0|P1|P2|P3")
    p.add_argument("--list-authors", action="store_true", help="List registered authors and sources")
    p.add_argument("--dry-run", action="store_true", help="Print plan without downloading")
    return p


def main() -> int:
    args = build_parser().parse_args()
    out_base = args.out if args.out.is_absolute() else ROOT / args.out

    if args.list_authors:
        for slug, author in sorted(AUTHOR_REGISTRY.items()):
            print(f"\n{slug} — {author.name_en} (MoE: {author.moe_domain})")
            for spec in author.sources:
                print(f"  [{spec.priority}] {spec.id}: {spec.title} ({spec.source_type}) ~{spec.est_lines:,} lines")
            if not author.sources:
                print("  (delegated — see download_olavo_sources.py)" if slug == "olavo" else "  (no sources registered yet)")
        print("\nUnregistered (see doc): cicero, aquinas, mises, hayek, oakeshott, kirk, friedman, buchanan, scruton, sowell, solzhenitsyn, roberto_campos, george_grant")
        return 0

    if args.dry_run:
        targets = []
        if args.author:
            targets = [args.author]
        elif args.phase:
            targets = DOWNLOAD_PHASES.get(args.phase, [])
        plan = {"out": str(out_base), "targets": targets, "priority": args.priority}
        for slug in targets:
            author = AUTHOR_REGISTRY.get(slug)
            if author:
                plan[slug] = [
                    {"id": s.id, "priority": s.priority, "type": s.source_type, "url": s.url}
                    for s in author.sources
                    if not args.priority or s.priority == args.priority
                ]
            else:
                plan[slug] = "NOT_REGISTERED — implement in AUTHOR_REGISTRY"
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    if not args.author and not args.phase:
        print("Specify --author SLUG or --phase N or --list-authors", file=sys.stderr)
        return 2

    session = _session()
    slugs = [args.author] if args.author else DOWNLOAD_PHASES.get(args.phase, [])
    exit_code = 0

    for slug in slugs:
        if slug not in AUTHOR_REGISTRY:
            print(f"SKIP {slug}: not in AUTHOR_REGISTRY (implement or use olavo script)", file=sys.stderr)
            exit_code = 1
            continue
        try:
            manifest = download_author(session, slug, out_base, priority_filter=args.priority)
            summary = manifest.to_dict()
            print(json.dumps({
                "author": slug,
                "ok_count": summary["ok_count"],
                "fail_count": summary["fail_count"],
                "total_bytes": summary["total_bytes"],
                "est_lines_total": summary["est_lines_total"],
            }, indent=2))
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL author {slug}: {exc}", file=sys.stderr)
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
